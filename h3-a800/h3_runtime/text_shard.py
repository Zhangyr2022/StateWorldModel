"""Layerwise-sharded Qwen3-VL text conditioner for 80 GB A800 ranks.

The upstream Sol-H3 engine keeps a full Qwen3-VL conditioner (~30B parameters,
~60 GB in BF16) on *every* rank: fine on 288 GB B300, impossible on 80 GB A800.
MiniMax-H3 reads `hidden_states[50]` — the output of decoder layer 49 — so only
`embed_tokens` plus the first 50 decoder layers ever produce anything. This
module therefore

* truncates the conditioner to the 50 layers that matter (the layer tail and
  the language-model head are dropped; the vision tower survives on rank 0
  only, where the first-frame keyframe path needs it),
* splits those layers into contiguous per-rank slices that stay resident on
  each rank's GPU (~6 GB at world size 8, ~13 GB at 4),
* relays the hidden states down the rank chain (`send`/`recv`, ~10 MB per hop)
  and broadcasts the final conditioning tensor from the last rank, and
* replaces `MiniMaxH3TextEncoderStep.encode_prompt` wholesale, so the pipeline
  keeps calling the step it always called while every rank now cooperates on
  one encoder instead of redundantly running eight full copies.

Everything a rank needs besides the relayed hidden states — token ids, the
causal mask, the M-RoPE tables — is a deterministic function of replicated
inputs, so every rank rebuilds it locally instead of communicating it. The
model shell (`Qwen3VLModel` with its weightless helpers: `get_rope_index`,
`compute_3d_position_ids`, `get_image_features`, `get_placeholder_mask`) is
kept on every rank with only that rank's weights inside it, which is what
makes the replicated side bit-identical to the unsharded model.

The text-only (`t2v`) path replicates `Qwen3VLTextModel.forward`'s
`position_ids is None` branch (1-D positions). The keyframe (`i2v`) path adds
rank-0 vision-embed merging, 3-D M-RoPE, and the deepstack feature injections
into the first decoder layers — all of which replicate the corresponding
`Qwen3VLModel`/`Qwen3VLTextModel` forwards line for line.

`H3_A800_TEXT_VERIFY=1` (default on) builds calibration references from the
full unsharded model at load time and checks the sharded relay against them
right after installation — a one-time gate that the slice extraction and the
replicated forward are the real thing.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


_CALIBRATION_TOKEN_IDS = list(range(1, 65))
_DEEPSTACK_MAX_ABS = 2e-2


def _split_contiguous(total: int, world: int) -> list[tuple[int, int]]:
    """Contiguous balanced [(start, end), ...] covering [0, total) in rank order."""
    base, remainder = divmod(total, world)
    ranges, start = [], 0
    for rank in range(world):
        size = base + (1 if rank < remainder else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


class TextEncoderShim(nn.Module):
    """Stands in for `components.text_encoder` so the pipeline never misses it.

    `.model` is the kept-alive `Qwen3VLModel` shell carrying this rank's
    weights (embed on rank 0, one layer slice, the vision tower on rank 0) and
    the weightless helpers the relay replicates the full forward with.
    """

    def __init__(self, config, dtype: torch.dtype, shell: nn.Module):
        super().__init__()
        self.config = config
        self._dtype = dtype
        self.model = shell

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        """`ModularPipeline.device` reads this off every nn.Module component."""
        for parameter in self.parameters():
            return parameter.device
        return torch.device("cpu")

    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        raise RuntimeError("the sharded conditioner is driven through text_shard, not forward()")


class ShardedConditioner:
    """The relay engine: embed on rank 0, layers spread over ranks, one broadcast."""

    def __init__(self, shim: TextEncoderShim, *, rank: int, world: int, device: torch.device,
                 condition_layer: int, hidden_size: int, text_config, layer_start: int,
                 relay_group=None):
        self.shim = shim
        self.rank = rank
        self.world = world
        self.device = device
        self.condition_layer = condition_layer
        self.hidden_size = hidden_size
        self.text_config = text_config
        self.layer_start = layer_start
        self.relay_group = relay_group

    # ------------------------------------------------------------- broadcast

    def _broadcast_from_zero(self, tensor: torch.Tensor | None, *, dtype, device) -> torch.Tensor:
        """Rank 0's tensor, delivered to every rank. All ranks call this.

        A fixed-size header `[ndim, d0, d1, d2]` carries the shape, so every
        rank allocates the receive buffer before the payload broadcast.
        """
        if self.world == 1:
            assert tensor is not None
            return tensor
        if self.rank == 0:
            assert tensor is not None
            out = tensor.to(device=device, dtype=dtype).contiguous()
            dims = list(out.shape)
            header = torch.tensor([len(dims)] + dims + [0] * (3 - len(dims)),
                                  dtype=torch.long, device=device)
        else:
            header = torch.empty(4, dtype=torch.long, device=device)
        dist.broadcast(header, src=0)
        if self.rank != 0:
            ndim = int(header[0].item())
            dims = header[1 : 1 + ndim].tolist()
            out = torch.empty(dims, dtype=dtype, device=device)
        dist.broadcast(out, src=0)
        return out

    def _broadcast_int_list(self, values: list[int] | None, device) -> list[int]:
        if self.world == 1:
            return values
        count = torch.tensor([len(values) if values is not None else 0], dtype=torch.long,
                             device=device)
        dist.broadcast(count, src=0)
        n = int(count.item())
        if n == 0:
            return []
        buffer = (torch.tensor(values, dtype=torch.long, device=device) if self.rank == 0
                  else torch.empty(n, dtype=torch.long, device=device))
        dist.broadcast(buffer, src=0)
        return buffer.tolist()

    def _broadcast_vision_flags(self, has_image: bool, has_video: bool) -> tuple[bool, bool]:
        """Whether image and/or video pixels are in play, agreed with every rank."""
        if self.world == 1:
            return has_image, has_video
        flags = self._broadcast_int_list(
            [int(has_image), int(has_video)], self.device
        )
        return bool(flags[0]), bool(flags[1])

    # ---------------------------------------------------------------- encode

    @torch.inference_mode()
    def encode(self, input_ids: torch.Tensor | None = None, *, pixel_values: torch.Tensor | None = None,
               image_grid_thw: torch.Tensor | None = None,
               pixel_values_videos: torch.Tensor | None = None,
               video_grid_thw: torch.Tensor | None = None,
               mm_token_type_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the conditioner. Rank 0 supplies everything; others supply nothing.

        `pixel_values`/`image_grid_thw` carry the i2v keyframe or Ref2VA image
        references; `pixel_values_videos`/`video_grid_thw` carry Ref2VA video
        references. Both may arrive in one request.

        Returns `(hidden, input_ids)`: the (1, seq, hidden) conditioning tensor
        on every rank and the (1, seq) token ids it came from.
        """
        has_image, has_video = self._broadcast_vision_flags(
            pixel_values is not None, pixel_values_videos is not None
        )
        input_ids = self._broadcast_from_zero(
            input_ids, dtype=torch.long, device=self.device
        )
        mm = None
        grid = None
        vgrid = None
        if has_image or has_video:
            mm = self._broadcast_from_zero(mm_token_type_ids, dtype=torch.long, device=self.device)
        if has_image:
            grid = self._broadcast_from_zero(image_grid_thw, dtype=torch.long, device=self.device)
        if has_video:
            vgrid = self._broadcast_from_zero(video_grid_thw, dtype=torch.long, device=self.device)

        # Rank 0: embeddings, with the vision block merged in for i2v/ref2va.
        deepstack = None
        visual_pos_masks = None
        if self.rank == 0:
            hidden = self.shim.model.language_model.embed_tokens(input_ids)
            if has_image or has_video:
                hidden, deepstack, visual_pos_masks = self._merge_vision(
                    input_ids, hidden, mm=mm,
                    pixel_values=pixel_values if has_image else None,
                    image_grid_thw=grid,
                    pixel_values_videos=pixel_values_videos if has_video else None,
                    video_grid_thw=vgrid,
                )

        # Vision artifacts (WORLD group) strictly before the relay: NCCL's eager
        # P2P mode serializes send/recv against collectives on the same group,
        # so a rank holding a recv while rank 0 issues a broadcast deadlocks.
        if (has_image or has_video) and self.world > 1:
            deepstack = self._broadcast_deepstack(deepstack)
            visual_pos_masks = self._broadcast_from_zero(
                visual_pos_masks, dtype=torch.bool, device=self.device
            )

        # The relay proper lives on its own group, ordered after every WORLD
        # collective on every rank.
        seq = input_ids.shape[1]
        if self.world > 1 and self.rank > 0:
            hidden = torch.empty((1, seq, self.hidden_size), dtype=self.shim.dtype, device=self.device)
            dist.recv(hidden, src=self.rank - 1, group=self.relay_group)

        hidden = self._run_slice(hidden, input_ids, mm=mm, grid=grid, vgrid=vgrid,
                                 deepstack=deepstack, visual_pos_masks=visual_pos_masks)

        if self.world == 1:
            return hidden, input_ids
        if self.rank < self.world - 1:
            dist.send(hidden.contiguous(), dst=self.rank + 1, group=self.relay_group)
            out = torch.empty((1, seq, self.hidden_size), dtype=self.shim.dtype, device=self.device)
        else:
            out = hidden.contiguous()
        dist.broadcast(out, src=self.world - 1, group=self.relay_group)
        return out, input_ids

    def _merge_vision(self, input_ids, inputs_embeds, *, mm, pixel_values, image_grid_thw,
                      pixel_values_videos, video_grid_thw):
        """Rank-0 replicate of `Qwen3VLModel.forward`'s image and video branches.

        An i2v keyframe arrives as an image; Ref2VA image references are images
        and video references are videos, and a request can mix both. When it
        does, the two placeholder masks merge into one `visual_pos_masks` and
        the two deepstack feature lists interleave into per-layer joint tensors
        — exactly `Qwen3VLModel.forward`'s aggregation.
        """
        shell = self.shim.model
        image_mask = None
        video_mask = None
        deepstack_image = None
        deepstack_video = None

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device, self.shim.dtype)
            image_outputs = shell.get_image_features(
                pixel_values, image_grid_thw.to(self.device), return_dict=True
            )
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = shell.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            deepstack_image = image_outputs.deepstack_features

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(self.device, self.shim.dtype)
            video_outputs = shell.get_video_features(
                pixel_values_videos, video_grid_thw.to(self.device), return_dict=True
            )
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = shell.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            deepstack_video = video_outputs.deepstack_features

        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            deepstack = []
            for img_embed, vid_embed in zip(deepstack_image, deepstack_video):
                embed_joint = img_embed.new_zeros(
                    int(visual_pos_masks.sum()), img_embed.shape[-1]
                ).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack.append(embed_joint.to(inputs_embeds.dtype))
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack = deepstack_image
        else:
            visual_pos_masks = video_mask[..., 0]
            deepstack = deepstack_video

        if deepstack is not None:
            deepstack = [feature.to(inputs_embeds.device, inputs_embeds.dtype)
                         for feature in deepstack]
        return inputs_embeds, deepstack, visual_pos_masks

    def _broadcast_deepstack(self, deepstack):
        """Deliver the per-layer visual injections to whoever owns those layers."""
        count = torch.tensor(
            [len(deepstack) if deepstack is not None else 0], dtype=torch.long,
            device=self.device,
        )
        dist.broadcast(count, src=0)
        if int(count.item()) == 0:
            return None
        if self.rank == 0:
            shapes = torch.tensor([list(t.shape) for t in deepstack], dtype=torch.long,
                                  device=self.device).reshape(-1)
            length = torch.tensor([len(shapes)], dtype=torch.long, device=self.device)
        else:
            length = torch.empty(1, dtype=torch.long, device=self.device)
        dist.broadcast(length, src=0)
        if self.rank != 0:
            shapes = torch.empty(int(length.item()), dtype=torch.long, device=self.device)
        dist.broadcast(shapes, src=0)
        shape_pairs = shapes.view(-1, 2).tolist()

        received = []
        for (rows, hidden), index in zip(shape_pairs, range(len(shape_pairs))):
            if self.rank == 0:
                tensor = deepstack[index].contiguous()
            else:
                tensor = torch.empty((rows, hidden), dtype=self.shim.dtype, device=self.device)
            dist.broadcast(tensor, src=0)
            received.append(tensor)
        return received

    def _run_slice(self, hidden: torch.Tensor, input_ids: torch.Tensor, *, mm, grid, vgrid,
                   deepstack, visual_pos_masks) -> torch.Tensor:
        """Replicate `Qwen3VLTextModel.forward`'s prelude, then run this rank's layers."""
        from transformers.masking_utils import create_causal_mask

        shell = self.shim.model
        language = shell.language_model
        batch = input_ids.shape[0]
        attention_mask = torch.ones_like(input_ids)

        if grid is not None or vgrid is not None:
            position_ids = shell.compute_3d_position_ids(
                input_ids=input_ids, image_grid_thw=grid, video_grid_thw=vgrid,
                inputs_embeds=hidden, attention_mask=attention_mask,
                past_key_values=None, mm_token_type_ids=mm,
            )
        else:
            position_ids = None
        if position_ids is None:
            position_ids = (
                torch.arange(input_ids.shape[1], device=hidden.device)
                .view(1, 1, -1).expand(4, batch, -1)
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        causal_mask = create_causal_mask(
            config=self.text_config,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = language.rotary_emb(hidden, position_ids)

        for local_index, layer in enumerate(language.layers):
            out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            hidden = out if torch.is_tensor(out) else out[0]
            global_index = self.layer_start + local_index
            if deepstack is not None and global_index < len(deepstack):
                hidden = self._deepstack_add(hidden, visual_pos_masks, deepstack[global_index])
        return hidden

    @staticmethod
    def _deepstack_add(hidden: torch.Tensor, visual_pos_masks, visual_embeds) -> torch.Tensor:
        """Replicate `Qwen3VLTextModel._deepstack_process`."""
        visual_pos_masks = visual_pos_masks.to(hidden.device)
        visual_embeds = visual_embeds.to(hidden.device, hidden.dtype)
        hidden = hidden.clone()
        hidden[visual_pos_masks, :] = hidden[visual_pos_masks, :] + visual_embeds
        return hidden


_ACTIVE: ShardedConditioner | None = None


def active() -> ShardedConditioner:
    if _ACTIVE is None:
        raise RuntimeError("no sharded conditioner installed in this process")
    return _ACTIVE


# ------------------------------------------------------------------ loading

def _reference_hidden(model, token_ids: list[int], device: torch.device,
                      condition_layer: int, *, pixel_values=None, image_grid_thw=None,
                      mm_token_type_ids=None) -> torch.Tensor:
    """The full unsharded model's own `hidden_states[condition_layer]`, on CPU."""
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    if mm_token_type_ids is None:
        mm_token_type_ids = torch.zeros_like(input_ids)
    with torch.inference_mode():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            output_hidden_states=True,
        )
    return outputs.hidden_states[condition_layer]


def _vision_calibration_inputs(config, device) -> dict[str, Any] | None:
    """A tiny synthetic image request for the vision-path calibration gate."""
    try:
        vision = config.vision_config
        merge = vision.spatial_merge_size
        patch = vision.patch_size
        temporal = vision.temporal_patch_size
        channels = vision.in_channels
        image_token = config.image_token_id
        vision_start = config.vision_start_token_id
        vision_end = config.vision_end_token_id
    except AttributeError:
        return None
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    num_tokens = int(grid.prod()) // (merge * merge)
    pixel_values = torch.randn(int(grid.prod()), temporal * channels * patch * patch,
                               dtype=torch.bfloat16, device=device)
    token_ids = (
        [100, 101] + [vision_start] + [image_token] * num_tokens + [vision_end] + [102, 103]
    )
    # Only the image pads carry the vision modality tag; the wrappers are text,
    # exactly as the processor derives them from the vision pad ids.
    mm = torch.tensor(
        [[0] * 3 + [1] * num_tokens + [0] * 3], dtype=torch.long, device=device
    )
    return {
        "token_ids": token_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": grid,
        "mm_token_type_ids": mm,
    }


def _load_slice(model_path: str, start: int, end: int, *, rank: int, device: torch.device,
                condition_layer: int, verify: bool):
    """One rank's turn: load the full conditioner, keep only this rank's slice."""
    from transformers import Qwen3VLForConditionalGeneration

    text_dir = os.path.join(model_path, "text_encoder")
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        text_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=None
    )
    shell = full.model
    language = shell.language_model
    text_config = language.config
    num_layers = text_config.num_hidden_layers
    if num_layers <= condition_layer:
        raise ValueError(
            f"conditioner needs more than {condition_layer} decoder layers, has {num_layers}"
        )

    references: dict[str, torch.Tensor] = {}
    verify_error = ""
    calibration = None
    if verify and rank == 0:
        started = time.perf_counter()
        # Run the reference forward on this rank's (still empty) GPU: the same
        # pass is minutes on a single CPU thread — torchrun pins OMP_NUM_THREADS
        # to 1 — and about a second here. The trim below then keeps the slice
        # where it already is.
        try:
            full.to(device)
            references["text"] = _reference_hidden(
                full, _CALIBRATION_TOKEN_IDS, device, condition_layer
            )
            calibration = _vision_calibration_inputs(full.config, device)
            if calibration is not None:
                try:
                    references["vision"] = _reference_hidden(
                        full, calibration["token_ids"], device, condition_layer,
                        pixel_values=calibration["pixel_values"],
                        image_grid_thw=calibration["image_grid_thw"],
                        mm_token_type_ids=calibration["mm_token_type_ids"],
                    )
                except Exception as error:  # best effort; the real request is the final gate
                    print(f"[h3a800.text] vision calibration skipped: {type(error).__name__}: {error}",
                          flush=True)
                    calibration = None
            references = {name: tensor.cpu() for name, tensor in references.items()}
            print(
                f"[h3a800.text] calibration references built in {time.perf_counter() - started:.1f}s"
                f" ({sorted(references)})",
                flush=True,
            )
        except Exception as error:
            # capture (not raise): the failure is aligned across ranks below so
            # no rank blocks in the staggered-install barriers
            verify_error = f"{type(error).__name__}: {error}"
    if dist.is_initialized():
        flag = torch.tensor([1 if verify_error else 0], dtype=torch.int32, device=device)
        dist.broadcast(flag, src=0)
        if int(flag.item()):
            raise RuntimeError(
                "rank 0 calibration-reference build failed during text_shard install "
                "(see rank 0 log for the cause)"
            )

    config = full.config  # captured before `full` is dropped

    # Trim to this rank's resident slice. Replacing the ModuleList drops the
    # only remaining references to the non-kept layers; the shell keeps its
    # weightless helpers and config.
    keep = [language.layers[index] for index in range(start, end)]
    language.layers = nn.ModuleList(keep)
    del language.norm
    if rank != 0:
        del shell.visual
        del language.embed_tokens

    shim = TextEncoderShim(config, torch.bfloat16, shell)
    shim.to(device)

    del full
    gc.collect()
    torch.cuda.empty_cache()
    return shim, text_config, references, calibration


def install(pipe, model_path: str, device: torch.device, *, condition_layer: int | None = None,
            verify: bool | None = None):
    """Load the sharded conditioner into `pipe` and take over `encode_prompt`.

    Every rank must call this in the same order: ranks take turns loading the
    full conditioner so host RAM only ever holds one ~60 GB copy, and each turn
    ends at a barrier.
    """
    global _ACTIVE
    from diffusers.modular_pipelines.minimax_h3 import encoders as h3_encoders
    from diffusers.modular_pipelines.minimax_h3.packing import (
        MINIMAX_H3_TEXT_ENCODER_LAYER, MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG,
    )

    condition_layer = int(
        condition_layer if condition_layer is not None
        else os.environ.get("H3_A800_TEXT_CONDITION_LAYER", MINIMAX_H3_TEXT_ENCODER_LAYER)
    )
    if verify is None:
        verify = os.environ.get("H3_A800_TEXT_VERIFY", "1") == "1"

    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    ranges = _split_contiguous(condition_layer, world)
    start, end = ranges[rank]

    # All ranks load concurrently by default: the transient host peak is
    # world x ~63 GB (~500 GB at eight ranks), which this machine's 1.5 TB
    # absorbs easily. `H3_A800_STAGGERED_TEXT=1` restores the one-rank-at-a-time
    # turn-taking for hosts with tighter RAM.
    staggered = os.environ.get("H3_A800_STAGGERED_TEXT", "0") == "1"
    if world > 1 and staggered:
        for _ in range(rank):
            dist.barrier()
    shim, text_config, references, calibration = _load_slice(
        model_path, start, end, rank=rank, device=device, condition_layer=condition_layer,
        verify=verify,
    )
    if world > 1:
        if staggered:
            for _ in range(rank, world - 1):
                dist.barrier()
        else:
            dist.barrier()

    relay_group = dist.new_group(ranks=list(range(world))) if world > 1 else None

    runner = ShardedConditioner(
        shim, rank=rank, world=world, device=device, condition_layer=condition_layer,
        hidden_size=text_config.hidden_size, text_config=text_config, layer_start=start,
        relay_group=relay_group,
    )
    _ACTIVE = runner

    if verify:
        _calibrate(runner, references, calibration)

    pipe.register_components(text_encoder=shim)

    def sharded_encode_prompt(components, prompt, images=None, device=None, dtype=None):
        conditioner = active()
        tokenizer = components.tokenizer
        processor = getattr(components, "processor", None)

        pixel_values, image_grid_thw = None, None
        token_ids, token_tags = [], []
        if images:
            vision = processor.image_processor(images=images, return_tensors="pt")
            pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            merge_size = processor.image_processor.merge_size**2
            for index in range(len(images)):
                num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
                label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                vision_ids = (
                    [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                    + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids += label_ids + vision_ids
                token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids) + [MINIMAX_H3_VIDEO_TAG] * len(vision_ids)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)

        mm_token_type_ids = torch.tensor(
            processor.create_mm_token_type_ids([token_ids]), dtype=torch.long
        )
        input_ids = (
            torch.tensor([token_ids], dtype=torch.long, device=conditioner.device)
            if conditioner.rank == 0 else None
        )
        pixel_values = (
            pixel_values if conditioner.rank == 0 else None
        )
        image_grid_thw = (
            image_grid_thw if conditioner.rank == 0 else None
        )
        mm_token_type_ids = (
            mm_token_type_ids if conditioner.rank == 0 else None
        )
        hidden, input_ids = conditioner.encode(
            input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        dtype = dtype or components.text_encoder.dtype
        return hidden.to(dtype=dtype), torch.tensor(token_tags, dtype=torch.long)

    h3_encoders.MiniMaxH3TextEncoderStep.encode_prompt = staticmethod(sharded_encode_prompt)

    if rank == 0:
        slice_bytes = sum(p.numel() * p.element_size() for p in shim.parameters())
        print(
            f"[h3a800.text] conditioner sharded: layers [0, {condition_layer}) over {world} ranks; "
            f"rank 0 slice (embed + {end - start} layers + vision tower) "
            f"{slice_bytes / 1024**3:.1f} GB",
            flush=True,
        )
    return runner


def _calibrate(runner: ShardedConditioner, references: dict, calibration: dict | None) -> None:
    """Collective: encode the calibration requests through the relay and compare."""
    rank0 = runner.rank == 0
    # Every rank must run the same number of collective encodes, so the case
    # list is agreed through a broadcast rather than read from rank-0-only data.
    vision_flag = runner._broadcast_int_list(
        [1] if "vision" in references else [], runner.device
    )
    cases = [("text", _CALIBRATION_TOKEN_IDS, None)]
    if vision_flag:
        # Only rank 0 holds the calibration data; the others just participate
        # in the collective encode.
        cases.append((
            "vision",
            calibration["token_ids"] if calibration is not None else None,
            calibration,
        ))
    for name, token_ids, vision_case in cases:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=runner.device) if rank0 else None
        hidden, _ = runner.encode(
            input_ids,
            pixel_values=vision_case["pixel_values"] if vision_case and rank0 else None,
            image_grid_thw=vision_case["image_grid_thw"] if vision_case and rank0 else None,
            mm_token_type_ids=vision_case["mm_token_type_ids"] if vision_case and rank0 else None,
        )
        if rank0:
            reference = references[name].to(hidden.device, hidden.dtype)
            maximum = float((hidden.float() - reference.float()).abs().max().item())
            ok = maximum <= _DEEPSTACK_MAX_ABS
            verdict = "PASS" if ok else "FAIL"
            print(
                f"[h3a800.text] {name} calibration: {verdict} max_abs={maximum:.3e} "
                f"over {tuple(hidden.shape)}",
                flush=True,
            )
            flag = torch.tensor([0 if ok else 1], dtype=torch.int32, device=runner.device)
        else:
            flag = torch.zeros(1, dtype=torch.int32, device=runner.device)
        # align the verdict across ranks: a rank-local raise here would leave
        # the other ranks blocked in the next collective until NCCL times out
        dist.broadcast(flag, src=0)
        if int(flag.item()):
            raise RuntimeError(
                f"sharded conditioner deviates on the {name} path "
                "(calibration threshold exceeded; see rank 0 for max_abs)"
            )


def install_ref2va_encoder(pipe) -> None:
    """Take over `MiniMaxH3Ref2VATextEncoderStep.encode_prompt` with the relay.

    The Ref2VA step builds its own presentation — a label per reference,
    numbered per modality, plus one timestamped vision block per merged frame
    pair of a video reference — and then fires the conditioner. The
    presentation is a deterministic function of the prepared references, so
    every rank rebuilds it locally; the timestamps come from
    `sample_reference_video_frames`, which reads only replicated frames. Rank 0
    alone runs the vision processors and owns the pixel tensors, and its two
    token-count lists ride a broadcast so every rank tokenizes the identical
    request. The conditioner forward itself goes through `encode`, whose image
    and video branches carry the Ref2VA references the way the i2v path carries
    the keyframe.
    """
    import numpy as np

    from diffusers.modular_pipelines.minimax_h3 import encoders as h3_encoders
    from diffusers.modular_pipelines.minimax_h3.packing_ref2va import (
        build_ref2va_presentation,
        sample_reference_video_frames,
    )

    def sharded_encode_prompt(components, prompt, references, device=None, dtype=None):
        conditioner = active()
        tokenizer = components.tokenizer
        processor = getattr(components, "processor", None)
        merge_size = processor.image_processor.merge_size**2

        images = [reference.image for reference in references if reference.kind == "image"]
        videos = [reference for reference in references if reference.kind == "video"]
        for reference in videos:
            _, reference.block_timestamps = sample_reference_video_frames(reference.frames)

        image_token_counts, video_block_token_counts = [], []
        pixel_values = image_grid_thw = None
        pixel_values_videos = video_grid_thw = None
        if conditioner.rank == 0:
            if images:
                vision = processor.image_processor(images=images, return_tensors="pt")
                pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
                image_token_counts = [int(grid.prod()) // merge_size for grid in image_grid_thw]
            if videos:
                sampled = [sample_reference_video_frames(reference.frames) for reference in videos]
                vision = processor.video_processor(
                    videos=[np.stack(frames) for frames, _ in sampled],
                    do_sample_frames=False,
                    return_tensors="pt",
                )
                pixel_values_videos, video_grid_thw = (
                    vision["pixel_values_videos"], vision["video_grid_thw"]
                )
                video_block_token_counts = [
                    int(grid[1]) * int(grid[2]) // merge_size for grid in video_grid_thw
                ]
                for reference, grid in zip(videos, video_grid_thw):
                    if int(grid[0]) != len(reference.block_timestamps):
                        raise ValueError(
                            f"the processor merged a reference video into {int(grid[0])} vision "
                            f"blocks, but MiniMax-H3 labels {len(reference.block_timestamps)} of them"
                        )
        image_token_counts = conditioner._broadcast_int_list(
            image_token_counts, conditioner.device
        )
        video_block_token_counts = conditioner._broadcast_int_list(
            video_block_token_counts, conditioner.device
        )

        token_ids, token_tags = build_ref2va_presentation(
            tokenizer, prompt, references, image_token_counts, video_block_token_counts
        )
        input_ids = (
            torch.tensor([token_ids], dtype=torch.long, device=conditioner.device)
            if conditioner.rank == 0
            else None
        )
        mm_token_type_ids = (
            torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long)
            if conditioner.rank == 0
            else None
        )
        hidden, _ = conditioner.encode(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        dtype = dtype or components.text_encoder.dtype
        return hidden.to(dtype=dtype), torch.tensor(token_tags, dtype=torch.long)

    h3_encoders.MiniMaxH3Ref2VATextEncoderStep.encode_prompt = staticmethod(sharded_encode_prompt)

    if (dist.get_rank() if dist.is_initialized() else 0) == 0:
        print("[h3a800.text] ref2va conditioner step routed through the sharded relay", flush=True)
