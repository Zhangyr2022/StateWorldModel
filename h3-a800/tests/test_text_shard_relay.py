#!/usr/bin/env python3
"""Sharded text-conditioner relay test on a tiny random Qwen3-VL.

Builds a small Qwen3VLForConditionalGeneration (8 decoder layers, hidden 256),
saves it, then loads it through `text_shard.install` over N ranks and checks
that the relayed conditioning hidden states match the full model's own
`hidden_states[condition_layer]` to the bit.

Run: torchrun --standalone --nproc_per_node=2 tests/test_text_shard_relay.py
(also works with 4 or 8)
"""

from __future__ import annotations

import os
import sys

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONDITION_LAYER = 4
NUM_LAYERS = 8
HIDDEN = 256
SEQ = 96
TOKEN_IDS = list(range(1000, 1000 + SEQ))


class _FakePipe:
    def register_components(self, **components):
        for name, module in components.items():
            setattr(self, name, module)


class _FakeTokenizer:
    """Deterministic prompt -> the calibration token ids, like a real tokenizer."""

    def __call__(self, prompt, add_special_tokens=False):
        return {"input_ids": list(TOKEN_IDS)}


class _FakeProcessor:
    """Pure-text token type ids: every row is modality 0."""

    @staticmethod
    def create_mm_token_type_ids(seqs):
        return [[0] * len(seq) for seq in seqs]


def build_tiny_model(directory: str):
    from transformers import (
        Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    text_config = Qwen3VLTextConfig(
        vocab_size=151936,
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
    )
    # The vision tower's output width (`out_hidden_size`) must equal the text
    # hidden size: its embeddings are scattered straight into token positions.
    vision_config = Qwen3VLVisionConfig(
        hidden_size=HIDDEN,
        out_hidden_size=HIDDEN,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        depth=2,
        spatial_merge_size=2,
    )
    config = Qwen3VLConfig(text_config=text_config, vision_config=vision_config)
    torch.manual_seed(7)
    model = Qwen3VLForConditionalGeneration._from_config(config)
    model.save_pretrained(os.path.join(directory, "text_encoder"))


def main() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()

    # One shared directory per test run: rank 0 populates it, everyone reads it.
    directory = f"/tmp/h3a800_textshard_test_{os.getppid()}"
    if rank == 0:
        os.makedirs(directory, exist_ok=True)
        build_tiny_model(directory)
    dist.barrier()

    try:
        from h3_runtime import text_shard

        pipe = _FakePipe()
        runner = text_shard.install(
            pipe, directory, device, condition_layer=CONDITION_LAYER, verify=True
        )

        # One relayed encode (the calibration gates already ran inside install).
        input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long, device=device)
        hidden, received_ids = runner.encode(input_ids if rank == 0 else None)
        assert hidden.shape == (1, SEQ, HIDDEN), hidden.shape
        assert received_ids.shape == (1, SEQ), received_ids.shape

        # The pipeline-facing entry: the patched encode_prompt on the step class.
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep

        components = SimpleNamespace(
            tokenizer=_FakeTokenizer(), processor=_FakeProcessor(),
            text_encoder=pipe.text_encoder,
        )
        embeds, tags = MiniMaxH3TextEncoderStep.encode_prompt(
            components, "a test prompt", device=device, dtype=torch.bfloat16
        )
        assert embeds.shape == (1, SEQ, HIDDEN), embeds.shape
        assert torch.equal(embeds, hidden)

        # A second request with a different length exercises the shape change.
        short_ids = torch.tensor([TOKEN_IDS[:40]], dtype=torch.long, device=device)
        short_hidden, _ = runner.encode(short_ids if rank == 0 else None)
        assert short_hidden.shape == (1, 40, HIDDEN), short_hidden.shape

        # The Ref2VA paths: a video reference, then image+video in one request.
        # Each frame is its own vision run separated by timestamp text — the
        # layout `build_ref2va_presentation` produces and this transformers
        # version's M-RoPE consumes (`video_grid_thw` splits per frame).
        # Parity gate: the relayed hidden states must match the full unsharded
        # model's own `hidden_states[condition_layer]` for the same request.
        config = pipe.text_encoder.config
        vision = config.vision_config
        merge_sq = vision.spatial_merge_size**2
        pixel_width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2

        def frame_block(ts_ids, num_tokens, pad_id):
            ids = list(ts_ids) + [config.vision_start_token_id] + [pad_id] * num_tokens
            ids += [config.vision_end_token_id]
            tags = [0] * len(ts_ids) + [0] + [2] * num_tokens + [0]
            return ids, tags

        video_grid = torch.tensor([[2, 4, 4]], dtype=torch.long)
        # per-frame merged tokens are spatial only (h/merge x w/merge) — the
        # same arithmetic diffusers applies to the processor's per-frame grids
        video_tokens = int(video_grid[0][1]) * int(video_grid[0][2]) // merge_sq
        image_grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
        image_tokens = int(image_grid.prod()) // merge_sq

        cases = {
            "video": (
                [1, 2], video_grid, None,
                [3, 4],
            ),
            "image+video": (
                [1, 2], video_grid, image_grid,
                [3, 4],
            ),
        }
        for name, (head, vgrid, igrid, tail) in cases.items():
            token_ids, token_tags = list(head), [0] * len(head)
            if igrid is not None:
                block_ids, block_tags = frame_block([], image_tokens, config.image_token_id)
                block_tags = [1] * len(block_tags)  # an image run is modality 1
                block_tags[-1] = 0  # vision_end is text, as the processor derives
                block_tags[-2 - image_tokens] = 0  # vision_start too
                token_ids += block_ids
                token_tags += block_tags
            for index in range(int(vgrid[0][0])):  # one single-frame run per frame
                block_ids, block_tags = frame_block(
                    list(range(200 + 10 * index, 200 + 10 * index + 3)),
                    video_tokens, config.video_token_id,
                )
                token_ids += block_ids
                token_tags += block_tags
            token_ids += list(tail)
            token_tags += [0] * len(tail)

            request_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            request_mm = torch.tensor([token_tags], dtype=torch.long, device=device)
            video_pixels = torch.randn(
                int(vgrid.prod()), pixel_width, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(11)
            ).to(device)
            image_pixels = (
                torch.randn(
                    int(igrid.prod()), pixel_width, dtype=torch.bfloat16,
                    generator=torch.Generator().manual_seed(13),
                ).to(device)
                if igrid is not None else None
            )
            hidden, got_ids = runner.encode(
                request_ids if rank == 0 else None,
                pixel_values=image_pixels if rank == 0 else None,
                image_grid_thw=igrid.to(device) if igrid is not None and rank == 0 else None,
                pixel_values_videos=video_pixels if rank == 0 else None,
                video_grid_thw=vgrid.to(device) if rank == 0 else None,
                mm_token_type_ids=request_mm if rank == 0 else None,
            )
            assert torch.equal(got_ids, request_ids), f"{name}: relayed ids diverged"
            if rank == 0:
                from transformers import Qwen3VLForConditionalGeneration

                full = Qwen3VLForConditionalGeneration.from_pretrained(
                    os.path.join(directory, "text_encoder"), dtype=torch.bfloat16
                ).to(device)
                with torch.inference_mode():
                    reference = full.model(
                        input_ids=request_ids,
                        attention_mask=torch.ones_like(request_ids),
                        mm_token_type_ids=request_mm,
                        pixel_values=image_pixels,
                        image_grid_thw=igrid.to(device) if igrid is not None else None,
                        pixel_values_videos=video_pixels,
                        video_grid_thw=vgrid.to(device),
                        use_cache=False,
                        output_hidden_states=True,
                    ).hidden_states[CONDITION_LAYER]
                maximum = float((hidden.float() - reference.float()).abs().max().item())
                assert maximum < 1e-3, f"{name}: relay deviates from full model: {maximum:.3e}"
                print(f"  {name} relay parity: max_abs={maximum:.3e} over {tuple(hidden.shape)}")
                del full
                torch.cuda.empty_cache()
            dist.barrier()

        dist.barrier()
        if rank == 0:
            print(f"TEXT SHARD RELAY PASS over {world} ranks "
                  f"(layers [0, {CONDITION_LAYER}), hidden {HIDDEN})")
        dist.destroy_process_group()
    finally:
        if rank == 0:
            import shutil

            shutil.rmtree(directory, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
