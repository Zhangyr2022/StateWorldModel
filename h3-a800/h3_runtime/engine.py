"""Resident MiniMax-H3 T2V/I2V/Ref2VA inference engine for 80 GB A800 ranks.

Forked from the Sol-H3 engine (validated on 8x B300). What changes for A800:

* **Dense attention everywhere.** The SOL/BSA backends bottom out in cuDNN
  block-sparse kernels that were never validated below SM103, so the default
  is SDPA; the Ulysses exchange is unchanged and runs BF16 on the wire.
  Ref2VA runs dense only: its sparse routing (the `text_audio` sink policy)
  was validated against the cuDNN BSA kernel, which SM80 cannot run.
* **Sharded text conditioner.** A full ~60 GB Qwen3-VL copy per rank does not
  fit an 80 GB card, so `text_shard` spreads its first 50 decoder layers over
  the ranks and relays the hidden states (see `text_shard.py`). The Ref2VA
  presentation rides the same relay (`install_ref2va_encoder`), including
  image and video references.
* **Staged device placement.** The transformer goes to the GPU at load; the
  VAEs move in lazily at first use, because the AdaLN precompute of the first
  denoise step frees ~26 GB per rank and that headroom is what the decode
  peak needs. Rank memory at steady state: ~37 GB DiT + text slice + VAEs.
  Ref2VA pins the video VAE earlier: its reference encode runs through the
  tile-sharded `_encode_clip`, which does not pass through the lazy wrapper.
* **Resolution is a parameter** (`width`/`height`, multiples of 32), for the
  480p profile this deployment targets.

Kept from upstream unchanged: the CP plan and Ulysses all-to-all, AdaLN
precompute, the Triton fusions, sharded VAE decode with global tile batching,
LoRA fusion, and the fragmented-MP4 encode handoff.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


FPS = 24
INFERENCE_STEPS = 5  # Five scheduler points execute four DiT forwards.
DURATION_FRAMES = {5: 124, 10: 243, 15: 362}
ATTENTION_BACKENDS = {"dense", "vsa", "sol"}
TASKS = {"t2v", "ref2va"}  # i2v is t2v driven through generate(image=...)
RESOLUTIONS = {"480p": (864, 480), "768p": (1344, 768)}
REFERENCE_IMAGE_RESIZE_MODES = {"match", "diffusers"}


def _resolve_reference_image_size(
    width: int,
    height: int,
    *,
    mode: str,
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int]:
    """Resolve Ref2VA image geometry for the selected deployment profile."""
    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")

    if mode == "match":
        # Match the fastest validated Ref2VA profile: never upscale a source,
        # and cap larger inputs to the generated canvas area.
        scale = min(1.0, math.sqrt((canvas_width * canvas_height) / (width * height)))
    elif mode == "diffusers":
        # Official Diffusers parity mode. This intentionally upscales every
        # reference image to a 2048-pixel short edge.
        scale = 2048 / min(width, height)
    else:
        raise ValueError(
            f"reference_image_resize_mode must be one of {sorted(REFERENCE_IMAGE_RESIZE_MODES)}"
        )

    multiple = 32
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def _media_crc(medium: Any) -> int:
    """A content CRC over one reference medium: PIL image, frame array or waveform.

    The three media a `MiniMaxH3Reference` can carry arrive as PIL `Image`,
    `(frames, h, w, 3)` `uint8` numpy and a `(channels, samples)` float tensor
    respectively; each converts to raw bytes here. bfloat16 tensors have no
    numpy view and are reinterpreted as int16 first.
    """
    import numpy as np
    import zlib

    if isinstance(medium, torch.Tensor):
        medium = medium.detach().cpu().contiguous()
        if medium.dtype == torch.bfloat16:
            medium = medium.view(torch.int16)
        payload = medium.numpy().tobytes()
    elif isinstance(medium, np.ndarray):
        payload = medium.tobytes()
    else:  # PIL Image (or anything with a raw byte dump)
        payload = medium.tobytes()
    return zlib.crc32(payload) & 0x7FFFFFFF


@dataclass
class GeneratedMedia:
    """Rank-zero output. Other ranks return ``None`` from ``generate``."""

    video: Any
    audio: Any
    audio_sample_rate: int | None
    elapsed_s: float
    duration: int
    seed: int
    stages: dict[str, float] | None = None

    def save_async(self, output_path: str | Path):
        """Stage output and encode it in the background."""
        from .encoding import start_fast_video_encode

        if self.video is None:
            raise RuntimeError("this output has already been handed to an encoder")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        job = start_fast_video_encode(
            self.video,
            fps=FPS,
            output_path=str(output_path),
            audio=self.audio,
            audio_sample_rate=self.audio_sample_rate,
            chunk_frames=8,
            fragmented=True,
            encoder_threads=36,
            pyav_zero_copy=True,
            overlap_audio=True,
        )
        job.wait_staged()
        self.video = None
        self.audio = None
        return job

    def save(self, output_path: str | Path) -> Path:
        job = self.save_async(output_path)
        job.wait()
        return Path(output_path)


def _state_value(state: Any, name: str) -> Any:
    if hasattr(state, "get"):
        value = state.get(name)
        if value is not None:
            return value
    return getattr(state, name, None)


def _lazy_to_device(module: torch.nn.Module, device: torch.device, label: str):
    """Move a module to the GPU on its first use; a no-op afterwards.

    Covers the clip-level entries the MiniMax-H3 pipeline actually calls
    (`_encode_clip` for i2v keyframes, `_decode` for the decode steps), plus
    the public `encode`/`decode`.
    """
    state = {"moved": False}

    for name in ("decode", "encode", "_decode", "_encode", "_decode_clip", "_encode_clip"):
        original = getattr(module, name, None)
        if original is None:
            continue

        def make(original=original):
            def wrapped(*args, **kwargs):
                if not state["moved"]:
                    module.to(device)
                    state["moved"] = True
                    print(f"[h3a800.lazy] {label} moved to {device}", flush=True)
                return original(*args, **kwargs)
            return wrapped

        setattr(module, name, make())


class MiniMaxH3Inference:
    """Load once and generate fixed-profile T2V/I2V/Ref2VA on 4 or 8 A800 ranks."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | Path,
        attention_backend: str = "dense",
        task: str = "t2v",  # i2v is t2v driven through generate(image=...)
        width: int = 864,
        height: int = 480,
        reference_image_resize_mode: str = "match",
    ) -> None:
        self._owns_process_group = False
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

        if dist.is_available() and not dist.is_initialized():
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size > 1:
                dist.init_process_group(backend="nccl", device_id=self.device)
                self._owns_process_group = True

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.world_size not in {4, 8}:
            raise ValueError(
                "the A800 profile needs 4 or 8 GPU processes: at world size 1-2 the "
                "61.7 GB dense DiT plus the text slice cannot share an 80 GB card "
                f"(got {self.world_size})"
            )
        task = task.lower()
        if task not in TASKS:
            raise ValueError(f"task must be one of {sorted(TASKS)} (i2v = t2v + --image)")
        attention_backend = attention_backend.lower()
        if attention_backend == "sol_bsa":
            # cuDNN BSA explicitly requires SM90-SM120 (verified 2026-09-12:
            # `BSA.block_sparse_attention_forward` raises "requires SM90-SM120,
            # found SM80" with nvidia-cudnn-frontend installed). The `sol`
            # backend covers SM80 through the sol_attn package's Triton kernel.
            raise ValueError(
                "sol_bsa requires SM90-SM120 (cuDNN BSA rejects SM80); "
                "use 'dense', 'vsa', or 'sol'"
            )
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {sorted(ATTENTION_BACKENDS)}")
        if task == "ref2va" and attention_backend != "dense":
            # Upstream also rejected `sol` for Ref2VA; its sparse routing
            # (`text_audio` sink) was only validated against the cuDNN BSA
            # kernel, and sol_bsa is architecture-excluded above. Dense is the
            # one Ref2VA backend this deployment can vouch for.
            raise ValueError("ref2va supports dense attention on this deployment")
        reference_image_resize_mode = reference_image_resize_mode.lower()
        if reference_image_resize_mode not in REFERENCE_IMAGE_RESIZE_MODES:
            raise ValueError(
                "reference_image_resize_mode must be one of "
                f"{sorted(REFERENCE_IMAGE_RESIZE_MODES)}"
            )
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError(
                f"width and height must be positive multiples of 32, got {width}x{height}"
            )
        self.task = task
        self.attention_backend = attention_backend
        self.reference_image_resize_mode = reference_image_resize_mode
        self.width = width
        self.height = height

        from diffusers import ComponentsManager, ModularPipeline

        from diffusers.modular_pipelines.minimax_h3 import before_encoder

        before_encoder.MINIMAX_H3_MAX_DURATION = 364 / FPS

        manager = ComponentsManager()
        if task == "ref2va":
            # before_encoder imports the resolver by name, so override that
            # binding without modifying the installed Diffusers package.
            before_encoder.resolve_reference_image_size = lambda image_width, image_height: (
                _resolve_reference_image_size(
                    image_width,
                    image_height,
                    mode=self.reference_image_resize_mode,
                    canvas_width=width,
                    canvas_height=height,
                )
            )
            from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Ref2VABlocks

            self.pipe = MiniMaxH3Ref2VABlocks().init_pipeline(
                model_path, components_manager=manager
            )
        else:
            self.pipe = ModularPipeline.from_pretrained(model_path, components_manager=manager)

        # The sharded conditioner registers itself as `text_encoder` *before*
        # `load_components`, whose default name selection skips components that
        # already exist — so the full 60 GB copy is never loaded, on any rank.
        from . import text_shard

        self.conditioner = text_shard.install(self.pipe, model_path, self.device)
        if task == "ref2va":
            text_shard.install_ref2va_encoder(self.pipe)
        # Resolve every component below the selected model root, ignoring any
        # absolute paths recorded in the checkpoint's component indexes. Without
        # this, a locally converted model_index.json pins its original
        # directory and `--model` would only be honored by some components.
        load_kwargs = {"dtype": torch.bfloat16, "pretrained_model_name_or_path": model_path}
        self.pipe.load_components(**load_kwargs)

        transformer_component = "transformer_ref" if task == "ref2va" else "transformer"
        self.transformer = getattr(self.pipe, transformer_component, None)
        if self.transformer is None:
            raise RuntimeError(
                f"failed to load {transformer_component} from {model_path!r}; "
                f"make sure {transformer_component}/config.json and all checkpoint shards exist"
            )
        self.transformer.to(self.device)
        self.pipe.scheduler.set_shift(12.0)
        self.pipe.audio_scheduler.set_shift(3.0)

        # VSA adapters carry 50 `to_gate_compress.set_weight` payloads. The
        # gate Linears are created on the CPU here (memory: they must not sit
        # on the GPU while the AdaLN-resident fusion runs below), fused
        # weights are copied in after the AdaLN eviction frees its ~26 GB.
        attention_fn = None
        self.sparse_attention = None
        if attention_backend == "sol":
            # Triton Sol-Attn: the sol_attn package (fetch_sol_attn.sh) dispatches SM80
            # (no specialized CuTe kernel, no CuTe runtime) to its triton_ref
            # kernel. sparse_attention owns routing/sink/residual-merge wiring
            # and plugs into the Ulysses exchange like the VSA callable does.
            # Upstream validated exactly this pairing on 4x A100 (tau=1.0,
            # exact threshold, full-prefix sink, dense prefix queries).
            from . import sparse_attention

            self.sparse_attention = sparse_attention.install(
                self.transformer,
                backend="sol",
                tau=1.0,
                dense_steps=1,
                dense_layers=2,
                sink_mode="prefix",
            )
            attention_fn = self.sparse_attention
        elif attention_backend == "vsa":
            from .vsa.h3_vsa import load_vsa_gates, make_vsa_attention_fn

            hidden = self.transformer.config.hidden_size
            width = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
            for block in self.transformer.transformer_blocks:
                # Zero-initialized by design: untrained gates reduce the VSA
                # compressed branch to zero, so feeding a dense-datafree
                # adapter here by mistake degrades to pure sparse attention
                # instead of polluting every layer with random weights.
                gate = torch.nn.Linear(hidden, width, bias=False)
                with torch.no_grad():
                    gate.weight.zero_()
                block.attn.to_gate_compress = gate
            attention_fn = make_vsa_attention_fn()
            self._install_vsa_layout_hook()

        from .lora import fuse_lora

        report = fuse_lora(
            self.transformer,
            adapter_path,
            alpha=8 if task == "ref2va" else 64,
            scale=1.0,
        )
        if self.rank == 0:
            print(f"[h3a800.lora] fused {report.pairs} lora pairs, {report.diffs} diffs, "
                  f"{report.replacements} gate replacements in {report.fuse_s:.1f}s", flush=True)

        # Evict the AdaLN projections (~26 GB) after fusion: their table is
        # rebuilt from these weights at the first denoise step anyway
        # (adaln.precompute streams each block back for its GEMMs), and the
        # reference/keyframe encode runs BEFORE that point — with the weights
        # resident it OOMs an 80 GB card.
        for block in self.transformer.transformer_blocks:
            block.adaln_proj.to("cpu")
        torch.cuda.empty_cache()

        if attention_backend == "vsa":
            from .vsa.h3_vsa import load_vsa_gates

            for block in self.transformer.transformer_blocks:
                block.attn.to_gate_compress.to(device=self.device, dtype=torch.bfloat16)
            filled = load_vsa_gates(self.transformer, str(adapter_path), verbose=self.rank == 0)
            expected = len(self.transformer.transformer_blocks)
            if filled != expected:
                raise RuntimeError(
                    f"the adapter provides gate weights for {filled} of {expected} blocks; "
                    "--attention-backend vsa requires the vsa-datafree adapter "
                    "(checkpoints/FastH3-4-step-Preview-v1-LoRA/vsa-datafree/)"
                )

        if self.world_size > 1:
            from diffusers.models._modeling_parallel import ContextParallelConfig

            from .cp_plan import MINIMAX_H3_CP_PLAN, assert_no_attention_mask

            assert_no_attention_mask(self.transformer)
            self.transformer.enable_parallelism(
                config=ContextParallelConfig(
                    ulysses_degree=self.world_size,
                    ulysses_anything=True,
                ),
                cp_plan=MINIMAX_H3_CP_PLAN,
            )

        from . import adaln, fusion_install, stage_timing, ulysses, vae_parallel

        if os.environ.get("H3_A800_DISABLE_FUSIONS", "0") != "1":
            fusion_install.install(self.transformer)
        adaln.enable_adaln_precompute(
            self.transformer,
            verbose=self.rank == 0,
            component_name=transformer_component,
        )
        if self.world_size > 1:
            ulysses.install(self.transformer, attention_fn=attention_fn)

        # The first denoise step's AdaLN precompute frees ~26 GB per rank; the
        # VAEs ride in only when they are first needed, after that headroom
        # exists. Ref2VA is the exception on the encode side: its reference
        # tiles go through the encode-sharded `_encode_clip`, which calls the
        # encoder directly and never fires the lazy wrapper — so the video VAE
        # must already be resident when the pipeline starts. The audio VAE has
        # no sharded path and stays lazy.
        if task == "ref2va":
            self.pipe.vae.to(self.device)
            if self.rank == 0:
                print("[h3a800.lazy] video_vae moved (ref2va reference encode)", flush=True)
        else:
            _lazy_to_device(self.pipe.vae, self.device, "video_vae")
        _lazy_to_device(self.pipe.audio_vae, self.device, "audio_vae")
        vae_parallel.install(
            self.pipe.vae,
            batched=True,
            compile_mode=os.environ.get("H3_VAE_COMPILE") or None,
            encode_parallel=task == "ref2va",
        )
        self._fix_execution_device()

        stage_timing.activate()
        stage_timing.install()
        self._adaln_checked = False

    def _install_vsa_layout_hook(self) -> None:
        """Feed the VSA tile geometry from the pipeline's own layout step.

        The packed layout ([text | conditioning | audio | generated video])
        and the video grid are decided inside the pipeline per request; this
        hook mirrors them into the attention core right after they are built.
        Every rank runs the same SPMD step, so the geometry agrees without a
        collective.
        """
        from diffusers.modular_pipelines.minimax_h3 import before_denoise

        from .vsa import h3_vsa

        if getattr(before_denoise.MiniMaxH3PrepareLayoutStep, "_h3a800_vsa_hooked", False):
            return

        original = before_denoise.MiniMaxH3PrepareLayoutStep.__call__

        def hooked(step_self, components, state):
            result = original(step_self, components, state)
            text = int(state.get("text_indices").shape[0])
            cond = int(state.get("num_condition_video_rows"))
            audio = int(state.get("audio_indices").shape[0])
            frames = int(state.get("num_latent_frames"))
            grid_h = int(state.get("latent_height")) // 2
            grid_w = int(state.get("latent_width")) // 2
            # diffusers packing invariants the VSA tiling depends on: video
            # rows sit after cond rows and exactly fill the (t, h/2, w/2) grid
            gen_video = int(state.get("video_indices").shape[0]) - cond
            if gen_video != frames * grid_h * grid_w:
                raise RuntimeError(
                    f"packed video rows {gen_video} != frames*grid {frames * grid_h * grid_w}; "
                    "diffusers packing semantics changed — VSA tiling would be misaligned"
                )
            h3_vsa.set_request_layout(
                (text, cond, audio),
                (frames, grid_h, grid_w),
            )
            return result

        before_denoise.MiniMaxH3PrepareLayoutStep.__call__ = hooked
        before_denoise.MiniMaxH3PrepareLayoutStep._h3a800_vsa_hooked = True

    def _fix_execution_device(self) -> None:
        """`pipe.device` is the *first* nn.Module component's device; make it CUDA.

        Without `pipe.to(device)` (deliberate — the VAEs stay on the CPU until
        first use) the property can still resolve to a CPU module, and every
        downstream tensor-creation call in the pipeline keys off it. Move the
        minimum set of components that flips it.
        """
        for _ in range(8):
            if self.pipe.device.type == "cuda":
                return
            for component in self.pipe.components.values():
                if isinstance(component, torch.nn.Module) and component.device.type != "cuda":
                    component.to(self.device)
                    break
            else:
                return

    def _check_request_consistency(self, duration: int, seed: int, width: int, height: int,
                                   has_image: bool, prompt: str, first_frame,
                                   references) -> None:
        """Fail loudly when ranks were handed diverging python-level requests.

        The pipeline is SPMD: setup, the conditional keyframe-VAE step and the
        token builders consume the request on every rank independently, and a
        rank that sees a different prompt or image builds a different packed
        layout than the rank that does — which only surfaces as an exotic
        index_copy error deep in the transformer. Compare a cheap signature
        that covers the content (prompt/image/reference CRCs), not just the
        shapes.
        """
        import zlib

        prompt_crc = zlib.crc32(prompt.encode()) & 0x7FFFFFFF
        if first_frame is not None:
            image_crc = zlib.crc32(first_frame.tobytes()) & 0x7FFFFFFF
        else:
            image_crc = 0
        reference_crcs = []
        for reference in references or []:
            for medium in (reference.image, reference.video, reference.audio):
                if medium is None:
                    continue
                reference_crcs.append(_media_crc(medium))
        # Fixed width: every rank allocates the same signature even when their
        # reference lists disagree — that disagreement is exactly what this
        # check exists to catch, and it must surface as a compare, not as an
        # all_gather shape error one rank raises while the rest hang.
        width_slots = 24  # <= 12 references x (video + soundtrack)
        if len(reference_crcs) > width_slots:
            raise ValueError(f"more reference media than the signature can carry: {reference_crcs}")
        reference_crcs += [0] * (width_slots - len(reference_crcs))
        signature = torch.tensor(
            [duration, int(seed), width, height, int(has_image), prompt_crc, image_crc,
            *reference_crcs],
            dtype=torch.long, device=self.device,
        )
        gathered = [torch.zeros_like(signature) for _ in range(self.world_size)]
        dist.all_gather(gathered, signature)
        if any(not torch.equal(item, gathered[0]) for item in gathered[1:]):
            raise RuntimeError(
                "ranks received different generate() requests "
                f"(rank 0: {gathered[0].tolist()}); the pipeline is SPMD and every "
                "rank must pass identical prompt/image/references/width/height/duration/seed"
            )

    def _align_rank_failures(self, local_err: str) -> None:
        """Raise on every rank if any rank recorded a local pre-flight failure.

        Without this alignment a single rank's ValueError/OOM leaves the other
        ranks blocked inside the next collective until the NCCL watchdog times
        out (~10 minutes), with the real cause scattered in one rank's log.
        """
        if self.world_size == 1:
            if local_err:
                raise RuntimeError(local_err)
            return
        buf = torch.zeros(256, dtype=torch.uint8, device=self.device)
        if local_err:
            encoded = local_err.encode()[:255]
            buf[:len(encoded)] = torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
        gathered = [torch.zeros(256, dtype=torch.uint8, device=self.device)
                    for _ in range(self.world_size)]
        dist.all_gather(gathered, buf)
        failures = [bytes(t.tolist()).decode(errors="replace").rstrip("\x00")
                    for t in gathered]
        failures = [f for f in failures if f]
        if failures:
            raise RuntimeError("rank-local pre-flight failures: " + " | ".join(failures))

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    @staticmethod
    def _load_image(image) -> Any | None:
        from PIL import Image, ImageOps

        if image is None:
            return None
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")
        with Image.open(image) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")

    def _check_adaln(self) -> None:
        if self._adaln_checked:
            return
        # T2V/I2V cache two conditioning schedules (base, visual-reference);
        # Ref2VA adds the audio-reference one.
        schedules = 3 if self.task == "ref2va" else 2
        for index, block in enumerate(self.transformer.transformer_blocks):
            table = getattr(block.adaln_proj, "table", None)
            if table is None or table.ndim != 4 or table.shape[0] != schedules:
                raise RuntimeError(f"invalid AdaLN cache at transformer block {index}")
        self._adaln_checked = True

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        duration: int,
        seed: int = 0,
        image: str | Path | Any | None = None,
        references: list[Any] | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> GeneratedMedia | None:
        """Generate one video with synchronized audio; `image` keys the first frame (i2v).

        ``references`` (list of `MiniMaxH3Reference`, in packed order) is
        required for, and only valid on, the ref2va task. `width`/`height`
        override the constructor canvas for this request only (multiples of
        32) — the benchmark matrix sweeps resolutions in one resident process
        rather than paying a reload per size.
        """
        # Pre-flight validation is captured (not raised) so that a rank-local
        # failure can be aligned across ranks before anyone enters a
        # collective — otherwise the other ranks block until the NCCL
        # watchdog times out (~10 minutes) instead of seeing this error.
        local_err = ""
        first_frame = None
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a non-empty string")
            if duration not in DURATION_FRAMES:
                raise ValueError("duration must be 5, 10, or 15 seconds")
            width = int(width or self.width)
            height = int(height or self.height)
            if width <= 0 or height <= 0 or width % 32 or height % 32:
                raise ValueError(
                    f"width and height must be positive multiples of 32, got {width}x{height}"
                )
            if self.task == "ref2va":
                if image is not None:
                    raise ValueError("image is a first-frame I2V input; use references for ref2va")
                if not references:
                    raise ValueError("ref2va requires at least one reference")
            elif references:
                raise ValueError("references are only valid when task='ref2va'")
            first_frame = self._load_image(image)
        except Exception as exc:
            if self.world_size == 1:
                raise
            local_err = f"{type(exc).__name__}: {exc}"
            width, height = int(self.width), int(self.height)
            references = None

        from . import stage_timing
        from .ulysses import reset_row_counts

        stage_timing.reset()
        if self.world_size > 1:
            reset_row_counts()
            self._check_request_consistency(duration, seed, width, height,
                                            first_frame is not None, prompt, first_frame,
                                            references)
            self._align_rank_failures(local_err)
            dist.barrier()
        if local_err:
            raise RuntimeError(local_err)

        request = {
            "prompt": prompt.strip(),
            "height": height,
            "width": width,
            "num_frames": DURATION_FRAMES[duration],
            "num_inference_steps": INFERENCE_STEPS,
            "generator": torch.Generator().manual_seed(int(seed)),
            "output_type": "pt",
        }
        if first_frame is not None:
            request["image"] = first_frame
        if self.task == "ref2va":
            request["references"] = list(references)

        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        state = self.pipe(**request)
        torch.cuda.synchronize(self.device)
        if self.world_size > 1:
            dist.barrier()
        elapsed = time.perf_counter() - started
        self._check_adaln()

        if not self.is_rank_zero:
            return None
        videos = _state_value(state, "videos")
        if videos is None:
            raise RuntimeError("pipeline returned no video")
        audio = _state_value(state, "audio")
        return GeneratedMedia(
            video=videos[0],
            audio=None if audio is None else audio[0],
            audio_sample_rate=_state_value(state, "sampling_rate"),
            elapsed_s=elapsed,
            duration=duration,
            seed=int(seed),
            stages=stage_timing.collect(),
        )

    def generate_to_file(self, output_path: str | Path, prompt: str, *, duration: int, seed: int = 0,
                         image: str | Path | Any | None = None,
                         references: list[Any] | None = None):
        """Generate and start background MP4 encoding on rank zero."""
        media = self.generate(prompt, duration=duration, seed=seed, image=image,
                              references=references)
        return None if media is None else media.save_async(output_path)

    def warmup(self, *, duration: int = 5, prompt: str | None = None,
               image: str | Path | Any | None = None,
               references: list[Any] | None = None) -> None:
        self.generate(
            prompt or "A continuous cinematic shot of morning light moving across a quiet room.",
            duration=duration,
            seed=0,
            image=image,
            references=references,
        )

    def close(self) -> None:
        sparse = getattr(self, "sparse_attention", None)
        if sparse is not None and getattr(sparse, "_call_stats", False) and sparse._call_count:
            print(f"[sol_attn_h3] final call stats (CPU-side): n={sparse._call_count} "
                  f"avg_total={sparse._call_total / sparse._call_count * 1e3:.2f} ms "
                  f"avg_inner={sparse._call_inner / sparse._call_count * 1e3:.2f} ms",
                  flush=True)
        if self._owns_process_group and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            self._owns_process_group = False

    def __enter__(self) -> "MiniMaxH3Inference":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
