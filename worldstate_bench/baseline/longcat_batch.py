"""LongCat-Video 13.6B I2V distilled 常驻批量: 480p, 93帧, 16步, seed=0。
需 torchrun --nproc_per_node=1 启动(demo 同款, dist 初始化依赖)。
环境变量: CODEBASE_ROOT(含 LongCat-Video/ 代码), MODELS_ROOT(含 LongCat-Video/ 权重),
见 baseline/env.sh。"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts_common import load_manifest, done  # noqa: E402

CODEBASE = Path(os.environ.get("CODEBASE_ROOT", "")) / "LongCat-Video"
MODELS = Path(os.environ.get("MODELS_ROOT", "")) / "LongCat-Video"
if not CODEBASE.is_dir() or not MODELS.is_dir():
    raise SystemExit(
        "Set CODEBASE_ROOT (containing LongCat-Video/ code) and MODELS_ROOT "
        "(containing LongCat-Video/ weights) — see baseline/env.sh")
sys.path.insert(0, str(CODEBASE))
import PIL.Image  # noqa: E402
import torch  # noqa: E404
import torch.distributed as dist  # noqa: E402
from torchvision.io import write_video  # noqa: E402
from diffusers.utils import load_image  # noqa: E402

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline  # noqa: E402
from longcat_video.context_parallel import context_parallel_util  # noqa: E402
from longcat_video.context_parallel.context_parallel_util import init_context_parallel  # noqa: E402
from longcat_video.modules.scheduling_flow_match_euler_discrete import (  # noqa: E402
    FlowMatchEulerDiscreteScheduler)
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan  # noqa: E402
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel  # noqa: E402
from transformers import AutoTokenizer, UMT5EncoderModel  # noqa: E402

MODEL = "longcat"
NEG = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, "
       "paintings, images, static, overall gray, worst quality, low quality, JPEG "
       "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
       "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
       "still picture, many people in the background, walking backwards")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
    init_context_parallel(context_parallel_size=1, global_rank=0, world_size=1)
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    ckpt = str(MODELS)
    tokenizer = AutoTokenizer.from_pretrained(ckpt, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(ckpt, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(ckpt, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ckpt, subfolder="scheduler", torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(ckpt, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
    pipe = LongCatVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder,
                                vae=vae, scheduler=scheduler, dit=dit)
    pipe.to(local_rank)
    pipe.dit.load_lora(os.path.join(ckpt, "lora/cfg_step_lora.safetensors"), "cfg_step_lora")
    pipe.dit.enable_loras(["cfg_step_lora"])
    if rank == 0:
        print("[longcat] model loaded", flush=True)

    entries = load_manifest(MODEL)
    if args.only:
        entries = [e for e in entries if args.only in f"{e['case']}-{e['side']}.{e['mode']}"]
    todo = [e for e in entries if not done(e)]
    print(f"[longcat] total={len(entries)} todo={len(todo)}", flush=True)

    t_all = time.time()
    for i, e in enumerate(todo, 1):
        if rank != 0:
            break
        image = load_image(e["image"])
        target_size = image.size
        generator = torch.Generator(device=local_rank)
        generator.manual_seed(0)
        t1 = time.time()
        frames = pipe.generate_i2v(
            image=image, prompt=e["prompt"], negative_prompt=NEG,
            resolution="480p", num_frames=93, num_inference_steps=16,
            use_distill=True, guidance_scale=1.0, generator=generator)[0]
        outp = Path(e["out"])
        outp.parent.mkdir(parents=True, exist_ok=True)
        proc = [PIL.Image.fromarray(fr).resize(target_size, PIL.Image.BICUBIC)
                for fr in (frames * 255).astype("uint8")]
        write_video(str(outp), torch.from_numpy(__import__("numpy").array(proc)),
                    fps=15, video_codec="libx264", options={"crf": "18"})
        el = time.time() - t1
        print(f"[longcat] {i}/{len(todo)} {e['case']}-{e['side']}.{e['mode']} "
              f"{el:.0f}s avg={(time.time()-t_all)/i:.0f}s eta={(time.time()-t_all)/i*(len(todo)-i)/3600:.1f}h",
              flush=True)
    dist.barrier()
    dist.destroy_process_group()
    print("[longcat] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
