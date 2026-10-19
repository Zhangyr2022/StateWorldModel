# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import os
import sys
from pathlib import Path

# 环境变量: CODEBASE_ROOT(含 HunyuanVideo-1.5/ 代码), 见 baseline/env.sh
_CODEBASE = Path(os.environ.get("CODEBASE_ROOT", "")) / "HunyuanVideo-1.5"
if not _CODEBASE.is_dir():
    raise SystemExit(
        "Set CODEBASE_ROOT (containing HunyuanVideo-1.5/ code) — see baseline/env.sh")
sys.path.insert(0, str(_CODEBASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import copy
import datetime
import json

import loguru
import torch
import argparse
import einops
import imageio
from torch import distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state

parallel_dims = initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

def save_video(video, path):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid, fps=24)

def rank0_log(message, level):
    if int(os.environ.get('RANK', '0')) == 0:
        loguru.logger.log(level, message)

def save_config(args, output_path, task, transformer_version):
    arguments = {}
    for key, value in vars(args).items():
        if not key.startswith('_') and not callable(value):
            try:
                json.dumps(value)
                arguments[key] = value
            except (TypeError, ValueError):
                arguments[key] = str(value)
    
    config = {
        'timestamp': datetime.datetime.now().isoformat(),
        'task': task,
        'transformer_version': transformer_version,
        'output_path': output_path,
        'arguments': arguments
    }
    
    base_path, _ = os.path.splitext(output_path)
    config_path = f"{base_path}_config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Saved generation config to: {config_path}")
    return config_path

def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True  # When --flag is provided without value, enable it
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")

def load_checkpoint_to_transformer(pipe, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    rank0_log(f"Loading checkpoint from {checkpoint_path}", "INFO")
    
    try:
        model_state_dict = get_model_state_dict(pipe.transformer)
        dcp.load(
            state_dict={"model": model_state_dict},
            checkpoint_id=checkpoint_path,
        )
        rank0_log("Transformer model state loaded successfully", "INFO")
    except Exception as e:
        rank0_log(f"Error loading checkpoint: {e}", "ERROR")
        raise

def load_lora_adapter(pipe, lora_path):
    rank0_log(f"Loading LoRA adapter from {lora_path}", "INFO")
    try:
        pipe.transformer.load_lora_adapter(
            pretrained_model_name_or_path_or_dict=lora_path,
            prefix=None,
            adapter_name="default",
            use_safetensors=True,
            hotswap=False,
        )
        rank0_log("LoRA adapter loaded successfully", "INFO")
    except Exception as e:
        rank0_log(f"Error loading LoRA adapter: {e}", "ERROR")
        raise
    

def generate_video(args):

    infer_state = initialize_infer_state(args)

    if args.sparse_attn and args.use_sageattn:
        raise ValueError("sparse_attn and use_sageattn cannot be enabled simultaneously. Please enable only one of them.")

    if args.use_fp8_gemm and 'sgl' in args.quant_type:
        try:
            import sgl_kernel
        except Exception:
            raise ValueError("sgl_kernel is not installed. Please install it using `pip install sgl-kernel==0.3.18`")
    
    # if args.enable_torch_compile:
    #     torch._logging.set_logs(graph_code=True)
    
    if args.enable_step_distill and args.enable_cache:
        raise ValueError("Enabling both step distilled model and cache will lead to performance degradation.")
    
    task = 'i2v'  # worldstate_bench: 恒为 i2v(首帧驱动)
    
    enable_sr = args.sr
    
    transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(args.resolution, task, args.cfg_distilled, args.enable_step_distill, args.sparse_attn)
    
    if args.dtype == 'bf16':
        transformer_dtype = torch.bfloat16
    elif args.dtype == 'fp32':
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Must be 'bf16' or 'fp32'")
    
    # Determine offloading settings
    enable_offloading = args.offloading
    if args.group_offloading is None:
        # Auto-detect based on offloading config
        offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
        enable_group_offloading = offloading_config['enable_group_offloading']
    else:
        enable_group_offloading = args.group_offloading
    
    overlap_group_offloading = args.overlap_group_offloading
    
    # Determine device and transformer_init_device based on offloading settings
    if enable_offloading:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    if enable_group_offloading:
        transformer_init_device = torch.device('cpu')
    else:
        transformer_init_device = device
    
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        create_sr_pipeline=enable_sr,
        transformer_dtype=transformer_dtype,
        device=device,
        transformer_init_device=transformer_init_device,
    )
    
    loguru.logger.info(f"{enable_offloading=} {enable_group_offloading=} {overlap_group_offloading=}")
    
    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=enable_offloading,
        enable_group_offloading=enable_group_offloading,
        overlap_group_offloading=overlap_group_offloading,
    )
    
    # Load checkpoint if provided
    if args.checkpoint_path:
        load_checkpoint_to_transformer(pipe, args.checkpoint_path)
    
    if args.lora_path:
        load_lora_adapter(pipe, args.lora_path)

    # Apply optimizations to SR pipeline if exists
    if enable_sr and hasattr(pipe, 'sr_pipeline'):
        sr_infer_state = copy.deepcopy(infer_state)
        sr_infer_state.enable_cache = False # SR pipeline does not require cache optimization yet
        pipe.sr_pipeline.apply_infer_optimization(
            infer_state=sr_infer_state,
            enable_offloading=enable_offloading,
            enable_group_offloading=enable_group_offloading,
            overlap_group_offloading=overlap_group_offloading,
        )

    # ---- worldstate_bench batch loop (raw / raw_state manifest, seed=0) ----
    from prompts_common import load_manifest, done

    entries = load_manifest("hy15")
    if getattr(args, "only", None):
        entries = [e for e in entries if args.only in f"{e['case']}-{e['side']}.{e['mode']}"]
    todo = [e for e in entries if not done(e)]
    rank0_log(f"hy15 total={len(entries)} todo={len(todo)}", "INFO")

    import time as _t
    t_all = _t.time()
    for i, e in enumerate(todo, 1):
        t1 = _t.time()
        out = pipe(
            enable_sr=False,
            prompt=e["prompt"],
            aspect_ratio=args.aspect_ratio,
            num_inference_steps=args.num_inference_steps,
            sr_num_inference_steps=None,
            video_length=args.video_length,
            negative_prompt=args.negative_prompt,
            seed=0,
            output_type="pt",
            prompt_rewrite=False,
            return_pre_sr_video=False,
            reference_image=e["image"],
        )
        Path(e["out"]).parent.mkdir(parents=True, exist_ok=True)
        save_video(out.videos, e["out"])
        del out
        el = _t.time() - t1
        avg = (_t.time() - t_all) / i
        rank0_log(f"hy15 {i}/{len(todo)} {e['case']}-{e['side']}.{e['mode']} "
                  f"{el:.0f}s avg={avg:.0f}s eta={avg * (len(todo) - i) / 3600:.1f}h", "INFO")
    rank0_log("hy15 ALL DONE", "INFO")

def main():
    parser = argparse.ArgumentParser(description='Generate video using HunyuanVideo-1.5')
    parser.add_argument('--only', type=str, default=None)

    parser.add_argument(
        '--prompt', type=str, default='',
        help='Text prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt', type=str, default='',
        help='Negative prompt for video generation (default: empty string)'
    )
    parser.add_argument(
        '--resolution', type=str, choices=['480p', '720p'],
        help='Video resolution (480p or 720p)'
    )
    parser.add_argument(
        '--model_path', type=str,
        help='Path to pretrained model'
    )
    parser.add_argument(
        '--aspect_ratio', type=str, default='16:9',
        help='Aspect ratio (default: 16:9)'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=None,
        help='Number of inference steps (default: 50)'
    )
    parser.add_argument(
        '--video_length', type=int, default=121,
        help='Number of frames to generate (default: 121)'
    )
    parser.add_argument(
        '--sr', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable super resolution (default: true). '
             'Use --sr or --sr true/1 to enable, --sr false/0 to disable'
    )
    parser.add_argument(
        '--save_pre_sr_video', type=str_to_bool, nargs='?', const=True, default=False,
        help='Save original video before super resolution (default: false). '
             'Use --save_pre_sr_video or --save_pre_sr_video true/1 to enable, '
             '--save_pre_sr_video false/0 to disable'
    )
    parser.add_argument(
        '--rewrite', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable prompt rewriting (default: true). '
             'Use --rewrite or --rewrite true/1 to enable, --rewrite false/0 to disable'
    )
    parser.add_argument(
        '--cfg_distilled', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable CFG distilled model (default: false). '
             'Use --cfg_distilled or --cfg_distilled true/1 to enable, '
             '--cfg_distilled false/0 to disable'
    )
    parser.add_argument(
        '--enable_step_distill', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable step distilled model (default: false). '
             'Use --enable_step_distill or --enable_step_distill true/1 to enable, '
             '--enable_step_distill false/0 to disable'
    )
    parser.add_argument(
        '--sparse_attn', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable sparse attention (default: false). '
             'Use --sparse_attn or --sparse_attn true/1 to enable, '
             '--sparse_attn false/0 to disable'
    )
    parser.add_argument(
        '--offloading', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable offloading (default: true). '
             'Use --offloading or --offloading true/1 to enable, '
             '--offloading false/0 to disable'
    )
    parser.add_argument(
        '--group_offloading', type=str_to_bool, nargs='?', const=True, default=None,
        help='Enable group offloading (default: None, automatically enabled if offloading is enabled). '
             'Use --group_offloading or --group_offloading true/1 to enable, '
             '--group_offloading false/0 to disable'
    )
    parser.add_argument(
        '--overlap_group_offloading', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable overlap group offloading (default: true). '
             'Significantly increases CPU memory usage but speeds up inference. '
             'Use --overlap_group_offloading or --overlap_group_offloading true/1 to enable, '
             '--overlap_group_offloading false/0 to disable'
    )
    parser.add_argument(
        '--dtype', type=str, default='bf16', choices=['bf16', 'fp32'],
        help='Data type for transformer (default: bf16). '
             'bf16: faster, lower memory; fp32: better quality, slower, higher memory'
    )
    parser.add_argument(
        '--seed', type=int, default=123,
        help='Random seed (default: 123)'
    )
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='Path to reference image for i2v (if provided, uses i2v mode)'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Output file path for generated video (if not provided, saves to ./outputs/output.mp4)'
    )
    parser.add_argument(
        '--use_sageattn', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable sageattn (default: false). '
             'Use --use_sageattn or --use_sageattn true/1 to enable, '
             '--use_sageattn false/0 to disable'
    )
    parser.add_argument(
        '--sage_blocks_range', type=str, default="0-53",
        help='Sageattn blocks range (e.g., 0-5 or 0,1,2,3,4,5)'
    )
    parser.add_argument(
        '--enable_torch_compile', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable torch compile for transformer (default: false). '
             'Use --enable_torch_compile or --enable_torch_compile true/1 to enable, '
             '--enable_torch_compile false/0 to disable'
    )
    parser.add_argument(
        '--enable_cache', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable cache for transformer (default: false). '
             'Use --enable_cache or --enable_cache true/1 to enable, '
             '--enable_cache false/0 to disable'
    )
    parser.add_argument(
        '--cache_type', type=str, default="deepcache",
        help='Cache type for transformer (e.g., deepcache, teacache, taylorcache)'
    )
    parser.add_argument(
        '--no_cache_block_id', type=str, default="53",
        help='Blocks to exclude from deepcache (e.g., 0-5 or 0,1,2,3,4,5)'
    )
    parser.add_argument(
        '--cache_start_step', type=int, default=11,
        help='Start step to skip when using cache (default: 11)'
    )
    parser.add_argument(
        '--cache_end_step', type=int, default=45,
        help='End step to skip when using cache (default: 45)'
    )
    parser.add_argument(
        '--total_steps', type=int, default=50,
        help='Total inference steps (default: 50)'
    )
    parser.add_argument(
        '--cache_step_interval', type=int, default=4,
        help='Step interval to skip when using cache (default: 4)'
    )
    parser.add_argument(
        '--save_generation_config', type=str_to_bool, nargs='?', const=True, default=True,
        help='Save generation config file (default: true). '
             'Use --save_generation_config or --save_generation_config true/1 to enable, '
             '--save_generation_config false/0 to disable'
    )
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Path to checkpoint directory containing transformer weights (e.g., ./outputs/checkpoint-1000/transformer). '
             'The checkpoint directory should contain a "transformer" subdirectory. '
             'If provided, the transformer model weights will be loaded from this checkpoint.'
    )
    parser.add_argument(
        '--lora_path', type=str, default=None,
        help='Path to LoRA adapter directory or checkpoint directory containing LoRA adapter. '
             'If provided, the LoRA adapter will be loaded to the transformer model.'
    )

    # fp8 gemm related
    parser.add_argument(
        '--use_fp8_gemm', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable fp8 gemm for transformer (default: false). '
             'Use --use_fp8_gemm or --use_fp8_gemm true/1 to enable, '
             '--use_fp8_gemm false/0 to disable'
    )
    parser.add_argument(
        '--quant_type', type=str, default="fp8-per-token-sgl",
        help='Quantization type for fp8 gemm (e.g., fp8-per-tensor-weight-only, fp8-per-tensor, fp8-per-token-sgl)'
    )
    parser.add_argument(
        '--include_patterns', type=str, default="double_blocks",
        help='Include patterns for fp8 gemm (default: double_blocks)'
    )

    args = parser.parse_args()
    
    # Convert string "none" to None for image_path
    if args.image_path is not None and args.image_path.lower().strip() == 'none':
        args.image_path = None
    
    
    generate_video(args)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
