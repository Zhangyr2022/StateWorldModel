#!/usr/bin/env bash
# h3-a800 launcher: bakes in the environment variables and defaults that this
# deployment was validated with.
#
# Usage:
#   ./run.sh <nproc> <output.mp4> [further infer.py args...]
# Example:
#   ./run.sh 8 output/demo_480p.mp4 --task t2v --duration 5 --res 480p \
#            --prompt-file prompt.txt --seed 42 --warmup
set -euo pipefail
cd "$(dirname "$0")"

NPROC="${1:?usage: run.sh <nproc> <output.mp4> [infer.py args...]}"
OUTPUT="${2:?usage: run.sh <nproc> <output.mp4> [infer.py args...]}"
shift 2

# deployment-default env vars: pre-set values win (a target machine can turn
# off the local workaround or pick a different mirror)
: "${HF_HUB_OFFLINE:=1}"                    # everything local; an unreachable huggingface.co would hang in TCP connect
: "${HF_ENDPOINT:=https://hf-mirror.com}"   # mirror used when offline mode is unset; also never hangs
: "${DISABLE_ADDMM_CUDA_LT:=1}"             # this torch cu130's cublasLt fails every bias-carrying F.linear
: "${TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING:=false}"
export HF_HUB_OFFLINE HF_ENDPOINT DISABLE_ADDMM_CUDA_LT TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING

case "$NPROC" in
  4|8) ;;  # the engine itself rejects other world sizes, but fail fast here
  *) echo "run.sh: nproc must be 4 or 8 (got $NPROC)"; exit 2 ;;
esac

# inject defaults only for flags the user did not pass (argparse is last-wins)
USER_ARGS="$*"
have() { case " $USER_ARGS " in *" $1 "*) return 0;; *) return 1;; esac; }
# resolve the task the same way argparse will: last `--task X` / `--task=X` wins
TASK=""
PREV=""
for a in "$@"; do
  case "$PREV" in --task) TASK="$a" ;; esac
  case "$a" in --task=*) TASK="${a#--task=}" ;; esac
  PREV="$a"
done
EXTRA=()
if ! have --model; then
  EXTRA+=(--model ./checkpoints/MiniMax-H3)
fi
if ! have --adapter; then
  # the two tasks fuse different adapters at different alphas: FastH3
  # dense-datafree (alpha 64) for t2v/i2v, the lightx2v ref2v turbo LoRA
  # (alpha 8, 4-step) for ref2va
  if [ "$TASK" = "ref2va" ]; then
    EXTRA+=(--adapter "./checkpoints/Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors")
  else
    EXTRA+=(--adapter ./checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors)
  fi
fi
if ! have --output; then
  EXTRA+=(--output "$OUTPUT")
fi

# prefer torchrun from PATH (i.e. an activated environment), H3_TORCHRUN
# overrides it; if neither exists, error out rather than guessing a path.
# note: a failing command substitution makes the assignment return nonzero,
# which would silently exit under set -e — hence the `|| true`.
TORCHRUN="$(command -v torchrun || true)"
TORCHRUN="${H3_TORCHRUN:-$TORCHRUN}"
if [ -z "$TORCHRUN" ]; then
  echo "run.sh: torchrun not found on PATH; activate the environment or set H3_TORCHRUN" >&2
  exit 2
fi
exec "$TORCHRUN" --standalone --nproc_per_node="$NPROC" infer.py \
  "${EXTRA[@]}" "$@"
