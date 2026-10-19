"""Minimal MiniMax-H3 inference runtime for A800 (SM80) deployment.

Environment note: this machine's torch 2.10.0+cu130 stack fails every
bias-carrying `F.linear` through cuBLASLt (`CUBLAS_STATUS_NOT_INITIALIZED`
from `cublasLtMatmulAlgoGetHeuristic`, all shapes and dtypes). Routing addmm
through the legacy cuBLAS path fixes it at a negligible cost — the heavy DiT
GEMMs are bias-free anyway. Set before any biased linear runs.
"""

import os

os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")

from .engine import GeneratedMedia, MiniMaxH3Inference  # noqa: E402

__all__ = ["GeneratedMedia", "MiniMaxH3Inference"]
