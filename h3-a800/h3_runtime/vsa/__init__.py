"""VSA (Video Sparse Attention) for MiniMax-H3 on A800.

Port of FastVideo's VSA-H3 backend (Apache-2.0) onto our diffusers + Ulysses
stack: the trained tile topology (90% video-tile sparsity, tile 64), the
learned per-row gate branch, and the vendored Triton block-sparse kernel
(`block_sparse_attn_triton.py`, validated on SM80 in Phase 0).
"""
