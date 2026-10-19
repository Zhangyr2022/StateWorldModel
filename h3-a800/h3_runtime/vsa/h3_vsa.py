"""The VSA-H3 attention core: tile geometry, topology mask, and the callable
that our Ulysses exchange feeds.

Semantics ported from FastVideo's `video_sparse_attn_h3.py` (Apache-2.0):

* tiles = segment-pure prefix chunks ([text | conditioning | audio]) followed
  by 3-D video tiles `(4, 4, 4)` with variable-size edge tiles,
* prefix queries are always dense and prefix keys are always selected
  (exempt mode); each video query tile keeps `ceil((1-sparsity) * n_video)`
  video key tiles chosen by pooled-Q/pooled-K scores,
* a learned per-row gate mixes a tile-level compressed attention branch into
  the sparse output (zero gates reduce it away; the VSA-distilled adapter
  ships trained gates),
* selection and execution run after the Ulysses all-to-all, on the full
  sequence with this rank's heads — the topology is per-head, so ranks never
  need to agree on anything.

This file is deliberately framework-free: geometry in, kernel out.
"""

from __future__ import annotations

import functools
import math
import os
import time

import torch
import triton

from .block_sparse_attn_triton import triton_block_sparse_attn_forward

_PROFILE = os.environ.get("H3_VSA_PROFILE", "0") == "1"


def _lap(t0: float | None, label: str) -> float | None:
    if t0 is None:
        return None
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"[vsa.profile] {label:14s} {(t1 - t0) * 1000:7.2f} ms", flush=True)
    return t1

TILE_SHAPE = (4, 4, 4)  # (t, h, w) patches per video tile = 64 tokens
TILE_ELEMS = math.prod(TILE_SHAPE)


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) == "1"


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# ----------------------------------------------------------------- geometry


def compute_topk(sparsity: float, num_blocks: int) -> int:
    """Blocks to keep for a sparsity level, clamped to [1, num_blocks]."""
    return max(1, min(math.ceil((1 - sparsity) * num_blocks), num_blocks))


def get_tile_partition_indices(seq_shape: tuple[int, int, int],
                               tile_shape: tuple[int, int, int]) -> torch.Tensor:
    """Tile-order row indices for a (t, h, w) row-major grid."""
    big_t, big_h, big_w = seq_shape
    ts, hs, ws = tile_shape
    indices = torch.arange(big_t * big_h * big_w, dtype=torch.long).reshape(big_t, big_h, big_w)
    chunks = []
    for t in range(math.ceil(big_t / ts)):
        for h in range(math.ceil(big_h / hs)):
            for w in range(math.ceil(big_w / ws)):
                chunks.append(
                    indices[t * ts:min(t * ts + ts, big_t),
                            h * hs:min(h * hs + hs, big_h),
                            w * ws:min(w * ws + ws, big_w)].flatten())
    return torch.cat(chunks)


def construct_variable_block_sizes(seq_shape: tuple[int, int, int],
                                   tile_shape: tuple[int, int, int]) -> torch.Tensor:
    """Valid token count per tile; edge tiles may be smaller."""
    big_t, big_h, big_w = seq_shape
    ts, hs, ws = tile_shape

    def sizes(dim_len: int, tile: int) -> torch.Tensor:
        n = math.ceil(dim_len / tile)
        out = torch.full((n,), tile, dtype=torch.long)
        remainder = dim_len - (n - 1) * tile
        out[-1] = remainder if remainder > 0 else tile
        return out

    t_sizes = sizes(big_t, ts)
    h_sizes = sizes(big_h, hs)
    w_sizes = sizes(big_w, ws)
    return (t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]).reshape(-1)


def get_non_pad_index(variable_block_sizes: torch.Tensor, max_block_size: int) -> torch.Tensor:
    n = variable_block_sizes.shape[0]
    starts = torch.arange(n) * max_block_size
    grid = starts[:, None] + torch.arange(max_block_size)[None, :]
    keep = torch.arange(max_block_size)[None, :] < variable_block_sizes[:, None]
    return grid[keep]


class Geometry:
    """Cached tile layout for one request shape.

    * `slot_to_row`: for every *valid* slot (tile order), the packed row it
      holds. Length = number of packed rows.
    * `non_pad`: padded-buffer positions of the valid slots.
    * `untile`: for every packed row, its padded slot (`argsort(slot_to_row)`).
    """

    def __init__(self, prefix_segments: tuple[int, ...], video_seq_shape: tuple[int, int, int]):
        prefix_sizes: list[int] = []
        for segment in prefix_segments:
            full, rem = divmod(segment, TILE_ELEMS)
            prefix_sizes.extend([TILE_ELEMS] * full)
            if rem:
                prefix_sizes.append(rem)
        self.num_prefix_tiles = len(prefix_sizes)

        video_indices = get_tile_partition_indices(video_seq_shape, TILE_SHAPE)
        self.variable_block_sizes = torch.cat([
            torch.tensor(prefix_sizes, dtype=torch.long),
            construct_variable_block_sizes(video_seq_shape, TILE_SHAPE),
        ])
        self.slot_to_row = torch.cat([
            torch.arange(sum(prefix_segments), dtype=torch.long),
            video_indices + sum(prefix_segments),
        ])
        self.n_tiles = self.variable_block_sizes.shape[0]
        self.num_video_tiles = self.n_tiles - self.num_prefix_tiles
        self.non_pad = get_non_pad_index(self.variable_block_sizes, TILE_ELEMS)
        # packed row -> PADDED slot: valid-slot index first, then its padded
        # position (upstream's untile_combined_index construction)
        self.untile = self.non_pad[torch.argsort(self.slot_to_row)]
        assert self.slot_to_row.shape[0] == self.non_pad.shape[0]
        self.seq_pad = self.n_tiles * TILE_ELEMS

    def to(self, device: torch.device) -> "Geometry":
        moved = Geometry.__new__(Geometry)
        for name in ("variable_block_sizes", "slot_to_row", "non_pad", "untile"):
            setattr(moved, name, getattr(self, name).to(device, non_blocking=True))
        moved.num_prefix_tiles = self.num_prefix_tiles
        moved.n_tiles = self.n_tiles
        moved.num_video_tiles = self.num_video_tiles
        moved.seq_pad = self.seq_pad
        return moved


@functools.lru_cache(maxsize=16)
def _tile_geometry(prefix_segments: tuple[int, ...], video_seq_shape: tuple[int, int, int]):
    return Geometry(prefix_segments, video_seq_shape)


def _pool_tiles(x_tiled: torch.Tensor, variable_block_sizes: torch.Tensor) -> torch.Tensor:
    """fp32 tile means. x_tiled: (1, H, n_tiles * tile, D) -> (1, H, n_tiles, D).

    Pad slots are zeroed at tiling time, so a sum over the padded window
    divided by the tile's true size is exactly the masked mean.
    """
    n_tiles = variable_block_sizes.shape[0]
    pooled = x_tiled.view(1, x_tiled.shape[1], n_tiles, TILE_ELEMS, x_tiled.shape[-1])
    pooled = pooled.sum(dim=3, dtype=torch.float32)
    return pooled / variable_block_sizes.view(1, 1, -1, 1)


def build_block_mask(scores: torch.Tensor, num_prefix_tiles: int, num_video_tiles: int,
                     sparsity: float, exempt: bool) -> torch.Tensor:
    """scores (1, H, n_tiles, n_tiles) -> bool keep-mask."""
    n_tiles = scores.shape[-1]
    k_video = compute_topk(sparsity, num_video_tiles)
    if k_video >= num_video_tiles:
        return torch.ones_like(scores, dtype=torch.bool)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if exempt or num_prefix_tiles == 0:
        video_cols = scores[..., num_prefix_tiles:]
        idx = video_cols.topk(k_video, dim=-1).indices + num_prefix_tiles
        mask.scatter_(-1, idx, True)
        mask[..., :num_prefix_tiles] = True
    else:
        k_total = min(k_video + num_prefix_tiles, n_tiles)
        idx = scores.topk(k_total, dim=-1).indices
        mask.scatter_(-1, idx, True)
    mask[:, :, :num_prefix_tiles, :] = True
    return mask


def map_to_index(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bool block mask (1, H, Nq, Nk) -> (q2k_index int32 with -1 pad, q2k_num)."""
    counts = mask.sum(-1)
    max_n = int(counts.max().item())
    order = torch.argsort(mask.to(torch.int8), dim=-1, descending=True, stable=True)
    order = order[..., :max_n]
    arange = torch.arange(max_n, device=mask.device).view(1, 1, 1, -1)
    index = torch.where(arange < counts[..., None], order, torch.full_like(order, -1))
    return index.to(torch.int32), counts.to(torch.int32)


# ------------------------------------------------------------- request state


class RequestState:
    """Per-request geometry, the layer/step clock (blocks run strictly in
    order, so a counter recovers both), and reusable tile buffers."""

    def __init__(self) -> None:
        self.reset()
        self.prefix_segments: tuple[int, ...] | None = None
        self.video_seq_shape: tuple[int, int, int] | None = None
        self._buffers: dict[tuple, torch.Tensor] = {}

    def reset(self) -> None:
        self.clock = 0
        self.steps = _env_int("H3_VSA_INFERENCE_STEPS", 4)
        self.layers = _env_int("H3_VSA_NUM_LAYERS", 50)

    def set_layout(self, prefix_segments: tuple[int, ...], video_seq_shape: tuple[int, int, int]) -> None:
        self.prefix_segments = prefix_segments
        self.video_seq_shape = video_seq_shape
        self._buffers.clear()
        self.reset()

    def next_layer(self) -> tuple[int, int]:
        step, layer = divmod(self.clock, self.layers)
        self.clock += 1
        return step, layer

    def scratch(self, tag: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """A zeroed scratch buffer, reused across layers and steps. `tag`
        keeps simultaneously-live buffers (q/k/v/gate) apart."""
        key = (tag, shape, dtype, device)
        buf = self._buffers.get(key)
        if buf is None:
            buf = torch.zeros(shape, dtype=dtype, device=device)
            self._buffers[key] = buf
        else:
            buf.zero_()
        return buf


_STATE: RequestState | None = None


def state() -> RequestState:
    global _STATE
    if _STATE is None:
        _STATE = RequestState()
    return _STATE


def set_request_layout(prefix_segments: tuple[int, ...], video_seq_shape: tuple[int, int, int]) -> None:
    state().set_layout(prefix_segments, video_seq_shape)


def reset_clock() -> None:
    state().reset()


# ------------------------------------------------------------- the attention


def _sparsity_knobs() -> tuple[float, bool]:
    return (
        _env_float("H3_VSA_SPARSITY", 0.9),
        _env_flag("H3_VSA_EXEMPT", "1"),
    )


@torch.no_grad()
def vsa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  gate: torch.Tensor | None = None) -> torch.Tensor:
    """One VSA attention call, post-Ulysses layout.

    q/k/v: (S, H_local, 128) strided views; gate: (S, H_local, 128) or None.
    Returns (S, H_local, 128).

    Structure: prefix rows ([text|cond|audio], packed-contiguous at the head of
    the sequence) are semantically dense and run through flash SDPA over the
    packed K/V directly; only the video rows go through the tile scatter and
    the block-sparse kernel — which keeps the kernel's longest program at
    ~56 block iterations instead of 544, eliminating the latency-bound tail.
    """
    _t0 = time.perf_counter() if _PROFILE else None
    request = state()
    assert request.prefix_segments is not None, "VSA request layout was never set"
    layer_step, layer_idx = request.next_layer()

    dense_first_steps = _env_int("H3_VSA_DENSE_FIRST_STEPS", 1)
    dense_layers = _env_int("H3_VSA_DENSE_LAYERS", 2)
    force_dense = layer_step < dense_first_steps or layer_idx < dense_layers
    sparsity, exempt = _sparsity_knobs()
    if force_dense:
        sparsity, exempt = 0.0, True  # keep every tile

    seq, heads, dim = q.shape
    geometry = _tile_geometry(
        request.prefix_segments, request.video_seq_shape).to(q.device)
    vbs = geometry.variable_block_sizes
    n_v_tiles = geometry.num_video_tiles
    prefix_rows = sum(request.prefix_segments)
    video_rows = seq - prefix_rows
    npt_slots = geometry.num_prefix_tiles * TILE_ELEMS  # padded prefix slots

    # video-segment tile indices. Two boundaries:
    #   valid-slot / packed-row: prefix_rows (every prefix row is valid)
    #   padded-position / tile-count: num_prefix_tiles * TILE_ELEMS
    video_slot_to_row = geometry.slot_to_row[prefix_rows:] - prefix_rows
    video_non_pad = geometry.non_pad[prefix_rows:] - npt_slots
    vbs_video = vbs[geometry.num_prefix_tiles:]

    _t1 = _lap(_t0, "setup")

    # 1. prefix rows: dense SDPA over all packed rows (no padding, no mask).
    prefix_out = torch.nn.functional.scaled_dot_product_attention(
        q[:prefix_rows].transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        dropout_p=0.0, is_causal=False)[0].transpose(0, 1)  # (prefix_rows, H, D)
    _t2 = _lap(_t1, "prefix SDPA")

    # 2. video rows: only the QUERIES are video-only and tiled; the K/V stay
    # tiled over the FULL sequence, because prefix keys are exempt-selected
    # (always attended) and appear in every video query row's key set.
    def tile_full(x: torch.Tensor, tag: str) -> torch.Tensor:
        buf = request.scratch(tag, (1, heads, geometry.seq_pad, dim), x.dtype, x.device)
        src = x.transpose(0, 1).unsqueeze(0)
        buf[:, :, geometry.non_pad] = src[:, :, geometry.slot_to_row]
        return buf

    def tile_video_q(x: torch.Tensor) -> torch.Tensor:
        buf = request.scratch('q', (1, heads, n_v_tiles * TILE_ELEMS, dim), x.dtype, x.device)
        src = x[prefix_rows:].transpose(0, 1).unsqueeze(0)
        buf[:, :, video_non_pad] = src[:, :, video_slot_to_row]
        return buf

    qv = tile_video_q(q)
    kt = tile_full(k, 'k')
    vt = tile_full(v, 'v')
    _t3 = _lap(_t2, "video tile q + full tile k/v")

    if sparsity <= 0.0:
        q_pooled = k_pooled = None
        mask = torch.ones((1, heads, n_v_tiles, geometry.n_tiles), dtype=torch.bool, device=q.device)
    else:
        q_pooled = _pool_tiles(qv, vbs_video)
        k_pooled = _pool_tiles(kt, vbs)
        tile_scores = torch.matmul(q_pooled, k_pooled.transpose(-1, -2)) / math.sqrt(dim)
        # video query rows: top-k among video key columns, prefix key columns
        # exempt-selected. (All query rows here are video rows — the prefix
        # rows already went through the dense SDPA above.)
        k_video = compute_topk(sparsity, n_v_tiles)
        mask = torch.zeros_like(tile_scores, dtype=torch.bool)
        if exempt:
            # default quality profile: every video query row keeps ALL prefix
            # key tiles and ranks only the video key tiles against each other
            vidx = tile_scores[..., geometry.num_prefix_tiles:] \
                .topk(k_video, dim=-1).indices + geometry.num_prefix_tiles
            mask.scatter_(-1, vidx, True)
            mask[..., :geometry.num_prefix_tiles] = True
        else:
            # FLOP-matched compete mode: video query rows rank prefix and
            # video key tiles together under a single budget
            k_total = min(k_video + geometry.num_prefix_tiles, geometry.n_tiles)
            vidx = tile_scores.topk(k_total, dim=-1).indices
            mask.scatter_(-1, vidx, True)
        del tile_scores
    _t4 = _lap(_t3, "pool+scores+mask")
    q2k_index, q2k_num = map_to_index(mask)
    del mask
    _t5 = _lap(_t4, "map_to_index")

    if _PROFILE:
        torch.cuda.synchronize(); _tk = time.perf_counter()
    video_out_tiled, _ = triton_block_sparse_attn_forward(
        qv, kt, vt, q2k_index, q2k_num, vbs.to(torch.int32))
    if _PROFILE:
        torch.cuda.synchronize()
        print(f"[vsa.profile]   kernel alone: {(time.perf_counter()-_tk)*1000:7.2f} ms", flush=True)
    video_untile = geometry.untile[prefix_rows:] - npt_slots
    video_out = video_out_tiled[:, :, video_untile]  # (1, H, video_rows, D), packed order
    if _PROFILE:
        torch.cuda.synchronize(); print(f"[vsa.profile]   untile: {(time.perf_counter()-_tk)*1000:7.2f} ms cum", flush=True)

    # 3. gate branch: tile-level compressed attention over the full tiled K/V,
    # applied per packed row through the learned gate — for ALL rows (prefix
    # rows included), matching the FastVideo semantics.
    if gate is not None:
        gate_video = gate[prefix_rows:]
        if q_pooled is None:
            q_pooled = _pool_tiles(qv, vbs_video)
            k_pooled = _pool_tiles(kt, vbs)
        v_pooled = _pool_tiles(vt, vbs)
        tile_scores = torch.matmul(q_pooled, k_pooled.transpose(-1, -2)) / math.sqrt(dim)
        out_c = torch.matmul(torch.softmax(tile_scores, dim=-1), v_pooled)  # (1, H, n_tiles, D)
        # packed video row -> its (video-)query-tile index into out_c
        tile_of_row = (geometry.untile[prefix_rows:] - npt_slots) // TILE_ELEMS
        # per packed video row: its tile's compressed attention, scaled by the row's gate
        contrib = out_c[0][:, tile_of_row, :]  # (H, video_rows, D)
        video_out = video_out + contrib.unsqueeze(0).to(video_out.dtype) \
            * gate_video.permute(1, 0, 2).unsqueeze(0).to(video_out.dtype)

        # prefix rows: their tiles are the leading prefix tiles of the full
        # tiling; pooled prefix Q against the same full pooled K/V. Prefix
        # tiles are segment-pure and contiguous in packed order.
        vbs_prefix = vbs[:geometry.num_prefix_tiles]
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=q.device),
                            torch.cumsum(vbs_prefix, 0)[:-1]])
        q_prefix = q[:prefix_rows].transpose(0, 1).unsqueeze(0)  # (1, H, prefix_rows, D)
        q_pooled_prefix = torch.zeros(1, heads, geometry.num_prefix_tiles, dim,
                                      dtype=torch.float32, device=q.device)
        for t in range(geometry.num_prefix_tiles):
            rows = slice(int(starts[t]), int(starts[t] + vbs_prefix[t]))
            q_pooled_prefix[:, :, t] = q_prefix[:, :, rows].mean(dim=2)
        out_c_prefix = torch.matmul(
            torch.softmax(q_pooled_prefix @ k_pooled.transpose(-1, -2) / math.sqrt(dim), dim=-1),
            v_pooled)  # (1, H, npt, D)
        tile_of_prefix = (geometry.untile[:prefix_rows]) // TILE_ELEMS
        contrib_p = out_c_prefix[0][:, tile_of_prefix, :]  # (H, prefix_rows, D)
        prefix_out = prefix_out + contrib_p.permute(1, 0, 2).to(prefix_out.dtype) \
            * gate[:prefix_rows].to(prefix_out.dtype)
        del tile_scores, out_c, v_pooled, q_pooled, k_pooled

    _t8 = _lap(_t5, "gate branch")

    out = torch.empty(1, heads, seq, dim, dtype=q.dtype, device=q.device)
    out[:, :, :prefix_rows] = prefix_out.to(q.dtype).transpose(0, 1).unsqueeze(0)
    out[:, :, prefix_rows:] = video_out.to(q.dtype)
    del qv, kt, vt, video_out
    _lap(_t8, "assemble")
    return out[0].transpose(0, 1)  # (S, H_local, D)


def make_vsa_attention_fn():
    """Factory matching the ulysses `attention_fn(q, k, v, gate=None)` contract."""
    return vsa_attention


def load_vsa_gates(transformer, adapter_path: str, verbose: bool = True) -> int:
    """Copy the adapter's `to_gate_compress.set_weight` payloads into the gate
    Linears (which the engine pre-creates, zero-initialized, after the AdaLN
    eviction — the memory-ordering constraint keeps this pass separate)."""
    import re

    from safetensors import safe_open

    filled = 0
    with safe_open(adapter_path, framework="pt") as handle:
        for key in handle.keys():
            if ".to_gate_compress.set_weight" not in key:
                continue
            match = re.search(r"transformer_blocks\.(\d+)\.", key)
            if match is None:
                continue
            attn = transformer.transformer_blocks[int(match.group(1))].attn
            module = getattr(attn, "to_gate_compress", None)
            if module is None:
                raise RuntimeError("gate Linears were not pre-created by the engine")
            with torch.no_grad():
                module.weight.copy_(handle.get_tensor(key).to(device=module.weight.device,
                                                              dtype=module.weight.dtype))
            filled += 1
    if verbose:
        print(f"[h3a800.vsa] loaded {filled} to_gate_compress gate weights", flush=True)
    return filled
