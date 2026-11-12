"""Where the GPU time of one engine step goes: each stage of BatchStepper._run captured as its own CUDA graph,
replayed and timed with CUDA events, next to a replay of the real whole-step graph.

Stages follow _run for a batch of later chunks: log-mel, subsampling (+ its cache gather/scatter), cache gather
(K/V, conv, kv_len, prompt one-hot, mask), the 24 Conformer layers split by component, projectors, cache scatter,
greedy (12 steps) and one continuation graph (4 steps). Stage inputs are static tensors computed once, so the sum
of the stages is the step without the boundaries between them.

    python bench/profiling/graph_breakdown.py --batch 64 32 --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)

# isort: split

from nemotron_engine.batch import BatchStepper  # noqa: E402
from nemotron_engine.model import NEG_INF, NemotronModel  # noqa: E402


def time_graph(fn, iters: int) -> float:
    """Capture fn as a CUDA graph, replay it; mean ms per replay."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


@torch.no_grad()
def breakdown(m: NemotronModel, st: BatchStepper, bp: int, iters: int, do_stages: bool = True) -> dict:
    C, T, L = st.cache_len, st.chunk, len(m.encoder.layers)
    for s in range(bp):
        st.reset(s, "vi-VN")
    st._idx[:bp] = torch.arange(bp, device=st.device)
    st._win[False][:bp] = torch.randn_like(st._win[False][:bp]) * 0.05
    idx = st._idx[:bp]
    kb, vb = st._kb[:bp], st._vb[:bp]
    layers = m.encoder.layers

    # static intermediates, computed once eagerly
    mel = m.frontend(st._win[False][:bp]).to(st.dtype)
    x0, _ = m.encoder.subsampling(mel, [s.index_select(0, idx) for s in st.sub])
    conv0 = st.conv.index_select(0, idx)
    kv_len = st.kv_len.index_select(0, idx)
    cols = torch.arange(st.keys_pad, device=st.device)
    km = ~((cols[None, :] >= (C - kv_len)[:, None]) & (cols[None, :] < st.n_keys))[:, None, None, :]
    oh = st.lang.index_select(0, idx)
    a0 = layers[0].self_attn
    xa = layers[0].norm_self_att(x0)
    q, k, v = F.linear(xa, a0.w_qkv).view(bp, T, 3, a0.h, a0.dk).unbind(2)
    scale = 1.0 / math.sqrt(a0.dk)
    bias = torch.where(km, NEG_INF, torch.matmul((q + a0.bias_v).permute(2, 1, 0, 3), st.pos_keys[0]).permute(2, 0, 1, 3) * scale)
    att = torch.nn.functional.scaled_dot_product_attention((q + a0.bias_u).transpose(1, 2), kb[:, 0], vb[:, 0],
                                                           attn_mask=bias[..., : st.n_keys], scale=scale)
    y = m.encoder_projector(m.prompt_projector(x0, oh))

    def attn_qkv():
        for li, l in enumerate(layers):
            a = l.self_attn
            qq, kk, vv = F.linear(l.norm_self_att(x0), a.w_qkv).view(bp, T, 3, a.h, a.dk).unbind(2)
            kb[:, li, :, st.n_keys - T :] = kk.transpose(1, 2)
            vb[:, li, :, st.n_keys - T :] = vv.transpose(1, 2)

    def attn_bias():
        for li, l in enumerate(layers):
            a = l.self_attn
            bd = torch.matmul((q + a.bias_v).permute(2, 1, 0, 3), st.pos_keys[li]).permute(2, 0, 1, 3)
            torch.where(km, NEG_INF, bd * scale)

    def attn_sdpa():
        for li, l in enumerate(layers):
            torch.nn.functional.scaled_dot_product_attention((q + l.self_attn.bias_u).transpose(1, 2), kb[:, li], vb[:, li],
                                                             attn_mask=bias[..., : st.n_keys], scale=scale)

    def attn_out():
        for l in layers:
            x0 + l.self_attn.o_proj(att.transpose(1, 2).reshape(bp, T, -1))

    def scatter():
        st.k.index_copy_(0, idx, kb[:, :, :, T:])
        st.v.index_copy_(0, idx, vb[:, :, :, T:])
        st.conv.index_copy_(0, idx, conv0)
        st.kv_len.index_copy_(0, idx, torch.clamp(kv_len + T, max=C))

    def greedy(steps):
        def run():
            torch.index_select(st.pred_out, 0, idx, out=st._p[:bp])
            st._h[:, :bp] = st.h.index_select(1, idx)
            st._c[:, :bp] = st.c.index_select(1, idx)
            for buf in (st._t, st._sym, st._n):
                buf[:bp].zero_()
            st._lab[:bp].fill_(-1)
            st._greedy(bp, steps)
        return run

    st._y[:bp] = y
    stages = {
        "log-mel (preemphasis + STFT)": lambda: m.frontend(st._win[False][:bp]),
        "subsampling + cache": lambda: [p.index_copy_(0, idx, n) for p, n in zip(st.sub, m.encoder.subsampling(mel, [s.index_select(0, idx) for s in st.sub])[1])],
        "gather cache + mask": lambda: (torch.index_select(st.k, 0, idx, out=kb[:, :, :, :C]), torch.index_select(st.v, 0, idx, out=vb[:, :, :, :C]),
                                        st.conv.index_select(0, idx), st.kv_len.index_select(0, idx), st.lang.index_select(0, idx)),
        "FFN 1 (24 lớp)": lambda: [x0 + 0.5 * l.feed_forward1(l.norm_feed_forward1(x0)) for l in layers],
        "attn: QKV + ghi cache (24 lớp)": attn_qkv,
        "attn: bias rel-pos (24 lớp)": attn_bias,
        "attn: SDPA (24 lớp)": attn_sdpa,
        "attn: o_proj (24 lớp)": attn_out,
        "conv module (24 lớp)": lambda: [l.conv(l.norm_conv(x0), conv0[:, li]) for li, l in enumerate(layers)],
        "FFN 2 (24 lớp)": lambda: [x0 + 0.5 * l.feed_forward2(l.norm_feed_forward2(x0)) for l in layers],
        "LayerNorm ra (24 lớp)": lambda: [l.norm_out(x0) for l in layers],
        "prompt + encoder projector": lambda: m.encoder_projector(m.prompt_projector(x0, oh)),
        "scatter cache": scatter,
        "greedy 12 bước": greedy(st.greedy_steps),
    }
    res = {name: time_graph(fn, iters) for name, fn in stages.items()} if do_stages else {}
    cont = time_graph(greedy(st.chunk), iters)  # a continuation graph runs `chunk` steps (over max_batch rows)
    whole = time_graph(lambda: st._run("later", bp), iters)
    for _ in range(3):  # the same step run eagerly (cuda_graphs=False): host launches every kernel
        st._run("later", bp)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        st._run("later", bp)
    torch.cuda.synchronize()
    eager = (time.perf_counter() - t) * 1000 / iters
    return {"batch": bp, "stages_ms": res, "sum_ms": sum(res.values()), "whole_step_ms": whole, "eager_step_ms": eager,
            "continuation_4_steps_ms": cont, "layers": L}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, nargs="+", default=[64])
    p.add_argument("--stages", action="store_true", help="also time every stage (default: whole step graph vs eager only)")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--matmul-precision", default="high")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision(args.matmul_precision)
    m = NemotronModel.load(None, "cuda", torch.float32)
    st = BatchStepper(m, 3, 128, max(args.batch), cuda_graphs=False)
    out = []
    for bp in args.batch:
        r = breakdown(m, st, bp, args.iters, args.stages)
        out.append(r)
        print(f"== batch {bp}: whole step {r['whole_step_ms']:.2f} ms (eager {r['eager_step_ms']:.2f} ms), sum of stages {r['sum_ms']:.2f} ms", flush=True)
        for k, v in sorted(r["stages_ms"].items(), key=lambda kv: -kv[1]):
            print(f"  {v:7.2f} ms  {100 * v / r['sum_ms']:5.1f}%  {k}", flush=True)
        print(f"  continuation (4 steps): {r['continuation_4_steps_ms']:.2f} ms", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "graph_breakdown.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
