"""Layer-level check of RelPosAttention.forward_stream against forward, plus SDPA backend availability.

Run on a GPU: python tests/check_fused_attn.py
"""
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import repo_paths  # noqa: E402,F401  (repo code on sys.path)
from nemotron_lite import NemotronLite  # noqa: E402

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
m = NemotronLite.from_pretrained(device="cuda", dtype=torch.float32)
enc = m.encoder
B, T, C, H, dk, D = 64, 4, 56, 8, 128, 1024
K, Kp = C + T, 64
pos = enc.pos_emb(K, "cuda", torch.float32)
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn(B, T, D, device="cuda", generator=g)
kc = torch.randn(B, H, C, dk, device="cuda", generator=g)
vc = torch.randn(B, H, C, dk, device="cuda", generator=g)
kv_len = torch.randint(0, C + 1, (B,), device="cuda", generator=g)
kv_len[0], kv_len[1] = 0, C

pos_ids = torch.arange(C, device="cuda")
cache_valid = pos_ids[None, :] >= (C - kv_len)[:, None]
key_valid = torch.cat([cache_valid, cache_valid.new_ones(B, T)], 1)
att_mask = ~key_valid[:, None, :].expand(B, T, -1)
cols = torch.arange(Kp, device="cuda")
valid = (cols[None, :] >= (C - kv_len)[:, None]) & (cols[None, :] < K)
key_masked = ~valid[:, None, None, :]

worst = 0.0
for li in (0, 11, 23):
    att = enc.layers[li].self_attn
    ref, (kr, vr) = att(x, pos, att_mask, (kc, vc))
    pk = att.stream_pos_keys(pos, T, K, Kp)
    for name, backends in (("auto", None), ("efficient", [SDPBackend.EFFICIENT_ATTENTION]), ("math", [SDPBackend.MATH])):
        kb = torch.empty(B, H, K, dk, device="cuda"); vb = torch.empty_like(kb)
        kb[:, :, :C] = kc; vb[:, :, :C] = vc
        try:
            if backends is None:
                out = att.forward_stream(x, kb, vb, pk, key_masked)
            else:
                with sdpa_kernel(backends):
                    out = att.forward_stream(x, kb, vb, pk, key_masked)
        except RuntimeError as e:
            print(f"layer {li} {name}: unavailable ({str(e).splitlines()[0][:120]})")
            continue
        d = (out - ref).abs().max().item() / ref.abs().max().item()
        dk_ = (kb - kr).abs().max().item()
        worst = max(worst, d)
        print(f"layer {li:2d} sdpa={name:9s} out rel-err {d:.2e}  kv buffer max diff {dk_:.1e}")

# can index_select write straight into a strided view of a bigger buffer?
pool = torch.randn(300, 24, H, C, dk, device="cuda")
buf = torch.zeros(64, 24, H, K, dk, device="cuda")
idx = torch.randperm(300, device="cuda")[:64]
ptr = buf.data_ptr()
try:
    torch.index_select(pool, 0, idx, out=buf[:, :, :, :C])
    ok = torch.equal(buf[:, :, :, :C], pool[idx]) and buf.data_ptr() == ptr
    print("index_select into strided view:", "OK" if ok else "WRONG RESULT")
except RuntimeError as e:
    print("index_select into strided view: error", str(e).splitlines()[0][:120])
for name, fn in (("index_select+copy", lambda: buf[:, :, :, :C].copy_(pool.index_select(0, idx))),
                 ("index_select out=view", lambda: torch.index_select(pool, 0, idx, out=buf[:, :, :, :C])),
                 ("index_select contiguous (old)", lambda: torch.index_select(pool, 0, idx, out=torch.empty(64, 24, H, C, dk, device="cuda")))):
    try:
        fn(); torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(20):
            fn()
        e.record(); torch.cuda.synchronize()
        print(f"gather {name:30s} {s.elapsed_time(e) / 20:.3f} ms")
    except RuntimeError as ex:
        print(f"gather {name}: error {str(ex).splitlines()[0][:100]}")
print("WORST rel-err", f"{worst:.2e}")
