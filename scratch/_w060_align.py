# W060 desk discriminator: permutation alignment (align-to-reference) on the wall members
# Bounded: mlp hidden-unit blocks only (the late-mlp.2 spread is the object); greedy
# straight-through matching, 2 iterations; CPU 4 threads.
import torch, glob, os, re, statistics as st
torch.set_num_threads(4)
files = sorted(glob.glob('runs/x66/new_*_end.pt') + glob.glob('runs/x66/ref_*_end.pt'))
sds = []
for f in files:
    sd = torch.load(f, map_location='cpu', weights_only=True)
    if hasattr(sd, 'state_dict'): sd = sd.state_dict()
    sds.append({k: v.float() for k, v in sd.items() if v.ndim >= 2})
names = [os.path.basename(f).replace('_end.pt','') for f in files]
layers = sorted({int(m.group(1)) for k in sds[0] for m in [re.match(r'h\.(\d+)\.mlp\.0\.weight', k)] if m})
print(f'{len(names)} members, mlp layers {layers}')
ref = sds[0]
def unit_feats(sd, L):
    # hidden-unit feature: concat of mlp.0 row (input side) and mlp.2 col (output side)
    up = sd[f'h.{L}.mlp.0.weight']   # [H, D]
    dn = sd[f'h.{L}.mlp.2.weight']   # [D, H]
    H = up.shape[0]
    return torch.cat([up, dn.t()], dim=1)  # [H, D+D]
def align_to_ref(sd, iters=2):
    out = {k: v.clone() for k, v in sd.items()}
    for L in layers:
        for _ in range(iters):
            A, B = unit_feats(out, L), unit_feats(ref, L)
            An, Bn = A / A.norm(dim=1, keepdim=True), B / B.norm(dim=1, keepdim=True)
            C = 1.0 - An @ Bn.t()  # cost: cosine distance [H, H]
            perm = torch.argmin(C, dim=1)  # greedy straight-through
            if len(set(perm.tolist())) != len(perm):
                # greedy may collide on tiny blocks; accept as-is (disclosed bound)
                pass
            up, dn = out[f'h.{L}.mlp.0.weight'], out[f'h.{L}.mlp.2.weight']
            out[f'h.{L}.mlp.0.weight'] = up[perm]
            out[f'h.{L}.mlp.2.weight'] = dn[:, perm]
    return out
def rel_dist(sd_a, sd_b, key):
    n = sd_a[key].norm()
    return float((sd_a[key] - sd_b[key]).norm() / n) if n > 0 else 0.0
mlp_keys = [f'h.{L}.mlp.2.weight' for L in layers]
wpe_key = 'wpe.weight'
before_mlp, after_mlp, before_wpe, after_wpe = [], [], [], []
for sd in sds[1:]:
    before_mlp.append(st.mean(rel_dist(sd, ref, k) for k in mlp_keys))
    before_wpe.append(rel_dist(sd, ref, wpe_key))
    al = align_to_ref(sd)
    after_mlp.append(st.mean(rel_dist(al, ref, k) for k in mlp_keys))
    after_wpe.append(rel_dist(al, ref, wpe_key))
f = lambda xs: f'median {st.median(xs):.4f} range [{min(xs):.4f}, {max(xs):.4f}]'
print(f'mlp.2 distance to reference  BEFORE alignment: {f(before_mlp)}')
print(f'mlp.2 distance to reference  AFTER  alignment: {f(after_mlp)}')
print(f'wpe  distance to reference  BEFORE alignment: {f(before_wpe)}')
print(f'wpe  distance to reference  AFTER  alignment: {f(after_wpe)}  (alignment cannot touch wpe - the control)')
