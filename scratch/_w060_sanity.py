# W060 sanity: can the aligner recover a KNOWN random permutation of the reference?
import torch, re, statistics as st
torch.set_num_threads(4)
sd = torch.load('runs/x66/new_d_100_end.pt', map_location='cpu', weights_only=True)
if hasattr(sd,'state_dict'): sd = sd.state_dict()
sd = {k: v.float() for k, v in sd.items() if v.ndim >= 2}
layers = sorted({int(m.group(1)) for k in sd for m in [re.match(r'h\.(\d+)\.mlp\.0\.weight', k)] if m})
g = torch.Generator().manual_seed(7)
shuf = {k: v.clone() for k, v in sd.items()}
for L in layers:
    H = sd[f'h.{L}.mlp.0.weight'].shape[0]
    perm = torch.randperm(H, generator=g)
    shuf[f'h.{L}.mlp.0.weight'] = sd[f'h.{L}.mlp.0.weight'][perm]
    shuf[f'h.{L}.mlp.2.weight'] = sd[f'h.{L}.mlp.2.weight'][:, perm]
def unit_feats(s, L):
    up, dn = s[f'h.{L}.mlp.0.weight'], s[f'h.{L}.mlp.2.weight']
    return torch.cat([up, dn.t()], dim=1)
mlp_keys = [f'h.{L}.mlp.2.weight' for L in layers]
def rd(a, b):
    n = a.norm(); return float((a-b).norm()/n)
print('shuffled-vs-ref mlp.2 mean rel-dist (pre-align):', st.mean(rd(shuf[k], sd[k]) for k in mlp_keys))
out = {k: v.clone() for k, v in shuf.items()}
for L in layers:
    for _ in range(2):
        A, B = unit_feats(out, L), unit_feats(sd, L)
        An, Bn = A/A.norm(dim=1,keepdim=True), B/B.norm(dim=1,keepdim=True)
        perm = torch.argmin(1.0 - An @ Bn.t(), dim=1)
        out[f'h.{L}.mlp.0.weight'] = out[f'h.{L}.mlp.0.weight'][perm]
        out[f'h.{L}.mlp.2.weight'] = out[f'h.{L}.mlp.2.weight'][:, perm]
print('shuffled-vs-ref mlp.2 mean rel-dist (post-align):', st.mean(rd(out[k], sd[k]) for k in mlp_keys))
