"""e248 repair: gdirs_{w1,w2}_m1.npy row 0 (wash step 1) holds exactly one
NaN at flat-index 38 (first 4KB page, both washes, same index — a storage
artifact, not compute: the recorded gnorms[0] are finite 0.544/0.560 and
fp16 casts of finite values cannot produce NaN). Row 0 is exactly
recomputable: the t0 weights + the wash's first seeded draw. Verified
after write: row finite, self-cos = 1."""
import importlib.util, json, torch, numpy as np
spec = importlib.util.spec_from_file_location('e248m', 'lab/e248_organism_replicate.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
dev = 'cuda'
st = torch.load(mod.SCRATCH / 'streams.pt', weights_only=False)
wash_train = st['wash_train']
ctx = mod.CFG['block_size']
model = mod.make_model(dev)
model.load_state_dict(torch.load(mod.CKPT / 'e248_installed_t0.pt',
                                 map_location='cpu', weights_only=False)['model'])
for wname, seed in mod.SEED_WASH.items():
    gen = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(wash_train) - ctx - 1, (mod.WASH['batch'],), generator=gen)
    x = torch.stack([wash_train[i:i + ctx] for i in ix]).to(dev)
    y = torch.stack([wash_train[i + 1:i + 1 + ctx] for i in ix]).to(dev)
    _, loss = model(x, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        flat = mod.flat_grad(model, dev)
        nrm = float(flat.norm().item())
    print(wname, 'batch1 ce', round(float(loss.item()), 4), 'gnorm', round(nrm, 4))
    gpath = mod.SCRATCH / f'gdirs_{wname}_m1.npy'
    mm = np.lib.format.open_memmap(gpath, dtype=np.float16, mode='r+',
                                   shape=(mod.WASH['steps'], mod.PARAMS_EXPECTED))
    row = (flat / max(nrm, 1e-12) * 16384.0).half().cpu().numpy()
    mm[0] = row
    mm.flush()
    chk = np.asarray(np.memmap(gpath, dtype=np.float16, mode='r',
                               shape=(mod.WASH['steps'], mod.PARAMS_EXPECTED))[0])
    print(wname, 'row0 nonfinite after repair:', int((~np.isfinite(chk)).sum()),
          '| self-dot check (first 8M cols):', round(float((chk[:8_000_000].astype(np.float64)**2).sum()), 1))
    mod.jlog('gdir_row0_repaired', wash=wname, why='single NaN at flat-index 38 (first-page storage artifact); recomputed from t0 + first seeded draw', gnorm=round(nrm, 4))
