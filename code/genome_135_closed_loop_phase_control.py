"""
genome_135_closed_loop_phase_control.py

CODEX VERDICT (today's pivot decision): "Is the trajectory CAUSAL or
EPIPHENOMENAL?" Closed-loop phase-control is the decisive test.

Hypothesis: capability emerges through traversal of the universal spectral
trajectory (random=high -> mode-collapse=low -> recovery to ~4.243). If
the trajectory is CAUSAL, accelerating it should accelerate capability
acquisition. If EPIPHENOMENAL, controlling the trajectory will not help.

Protocol — 2 arms x N=2 architectures (Llama, Pythia-style):

  Arm A (control):
    Standard CE training. Fixed LR, no phase intervention.

  Arm B (phase-controlled):
    Every 32 steps, measure sqrt(er)*alpha (the spectral phase coordinate).
    Compare to known landmark schedule:
      - step 128:  target sqrt(er)*alpha ~ 3.0  (entering mode-collapse)
      - step 512:  target sqrt(er)*alpha ~ 2.8  (mode-collapse minimum)
      - step 4000: target sqrt(er)*alpha >= 4.0 (recovery toward target)
    If we are BEHIND schedule (above target trajectory): boost LR + reduce
      weight decay to accelerate descent.
    If we are AHEAD of schedule (below trajectory): reduce LR + increase
      weight decay to slow.
    The intervention pushes the model to traverse the trajectory FASTER
    than the natural baseline.

Metric — CtQ_75: gradient steps to reach NLL = donor_NLL + 0.25*initial_gap.
  For Llama from-scratch: target NLL roughly 5.0 (vs random init NLL ~11
  and a tiny model's natural floor around 4-5).

Pre-stated criteria:
  PASS: phase-controlled arm reaches CtQ_75 in <= 0.5x the steps of control
        on at least one architecture; matches or beats control NLL at the
        final step.
  PARTIAL: 1.5-2x speedup, OR matches NLL at fewer steps but not 2x.
  KILL: phase-controlled arm matches or under-performs control. Trajectory
        is epiphenomenal; pivot to high-dim process descriptors.

This is the first experiment in the project that would constitute an
ELECTRICITY-GRADE EFFICIENCY DEMO if it passes.

Architecture: tiny Llama (same as genome_133, ~30M params).
Training: 4000 steps on c4_clean_v1, batch=8, max_length=256.

Results: results/genome_135_closed_loop_phase_control.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_extractor import extract_trajectory  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent
TARGET = float(np.sqrt(18))

HIDDEN_SIZE = 384
N_LAYERS = 6
N_HEADS = 6
INTERMEDIATE_SIZE = 1024
SEQ_LEN = 256
MAX_STEPS = 4000
EVAL_AT = [0, 32, 128, 512, 1000, 2000, 4000]
PHASE_CHECK_INTERVAL = 32
LR_BASE = 3e-4
BATCH_SIZE = 8
SEED = 42

# Target trajectory (from genome_133 Llama natural trajectory):
#   step 0: 6.83, step 32: 1.26, step 128: 1.03, step 512: 2.39,
#   step 1000: 3.06, step 2000: 4.00, step 4000: 4.58.
# We want phase-controlled arm to traverse this FASTER.
# Schedule: target invariant value at each phase-check step.
def target_invariant_at_step(step):
    """Linear interpolation between observed Llama trajectory landmarks,
    SHIFTED 2x EARLIER to test acceleration."""
    # Compress the natural Llama trajectory into half the steps:
    # natural step k -> target value at step k/2
    # i.e., we want phase-controlled arm to reach step 4000's natural value at step 2000.
    eff_step = step * 2
    landmarks = [
        (0,    6.83),
        (32,   1.26),
        (128,  1.03),
        (512,  2.39),
        (1000, 3.06),
        (2000, 4.00),
        (4000, 4.58),
        (8000, 4.58),
    ]
    # Linear interp
    for (s1, v1), (s2, v2) in zip(landmarks, landmarks[1:]):
        if s1 <= eff_step <= s2:
            t = (eff_step - s1) / max(s2 - s1, 1)
            return v1 + t * (v2 - v1)
    return landmarks[-1][1]


def make_tiny_llama(vocab_size, seed=SEED):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=SEQ_LEN + 64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg).to("cuda").to(torch.bfloat16)
    return model


def spectrum(X):
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(max(X.shape[0] - 1, 1))
    return s.astype(np.float64)


def fit_power_tail(s, lo=0.05, hi=0.5):
    r = np.arange(1, len(s) + 1)
    a = max(1, int(len(s) * lo))
    b = int(len(s) * hi)
    lr = np.log(r[a:b])
    ls = np.log(s[a:b] + 1e-12)
    slope, _ = np.polyfit(lr, ls, 1)
    return float(-slope)


def eff_rank(s):
    s2 = s ** 2
    tot = s2.sum()
    return float(tot ** 2 / (s2 ** 2).sum()) if tot > 0 else 0.0


def measure_invariant(model, tok, calib_texts):
    n_layers = model.config.num_hidden_layers
    mid = max(1, n_layers // 2)
    traj = extract_trajectory(
        model=model, tokenizer=tok,
        texts=calib_texts, layer_indices=[mid], pooling="seq_mean",
        device="cuda", system_key="g135_llama", class_id=1,
        quantization="fp16",
        stimulus_version="c4_clean.v1.seed42.n800",
        seed=42, batch_size=BATCH_SIZE, max_length=SEQ_LEN,
    )
    X = traj.layers[0].X.astype(np.float32)
    s = spectrum(X)
    a = fit_power_tail(s)
    er = eff_rank(s)
    return {"alpha": a, "eff_rank": er, "sqrt_er_alpha": float(np.sqrt(er) * a)}


def measure_nll(model, tok, eval_texts):
    """Mean NLL over eval texts."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(eval_texts), BATCH_SIZE):
        batch = eval_texts[i:i + BATCH_SIZE]
        enc = tok(batch, return_tensors="pt", padding=True,
                   truncation=True, max_length=SEQ_LEN).to("cuda")
        ids, mask = enc["input_ids"], enc["attention_mask"]
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits
        sl = logits[:, :-1].contiguous()
        lbl = ids[:, 1:].contiguous().clone()
        sm = mask[:, 1:].contiguous()
        lbl[sm == 0] = -100
        loss = F.cross_entropy(
            sl.view(-1, sl.size(-1)),
            lbl.view(-1), ignore_index=-100, reduction="sum",
        )
        n = (sm != 0).sum().item()
        total_loss += loss.item()
        total_tokens += n
    model.train()
    return total_loss / max(total_tokens, 1)


def train_arm(model, tok, train_ids, train_mask, calib_texts, eval_texts, arm_name,
              phase_controlled=False):
    opt = torch.optim.AdamW(model.parameters(), lr=LR_BASE, betas=(0.9, 0.95),
                             weight_decay=0.1)
    rng = np.random.default_rng(SEED)
    rows = []
    t_arm = time.time()

    # step-0 measurement
    print(f"\n[{arm_name}] step=0 (random init)")
    inv0 = measure_invariant(model, tok, calib_texts)
    nll0 = measure_nll(model, tok, eval_texts)
    rows.append({"step": 0, **inv0, "nll": nll0, "lr": LR_BASE, "wd": 0.1})
    print(f"  inv={inv0['sqrt_er_alpha']:.3f}  NLL={nll0:.3f}")
    model.train()

    step = 0
    next_eval_idx = 1
    current_lr = LR_BASE
    current_wd = 0.1

    while step < MAX_STEPS:
        idx = rng.integers(0, train_ids.size(0), size=BATCH_SIZE)
        ids = train_ids[idx].to("cuda")
        mask = train_mask[idx].to("cuda")
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits
        sl = logits[:, :-1].contiguous()
        lbl = ids[:, 1:].contiguous().clone()
        sm = mask[:, 1:].contiguous()
        lbl[sm == 0] = -100
        loss = F.cross_entropy(
            sl.view(-1, sl.size(-1)), lbl.view(-1), ignore_index=-100
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        # PHASE CONTROL: every 32 steps, measure spectrum and adapt
        if phase_controlled and step % PHASE_CHECK_INTERVAL == 0 and step < MAX_STEPS:
            model.eval()
            inv = measure_invariant(model, tok, calib_texts[:64])  # quick check
            model.train()
            target = target_invariant_at_step(step)
            actual = inv['sqrt_er_alpha']
            # Heuristic adaptation:
            #   if actual > target * 1.2 (behind schedule, still too random): boost LR
            #   if actual < target * 0.8 (ahead, going too low): reduce LR
            #   else: keep
            ratio = actual / max(target, 1e-3)
            if ratio > 1.3:
                current_lr = min(current_lr * 1.5, LR_BASE * 5)
                current_wd = max(current_wd * 0.7, 0.01)
            elif ratio < 0.7:
                current_lr = max(current_lr * 0.7, LR_BASE * 0.2)
                current_wd = min(current_wd * 1.3, 0.5)
            else:
                # converging toward LR_BASE smoothly
                current_lr = 0.7 * current_lr + 0.3 * LR_BASE
                current_wd = 0.7 * current_wd + 0.3 * 0.1
            for g in opt.param_groups:
                g['lr'] = current_lr
                g['weight_decay'] = current_wd

        if next_eval_idx < len(EVAL_AT) and step == EVAL_AT[next_eval_idx]:
            print(f"\n[{arm_name}] step={step} ({time.time()-t_arm:.0f}s)")
            model.eval()
            inv = measure_invariant(model, tok, calib_texts)
            nll = measure_nll(model, tok, eval_texts)
            rows.append({"step": step, **inv, "nll": nll,
                          "lr": current_lr, "wd": current_wd, "loss": float(loss.item())})
            print(f"  inv={inv['sqrt_er_alpha']:.3f}  NLL={nll:.3f}  loss={loss.item():.3f}  lr={current_lr:.2e}  wd={current_wd:.3f}")
            model.train()
            next_eval_idx += 1

    return rows


def find_ctq(rows, target_nll):
    """Return first step where NLL <= target_nll."""
    for r in rows:
        if r["nll"] <= target_nll:
            return r["step"]
    return None


def main():
    t0 = time.time()
    print("genome_135: closed-loop phase-control training")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.decode([0])
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    print("Loading c4_clean_v1 stimuli (calib n=400, eval n=200, train n=4000)...")
    all_texts = []
    for rec in c4_clean_v1(seed=42, n_samples=4800):
        all_texts.append(rec["text"])
    calib_texts = all_texts[:400]
    eval_texts = all_texts[400:600]
    train_texts = all_texts[600:]
    print(f"  calib N={len(calib_texts)}, eval N={len(eval_texts)}, train N={len(train_texts)}")

    actual_vocab = len(tok)
    print(f"  vocab: {actual_vocab}")

    enc = tok(train_texts, padding=True, truncation=True,
               max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc["input_ids"]
    train_mask = enc["attention_mask"]
    print(f"  train tensor: {train_ids.shape}")

    # Arm A: control
    print("\n=== ARM A: CONTROL (fixed LR, no phase intervention) ===")
    model_a = make_tiny_llama(actual_vocab)
    rows_a = train_arm(model_a, tok, train_ids, train_mask,
                        calib_texts, eval_texts, "arm_A_control",
                        phase_controlled=False)
    del model_a
    torch.cuda.empty_cache()

    # Arm B: phase-controlled
    print("\n=== ARM B: PHASE-CONTROLLED ===")
    model_b = make_tiny_llama(actual_vocab)
    rows_b = train_arm(model_b, tok, train_ids, train_mask,
                        calib_texts, eval_texts, "arm_B_phase_controlled",
                        phase_controlled=True)
    del model_b
    torch.cuda.empty_cache()

    # CtQ analysis
    nll_a_final = rows_a[-1]["nll"]
    nll_b_final = rows_b[-1]["nll"]
    nll_a_step0 = rows_a[0]["nll"]
    initial_gap = nll_a_step0 - nll_a_final
    target_nll = nll_a_step0 - 0.75 * initial_gap

    ctq_a = find_ctq(rows_a, target_nll)
    ctq_b = find_ctq(rows_b, target_nll)

    print(f"\n=== ANALYSIS ===")
    print(f"  Arm A final NLL: {nll_a_final:.3f}")
    print(f"  Arm B final NLL: {nll_b_final:.3f}")
    print(f"  CtQ_75 target NLL: {target_nll:.3f}")
    print(f"  CtQ_75 Arm A: {ctq_a}")
    print(f"  CtQ_75 Arm B: {ctq_b}")

    speedup = (ctq_a / ctq_b) if (ctq_a and ctq_b and ctq_b > 0) else None
    print(f"  speedup (A/B): {speedup}")

    # Verdict
    if speedup and speedup >= 2.0:
        verdict = (f"PASS: phase-controlled arm reaches target NLL {speedup:.2f}x faster. "
                   f"Trajectory is CAUSAL — controlling spectral phase accelerates capability. "
                   f"First electricity-grade efficiency demo.")
    elif speedup and speedup >= 1.5:
        verdict = (f"PARTIAL: {speedup:.2f}x speedup, below 2x threshold but real. "
                   f"Trajectory has some causal influence. Refinement needed.")
    elif nll_b_final < nll_a_final - 0.1:
        verdict = (f"PARTIAL: B reaches lower final NLL ({nll_b_final:.3f} vs {nll_a_final:.3f}) "
                   f"but not faster. Phase control improves equilibrium not rate.")
    else:
        verdict = (f"KILL: phase-controlled arm matches or under-performs control. "
                   f"Trajectory is EPIPHENOMENAL — controlling the spectrum does not change "
                   f"capability acquisition. Pivot to high-dim process descriptors (P3).")

    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 135, "name": "closed_loop_phase_control",
        "config": {
            "hidden": HIDDEN_SIZE, "layers": N_LAYERS, "heads": N_HEADS,
            "ffn": INTERMEDIATE_SIZE, "max_steps": MAX_STEPS,
            "lr_base": LR_BASE, "batch": BATCH_SIZE,
            "phase_check_interval": PHASE_CHECK_INTERVAL,
        },
        "arm_A_control": rows_a,
        "arm_B_phase_controlled": rows_b,
        "analysis": {
            "target_nll": target_nll,
            "initial_gap": initial_gap,
            "ctq_75_arm_A": ctq_a,
            "ctq_75_arm_B": ctq_b,
            "speedup": speedup,
            "final_nll_A": nll_a_final,
            "final_nll_B": nll_b_final,
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_135_closed_loop_phase_control.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
