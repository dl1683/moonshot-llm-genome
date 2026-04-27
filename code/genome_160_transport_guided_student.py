"""
genome_160_transport_guided_student.py

POST-CHAIN MANIFESTO CASH-OUT: transport-guided student vs local-heavy student.

Pre-reg LOCKED: research/prereg/genome_160_transport_guided_student_2026-04-26.md
Theory: research/derivations/prefix_information_transport.md
Program: research/programs/post_g156_pass_program.md section g160

If the transport theory is a real design law (validated by g156, g157,
g158, g159), then under matched inference FLOPs and matched distillation
budget, a transport-heavy student should beat a local-heavy student on:
  - C3_macro = mean(HellaSwag, PIQA, Winogrande) accuracy (full validation)
  - CtQ_90: compute to reach 90% of own-final C3_macro

Two students at matched inference FLOPs (within +/- 2%):
  transport_heavy:  6L_noMLP_wide   hidden=512, 6 layers, no MLP, ~50-70M
  local_heavy:      4L_MLP          hidden=384, 4 layers, ffn=1024, ~50-70M

Teacher: Qwen3-0.6B (matches g154 smoke). Distillation: top-k=64 KD with
gamma=0.5, T=2.0. 8192 c4 train windows. Seeds {42,7,13}.

Pre-stated criteria:
  PASS: C3_macro_transport - C3_macro_local >= +1.0pp
        AND CtQ_90_transport <= 0.80 * CtQ_90_local in >=2/3 seeds
  PARTIAL: C3 gain >= +0.5pp OR only the CtQ_90 criterion lands
  KILL: local_heavy ties or wins on both metrics

Compute: ~3-3.5 hr per program estimate.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent

TEACHER_HF = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
BATCH_SIZE = 8
# PILOT scope: 1 seed first, expand to 3 seeds in g160b only if PASS.
# Original 3-seed run > 4hr envelope (Codex pre-flight Severity 8).
SEEDS = [42]
N_TRAIN = 8192
KD_TEMP = 2.0
KD_GAMMA = 0.5
KD_TOPK = 64
LR_WARMUP_STEPS = 200
TRAIN_STEPS = 8000
LR = 3e-4

# CtQ_90 measurement: evaluate at these step checkpoints and find the step at
# which C3_macro reaches >= 0.90 * own_final. Per Codex: also report cumulative
# train FLOPs at each checkpoint so the metric can be expressed in FLOPs (the
# fair currency for comparing two architectures).
CTQ_EVAL_STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]


class ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def build_student(name: str, vocab_size: int, seed: int = 42):
    """Build matched-inference-FLOPs students."""
    from transformers import LlamaConfig, LlamaForCausalLM
    if name == "transport_heavy":
        cfg = LlamaConfig(
            vocab_size=vocab_size, hidden_size=512, num_hidden_layers=6,
            num_attention_heads=8, num_key_value_heads=8, intermediate_size=1024,
            max_position_embeddings=SEQ_LEN + 64, rms_norm_eps=1e-6,
            tie_word_embeddings=True, attn_implementation="eager",
        )
    elif name == "local_heavy":
        # FLOP-matched to transport_heavy (h=512, L=6, noMLP) at intermediate_size=1024.
        # Verified: both arms = 4.027 GFLOP/seq inference at SEQ_LEN=256.
        cfg = LlamaConfig(
            vocab_size=vocab_size, hidden_size=384, num_hidden_layers=4,
            num_attention_heads=6, num_key_value_heads=6, intermediate_size=1024,
            max_position_embeddings=SEQ_LEN + 64, rms_norm_eps=1e-6,
            tie_word_embeddings=True, attn_implementation="eager",
        )
    else:
        raise ValueError(f"unknown student: {name}")
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg).to("cuda").to(torch.bfloat16)
    if name == "transport_heavy":
        for layer in model.model.layers:
            layer.mlp = ZeroMLP()
    return model


def count_inference_flops(model, seq_len=SEQ_LEN):
    """Approximate per-token forward FLOPs for inference."""
    cfg = model.config
    h = cfg.hidden_size
    L = cfg.num_hidden_layers
    m = cfg.intermediate_size
    # Attention per layer per token: 4 * h^2 + 2 * seq_len * h (qkv proj, attn matmul, output proj)
    attn = 4 * h * h + 2 * seq_len * h
    # MLP per layer per token (SwiGLU = 3 * h * m): 3 * h * m
    has_mlp = not isinstance(model.model.layers[0].mlp, ZeroMLP)
    mlp = 3 * h * m if has_mlp else 0
    per_layer = 2 * (attn + mlp)  # x2 for forward (mul-add)
    return L * per_layer * seq_len


def precompute_teacher_logits(teacher, tok, train_ids, train_mask, top_k=KD_TOPK):
    teacher.eval()
    n = train_ids.shape[0]
    topk_idx = torch.zeros((n, train_ids.shape[1] - 1, top_k), dtype=torch.int64)
    topk_lg = torch.zeros((n, train_ids.shape[1] - 1, top_k), dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            ids_b = train_ids[i:i+BATCH_SIZE].to("cuda")
            msk_b = train_mask[i:i+BATCH_SIZE].to("cuda")
            out = teacher(input_ids=ids_b, attention_mask=msk_b, use_cache=False)
            logits = out.logits[:, :-1].float()
            tk = logits.topk(top_k, dim=-1)
            topk_idx[i:i+BATCH_SIZE] = tk.indices.cpu()
            topk_lg[i:i+BATCH_SIZE] = tk.values.cpu()
            if i % (BATCH_SIZE * 50) == 0:
                print(f"    teacher cache: {i}/{n}")
    return topk_idx, topk_lg


def kd_loss(student_logits, teacher_idx, teacher_lg, mask, T):
    teacher_idx = teacher_idx.to(student_logits.device)
    teacher_lg = teacher_lg.to(student_logits.device).float()
    s_at = student_logits.gather(2, teacher_idx)
    s_lp = F.log_softmax(s_at / T, dim=-1)
    t_p = F.softmax(teacher_lg / T, dim=-1)
    kl = (t_p * (t_p.clamp_min(1e-10).log() - s_lp)).sum(dim=-1)
    m = mask[:, 1:].float()
    return (kl * m).sum() / m.sum().clamp(min=1) * (T ** 2)


def warmup_lr(step, target_lr, warmup_steps):
    if step < warmup_steps:
        return target_lr * (step + 1) / warmup_steps
    return target_lr


def measure_capability(model, tok, c3_data):
    """Compute multiple-choice log-likelihood accuracy on each task.

    Handles empty-context items (Winogrande): when item['context'] is "",
    ctx tokenization may return [] and the off-by-one breaks. We use BOS-only
    as a 1-token prefix in that case so logits indexing is well-defined.
    """
    model.eval()
    results = {}
    for task_name, items in c3_data.items():
        correct = 0
        for item in items:
            ll_per_choice = []
            for choice in item["choices"]:
                ctx = item["context"]
                if not ctx.strip():
                    ctx = tok.bos_token if tok.bos_token else (tok.eos_token or " ")
                full = ctx + choice
                ids = tok(full, return_tensors="pt", truncation=True, max_length=SEQ_LEN+128)
                ids = ids["input_ids"].to("cuda")
                ctx_ids = tok(ctx, return_tensors="pt", truncation=True, max_length=SEQ_LEN+128)["input_ids"]
                ctx_len = max(1, ctx_ids.shape[1])
                if ids.shape[1] <= ctx_len:
                    # Defensive: full string didn't add tokens beyond ctx (rare)
                    ll_per_choice.append(float("-inf"))
                    continue
                with torch.no_grad():
                    out = model(input_ids=ids, use_cache=False)
                    logits = out.logits[0, ctx_len-1:-1].float()
                    targets = ids[0, ctx_len:]
                    if logits.shape[0] != targets.shape[0]:
                        ll_per_choice.append(float("-inf"))
                        continue
                    ll = -F.cross_entropy(logits, targets, reduction="sum").item()
                ll_per_choice.append(ll)
            pred = int(np.argmax(ll_per_choice))
            if pred == item["label"]:
                correct += 1
        results[task_name] = correct / max(len(items), 1)
    model.train()
    macro = float(np.mean(list(results.values()))) if results else 0.0
    return {"per_task": results, "C3_macro": macro}


def load_c3_validation(n_per_task=None):
    """Load HellaSwag, PIQA, Winogrande validation sets.
    n_per_task: if None, full validation. For smoke runs, can subsample."""
    from datasets import load_dataset
    print("Loading C3 validation sets...")
    out = {}

    # HellaSwag
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation")
        items = []
        for ex in ds:
            ctx = ex["ctx"] if ex.get("ctx") else (ex.get("activity_label", "") + " " + ex.get("ctx_a", "") + " " + ex.get("ctx_b", ""))
            items.append({"context": ctx, "choices": ex["endings"], "label": int(ex["label"])})
            if n_per_task and len(items) >= n_per_task:
                break
        out["hellaswag"] = items
        print(f"  hellaswag: {len(items)}")
    except Exception as e:
        print(f"  hellaswag load failed: {e}")
        out["hellaswag"] = []

    # PIQA
    try:
        ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
        items = []
        for ex in ds:
            items.append({"context": ex["goal"] + " ", "choices": [ex["sol1"], ex["sol2"]], "label": int(ex["label"])})
            if n_per_task and len(items) >= n_per_task:
                break
        out["piqa"] = items
        print(f"  piqa: {len(items)}")
    except Exception as e:
        print(f"  piqa load failed: {e}")
        out["piqa"] = []

    # Winogrande
    try:
        ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="validation", trust_remote_code=True)
        items = []
        for ex in ds:
            ctx = ex["sentence"].replace("_", "{}")
            items.append({"context": "", "choices": [ctx.format(ex["option1"]), ctx.format(ex["option2"])],
                          "label": int(ex["answer"]) - 1})
            if n_per_task and len(items) >= n_per_task:
                break
        out["winogrande"] = items
        print(f"  winogrande: {len(items)}")
    except Exception as e:
        print(f"  winogrande load failed: {e}")
        out["winogrande"] = []

    # Per heartbeat code review Sev-8: raise on any missing C3 task
    required = ("hellaswag", "piqa", "winogrande")
    missing = [t for t in required if len(out.get(t, [])) == 0]
    if missing:
        raise RuntimeError(f"C3 validation incomplete; missing/empty tasks: {missing}")

    return out


def train_student_with_kd(student, train_ids, train_mask, topk_idx, topk_lg,
                           tok, c3_data, ctq_eval_steps, seed, n_steps=TRAIN_STEPS):
    opt = torch.optim.AdamW(student.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    student.train()
    n_train = train_ids.size(0)
    history = []  # list of {step, c3_macro, per_task}

    for step in range(1, n_steps + 1):
        cur_lr = warmup_lr(step, LR, LR_WARMUP_STEPS)
        for g in opt.param_groups:
            g['lr'] = cur_lr
        idx = rng.integers(0, n_train, size=BATCH_SIZE)
        ids = train_ids[idx].to("cuda")
        mask = train_mask[idx].to("cuda")
        opt.zero_grad()
        out = student(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits
        sl = logits[:, :-1].contiguous()
        lbl = ids[:, 1:].clone()
        sm = mask[:, 1:]
        lbl_ce = lbl.clone()
        lbl_ce[sm == 0] = -100
        ce = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lbl_ce.reshape(-1), ignore_index=-100)
        kd = kd_loss(sl, topk_idx[idx], topk_lg[idx], mask, KD_TEMP)
        loss = (1 - KD_GAMMA) * ce + KD_GAMMA * kd
        if not torch.isfinite(loss):
            print("NaN seen; stopping arm")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        if step in ctq_eval_steps or step == n_steps:
            metrics = measure_capability(student, tok, c3_data)
            metrics["step"] = step
            metrics["wallclock_s"] = time.time() - t0
            history.append(metrics)
            print(f"    step={step:5d} ce={ce.item():.3f} kd={kd.item():.3f} C3_macro={metrics['C3_macro']:.4f}")

    return history


def main():
    t0 = time.time()
    print("genome_160: transport-guided student vs local-heavy student")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading teacher tokenizer {TEACHER_HF}...")
    tok = AutoTokenizer.from_pretrained(TEACHER_HF)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab = len(tok)
    print(f"  vocab={vocab}")

    # Use c4_clean_v1 seed=160 (different from g141..g158 seed=42 / g159 seed=159)
    # to minimize overlap with prior experiments.
    print(f"\nLoading {N_TRAIN} c4 train windows (seed=160)...")
    pool = []
    for rec in c4_clean_v1(seed=160, n_samples=N_TRAIN):
        pool.append(rec["text"])
        if len(pool) >= N_TRAIN:
            break
    enc = tok(pool, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    train_ids = enc["input_ids"]
    train_mask = enc["attention_mask"]

    # Per Codex pre-flight: 13-token rolling-hash dedup audit vs c4_clean_v1 seed=42
    print("Running 13-token dedup audit vs c4_clean_v1 seed=42 train slice...")
    g42 = []
    for rec in c4_clean_v1(seed=42, n_samples=2048):
        g42.append(rec["text"])
        if len(g42) >= 2048:
            break
    enc42 = tok(g42, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    def _hashes(ids, mask):
        H = set()
        for r in range(ids.shape[0]):
            v = mask[r].sum().item()
            if v < 13:
                continue
            row = ids[r, :v].tolist()
            for i in range(len(row) - 12):
                H.add(tuple(row[i:i+13]))
        return H
    h_train = _hashes(train_ids, train_mask)
    h_42 = _hashes(enc42["input_ids"], enc42["attention_mask"])
    overlap = h_train & h_42
    pct = 100.0 * len(overlap) / max(len(h_train), 1)
    print(f"  13-gram overlap: {pct:.2f}%")
    if pct > 5.0:
        raise RuntimeError(f"dedup FAIL: {pct:.2f}% > 5%")

    print(f"\nLoading teacher {TEACHER_HF}...")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_HF, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda")
    teacher.eval()
    n_t = sum(p.numel() for p in teacher.parameters())
    print(f"  teacher params: {n_t/1e6:.1f}M")

    # Per cycle 9 code review Sev-7: cache key must include all relevant config
    # so changing TEACHER_HF/SEQ_LEN/KD_TOPK/seed doesn't silently load stale logits.
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(exist_ok=True)
    teacher_slug = TEACHER_HF.replace("/", "-")
    # Per cycle 12 code review Sev-8: train_seed must match the actual c4_clean_v1
    # seed used for loading (line 319 uses seed=160). Was incorrectly recorded as 42.
    C4_TRAIN_SEED = 160
    cache_path = cache_dir / f"g160_teacher_topk_{teacher_slug}_n{N_TRAIN}_seq{SEQ_LEN}_topk{KD_TOPK}_seed{C4_TRAIN_SEED}.pt"
    expected_meta = {
        "teacher_hf": TEACHER_HF, "n_train": N_TRAIN, "seq_len": SEQ_LEN,
        "kd_topk": KD_TOPK, "train_seed": C4_TRAIN_SEED,
    }
    if cache_path.exists():
        print(f"\nLoading cached teacher top-{KD_TOPK} logits from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached_meta = cached.get("meta", {})
        if cached_meta != expected_meta:
            print(f"  cache meta mismatch ({cached_meta} vs {expected_meta}); discarding and recomputing")
            cache_path.unlink()
            cached = None
    else:
        cached = None
    if cached is not None:
        topk_idx, topk_lg = cached["idx"], cached["lg"]
    else:
        print(f"\nPrecomputing teacher top-{KD_TOPK} logits over {N_TRAIN} sequences...")
        topk_idx, topk_lg = precompute_teacher_logits(teacher, tok, train_ids, train_mask, KD_TOPK)
        torch.save({"idx": topk_idx, "lg": topk_lg, "meta": expected_meta}, cache_path)
        print(f"  cached -> {cache_path}")
    del teacher
    torch.cuda.empty_cache()

    c3_data = load_c3_validation()

    students = ["transport_heavy", "local_heavy"]
    results = {s: {} for s in students}
    for student_name in students:
        ref = build_student(student_name, vocab, seed=42)
        flops = count_inference_flops(ref)
        n_params = sum(p.numel() for p in ref.parameters())
        del ref
        torch.cuda.empty_cache()
        print(f"\n  {student_name}: ~{n_params/1e6:.1f}M params, ~{flops/1e9:.2f} G inference FLOPs/seq")

    # FLOP-match check (warn if outside +/- 2%)
    th_ref = build_student("transport_heavy", vocab, 42)
    lh_ref = build_student("local_heavy", vocab, 42)
    th_flops = count_inference_flops(th_ref)
    lh_flops = count_inference_flops(lh_ref)
    flop_diff_pct = abs(th_flops - lh_flops) / max(th_flops, lh_flops) * 100
    print(f"\n  FLOP match: transport={th_flops/1e9:.2f}G  local={lh_flops/1e9:.2f}G  diff={flop_diff_pct:.1f}%")
    if flop_diff_pct > 5:
        print("  WARNING: FLOP diff > 5%; results may not be matched-FLOP comparison")
    del th_ref, lh_ref
    torch.cuda.empty_cache()

    for student_name in students:
        for seed in SEEDS:
            print(f"\n=== {student_name} seed={seed} ===")
            student = build_student(student_name, vocab, seed=seed)
            history = train_student_with_kd(
                student, train_ids, train_mask, topk_idx, topk_lg,
                tok, c3_data, set(CTQ_EVAL_STEPS), seed,
            )
            results[student_name][seed] = {"history": history}
            del student
            torch.cuda.empty_cache()

    # Analysis — per Codex pre-flight Severity-7: CtQ_90 measured in train FLOPs,
    # not steps (so the comparison across architectures is FLOP-fair, not
    # step-fair which advantages whichever arm has cheaper steps).
    th_per_step_train_flops = th_flops * 3  # forward + backward ~ 3x forward FLOPs
    lh_per_step_train_flops = lh_flops * 3

    def per_step_train_flops(student_name):
        return th_per_step_train_flops if student_name == "transport_heavy" else lh_per_step_train_flops

    # Per cycle 6 code review Sev-8: completeness guard before verdict.
    # Each (student, seed) must have history reaching TRAIN_STEPS — no NaN-truncated runs.
    incomplete = []
    for student_name in students:
        for s in SEEDS:
            hist = results[student_name][s].get("history", [])
            if not hist or hist[-1].get("step") != TRAIN_STEPS:
                last_step = hist[-1].get("step") if hist else None
                incomplete.append((student_name, s, last_step))
    if incomplete:
        raise RuntimeError(f"g160 incomplete (training did not reach TRAIN_STEPS): {incomplete}; cannot emit verdict")

    print(f"\n=== ANALYSIS ===")
    summary = {}
    for student_name in students:
        finals = [results[student_name][s]["history"][-1]["C3_macro"]
                  for s in SEEDS if results[student_name][s]["history"]]
        ctq_steps = []
        ctq_flops = []
        for s in SEEDS:
            hist = results[student_name][s]["history"]
            if not hist:
                continue
            final_c3 = hist[-1]["C3_macro"]
            target = 0.9 * final_c3
            ctq_step = next((h["step"] for h in hist if h["C3_macro"] >= target), hist[-1]["step"])
            ctq_steps.append(ctq_step)
            ctq_flops.append(ctq_step * per_step_train_flops(student_name))
        summary[student_name] = {
            "C3_final_mean": float(np.mean(finals)) if finals else float("nan"),
            "C3_final_std": float(np.std(finals)) if finals else float("nan"),
            "CtQ_90_steps_mean": float(np.mean(ctq_steps)) if ctq_steps else float("nan"),
            "CtQ_90_flops_mean": float(np.mean(ctq_flops)) if ctq_flops else float("nan"),
        }
        print(f"  {student_name}: C3={summary[student_name]['C3_final_mean']*100:.2f}%  "
              f"CtQ_90_flops={summary[student_name]['CtQ_90_flops_mean']/1e12:.2f} TFLOP "
              f"({summary[student_name]['CtQ_90_steps_mean']:.0f} steps)")

    th = summary.get("transport_heavy", {})
    lh = summary.get("local_heavy", {})
    c3_gap_pp = (th.get("C3_final_mean", 0) - lh.get("C3_final_mean", 0)) * 100
    # CtQ ratio in FLOPs (architecture-fair) per Codex pre-flight
    ctq_ratio = th.get("CtQ_90_flops_mean", float("inf")) / max(lh.get("CtQ_90_flops_mean", 1), 1)

    # Per cycle 12 code review Sev-7: SEEDS=[42] is single-seed PILOT scope.
    # Verdict labels reflect PILOT directionality, NOT canonical PASS.
    # A canonical 3-seed verdict requires a separate g160c prereg.
    pilot_scope = (len(SEEDS) == 1)
    pilot_prefix = "PILOT_" if pilot_scope else ""

    if c3_gap_pp >= 1.0 and ctq_ratio <= 0.80:
        verdict = (f"{pilot_prefix}DIRECTIONAL_SUPPORT: C3 gap={c3_gap_pp:+.2f}pp (>=1.0pp), CtQ_90 ratio={ctq_ratio:.2f} (<=0.80). "
                   f"{'PILOT directional support — write 3-seed canonical prereg.' if pilot_scope else 'Transport principle confirmed as model-selection rule.'}")
    elif c3_gap_pp >= 0.5 or ctq_ratio <= 0.85:
        verdict = (f"{pilot_prefix}PARTIAL: C3 gap={c3_gap_pp:+.2f}pp, CtQ_90 ratio={ctq_ratio:.2f}.")
    else:
        verdict = (f"{pilot_prefix}KILL: C3 gap={c3_gap_pp:+.2f}pp, CtQ_90 ratio={ctq_ratio:.2f}. "
                   f"Theory does not select better matched-cost design at this scale.")
    print(f"\n  verdict: {verdict}")

    out = {
        "genome": 160, "name": "transport_guided_student",
        "config": {"teacher": TEACHER_HF, "students": students, "seeds": SEEDS,
                    "n_train": N_TRAIN, "train_steps": TRAIN_STEPS, "lr": LR,
                    "kd_temp": KD_TEMP, "kd_gamma": KD_GAMMA, "kd_topk": KD_TOPK,
                    "ctq_eval_steps": CTQ_EVAL_STEPS,
                    "th_inference_flops": int(th_flops), "lh_inference_flops": int(lh_flops),
                    "flop_diff_pct": flop_diff_pct},
        "results": {s: {str(k): v for k, v in d.items()} for s, d in results.items()},
        "summary": summary,
        "c3_gap_pp": c3_gap_pp, "ctq_ratio": ctq_ratio,
        "verdict": verdict, "elapsed_s": time.time() - t0,
    }
    out_path = ROOT / "results" / "genome_160_transport_guided_student.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
