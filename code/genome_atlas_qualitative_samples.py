"""Qualitative demo: text samples from the atlas-patched model.

NLL is a scalar. Partners want to SEE text. Generate completions from:
 (A) Teacher (pretrained Qwen3-0.6B)
 (B) All-28-layer-lesioned student (garbage)
 (C) Student + last-7 atlas (28 KB)
 (D) Student + all-28 atlas (112 KB)

Show side-by-side for 5 prompts. If the atlas-patched models produce
even partially-coherent text, that's the qualitative demo for outreach.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from genome_capability_patch_k48 import (  # noqa: E402
    extract_mid_activations, lesion_midblock,
)
from genome_capability_patch_k48_v2 import MeanShiftHook  # noqa: E402
from genome_loaders import load_system  # noqa: E402
from stimulus_banks import c4_clean_v1  # noqa: E402

_ROOT = _THIS_DIR.parent

PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "Two plus three equals",
    "The Eiffel Tower is located in",
    "Once upon a time,",
]


def generate(model, tokenizer, prompt, max_new=20, temperature=0.0):
    enc = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new,
                              do_sample=False if temperature == 0 else True,
                              temperature=max(temperature, 1e-5),
                              pad_token_id=tokenizer.eos_token_id
                              if tokenizer.pad_token_id is None else tokenizer.pad_token_id)
    text = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return text


def run_condition(hf_id, sents, label, atlas=None, student_atlas=None, patch_layers=None,
                  lesion=False, n_layers=28):
    sys_m = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    if lesion:
        sd = sys_m.model.state_dict()
        for L in range(n_layers):
            lesion_midblock(sd, f"model.layers.{L}.")
        sys_m.model.load_state_dict(sd, strict=False)
    handles = []
    if atlas is not None and patch_layers is not None:
        for L in patch_layers:
            shift = (atlas[L] - student_atlas[L]).astype(np.float32)
            hook = MeanShiftHook(shift)
            h = sys_m.model.model.layers[L].register_forward_hook(hook)
            handles.append(h)
    completions = {}
    for p in PROMPTS:
        try:
            c = generate(sys_m.model, sys_m.tokenizer, p, max_new=20)
        except Exception as e:
            c = f"<gen error: {e}>"
        completions[p] = c
        # ASCII-safe print: replace non-cp1252 chars for Windows console
        safe_p = p.encode("ascii", "backslashreplace").decode("ascii")
        safe_c = c.encode("ascii", "backslashreplace").decode("ascii")
        print(f"  [{label}] {safe_p!r} -> {safe_c!r}")
    for h in handles:
        h.remove()
    sys_m.unload(); torch.cuda.empty_cache()
    return completions


def main():
    hf_id = "Qwen/Qwen3-0.6B"
    sents = []
    for rec in c4_clean_v1(seed=42, n_samples=5000):
        sents.append(rec["text"])
        if len(sents) >= 300:
            break

    # Teacher atlas
    print("\n-- teacher --")
    sys_t = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    n_layers = sys_t.n_hidden_layers()
    teacher_out = {}
    for p in PROMPTS:
        teacher_out[p] = generate(sys_t.model, sys_t.tokenizer, p)
        print(f"  [TEACHER] {p!r} -> {teacher_out[p]!r}")
    atlas = {L: extract_mid_activations(sys_t, sents, L, f"t{L}").mean(axis=0).astype(np.float32)
             for L in range(n_layers)}
    sys_t.unload(); torch.cuda.empty_cache()

    # Fully-lesioned student atlas
    print("\n-- building student_atlas from full-lesion --")
    sys_s = load_system(hf_id, quant="fp16", untrained=False, device="cuda")
    sd = sys_s.model.state_dict()
    for L in range(n_layers):
        lesion_midblock(sd, f"model.layers.{L}.")
    sys_s.model.load_state_dict(sd, strict=False)
    student_atlas = {L: extract_mid_activations(sys_s, sents, L, f"s{L}").mean(axis=0).astype(np.float32)
                      for L in range(n_layers)}
    sys_s.unload(); torch.cuda.empty_cache()

    # Condition: all-lesioned (no patch)
    print("\n-- all-lesioned (no patch) --")
    lesion_out = run_condition(hf_id, sents, "LESIONED", lesion=True, n_layers=n_layers)

    # Condition: all-lesioned + last-7 atlas
    print("\n-- all-lesioned + last-7 atlas --")
    last7_out = run_condition(hf_id, sents, "LAST-7",
                               atlas=atlas, student_atlas=student_atlas,
                               patch_layers=list(range(n_layers - 7, n_layers)),
                               lesion=True, n_layers=n_layers)

    # Condition: all-lesioned + all-28 atlas
    print("\n-- all-lesioned + all-28 atlas --")
    all28_out = run_condition(hf_id, sents, "ALL-28",
                               atlas=atlas, student_atlas=student_atlas,
                               patch_layers=list(range(n_layers)),
                               lesion=True, n_layers=n_layers)

    # Save side-by-side
    out = {"prompts": PROMPTS,
           "teacher": teacher_out,
           "lesioned": lesion_out,
           "last_7_atlas_28kb": last7_out,
           "all_28_atlas_112kb": all28_out}
    out_path = _ROOT / "results/gate2/atlas_qualitative_samples.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}")

    print("\n=== SIDE-BY-SIDE ===")
    for p in PROMPTS:
        print(f"\n  PROMPT: {p!r}")
        print(f"  TEACHER  -> {teacher_out[p]!r}")
        print(f"  LESIONED -> {lesion_out[p]!r}")
        print(f"  LAST-7   -> {last7_out[p]!r}")
        print(f"  ALL-28   -> {all28_out[p]!r}")


if __name__ == "__main__":
    main()
