# -*- coding: utf-8 -*-
"""Fold the literature beat: paper risks sharpened, T138 card, e188 spec
addition (install-vs-wash cosine), STATE stamp."""
import io, json

# --- T138 card ---
t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T137 — g3K:"
card = """## T138 — the literature pass: the trajectory law is new as a CONTROL, predicted as THEORY (2026-09-30 ~10:55Z)

scratch/lit_beat_20260930.md, ~20 searches, all supervisor anchors
pinned. THE GATE: six phrasings of trajectory-vs-static /
learned-vs-random displacement — the dissociation appears NOWHERE.
THE POSITIONING THAT MATTERS: Evron (COLT 2022) and Goldfarb & Hand
(AISTATS 2023) already state IN THEORY that forgetting is governed by
task/gradient geometry rather than displacement magnitude — linear/
overparameterized theory, no static controls, no fact-readout assay.
So the lab's claim is not "alignment was unsuspected"; it is "the
theory predicted it and nobody ran the control": the 3-way controlled
dissociation at matched per-coordinate RMS (training kills — corpus
or noise-label; random jump survives, graded kappa 5-6) plus the
census form. CITE THE THEORY LINE AND LEAD WITH THE CONTROL —
uncited, T137 reads under-theorized; cited, it reads as the
experiment the theory was waiting for. The other sharpenings: the
wall's family is hard-constraint methods (Wolczyk ICML'22, Elsayed
RLC'24 — defended by minimality: one commit + one scalar ball, zero
old-task statistics, plus the survival assay); the rhythm's is
learned/interference-based replay scheduling (Klasson, MIR, PER —
defended by the zero-parameter self-timed gate and the g2g
head-to-head); the cone's is task arithmetic (Ilharco — the defence
is to ADOPT the vocabulary: report the install-vs-wash cosine; if
the wash approximates minus-the-install-vector, forgetting at this
scale IS task arithmetic, which would be a simplification, not a
refutation). FLAGGED for full-text re-check before submission: SFAO
(OpenReview Feb 2026) and Elsayed RLC 2024. The e188 co-read gains
the install-vs-wash cosine (cheap, decisive for the vocabulary
choice).

"""
assert anchor in t and "## T138" not in t
t = t.replace(anchor, card + anchor, 1)

# --- e188 spec addition (W022b) ---
old = "passes. Name when dispatched: e188."
new = ("passes. Name when dispatched: e188. [LIT-BEAT ADDITION 10:55Z: "
       "add the INSTALL-vs-WASH cosine as a co-read — cos(install_direction, "
       "wash_direction) per organism; if wash ~ -install, forgetting here IS "
       "task arithmetic (Ilharco ICLR'23) and the paper adopts that vocabulary "
       "(T138); if not, the wash is the corpus's adaptation direction, not the "
       "fact's negation — either answer decides the framing.]")
assert old in t
t = t.replace(old, new, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# --- paper risks sharpened ---
p = "scratch/day6_paper_skeleton.md"
t2 = io.open(p, encoding="utf-8").read()
old_r3 = """R3 "Known phenomenon" (attention sinks; memory types) — the novelty is the
   CAUSAL compass + presence-typed routing + the failure-mode inversion, not
   the sink's existence. Position against sink/StreamLLM and
   complementary-learning-systems literature (scratch/massaction_key_lit.md,
   W003's CLS analogy, now narrowed to replay-only)."""
new_r3 = """R3 "Known phenomenon" (attention sinks; memory types) — the novelty is the
   CAUSAL compass + presence-typed routing + the failure-mode inversion, not
   the sink's existence. Position against sink/StreamLLM (Xiao ICLR'24) and
   the CLS/replay line (now literature-mapped: scratch/lit_beat_20260930.md).
   R3b (T137, NEW — the most dangerous overlap): the trajectory law's
   nearest prior is THEORY — Evron COLT'22 + Goldfarb & Hand AISTATS'23
   state that forgetting follows task/gradient geometry, not displacement
   magnitude. MUST cite and lead with the controlled 3-way dissociation
   (theory predicted; nobody ran the static control). R3c: the wall reads
   as hard-constraint methods (Wolczyk ICML'22, Elsayed RLC'24) without
   minimality + the survival assay foregrounded. R3d: "wash = negative task
   vector" (Ilharco ICLR'23) — pre-empted by reporting the install-vs-wash
   cosine (e188 co-read) and adopting the vocabulary if it fits.
   FULL-TEXT RE-CHECKS before submission: SFAO (OpenReview Feb 2026),
   Elsayed RLC 2024."""
assert old_r3 in t2
t2 = t2.replace(old_r3, new_r3, 1)
io.open(p, "w", encoding="utf-8").write(t2)

# --- STATE ---
s = json.load(io.open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = "2026-09-30T10:55:00Z"
s["current_experiment"] = ("Fleet 2: g1bW (GPU, finalizing) + opt1 (CPU, computing). "
                           "Literature beat FOLDED (T138: the trajectory law is new as a CONTROL, "
                           "predicted as THEORY - Evron/Goldfarb must be cited; novelty gate PASSED; "
                           "e188 gains the install-vs-wash cosine; paper risks R3b-d + two full-text "
                           "re-checks flagged).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(s, indent=2, ensure_ascii=False))
print("lit beat folded")
