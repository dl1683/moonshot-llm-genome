# -*- coding: utf-8 -*-
"""Lit-beat fold part 2 (THINKING already written): paper R3 sharpening + STATE."""
import io, json

p = "scratch/day6_paper_skeleton.md"
t = io.open(p, encoding="utf-8").read()
old = """complementary-learning-systems literature (scratch/massaction_key_lit.md,
  W003's CLS analogy, now narrowed to replay-only)."""
new = """complementary-learning-systems literature (scratch/massaction_key_lit.md,
  W003's CLS analogy, now narrowed to replay-only; full claim-by-claim
  mapping: scratch/lit_beat_20260930.md).
R3b (T137, the most dangerous overlap): the trajectory law's nearest
  prior is THEORY — Evron COLT'22 + Goldfarb & Hand AISTATS'23 state
  that forgetting follows task/gradient geometry, not displacement
  magnitude. MUST cite and lead with the controlled 3-way dissociation
  (the theory predicted; nobody ran the static control).
R3c: the wall reads as hard-constraint methods (Wolczyk ICML'22,
  Elsayed RLC'24) without minimality (one commit + one scalar ball,
  zero old-task statistics) + the survival assay foregrounded.
R3d: "wash = negative task vector" (Ilharco ICLR'23) — pre-empted by
  reporting the install-vs-wash cosine (e188 co-read) and adopting
  the vocabulary if it fits.
FULL-TEXT RE-CHECKS before submission: SFAO (OpenReview Feb 2026),
  Elsayed RLC 2024."""
assert old in t, "r3 tail"
t = t.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(t)

s = json.load(io.open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = "2026-09-30T10:55:00Z"
s["current_experiment"] = ("Fleet 2: g1bW (GPU, finalizing) + opt1 (CPU, computing). "
                           "Literature beat FOLDED (T138: the trajectory law is new as a CONTROL, "
                           "predicted as THEORY - Evron/Goldfarb must be cited; novelty gate PASSED; "
                           "e188 gains the install-vs-wash cosine; paper risks R3b-d + two full-text "
                           "re-checks flagged).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(s, indent=2, ensure_ascii=False))
print("part2 OK")
