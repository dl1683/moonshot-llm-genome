# -*- coding: utf-8 -*-
"""Fold opt1: NOTES entry, T139 card, QUEUE row, paper clause note, STATE."""
import io, json

# --- NOTES ---
n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## opt1 — the optimizer controls: the clock is ADAM'S ARITHMETIC, the kill gate is DISPLACEMENT, and the killer stream TEACHES under SGD (2026-09-30 ~11:25Z) — DONE

WHAT WE DID: the licensed e185 wash cell VERBATIM with only the
optimizer swapped (7 arms, CPU-only); every gate green — A0
bit-reproduces e185's stored control (max|diff| 3.2e-13), input
batches bit-identical across arms and vs e185's stored hashes, and A5
(moment-reset) is bit-identical to A0: no consolidation-time
optimizer state is inherited, now gated proof not assumption.

WHAT WE SAW (T139): VERDICT per the registered letter: ADAM-
AMPLIFIES (the warmup knob shifted t* 10.08x >= 2x; OPT-AGNOSTIC
could not fire — no SGD arm killed within the CPU window, all three
cap-limited; their clocks located only by clearly-labeled
extrapolation ~379/1174/3228 steps, never adjudicated). THE
DECOMPOSITION the co-reads carry: (1) THE CLOCK IS ADAM'S
ARITHMETIC — step-1 pre-clip grad norm 0.9829 in EVERY arm; AdamW
moved 1.6543/step (lr*sqrt(N), sign-normalized) vs SGD-1e-3's
0.0010: 1687x at the same lr; the two-step clock is the normalizer,
not the memory. (2) THE GATE IS DISPLACEMENT — every Adam variant
kills at D ~ 2.49-2.84; warmup stretched the step clock 10x and the
kill still arrived at the same displacement within ~15%
(rate-carried, not clock-carried). (3) beta2 and inherited moments
carry NOTHING (A4 indistinguishable; A5 bit-identical). (4) THE
BOMBSHELL: matched-lr SGD ran the SAME corpus wash with the fact
RISING (g-12 0.916 -> 0.940-0.955 at |d| <= 0.29) — the stream that
kills in two steps under Adam TEACHES under SGD; the raw gradient is
weakly fact-positive at small displacement, the kill is the sign-
normalization leaving the basin at full speed. "The stream chooses
signs, the normalizer chooses the clock." (5) Alignment co-read:
mildly negative everywhere (-0.015..-0.105, SGD included) — never
the store's -0.44; W023's rise-prediction undecided pending the
per-arm curves. HONESTY: n=1 per arm, one root, one stream, CPU fp32
texture (gated on-device via A0). OWED (dispatched, opt1b): the
DIRECT measured SGD kill at lr 1e-2 (~380 steps, ckpt-resumable CPU
chunks) — does the displacement gate generalize across learned
trajectory classes, or is it sign-normalization-typed?
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

# --- T139 ---
t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T138 —"
card = """## T139 — opt1: the clock is Adam's arithmetic; the gate is displacement; the "killer" stream teaches under SGD (2026-09-30 ~11:30Z)

The nine-times-asked control lands as a decomposition of the wash
kill into CLOCK and GATE. THE CLOCK: step-1 pre-clip grad norm 0.9829
in every arm; AdamW moves 1.6543/step (lr*sqrt(N), sign-normalized),
matched-lr SGD 0.0010 — 1687x at the same lr. THE GATE: every Adam
variant kills at D ~ 2.49-2.84; warmup stretched the clock 10.08x
(the registered ADAM-AMPLIFIES fire) and the kill still arrived at
the same displacement within ~15%. THE BOMBSHELL: matched-lr SGD ran
the SAME corpus wash with the fact RISING (g-12 0.916 -> 0.940-0.955)
— the raw stream gradient is weakly fact-POSITIVE at small
displacement; the kill is Adam's sign-normalization exiting the
basin at full speed. A fresh AdamW's first step is +/- lr on every
coordinate; the basin (~2.5 L2) meets per-step 1.65 and dies in ~1.6
steps — the stream chooses signs, the normalizer chooses the clock.
THE TRAJECTORY HYPOTHESIS (T137) REFINES: never "any learned path
kills" — it is "any path that REACHES the gate kills; Adam reaches
it in ~1.6 steps BY CONSTRUCTION; static jumps need 4-10x (g3K);
SGD's slow path is fact-positive in-window." THE OWED DISCRIMINATOR
(opt1b, dispatched): the direct measured SGD kill at lr 1e-2 — if
SGD dies at D ~ 2.5 the displacement gate generalizes across
trajectory classes; if SGD reaches D = 2.5 ALIVE, the gate is
trajectory-class-typed (sign-normalized vs raw-gradient paths
differ) and the law gains a third clause. W023 stands untested (the
summary alignment read is flat-negative; the per-arm curves decide).
beta2, inherited moments: nothing (A4 indistinguishable; A5
bit-identical). HONESTY: n=1 per arm; SGD clocks only by labeled
projection; CPU fp32 texture gated on-device.

"""
assert anchor in t and "## T139" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# --- QUEUE ---
q = io.open("QUEUE.md", encoding="utf-8").read()
old = q[q.index("| opt1 |"):]
old = old[:old.index("\n")]
new = ("| opt1 | OPTIMIZER CONTROLS | DONE 11:25Z (T139: ADAM-AMPLIFIES via the registered knob (warmup 10.08x); "
       "the CLOCK is Adam's sign-normalization (1687x/step at matched lr), the GATE is displacement (~2.49-2.84 "
       "every Adam arm); beta2 + inherited moments NOTHING (A5 bit-identical); THE STREAM TEACHES UNDER SGD "
       "(g-12 0.916->0.955); SGD clocks projected only (~379-3228 steps, never adjudicated)) |")
q = q.replace(old, new, 1)
opt1b = ("| opt1b | THE DIRECT SGD KILL (opt1's owed discriminator: lr 1e-2, ~380 steps, ckpt-resumable CPU chunks) | "
         "DISPATCHED 11:30Z — bars: SGD-KILLS-AT-GATE (dies at D ~ 2.5 ± 15% -> the displacement gate generalizes "
         "across trajectory classes) / SGD-SPARED-AT-GATE (reaches D=2.5 alive -> the gate is sign-normalization-typed; "
         "the law gains a third clause); co-reads: alignment along the SGD path, fact-vs-D curve vs Adam's |\n")
j = q.index("\n", q.index(new)) + 1
q = q[:j] + opt1b + q[j:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

# --- paper: the optimizer clause is now evidenced ---
p = "scratch/day6_paper_skeleton.md"
t2 = io.open(p, encoding="utf-8").read()
old10 = """11. Trajectory-hypothesis replication (C13-1, NEW):"""
new10 = """12. Optimizer clause EVIDENCED (opt1, DONE): the two-step clock is
   Adam's sign-normalization (1687x/step at matched lr); the kill
   gate is displacement (~2.5, clock-invariant); under matched SGD
   the same stream STRENGTHENS the fact — paragraph (2)'s "at every
   lr tested" must become "under AdamW at every lr tested" and cite
   the decomposition; opt1b (the direct SGD kill) decides whether
   the gate generalizes across trajectory classes.
11. Trajectory-hypothesis replication (C13-1, NEW):"""
assert old10 in t2
t2 = t2.replace(old10, new10, 1)
io.open(p, "w", encoding="utf-8").write(t2)

# --- STATE ---
s = json.load(io.open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = "2026-09-30T11:30:00Z"
s["current_experiment"] = ("Fleet 2: g1bW (GPU, alive, PARTIAL metrics) + opt1b DISPATCHED (CPU: the direct SGD "
                           "kill - opt1's owed discriminator). opt1 FOLDED (T139): the clock is Adam's arithmetic, "
                           "the gate is displacement, THE STREAM TEACHES UNDER SGD; paper gap 12 (optimizer clause evidenced).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(s, indent=2, ensure_ascii=False))
print("opt1 folded")
