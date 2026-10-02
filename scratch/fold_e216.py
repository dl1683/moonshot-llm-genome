# -*- coding: utf-8 -*-
"""Fold e216: NOTES, T190, W028 update, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e216 — the within-family residual: RESIDUAL-STRUCTURED — THE RESIDUAL IS THE THIRD SORTING DIMENSION, AND IT REPLICATES (family x height explains only ~2/3 (R2 0.67/0.62, below the bar; p0 adds only +0.06); the per-probe misses replicate across washes at rho 0.934 — essentially the full reliability surviving the model; no surface feature names the dimension) (2026-10-02 ~16:05Z) — DONE

WHAT WE DID: pure desk on the committed 54-probe records (the OLS
hr ~ family + p0 per wash; the residual structure; the predictor
ladder; the cross-wash residual); 9.6s, zero model loads; the desk
recompute certified at dp 0.0.

WHAT WE SAW (T190): THE FULL MODEL R2 0.671/0.621 — BELOW the
0.75 bar; the height term adds only +0.057/+0.085 over family
(p0-only R2 0.04/0.09: family is the load-bearing term; linear-p0
a poor within-family account despite the rank association
0.51-0.55). THE NAMED SPLITS SURVIVE: the product contrast
(Gmail+PlayStation minus iPhone) at +2.92/+2.29 residual SD
(Gmail the archive's single largest residual); the tmpl width
ratio 0.89/0.86 (the common slope leaves rev-capital's width
intact). NO REGISTERED PREDICTOR NAMES THE DIMENSION (vocab
overlap / entrenchment / position all |rho| < 0.09). THE DECISIVE
CLAUSE: THE CROSS-WASH RESIDUAL rho 0.934 (per-family: rev-capital
0.96, lang 0.89, near 1.00) — the per-probe idiosyncrasy IS the
residual, and it replicates: A GENUINE THIRD SORTING DIMENSION.
W028'S READING REFINED ONE FLOOR DOWN: the height layer was the
wrong second term — p0-as-linear-height is poor; the conserved
object is the PER-PROBE RANK, family x probe-idiosyncrasy, not
family x linear-p0 — the rank-conserved/scale-destroyed cut
reaching into the probe-level sorting itself. THE NEXT CUT NAMED:
within-family nonlinear height (p0 as rank or logit) vs genuinely
new probe features. HONESTY: n=2 washes; the hand-registered
typology caveat; plain OLS on bounded hr with unbalanced n's; the
residual's mechanism open (probe physics vs shared-denominator
texture).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T189 —"
card = """## T190 — e216: the third dimension — per-probe rank the model cannot carry (2026-10-02 ~16:05Z)

The residual cell closes T189's thread hard: family x height
explains two-thirds; what remains is NOT noise — the per-probe
misses replicate across independent washes at rho 0.934 (the full
reliability surviving the model). A THIRD SORTING DIMENSION,
unnamed by every surface feature, real as physics. THE LAW'S
DEEPEST CUT YET: the conserved object at the probe level is the
PER-PROBE RANK — which specific probes hold is wash-independent
to three digits of reliability — and linear-height terms cannot
carry it (rank 0.51 but R2 +0.06); the family label organizes the
tiers, the probe idiosyncrasy sorts within, and the idiosyncrasy
replicates. Gmail holds and iPhone dies on BOTH washes — not
because of exposure, entrenchment, or order — for reasons the lab
has not yet named. THE CUT NAMED: nonlinear within-family height
(p0 as rank/logit) vs genuinely new probe features — the third
dimension's identity is the relational signature's last question.

"""
assert anchor in t and "## T190" not in t
t = t.replace(anchor, card + anchor, 1)

# W028 update
i = t.index("## W028 ")
line = t[i:t.index("\n", i)]
t = t.replace(line, line + " [E214+E216 UPDATE ~16:05Z: the law at probe level — the baseline rank is HEIGHT (dies with the scale); the conserved objects are the EROSION ORDER (relational, replicating) and now the PER-PROBE IDIOSYNCRASY (the third dimension, rho 0.934 — which probe holds within a family is physics the surface features do not name)]", 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e216 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e216 | THE WITHIN-FAMILY RESIDUAL | DONE 16:05Z (T190: RESIDUAL-STRUCTURED — the third sorting dimension, replicating at rho 0.934; family x height only ~2/3 (p0 adds +0.06); the named splits survive; no surface feature names it; THE PER-PROBE RANK is the conserved object one floor down) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T16:06:00Z"
st["current_experiment"] = ("e216 FOLDED (T190: the third dimension - per-probe rank, replicating, unnamed). Fleet: "
                            "e217 (GPU, the third wash draw) alone."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e216 folded")
