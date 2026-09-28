import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

t = open("THINKING.md", encoding="utf-8").read()

# T097 correction — the home-graft counterexample
o1 = "## T097 — E158: the door closes when a novel graft forms — site-store construction and geometry-access destruction are one event (2026-09-28 ~12:40Z)"
n1 = """## T097 — [CORRECTED per R48 critic — the headline was contradicted by e158's own unread census: locked@band DID re-form a home graft (row 129: brake -0.132 -> content +0.057, site_pos TRUE, peak 129) while the door stayed OPEN; TWO grafts, different door outcomes — the operative variable is SITE NOVELTY (or occupied-slot history), NOT graft formation; 'one event two faces' is FALSE as written] E158: the two-factor gate — closure requires novelty AND zero-variance (2026-09-28 ~12:40Z)

THE HONEST RESTATEMENT: home-site and novel-site grafts dissociate
from door closure. The registered verdict (SITE-INDEPENDENT: the
two-factor gate on novelty+zero-variance) STANDS — it was the
pre-committed cell and the metrics' own adjudication. What falls
is the MECHANISM STORY layered on top: 'graft formation closes
the door' must read 'NOVEL-SITE teaching closes the door, with
or without a graft.' The novelty axis (distance-from-home vs
occupied-slot vs first-novel-site) is unresolved until e165.
THE FRAME-BREAKING ALTERNATIVE (R48's final line, adopted): the
door may decay by DISUSE while the graft grows by USE — two
independently-trained things, not one machinery seen twice. The
build-travel frame is PROVISIONAL on exactly this; the deciding
cells are already running/queued: e166 (inverse event), e161's
freeze-cell (plain corpus, no teaching — does the door close
anyway?), e154 (different fact). e158's at-boundary honesty:
jitter@183 0.5048 (median 0.408, most prompts below bar) and
locked@band 0.546 — margins 0.005/0.046 against a +-0.029
device bound; 'OPEN' labels are at-or-near-boundary; locked@home
cost 40% of the door ('harmless' was generous — corrected)."""
assert o1 in t, "T097"
t = t.replace(o1, n1, 1)

# T095 correction — profile mismatch
o2 = "The fork resolves as MIXED, and the resolution is better than\neither branch: the poison kills through BOTH channels, EACH\nINDIVIDUALLY SUFFICIENT."
n2 = """R48 CORRECTION (~13:10Z — the sufficiency claim fails internal
consistency): cell (i) KEPT the poison's front-loaded q/k
absorption at CE -0.0004 — absorbed mass at the REAL profile is
FREE; cell (ii) killed only with a FLAT profile carrying 11x the
poison's L0 absorption. The honest form: HEALTHY CONTENT FULLY
RESCUES (supply edge, matched conditions); TOTAL-DOSE absorption
on a FLATTENED profile kills (allocation edge, mismatched
conditions); the per-layer DISTRIBUTION is untested and is where
the divergence lives (e167 queued: per-layer-matched bias).
The original fold text follows with that qualifier attached."""
assert o2 in t, "T095"
t = t.replace(o2, n2, 1)

# T096 bound
o3 = "## T096 — E125a: the asymmetry of existence"
n3 = "## T096 — [R48 BOUND: 'no kill set EXISTS' is a 0.03% sample of the pair space (the consolidated kill was a superadditive pair INVISIBLE to singles ranking — the same hiding place unsearched for the site fact); B5/B6 at moderate CE never run; the MLP surface (33% of load) untouched — scope: head-coordinate surgery, sets <= 4; e168 queued: exhaustive 630-pair scan + MLP-neuron ablation] E125a: the asymmetry of existence"
assert o3 in t, "T096"
t = t.replace(o3, n3, 1)

# W017 header restatement
o4 = "## W017 — WONDER: variance concentrates credit — concentration is portability and vulnerability; redundancy is robustness and immobility (2026-09-28 ~12:28Z; the unifying thread of e125a/e160/e147/e151)"
n4 = "## W017 — WONDER: [R48 MINIMAL RESTATEMENT: variance-trained readouts recruit fewer heads with heavier top-load; killability tracks circuit COMPLEMENTARITY (the top-loaded head L0H3 is DISPENSABLE — the kill set is ranks 2-3), which concentration neither predicts nor explains; the coding-removability link is OPEN pending e154/e169; keep out of paper text beyond the operative top-load form] variance concentrates the top of the load distribution (2026-09-28 ~12:28Z)"
assert o4 in t, "W017"
t = t.replace(o4, n4, 1)

# T092 amendment — the layer-2 fork
o5 = "The paper's model figure: four stacked layers"
n5 = "R48 AMENDMENT: T096 forks layer 2 by phase (killable complementary circuit vs unkillable redundant population — the readout layer was one layer too flat as drawn). The paper's model figure: four stacked layers (layer 2 drawn forked)"
assert o5 in t, "T092"
t = t.replace(o5, n5, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- paper fixes ----------
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
fixes = [
 ("the types are PHASES of one substrate, converted bidirectionally by\ntraining",
  "the types are phases of one substrate, with conversions demonstrated in\nboth directions across the lineage (same-net reversibility is e155's\nqueued cell)"),
 ("e158 RESOLVED: SITE-INDEPENDENT — closure requires novelty AND zero-variance together: the geometry door closes exactly when a NOVEL GRAFT forms; neither variance nor placement alone suffices",
  "e158 RESOLVED: SITE-INDEPENDENT — closure requires novelty AND zero-variance together (jitter@novel open, locked@home open, locked@novel shut); the door's closure ACCOMPANIES novel-site teaching — graft-formation per se is not the closer (a home graft formed with the door open); mechanism pending e165/e166/e161"),
 ("type-selective head surgery", "circuit-selective head surgery"),
 ("and the knife is\n  TYPE-SELECTIVE (the site-stored fact survives the same\n  coordinates)", "and the knife is\n  CIRCUIT-selective (install dies too; the site-stored fact survives)"),
]
for old, new in fixes:
    if old in p:
        p = p.replace(old, new, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
r = r.replace("the surgery is type-selective", "the surgery is circuit-selective", 1)
r = r.replace("closing the geometry door globally [R46 critic: n=1 same-fact\ncell; the different-fact and jitter-at-183 cells are queued —\n\"globally\" is provisional]",
              "closing the geometry door [R48: novelty+zero-variance gate; a home graft formed with the door OPEN — graft-formation is not the closer; the disuse alternative is live and e161/e154/e166 decide]", 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

# ---------- queue: e167/e168 ----------
q = open("QUEUE.md", encoding="utf-8").read()
o_q = "| e163 | THE SATURATION CONTROL"
rows = """| e167 | PER-LAYER-MATCHED MASS (R48 critic attack 3's missing cell — does mass kill at the poison's OWN profile?) | READY (CPU eval-only, minutes) | six biases set to the poison's per-layer masses (front-loaded 0.48/0.45/0.33) on healthy values. Bars: PROFILE-MATCHED-KILLS = fact dies (allocation edge real at matched conditions); PROFILE-MATCHED-SPARES = MASS-COUPLED collapses, the kill is value-mediated end-to-end (T095's allocation edge dies) |
| e168 | THE EXHAUSTIVE SITE-KNIFE SCAN (R48 critic attack 2 — discharges the 0.03%-sample bound on T096) | READY (CPU eval-only, ~1h; splittable) | (a) exhaustive 630-pair zero-mode scan on arm_b (the superadditive-pair hiding place, machinery exists); (b) top-k MLP-neuron ablation at flat CE ranked by site-fact counterfactual delta; (c) B5/B6 mean-mode at CE <= +0.5. Bars: HIDDEN-PAIR = any pair >= 60% at CE <= +0.35 (T096's null inverts); MLP-KNIFE = MLP set kills at flat CE (the incorrigible substrate found); BOUND-CONFIRMED = none (the asymmetry of existence survives its exhaustive test) |
""" + o_q
assert o_q in q
q = q.replace(o_q, rows, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 3 (CPU): e164 + e154 + e166. R48 critic folded: T097 corrected (home-graft counterexample — novelty not graft-formation), T095 profile-qualified, T096 bounded; build-travel frame PROVISIONAL on the disuse alternative."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("R48 critic repairs applied")
