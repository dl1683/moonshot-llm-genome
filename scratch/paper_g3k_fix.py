# -*- coding: utf-8 -*-
"""Paper skeleton: fold g3K's corrections (T137) into the g-series
integration amendment + abstract paragraph (2) trajectory clause."""
import io

p = "scratch/day6_paper_skeleton.md"
t = io.open(p, encoding="utf-8").read()

# --- R6(c) correction ---
old_c = """(c) THE CONE (rescoped R56): the store's damage anisotropy is
   EXTREME — the wash direction and its 45-degree tilts kill at ~2x
   displacement (g0 0.10-0.16) where matched-L2 ISOTROPIC noise needs
   ~25-32x (concentration of measure; the registered 4x leg sat below
   the isotropic threshold and is never cited) — n=3 wash-draw seeds,
   one organ (redraw queued); the lambda edge is draw-sensitive in
   (1x,2x); effect size stated ONLY as the threshold ratio ~12-16x."""
new_c = """(c) THE CONE + THE TRAJECTORY LAW (g3R-amended + g3K): killing is
   DIRECTION-typed (the wash direction and its 45-degree tilts kill at
   ~2x displacement; g3R n=3 wash-draw seeds, one organ) AND
   TRAJECTORY-typed (g3K: static random displacement is 4-10x more
   forgivable at the organism — kappa_store 5.0, kappa_host 6.0 at
   matched mean per-coordinate RMS; the earlier 24-32x was the
   STORE-ISOLATED leg, so the wide cone is an ORGAN property; no
   memory state survives continued TRAINING on any learned path —
   corpus or noise-label — while static jumps show graded basins).
   Effect sizes stated ONLY as organism-level threshold ratios; the
   store-isolated organ reading cited as the design's property, never
   as the organism's."""
assert old_c in t, "R6c"
t = t.replace(old_c, new_c, 1)

# --- framing sentence ---
old_f = """Framing sentence: forgetting is not distance; it is direction —
   a wide-angle, low-measure sensitive set — and it can be walled
   (g1, battery-channel; sequential pending g1bW), detected (g2, with
   the R56 construction disclosures), or measured as anisotropy (g3)."""
new_f = """Framing sentence: forgetting is not distance and not even
   displacement — it is ALIGNED TRAINING: any learned path to a
   displacement kills where a random jump of the same size is 4-10x
   more forgivable; the aligned front can be walled (g1,
   battery-channel; sequential pending g1bW), detected (g2, with the
   R56 construction disclosures), and its anisotropy measured (g3)."""
assert old_f in t, "framing"
t = t.replace(old_f, new_f, 1)

# --- abstract clause 4 correction ---
old_a = """Abstract clause (4), draft (R56-corrected): "(4) The same laws are
   generative: a projection ball (channel-scoped protection at a
   standing tax; sequential memory tested), a self-timed rehearsal
   organ (timing replicates across seeds and roots; amplitude
   root-draw-bound), and an extreme damage anisotropy (wash-aligned
   ~2x vs isotropic ~25-32x displacement) each convert a dissected
   failure law into an architectural positive, replicated at n>=3 on
   single roots/organs — memory in these nets is not fragile by
   necessity but by default.\""""
new_a = """Abstract clause (4), draft (R56+g3K-corrected): "(4) The same laws
   are generative: a projection ball (channel-scoped protection at a
   standing tax; sequential memory tested in g1bW), a self-timed
   rehearsal organ (timing replicates across seeds and roots;
   amplitude root-draw-bound), and a direction-and-trajectory-typed
   forgetting law (wash-aligned training kills at ~2x where static
   random displacement is 4-10x more forgivable; no state survives
   any learned path) each convert a dissected failure law into an
   architectural positive, replicated at n>=3 on single roots/organs
   — memory in these nets is not fragile by necessity but by
   default.\""""
assert old_a in t, "abstract4"
t = t.replace(old_a, new_a, 1)

# --- Fig 4(iii) ---
old_fig = """(iii) cone: the dissociation bars (wash vs
   isotropic at 1x/2x/4x matched L2), n=3 with the draw-artifact note."""
new_fig = """(iii) cone/trajectory: the
   kappa pair plot (organ-level store-isolated vs organism-level,
   wash vs isotropic rung ladders from g3K, with g3R's n=3 wash legs)
   — the trajectory-vs-static dissociation panel."""
assert old_fig in t, "fig4iii"
t = t.replace(old_fig, new_fig, 1)

# --- abstract paragraph (2): the trajectory clause, gap 10 executed ---
old_p2 = "the mechanism per e185+e180: a NARROW robustness basin"
new_p2 = ("the mechanism per e185+e180+g3K: no robustness basin against "
          "LEARNED displacement (the law is a TRAJECTORY law — static random "
          "displacement is 4-10x more forgivable, kappa ~5-6 at matched "
          "per-coordinate RMS; every e185 arm was a trajectory), ")
assert old_p2 in t, "p2"
t = t.replace(old_p2, new_p2, 1)

# --- gap 7 note ---
old_g7 = """7. Cone is organ-draw n=1 (g3R's split robustness is over wash-draw
   seeds on ONE reused organ) — one organ redraw (g3O queued) or an
   explicit scope sentence (prefer both; cheap); the R56 ruler
   correction is ADOPTED (threshold-ratio form, isotropic-spare never
   cited)."""
new_g7 = """7. Cone/trajectory is organ-draw n=1 (g3R+g3K's split robustness is
   over wash-draw seeds on ONE reused organ) — one organ redraw (g3O
   queued, now carrying g3K's organism-level ruler); the R56+g3K
   corrections are ADOPTED (organism threshold ratios; the 24-32x
   store-isolated reading cited only as the organ's design property)."""
assert old_g7 in t, "gap7"
t = t.replace(old_g7, new_g7, 1)

io.open(p, "w", encoding="utf-8").write(t)
print("paper g3K corrections folded")
