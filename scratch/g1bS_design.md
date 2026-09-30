# g1bS — THE WALL AT 10x (design draft v1, 2026-09-30 ~11:08Z)

Status: DESIGN-DRAFT per C13-2 ("design first", next GPU slot after
g1bW). Builds on: g1/g1b/g1bR (the commit-and-project wall; WALL-
REPLICATES at n=3 seeds, mins 0.746/0.777/0.803; WALL-TAXES +0.53),
e182 (the only >=10x result in the lab: GPT-2 124M wash TEXTURE —
direction translates, clock does not), supervisor directives C11-2/
C12-4/C13-2 (the scale debt: every architectural law minted at
0.86-2.74M). What is NEW: the wall's first scale rung — no prior
g-claim has been tested above 2.74M.

## The question

Does the commit-and-project L2 ball hold a consolidated fact through
a wash that kills the control, at ~10x the parameters? Three outcomes
rewrite something: WALL-SCALES (the positive generalizes — the
architectural-law wording survives, with the tax re-priced at scale);
WALL-FADES (protection decays with scale — the R-dial or the wall's
premise is size-bound; the claim scopes); WALL-TIGHTENS (protection
improves — the basin grows with dimension, echoing g3K's graded
static basins in higher d).

## The instrument trap (Rule 12 / W021 — the design's core decision)

R=0.7 L2 was tuned at 2.74M params. Naively porting it to ~10M makes
the per-coordinate RMS budget 1.9x TIGHTER (R/sqrt(N) scaling) — the
cell would test a different wall and any failure would be an
instrument artifact. THE CONVENTION (frozen here): the R-dial is
carried in PER-COORDINATE RMS units, not raw L2. R_rms(g1bR @ 2.74M)
= 0.7/sqrt(2.74e6) ~ 4.2e-4. The g1bS ladder tests R in RMS-matched
units: {R_rms_match, 2x R_rms, 4x R_rms} (the g1 dial {0.7, 1.4,
4.2} L2 was ~{1,2,6}x — the same shape, dimension-matched). All
readouts report BOTH conventions (R_rms and raw L2) so no reader can
mistake the ruler. The wash itself is UNTOUCHED (the e1xx wash recipe
verbatim — the wash is the treatment, not the dial).

## The cell

HOST: a fresh ~10M TinyGPT char-LM (target 10M +/- 15%; exact config
recorded; corpus seed 1337 family). Envelope: single runs <= 180s;
the base trains in ckpt-RESUMABLE 180s chunks to a completed cosine
(a stated reason per directive 2: the wall assay needs a trained
base, not a pretrained LM — the fact must be INSTALLED, and GPT-2's
tokenizer/geometry would change three variables at once; e182
already covers the pretrained-LM lane).
ARMS (g1b verbatim in form): install fact (e043-Dmix convention) ->
consolidate (e113 jitter convention) -> commit(theta_anchor) -> W1
{R_rms ladder} vs C (unwalled) under the wash (seed 10902 family);
checkpoint cadence + fact battery + CE reads verbatim from g1b.
BARS (frozen at registration; the g1b form, dimension-matched):
  WALL-SCALES: "fires if at R_rms_match the fact's battery-channel
      p(Z) stays >= 0.9-equivalent (scaled to this host's root
      strength) at every checkpoint through +300 while C dies by +50."
  WALL-FADES: "fires if C dies but no R on the ladder holds the bar
      (or only the tightest R holds it while freezing CE)."
  WALL-TIGHTENS: "fires if a LOOSER R than rms-match holds (the basin
      grew with dimension) — co-report the CE tax curve at scale (the
      +0.53 reference re-priced)."
Honesty: n=1 host, one wash seed, one fact; the ladder rung is SCALE,
not seeds — replicate ladder follows only if a bar fires (the lab's
standard first-cell scoping). The tax comparison to +0.53 carries the
CPU/GPU-texture caveat.

## Cost/envelope

~6-10 GPU runs of <= 180s (base chunks + install + consolidate + wash
arms), cooldown(120) between; total wall ~25-40 min. GPU slot after
g1bW (C13-2). The design freezes the convention; the builder may not
re-tune the dial (anti-shopping: the ladder is fixed; if no rung
holds, WALL-FADES is the verdict, not a dial search).
