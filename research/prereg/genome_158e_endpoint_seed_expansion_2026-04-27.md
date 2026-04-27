# genome_158e — endpoint seed expansion (Path B: WEAK_canonical)

**Status.** DRAFT 2026-04-27 — pending g158c WEAK_canonical verdict + Codex pre-flight sign-off before LOCK.
**Trigger.** Activates iff g158c returns WEAK_canonical (mean rho >= +0.5 AND mean Delta_256 >= +1.0pp AND CI(Delta_256) does not flip sign, but does NOT clear PASS_canonical).
**Source.** Codex direction consult 2026-04-27 (`codex_outputs/heartbeats/post_g158c_design_20260427T090500.md`) Path B.

## Hypothesis

If the architecture-prior inversion is real but variance-fragile at canonical scale, more seeds at the ENDPOINT contexts (where the predicted effect is sharpest) will either rescue the claim by tightening CIs or cleanly kill it. Adding 3 new seeds {17, 23, 37} to the existing canonical {42, 7, 13} for endpoint contexts ONLY (drop intermediate L=64, 128) yields N=6 endpoint observations per arm.

## Setup

- **Arms:** baseline_6L+MLP, minimal_3L_noMLP (same as g158c).
- **Contexts:** L ∈ {32, 256} ONLY (no intermediate).
- **Seeds:** [17, 23, 37] (3 NEW seeds; aggregate with existing {42, 7, 13} for N=6 endpoint).
- **Per-cell FLOP budget:** matched at 193.27 TFLOP (same as g158c).
- **LR selection:** per-arm at L=32 and L=256 separately, take min (same as g158c).
- **Eval:** C4 + Wikitext val (top-1 acc, NLL).

## Cells

12 train cells total: 2 contexts × 2 arms × 3 new seeds.

## Locked PASS / FAIL criteria (from Codex direction consult)

**PASS:**
- mean Delta_256(c4) across N=6 seeds >= +1.2pp
- mean Delta_32(c4) across N=6 seeds <= 0.0pp
- endpoint contrast `C = Delta_256 - Delta_32 >= +1.5pp`
- bootstrap 95% CI(C) excludes 0 (paired bootstrap on N=6 endpoint pairs)

**FAIL:**
- CI(C) crosses 0 OR
- mean Delta_32 > +0.2pp

## Compute envelope (COMPUTE.md §9 compliance)

- Wall-clock estimate: 3.0–3.5h (Codex direction consult estimate)
- Peak VRAM: < 4 GB (one ~30M BF16 model at a time)
- Peak RAM: < 16 GB

## Why this is highest-leverage Path B move (Codex rationale)

Per Codex direction consult: "weak canonical means the live question is variance, not 'maybe more train tokens would save it.' Budget expansion is post-hoc and easy to attack as moving the goalposts. Endpoint seed expansion keeps the estimand fixed, tests the sharpest unique prediction directly, and either rescues or kills the inversion honestly."

**Explicitly REJECTED alternative:** g158d (probe-budget expansion at 2x training tokens). Codex flagged this as p-hacking-vulnerable: "post-hoc and easy to attack as moving the goalposts."

## Codex score

- PASS=6.4/10. Upgrades a weak full-curve claim to a clean replicated endpoint inversion; still doesn't reopen "measured design law."
- FAIL=6.0/10. Kills last transport-demand claim without p-hacking or budget drift; honest thing to know.

## Files (to be created when LOCKED)

- `code/genome_158e_endpoint_seed_expansion.py` — fork of `code/genome_158c_3seed_canonical.py`, restrict L to {32, 256}, replace SEEDS with [17, 23, 37], add bootstrap 95% CI on contrast.
- `results/genome_158e_endpoint_seed_expansion.json`
- `code/integrate_g158e.py` — verdict integration helper. Reads BOTH `genome_158c_3seed_canonical.json` AND `genome_158e_endpoint_seed_expansion.json`, aggregates to N=6 endpoint, applies decision rule.

## Pre-flight protocol

Before LOCK, fire Codex consult: "Read `code/genome_158e_endpoint_seed_expansion.py` (when written) and this prereg. Audit (a) bootstrap CI methodology on paired N=6 contrast, (b) seed independence (no overlap with {42, 7, 13}), (c) corpus slice handling — should the new seeds reuse `c4_clean_v1(seed=77)` for protocol consistency, or use a different corpus seed to add data-slice variance?"

## Provenance

- g158 PILOT: `results/genome_158_context_length_inversion.json` (single-seed)
- g158c canonical: `code/genome_158c_3seed_canonical.py` (running)
- Codex direction consult: `codex_outputs/heartbeats/post_g158c_design_20260427T090500.md`
- Decision tree: `research/programs/post_g158c_decision_tree.md` Path B
