# Experiment Queue

Statuses: `READY` (next up), `RUNNING`, `DONE (see NOTES.md)`, `PARKED`
(idea only, no live-hypothesis discrimination), `GATED` (waiting on a
prerequisite). Rewritten at Review 1 (2026-09-24T11:20Z) to fix drift.

| id | experiment | status | one-liner |
|---|---|---|---|
| e013a | attention census | DONE | funnel replicates at scale (far-mass U 0.80→0.09→0.54); L5 abandons local in 82.5% of prompts; rare-token story DEAD (0 concentrated heads, flat surprisal) |
| e013 | context-truncation calibration test | DONE | REFUTED: L5 calibration is local (KL −6.9%); 16-token sufficiency — far context worth ≈0 nats at char level; mid-stack readouts anti-informative |
| e013c | far-value tail distribution | READY | per-position (full−trunc) ΔCE: uniform ≈0 or rare far-dependent positions? gates any future long-range claim |
| e028 | cross-anatomy transplant | READY | design memo complete (scratch/e028_transplant_design.md): C0 self-transplant gate, seed-43 within-control, R=transplant/ablation bands; tests T006 P3 |
| e019 | MLP-5 thermostat | READY | scale MLP-5 write by α∈{0,.5,1,2} + rotate; entropy/top-k/CE response — direct causal test of the energy-carrier claim (eval-only, minutes) |
| e003b | corrected ascent instruments | READY | projected + masked (top-k A-specific) ascent, dense steps 0–30; target=train-A CE, collateral=val_B CE (labels fixed per critique) |
| e013 | rare-token causal mask | SUPERSEDED | census found no concentrated rare-token heads; replaced by context-truncation design |
| e014c | write-clamp training | READY | clamp ‖w‖ ≤ α·‖x_in‖ during training (or eval-time rescale L0/L5 writes ×{0.5,2,4}) — decisive test of "damage tracks write allocation" (P3 passed correlationally) |
| e018 | causal depth | READY | activation-patching depth: shallowest d where splicing a counterfactual context switches the decision — upgrades T004 past the depth-6/L5 circularity |
| e014b.1 | replication seed | READY (debt) | second seed for the renorm-plasticity result (anatomy plasticity is single-seed) |
| e011c-ci | bootstrap CIs | READY (debt) | resample eval batches for e011c rotate/zero ratios (MLP-L1 ×2.95, MLP-L5 ×0.24 beyond noise?) |
| e001–e003, e011a/b/c, e012, e014b | — | DONE | see NOTES.md |
| e027 | predict-and-poke | PARKED | (was e013 collision) pick a direction that should flip a behavior, poke it, score prediction vs surprise |
| e004–e011, e015, e016 | — | PARKED | no live-hypothesis discrimination; e011 refolded into e019 |

## Visualization thread (`lab/vNNN_*`, standing — see README VISUALIZER)

| id | viz | status | one-liner |
|---|---|---|---|
| v001 | token journey | DONE | decision-depth observable discovered (T004); authority schedule visualized |
| v002 | attention atlas | DONE | locality funnel (L0→L3→L4/L5); L5 local-abandonment real, rarity = one head (Review 1) |
| v006 | decision-depth passage map | DONE | letters late / structural chars early; 51% finalize at L5 |
| v007 | funnel film | PARKED | animation through depth: distance histograms + rare-token spotlight on dialogue |
| v008 | anatomy phylogeny | PARKED | 8-12 seeds × regimes; embed lesion-map vectors; which organs are conserved homologs vs plastic |
| v003 | write-space geometry | PARKED | PCA/dimensionality of each block's writes |
| v004 | lesion atlas explorer | PARKED | composite anatomy poster |
| v005 | forgetting animation | PARKED | animate ascent trajectories + generation decay |

## Parking lot (raw ideas, unranked)

- e020 context surgery: duplicate/delete the distant rare token; depth + KL response (R1 vs R2)
- e021 task-swap: copy-task vs word-shuffled vs Shakespeare — is front-loading task-dependent?
- e022 frequency chase: make 'O' common (or rare-ify a common char), re-run atlas
- e023 entity-granularity forgetting: anchored ascent on one name's windows vs embedding+lm_head row surgery
- e024 reconsolidation window (bio-analogue): retrieval-gated labilization then matched-dose noise vs control
- e026 selection-on-depth (evolution): lineages selected under L5-ablated loss; is depth selectable?
- dual-culture nets (successor if e003b refutes selective forgetting at all granularities)
- replay buffers vs catastrophic forgetting; sleep replay; curriculum scars; lottery tickets; scar tissue (retrain after unlearning)

## Parking-lot promotions policy
Reviews promote at most 1-3 items to READY; ideas that discriminate live
hypotheses (THINKING.md) outrank new topics. Replication debt outranks new
lines when a load-bearing claim is single-seed.
