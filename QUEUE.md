# Experiment Queue

Statuses: `READY` (next up), `RUNNING`, `DONE (see NOTES.md)`, `PARKED`
(idea only). The hourly frontier review promotes/parks entries here.

| id | experiment | status | one-liner |
|---|---|---|---|
| e003 | forgetting selectivity frontier | DONE | P1 refuted (cos A-B 0.345), P2 confirmed (no selective LR), P3 refuted (French unlearning not selective either) — see NOTES.md |
| e003b | transient-selectivity audit + repair | READY (pending T002 final) | read early trajectory steps for a selectivity window; if found: early-stop + fluency-anchor ascent; if not: weight-targeted (low-overlap) or second-order ascent |
| e011a | write norms vs damage | DONE | H4 refuted: norms not monotone; damage/write falls 11× across attn layers; MLP-5 writes most, matters least |
| e011b | L0 redundancy + orthogonal innovation | DONE | 7.6× superadditive head ensemble; same-norm noise > zero everywhere; stream 8.4× jump at L0 — see T003 |
| e011c | matched-perturbation control | DONE | P1 refuted both ways: energy (not content) dominates; MLP-L1 direction-sensitive, MLP-L5 energy-carrier; bootstrap CIs pending |
| e014b | stream-renorm training | READY | train fresh net with constant-norm residual stream at block inputs; T003-B predicts lesion map flattens |
| e003b | corrected ascent instruments | READY | dense steps 0–30; projected + masked (top-k A-specific) ascent; target=train-A CE (memorization), collateral=val_B CE (labels fixed per critique) |
| e011 | MLP-0 anatomy | READY | what does the keystone organ (+4.08 nat lesion) store? probe, ablate-then-finetune recovery cost |
| e012 | decision-depth census | DONE | P2 confirmed (late-decided 3.04× more ablation damage), P3 confirmed (L5 calibrator, KL 1.03 nats), P1 refuted with sign flip (+0.32) |
| e001 | lesion map | DONE | see NOTES.md — front-loaded attention, keystone MLP-0, 16/48 dispensable |
| e002 | forgetting pilot | DONE | see NOTES.md — ascent is anti-selective at lr 2e-5; anchor slows but doesn't save |
| e004 | organ transplant | READY | train two nets on two corpora; swap layers/MLPs/heads; what transfers? |
| e005 | scaling ladder | READY | 1M→3M→10M→30M on same corpus; how does the lesion map change with depth/width? |
| e006 | neuron census | READY | rank individual MLP neurons by ablation damage; how sparse is the critical set? |
| e007 | grokking watch | READY | modular-arithmetic dataset; catch the generalization phase transition; dissect before/after |
| e008 | stitching | READY | two nets, same data, different seeds; can layer i of net A feed layer i+1 of net B? |
| e009 | evolution pilot | READY | mutate weights of a trained net, select by val loss, breed a lineage; Lamarck vs descent |
| e010 | growing nets | READY | start with 2 layers, grow to 8, retrain lightly; does knowledge survive growth spurts? |
| e013 | predict-and-poke | READY | pick a direction that "should" flip a behavior, poke it, score prediction vs surprise |
| e014 | pruning row-death + liveness repair | READY | magnitude-prune our LM to 95-99%; do whole organs (cf. MLP-0) die? does liveness vs feature-subspace repair rescue? (NEURO_AI_LAB #1/#2) |
| e015 | variable-depth training | READY | sample active layer-count during training; does it make the net robust to layer-drop/early exit? (NEURO_AI_LAB #6; pairs with E001 dead-late-attention) |
| e016 | reconsolidation vs real-EWC | READY | retrieval-gated labilize→reconsolidate vs properly-tuned EWC on sequential tasks (NEURO_AI_LAB #4) |
| e017 | sleep cycles | PARKED | periodic NREM-prune + REM-noisy-replay phases; forgetting/generalization at matched compute (NEURO_AI_LAB #5) |

## Visualization thread (`lab/vNNN_*`, standing — see README VISUALIZER)

| id | viz | status | one-liner |
|---|---|---|---|
| v001 | token journey | DONE | decision-depth observable discovered (T004); authority schedule visualized; polish: arrowheads + L0 headroom |
| v006 | decision-depth passage map | READY (viz) | color every char of a passage by its decision depth — does depth cluster on names, line-ends, dialogue turns? |
| v002 | attention atlas | READY | per-prompt 6×6 grid of head attention maps, annotated; find what attn-L5 actually attends to |
| v003 | write-space geometry | PARKED | PCA/dimensionality of each block's writes; overlap between blocks (who writes where) |
| v004 | lesion atlas explorer | PARKED | combine e001 damage + e011b subsets + write norms into one annotated anatomy poster |
| v005 | forgetting animation | PARKED | animate E003 ascent trajectories (Δtarget/Δcollateral + generated text decaying) |

## Parking lot (raw ideas, unranked)

- replay buffers vs catastrophic forgetting: how much replay is enough?
- curriculum as evolution pressure: does task order leave anatomical scars?
- pruning anatomy: what dies first when you remove 90% of weights?
- lottery tickets: do winning subnetworks exist at 1M scale?
- weight noise vaccination: does noise during training buy robustness to lesions?
- representation drift: same net retrained from same seed — how different are the organs?
- context-length scaling: what breaks first as sequence length grows?
- dual-culture nets: train on two languages/scripts, then selectively lesion one
- sleep: interleaved random replay after task B — does it rescue task A?
- scar tissue: after unlearning, retrain on the forgotten material — faster than fresh?
