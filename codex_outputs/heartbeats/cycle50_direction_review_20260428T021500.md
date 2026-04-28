**Q1.** Run a 2×2 donor-anchor attribution, because trained-on-C4 donor confounds “basin” and “content.”

Arms:
- natural trained donor: low-loss + correct content
- wrong-task trained donor: matched NLL/norm, trained on fixed vocab-permuted or shuffled-target C4
- task-structured but not full-basin donor: trained trunk/body anchor with randomized/refit head, body-only anchor
- random/permuted nulls

Readout:
- wrong-task low-loss helps → basin/landscape dominates
- body-only task structure helps while wrong-task fails → content transfer dominates
- only natural trained donor is strong → (c), basin and learned content are coupled

**Q2.** Do not do per-shape Frobenius as the cross-arch extension. That tests an implementation artifact and likely produces “no shared coordinates.” The same mechanism cross-arch is: continuous donor-specific information in the loss, expressed in a shared functional space. So use Qwen teacher → Llama-arch student KD/logit constraint with trained/random/uniform teacher nulls and honest train+inference FLOPs. Treat raw cross-arch Frobenius only as a predicted FAIL boundary for weight-space anchoring.

**Q3.** No repo-recorded procurement update. The case is stronger, not weaker: g174 PART A makes donor-specificity credible, so g155 becomes less like an efficiency stunt and more like electricity-grade cash-out of a real transfer law. Still blocked until external AC wall-power logging exists.

**Q4.** Yes, g172 deserves its own CLAIM_EVIDENCE_MAP entry. Wording: “late-stage KD recovers 69% of full-KD gain with 33% KD-active compute; early KD retains 14%.” Universality level: Level-0 / single-family mechanism finding now, not Level-1/2/3. Upgrade only after cross-architecture or cross-task replication.

**Queue Score**
- g174 PART B: 9.5/10. Finish; gates KD donor-specificity narrative.
- g173 cross-arch FLOP cash-out: 8.2/10 if PART B passes; 7.5 if PART B fails.
- g155 wall-power TEI/kJ: 8.2 ceiling, 0 executable until meter.
- Q1 2×2 basin/content attribution: 7.4, high theory value but secondary to cash-out.
- per-shape cross-arch Frobenius: 4.5, mostly boundary-test.

**One Concrete Change**
Add g172 as a standalone C20 claim now, with Level-0/single-family status and the PART B-dependent caveat.