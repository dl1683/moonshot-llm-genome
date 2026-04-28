Local note: `g175` JSON is still incremental/incomplete, so this remains pre-commit.

**Decision tree**

- **PASS:** donor-identity specificity locked enough. Do not spend GPU on g176. Go straight to **g173 cross-arch FLOP cash-out**.
- **INTERMEDIATE:** run **one stronger alt-donor test**, not g176 yet. Use a heavily trained same-shape donor if available, with Frobenius/gradient-norm matching. The current Wikitext scratch-pretrained alt can be attacked as “too weak / wrong corpus / undertrained.” Higher lambda or longer training muddies the readout.
- **FAIL:** accept “trained-like basin” as the active ingredient. Next best experiment is **ATTN-only vs MLP-only anchoring** to localize the basin. That gives a concrete mechanism and an engineering handle; cross-task can come after.

**g176**

Only warranted if `g175` is intermediate after a stronger-alt check. If `g175` PASSes or FAILs cleanly, g176 is too broad. `g175 + one targeted follow-up` is cleaner.

**Wall Power**

No repo-recorded procurement update. Yes: a clean **8.5+ electricity-grade ceiling** is still gated on external AC wall-power logging. Without it, the best runnable path remains g173 FLOP cash-out, likely **8.1-8.4**, not 8.5+.

**g172 Follow-Up**

Most powerful: **cross-arch student, late-KD only**. Scaling to 12000 steps and last-1000 KD are useful schedule probes, but cross-arch late-KD upgrades C20 from single-family timing quirk to transferable efficiency mechanism.

**Queue Score**

- g173 cross-arch FLOP cash-out, with late-KD arm: **8.3**
- g175 stronger-alt donor, if intermediate: **7.6**
- ATTN-only vs MLP-only anchor, if fail: **7.3**
- g176 basin/content 2x2: **7.1**, only if ambiguity persists
- g172 12000-step scale: **6.7**
- g172 last-1000 KD: **6.5**
- g155 wall-power: **8.5 ceiling, 0 executable**

**One concrete change:** make `g173` include a **late-KD-only cross-arch arm** alongside full-KD and scratch, then score C3_macro per total train+teacher-query FLOPs.