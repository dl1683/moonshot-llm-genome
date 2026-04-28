**Q1.** Pattern is real, but it is narrower than “all PASSes are bad.” The failures are from **claim overreach beyond the comparator**, not from the raw effects disappearing. g174 fixed “generic regularization/dense target” for C18/C19. g177 is fixing “identity” for C22. I would not GPU-rerun every PASS. Do a desk “matched-null audit” on g158c/g165/g167/g172, then only rerun if the audit finds an actual missing comparator.

**Q2.** Without the wall-power meter, there is no honest 8.5+ path currently runnable. g173 can plausibly get to ~8.2, maybe 8.4 if cross-arch C3/FLOP results are clean and g177 preserves donor specificity. But energy-grade flagship requires g155, and g155 requires wall power. The narrative should explicitly say: **8.5+ is hardware-gated; otherwise the ceiling is 8.1-8.4.**

**Q3.** Budget is being well-spent because the adversarial cadence caught two publish-killing overclaims before external review did. But it should stay bounded: one adversarial cycle per strongest live claim, not a blanket rerun tax. Right now: finish g177, then run g173 if g177 passes or is intermediate. New mechanisms g178+ should wait unless g177 fails and forces a pivot.

**Q4.** I found no procurement update beyond the existing blocker note. `WIKI.md` still lists the external AC wall-power meter as the hard prerequisite for g155 and the top procurement priority. C22 provisional weakens g155’s donor-identity story slightly, but not the efficiency-demo value if KD/anchor transfer still works.

**Queue Score**
- g177 matched-alt falsifier: **9.5/10**. Must finish; it determines whether C22 survives.
- Wall-power procurement for g155: **9.0/10 action**, **blocked experiment**. Only honest 8.5+ unlock.
- g173 cross-arch FLOP cash-out: **8.0/10 if g177 PASS**, **6.5 if intermediate**, **5.0 if fail**.
- Matched-null desk audit of prior PASSes: **7.5/10**, cheap and protective.
- Full rerun audit of every PASS: **4/10**, too much compute unless desk audit flags a gap.
- g178+ new mechanisms: **5.5/10 now**, higher only after g177/g173 clarify the surviving claim.

**One Concrete Change**
Add a PASS-promotion rule immediately: any claim using “specificity,” “identity,” “mechanism,” “cash-out,” or “transfer” cannot be marked locked until the claim map includes a named matched-null/control row. Apply this as a desk audit to g158c, g165, g167, and g172 before publication, without preempting g177.