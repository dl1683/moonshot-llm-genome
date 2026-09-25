# Paper bibliography — verified citation set — RESEARCHER, 2026-09-25

Source of truth for the P-A draft (day3_paper_draft.md). Every entry below was
checked against arXiv/OpenReview/ACL Anthology via live web search on
2026-09-25. Status tags: VERIFIED (exists exactly as cited), CORRECTED-FROM
(exists but our scans had something wrong — old value noted), ADDED (new
must-cite for a draft claim with no prior scan coverage), UNVERIFIED (could
not be located; do not cite without manual confirmation).

Counts: 16 VERIFIED / 5 CORRECTED / 7 ADDED (+1 scan item whose missing ID was
supplied) / 2 UNVERIFIED (probable scan hallucinations — see Section E).

---

## A. The eight primary citations (draft intro + contribution scoping)

1. **Orgad, Toker, Gekhman, Reichart, Szpektor, Kotek, Belinkov. "LLMs Know
   More Than They Show: On the Intrinsic Representation of LLM
   Hallucinations." ICLR 2025. arXiv:2410.02707.**
   — VERIFIED. Author list confirmed character-exact against the arXiv abs
   page (incl. Idan Szpektor, Hadas Kotek, Yonatan Belinkov; v4 May 2025).
   OpenReview forum KRnsX5Em3W confirmed; ICLR 2025 venue confirmed via
   OpenReview/citation records. (Note: one reproduction claiming code bugs
   circulates on X — keep the effect-size caveat from frontier scan.)

2. **Buckmann, Nguyen, Hill. "Revealing economic facts: LLMs know more than
   they say." arXiv:2505.08662 (May 2025).**
   — VERIFIED. Title/authors/ID exact. Coins "elicitation failure" as claimed.
   Preprint only (34 pp, journal-format); no conference venue — cite as arXiv.

3. **Luo, Chu, He, Wang, Qin, Wu, Chen. "You Only Pass Once: Answering and
   Abstaining Together in a Single Forward Pass of a Frozen Language Model."
   arXiv:2608.14465 (14 Aug 2026).** [YOPO]
   — CORRECTED-FROM: scans called it "You Only Pass Once: Answering and
   Abstaining" (truncated title). Full title above per arXiv abs page. Two
   more scope fixes: backbone is Qwen2.5 {1.5B, 3B, 7B}, not just 7B; and no
   venue is listed (arXiv preprint, 24 pp; Luo and Chu equal contribution).
   Domain claims in the scan (abstention/sufficiency, RepLiQA anchoring,
   steering-vs-read interference) match the abstract; the "layer 19" /
   "124/125" specifics come from our earlier HTML-v1 read — fine to keep, but
   they are v1 numbers.

4. **Yan (Tianyi Lorena Yan), Jia. "Promote, Suppress, Iterate: How Language
   Models Answer One-to-Many Factual Queries." EMNLP 2025 (Main).
   arXiv:2502.20475.**
   — VERIFIED. Anthology ID 2025.emnlp-main.815 confirmed. Two authors (not
   "Yan & Jia" as a single name — it is Yan and Jia, USC).

5. **Gu, Pang, Du, Liu, Guo, Pai, Bai, Jiao. "When Attention Sink Emerges in
   Language Models: An Empirical View." ICLR 2025. arXiv:2410.10781.**
   — VERIFIED. Full author list confirmed via ACM/OpenReview citation records.

6. **"KVSink: Understanding and Enhancing the Preservation of Attention Sinks
   in KV Cache Quantization for LLMs." COLM 2025. arXiv:2508.04257.**
   — VERIFIED. Scan had only "KVSink (COLM 2025)"; full title recovered.
   (Author list not individually captured in this pass — pull from arXiv page
   at camera-ready.)

7. **Guo (Phyllicia), Syed, Sheshadri, Ewart, Dziugaite. "Mechanistic
   Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic
   Localization." ICML 2025. arXiv:2410.12949.**
   — VERIFIED. Full author list + venue confirmed (OpenReview 92oBV5HAGl,
   ICML 2025 per DBLP/open-access record).

8. **Modarressi et al. "NoLiMa: Long-Context Evaluation Beyond Literal
   Matching." ICML 2025. arXiv:2502.05167.**
   — VERIFIED. First author Ali Modarressi; 13 evaluated LLMs; ICML 2025.

## B. Secondary citations named in the draft/scan chain

9. **Li (Kenneth), Patel, et al. "Inference-Time Intervention: Eliciting
   Truthful Answers from a Language Model." NeurIPS 2023 (spotlight).
   arXiv:2306.03341.** — VERIFIED. [ITI]

10. **Chuang, Xie, Luo, Kim, Glass, He. "DoLa: Decoding by Contrasting Layers
    Improves Factuality in Large Language Models." ICLR 2024.
    arXiv:2309.03883.** — VERIFIED.

11. **Li (Xiang Lisa), Holtzman, Fried, Liang (P.), Eisner, Hashimoto,
    Zettlemoyer, Liang (P.). "Contrastive Decoding: Open-ended Text Generation
    as Optimization." ACL 2023. arXiv:2210.15097.**
    — VERIFIED (ID supplied — scan named the paper without an ID; note the ID
    is 2210.15097; the adjacent 2210.15091 is an unrelated MS-imaging paper).

12. **Heimersheim & Nanda. "Towards Best Practices of Activation Patching in
    Language Models: Metrics and Methods." arXiv:2309.16042.**
    — VERIFIED. Companion tutorial: arXiv:2404.15255 ("How to use and
    interpret activation patching").

13. **Bürger, Hamprecht, Nadler. "Truth is Universal: Robust Detection of Lies
    in LLMs." NeurIPS 2024. arXiv:2407.12831.**
    — CORRECTED-FROM: scan said "Nadler et al." with no ID/venue. First
    author is Bürger; NeurIPS 2024. Claim-level warning: the paper's result is
    that a truth direction supports robust *lie detection* even in models
    instructed/aligned to lie; the scan's gloss "argues RLHF widens the
    elicitation gap" is NOT what the located abstract says — soften or drop
    that gloss if cited.

14. **Afzal, Matthes, Chechik, Ziser. "Knowing Before Saying: LLM
    Representations Encode Information About Chain-of-Thought Success Before
    Completion." Findings of ACL 2025. arXiv:2505.24362.**
    — VERIFIED (ID supplied; scan had title+venue but no ID). Anthology:
    2025.findings-acl.662.

15. **Cundy & Gleave (FAR AI). "Preference Learning with Lie Detectors can
    Induce Honesty or Evasion." arXiv:2505.13787 (May 2025); NeurIPS 2025.**
    — CORRECTED-FROM: scan title was "Preference Learning with Lie Detectors
    can Induce Deception" — WRONG. Actual title above. Scan date "Nov 2025"
    also wrong (May 2025 preprint; NeurIPS 2025). Substance (training against
    probes yields evasion; probe-visible knowledge driven underground, not
    deleted) survives.

16. **Duan (Jinhao), et al. "TruthPrInt: Mitigating Large Vision-Language
    Models Object Hallucination via Latent Truthful-Guided Pre-Intervention."
    CVPR 2025. arXiv:2503.10602.** — VERIFIED (ID supplied; 9 authors).

17. **Yang (Tianyun), et al. "Understanding and Mitigating Hallucination in
    Large Vision-Language Models via Modular Attribution and Intervention."
    ICLR 2025. OpenReview Bjq4W7P2Us.**
    — VERIFIED. (No arXiv ID surfaced in this pass; cite via OpenReview, or
    pull the arXiv mirror at camera-ready.)

18. **Sarkar, Che, Gavin, Beerel, Kundu. "Mitigating Hallucinations in
    Vision-Language Models through Image-Guided Head Suppression." EMNLP 2025
    (Main). arXiv:2505.16411; anthology 2025.emnlp-main.631, pp. 12481–12500.**
    — VERIFIED, including the exact anthology ID the scan gave (checked
    directly — it is correct).

19. **Zhao, Köksal, Modarressi, Hedderich, Schütze. "Do We Know What LLMs
    Don't Know? A Study of Consistency in Knowledge Probing."
    arXiv:2505.21701 (May 2025).**
    — CORRECTED-FROM: scan said "ACL 2025"; the arXiv page lists NO venue
    (preprint). Intra-method agreement ~40%, cross-method consistency as low
    as 7% — supports our battery-overstatement control as claimed.

20. **Miao (Miranda Muqing), et al. "Correctness-Optimized Residual Activation
    Lens (CORAL): Transferrable and Calibration-Aware Inference-Time
    Steering." arXiv:2602.06022 (Feb 2026).**
    — VERIFIED. Use as the ITI-generalization critique (scan's intent).

21. **"The Mirage of Performance Gains: Why Contrastive Decoding Fails to
    Mitigate Object Hallucinations in MLLMs?" arXiv:2504.10020 (Apr 2025).**
    — CORRECTED-FROM: scan shorthand "Why contrastive decoding fails".
    Scope correction: this critiques contrastive decoding for *MLLM object
    hallucination* (gains = MCQ-format artifacts); it is adjacent to, not a
    direct rebuttal of, DoLa. Keep the DoLa "assumes, never measures" point as
    OUR argument, not this citation's.

## C. ADDED must-cites (draft claims with no prior scan coverage)

### C1. Mechanistic unlearning / RMU obfuscation (backs §5.1/§5.2 framing:
### "representation-level methods hide rather than delete; parametric surgery
### is the alternative")

22. **ADDED — Li (Nathaniel), Pan, Gopal, Yue, ... Hendrycks (56 authors).
    "The WMDP Benchmark: Measuring and Reducing Malicious Use With
    Unlearning." arXiv:2403.03218 (2024).**
    — VERIFIED via arXiv abs page. Origin of RMU (Representation
    Misdirection for Unlearning) — cite this whenever RMU is named.

23. **ADDED — Fan (Chongyu), Wang, Huang, Pal, Liu, et al. "LLM Unlearning
    Under the Microscope: A Full-Stack View on Methods and Metrics."
    arXiv:2510.07626 (8 Oct 2025).**
    — VERIFIED. The 2025 consensus paper that unlearning evaluation must
    check latent-knowledge recovery (obfuscation, not deletion).

24. (Scan-completed, not new) **Shilov, Cloud, Gema, Goldman-Wetzler,
    Panickssery, Sleight, et al. (Anthropic). "Beyond Data Filtering:
    Knowledge Localization for Capability Removal in LLMs."
    arXiv:2512.05648 (Dec 2025).**
    — VERIFIED (scan named the paper without an ID; ID supplied). Closest
    big-lab counterpart to our localization-vs-suppression stance.

### C2. Model editing side effects (backs §5.2 "constitutionally silent" and
### the editing-law framing: edits ripple/side-effect)

25. **ADDED — Cohen, Biran, Yoran, Globerson, Geva. "Evaluating the Ripple
    Effects of Knowledge Editing in Language Models." TACL 12:283–298, 2024.
    arXiv:2307.12976.**
    — VERIFIED (anthology 2024.tacl-1.16; RippleEdits benchmark). The
    canonical "edits have side effects" citation.

26. **ADDED — Zhong, Wu, Manning, Potts, Liang. "MQuAKE: Assessing Knowledge
    Editing in Language Models via Multi-Hop Questions." EMNLP 2023.
    arXiv:2305.14795.**
    — VERIFIED (anthology 2023.emnlp-main.971). Multi-hop propagation
    failure; pairs with (not replaces) MQuAKE-Remastered ICLR 2025. (Scan had
    implied ICLR 2024 for MQuAKE — wrong; original is EMNLP 2023.)

### C3. Exposure bias / teacher-forcing gap (backs §2.1, §4.4, Risk 3: the
### TF/free-run gap is textbook — but behavioral/training-side only)

27. **ADDED — Ranzato, Chopra, Auli, Zaremba. "Sequence Level Training with
    Recurrent Neural Networks." ICLR 2016. arXiv:1511.06732.**
    — VERIFIED via arXiv abs page. Coined/named the exposure-bias
    (train-test mismatch) problem — the "2016 textbook" anchor for Risk 3.

28. **ADDED — Bengio, Vinyals, Jaitly, Shazeer. "Scheduled Sampling for
    Sequence Prediction with Recurrent Neural Networks." NeurIPS 2015.
    arXiv:1506.03099.** — VERIFIED via arXiv abs page.

29. **ADDED — Lamb, Goyal, Zhang (Ying), Zhang (Saizheng), Courville, Bengio.
    "Professor Forcing: A New Algorithm for Training Recurrent Networks."
    NIPS 2016. arXiv:1610.09038.** — VERIFIED via arXiv abs page. The
    adversarial TF/free-run-state match — the closest *state-level* prior to
    our transplant framing; cite-and-distinguish (they train a discriminator
    on state distributions; we transplant a state at a named token).

## D. Venue-only notes

- Orgad: cite "ICLR 2025" (forum KRnsX5Em3W); the arXiv page itself carries no
  venue comment — venue from OpenReview record.
- YOPO and Buckmann: cite as arXiv preprints (no venue on arXiv as of
  2026-09-25).
- WMDP: no venue on arXiv page; commonly cited as ICML 2024 — do not assert
  the venue without checking the proceedings at camera-ready.

## E. UNVERIFIED — probable scan hallucinations; do NOT cite as-is

- **"Bridge-Garden" (claimed: ICML 2026, distillation gains attributed to
  reduced exposure bias)** — NOT FOUND in any search. The draft's Risk-3
  parenthetical "(self-recovery, concavity theory, distillation attribution)"
  leans on this and the next item; replace with the verified C3 classics
  (Ranzato/Bengio/Lamb), which fully carry the "exposure bias is textbook and
  behavioral-only" point.
- **OpenReview EGz8InJz6F (claimed: teacher-forced log-prob concavity
  theory)** — ID not confirmed to resolve to that paper; the one candidate
  match found ("Token-Level Control or Just a Better Mean?") is a different
  topic. Drop the "concavity theory" phrase or re-source it manually.

---

### Camera-ready micro-checklist (carry into day3_paper_draft checklist)

- [ ] Fix FAR AI title (→ "…can Induce Honesty or Evasion", arXiv:2505.13787).
- [ ] Fix YOPO title (full title) + write "Qwen2.5 1.5B/3B/7B" not "7B".
- [ ] "Truth is Universal" → Bürger et al., NeurIPS 2024, arXiv:2407.12831;
      drop the "RLHF widens the elicitation gap" gloss.
- [ ] "Do We Know What LLMs Don't Know?" → arXiv:2505.21701, no venue.
- [ ] Contrastive Decoding ID = arXiv:2210.15097 (not …091).
- [ ] MQuAKE original = EMNLP 2023, arXiv:2305.14795; MQuAKE-Remastered =
      ICLR 2025 (separate paper, S. Zhong et al.).
- [ ] Remove/replace "Bridge-Garden" and "EGz8InJz6F" references (Risk 3).
