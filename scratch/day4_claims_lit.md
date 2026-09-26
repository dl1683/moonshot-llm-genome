# Day-4 claims prior-art scan (2026-09-25)

Online scan for the two day-4 promotion candidates. Companion to
`positional_binding_lit.md` (which covers the activation-level transplant);
this file covers the NEW weight-level single-row claim and the entry-level
exposure-bias claim. CPU-only, no repo edits. Search engines: 3 (web search
prime, arXiv API, direct fetches). Honesty note: absence of evidence in
~10 targeted queries per claim, not proof of absence; two items flagged
below for full-text re-check.

---

## Claim 1 — "The single-row address" (wpe-129 portability)

Claim under check: an installed fact's address localizes to ONE portable
position-embedding row (wpe-129, the decision-position code); necessary
(one-char shift kills expression), sufficient (~70% expression restored by
copying that row alone), surgically targeted in place (KL 0.2–0.5) vs
generic window-start row (KL 2.6–7.2).

### (a) Positional-embedding row-level causal edits

**No prior art found.** Nothing in 2023–2026 does per-row transplant,
rebinding, or in-place surgical retargeting of learned position-embedding
rows as a causal test of where a trained memory's address lives. Closest
encounters, all distinguishable:

- **TAPE — "Rethinking Addressing in Language Models via Contextualized
  Equivariant Positional Encoding"** (Zhu et al., arXiv:2501.00712, ICML
  2025). The nearest framing: explicitly argues transformers use
  "content-based and position-based addressing" and that standard PEs are
  generic biases rather than instance-specific addresses. BUT: it *builds*
  dynamic context-aware encodings (a training method); it performs no
  row-level causal edits, no transplants, and makes no claim that knowledge
  localizes to specific positions. Useful to us as vocabulary ("addressing")
  and as evidence the field sees the addressing gap — it does not touch our
  experimental move.
- **Knowledge editing (ROME arXiv:2202.05262; MEMIT; FiNE arXiv:2503.01090).**
  Edits are rank-one/low-rank updates to MLP matrices; facts are implicitly
  anchored to "where the subject token ends" via causal tracing, but no one
  edits position-embedding rows, and no one frames an edit as *rebinding an
  address row*. FiNE reports edits are entity-bound, not position-bound —
  different binding axis from ours.
- **Whole-matrix position-embedding interventions** (opposite direction,
  gross scale): "Extending the Context of Pretrained LLMs by Dropping Their
  Positional Embedding" (arXiv:2512.12167) removes wpe entirely for context
  extension; NoPE (arXiv:2305.19466, NeurIPS 2023) trains without PE. These
  support "wpe is exploitable as a cheap absolute address" but never isolate
  a single row.
- **Descriptive wpe geometry** (helix structure, LessWrong 2023; orthogonality
  to token space, Simon 2024; Dufter et al. survey arXiv:2010.04903): rows
  are smooth manifold points, not individually causal handles. No one asks
  "what does row i *do* when a fact is installed at position i".
- Unverified rumor: a search summarizer claimed a paper "Frozen Transformers
  are Even More Universal Computers" programs frozen transformers via
  position embeddings. arXiv title/full-text API search found NO such paper
  (only Lu, Grover, Abbeel, Mordatch, "Pretrained Transformers as Universal
  Computation Engines", arXiv:2103.05247, which trains linear I/O layers
  around a frozen backbone — no position rows). Treat the rumor as a
  hallucination unless a human finds the paper; re-check before submission.

### (b) KV-cache / attention "position anchors" vs our row-level weight edit

- **Attention sinks / StreamingLLM** (Xiao et al., arXiv:2309.17453) + 2026
  survey: first tokens act as attention "anchors"/sinks, kept permanently in
  the KV cache; KV-pruning methods (H2O arXiv:2306.14048, SnapKV
  arXiv:2404.14469; "practical robust KV pruning" retains first-A tokens as
  anchors) manage anchor tokens at runtime. ALL of this is inference-time
  cache/activation management; none of it writes a persistent weight edit,
  and none relocates an installed memory's address. Our result is the
  complementary statement: the anchor-like role of a position can be *in the
  weight table itself*, and can be moved by editing one row.
- **Inference-time position re-mapping** (SelfExtend, PoSE/ICLR, NTK-aware
  tricks): re-map position indices coarsely (grouped neighbors, skipped
  offsets) for length extension. Global, input-agnostic re-indexing — not
  per-fact addressing, no transplant assay, no KL-surgical claim.

### (c) Soft-prompt / prompt-tuning moving function between positions

- **Prefix tuning** (Li & Liang, arXiv:2101.12090), **prompt tuning**
  (Lester et al., arXiv:2104.08691), **P-tuning v2** (arXiv:2110.07677):
  learn continuous vectors *at chosen positions*, added to activations at
  runtime. Closest conceptual neighbor to "a learned per-position vector
  carries function". Distinguishing axes, all in our favor: (i) they are
  runtime input/activation objects, not rows of the position-embedding
  weight matrix; (ii) function typically occupies a prefix *block*, not a
  single row; (iii) no necessity/sufficiency claim about an *installed-by-
  training* memory's address; (iv) no in-place retarget of an existing
  memory with a distribution-shift cost (our KL 0.2–0.5 vs 2.6–7.2
  surgical-versus-generic contrast has no analogue in that literature).
- **Function vectors** (Todd et al., arXiv:2310.15213, ICLR 2024) and
  **task vectors** (Hendel, Geva, Globerson, arXiv:2310.15916, EMNLP
  Findings 2023): portable activation vectors, explicitly position-
  invariant. This is the contrast pole already identified in
  positional_binding_lit.md — the field's default is position-robust
  portability; our single-row address is position-*definite* portability,
  and the row itself is the portable object.

### Verdict — Claim 1

**NOVEL / claimable.** "Single-row portability of an installed memory
address" — necessary + sufficient single wpe row, transplantable to a
shifted window, surgically retargetable in place with small KL — has no
found prior. Perimeter to state explicitly in the paper: (1) scoped to
learned absolute position embeddings (GPT-2-style wpe); RoPE-era LLMs have
no rows to edit, so frame generality accordingly; (2) distinguish from
ROME-family MLP edits (content-side, position only implicit), prompt/prefix
tuning (runtime vectors, not weight rows), attention-sink anchors (runtime
cache, not weights), TAPE (addressing vocabulary without row-level
causality); (3) the activation-level companion (matched-vs-shifted
transplant) is separately novel per positional_binding_lit.md — together
they make an "address lives in the position row, content lives elsewhere"
two-factor claim nobody else has run. Re-check flag: the unverified
"frozen transformers + position-embedding programming" rumor above.

---

## Claim 2 — "Entry-level exposure bias" (self-generated poison)

Claim under check: in free-running generation, negative-utility cache
entries concentrate in the model's OWN generated tokens (4/4 nets; up to
36.7% of beyond-onset generated entries lesion-helpful at 10M), while
corpus-prompt entries almost never hurt.

### (a) Exposure-bias literature — is the poison localized anywhere?

No. The lineage is behavioral/distributional, never entry-level:

- **Scheduled sampling** (Bengio et al., arXiv:1506.03099, 2015) and
  **sequence-level RL** (Ranzato et al., arXiv:1511.06732, 2016): define
  the train/test mismatch (teacher forcing vs own tokens) and mitigate it.
- **"Why Exposure Bias Matters: An Imitation Learning Perspective of Error
  Accumulation in Language Generation"** (Arora, El Asri et al., AAAI 2022,
  arXiv:2110.05978): the deepest analysis found — imitation-learning bound
  on error accumulation from conditioning on own generations. Still
  aggregate-level: no localization of WHERE in the context harmful entries
  sit, no cache-entry analysis, no token-source split.
- **Degeneration work** (Holtzman et al. arXiv:1904.09751; Welleck et al.
  unlikelihood arXiv:1908.04319): documents that free-running text goes
  repetitively bad, i.e., self-generated junk exists — but at the text
  level, with no per-entry utility measurement.

### (b) KV-cache pruning/compression — junk split by token source?

No source split found anywhere in the pruning/compression literature:

- **H2O** (arXiv:2306.14048), **StreamingLLM** (arXiv:2309.17453),
  **SnapKV** (arXiv:2404.14469), **Scissorhands** (arXiv:2305.17118),
  **KIVI** (arXiv:2402.02750), **SCBench** KV-cache-centric benchmark
  (Li et al., OpenReview), **KVzip** (2025): eviction/importance is by
  attention mass, recency, or sinks — the *origin* of a token (self-
  generated vs prompt-given) is never a variable. Multiple targeted
  searches ("self-generated vs prompt tokens eviction") returned nothing;
  one summarizer explicitly noted the asymmetry we study "is precisely the
  gap".
- **SinkProbe — "Attention Sinks as Internal Signals for Hallucination
  Detection"** (Binkowski, Adamczewski, Kajdanowicz, arXiv:2604.10697,
  ICML 2026): the nearest 2026 work. Links sinks (incl. sinks among
  *generated* tokens, per a secondary description) to hallucination as a
  detection signal. Distinguishers: it is detection-oriented (probe on
  attention maps), not causal lesioning; no per-entry utility scoring; the
  abstract does not split sinks by token source. FLAG: full text not read —
  verify the generated-token-sink analysis before claiming the source-split
  gap against this paper specifically.
- **Agent-security "context/memory poisoning"** (2025–2026 practitioner +
  arXiv literature, e.g. GhostWriter-style delayed-activation memory
  poisoning, sleeper-memory papers): adversarial injected content
  persisting across turns. Different phenomenon (attacker-supplied, not
  self-generated; security framing, no lesion assay), but shows the
  community cares about *which* context entries corrupt later behavior —
  cite as motivation.

### (c) Hallucination snowballing / self-reinforcing error

- **Zhang, Press et al., "How Language Model Hallucinations Can Snowball"**
  (arXiv:2305.13534, ICML 2024, ~570 citations): states the BEHAVIORAL
  version outright — an LM over-commits to its own early mistakes, and
  generates further mistakes it would not otherwise make; models often can
  separately recognize the errors. So "the run's own outputs poison its own
  subsequent generation" is STATED here, at the behavior level. NOT stated:
  where the poison sits (which entries), that removal of specific entries
  helps (lesion utility), or the self-vs-given asymmetry. Our claim is the
  mechanistic entry-level refinement of this paper.
- **Farquhar, Kossen, Kuhn, Gal** "Detecting hallucinations in LLMs using
  semantic entropy" (Nature 630:625–630, 2024; arXiv:2406.15927): entropy-
  over-meanings detection; says nothing about localization in context. Cite
  as the detection-era context, not prior for our claim.
- **Model collapse** (Shumailov et al., Nature 2024; self-consuming loops,
  Alemohammad et al.): recursive *training* on own outputs — different
  level (weights over generations, not one run's cache). Cite to preempt
  conflation.

### Verdict — Claim 2

**PARTIALLY KNOWN (behavioral) / NOVEL (entry-level localization).**
Known: exposure bias (2015–), imitation-learning error accumulation
(2022), and snowballing (2023/ICML 2024) already assert that a model's own
outputs degrade its subsequent generation. Novel (no prior found): (1) the
localization assay — per-cache-entry lesion utility during free-run;
(2) the source split — negative-utility entries concentrate in self-
generated tokens while corpus-prompt entries almost never hurt (4/4 nets,
up to 36.7%); (3) the implication for KV-cache pruning that token source is
an unexploited, cheap eviction prior. Position the claim as "entry-level
account of exposure bias / snowballing", citing Zhang+Press and Arora as
the behavioral priors we mechanize. Naming it "entry-level exposure bias"
is defensible. Re-check flag: SinkProbe (2604.10697) full text.

---

## Sources

Claim 1:
- TAPE, arXiv:2501.00712 — https://arxiv.org/abs/2501.00712
- ROME, arXiv:2202.05262 — https://arxiv.org/abs/2202.05262
- FiNE, arXiv:2503.01090 — https://arxiv.org/abs/2503.01090
- Dropping positional embeddings for context extension, arXiv:2512.12167 — https://arxiv.org/html/2512.12167v1
- NoPE, arXiv:2305.19466 — https://arxiv.org/abs/2305.19466
- GPT-2 wpe helix — https://www.lesswrong.com (post "GPT-2's positional embedding matrix is a helix", Jul 2023)
- GPT-2 positional encodings orthogonality — https://jamiesimon.io (2024)
- Dufter et al. position survey, arXiv:2010.04903 — https://arxiv.org/abs/2010.04903
- StreamingLLM, arXiv:2309.17453 — https://arxiv.org/abs/2309.17453 (attention sinks)
- Attention-sink survey 2026 — https://www.alphaxiv.org ("Attention Sink in Transformers: A Survey on Utilization")
- H2O, arXiv:2306.14048 — https://arxiv.org/abs/2306.14048
- SnapKV, arXiv:2404.14469 — https://arxiv.org/abs/2404.14469
- Robust KV pruning with anchor tokens — https://dl.acm.org
- SelfExtend / PoSE context-extension overview — https://zilliz.com (context-engineering survey, Jan 2026) and PoSE ICLR
- Prefix tuning, arXiv:2101.12090 — https://arxiv.org/abs/2101.12090
- Prompt tuning (Lester), arXiv:2104.08691 — https://arxiv.org/abs/2104.08691
- P-tuning v2, arXiv:2110.07677 — https://arxiv.org/abs/2110.07677
- Function vectors, arXiv:2310.15213 — https://arxiv.org/abs/2310.15213 (project: https://functions.baulab.info)
- Task vectors, arXiv:2310.15916 — https://arxiv.org/abs/2310.15916
- Pretrained Transformers as Universal Computation Engines, arXiv:2103.05247 — https://arxiv.org/abs/2103.05247 (verified; the "even more universal + position-embedding programming" paper could NOT be verified on arXiv — treat as hallucinated until found)

Claim 2:
- Scheduled sampling, arXiv:1506.03099 — https://arxiv.org/abs/1506.03099
- Ranzato et al., arXiv:1511.06732 — https://arxiv.org/abs/1511.06732
- Arora et al., exposure bias as imitation learning, arXiv:2110.05978, AAAI 2022 — https://arxiv.org/abs/2110.05978
- Holtzman et al., arXiv:1904.09751 — https://arxiv.org/abs/1904.09751
- Welleck et al., arXiv:1908.04319 — https://arxiv.org/abs/1908.04319
- Scissorhands, arXiv:2305.17118 — https://arxiv.org/abs/2305.17118
- KIVI, arXiv:2402.02750 — https://arxiv.org/abs/2402.02750
- SCBench (KV-cache-centric benchmark) — https://openreview.net
- KVzip (2025) — via scribd/openreview listings
- SinkProbe, arXiv:2604.10697 — https://arxiv.org/abs/2604.10697 (ICML 2026)
- Zhang, Press et al., snowballing, arXiv:2305.13534 — https://arxiv.org/abs/2305.13534 (ICML 2024: https://proceedings.mlr.press/v235/zhang24ay.html)
- Farquhar et al., Nature 2024 — https://www.nature.com/articles/s41586-024-07421-0 (arXiv:2406.15927)
- Model collapse (Shumailov et al., Nature 2024) — nature.com; self-consuming loops (Alemohammad et al.)
- Agent memory/context poisoning (2026) — dev.to/Redis/Knostic practitioner refs; sleeper-memory poisoning arXiv May 2026
