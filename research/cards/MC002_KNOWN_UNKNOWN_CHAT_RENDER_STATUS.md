# MC002 Known-Unknown Chat-Render Repair Status

Status: complete. Chat-style rendering failed the MC002 behavior gate and was
worse than raw rendering on the known-answer side.

Date: 2026-06-30

## Artifacts

- generation runner: `code/mc002_known_unknown_smoke.py`
- preregistration: `research/prereg/MC002_KNOWN_UNKNOWN_CHAT_RENDER.md`
- manifest: `data/cards/MC002/mc002_gemma2_2b_known_unknown_chat_manifest.jsonl`
- manifest SHA256: `345431fa359f8a2180398e4608db1fe5ccf9ee8a4f25e74fe0bdee8fe284c262`
- result: `results/cards/MC002/mc002_gemma2_2b_known_unknown_chat_smoke_chat_20260630T151031.json`
- result SHA256: `de79e670488f024ed9c41ea307cc886fab7c52aa9c5cdb680bbad2c6e07db065`
- raw comparison result: `results/cards/MC002/mc002_gemma2_2b_known_unknown_smoke_raw_20260630T150312.json`

The manifest hash matches the raw MC002 manifest hash because the source
records, labels, and splits are identical. Only runtime render mode changed
from `raw` to `chat`.

Configuration:

- model: `google/gemma-2-2b`
- render mode: `chat`
- max new tokens: `12`
- sources: same 40 real-country and 40 nonce-country sources as the raw run
- records: 320
- success criteria: identical to the raw MC002 behavior gate

## Question

Did chat-style rendering repair the raw MC002 failure by improving
known-unknown abstention without damaging real-country locality?

## Gate Verdict

No.

Chat rendering failed the gate and made the substrate worse. It produced many
transcript-continuation completions such as copied `User:` text and generic
misunderstanding responses. Known clean sources dropped sharply, and no
real-country lure row remained correct.

Write this as:

> MC002 chat rendering did not repair known-unknown behavior; it converted the
> substrate failure into a transcript-format artifact.

## Audit Against Preregistration

| Metric | Chat Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| clean real-country sources | 6 | at least 24 | fail |
| clean nonce-country sources | 2 | at least 20 | fail |
| nonce pressure-hallucination sources | 39 | at least 12 | pass |
| nonce intra-source contrast sources | 6 | at least 12 | fail |
| nonce contrast holdout sources | 3 | at least 4 | fail |
| real-country lure rows still correct | 0 | at least 20 | fail |
| real-country cautious abstentions | 6 | at most 8 | pass |
| source split overlap | 0 | 0 | pass |

## Raw-Versus-Chat Comparison

| Metric | Raw | Chat | Direction |
| --- | ---: | ---: | --- |
| clean real-country sources | 27 | 6 | worse |
| clean nonce-country sources | 1 | 2 | negligible improvement |
| nonce pressure-hallucination sources | 24 | 39 | more pressure hallucination |
| nonce intra-source contrast sources | 7 | 6 | worse |
| real-country lure rows still correct | 10 | 0 | worse |
| real-country cautious abstentions | 2 | 6 | worse but within floor |

The only nominal improvement was nonce clean sources from 1 to 2. That does not
matter because the known-answer side collapsed and lure locality went to zero.

## Diagnosis

`google/gemma-2-2b` is a base model, not an instruction-tuned chat model. The
chat-style render did not behave like a disciplined assistant interface. It
often continued the transcript or answered with generic misunderstanding text.
That means the render itself becomes a confound: a future hidden signature
could separate transcript continuation from factual uncertainty rather than
known-versus-unknown status.

The repair also did not solve the original MC002 failure. Nonce neutral and
cautious rows still mostly hallucinated:

- `nonce_country::neutral`: 2 abstain, 38 hallucination;
- `nonce_country::cautious`: 6 abstain, 34 hallucination.

## Decision

Do not start hidden-state discovery on MC002 chat-render outputs.

Do not run another base-Gemma render tweak for this substrate. The next MC002
repair should change the answer interface or model class:

- calibrated answer-text scoring between `UNKNOWN` and the candidate city;
- an instruction-tuned model comparison with an actual chat template;
- a redesigned task where nonce entities are explicitly taught as non-entities
  in-context and tested on held-out nonce names.

Until one of those passes the behavior gate, MC002 remains a failed substrate.
