I would run g190, but with one correction: if Phase 1 freezes the real trained Qwen3-0.6B decoder, Phase 2 should use the real Qwen3-0.6B config with GPT-2 vocab too. The current g188 code uses an 8-layer Qwen-like shell, not the full 28-layer Qwen3-0.6B config, so using it makes the result a useful pilot but a weaker decoder-interface test.

**Recommended Design**

Phase 1: train one GPT-2-vocab embed/head against frozen trained Qwen3 decoder.

- Freeze: all transformer blocks, final norm, RoPE/config behavior. Train only `model.embed_tokens.weight` and `lm_head.weight`.
- Use tied embed/head, matching Qwen3/g188 `tie_word_embeddings=True`.
- GPT-2 vocab shape: `len(tok_gpt2) x 1024`, about `50k x 1024`.
- Init randomly, scaled to trained Qwen embedding row-norm/Frobenius statistics. No OT, no PPMI, no string map.
- Train on GPT-2-tokenized C4 with same 13-gram train/val dedup discipline.

**1. Phase 1 Steps**

Use `2000` Phase 1 steps as the locked primary target, with checkpoints at `250, 500, 1000, 1500, 2000`.

Convergence criterion should be Phase-1-only:

- validation NLL improves by `<0.03 nats` over the last 500 steps, or
- normalized embedding movement `||E_t - E_{t-500}||_F / ||E_t||_F < 0.01`.

If not converged by `2000`, still use step `2000` as the preregistered target and mark `phase1_converged=false`. Do not choose a Phase 1 checkpoint based on Phase 2.

**2. Freeze All Or Most?**

Freeze all decoder params. Do not unfreeze RMSNorm/final norm in the main experiment.

Reason: the hypothesis is that embeddings learn the decoder’s existing interface format. If norms or blocks move, the interface is adapting to the embeddings, which muddies the result.

**3. Vocab Mismatch**

There is no target mismatch in the main path: Phase 1 creates GPT-2-vocab embeddings directly.

Use Qwen native embeddings only as a diagnostic string-overlap control, not a main control. Exact token overlap will be sparse and frequency-biased.

**4. Include `lm_head`?**

Yes, mandatory. New tokenizer adaptation is both input codebook and output classifier. With tied weights this is one matrix, but gradients come from both input and output usage.

An embed-only Phase 1 is not a serious main arm.

**5. Phase 2 PASS Criterion**

Primary arms:

1. `scratch_ce`
2. `relearned_embed_init_anchor`
3. `relearned_embed_anchor_only`
4. `flow_bridge_init_anchor` rerun under the exact same Phase 2 architecture/protocol

Use paired seed gaps:

`gain = scratch_final_nll - arm_final_nll`

PASS:

- `relearned_embed_init_anchor` mean gain `>= +0.15 nats`
- 3/3 seeds positive vs scratch
- beats `flow_bridge_init_anchor` by `>= +0.20 nats`
- `relearned_embed_anchor_only` mean gain `>= +0.05 nats` and does not harm

STRONG PASS:

- init+anchor gain `>= +0.257 nats`, i.e. at least 50% of g181b’s `+0.513`
- anchor-only gain `>= +0.15 nats`

FAIL:

- relearned init+anchor `< +0.05 nats`, or
- not better than flow bridge by at least `+0.05`, or
- anchor-only harms.

I would add `relearned_row_shuffle_anchor` as a gated Stage B adversarial control if Stage A passes.

**6. Wall-Clock**

Full-Qwen clean version:

- Phase 1: roughly 10-20 min for 2000 frozen-decoder steps.
- Phase 2 Stage A: 3 arms x 3 seeds x 5000 steps, about 3.2-3.8 h.
- Anchor-only Stage B: another ~1.1-1.3 h if run separately.

So: Stage A fits the envelope if staged carefully; all controls in one run probably exceeds the 4 h target.

If using g188’s reduced 8-layer shell, all main arms likely fit in ~2 h, but interpretation is weaker.

**7. §0.1 Movement**

Moderate but real.

A strong pass would move the project from “tokenizer-prior artifact” toward “decoder-conditioned codebooks are transferable training priors.” I’d score that as roughly `+0.8 to +1.2` §0.1 movement, maybe taking the current `4.0-4.5/10` to `5.0-5.7/10`.

It does not get to `7+` unless it later produces compute savings, downstream capability gains, or cross-architecture portability.

**8. Interpretation**

PASS means static lexical/corpus geometry is insufficient, but decoder-conditioned geometry can make a new tokenizer usable as an anchor target. That supports the codebook+decoder thesis: embeddings are not just semantic rows; they are an interface compiled against decoder dynamics.

FAIL means one of two things:

- decoder-conditioned embeddings are specific to the frozen trained decoder and do not transfer to a fresh decoder, or
- g181b’s effect is native-tokenizer-specific and cannot be ported through GPT-2 tokenization.

Most informative split:

- Phase 1 succeeds, Phase 2 fails: adaptation works, but not as a transferable anchor.
- init+anchor passes, anchor-only fails: useful warm start, weak anchor thesis.
- anchor-only passes: strongest evidence for decoder-conditioned embedding targets as basin-shaping objects.