Distillation was prematurely deprioritized.

`g160` is not evidence against distillation as transfer. It tests a different claim: whether transport theory selects the better student architecture under a fixed KD setup, not whether KD transfers capability at all ([g160 spec](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/programs/post_g156_pass_program.md:86>), [g160 code](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_160_transport_guided_student.py:23>)). The code explicitly downscoped to `SEEDS=[42]` and says any verdict is PILOT-only, with canonical 3-seed requiring a separate prereg ([g160 code](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_160_transport_guided_student.py:51>)). The artifact makes it weaker still: `CtQ_90` was non-discriminative because both arms hit 90% of own-final by the first 1000-step checkpoint; the pilot is basically a single-seed `C3` comparison with `-0.34pp` gap ([g160 result](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/results/genome_160_transport_guided_student.json:209>)). That can kill R8 as a design-rule pilot. It does not kill KD.

`g154` already is a transfer experiment: random-init student, scratch control, KD treatment, and KD wins `+0.586pp` top-1 and `+0.058` NLL ([g154 code](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_154_distillation_smoke.py:20>), [g154 result](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/results/genome_154_distillation_smoke.json:32>)). The reason it was not made canonical is also real: the script itself labels it a smoke test, single seed, `4096` train windows, `200` C4 eval windows, and a tokenizer-compatibility workaround. So no, `g154` should not already be the canonical proof. But yes, the next move should have been “scale `g154` with proper protocol,” not “downgrade distillation because `g160` failed.”

**g167**
- Question: does top-k logit KD transfer teacher next-token capability to a random-init recipient at canonical multi-seed scale?
- Teacher: `Qwen/Qwen3-0.6B`.
- Student: same `minimal_3L` family as `g154`, random-init each seed.
- Arms: `scratch CE` vs `CE + top-k=64 KD`, `T=2.0`, `gamma=0.5`.
- Data: `8192` train windows from deduped C4 train; eval on deduped C4 val plus Wikitext-103 val, `1000` windows each, len `256`.
- Seeds: `{42, 7, 13}`.
- Train: `6000` steps, batch `8`.
- PASS: mean `KD - scratch` top-1 on C4-val `>= +0.40pp`, paired 95% CI excludes zero, mean NLL gain `>= +0.03` on both C4-val and Wikitext-val, and positive primary effect in at least `2/3` seeds.
- Compute: about `3.0-3.5 hr`, `<22 GB` VRAM. This is consistent with observed `g154` per-arm timings and the fact that `g154/g160` already fit this teacher/student regime.
- Score on PASS: `7/10`.

Brutally honest version: elevate distillation as the primary engineering axis for §0 now. Do not elevate `g160`’s transport-guided student-selection rule on the basis of current evidence. If `g167` fails, then the cheap-transfer story is in trouble in your actual budget regime, because even the literature’s strongest baseline mechanism would not be cashing out here.