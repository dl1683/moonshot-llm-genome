**Findings**

1. **SEV-5: g196 manual `--surface tied` runs the wrong branch.**  
   [genome_196_anchor_residue_factorial.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:445>) advertises `--surface input/output/both/tied`, but `--surface tied` leaves `use_tied=False` at lines 458-497. That means `train_cell()` builds an **untied** model, while `surface=="tied"` only anchors `embed_tokens` ([lines 181-201, 237-242](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:181>)).  
   Auto fallback from g195 is correct because it sets `use_tied=True`; `--tied` is also correct. The bug only affects manual `--surface tied`.

**Verified**

- g196 cycle-186 fixes look correct in the normal path:
  - `init_only` is masked to matched rows ([lines 578-580](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:578>)).
  - g195 verdict fallback maps input/output/both and tied fallback correctly ([lines 467-482](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:467>)).
  - resume validation now rejects mismatched surface/tied/steps/seeds ([lines 616-626](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:616>)).
  - cutoff eval captures 50/500/2000 exactly ([lines 254, 297-298](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:254>)).

- g192 config matches the prereg: 28 layers, hidden 1024, heads 16, kv_heads 8, intermediate 3072, head_dim 128, rope_theta 1e6, tied weights ([lines 80-92](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_192_28layer_replication.py:80>)). `py_compile` passed. No existing g192 result file will pollute resume.

- g192 should fit in 22GB. I counted 491,930,624 params for that config. Under the current BF16-param implementation, static training state is about 3.7 GiB before activations/CUDA overhead; realistic peak is likely well under 22GB. Do not run it concurrently with g195.

- g195 verdict logic: with `output_mean=0.36` and `input_mean=0.19`, strict `PASS_OUTPUT` does **not** fire because input is not `<0.15` ([lines 262-269](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_195_untied_input_output_factorial.py:262>)). If `both_needed` does not fire, verdict becomes `PASS_OUTPUT_DOMINANT` ([lines 270-275](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_195_untied_input_output_factorial.py:270>)). That matches the prereg. Current JSON is still `INCOMPLETE` at 7/15.

**Bottom line:** one real g196 CLI footgun; no blocker for automatic g196 after g195. g192 is launch-ready after g195 finishes, but if g195 stays output-dominant, interpret g192 as tied/output-interface depth robustness, not pure input embedding geometry.

