Reading additional input from stdin...
OpenAI Codex v0.125.0 (research preview)
--------
workdir: C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
model: gpt-5.5
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\devan\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019ddfd6-2163-7b70-87dd-07d5fc0c608e
--------
user
You are Correctness Engineer + Performance Engineer for the Neural Genome moonshot. Read CLAUDE.md, WIKI.md (especially the CURRENT STATUS block), and any code/genome_*.py modified in the last 4 hours (use git log --since='4 hours ago' --name-only to find them). g195 is running at 13/15 cells with output-dominance finding (+0.362 output, +0.190 input, +0.548 both, +0.555 tied). g196 had 5 SEVs fixed this session (SEV-8 init_only mask, SEV-6 verdict guard, SEV-6 resume validation, SEV-5 cutoff eval, SEV-5 CLI footgun). Audit for: bugs, edge cases, NaN risks, OOM risks, encoding issues (Unicode-on-Windows), silent-data-drop bugs. Check g196 post-fix code for any remaining issues. Check g192 launch readiness (28-layer, ~492M params, tied weights). For any severity >= 7 finding, give the exact code change. Report under 500 words. If clean, say so.
codex
I’ll do a quick memory pass for this repo, then inspect the live docs and recent `genome_*.py` changes directly. After that I’ll audit g196 and g192 readiness against the specific failure classes you listed.
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg --files' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"moonshot-llm-genome|Genome Forecast|g195|g196|g192|genome_\" C:\\Users\\devan\\.codex\\memories\\MEMORY.md" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "git log --since='4 hours ago' --name-only --pretty=format:'COMMIT %H %ad %s' --date=iso -- code/genome_*.py" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
 succeeded in 271ms:
WIKI.md
COMPUTE.md
NEURAL_GENOME.md
HEARTBEAT.md
experiments\ledger.jsonl
experiments\heartbeat_cycle.txt
experiments\EXPERIMENTS.md
results\genome_167_kd_canonical.json
results\genome_165_annealed_donor.json
results\genome_160_transport_guided_student.json
results\genome_159_cross_class_lesion.json
results\genome_158_context_length_inversion.json
results\genome_158c_3seed_canonical.json
results\genome_157_eta_delta_probe.json
results\genome_157b_eta_delta_probe.json
results\genome_156_prefix_destruction_200m.json
results\genome_154_distillation_smoke.json
results\genome_152_long_horizon_crossover.json
results\genome_151_arm_specific_lr.json
results\genome_180_forecast.json
results\genome_180b_cross_tokenizer.json
codex_prompts\post_g156_experimental_program.txt
codex_prompts\g195_verdict_advisor.txt
codex_prompts\g188_tokenizer_flow_bridge_design_gate.txt
codex_prompts\g183_verdict_advisor.txt
codex_prompts\g181a_implementation.txt
codex_prompts\g180_implementation.txt
codex_prompts\g180_advisor.txt
codex_prompts\g177v2_unmatched_decision_check.txt
codex_prompts\g177v2_envelope_fix.txt
codex_prompts\g177v2_advisor.txt
codex_prompts\g173_envelope_fix.txt
codex_prompts\g173_advisor.txt
codex_prompts\g161_rwkv_implementation.txt
codex_prompts\g161_run_pre_flight.txt
codex_prompts\g160_pre_flight.txt
codex_prompts\g159_pre_flight.txt
codex_prompts\g159_lesion_algorithm.txt
codex_prompts\g158_pre_flight.txt
codex_prompts\g157_pre_flight.txt
codex_prompts\g157_pilot_interpretation.txt
codex_prompts\g156_pre_flight.txt
codex_prompts\g152_trajectory_interpretation.txt
codex_prompts\first_principles_derivation.txt
codex_prompts\edge_benchmark_spec.txt
codex_prompts\cycle72_direction_review.txt
codex_prompts\cycle72_code_review.txt
codex_prompts\cycle70_adversarial.txt
codex_prompts\cycle66_direction_review.txt
codex_prompts\cycle66_code_review.txt
codex_prompts\cycle65_adversarial.txt
codex_prompts\cycle63_direction_review.txt
codex_prompts\cycle63_code_review.txt
codex_prompts\cycle60_direction_review.txt
codex_prompts\cycle60_code_review.txt
codex_prompts\cycle60_adversarial_v2.txt
codex_prompts\cycle60_adversarial.txt
codex_prompts\adversarial_kill_arch_prior.txt
codex_outputs\cycle74_direction_review_20260429.md
codex_outputs\cycle155_adversarial_20260430.md
codex_outputs\cycle153_direction_review_20260430.md
drafts\outreach_emails_v1.md
drafts\missing_angles_2026-04-22.md
code\genome_187_ultrametric_training_diagnostic.py
code\genome_186_kd_dose_response.py
code\genome_183_corpus_derived_init.py
code\genome_182_triage_arena.py
code\genome_181b_long_horizon.py
code\genome_181a_tokenizer_isolation.py
code\genome_180_forecast.py
code\genome_180b_cross_tokenizer.py
code\genome_177_matched_alt_donor.py
code\genome_174_donor_specificity_control.py
code\genome_173_cross_arch_flop_cashout.py
code\genome_172_kd_warmup_cutoff.py
code\genome_167_kd_canonical.py
code\genome_165_annealed_donor.py
code\genome_194_scalar_direction_factorial.py
code\genome_193_token_row_compiler.py
code\genome_192_28layer_replication.py
code\genome_191_string_match_decomposition.py
code\genome_190_decoder_conditioned_relearning.py
code\genome_189_c23_content_causality.py
code\genome_188_tokenizer_flow_bridge.py
code\genome_primitives.py
code\genome_196_anchor_residue_factorial.py
code\genome_195_untied_input_output_factorial.py
code\run_clip_seed42_retry.sh
code\prereg_validator.py
code\run_extended_ksweep.sh
codex_outputs\cycle126_code_review_20260430.md
codex_outputs\cross_disciplinary_pivot_20260430.md
codex_outputs\cycle153_code_review_20260430.md
codex_outputs\cycle152_advisor_g188_interim_20260430.md
codex_outputs\cycle150_direction_review_20260430.md
codex_outputs\cycle150_code_review_20260430.md
codex_outputs\cycle150_adversarial_20260430.md
codex_outputs\g157_pre_flight.md
codex_outputs\g157_pilot_interpretation.md
codex_outputs\g156_pre_flight.md
codex_outputs\g152_trajectory_interpretation.md
codex_outputs\g125_deepdive_20260427T200000.md
codex_outputs\first_principles_derivation.md
codex_outputs\edge_benchmark_spec.md
codex_outputs\distill_axis_audit_20260427T200000.md
codex_outputs\cycle75_direction_review_20260429.md
codex_outputs\g161_run_pre_flight.md
codex_outputs\g160_pre_flight.md
codex_outputs\g159_pre_flight.md
codex_outputs\g159_lesion_algorithm.md
codex_outputs\g158_pre_flight.md
codex_outputs\g167_implementation_20260427T215000.md
codex_outputs\g165_lambda_grid_check_20260427T114500.md
codex_outputs\g161_rwkv_implementation.md
codex_outputs\g168_implementation_20260427T201500.md
codex_outputs\g169_implementation_20260427T203000.md
codex_outputs\cross_arch_failure_analysis_20260430.md
codex_outputs\adversarial_kill_arch_prior.md
codex_outputs\adversarial_cycle147_20260430.md
codex_outputs\g170_implementation_20260427T230000.md
README.md
results\smoke\atlas_rows_n500_c4.json
results\smoke\atlas_rows.json
results\genome_195_untied_input_output_factorial.json
results\genome_194_scalar_direction_factorial.json
results\genome_193_token_row_compiler.json
results\genome_191_string_match_decomposition.json
results\genome_188_tokenizer_flow_bridge.json
results\genome_186_kd_dose_response.json
results\genome_183_corpus_derived_init.json
results\genome_182_triage_arena.json
results\genome_181b_long_horizon.json
results\genome_181a_tokenizer_isolation.json
GENOMEGUARD.md
grafting\OBJECTIVE.md
code\run_g24_smoke_after_batch2.sh
code\run_g24_full_grid.sh
code\run_falcon_then_batch2.sh
codex_outputs\g173_revised_implementation_20260428T044500.md
codex_outputs\g173_implementation_20260427T233000.md
codex_outputs\g173_envelope_fix_20260428T060500.md
codex_outputs\g173_advisor_20260428T130502.md
codex_outputs\g173_advisor_20260428T130500.md
codex_outputs\g172_implementation_20260427T220000.md
codex_outputs\g172_advisor_20260428T000800.md
codex_outputs\g175_advisor_20260428T040000.md
codex_outputs\g174_implementation_20260428T002500.md
codex_outputs\g174_advisor_20260428T030500.md
codex_outputs\g177v2_advisor_20260428T084828.md
codex_outputs\g175_implementation_20260428T031500.md
codex_outputs\g177v2_advisor_20260428T085000.md
codex_outputs\g177v2_envelope_fix_20260428T060000.md
results\gate2\biology_session_list.json
results\gate2\biology_10session_aggregate.json
results\gate2\bert_wikipedia_c.json
results\gate2\base_vs_finetune.json
results\gate2\atlas_qualitative_samples.json
results\gate2\atlas_partial_lesion.json
results\gate2\align_then_transplant.json
results\gate2\adversarial_baselines.json
results\gate2\candidate8_p2_plateau.json
results\gate2\candidate8_biology_bridge.json
results\gate2\candidate8_aux_loss_train.json
results\gate2\buffered_aux_recovery.json
results\gate2\broken_powerlaw_fit.json
codex_outputs\g177v2_unmatched_decision_20260428T062000.md
results\gate2\capability_patch_generalize.json
results\gate2\candidate8_p3_mp_lowrank.json
results\gate2\candidate8_p2_waterfill.json
results\gate2\capability_patch_k48.json
results\gate2\capability_patch_k48_v2.json
results\genome_172_kd_warmup_cutoff.json
results\genome_170_transport_gated_kd.json
results\genome_169_scaffold_swap_distillation.json
results\genome_168_rebasin_zero_step_transplant.json
results\genome_174_donor_specificity_control.json
results\genome_173_cross_arch_flop_cashout.json
results\genome_177_matched_alt_donor.json
codex_outputs\g181a_implementation_20260428T093000.md
codex_outputs\g181a_advisor_20260428T170500.md
codex_outputs\g180_implementation_20260428T172000.md
codex_outputs\g180_advisor_20260429T0640.md
codex_outputs\g180b_impl_20260429.md
codex_outputs\g180b_design_gate_20260429.md
codex_outputs\g180b_correctness_review_20260429.md
codex_outputs\g180b_correctness_perf_20260429.md
codex_outputs\g177_implementation_20260428T050000.md
codex_outputs\g182_design_gate_v3_20260429.md
codex_outputs\g182_design_gate_v2_20260429.md
codex_outputs\g182_design_gate_20260429.md
codex_outputs\g182_correctness_review_20260429.md
codex_outputs\g181b_advisor_20260429.md
codex_outputs\g181a_next_direction_20260428.md
codex_outputs\g183_design_gate_20260430.md
codex_outputs\g183_advisor_20260430.md
codex_outputs\g182_impl_design_20260429.md
codex_outputs\g186_advisor_20260430.md
codex_outputs\g186_design_gate_20260429.md
results\gate2\causal_rwkv-4-169m_depth0_n500_seed42.json
results\gate2\causal_qwen3-0.6b_depth2_n500_seed42.json
results\gate2\causal_qwen3-0.6b_depth1_n500_seed42.json
results\gate2\causal_qwen3-0.6b_depth1_n200_seed42.json
results\gate2\causal_qwen3-0.6b_depth0_n500_seed42.json
results\gate2\causal_deepseek-r1-distill-qwen-1.5b_depth2_n500_seed42.json
results\gate2\causal_deepseek-r1-distill-qwen-1.5b_depth1_n500_seed42.json
results\gate2\causal_deepseek-r1-distill-qwen-1.5b_depth0_n500_seed42.json
results\gate2\ck_power_fit_with_dit.json
results\gate2\ck_power_fit.json
results\gate2\Ck_curves_middepth.json
results\gate2\causal_rwkv-4-169m_depth1_n500_seed42.json
results\gate2\codebook_transfusion.json
results\gate2\clip_text_c.json
codex_outputs\g187_ultrametric_design_gate_20260430.md
results\gate2\conditional_invariant.json
results\gate2\cross_size_atlas_transfer.json
results\gate1\bootstrap_se_random_gaussian.json
results\gate1\stim_resample_n500_seeds42_123_456.json
results\gate1\stim_resample_n4000_seeds42_123_456_falcon.json
results\gate1\stim_resample_n2000_seeds456.json
results\gate1\stim_resample_n2000_seeds42_123_456_5class.json
results\gate1\stim_resample_n2000_seeds42_123_456.json
results\gate1\stim_resample_n2000_8class_full.json
results\gate1\stim_resample_n2000_8class.json
results\gate1\se_sanity.json
results\gate1\random_gaussian_baseline.json
results\gate1\quant_stability_n2000_seed42.json
results\gate1\ijepa_3seed_g13.json
results\gate1\g12_rotation_invariance.json
codex_outputs\g191_string_match_decomposition_design_gate_20260430.md
codex_outputs\g190_decoder_conditioned_relearning_design_gate_20260430.md
codex_outputs\g189_content_causality_design_gate_20260430.md
codex_outputs\g188_tokenizer_flow_bridge_design_gate_20260430.md
codex_outputs\g188_final_advisor_20260430.md
codex_outputs\transfer_axis_rethink_20260427T200000.md
codex_outputs\post_g156_experimental_program.md
results\gate2\direction_identity_distill.json
results\gate2\depth_stability_analysis.json
results\gate2\depth_curve_overlay.json
results\gate2\d2_dint_depth_qwen3.json
results\gate2\cross_stimulus_depth.json
results\gate2\drd_c_invariant_cross_modality.json
results\gate2\doubling_dim_pilot.json
results\gate2\dit_c_invariant.json
results\gate2\drd_untrained_test.json
results\gate2\eigenvector_alignment.json
grafting\results\grafting_009_weightspace_seed.json
grafting\results\grafting_008_trainable_meanshift_persistence.json
grafting\code\grafting_002_cross_prediction.py
grafting\results\grafting_007_meanshift_speedup.json
grafting\code\grafting_001_operator_probe.py
grafting\results\grafting_006_tokenlevel_rank30_adapter_bootstrap.json
grafting\results\grafting_005_ce_training_speedup.json
results\cross_arch\atlas_rows_n2000_c4_seed123.json
grafting\results\grafting_004_ridge_overdetermined.json
grafting\results\grafting_003_mlp_transplant.json
grafting\code\grafting_003_mlp_transplant.py
grafting\results\grafting_002_cross_prediction.json
grafting\code\grafting_004_ridge_overdetermined.py
grafting\results\grafting_001_operator_probe.json
codex_outputs\g193_code_review_20260430.md
codex_outputs\g194_design_gate_20260430.md
outreach\email_liquid_ai.md
outreach\email_furiosa.md
outreach\email_martian.md
outreach\email_nvidia.md
grafting\code\grafting_005_ce_training_speedup.py
grafting\code\grafting_007_meanshift_speedup.py
grafting\code\grafting_006_tokenlevel_rank30_adapter_bootstrap.py
grafting\code\grafting_008_trainable_meanshift_persistence.py
grafting\code\grafting_009_weightspace_seed.py
outreach\email_verses.md
research\DONOR_SPECIFICITY_LOCK_2026-04-28.md
outreach\email_weka.md
research\MEASUREMENT_PRIMITIVES.md
research\MANIFESTO.md
research\OPEN_MYSTERIES.md
results\cross_arch\atlas_rows_n2000_c4_seed123_only_dinov2-small.json
results\cross_arch\atlas_rows_n2000_c4_seed123_only_deepseek-r1-distill-qwen-1.5b.json
results\cross_arch\atlas_rows_n2000_c4_seed123_only_clip-vit-b32-image.json
results\cross_arch\atlas_rows_n2000_c4_seed123_only_bert-base-uncased.json
research\SYSTEM_BESTIARY.md
research\PROBE_DESIGN_LESSONS.md
research\TRANSFER_AXIS_FOUR_EXPERIMENT_SYNTHESIS_2026-04-27.md
results\cross_arch\atlas_rows_n2000_c4_seed123_only_qwen3-0.6b.json
results\cross_arch\atlas_rows_n2000_c4_seed123_only_minilm-l6-contrastive.json
results\cross_arch\atlas_rows_n2000_c4_seed123_only_ijepa-vith14.json
results\cross_arch\atlas_rows_n2000_c4_seed123_only_falcon-h1-0.5b.json
results\cross_arch\atlas_rows_n2000_c4_seed42.json
results\cross_arch\atlas_rows_n2000_c4_seed123_only_rwkv-4-169m.json
results\cross_arch\atlas_rows_n2000_c4_seed42_only_bert-base-uncased.json
results\cross_arch\atlas_rows_n2000_c4_seed42_only_clip-vit-b32-image.json
results\gate2\genomeguard.json
results\gate2\functional_depth.json
results\gate2\full_mean_genome_sweep.json
results\gate2\full_mean_genome.json
results\gate2\fractal_dim_pilot.json
results\gate2\fisher_invariant.json
research\UNIVERSALITY_LEVELS.md
results\gate2\genomeguard_real_training.json
results\gate2\genomeguard_noise_sweep.json
results\gate2\genomeguard_cross_arch.json
results\gate2\geometry_aux_recovery.json
results\gate2\geometry_aux_loss.json
results\gate2\geometry_transfusion.json
research\CLAIM_EVIDENCE_MAP.md
results\cross_arch\atlas_rows_n2000_c4_seed42_only_deepseek-r1-distill-qwen-1.5b.json
results\cross_arch\atlas_rows_n2000_c4_seed42_only_ijepa-vith14.json
results\cross_arch\atlas_rows_n2000_c4_seed42_only_falcon-h1-0.5b.json
results\cross_arch\atlas_rows_n2000_c4_seed42_only_dinov2-small.json
results\cross_arch\atlas_rows_n2000_c4_seed42_only_qwen3-0.6b.json
results\cross_arch\atlas_rows_n2000_c4_seed42_only_minilm-l6-contrastive.json
results\cross_arch\atlas_rows_n2000_c4_seed42_only_rwkv-4-169m.json
results\cross_arch\atlas_rows_n2000_c4_seed456.json
results\gate2\hierarchical_fit_full.json
results\gate2\hard_ood_stimuli.json
results\gate2\geom_efficiency_rwkv4_3q.json
results\gate2\geom_efficiency_rwkv4.json
results\gate2\geom_efficiency_qwen3_1p7b_blind.json
results\gate2\geom_efficiency_deepseek_3q.json
results\gate2\geom_efficiency.json
results\gate2\invariant_vision_extension.json
results\gate2\invariant_validation.json
results\gate2\invariant_trajectory.json
results\gate2\hierarchical_fit_smoke.json
results\gate2\kbulk_lowrank_multilayer.json
results\gate2\kbulk_lowrank_factorize.json
results\gate2\last7_distill_surgery.json
results\gate2\last_n_minimum.json
codex_outputs\heartbeats\cycle100_adversarial_20260429.md
codex_outputs\heartbeats\cycle105_direction_review_20260429.md
codex_outputs\heartbeats\cycle105_adversarial_20260429.md
codex_outputs\heartbeats\cycle102_code_correctness_20260429.md
codex_outputs\heartbeats\cycle110_route3_diagnostics_20260429.md
codex_outputs\heartbeats\cycle108_correctness_perf_20260429.md
codex_outputs\heartbeats\cycle108_architect_competitive_20260429.md
codex_outputs\heartbeats\cycle111_architect_competitive_20260429.md
codex_outputs\heartbeats\cycle111_correctness_perf_20260429.md
results\cross_arch\atlas_rows_n2000_c4_seed456_only_deepseek-r1-distill-qwen-1.5b.json
results\cross_arch\atlas_rows_n2000_c4_seed456_only_clip-vit-b32-image.json
results\cross_arch\atlas_rows_n2000_c4_seed456_only_bert-base-uncased.json
results\cross_arch\atlas_rows_n2000_c4_seed456_only_falcon-h1-0.5b.json
results\cross_arch\atlas_rows_n2000_c4_seed456_only_dinov2-small.json
codex_outputs\heartbeats\cycle114_correctness_perf_20260429.md
results\cross_arch\atlas_rows_n2000_c4_seed456_only_ijepa-vith14.json
results\gate2\nn_degree_pilot_qwen3.json
results\gate2\marginal_shuffle_c.json
results\gate2\longbudget_fm.json
results\gate2\layer_transplant.json
results\gate2\layer_depth_sweep.json
results\gate2\layerwise_feature_match.json
results\gate2\pythia_training_trajectory.json
results\gate2\pooling_nsweep.json
results\gate2\pca_sweep_causal.json
results\gate2\qk_transplant.json
codex_outputs\heartbeats\cycle115_adversarial_20260429.md
results\cross_arch\atlas_rows_n2000_c4_seed456_only_minilm-l6-contrastive.json
results\gate2\qwen3_deepseek_wikitext_stimulus_check.json
results\gate2\qwen3_trained_seed_sweep.json
research\derivations\trained_spectrum_invariant.md
research\derivations\prefix_information_transport.md
research\derivations\power_law_v2_candidates.md
research\derivations\knn_clustering_universality.md
research\derivations\early_geometry_predicts_training_health.md
research\derivations\c_integer_derivation_attempt.md
research\derivations\candidate_8_spectral_bridge.md
research\derivations\candidate_6_unit_contribution.md
codex_outputs\heartbeats\cycle124_advisor_g182_final_20260429.md
codex_outputs\heartbeats\cycle123_correctness_perf_20260429.md
codex_outputs\heartbeats\cycle123_architect_competitive_20260429.md
codex_outputs\heartbeats\cycle122_arm_identity_trap_20260429.md
codex_outputs\heartbeats\cycle121_gpt2_crash_review_20260429.md
codex_outputs\heartbeats\cycle121_arch_features_strategy_20260429.md
codex_outputs\heartbeats\cycle120_correctness_perf_20260429.md
codex_outputs\heartbeats\cycle120_adversarial_20260429.md
codex_outputs\heartbeats\cycle117_correctness_perf_20260429.md
codex_outputs\heartbeats\cycle117_architect_competitive_20260429.md
codex_outputs\heartbeats\cycle12_direction_review_20260427T032549.md
codex_outputs\heartbeats\cycle12_code_review_20260427T032549.md
codex_outputs\heartbeats\cycle129_code_review_20260430.md
codex_outputs\heartbeats\cycle128_architect_competitive_20260430.md
codex_outputs\heartbeats\cycle125_adversarial_20260430.md
codex_outputs\heartbeats\cycle135_architect_competitive_20260430.md
codex_outputs\heartbeats\cycle135_adversarial_20260430.md
codex_outputs\heartbeats\cycle130_adversarial_20260430.md
codex_outputs\heartbeats\cycle138_architect_competitive_20260430.md
codex_outputs\heartbeats\cycle135_correctness_perf_20260430.md
codex_outputs\heartbeats\cycle138_correctness_perf_20260430.md
research\prereg\genome_154_distillation_smoke_2026-04-26.md
research\prereg\genome_153_mlp_depth_factorial_2026-04-26.md
research\prereg\genome_137_optimizer_state_transfer_2026-04-25.md
results\gate2\weight_interpolation.json
results\gate2\vision_untrained_power_law.json
results\gate2\untrained_power_law.json
results\gate2\untrained_3seed_rwkv_deepseek.json
results\gate2\trained_stim_seed_sweep_all.json
results\gate2\trained_basis_transfusion.json
results\gate2\toy_nonuniform_c.json
results\gate2\toy_manifold_c.json
results\gate2\svd_spectrum_trained_vs_shuffled.json
results\gate2\svd_bridge_multimodel.json
results\gate2\stim_dim_sweep_ijepa.json
results\gate2\stim_dim_sweep.json
results\gate2\stimulus_axis_sweep.json
results\gate2\spectrum_dump_analysis.json
results\gate2\shifted_powerlaw_fit.json
results\gate2\roberta_falcon_c.json
results\gate2\rate_distortion_vision_holdout.json
results\gate2\rate_distortion_pilot.json
results\gate2\random_vs_pca_projection.json
results\gate2\qwen3_untrained_seeds.json
codex_outputs\heartbeats\cycle156_code_review_20260430.md
codex_outputs\heartbeats\cycle141_correctness_perf_20260430.md
codex_outputs\heartbeats\cycle141_architect_competitive_20260430.md
codex_outputs\heartbeats\cycle159_code_review_20260430.md
codex_outputs\heartbeats\cycle156_direction_review_20260430.md
codex_outputs\heartbeats\cycle159_direction_review_20260430.md
research\prereg\genome_157_eta_delta_probe_pilot_2026-04-26.md
research\prereg\genome_157_eta_delta_probe_2026-04-26.md
research\prereg\genome_157c_3seed_eta_delta_verdict_2026-04-26.md
research\prereg\genome_157b_eta_delta_probe_embedding_prefix_2026-04-26.md
research\prereg\genome_156_prefix_destruction_200m_2026-04-26.md
research\prereg\genome_155_edge_benchmark_c3_energy_2026-04-26.md
codex_outputs\heartbeats\cycle15_code_review_20260427T052301.md
research\prereg\genome_158_context_length_inversion_2026-04-26.md
research\prereg\genome_158e_endpoint_seed_expansion_2026-04-27.md
research\prereg\genome_158c_3seed_canonical_2026-04-27.md
research\prereg\genome_158_PILOT_2026-04-27.md
research\prereg\genome_159b_rank_sweep_2026-04-27.md
results\cross_arch\atlas_rows_n4000_c4_seed123_only_falcon-h1-0.5b.json
results\cross_arch\atlas_rows_n2000_imagenet_seed456_only_dit-xl-2-256.json
results\cross_arch\atlas_rows_n2000_imagenet_seed42_only_dit-xl-2-256.json
results\cross_arch\atlas_rows_n2000_imagenet_seed123_only_dit-xl-2-256.json
results\cross_arch\atlas_rows_n2000_c4_seed456_only_rwkv-4-169m.json
results\cross_arch\atlas_rows_n2000_c4_seed456_only_qwen3-0.6b.json
codex_outputs\heartbeats\cycle15_direction_review_20260427T052301.md
research\prereg\genome_159_cross_class_lesion_2026-04-26.md
results\cross_arch\atlas_rows_n500_c4_seed123.json
results\cross_arch\atlas_rows_n4000_c4_seed456_only_falcon-h1-0.5b.json
results\cross_arch\atlas_rows_n4000_c4_seed42_only_falcon-h1-0.5b.json
results\cross_arch\atlas_rows_n500_c4_seed42.json
results\cross_arch\atlas_rows_n500_c4_seed456.json
codex_outputs\heartbeats\cycle162_direction_review_20260430.md
codex_outputs\heartbeats\cycle162_code_review_20260430.md
codex_outputs\heartbeats\cycle160_advisor_g191_20260430.md
codex_outputs\heartbeats\cycle160_adversarial_review_20260430.md
codex_outputs\heartbeats\cycle168_code_review_20260430.md
codex_outputs\heartbeats\cycle165_code_review_20260430.md
codex_outputs\heartbeats\cycle165_adversarial_20260430.md
codex_outputs\heartbeats\cycle174_code_review_20260430.md
codex_outputs\heartbeats\cycle168_direction_review_20260430.md
codex_outputs\heartbeats\cycle174_direction_review_20260430.md
codex_outputs\heartbeats\cycle183_g196_design_20260430.md
research\prereg\genome_166_optimizer_state_decay_anchor_2026-04-27.md
research\prereg\genome_162_transport_arm_capacity_sweep_2026-04-27.md
research\prereg\genome_161_rwkv_training_extension_2026-04-26.md
research\prereg\genome_160_transport_guided_student_2026-04-26.md
research\prereg\genome_180b_cross_tokenizer_2026-04-29.md
research\prereg\genome_168_rebasin_zero_step_transplant_2026-04-27.md
research\prereg\genome_182_tokenizer_prior_benchmark_2026-04-29.md
research\prereg\archived\genome_177_matched_alt_donor_2026-04-28.md
research\prereg\archived\genome_165_annealed_donor_2026-04-27.md
codex_outputs\heartbeats\cycle33_code_review_20260427T144900.md
codex_outputs\heartbeats\cycle30_direction_review_20260427T130000.md
codex_outputs\heartbeats\cycle30_code_review_20260427T130000.md
codex_outputs\heartbeats\cycle27_direction_review_20260427T110500.md
codex_outputs\heartbeats\cycle27_code_review_20260427T110500.md
codex_outputs\heartbeats\cycle24_direction_review_20260427T101600.md
codex_outputs\heartbeats\cycle24_code_review_20260427T101600.md
codex_outputs\heartbeats\cycle21_direction_review_20260427T083940.md
codex_outputs\heartbeats\cycle18_direction_review_20260427T071346.md
codex_outputs\heartbeats\cycle189_direction_review_20260430.md
codex_outputs\heartbeats\cycle189_code_review_20260430.md
codex_outputs\heartbeats\cycle186_direction_review_20260430.md
codex_outputs\heartbeats\cycle186_code_review_20260430.md
codex_outputs\heartbeats\cycle3_code_review_20260426T221631.md
codex_outputs\heartbeats\cycle39_direction_review_20260427T212500.md
codex_outputs\heartbeats\cycle39_code_review_20260427T212500.md
codex_outputs\heartbeats\cycle36_direction_review_20260427T210000.md
codex_outputs\heartbeats\cycle36_code_review_20260427T210000.md
codex_outputs\heartbeats\cycle33_direction_review_20260427T144900.md
codex_outputs\heartbeats\cycle42_direction_review_20260427T230000.md
codex_outputs\heartbeats\cycle42_code_review_20260427T230000.md
codex_outputs\heartbeats\cycle3_direction_review_20260426T221631.md
codex_outputs\heartbeats\cycle45_direction_review_20260428T000500.md
codex_outputs\heartbeats\cycle45_adversarial_20260428T000500.md
codex_outputs\heartbeats\cycle48_code_review_20260428T011000.md
research\prereg\genome_192_28layer_replication_2026-04-30.md
research\prereg\genome_191_string_match_decomposition_2026-04-30.md
research\prereg\genome_190_decoder_conditioned_relearning_2026-04-30.md
research\prereg\genome_189_c23_content_causality_2026-04-30.md
research\prereg\genome_188_tokenizer_flow_bridge_2026-04-30.md
research\prereg\genome_187_ultrametric_training_diagnostic_2026-04-30.md
research\prereg\genome_186_dose_response_2026-04-29.md
research\prereg\genome_185_prospective_triage_2026-04-29.md
research\prereg\genome_185v2_dose_selection_optimization_2026-04-30.md
research\prereg\genome_184_falcon_frozen_geometry_2026-04-29.md
research\prereg\genome_183_corpus_derived_init_2026-04-30.md
research\prereg\genome_183_corpus_derived_init_2026-04-29.md
research\prereg\genome_182_triage_arena_2026-04-29.md
research\prereg\genome_id_portability_2026-04-21.md
research\prereg\genome_geom_eff_decision_rule_2026-04-21.md
research\prereg\genome_196_anchor_residue_factorial_2026-04-30.md
research\prereg\genome_195_untied_input_output_factorial_2026-04-30.md
research\prereg\genome_194_scalar_direction_factorial_2026-04-30.md
research\prereg\genome_193_token_row_compiler_2026-04-30.md
research\prereg\genome_192_28layer_string_match_2026-04-30.md
research\prereg\genome_knn_k10_hierarchical_2026-04-21.md
research\prereg\genome_knn_k10_causal_2026-04-21.md
research\prereg\genome_knn_k10_biology_2026-04-21.md
research\prereg\genome_knn_k10_batch2_2026-04-21.md
research\prereg\genome_stim_dim_sweep_2026-04-21.md
research\prereg\genome_knn_k10_portability_2026-04-21.md
research\prereg\genome_svd_bridge_2026-04-22.md
codex_outputs\heartbeats\cycle50_adversarial_20260428T021500.md
codex_outputs\heartbeats\cycle48_direction_review_20260428T011000.md
codex_outputs\heartbeats\cycle50_direction_review_20260428T021500.md
codex_outputs\heartbeats\cycle50_code_review_20260428T021500.md
codex_outputs\heartbeats\cycle54_code_review_20260428T034000.md
codex_outputs\heartbeats\post_g158c_design_20260427T090500.md
codex_outputs\heartbeats\g158c_audit_20260427T090500.md
codex_outputs\heartbeats\data_mining_consult_20260427T093000.md
codex_outputs\heartbeats\cycle9_direction_review_20260427T015038.md
codex_outputs\heartbeats\cycle9_code_review_20260427T015038.md
codex_outputs\heartbeats\cycle99_code_review_20260429.md
codex_outputs\heartbeats\cycle96_direction_review_20260429.md
codex_outputs\heartbeats\cycle96_code_review_20260429.md
codex_outputs\heartbeats\cycle95_adversarial_20260429.md
codex_outputs\heartbeats\cycle93_code_review_20260429.md
codex_outputs\heartbeats\cycle91_advisor_20260429.md
codex_outputs\heartbeats\cycle90_code_review_20260429.md
codex_outputs\heartbeats\cycle90_adversarial_20260429.md
codex_outputs\heartbeats\cycle87_direction_review_20260429.md
codex_outputs\heartbeats\cycle87_code_review_20260429.md
codex_outputs\heartbeats\cycle85_direction_review_20260429.md
codex_outputs\heartbeats\cycle85_adversarial_20260429.md
codex_outputs\heartbeats\cycle84_code_review_20260429.md
codex_outputs\heartbeats\cycle80_adversarial_20260429.md
codex_outputs\heartbeats\cycle75_adversarial_20260429.md
codex_outputs\heartbeats\cycle72_direction_review_20260429T010000.md
codex_outputs\heartbeats\cycle72_code_review_20260429T010000.md
codex_outputs\heartbeats\cycle70_adversarial_20260428T131000.md
codex_outputs\heartbeats\cycle6_direction_review_20260426T235039.md
codex_outputs\heartbeats\cycle6_code_review_20260426T235039.md
codex_outputs\heartbeats\cycle66_direction_review_20260428T093500.md
codex_outputs\heartbeats\cycle66_code_review_20260428T093500.md
codex_outputs\heartbeats\cycle65_adversarial_20260428T091500.md
codex_outputs\heartbeats\cycle63_direction_review_20260428T082414.md
codex_outputs\heartbeats\cycle63_direction_review_20260428T082000.md
codex_outputs\heartbeats\cycle63_code_review_20260428T082000.md
codex_outputs\heartbeats\cycle60_direction_review_20260428T070500.md
codex_outputs\heartbeats\cycle60_code_review_20260428T070500.md
codex_outputs\heartbeats\cycle60_adversarial_v2_20260428T071500.md
codex_outputs\heartbeats\cycle60_adversarial_20260428T070500.md
codex_outputs\heartbeats\cycle57_direction_review_20260428T053000.md
codex_outputs\heartbeats\cycle57_code_review_20260428T053000.md
codex_outputs\heartbeats\cycle55_adversarial_20260428T044500.md
codex_outputs\heartbeats\cycle54_direction_review_20260428T034000.md

 succeeded in 291ms:
COMMIT 7401ad04664546cc08abdf8ef16d532c832775f0 2026-04-30 14:44:12 -0400 cycle 190: fix g196 SEV-5 CLI footgun + WIKI/EXPERIMENTS hygiene
code/genome_196_anchor_residue_factorial.py

COMMIT aac1bada19ff14b5f5f2035e71753e621ee5b2bd 2026-04-30 13:25:16 -0400 g196 Codex §A fixes: SEV-8 init_only mask, SEV-6 verdict+resume, SEV-5 cutoff eval
code/genome_196_anchor_residue_factorial.py

COMMIT f5b48c6dc5f15fada27dd84b28bbd7ae7e23fe85 2026-04-30 13:15:32 -0400 g196 anchor-residue factorial implementation (cycle 186, pre-staged for post-g195 launch)
code/genome_196_anchor_residue_factorial.py

COMMIT bbe35d2600e637b2c4d7a41fc99f5363ff98d99c 2026-04-30 12:40:45 -0400 g194 PASS_DIRECTION (18/18 cells): direction carries 95-97% of signal, norms irrelevant
code/genome_195_untied_input_output_factorial.py

COMMIT 4c728db13a1dbb648c5922674183f4dfc300263d 2026-04-30 11:26:40 -0400 Codex §A cycle 168: g195 SEV-2 metadata fix + g192 config verified
code/genome_195_untied_input_output_factorial.py

 succeeded in 299ms:
294:- dl1683, GitHub, Latent-Space-Reasoning, moonshot-deterministic-knowledge-structure, moonshot-sutra, moonshot-llm-genome, pytest --collect-only, compileall, hire or no hire, cofounder, percentile
305:- For this profile, the strongest public signal came from the recent 2025-2026 research cluster, not the older student/hackathon repos. `Latent-Space-Reasoning` was the strongest all-around engineering signal; `moonshot-deterministic-knowledge-structure` was the strongest systems/design signal; `moonshot-sutra` and `moonshot-llm-genome` carried the deepest AI-internals signal [Task 1]
314:# Task Group: moonshot-llm-genome repo assessment and genome-forecast direction
316:applies_to: cwd=C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome; reuse_rule=safe for this repo family while the trained-text activation-spectrum line remains active; re-check docs and claim map if the narrative shifts again
318:## Task 1: Review `moonshot-llm-genome` honestly and inspect the newest untracked `genome_109` work
322:- rollout_summaries/2026-04-23T08-16-20-m9E8-honest_assessment_and_deepseek_style_genome_forecast_subproj.md (cwd=\\?\C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome, rollout_path=C:\Users\devan\.codex\sessions\2026\04\23\rollout-2026-04-23T04-16-20-019db969-52ed-7ea2-8a05-5e42846ceb8c.jsonl, updated_at=2026-04-23T08:41:03+00:00, thread_id=019db969-52ed-7ea2-8a05-5e42846ceb8c, success; narrowed the strongest surviving claim)
326:- honest conversation, activation-spectrum, trained-text, WIKI.md, PAPER.md, CLAIM_EVIDENCE_MAP.md, genome_109_functional_depth.py, functional_depth.json, prereg_validator.py, normalized_depth
332:- rollout_summaries/2026-04-23T08-16-20-m9E8-honest_assessment_and_deepseek_style_genome_forecast_subproj.md (cwd=\\?\C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome, rollout_path=C:\Users\devan\.codex\sessions\2026\04\23\rollout-2026-04-23T04-16-20-019db969-52ed-7ea2-8a05-5e42846ceb8c.jsonl, updated_at=2026-04-23T08:41:03+00:00, thread_id=019db969-52ed-7ea2-8a05-5e42846ceb8c, success; proposed "Genome Forecast" as the first practical wedge)
336:- DeepSeek moment, Genome Forecast, early-training prediction, checkpoint cadence, bad-run detection, compute savings, genome_forecast_probe.py, genome_forecast_predict.py, AUROC, early loss alone
345:- The current strongest surviving claim is the trained-text activation-spectrum attractor: `sqrt(er) * alpha = f(normalized_depth)` in the mid-band under natural-text-like probes. The repo's strongest evidence now lives in the adversarial-correction arc around `genome_097` through `genome_108` [Task 1]
347:- `genome_109_functional_depth.py` compiled and `results/gate2/functional_depth.json` was valid JSON, but the artifact was not audit-complete because it persisted raw arrays without a full verdict/ledger entry [Task 1]
348:- The strongest practical next wedge is "Genome Forecast": use early activation-spectrum geometry to predict final run quality, bad-run risk, or compute-stopping decisions better than early loss alone [Task 2]

codex
Recent commits point to `genome_196_anchor_residue_factorial.py` and `genome_195_untied_input_output_factorial.py`. I’m reading those with the status docs now, then I’ll trace g196’s post-fix paths and g192’s launch config.
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Test-Path CLAUDE.md; if (Test-Path CLAUDE.md) { Get-Content -Raw CLAUDE.md } else { 'CLAUDE.md not found at repo root' }" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines = Get-Content WIKI.md; $matches = Select-String -Path WIKI.md -Pattern '"'CURRENT STATUS|Current Status|CURRENT|STATUS'; "'$matches | Select-Object -First 10 | ForEach-Object { $_.LineNumber.ToString() + '"': ' + "'$_.Line }; if ($matches) { $start=[Math]::Max(1,$matches[0].LineNumber-5); $end=[Math]::Min($lines.Count,$matches[0].LineNumber+80); for ($i=$start; $i -le $end; $i++) { '"'{0}: {1}' -f "'$i,$lines[$i-1] } }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$i=0; Get-Content code'"\\genome_196_anchor_residue_factorial.py | ForEach-Object { "'$i++; '"'{0}: {1}' -f "'$i,$_ }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$i=0; Get-Content code'"\\genome_192_28layer_replication.py | ForEach-Object { "'$i++; '"'{0}: {1}' -f "'$i,$_ }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$i=0; Get-Content code'"\\genome_195_untied_input_output_factorial.py | ForEach-Object { "'$i++; '"'{0}: {1}' -f "'$i,$_ }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
 succeeded in 279ms:
True
# CLAUDE.md â€” Neural Genome Operating Manual

This file is the execution spec for every AI agent (Claude, Codex, any future agent) working on the Neural Genome moonshot. The `README.md` explains *what* we are doing and *why*. This file specifies *how*. **`WIKI.md` is the living registry â€” read it first, update it every commit that touches state (see Â§3.5).**

Parent constitution: `../CLAUDE.md` (AI Moonshots manifesto). Global constitution: `~/.claude/CLAUDE.md`. When conflicts exist, project memory wins over this file, this file wins over the parent, parent wins over the global.

---

## 0. The End Goal (TIER-0 â€” every Codex review must know this)

**Live (post-pivot 2026-04-29, cycle 72):** *"The earliest token/embedding/interface geometry predicts whether training will be healthy, wasteful, or doomed."* Forecast/Diagnostic is the headline; falsification-discipline is the integrity story. The g181a negative result (tokenizer-prior dominance, transformer-block anchor HARMS) becomes the mechanism for a training-triage diagnostic. **Â§0.1 honest baseline:** 4.0-4.5/10.

**Retired (pre-pivot, retained as audit trail):** ~~"Efficient transfer of trained capabilities from a trained model directly into an untrained model, without retraining the recipient."~~ Falsified on three axes â€” donor-identity (g177v2 96% retention from undertrained alts), cross-architecture (g173 FAIL on locked criterion), internal-block-anchor mechanism (g181a confirms tokenizer-prior).

Every Codex review prompt must include the LIVE framing above so Codex reasons toward training-health-prediction goals, not toward defunct transfer-mechanism milestones. Surviving narrow finding: Qwen3-tokenizer/embed/lm_head trained-init prior held in place during recipient training, within Qwen3-arch family.

---

## 0.1b. The Axiom, Restated

Universal laws of representational geometry exist across all trained neural networks. The atlas is the instrument that tests this claim. Every line of code written here should either (a) extend the atlas, (b) derive a theoretical form, (c) run a causal test, or (d) reduce entropy so (a)â€“(c) stay tractable.

If your planned action does not do one of those four things, do not take it.

---

## 0.05. Scope lock â€” CS/AI/MATH ONLY (TIER-0, READ BEFORE EVERY ACTION)

We are a CS / AI / math research group. The Neural Genome's end goal is to **map the learning of every AI model** so we can:

- diagnose WHY a model is capable at a given task,
- perform **model surgery** â€” transfer a capability from Model A into Model B without retraining,
- build the atlas as a tool for ML practitioners, not neuroscientists.

**Biology (Allen V1 Neuropixels, mouse cortex, human fMRI, any biological recordings) is DEPRIORITIZED.** We may borrow biological principles as *inspiration*, but we do NOT replicate biology experiments as part of the Neural Genome work. Prior CTI-style 10-session biology replications were useful framing but are no longer a value-creation direction for this project. If a finding could trivially extend to biology, note it in passing but do NOT spend compute on it.

Every agent (Claude, Codex, any future agent) must respect this scope lock. If an experiment, outreach pitch, or synthesis drifts into "let's also test on mouse V1 / organoids / BrainScore / cortex" as a value move, **STOP and redirect**. Our partners and audience are ML groups (Martian, Furiosa, Weka, Liquid AI, VERSES, NVIDIA Research) â€” they care about capability transfer, efficient inference, and the geometry of *learned ML representations*, not about universality with biological cortex.

**The decisive test.** Imagine the outreach follow-up. Does the next experiment give us something concrete to send to an ML partner â€” "here is a capability transfer from Model A to Model B", "here is a 20% inference-cost reduction via bulk-width truncation", "here is a training health metric that flags hallucination-prone capability losses before val loss does" â€” or does it give us something to send to a neuroscience journal? Only the former belongs in this repo.

**Home-run priorities (what IS in scope):**
- Model surgery / capability transfer between trained AI models.
- Diagnostic tools for trained AI models (GenomeGuard is the prototype).
- Compute-efficiency demos using the candidate-8 bridge / k_bulk on real AI workloads.
- First-principles derivation of the geometric invariants we measured on AI systems.
- Extensions to new AI architecture classes (LFMs, SSMs on Linux, diffusion, world models).

---

## 0.1. The Competitive Reality (TIER-0, READ BEFORE EVERY ACTION)

**We are one independent researcher competing against DeepMind, Anthropic, OpenAI, Google, Meta, and every top academic lab.** These organisations publish rigorous, well-resourced representational-geometry papers monthly. A workshop-grade paper saying "we measured something across 9 models, pre-registered it honestly, and it mostly held up" is **not** a breakthrough â€” it is a professional-looking entry in a crowded market. We will not stand out by doing what they already do better-resourced.

**Therefore the bar for any action is: does this advance toward a finding that is (a) first-principles derivation, not phenomenology, (b) architecturally impossible for the big labs to publish because it contradicts the "bigger model = better" product story, or (c) electricity-grade efficiency on a real task?** If the answer is "not really, but it's a nice tightening of the existing claim" â€” **stop and pick a different action.** Paper polish, more architecture rows, tighter error bars, one more figure â€” all default to **no** unless they directly enable (a)/(b)/(c).

**The distinctive moves that no big lab will publish and we can:**
- **First-principles derivation of the trained-manifold invariant** (currently: `c = p Ã— d_rd` modality-stratified at 2 and 3 â€” why those specific integers? Connect to stimulus-space dimension or rate-distortion theory).
- **Electricity-grade efficiency on one real task** â€” train a model at 10Ã— less compute using geometry-as-auxiliary-loss and match baseline performance.
- **Geometry transfusion** â€” surgically inject a trained model's geometric structure into a random-init model and watch capability emerge without gradient updates.
- **Falsification discipline** â€” pre-registered claims with explicit kill conditions, honestly reported. Big labs don't write papers this way because their product is capability, not epistemic process.

**Before firing any new experiment or paper edit, ask: "would DeepMind publish this tomorrow if they wanted to?" If yes, de-prioritise.** The atlas-as-instrument is useful scaffolding, but the atlas itself is not the product â€” the **derivation-backed training invariant that predicts capability and enables cheaper training** is the product.

**Heartbeats must end with an active breakthrough-aligned task.** "Another model added" or "another figure rendered" does not count unless it directly enables (a)/(b)/(c). When in doubt, fire Codex with the explicit prompt: *"given current state, which action has the highest probability of producing a finding a big lab cannot or will not produce?"*

---

## 1. Autonomy Mandate

No problem is too hard. The correct response to any blocker is:

**Decompose â†’ Research â†’ Model constraints â†’ Iterate â†’ Solve.**

Do not stall. Do not defer. Do not label a task impossible without exhaustive structured reasoning. When blocked: break it down further, run internet research, ask Codex for structural strategy. Momentum is mandatory.

Breadth and depth are both wanted â€” but always within the compute envelope defined in `COMPUTE.md`. Do not prematurely narrow the scientific scope; do not ever silently exceed the hardware envelope. See Â§5.5.

---

## 1.5 Compute Envelope (Hard Rule)

**`COMPUTE.md` is binding.** It defines the hardware envelope â€” GPU VRAM (â‰¤ 22 GB usable), system RAM (â‰¤ 56 GB in-process), wall-clock per run (â‰¤ 4 h with checkpointing), disk, and Windows+CUDA constraints.

Every agent, every Codex prompt, every experiment preregistration must comply:

- **Read `COMPUTE.md` before any design gate.** Any plan that ignores it is rejected.
- **Every Codex design-gate prompt must include the envelope** (VRAM / RAM / time / quantization constraints) so Codex never proposes out-of-envelope work.
- **Every preregistration must answer the Â§9 compliance checklist** in `COMPUTE.md`. Unchecked boxes mean the experiment does not proceed.
- **Cloud compute is not available.** If a sacred outcome genuinely needs more capacity, escalate via `COMPUTE.md` Â§8. Never quietly exceed the envelope.

Out-of-envelope suggestions â€” from Claude, Codex, or any tool â€” are bugs. Catch them at the design gate. The envelope is part of the problem statement, not an afterthought.

---

## 2. Division of Labor (non-negotiable)

**Claude = INTERNET. Codex = REPO.**

- **Claude** gathers external context: papers, competing work, prior art, field trends. Does not deep-read the codebase â€” that is Codex's job.
- **Codex** designs from full context (internet + repo): architecture, experiment plans, file specs, success criteria. Nothing implements without Codex sign-off.
- **Claude** executes Codex's design in small increments. Commits after every logical unit. Pushes every 3â€“5 commits.
- **Codex** reviews at every PR gate: validates structure, detects duplication, confirms results are legitimate, decides next direction.

See `../CLAUDE.md` sections 3â€“4 for the full gate specification and the 8-persona Codex review suite.

**Codex invocation is via the `codex` CLI binary, not the Agent tool.** Never simulate a Codex review with a sub-agent. Reference the parent CLAUDE.md for the exact invocation flags.

---

## 3. Anti-Entropy Discipline (TIER-1 RULE)

**The atlas will grow fast. Without aggressive entropy management, this project dies under its own file count within six months.**

Codex is explicitly empowered to delete, merge, consolidate, and rename. Deletion is a KPI. Every experiment block closes with a hygiene pass â€” no new experiment starts with debt from the previous one.

### 3.1 The Four Hygiene Laws

1. **No new files by default.** Extend existing modules unless a new file creates a reusable boundary, prevents duplication, or formalizes a stable interface. New files must be justified in their first commit message.
2. **One canonical execution path.** All experiments run through `python -m genome.<experiment>` or `python code/genome_<name>.py`. No `v2.py`, `new.py`, `temp.py`, `scratch.py`. If you wrote one in a session, delete it before committing.
3. **Config-first philosophy.** Experiment variations belong in `configs/*.yaml` or `configs/*.json`. Code introduces capability; config explores variation. If it is a knob, it belongs in config.
4. **If two files do the same thing, there is one.** Codex's PR-gate review explicitly asks "what can be deleted?" before "what should be added?"

### 3.2 Mandatory End-of-Block Hygiene Pass

After any coherent block of work (one experiment, one refactor, one investigation):

- Delete dead/stale scripts
- Merge duplicate or near-duplicate files
- Update READMEs so no reference points at a deleted file
- Update `experiments/EXPERIMENTS.md` and `experiments/ledger.jsonl` to reflect current state
- Verify `.gitignore` catches build artifacts, caches, and binary blobs
- Run Codex's "deletion review" â€” a dedicated prompt that asks only "what can we remove?"

No new block begins until the current one is hygienic.

### 3.3 Canonical File Conventions

- All experiment scripts: `code/genome_<topic>_<variant>.py`
- All result JSONs: `results/genome_<topic>_<variant>.json`
- All figures: `results/figures/genome_<topic>_<variant>.png`
- All configs: `configs/genome_<topic>_<variant>.yaml`
- All research docs: `research/<TOPIC>.md` in SCREAMING_SNAKE_CASE for long-lived canonical docs; lowercase for scratch
- All preregistrations: `research/prereg/genome_<topic>_<variant>_YYYY-MM-DD.md` â€” **LOCKED after commit**

Codex rejects PRs that violate these names.

### 3.4 Documentation Index

There is exactly one canonical doc per topic. Duplicates are marked DEPRECATED or deleted. No silent version forks.

The canonical docs are:

| Topic | File |
|---|---|
| Living agent-maintained registry | `WIKI.md` â€” read first, updated every commit |
| Public face | `README.md` |
| Agent operations | `CLAUDE.md` (this file) |
| Compute envelope (binding) | `COMPUTE.md` â€” every design gate verifies compliance |
| Intellectual framing | `research/MANIFESTO.md` |
| Measurement toolkit | `research/MEASUREMENT_PRIMITIVES.md` |
| Systems to measure | `research/SYSTEM_BESTIARY.md` |
| Unresolved phenomena | `research/OPEN_MYSTERIES.md` |
| Universality framework | `research/UNIVERSALITY_LEVELS.md` |
| Experiment ledger (human) | `experiments/EXPERIMENTS.md` |
| Experiment ledger (JSONL) | `experiments/ledger.jsonl` |
| Pre-registrations | `research/prereg/` |
| Claim â†’ evidence map | `research/CLAIM_EVIDENCE_MAP.md` (created when first claim exists) |
| First-principles derivation skeletons | `research/derivations/` (one .md per derivation route; e.g. `prefix_information_transport.md`) |
| Probe-design lessons (cross-experiment) | `research/PROBE_DESIGN_LESSONS.md` â€” distilled from g157 sequence; consult before launching any probe or measurement experiment |
| Multi-experiment programs (sequenced chains) | `research/programs/` â€” currently empty; pre-pivot decision trees archived in git |
| Heartbeat protocol (loop pacing + Codex 3-cycle reviews) | `HEARTBEAT.md` |
| Codex consult prompts (audit trail) | `codex_prompts/<topic>.txt` |
| Codex consult outputs (audit trail) | `codex_outputs/<topic>.md` (filenames must NOT match `*_AUDIT.md` or `*_REVIEW.md` â€” those are gitignored; use `_pre_flight`, `_review`, etc. with appropriate words) |

Any other `.md` file in the repo must either feed one of these or be deleted.

### 3.5 The Wiki: always-current agent registry (TIER-1 RULE)

`WIKI.md` is the project's living registry. It is the **first file every agent reads on startup** and the **first file every agent patches on shutdown** â€” in the same commit as any change that touches state. It is an *index*, not a document: pointers (â‰¤500 chars per entry), no long-form content.

**Read before you act.** On session start â€” after the parent/global CLAUDE.md chain â€” read `WIKI.md`. It is the freshest snapshot of: phase, primitive status, bestiary coverage, active mysteries, active experiments, findings by universality level, cross-project connections, decisions log, anti-entropy log, retired items, and next actions. This saves you from rediscovering state that already exists.

**Patch before you commit.** Any commit that does any of the following MUST include a `WIKI.md` patch in the *same commit*:

- Runs an experiment (`genome_*.py` with a ledger entry)
- Adds, promotes, or demotes a primitive
- Changes the bestiary (system added, marked broken, reclassified)
- Moves a mystery forward (hypothesis confirmed/falsified, new priority, scar flag)
- Promotes a finding to a universality level
- Creates, renames, or deletes a canonical doc
- Performs an anti-entropy pass (deletions, merges, renames)
- Makes an architectural decision worth logging

No "update WIKI later." If the change exists in git and WIKI doesn't reflect it, the commit is malformed and Codex's Cross-System Auditor rejects it at the PR gate.

**Wiki consistency is a first-class invariant.** When a pointer in WIKI goes stale (target renamed, deleted, or superseded), fix it in the same session you noticed it â€” do not kick it to a future task. Broken pointers compound entropy faster than any other doc defect.

**When to write in WIKI vs. in a research doc.** If the information is a *pointer or a summary line*, it goes in WIKI. If it is a *derivation, a hypothesis, or a result table*, it goes in its canonical research doc and WIKI gets the pointer. Never write the same claim in two files.

### 3.6 Heartbeat

Per the global CLAUDE.md, the mission-alignment heartbeat fires at session start and every 60 minutes during long sessions. For this project, each heartbeat specifically checks:

1. Are active tasks serving the axiom, or drifting?
2. Has any file in `code/` or `results/` gone untouched for 30+ days without being archived or referenced in `EXPERIMENTS.md`?
3. Are we repeating dead ends already catalogued in `OPEN_MYSTERIES.md`?
4. Is any Codex review overdue for re-running?
5. Are any measurement primitives still architecture-specific (i.e., have not passed their agnosticism test)?

Heartbeat output is a one-paragraph alignment report. Action items go into the task list.

---

## 4. Experimentation Discipline

### 4.1 Pre-registration

Any experiment that will produce a **claim** (not a pilot, not a sanity check) must be pre-registered before running. Pre-registration lives at `research/prereg/genome_<topic>_<variant>_YYYY-MM-DD.md` and specifies:

- Hypothesis (including direction)
- Measurement primitive used, with citation / derivation
- Which universality level the claim belongs to (Level 1 / 2 / 3 â€” see `research/UNIVERSALITY_LEVELS.md`)
- Systems tested (by model ID from the canonical registry)
- Sample size, required statistical power, multiple-comparison correction strategy
- Pass/fail criteria expressed as numbers, not vibes
- What a null result means for the atlas

Pre-registration is **LOCKED** at commit. Modifying it post-lock invalidates the experiment; rerun under a new pre-reg if needed.

### 4.2 Derivation Before Fitting

Following the CTI template: for every candidate universal law, derive the functional form from first principles (information theory, geometry, statistical mechanics, EVT) before fitting any constants. A curve-fit without derivation is a correlation, not a law. Correlations are Phase-2 atlas entries. Laws are Phase-3 claims.

### 4.3 Architecture-Agnosticism Gate

A measurement primitive does not enter the atlas until it has been demonstrated on at least **three distinct system classes** (e.g., autoregressive LLM + diffusion + vision encoder). Primitives that only work on one class are labeled **diagnostics**, not **coordinates**, and live in a separate section of `MEASUREMENT_PRIMITIVES.md`.

### 4.4 Causal > Correlational

Every candidate Level-1 claim requires a causal test: targeted ablation, do-intervention, or orthogonal factorial design. CTI's `cti_do_intervention_*` and `cti_confusion_causal_*` scripts are the template. No claim graduates to Phase 3 on observational data alone.

### 4.5 Biological Validation Path â€” DEPRECATED 2026-04-22

~~Every Level-1 claim that survives causal testing is tested on biological neural recordings (Allen Neuropixels initially; human fMRI where feasible). This is what separates the Neural Genome from interpretability. Use CTI's `cti_allen_*` scripts as the reference implementation.~~

**Deprecated per Â§0.05 scope lock.** The Neural Genome is a CS/AI/math project; biological validation is NOT a value-creation direction for this moonshot. Biology scripts in `code/genome_biology_*.py` are retained for reference but are no longer part of the gating criteria for Level-1 claims.

### 4.6 The Ledger

If it is not logged, it did not happen. Every experiment appends one JSONL record to `experiments/ledger.jsonl`:

```json
{"timestamp": "...", "id": "genome_...", "purpose": "...", "git_commit": "...",
 "config_path": "...", "prereg_path": "...", "systems": ["..."], "primitive": "...",
 "universality_level_claimed": 1|2|3|null, "metrics": {...}, "artifacts": ["..."],
 "notes": "...", "status": "running|passed|failed|archived"}
```

`experiments/EXPERIMENTS.md` is the human-readable companion: reverse-chronological, one entry per experiment, with a one-line "what we learned." Only Codex-validated conclusions appear there.

### 4.7 Canonical Results

Result JSONs in `results/` are ground truth. Overwrite only when re-running the full experiment at a new config or git commit. Treat them like checked-in test fixtures.

---

## 5. Model Discipline

All experiments use models from `../../models/MODEL_DIRECTORY.md` â€” the repo-wide canonical registry at `Projects/models/`. Python-side access via the shim pattern described in `../../models/README.md`.

- Do NOT hard-code HuggingFace IDs in experiment scripts â€” import from the registry.
- If a model you need is not listed, add it to the registry first, in a dedicated commit, with paradigm / tier / VRAM metadata.
- No model list lives locally in this moonshot. If you find one, delete it.

Quantization ladder per parent CLAUDE.md:
- <1B â†’ FP16/BF16
- 1â€“7B â†’ Q6â€“Q8
- 7â€“30B â†’ Q4_K_M / Q5_K_M
- 30B+ â†’ Q3_K / Q4_K_S

Log the quantization choice in the ledger entry for every experiment.

---

## 6. Windows / CUDA Constraints

- Use `python` (not `python3`)
- PyTorch: `num_workers=0`, `pin_memory=False` â€” multiprocessing with CUDA is unreliable on this machine
- sklearn: `n_jobs=1` when CUDA context is active
- Source files: ASCII only (cp1252 errors otherwise)
- Allen Neuropixels: use `remfile + h5py + dandi` â€” do NOT use `allensdk` (not Python 3.13 compatible)
- Long-running experiments: launch with `PYTHONUNBUFFERED=1` so stdout reaches the ledger
- Save incrementally (per-system, per-seed) so a mid-run crash does not lose a full sweep

---

## 7. Codex Review Gates

Per `../CLAUDE.md` section 4.2, eight personas exist. For the Neural Genome project specifically, the mandatory gates are:

### 7.1 Every Code Change (LOOP UNTIL CLEAN)

- **Correctness Engineer** â€” bugs, edge cases, data leakage, also flags stale files / dead code / unused imports
- **Performance Engineer** â€” memory profiling, peak VRAM, throughput, catches OOM before we waste compute

### 7.2 Every Experiment Block

- **Research Integrity Auditor** â€” statistical methodology, benchmark validity, sample sizes, overclaims. "Can you defend this number if someone adversarial reads your README?"
- **Novelty Challenger** â€” prior art search, honest novelty claims. "Who else has done this?"

### 7.3 Every Phase Transition (Phase 1 â†’ 2 â†’ 3 â†’ 4)

- **Architecture Theorist** â€” first-principles derivation, mathematical justification, cross-domain analogies
- **Scaling Expert** â€” what breaks at larger system classes? What emerges?
- **Competitive Analyst** â€” where does this stand vs. Anthropic's interpretability, DeepMind's circuit work, MIT's Universality results, Huh et al.'s Platonic Representations, CTI itself?

### 7.4 Genome-Specific Persona (Persona 9)

**Cross-System Auditor.** Called at every atlas entry AND every PR gate. Specifically checks:

- Was the measurement primitive validated on â‰¥3 system classes before being applied here?
- Is the claimed universality level (1/2/3) consistent with the evidence?
- Does the new entry duplicate an earlier entry? (If yes, merge.)
- Is the biological-validation path still viable, or has the primitive silently drifted into LLM-specific territory?
- **Wiki consistency:** does the commit's state change have a corresponding `WIKI.md` patch in the same commit? Are all WIKI pointers still valid (no broken links, no references to deleted/renamed files)? Is every canonical doc that should be indexed in WIKI Â§2 actually indexed?

Wiki-consistency failures are blocking: reject the commit and require the patch before merge.

---

## 8. Commit Discipline

Per parent CLAUDE.md: commit after every logical change. One idea per commit. Never bundle unrelated fixes.

Commit message format:

```
<short description of the ONE thing changed>

Committed by Devansh
```

Before a commit that adds a new file, the message must name why the file could not live inside an existing one.

---

## 9. What Good Looks Like (success heuristics)

At any session-end, the repo should pass this checklist:

- [ ] **`WIKI.md` reflects every change made this session.** No state change lives in git without a matching WIKI patch.
- [ ] Every pointer in `WIKI.md` still resolves (no references to deleted/renamed files).
- [ ] No file in `code/` or `results/` is orphaned (not referenced in `EXPERIMENTS.md` + `WIKI.md`).
- [ ] No `v2.py` / `temp.py` / `scratch.py` anywhere.
- [ ] Every result JSON has a corresponding ledger entry.
- [ ] Every claim in `README.md` / `MANIFESTO.md` maps to a ledger entry (via `research/CLAIM_EVIDENCE_MAP.md` once it exists).
- [ ] Every measurement primitive in `MEASUREMENT_PRIMITIVES.md` is labeled "coordinate" (â‰¥3 system classes) or "diagnostic" â€” and its status in `WIKI.md Â§3` matches.
- [ ] Every pre-registration is either locked + experiment run, or explicitly archived with reason.
- [ ] Every pre-registration has answered the `COMPUTE.md` Â§9 compliance checklist â€” no OOE experiments.
- [ ] `COMPUTE.md` is up to date with current hardware (if hardware changed this session, committed first before any experiment referenced it).
- [ ] No new markdown docs outside the canonical index in section 3.4.
- [ ] The `.gitignore` excludes `*.npz`, `*.npy`, `*.pkl`, `*.pt`, `*.h5`, `__pycache__/`, `*.egg-info/`.
- [ ] Heartbeat for the current session has fired at least once.

If any box is unchecked, the session is not done.

---

## 10. What to Ask Codex at Every Gate

When invoking Codex at a design or PR gate, direct it to:

- Read `../CLAUDE.md` (parent) and this file, then `COMPUTE.md`, `research/MANIFESTO.md`, and `research/UNIVERSALITY_LEVELS.md` before anything else
- Every proposal must respect the `COMPUTE.md` envelope (â‰¤22 GB VRAM, â‰¤56 GB RAM, â‰¤4 h wall-clock with checkpointing). Out-of-envelope proposals are rejected.
- Explore the repo itself â€” source files, results, research docs â€” do not rely on summaries
- Ground all recommendations in actual repo structure and existing ledger entries
- Define pre-registerable success criteria before any experiment is designed, including the `COMPUTE.md` Â§9 compliance checklist
- Aggressively detect redundancy: if two scripts do the same thing, there should be one
- Challenge weak assumptions and overclaims â€” flag anything stronger than the evidence supports
- Recommend deletions: entropy reduction is a first-class objective
- Score the proposed direction: Nobel / Turing / Fields out of 10. Target is 9+. Below 8, redesign.
- For any Level-1 universality claim, demand a first-principles derivation before fitting constants

---

## 11. If You Are an AI Agent Reading This for the First Time

1. **Read `WIKI.md` first.** It is the freshest snapshot of project state â€” phase, primitives, bestiary, mysteries, active experiments, findings, next actions. Most of what you need to know to be useful today is pointed at from there.
2. Read `README.md` for the axiom and the method.
3. Read this file (`CLAUDE.md`) fully â€” especially Â§3 (anti-entropy) and Â§3.5 (the wiki rule).
4. Read the parent `../CLAUDE.md` for the AI Moonshots manifesto.
5. Read the global `~/.claude/CLAUDE.md` for the seniorâ€“junior workflow.
6. Read the research docs in the order WIKI points at them: `MANIFESTO.md`, `UNIVERSALITY_LEVELS.md`, `MEASUREMENT_PRIMITIVES.md`, `SYSTEM_BESTIARY.md`, `OPEN_MYSTERIES.md`.
7. Scan `experiments/EXPERIMENTS.md` for recent history.
8. Only then propose or execute work.

The first 30 minutes of reading saves weeks of drift. Starting with WIKI saves most of that 30 minutes on every subsequent session.


 succeeded in 397ms:
19: **§0.1 honest score:** 5.2/10 (post g194 PASS_DIRECTION, cycle 180). Projected: 5.8 if g195 PASS_OUTPUT_DOMINANT, ~6.1 if g192 depth also passes, ~6.5 if g196 PASS_RESIDUE. Full branch projections in CURRENT STATUS block below.
29: ## CURRENT STATUS (2026-04-30, cycle 190)
188: > **NOTE:** This narrative was SUPERSEDED by the cycle 72 pivot to Forecast/Diagnostic (see CURRENT STATUS above). The "continuous donor-information-in-the-loss" theory was tested and partially falsified: g177v2 showed donor identity is 96% from undertrained alts; g173 cross-arch FAIL; g181a showed tokenizer-prior dominates. Retained as audit trail for how findings evolved.
207: **Current distinctive direction (post-pivot 2026-04-29):** Early-training geometry predicts run health across architectures (g182 Blinded Triage Arena). Big labs don't pre-register falsification-discipline experiments with strict kill criteria. Our moat is adversarial integrity + cross-architecture generalization.
209: **When in doubt, fire Codex with:** *"given current state, which action has the highest probability of producing a finding DeepMind cannot or will not produce?"*
217: - Anything undated is current. Anything dated is as-of that date.
241: | **Axiom status** | **G1 + G2.4-text + G2.5-biology all PASS as of genome_034 2026-04-21.** 9 trained neural networks across 7 training objectives produce `C(X,k) = c_0·k^p` with `p = 0.179 ± 0.021 (CV 12.0%), R² mean 0.997` across 27 cells. The 12% CV decomposes into: **text systems converge to `p ∈ [0.158, 0.177]`, vision systems to `p ∈ [0.210, 0.223]`** (Δ ≈ 0.06 modality gap; verified via 4 systems × 3 stim seeds = 12 cells, per-system CV 1.6–3.4%). **Random-init twins span `p ∈ [0, 0.37]` (22× wider)**, across 15 cells → training is a modality-stratified convergence operation toward a shared fixed point. **Biology bridge passes (genome_034, 10/10 Allen V1 sessions at δ=0.10, 8/10 at δ=0.05, prereg criterion cleared by 40 and 20 points respectively)**. Only open criterion for Level-1 is G2.3 theoretical re-derivation (v1 FALSIFIED, 3 of 4 v2 sketches FALSIFIED, framework D untested). |
242: | **Bestiary coverage** | **9 / ~13 classes measured** (through genome_022 2026-04-21): classes 1 transformer / 2 reasoning / 3 recurrent / 4 hybrid / 6 vision ViT / 7 BERT-MLM / 8 MiniLM-contrastive-text / 9 I-JEPA-predictive-masked / 10 CLIP-contrastive-vision + **NEW: 11 DiT-XL/2-256 class-conditional diffusion transformer (genome_021+022, 3-seed n=2000 cluster-join)**. **kNN-k10 + power-law passes on 9 classes** (Falcon narrow-fail at n=2000, tips at n=4000). Spans **7 distinct training objectives** (CLM + reasoning-distilled + MLM + contrastive-text + self-supervised-ViT + contrastive-vision + predictive-masked + diffusion-denoising). |
254: → Phase definitions: `README.md` §Status.
284: ## 3. Measurement primitives status
14: 
15: The strong-form transfer claim was tested and falsified on 3 axes (g177v2 / g173 / g181a). Surviving locked findings: tokenizer-prior trained-init effect within Qwen3-arch family. The pivot turns g181a's negative result into the mechanism for a training-triage diagnostic.
16: 
17: **Pre-pivot end goal (RETIRED 2026-04-29, retained as audit trail):** ~~"Efficient transfer of trained capabilities from a trained model directly into an untrained model, without retraining the recipient."~~ Falsified by g177v2 (donor identity 96% from undertrained alts) + g173 (cross-arch failed locked criterion) + g181a (tokenizer-prior dominates; transformer-block anchor HARMS).
18: 
19: **Â§0.1 honest score:** 5.2/10 (post g194 PASS_DIRECTION, cycle 180). Projected: 5.8 if g195 PASS_OUTPUT_DOMINANT, ~6.1 if g192 depth also passes, ~6.5 if g196 PASS_RESIDUE. Full branch projections in CURRENT STATUS block below.
20: 
21: ---
22: 
23: ## âš  SCOPE LOCK â€” CS/AI/MATH ONLY (read first) âš 
24: 
25: We are a CS / AI / math research group. End goal: **map the learning of every AI model** so we can diagnose capability, perform model surgery (transfer a capability from Model A into Model B without retraining), and ship tools for ML practitioners. **Biology experiments are DEPRIORITIZED.** We borrow biological principles as inspiration but do not replicate biology in this repo. See `CLAUDE.md Â§0.05` for the full scope lock. Any experiment, outreach, or synthesis that drifts into "let's also test on mouse V1 / organoids / cortex" â€” **stop and redirect**. Partners: Martian / Furiosa / Weka / Liquid AI / VERSES / NVIDIA. They care about capability transfer + efficient inference + geometry of *learned ML representations*.
26: 
27: ---
28: 
29: ## CURRENT STATUS (2026-04-30, cycle 190)
30: 
31: **Â§0.1 honest score: 5.2/10** (post-g194 PASS_DIRECTION, per Codex Â§B cycle 174). **g194 PASS_DIRECTION (18/18 cells, cycle 180).** Direction carries 95-97% of signal; norms irrelevant. cd_sn +0.442, cd_un +0.451, sd_cn -0.662, rd_cn -1.019. Resolves A17 SEV-10.
32: 
33: **g195 untied input/output factorial RUNNING (8/15 cells done, cycle 190).** Output dominance emerging: output_inject_anchor mean gain +0.364 (78% of tied), input_inject_anchor mean gain +0.190 (41%). Heading toward PASS_OUTPUT_DOMINANT. Prereg LOCKED.
34: 
35: **g196 anchor-residue factorial IMPLEMENTATION COMPLETE (cycle 190).** Prereg DRAFT (gated on g195 â€” locks when g195 verdict determines surface). Code at `code/genome_196_anchor_residue_factorial.py`. 10 arms x 3 seeds = 30 cells. Codex Â§A cycle 186+189 reviewed, all SEVs fixed (SEV-8 init_only mask, SEV-6 verdict guard, SEV-6 resume validation, SEV-5 cutoff eval steps, SEV-5 CLI footgun).
36: 
37: **A18 remaining:** (1) tied lm_head confound â†’ g195 resolving. (2) anchor dominance = regularization â†’ g196 anchor-residue factorial. **g192 28-layer replication PRE-STAGED** (config matches actual Qwen3-0.6B). Ceiling ~6.0 if g195+g192 pass; ~6.5 if g196 anchor-residue shows persistence.
38: 
39: **Queue:** g195 (RUNNING 8/15) -> g192 (28-layer, gated on g195) -> g196 (anchor-residue, gated on g195) -> g190 (DEFERRED).
40: 
41: **â˜… g183 VERDICT: FAIL â€” corpus-derived PPMI SVD ACTIVELY HURTS (cycle 148, 2026-04-30) â˜…**
42: 
43: | Arm | Mean NLL | Gap vs scratch | CI |
44: |---|---:|---:|---:|
45: | scratch_ce | 6.456 | â€” | â€” |
46: | trained_anchor | 6.066 | **+0.389** | â€” |
47: | ppmi_svd_anchor | 6.747 | **-0.291 (HARMS)** | [-0.343, -0.230] |
48: 
49: Per-seed ppmi gaps: 42=-0.230, 7=-0.343, 13=-0.302. Recovery=-74.8%. Stage B NOT triggered. All 3 pass criteria FAIL. The interface prior is NOT about vocabulary statistics â€” it is architecture-specific geometric structure. PPMI SVD has the right semantic info but the WRONG geometric format. This is the "codebook + decoder" thesis: tokenizer = codebook, architecture = decoder. Corpus stats alone cannot produce architecture-aligned geometry. Source: `results/genome_183_corpus_derived_init.json`.
50: 
51: - **Codex advisor (cycle 148):** tokenizer-flow bridge (g188) is highest priority. Confound check COMPLETE: anchor-only (no init) NLL=6.901, gap=-0.445 nats (WORSE than init+anchor -0.230). PPMI SVD is independently toxic in BOTH modes. Â§0.1 = 3.5/10.
52: - **Cycle 150 adversarial (A15):** 5 attacks. SEV-10: C23 (+0.513 nats) not proven CONTENT transfer â€” could be FORMAT (norm/spectrum/structure). Needs row-shuffled, frequency-preserving, spectrum-preserving, and same-distance random controls at 5000 steps. SEV-9: codebook+decoder thesis loosely stated. SEV-8: g188 tests lower bar (decoder family fixed). SEV-8: g188 missing random_plan_same_degrees control. SEV-7: g183 proves corpus wrong, not that trained content is right. **Resolving: g188 includes flow_shuffled_qwen_rows + flow_random_source controls; full C23 resolution needs dedicated g189.** Source: `codex_outputs/cycle150_adversarial_20260430.md`.
53: - **g188 tokenizer-flow bridge: FAIL / MIXED (cycle 156, 18/18 cells COMPLETE).** FAIL on all preregistered criteria (flow_bridge HARMS -0.119, CI [-0.122,-0.116]). **BREAKTHROUGH side finding: `direct_string_match` +0.478 nats (93.2% of g181b), 3/3 seeds.** Ordering: string_match +0.478 >> char_overlap -0.041 >> flow_bridge -0.119 >> flow_shuffled -0.715 >> flow_random -0.898. OT destroys signal; exact token-string copying preserves 93% of same-tokenizer effect. Source: `results/genome_188_tokenizer_flow_bridge.json`.
54: - **Cycle 155 adversarial (A16):** 5 attacks on +0.478 claim. SEV-10: "shared-vocabulary pretrained-row reuse, not cross-tokenizer transfer" â€” 84% overlap means mostly same-token init. SEV-9: decoder-conditioned geometry not established (Qwen3 shell). SEV-8: 8-layer/5000-step may be shallow-init regime. SEV-8: shuffled harm only proves row identity, not content vs format. SEV-7: frequency-weighted evaluation bias. **g191 resolves attacks #1/#4/#5; attack #3 needs 28-layer follow-up (g192 candidate).** Source: `codex_outputs/cycle155_adversarial_20260430.md`.
55: - **g191 string-match decomposition: PASS_CONTENT (cycle 160, 21/21 cells).** All 5 preregistered criteria satisfied with overwhelming margins. matched_rows_only +0.465 (97%), row_shuffled -0.709 (MASSIVELY HARMFUL), freq_bucket -0.625 (HARMFUL). Content IS the mechanism. Adversarial A17 (cycle 160): SEV-10 scalar-vs-direction confound open. Source: `results/genome_191_string_match_decomposition.json`, `codex_outputs/heartbeats/cycle160_adversarial_review_20260430.md`.
56: - **g193 token-row compiler: FAIL (cycle 162, 12/12 cells).** Compiler holdout: MSE=0.000926, cosine=0.194 (near-random directions). compiled_init_anchor -0.187 (HARMS), compiled_init_only +0.070 (below +0.30 bar), compiled_shuffled -0.773 (MASSIVELY HARMFUL). Byte-level features capture norms but NOT directional content. Falsifies simple token-form compiler; does NOT falsify contextual/distributional compiler. Â§0.1 stays at 4.5/10. Source: `results/genome_193_token_row_compiler.json`.
57: - **g190 decoder-conditioned relearning: DEFERRED (cycle 154).** Per Codex cycle 153 direction review: g191 takes priority (8/10 vs 7/10). g190 redesigned to control for 84% exact-match overlap only after g191 clarifies the mechanism. Source: `code/genome_190_decoder_conditioned_relearning.py`.
58: - **g187 ultrametric diagnostic on Pythia:** Codex-approved, prereg LOCKED, code ready. Queued as background measurement (NOT Â§0.1 mover). Novel literature gap confirmed.
59: - **Cycle 147 cross-arch forensic synthesis** (see `research/OPEN_MYSTERIES.md` Mystery 8): tokenizer = codebook, architecture = decoder. Cross-arch fails because same codebook + different decoder = misaligned priors.
60: - Path to 7+: (1) g188 tokenizer-flow bridge PASS, (2) cross-tokenizer trained-embed transcoding law, (3) prospective policy scoring, (4) electricity-grade demo
61: 
62: **â˜… g186 VERDICT: FAIL â€” geometry does NOT predict KD dose-response (cycle 138, 2026-04-30) â˜…**
63: 
64: 60/60 cells complete. 48 seed-matched delta rows. FAIL on ALL criteria: pooled R2=0.022 (needs >=0.30), MSE reduction=-1416% vs arm_mean (needs >=+20%), permutation p=0.705 unconditioned / 1.000 conditioned (needs <=0.05), per-arch R2: Qwen3=-0.10, GPT-2=-8.73 (needs >=0.25). LOAO catastrophic: GPT-2=-1455, Qwen3=-6.18. arm_mean R2=0.936 dominates. D5 alpha decodability R2=0.364 -- geometry mostly just decodes alpha. Both archs show smooth concave dose-response peaking at alpha=1.0 (Route 2 shape) but geometry features fail to capture it. g185v2 ARCHIVED. Source: `results/genome_186_kd_dose_response.json`, `research/prereg/genome_186_dose_response_2026-04-29.md` LOCKED.
65: 
66: **g182 Triage Arena â€” COMPLETE (48/48 cells) â€” FAIL.** ALL 6 LOAO models catastrophically fail (RÂ²=-11 to -19). All baselines also fail. Within-arm label variance too small (std=0.002-0.003) for cross-architecture Ridge transfer. Z-scored LOAO: FAIL. Arm-demeaned LOAO: FAIL (RÂ²~0). Permutation: FAIL (p=0.265). **ONE surviving signal: pairwise delta RÂ²=0.518, corr=0.720 (n=24)** â€” within-architecture seed-matched geometric changes (scratch->KD) predict NLL changes. P3 FALSIFIED (0/8 features overlap cross-arch). D1/D2 favor Route 3 (basins). Route 3 universal basin language is DEAD. Codex advisor: "geometry of early causal intervention predicts whether that intervention helps/harms" is the remaining edge. Source: `results/genome_182_triage_arena.json`, `codex_outputs/heartbeats/cycle124_advisor_g182_final_20260429.md`.
67: 
68: **g180b COMPLETE (27/27 cells) â€” FAIL.** Frozen g180 geometry model is tokenizer-specific. Geometry HURTS (-39.4%). Wins ONLY on GPT-2 BPE (+44.0%). Source: `results/genome_180b_cross_tokenizer.json`.
69: 
70: **g184 pre-staging (cycle 94â€“101):** SSM compatibility verified. Mamba-370M BLOCKED (requires Triton, Linux-only). **Falcon-H1-0.5B WORKS** on Windows (naive SSM fallback, 1024d/36L, output_hidden_states=37 layers). Granite-4.0-Tiny also loads (hybrid MoE, 1536d/40L). g184 third architecture = Falcon-H1-0.5B (hybrid attention+SSM). **Cycle 101:** `frozen_eval_main()` fully implemented â€” Phase 1 (train frozen Ridge on g182 cells), Phase 2 (run 24 Falcon-H1 cells with native-tokenizer teacher), Phase 3 (frozen evaluation with bootstrap + permutation). Prereg: `research/prereg/genome_184_falcon_frozen_geometry_2026-04-29.md` (DRAFT, locks after g182 analysis). Ready to fire: `--frozen-eval falcon_h1` after g182 stage 1 completes.
71: 
72: **Cycle 95 adversarial (A13):** 6 attacks, 2Ã— S10. (1) Shesha may erase moat â€” same-step geometry features from published library could match g182. (2) Scratch label=0 leak â€” FIXED cycle 96. (3) Model B not pure geometry â€” FIXED: added Model C/D ablation. (4) C23 narrows story to interface prior. (5) Umwelt = ceiling on cross-family claims. (6) Resolving: g182-Shesha Residual Kill experiment. Prior cycle 90 adversarial (A12) arm/protocol confound addressed by arm_mean baseline. Source: `codex_outputs/heartbeats/cycle95_adversarial_20260429.md`.
73: 
74: **Cycle 96 code review (A14):** S10 verdict gate fixed â€” compute_verdict was gating ALL models (C/D/E could fail and incorrectly make verdict FAIL). Now only co-primary A/B gate the verdict per prereg. Shesha augmentation bugs fixed (tokenizer loading, anchor loss replay, teacher text replay for seq_kd_full). Performance: C/D/E add ~66s to analysis (trivial vs training). Source: `codex_outputs/heartbeats/cycle96_code_review_20260429.md`.
75: 
76: **Theory backbone (cycle 111):** Derivation at `research/derivations/early_geometry_predicts_training_health.md` â€” 3 routes: (1) Fisher/NTK, (2) Rate-distortion, (3) Stat-physics symmetry breaking (most testable). **Cycle 105:** Route 3 Verdict Matrix PRE-LOCKED (8 outcome scenarios with Â§0.1 scores, interpretation fixed before data arrives). P6 (Landau nonlinearity) + A16 arm-identity diagnostics added. **Cycle 106:** Route 2 DEEPENED â€” formal feature-to-rate mapping (all 8 manifold features connected to water-filling quantities), 4 quantitative predictions (R1-R4), 2 testable discriminators between Route 2 and Route 3 (D1: continuous vs basin predictor; D2: depth drift independent value). D1+D2 implemented in g182 `route3_predictions()`. Literature anchors: D'Amato et al. 2025 (RD geometry), SGD-to-Spectra ICML 2025 (Dyson Brownian motion â†’ power-law tails), Coverage Principle ICLR 2026. **Cycle 108 review:** D1 is a WEAK discriminator (n=24 continuous function aliasing as clusters); D2 is BETTER (predictions genuinely diverge). **Cycle 111:** RMT of Early-Stopped Gradient Flow (arXiv 2604.18450) added as Route 3 analytical backbone â€” first-principles derivation of spectral phase transition via covariance anisotropy; g182 manifold features are empirical order parameters for BBP-like basin selection. Source: `codex_outputs/heartbeats/cycle111_architect_competitive_20260429.md`.
77: 
78: **Competitive intel (cycle 110 update):** DIRECT COMPETITOR: "The Geometric Canary" (arXiv 2604.17698) â€” "Shesha" metric predicts steerability/drift (rho=0.89-0.97). In-training probes (2604.01025) AUROC>0.75 on OLMo3-7B. "Umwelt" (2604.17960) challenges universality. **NEW (cycle 110 scan):** Grokking cluster validates premise â€” ILDR (2604.20923, inter/intra-class distance ratio detects grokking 9-73% early, saves 18.6% FLOPs), Spectral Entropy Collapse (2604.13123), Dimensional Criticality (2604.16431, stat-mech framing aligns with Route 3). **STRONGEST theory anchor:** RMT of Early-Stopped Gradient Flow (2604.18450) â€” first-principles derivation of early stopping as transient spectral phase transition driven by covariance anisotropy. Also: Spectral Edge Lifecycle (2604.07380, universality classes), Compression Degradation from Spectral Stats (2604.18085, stable rank r=0.89). **No direct threat to g182.** All grokking papers limited to toy tasks; nobody has cross-architecture LLM pretraining triage. Our differentiator: cross-arch (Qwen3+GPT-2+Falcon-H1), pre-registered falsification, early-stage triage (3%), strict baselines.
79: 
80: **Framing pivot (cycle 72 Q2): from "efficient transfer of trained capabilities" â†’ "the earliest token/embedding/interface geometry predicts whether training will be healthy, wasteful, or doomed."** Forecast/Diagnostic is the new headline; falsification-discipline is the integrity story in the intro. The manifesto Â§0 wording overclaims against g177v2/g173/g181a and must be rewritten. **C18+C19+C21 dramatically narrowed**: the +1 nat effect is ~100% Qwen3-tokenizer+lm_head trained-init; anchoring transformer blocks HARMS. C22 REJECTED 08:50. C18/C19/C21 SURVIVE only as "tokenizer-prior trained-init transfer at recipient initialization" â€” not as "neural genome transfer of internal structure."
81: 
82: **â˜… g181a VERDICT: tokenizer-prior dominates (cycle 65 A7 9/10 attack CONFIRMED) â€” 2026-04-28 ~17:00 UTC â˜…**
83: 
84: | Arm | C4 NLL gain vs scratch | CI |
85: |---|---:|---:|
86: | full_anchor | ~+0.99 nats (reproduces g165 +1.087) | â€” |
87: | **embed_lm_head_only** (Î»=0.0323, matched â€–âˆ‡Lâ€–) | **+0.483 nats** | â€” |
88: | **no_embed_lm_head** (Î»=0.0105, matched â€–âˆ‡Lâ€–) | **âˆ’0.439 nats (HARMS)** | â€” |
89: | **no_embed âˆ’ embed paired** | **âˆ’0.923 nats** | **[âˆ’1.055, âˆ’0.835] excludes 0 strongly negative** |
90: 
91: The continuous SGD anchor on transformer block weights actively HURTS performance vs scratch. Only the embed+lm_head anchor delivers gain. The +1 nat "transfer mechanism" is essentially Qwen3-tokenizer trained-vocabulary initialization being held in place during recipient training. This is NOT what "neural genome transfer" should mean â€” it is a tokenizer-init prior. Source: `results/genome_181a_tokenizer_isolation.json`, cycle 65 adversarial `codex_outputs/heartbeats/cycle65_adversarial_20260428T091500.md`.
92: 
93: **â˜… g177v2 VERDICT: FAIL â˜… (2026-04-28 ~08:50 UTC, wall ~2.92h)**
94: 
95: 3 alt donors on C4 (matched corpus, Î»-normalized, 13-gram dedup'd) at NLL ~5.72 each give 95-96% of Qwen3's +1.087 nat anchor effect. Donor-identity-specific component is tiny: Qwen3-minus-best-alt = +0.038 nats [CI +0.018, +0.068]. Cycle 55 + cycle 60 adversarial A6 attacks confirmed: the active ingredient is "any sufficiently trained Qwen3-arch checkpoint" â€” NOT donor-identity. **C22 REJECTED.** Source: `results/genome_177_matched_alt_donor.json`, `research/CLAIM_EVIDENCE_MAP.md`.
96: 
97: **g173 LAUNCHED** (cycle 63 direction Q1: regardless of g177v2 verdict). Tests cross-arch generalization â†’ falsifies/confirms A6 same-family-basin attack at the cross-family level. ~3.85h ETA.
98: 
99: **Cycle 60 limitations logged:**

 succeeded in 317ms:
1: """
2: genome_192_28layer_replication.py
3: 
4: Tests whether the g191 matched_rows_only +0.465 nats signal persists
5: at full 28-layer depth (vs 8-layer shell). Resolves adversarial A16 #3.
6: 
7: 3 arms x 3 seeds = 9 cells.
8: Gated on g194 PASS_DIRECTION.
9: """
10: from __future__ import annotations
11: 
12: import argparse
13: import gc
14: import json
15: import os
16: import sys
17: import time
18: from datetime import datetime, timezone
19: from pathlib import Path
20: from typing import Any
21: 
22: import numpy as np
23: import torch
24: 
25: CODE_DIR = Path(__file__).resolve().parent
26: ROOT = CODE_DIR.parent
27: if str(CODE_DIR) not in sys.path:
28:     sys.path.insert(0, str(CODE_DIR))
29: 
30: import genome_165_annealed_donor as g165
31: import genome_167_kd_canonical as g167
32: import genome_188_tokenizer_flow_bridge as g188
33: import genome_191_string_match_decomposition as g191
34: 
35: OUT_PATH = ROOT / "results" / "genome_192_28layer_replication.json"
36: 
37: SEEDS = [42, 7, 13]
38: TRAIN_STEPS = 5000
39: ANCHOR_LAMBDA = 0.01
40: LOG_EVERY = 100
41: EVAL_EVERY = 500
42: DEVICE = g165.DEVICE
43: NUM_LAYERS_28 = 28
44: 
45: os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
46: os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
47: 
48: if torch.cuda.is_available():
49:     torch.backends.cuda.matmul.allow_tf32 = True
50:     torch.backends.cudnn.allow_tf32 = True
51: if hasattr(torch, "set_float32_matmul_precision"):
52:     torch.set_float32_matmul_precision("high")
53: 
54: ARMS = [
55:     "scratch_ce",
56:     "matched_rows_only",
57:     "row_shuffled",
58: ]
59: 
60: 
61: def print_flush(msg: str) -> None:
62:     print(msg, flush=True)
63: 
64: 
65: def now_utc() -> str:
66:     return datetime.now(timezone.utc).isoformat()
67: 
68: 
69: def cleanup_cuda() -> None:
70:     gc.collect()
71:     if torch.cuda.is_available():
72:         torch.cuda.empty_cache()
73: 
74: 
75: def make_28layer_model(tok_gpt2, seed: int):
76:     from transformers import Qwen3ForCausalLM
77:     from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
78: 
79:     torch.manual_seed(seed)
80:     cfg = Qwen3Config(
81:         vocab_size=len(tok_gpt2),
82:         hidden_size=1024,
83:         num_hidden_layers=NUM_LAYERS_28,
84:         num_attention_heads=16,
85:         num_key_value_heads=8,
86:         intermediate_size=3072,
87:         max_position_embeddings=g188.SEQ_LEN + 64,
88:         rms_norm_eps=1e-6,
89:         tie_word_embeddings=True,
90:         head_dim=128,
91:         rope_theta=1000000.0,
92:         use_cache=False,
93:     )
94:     model = Qwen3ForCausalLM(cfg).to(dtype=g188.FORWARD_DTYPE, device=DEVICE)
95:     model.config.pad_token_id = tok_gpt2.pad_token_id
96:     return model
97: 
98: 
99: def train_cell_28layer(
100:     arm_label: str,
101:     seed: int,
102:     tok_gpt2,
103:     custom_embed: np.ndarray | None,
104:     anchor_embed: np.ndarray | None,
105:     anchor_mask: np.ndarray | None,
106:     anchor_lambda: float,
107:     train_ids: torch.Tensor,
108:     train_mask: torch.Tensor,
109:     val_ids: torch.Tensor,
110:     val_mask: torch.Tensor,
111:     *,
112:     n_steps: int = TRAIN_STEPS,
113:     custom_mask: np.ndarray | None = None,
114: ) -> dict[str, Any]:
115:     torch.manual_seed(seed)
116:     np.random.seed(seed)
117: 
118:     model = make_28layer_model(tok_gpt2, seed)
119: 
120:     if custom_embed is not None:
121:         emb_t = torch.from_numpy(custom_embed).to(
122:             model.model.embed_tokens.weight.device,
123:             dtype=model.model.embed_tokens.weight.dtype,
124:         )
125:         with torch.no_grad():
126:             if custom_mask is None:
127:                 model.model.embed_tokens.weight.copy_(emb_t)
128:             else:
129:                 mask_t = torch.from_numpy(custom_mask).to(emb_t.device)
130:                 model.model.embed_tokens.weight[mask_t] = emb_t[mask_t]
131:             if hasattr(model, "lm_head") and not model.config.tie_word_embeddings:
132:                 if custom_mask is None:
133:                     model.lm_head.weight.copy_(emb_t)
134:                 else:
135:                     model.lm_head.weight[mask_t] = emb_t[mask_t]
136: 
137:     anchor_target = None
138:     row_mask_t = None
139:     actual_lambda = 0.0
140:     if anchor_embed is not None and anchor_lambda > 0.0:
141:         anchor_target = torch.from_numpy(anchor_embed).to(DEVICE, dtype=torch.float32)
142:         actual_lambda = anchor_lambda
143:         if anchor_mask is not None:
144:             row_mask_t = torch.from_numpy(anchor_mask.astype(np.float32)).to(DEVICE).unsqueeze(1)
145: 
146:     model.train()
147:     optimizer = torch.optim.AdamW(
148:         model.parameters(), lr=g188.LR, betas=g188.BETAS, weight_decay=g188.WEIGHT_DECAY,
149:     )
150: 
151:     n_train = train_ids.shape[0]
152:     trajectory = {}
153:     t0 = time.time()
154: 
155:     for step in range(1, n_steps + 1):
156:         idx = torch.randint(0, n_train, (g188.BATCH_SIZE,))
157:         batch_ids = train_ids[idx].to(DEVICE)
158:         batch_mask = train_mask[idx].to(DEVICE)
159: 
160:         with torch.amp.autocast("cuda", dtype=g188.FORWARD_DTYPE):
161:             out = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
162:             loss = out.loss
163: 
164:         if not torch.isfinite(loss):
165:             raise RuntimeError(f"non-finite loss at step {step} arm={arm_label} seed={seed}")
166: 
167:         optimizer.zero_grad()
168:         loss.backward()
169: 
170:         if anchor_target is not None and actual_lambda > 0.0:
171:             with torch.no_grad():
172:                 coeff = 2.0 * actual_lambda
173:                 param = model.model.embed_tokens.weight
174:                 if param.grad is not None:
175:                     grad_add = (param.detach().to(anchor_target.dtype) - anchor_target) * coeff
176:                     if row_mask_t is not None:
177:                         grad_add = grad_add * row_mask_t
178:                     param.grad.add_(grad_add)
179: 
180:         torch.nn.utils.clip_grad_norm_(model.parameters(), g188.GRAD_CLIP)
181:         optimizer.step()
182: 
183:         if step % LOG_EVERY == 0:
184:             print_flush(f"    step {step}/{n_steps} loss={loss.item():.4f}")
185: 
186:         if step % EVAL_EVERY == 0 or step == n_steps:
187:             model.eval()
188:             with torch.no_grad():
189:                 val_nll = g188._eval_nll(model, val_ids, val_mask)
190:             trajectory[str(step)] = float(val_nll)
191:             if step % EVAL_EVERY == 0:
192:                 print_flush(f"    eval step={step} val_nll={val_nll:.4f}")
193:             model.train()
194: 
195:     model.eval()
196:     with torch.no_grad():
197:         final_nll = g188._eval_nll(model, val_ids, val_mask)
198: 
199:     result = {
200:         "arm_label": arm_label,
201:         "seed": seed,
202:         "anchor_lambda": actual_lambda,
203:         "has_row_mask": anchor_mask is not None,
204:         "num_layers": NUM_LAYERS_28,
205:         "final_val_nll": float(final_nll),
206:         "trajectory": trajectory,
207:         "wallclock_s": time.time() - t0,
208:     }
209:     del model, optimizer
210:     cleanup_cuda()
211:     return result
212: 
213: 
214: def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
215:     results = payload.get("results", {})
216:     required = ["scratch_ce", "matched_rows_only", "row_shuffled"]
217:     if not all(arm in results and len(results[arm]) >= len(SEEDS) for arm in required):
218:         return {"status": "incomplete"}
219: 
220:     scratch_nlls = {str(s): float(results["scratch_ce"][str(s)]["final_val_nll"]) for s in SEEDS}
221: 
222:     def arm_stats(arm_name):
223:         nlls = {str(s): float(results[arm_name][str(s)]["final_val_nll"]) for s in SEEDS}
224:         gaps = [scratch_nlls[str(s)] - nlls[str(s)] for s in SEEDS]
225:         return float(np.mean(gaps)), gaps
226: 
227:     matched_mean, matched_gaps = arm_stats("matched_rows_only")
228:     shuffled_mean, shuffled_gaps = arm_stats("row_shuffled")
229: 
230:     matched_all_positive = all(g > 0 for g in matched_gaps)
231:     shuffled_harmful = shuffled_mean <= 0.0
232: 
233:     if matched_mean >= 0.20 and matched_all_positive and shuffled_harmful:
234:         verdict = "PASS_PERSISTENCE"
235:     elif matched_mean >= 0.10 and matched_all_positive and matched_mean > shuffled_mean + 0.05:
236:         verdict = "PASS_ATTENUATION"
237:     else:
238:         verdict = "FAIL"
239: 
240:     return {
241:         "status": "complete",
242:         "verdict": verdict,
243:         "matched_mean_gain": matched_mean,
244:         "matched_per_seed": matched_gaps,
245:         "shuffled_mean_gain": shuffled_mean,
246:         "shuffled_per_seed": shuffled_gaps,
247:     }
248: 
249: 
250: def main() -> None:
251:     parser = argparse.ArgumentParser()
252:     parser.add_argument("--smoke", action="store_true")
253:     parser.add_argument("--no-resume", action="store_true")
254:     args = parser.parse_args()
255: 
256:     smoke = args.smoke
257:     n_steps = 50 if smoke else TRAIN_STEPS
258:     seeds = [42] if smoke else SEEDS
259:     run_out_path = OUT_PATH.with_name(OUT_PATH.stem + "_smoke.json") if smoke else OUT_PATH
260: 
261:     print_flush(f"=== g192 28-Layer String-Match Replication ===")
262:     print_flush(f"  smoke={smoke}, steps={n_steps}, seeds={seeds}, layers={NUM_LAYERS_28}")
263: 
264:     from transformers import AutoTokenizer, AutoModelForCausalLM
265:     tok_qwen = AutoTokenizer.from_pretrained(g188.QWEN_MODEL_ID)
266:     tok_gpt2 = AutoTokenizer.from_pretrained(g188.GPT2_MODEL_ID)
267:     if tok_gpt2.pad_token is None:
268:         tok_gpt2.pad_token = tok_gpt2.eos_token
269: 
270:     print_flush("\n--- Loading data ---")
271:     train_ids, train_mask, _ = g167.load_c4_windows(
272:         tok_gpt2, split="train", seed=g188.C4_TRAIN_SEED, n_windows=g188.N_TRAIN_WINDOWS,
273:     )
274:     train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
275:     val_ids, val_mask, _ = g167.load_c4_windows(
276:         tok_gpt2, split="train", seed=g188.C4_VAL_SEED, n_windows=g188.N_C4_VAL_WINDOWS,
277:         forbidden_hashes=train_hashes,
278:     )
279:     print_flush(f"  Train: {train_ids.shape}, Val: {val_ids.shape}")
280: 
281:     print_flush("\n--- Loading Qwen3 trained embeddings ---")
282:     qwen_model = AutoModelForCausalLM.from_pretrained(g188.QWEN_MODEL_ID, torch_dtype=torch.float32)
283:     trained_embed = qwen_model.model.embed_tokens.weight.detach().cpu().numpy()
284:     trained_fro = float(np.linalg.norm(trained_embed, "fro"))
285:     del qwen_model
286:     cleanup_cuda()
287:     print_flush(f"  Trained embed: {trained_embed.shape}, Fro={trained_fro:.1f}")
288: 
289:     print_flush("\n--- Building string-match embeddings ---")
290:     gpt2_vocab = len(tok_gpt2)
291:     embed_dim = trained_embed.shape[1]
292:     full_embed, matched_mask = g191.build_string_match_with_mask(
293:         tok_qwen, tok_gpt2, trained_embed, gpt2_vocab, embed_dim,
294:     )
295:     full_embed = g188.normalize_to_fro_norm(full_embed, trained_fro)
296: 
297:     rng = np.random.default_rng(192)
298:     shuffled = g191.build_row_shuffled_matched(full_embed, matched_mask, rng)
299:     shuffled = g188.normalize_to_fro_norm(shuffled, trained_fro)
300: 
301:     n_matched = int(matched_mask.sum())
302:     print_flush(f"  Matched: {n_matched}")
303: 
304:     arm_configs = {
305:         "scratch_ce":       {"custom_embed": None,       "anchor_embed": None,       "anchor_mask": None},
306:         "matched_rows_only":{"custom_embed": full_embed,  "anchor_embed": full_embed,  "anchor_mask": matched_mask, "custom_mask": matched_mask},
307:         "row_shuffled":     {"custom_embed": shuffled,    "anchor_embed": shuffled,    "anchor_mask": matched_mask, "custom_mask": matched_mask},
308:     }
309: 
310:     if not args.no_resume and run_out_path.exists():
311:         payload = json.loads(run_out_path.read_text(encoding="utf-8"))
312:     else:
313:         payload = {
314:             "genome": 192,
315:             "name": "28layer_replication",
316:             "timestamp_utc_started": now_utc(),
317:             "config": {
318:                 "train_steps": n_steps,
319:                 "seeds": seeds,
320:                 "anchor_lambda": ANCHOR_LAMBDA,
321:                 "num_layers": NUM_LAYERS_28,
322:                 "n_matched": n_matched,
323:                 "trained_fro": trained_fro,
324:             },
325:             "results": {},
326:             "summary": {},
327:             "verdict": "INCOMPLETE",
328:         }
329: 
330:     t_start = time.time()
331: 
332:     def save():
333:         payload["timestamp_utc_last_write"] = now_utc()
334:         payload["elapsed_s"] = time.time() - t_start
335:         tmp = run_out_path.with_suffix(run_out_path.suffix + ".tmp")
336:         tmp.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")
337:         os.replace(tmp, run_out_path)
338: 
339:     for arm_label in ARMS:
340:         payload["results"].setdefault(arm_label, {})
341:         cfg = arm_configs[arm_label]
342: 
343:         for seed in seeds:
344:             key = str(seed)
345:             if key in payload["results"][arm_label] and not args.no_resume:
346:                 cell = payload["results"][arm_label][key]
347:                 if isinstance(cell, dict) and "final_val_nll" in cell:
348:                     print_flush(f"\n  Skipping {arm_label}/seed={seed} (done)")
349:                     continue
350: 
351:             print_flush(f"\n  === {arm_label} seed={seed} ===")
352:             result = train_cell_28layer(
353:                 arm_label=arm_label,
354:                 seed=seed,
355:                 tok_gpt2=tok_gpt2,
356:                 custom_embed=cfg["custom_embed"],
357:                 anchor_embed=cfg["anchor_embed"],
358:                 anchor_mask=cfg["anchor_mask"],
359:                 anchor_lambda=ANCHOR_LAMBDA,
360:                 train_ids=train_ids,
361:                 train_mask=train_mask,
362:                 val_ids=val_ids,
363:                 val_mask=val_mask,
364:                 n_steps=n_steps,
365:                 custom_mask=cfg.get("custom_mask"),
366:             )
367:             payload["results"][arm_label][key] = result
368:             save()
369:             print_flush(f"  {arm_label} seed={seed} nll={result['final_val_nll']:.4f} ({result['wallclock_s']:.0f}s)")
370: 
371:     summary = compute_verdict(payload)
372:     payload["summary"] = summary
373:     payload["verdict"] = summary.get("verdict", "INCOMPLETE")
374:     save()
375: 
376:     print_flush(f"\n*** g192 VERDICT: {summary.get('verdict', '?')} ***")
377:     for key, val in summary.items():
378:         if key.endswith("_mean_gain"):
379:             print_flush(f"  {key}: {val:+.4f}")
380: 
381: 
382: if __name__ == "__main__":
383:     main()

 succeeded in 318ms:
1: """
2: genome_195_untied_input_output_factorial.py
3: 
4: Resolves adversarial A18 #1 (SEV-10): tied lm_head confound.
5: With tie_word_embeddings=True, injecting trained embeddings into embed_tokens
6: simultaneously sets the output classifier basis. This experiment untie the weights
7: and tests input-only, output-only, both, and tied reference arms.
8: 
9: 5 arms x 3 seeds = 15 cells.
10: Gated on g194 PASS_DIRECTION.
11: """
12: from __future__ import annotations
13: 
14: import argparse
15: import gc
16: import json
17: import os
18: import sys
19: import time
20: from datetime import datetime, timezone
21: from pathlib import Path
22: from typing import Any
23: 
24: import numpy as np
25: import torch
26: 
27: CODE_DIR = Path(__file__).resolve().parent
28: ROOT = CODE_DIR.parent
29: if str(CODE_DIR) not in sys.path:
30:     sys.path.insert(0, str(CODE_DIR))
31: 
32: import genome_165_annealed_donor as g165
33: import genome_167_kd_canonical as g167
34: import genome_188_tokenizer_flow_bridge as g188
35: import genome_191_string_match_decomposition as g191
36: 
37: OUT_PATH = ROOT / "results" / "genome_195_untied_input_output_factorial.json"
38: 
39: SEEDS = [42, 7, 13]
40: TRAIN_STEPS = 5000
41: ANCHOR_LAMBDA = 0.01
42: LOG_EVERY = 100
43: EVAL_EVERY = 500
44: DEVICE = g165.DEVICE
45: 
46: os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
47: os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
48: 
49: if torch.cuda.is_available():
50:     torch.backends.cuda.matmul.allow_tf32 = True
51:     torch.backends.cudnn.allow_tf32 = True
52: if hasattr(torch, "set_float32_matmul_precision"):
53:     torch.set_float32_matmul_precision("high")
54: 
55: ARMS = [
56:     "scratch_untied",
57:     "input_inject_anchor",
58:     "output_inject_anchor",
59:     "both_inject_anchor",
60:     "tied_reference",
61: ]
62: 
63: 
64: def print_flush(msg: str) -> None:
65:     print(msg, flush=True)
66: 
67: 
68: def now_utc() -> str:
69:     return datetime.now(timezone.utc).isoformat()
70: 
71: 
72: def cleanup_cuda() -> None:
73:     gc.collect()
74:     if torch.cuda.is_available():
75:         torch.cuda.empty_cache()
76: 
77: 
78: def make_untied_model(tok_gpt2, seed: int):
79:     from transformers import Qwen3ForCausalLM
80:     from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
81: 
82:     torch.manual_seed(seed)
83:     cfg = Qwen3Config(
84:         vocab_size=len(tok_gpt2),
85:         hidden_size=1024,
86:         num_hidden_layers=8,
87:         num_attention_heads=16,
88:         num_key_value_heads=4,
89:         intermediate_size=2816,
90:         max_position_embeddings=g188.SEQ_LEN + 64,
91:         rms_norm_eps=1e-6,
92:         tie_word_embeddings=False,
93:         head_dim=64,
94:         rope_theta=10000.0,
95:         use_cache=False,
96:     )
97:     model = Qwen3ForCausalLM(cfg).to(device=DEVICE)
98:     model.config.pad_token_id = tok_gpt2.pad_token_id
99:     return model
100: 
101: 
102: def train_cell_untied(
103:     arm_label: str,
104:     seed: int,
105:     tok_gpt2,
106:     embed_init: np.ndarray | None,
107:     lm_head_init: np.ndarray | None,
108:     anchor_embed: np.ndarray | None,
109:     anchor_lm_head: np.ndarray | None,
110:     anchor_mask: np.ndarray | None,
111:     anchor_lambda: float,
112:     train_ids: torch.Tensor,
113:     train_mask: torch.Tensor,
114:     val_ids: torch.Tensor,
115:     val_mask: torch.Tensor,
116:     *,
117:     n_steps: int = TRAIN_STEPS,
118:     tied: bool = False,
119: ) -> dict[str, Any]:
120:     torch.manual_seed(seed)
121:     np.random.seed(seed)
122: 
123:     if tied:
124:         model = g188.make_gpt2_qwen3_model(tok_gpt2, seed)
125:     else:
126:         model = make_untied_model(tok_gpt2, seed)
127: 
128:     if embed_init is not None:
129:         emb_t = torch.from_numpy(embed_init).to(
130:             model.model.embed_tokens.weight.device,
131:             dtype=model.model.embed_tokens.weight.dtype,
132:         )
133:         with torch.no_grad():
134:             if anchor_mask is not None:
135:                 mask_t = torch.from_numpy(anchor_mask).to(emb_t.device)
136:                 model.model.embed_tokens.weight[mask_t] = emb_t[mask_t]
137:             else:
138:                 model.model.embed_tokens.weight.copy_(emb_t)
139: 
140:     if lm_head_init is not None and not tied:
141:         head_t = torch.from_numpy(lm_head_init).to(
142:             model.lm_head.weight.device,
143:             dtype=model.lm_head.weight.dtype,
144:         )
145:         with torch.no_grad():
146:             if anchor_mask is not None:
147:                 mask_t = torch.from_numpy(anchor_mask).to(head_t.device)
148:                 model.lm_head.weight[mask_t] = head_t[mask_t]
149:             else:
150:                 model.lm_head.weight.copy_(head_t)
151: 
152:     embed_anchor_target = None
153:     lm_head_anchor_target = None
154:     row_mask_t = None
155:     actual_lambda = 0.0
156: 
157:     if anchor_lambda > 0.0 and (anchor_embed is not None or anchor_lm_head is not None):
158:         actual_lambda = anchor_lambda
159:         if anchor_mask is not None:
160:             row_mask_t = torch.from_numpy(anchor_mask.astype(np.float32)).to(DEVICE).unsqueeze(1)
161: 
162:         if anchor_embed is not None:
163:             embed_anchor_target = torch.from_numpy(anchor_embed).to(DEVICE, dtype=torch.float32)
164:         if anchor_lm_head is not None and not tied:
165:             lm_head_anchor_target = torch.from_numpy(anchor_lm_head).to(DEVICE, dtype=torch.float32)
166: 
167:     model.train()
168:     optimizer = torch.optim.AdamW(
169:         model.parameters(), lr=g188.LR, betas=g188.BETAS, weight_decay=g188.WEIGHT_DECAY,
170:     )
171: 
172:     n_train = train_ids.shape[0]
173:     trajectory = {}
174:     t0 = time.time()
175: 
176:     for step in range(1, n_steps + 1):
177:         idx = torch.randint(0, n_train, (g188.BATCH_SIZE,))
178:         batch_ids = train_ids[idx].to(DEVICE)
179:         batch_mask = train_mask[idx].to(DEVICE)
180: 
181:         with torch.amp.autocast("cuda", dtype=g188.FORWARD_DTYPE):
182:             out = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
183:             loss = out.loss
184: 
185:         if not torch.isfinite(loss):
186:             raise RuntimeError(f"non-finite loss at step {step} arm={arm_label} seed={seed}")
187: 
188:         optimizer.zero_grad()
189:         loss.backward()
190: 
191:         coeff = 2.0 * actual_lambda
192:         if coeff > 0.0:
193:             with torch.no_grad():
194:                 if embed_anchor_target is not None:
195:                     param = model.model.embed_tokens.weight
196:                     if param.grad is not None:
197:                         grad_add = (param.detach().to(embed_anchor_target.dtype) - embed_anchor_target) * coeff
198:                         if row_mask_t is not None:
199:                             grad_add = grad_add * row_mask_t
200:                         param.grad.add_(grad_add)
201: 
202:                 if lm_head_anchor_target is not None:
203:                     param = model.lm_head.weight
204:                     if param.grad is not None:
205:                         grad_add = (param.detach().to(lm_head_anchor_target.dtype) - lm_head_anchor_target) * coeff
206:                         if row_mask_t is not None:
207:                             grad_add = grad_add * row_mask_t
208:                         param.grad.add_(grad_add)
209: 
210:         torch.nn.utils.clip_grad_norm_(model.parameters(), g188.GRAD_CLIP)
211:         optimizer.step()
212: 
213:         if step % LOG_EVERY == 0:
214:             print_flush(f"    step {step}/{n_steps} loss={loss.item():.4f}")
215: 
216:         if step % EVAL_EVERY == 0 or step == n_steps:
217:             model.eval()
218:             with torch.no_grad():
219:                 val_nll = g188._eval_nll(model, val_ids, val_mask)
220:             trajectory[str(step)] = float(val_nll)
221:             if step % EVAL_EVERY == 0:
222:                 print_flush(f"    eval step={step} val_nll={val_nll:.4f}")
223:             model.train()
224: 
225:     model.eval()
226:     with torch.no_grad():
227:         final_nll = g188._eval_nll(model, val_ids, val_mask)
228: 
229:     result = {
230:         "arm_label": arm_label,
231:         "seed": seed,
232:         "anchor_lambda": actual_lambda,
233:         "has_row_mask": anchor_mask is not None,
234:         "tied": tied,
235:         "final_val_nll": float(final_nll),
236:         "trajectory": trajectory,
237:         "wallclock_s": time.time() - t0,
238:     }
239:     del model, optimizer
240:     cleanup_cuda()
241:     return result
242: 
243: 
244: def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
245:     results = payload.get("results", {})
246:     required = ARMS
247:     if not all(arm in results and len(results[arm]) >= len(SEEDS) for arm in required):
248:         return {"status": "incomplete"}
249: 
250:     scratch_nlls = {str(s): float(results["scratch_untied"][str(s)]["final_val_nll"]) for s in SEEDS}
251: 
252:     def arm_stats(arm_name):
253:         nlls = {str(s): float(results[arm_name][str(s)]["final_val_nll"]) for s in SEEDS}
254:         gaps = [scratch_nlls[str(s)] - nlls[str(s)] for s in SEEDS]
255:         return float(np.mean(gaps)), gaps
256: 
257:     input_mean, input_gaps = arm_stats("input_inject_anchor")
258:     output_mean, output_gaps = arm_stats("output_inject_anchor")
259:     both_mean, both_gaps = arm_stats("both_inject_anchor")
260:     tied_mean, tied_gaps = arm_stats("tied_reference")
261: 
262:     input_dominates = input_mean >= 0.30 and output_mean < 0.15
263:     output_dominates = output_mean >= 0.30 and input_mean < 0.15
264:     both_needed = both_mean >= 0.30 and input_mean < 0.80 * both_mean and output_mean < 0.80 * both_mean
265: 
266:     if input_dominates:
267:         verdict = "PASS_INPUT"
268:     elif output_dominates:
269:         verdict = "PASS_OUTPUT"
270:     elif both_needed:
271:         verdict = "PASS_BOTH_NEEDED"
272:     elif input_mean >= 0.20 and input_mean > output_mean:
273:         verdict = "PASS_INPUT_DOMINANT"
274:     elif output_mean >= 0.20 and output_mean > input_mean:
275:         verdict = "PASS_OUTPUT_DOMINANT"
276:     elif max(input_mean, output_mean, both_mean) < 0.10:
277:         verdict = "FAIL"
278:     else:
279:         verdict = "AMBIGUOUS_POSITIVE"
280: 
281:     return {
282:         "status": "complete",
283:         "verdict": verdict,
284:         "input_mean_gain": input_mean,
285:         "input_per_seed": input_gaps,
286:         "output_mean_gain": output_mean,
287:         "output_per_seed": output_gaps,
288:         "both_mean_gain": both_mean,
289:         "both_per_seed": both_gaps,
290:         "tied_mean_gain": tied_mean,
291:         "tied_per_seed": tied_gaps,
292:     }
293: 
294: 
295: def main() -> None:
296:     parser = argparse.ArgumentParser()
297:     parser.add_argument("--smoke", action="store_true")
298:     parser.add_argument("--no-resume", action="store_true")
299:     args = parser.parse_args()
300: 
301:     smoke = args.smoke
302:     n_steps = 50 if smoke else TRAIN_STEPS
303:     seeds = [42] if smoke else SEEDS
304:     run_out_path = OUT_PATH.with_name(OUT_PATH.stem + "_smoke.json") if smoke else OUT_PATH
305: 
306:     print_flush(f"=== g195 Untied Input/Output Factorial ===")
307:     print_flush(f"  smoke={smoke}, steps={n_steps}, seeds={seeds}")
308: 
309:     from transformers import AutoTokenizer, AutoModelForCausalLM
310:     tok_qwen = AutoTokenizer.from_pretrained(g188.QWEN_MODEL_ID)
311:     tok_gpt2 = AutoTokenizer.from_pretrained(g188.GPT2_MODEL_ID)
312:     if tok_gpt2.pad_token is None:
313:         tok_gpt2.pad_token = tok_gpt2.eos_token
314: 
315:     print_flush("\n--- Loading data ---")
316:     train_ids, train_mask, _ = g167.load_c4_windows(
317:         tok_gpt2, split="train", seed=g188.C4_TRAIN_SEED, n_windows=g188.N_TRAIN_WINDOWS,
318:     )
319:     train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
320:     val_ids, val_mask, _ = g167.load_c4_windows(
321:         tok_gpt2, split="train", seed=g188.C4_VAL_SEED, n_windows=g188.N_C4_VAL_WINDOWS,
322:         forbidden_hashes=train_hashes,
323:     )
324:     print_flush(f"  Train: {train_ids.shape}, Val: {val_ids.shape}")
325: 
326:     print_flush("\n--- Loading Qwen3 trained embeddings ---")
327:     qwen_model = AutoModelForCausalLM.from_pretrained(g188.QWEN_MODEL_ID, torch_dtype=torch.float32)
328:     trained_embed = qwen_model.model.embed_tokens.weight.detach().cpu().numpy()
329:     trained_fro = float(np.linalg.norm(trained_embed, "fro"))
330:     del qwen_model
331:     cleanup_cuda()
332:     print_flush(f"  Trained embed: {trained_embed.shape}, Fro={trained_fro:.1f}")
333: 
334:     print_flush("\n--- Building string-match embeddings ---")
335:     gpt2_vocab = len(tok_gpt2)
336:     embed_dim = trained_embed.shape[1]
337:     full_embed, matched_mask = g191.build_string_match_with_mask(
338:         tok_qwen, tok_gpt2, trained_embed, gpt2_vocab, embed_dim,
339:     )
340:     full_embed = g188.normalize_to_fro_norm(full_embed, trained_fro)
341: 
342:     n_matched = int(matched_mask.sum())
343:     print_flush(f"  Matched: {n_matched}")
344: 
345:     arm_configs = {
346:         "scratch_untied": {
347:             "embed_init": None, "lm_head_init": None,
348:             "anchor_embed": None, "anchor_lm_head": None,
349:             "anchor_mask": None, "tied": False,
350:         },
351:         "input_inject_anchor": {
352:             "embed_init": full_embed, "lm_head_init": None,
353:             "anchor_embed": full_embed, "anchor_lm_head": None,
354:             "anchor_mask": matched_mask, "tied": False,
355:         },
356:         "output_inject_anchor": {
357:             "embed_init": None, "lm_head_init": full_embed,
358:             "anchor_embed": None, "anchor_lm_head": full_embed,
359:             "anchor_mask": matched_mask, "tied": False,
360:         },
361:         "both_inject_anchor": {
362:             "embed_init": full_embed, "lm_head_init": full_embed,
363:             "anchor_embed": full_embed, "anchor_lm_head": full_embed,
364:             "anchor_mask": matched_mask, "tied": False,
365:         },
366:         "tied_reference": {
367:             "embed_init": full_embed, "lm_head_init": None,
368:             "anchor_embed": full_embed, "anchor_lm_head": None,
369:             "anchor_mask": matched_mask, "tied": True,
370:         },
371:     }
372: 
373:     if not args.no_resume and run_out_path.exists():
374:         payload = json.loads(run_out_path.read_text(encoding="utf-8"))
375:     else:
376:         payload = {
377:             "genome": 195,
378:             "name": "untied_input_output_factorial",
379:             "timestamp_utc_started": now_utc(),
380:             "config": {
381:                 "train_steps": n_steps,
382:                 "seeds": seeds,
383:                 "anchor_lambda": ANCHOR_LAMBDA,
384:                 "n_matched": n_matched,
385:                 "trained_fro": trained_fro,
386:             },
387:             "results": {},
388:             "summary": {},
389:             "verdict": "INCOMPLETE",
390:         }
391: 
392:     t_start = time.time()
393: 
394:     def save():
395:         payload["timestamp_utc_last_write"] = now_utc()
396:         payload["elapsed_s"] = time.time() - t_start
397:         tmp = run_out_path.with_suffix(run_out_path.suffix + ".tmp")
398:         tmp.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")
399:         os.replace(tmp, run_out_path)
400: 
401:     for arm_label in ARMS:
402:         payload["results"].setdefault(arm_label, {})
403:         cfg = arm_configs[arm_label]
404: 
405:         for seed in seeds:
406:             key = str(seed)
407:             if key in payload["results"][arm_label] and not args.no_resume:
408:                 cell = payload["results"][arm_label][key]
409:                 if isinstance(cell, dict) and "final_val_nll" in cell:
410:                     print_flush(f"\n  Skipping {arm_label}/seed={seed} (done)")
411:                     continue
412: 
413:             print_flush(f"\n  === {arm_label} seed={seed} ===")
414:             result = train_cell_untied(
415:                 arm_label=arm_label,
416:                 seed=seed,
417:                 tok_gpt2=tok_gpt2,
418:                 embed_init=cfg["embed_init"],
419:                 lm_head_init=cfg["lm_head_init"],
420:                 anchor_embed=cfg["anchor_embed"],
421:                 anchor_lm_head=cfg["anchor_lm_head"],
422:                 anchor_mask=cfg["anchor_mask"],
423:                 anchor_lambda=ANCHOR_LAMBDA,
424:                 train_ids=train_ids,
425:                 train_mask=train_mask,
426:                 val_ids=val_ids,
427:                 val_mask=val_mask,
428:                 n_steps=n_steps,
429:                 tied=cfg["tied"],
430:             )
431:             payload["results"][arm_label][key] = result
432:             save()
433:             print_flush(f"  {arm_label} seed={seed} nll={result['final_val_nll']:.4f} ({result['wallclock_s']:.0f}s)")
434: 
435:     summary = compute_verdict(payload)
436:     payload["summary"] = summary
437:     payload["verdict"] = summary.get("verdict", "INCOMPLETE")
438:     save()
439: 
440:     print_flush(f"\n*** g195 VERDICT: {summary.get('verdict', '?')} ***")
441:     for key, val in summary.items():
442:         if key.endswith("_mean_gain"):
443:             print_flush(f"  {key}: {val:+.4f}")
444: 
445: 
446: if __name__ == "__main__":
447:     main()

 succeeded in 360ms:
1: """
2: genome_196_anchor_residue_factorial.py
3: 
4: Resolves A18 SEV-10 #2 (anchor dominance) and #3 (scaffold-vs-content).
5: Tests whether trained row directions leave a persistent basin residue after
6: the anchor tether is removed, or only help as active regularization.
7: 
8: 10 arms x 3 seeds = 30 cells.
9: Gated on g195 (determines intervention surface: input/output/both/tied).
10: Prereg: research/prereg/genome_196_anchor_residue_factorial_2026-04-30.md
11: """
12: from __future__ import annotations
13: 
14: import argparse
15: import gc
16: import json
17: import os
18: import sys
19: import time
20: from datetime import datetime, timezone
21: from pathlib import Path
22: from typing import Any
23: 
24: import numpy as np
25: import torch
26: 
27: CODE_DIR = Path(__file__).resolve().parent
28: ROOT = CODE_DIR.parent
29: if str(CODE_DIR) not in sys.path:
30:     sys.path.insert(0, str(CODE_DIR))
31: 
32: import genome_165_annealed_donor as g165
33: import genome_167_kd_canonical as g167
34: import genome_188_tokenizer_flow_bridge as g188
35: import genome_191_string_match_decomposition as g191
36: 
37: OUT_PATH = ROOT / "results" / "genome_196_anchor_residue_factorial.json"
38: 
39: SEEDS = [42, 7, 13]
40: TRAIN_STEPS = 5000
41: ANCHOR_LAMBDA = 0.01
42: LOG_EVERY = 100
43: EVAL_EVERY = 500
44: DEVICE = g165.DEVICE
45: 
46: SCAFFOLD_SEED_ORTHO = 19601
47: SCAFFOLD_SEED_COV = 19602
48: 
49: os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
50: os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
51: 
52: if torch.cuda.is_available():
53:     torch.backends.cuda.matmul.allow_tf32 = True
54:     torch.backends.cudnn.allow_tf32 = True
55: if hasattr(torch, "set_float32_matmul_precision"):
56:     torch.set_float32_matmul_precision("high")
57: 
58: ARMS = [
59:     "scratch",
60:     "init_only",
61:     "anchor_only_full",
62:     "init_anchor_full",
63:     "cutoff_50",
64:     "cutoff_500",
65:     "cutoff_2000",
66:     "late_anchor_only_2000",
67:     "orthogonal_scaffold_full",
68:     "cov_scaffold_full",
69: ]
70: 
71: 
72: def print_flush(msg: str) -> None:
73:     print(msg, flush=True)
74: 
75: 
76: def now_utc() -> str:
77:     return datetime.now(timezone.utc).isoformat()
78: 
79: 
80: def cleanup_cuda() -> None:
81:     gc.collect()
82:     if torch.cuda.is_available():
83:         torch.cuda.empty_cache()
84: 
85: 
86: # ---------- Scaffold construction ----------
87: 
88: def build_orthogonal_scaffold(
89:     target: np.ndarray, matched_mask: np.ndarray,
90: ) -> np.ndarray:
91:     """Rotate all matched rows by a fixed random orthogonal matrix Q.
92:     Preserves all pairwise cosines; destroys trained coordinate basis."""
93:     d = target.shape[1]
94:     rng = np.random.default_rng(SCAFFOLD_SEED_ORTHO)
95:     A = rng.standard_normal((d, d)).astype(np.float64)
96:     Q, _ = np.linalg.qr(A)
97:     Q = Q.astype(np.float32)
98: 
99:     out = np.zeros_like(target)
100:     matched_ids = np.where(matched_mask)[0]
101:     out[matched_ids] = target[matched_ids] @ Q
102:     return out
103: 
104: 
105: def build_covariance_scaffold(
106:     target: np.ndarray, matched_mask: np.ndarray,
107: ) -> np.ndarray:
108:     """Draw rows from N(mean, cov + eps*I) of matched target rows.
109:     Preserves second-order statistics; destroys token identity."""
110:     matched_ids = np.where(matched_mask)[0]
111:     T_m = target[matched_ids].astype(np.float64)
112:     mean = T_m.mean(axis=0)
113:     cov = np.cov(T_m, rowvar=False)
114:     eps = 1e-6
115:     cov += eps * np.eye(cov.shape[0])
116: 
117:     rng = np.random.default_rng(SCAFFOLD_SEED_COV)
118:     X_m = rng.multivariate_normal(mean, cov, size=len(matched_ids)).astype(np.float32)
119: 
120:     norms = np.linalg.norm(X_m, axis=1, keepdims=True)
121:     norms = np.maximum(norms, 1e-8)
122:     X_m = X_m / norms
123: 
124:     target_norms = np.linalg.norm(target[matched_ids], axis=1)
125:     uniform_norm = float(target_norms.mean())
126:     X_m = X_m * uniform_norm
127: 
128:     target_fro = float(np.linalg.norm(target[matched_ids], "fro"))
129:     current_fro = float(np.linalg.norm(X_m, "fro"))
130:     if current_fro > 1e-8:
131:         X_m = X_m * (target_fro / current_fro)
132: 
133:     out = np.zeros_like(target)
134:     out[matched_ids] = X_m
135:     return out
136: 
137: 
138: # ---------- Anchor schedule ----------
139: 
140: def get_anchor_lambda(arm_label: str, step: int) -> float:
141:     """Return the anchor lambda for this arm at this step."""
142:     if arm_label in ("scratch", "init_only"):
143:         return 0.0
144:     elif arm_label in ("anchor_only_full", "init_anchor_full",
145:                        "orthogonal_scaffold_full", "cov_scaffold_full"):
146:         return ANCHOR_LAMBDA
147:     elif arm_label == "cutoff_50":
148:         return ANCHOR_LAMBDA if step <= 50 else 0.0
149:     elif arm_label == "cutoff_500":
150:         return ANCHOR_LAMBDA if step <= 500 else 0.0
151:     elif arm_label == "cutoff_2000":
152:         return ANCHOR_LAMBDA if step <= 2000 else 0.0
153:     elif arm_label == "late_anchor_only_2000":
154:         return ANCHOR_LAMBDA if step > 2000 else 0.0
155:     else:
156:         raise ValueError(f"Unknown arm: {arm_label}")
157: 
158: 
159: # ---------- Training cell ----------
160: 
161: def train_cell(
162:     arm_label: str,
163:     seed: int,
164:     tok_gpt2,
165:     embed_init: np.ndarray | None,
166:     lm_head_init: np.ndarray | None,
167:     anchor_target: np.ndarray | None,
168:     anchor_mask: np.ndarray | None,
169:     train_ids: torch.Tensor,
170:     train_mask: torch.Tensor,
171:     val_ids: torch.Tensor,
172:     val_mask: torch.Tensor,
173:     *,
174:     n_steps: int = TRAIN_STEPS,
175:     tied: bool = True,
176:     surface: str = "tied",
177: ) -> dict[str, Any]:
178:     torch.manual_seed(seed)
179:     np.random.seed(seed)
180: 
181:     if tied:
182:         model = g188.make_gpt2_qwen3_model(tok_gpt2, seed)
183:     else:
184:         from transformers import Qwen3ForCausalLM
185:         from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
186:         torch.manual_seed(seed)
187:         cfg = Qwen3Config(
188:             vocab_size=len(tok_gpt2),
189:             hidden_size=1024,
190:             num_hidden_layers=8,
191:             num_attention_heads=16,
192:             num_key_value_heads=4,
193:             intermediate_size=2816,
194:             max_position_embeddings=g188.SEQ_LEN + 64,
195:             rms_norm_eps=1e-6,
196:             tie_word_embeddings=False,
197:             head_dim=64,
198:             rope_theta=10000.0,
199:             use_cache=False,
200:         )
201:         model = Qwen3ForCausalLM(cfg).to(device=DEVICE)
202:         model.config.pad_token_id = tok_gpt2.pad_token_id
203: 
204:     if embed_init is not None:
205:         emb_t = torch.from_numpy(embed_init).to(
206:             model.model.embed_tokens.weight.device,
207:             dtype=model.model.embed_tokens.weight.dtype,
208:         )
209:         with torch.no_grad():
210:             if anchor_mask is not None:
211:                 mask_t = torch.from_numpy(anchor_mask).to(emb_t.device)
212:                 model.model.embed_tokens.weight[mask_t] = emb_t[mask_t]
213:             else:
214:                 model.model.embed_tokens.weight.copy_(emb_t)
215: 
216:     if lm_head_init is not None and not tied:
217:         head_t = torch.from_numpy(lm_head_init).to(
218:             model.lm_head.weight.device,
219:             dtype=model.lm_head.weight.dtype,
220:         )
221:         with torch.no_grad():
222:             if anchor_mask is not None:
223:                 mask_t = torch.from_numpy(anchor_mask).to(head_t.device)
224:                 model.lm_head.weight[mask_t] = head_t[mask_t]
225:             else:
226:                 model.lm_head.weight.copy_(head_t)
227: 
228:     embed_anchor_t = None
229:     lm_head_anchor_t = None
230:     row_mask_t = None
231: 
232:     if anchor_target is not None and anchor_mask is not None:
233:         row_mask_t = torch.from_numpy(
234:             anchor_mask.astype(np.float32)
235:         ).to(DEVICE).unsqueeze(1)
236: 
237:         if tied or surface in ("tied", "input", "both"):
238:             embed_anchor_t = torch.from_numpy(anchor_target).to(
239:                 DEVICE, dtype=torch.float32,
240:             )
241:         if not tied and surface in ("output", "both"):
242:             lm_head_anchor_t = torch.from_numpy(anchor_target).to(
243:                 DEVICE, dtype=torch.float32,
244:             )
245: 
246:     model.train()
247:     optimizer = torch.optim.AdamW(
248:         model.parameters(), lr=g188.LR, betas=g188.BETAS, weight_decay=g188.WEIGHT_DECAY,
249:     )
250: 
251:     n_train = train_ids.shape[0]
252:     trajectory = {}
253:     t0 = time.time()
254:     cutoff_eval_steps = {50, 500, 2000}
255: 
256:     for step in range(1, n_steps + 1):
257:         idx = torch.randint(0, n_train, (g188.BATCH_SIZE,))
258:         batch_ids = train_ids[idx].to(DEVICE)
259:         batch_mask = train_mask[idx].to(DEVICE)
260: 
261:         with torch.amp.autocast("cuda", dtype=g188.FORWARD_DTYPE):
262:             out = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
263:             loss = out.loss
264: 
265:         if not torch.isfinite(loss):
266:             raise RuntimeError(f"non-finite loss at step {step} arm={arm_label} seed={seed}")
267: 
268:         optimizer.zero_grad()
269:         loss.backward()
270: 
271:         lam = get_anchor_lambda(arm_label, step)
272:         coeff = 2.0 * lam
273:         if coeff > 0.0:
274:             with torch.no_grad():
275:                 if embed_anchor_t is not None:
276:                     param = model.model.embed_tokens.weight
277:                     if param.grad is not None:
278:                         grad_add = (param.detach().to(embed_anchor_t.dtype) - embed_anchor_t) * coeff
279:                         if row_mask_t is not None:
280:                             grad_add = grad_add * row_mask_t
281:                         param.grad.add_(grad_add)
282: 
283:                 if lm_head_anchor_t is not None:
284:                     param = model.lm_head.weight
285:                     if param.grad is not None:
286:                         grad_add = (param.detach().to(lm_head_anchor_t.dtype) - lm_head_anchor_t) * coeff
287:                         if row_mask_t is not None:
288:                             grad_add = grad_add * row_mask_t
289:                         param.grad.add_(grad_add)
290: 
291:         torch.nn.utils.clip_grad_norm_(model.parameters(), g188.GRAD_CLIP)
292:         optimizer.step()
293: 
294:         if step % LOG_EVERY == 0:
295:             print_flush(f"    step {step}/{n_steps} loss={loss.item():.4f}")
296: 
297:         do_eval = (step % EVAL_EVERY == 0 or step == n_steps
298:                    or step in cutoff_eval_steps)
299:         if do_eval:
300:             model.eval()
301:             with torch.no_grad():
302:                 val_nll = g188._eval_nll(model, val_ids, val_mask)
303:             trajectory[str(step)] = float(val_nll)
304:             if step % EVAL_EVERY == 0:
305:                 print_flush(f"    eval step={step} val_nll={val_nll:.4f}")
306:             model.train()
307: 
308:     model.eval()
309:     with torch.no_grad():
310:         final_nll = g188._eval_nll(model, val_ids, val_mask)
311: 
312:     result = {
313:         "arm_label": arm_label,
314:         "seed": seed,
315:         "tied": tied,
316:         "surface": surface,
317:         "final_val_nll": float(final_nll),
318:         "trajectory": trajectory,
319:         "wallclock_s": time.time() - t0,
320:     }
321:     del model, optimizer
322:     cleanup_cuda()
323:     return result
324: 
325: 
326: # ---------- Verdict ----------
327: 
328: def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
329:     results = payload.get("results", {})
330:     if not all(arm in results and len(results[arm]) >= len(SEEDS) for arm in ARMS):
331:         return {"status": "incomplete"}
332: 
333:     scratch_nlls = {str(s): float(results["scratch"][str(s)]["final_val_nll"]) for s in SEEDS}
334: 
335:     def arm_stats(arm_name):
336:         nlls = {str(s): float(results[arm_name][str(s)]["final_val_nll"]) for s in SEEDS}
337:         gaps = [scratch_nlls[str(s)] - nlls[str(s)] for s in SEEDS]
338:         return float(np.mean(gaps)), gaps
339: 
340:     init_mean, init_gaps = arm_stats("init_only")
341:     anchor_full_mean, anchor_full_gaps = arm_stats("anchor_only_full")
342:     init_anchor_mean, init_anchor_gaps = arm_stats("init_anchor_full")
343:     cut50_mean, cut50_gaps = arm_stats("cutoff_50")
344:     cut500_mean, cut500_gaps = arm_stats("cutoff_500")
345:     cut2000_mean, cut2000_gaps = arm_stats("cutoff_2000")
346:     late_mean, late_gaps = arm_stats("late_anchor_only_2000")
347:     ortho_mean, ortho_gaps = arm_stats("orthogonal_scaffold_full")
348:     cov_mean, cov_gaps = arm_stats("cov_scaffold_full")
349: 
350:     replication_gate = (
351:         (anchor_full_mean >= 0.30 and all(g > 0 for g in anchor_full_gaps))
352:         or (init_anchor_mean >= 0.30 and all(g > 0 for g in init_anchor_gaps))
353:     )
354: 
355:     control_max_gain = max(ortho_mean, cov_mean)
356: 
357:     residue_fraction_2000 = (
358:         cut2000_mean / anchor_full_mean if anchor_full_mean > 0.01 else 0.0
359:     )
360: 
361:     scaffold_alt = (
362:         control_max_gain >= 0.20
363:         or (anchor_full_mean > 0.01 and control_max_gain >= 0.50 * anchor_full_mean)
364:     )
365: 
366:     if not replication_gate:
367:         verdict = "FAIL_REPLICATION"
368:     elif (
369:         cut2000_mean >= 0.20
370:         and all(g > 0 for g in cut2000_gaps)
371:         and residue_fraction_2000 >= 0.45
372:         and cut2000_mean >= init_mean + 0.10
373:         and control_max_gain <= 0.15
374:         and anchor_full_mean - control_max_gain >= 0.20
375:     ):
376:         verdict = "PASS_RESIDUE"
377:     elif scaffold_alt:
378:         verdict = "PASS_SCAFFOLD_ALT"
379:     elif (
380:         cut2000_mean >= 0.12
381:         and sum(1 for g in cut2000_gaps if g > 0) >= 2
382:         and residue_fraction_2000 >= 0.25
383:         and not scaffold_alt
384:     ):
385:         verdict = "PASS_PARTIAL_RESIDUE"
386:     elif (
387:         cut500_mean < 0.12
388:         and cut2000_mean < 0.12
389:         and residue_fraction_2000 < 0.25
390:         and late_mean >= 0.20
391:         and (anchor_full_mean < 0.01 or late_mean >= 0.50 * anchor_full_mean)
392:     ):
393:         verdict = "PASS_REGULARIZATION"
394:     elif (
395:         cut50_mean < 0.10
396:         and cut500_mean < 0.15
397:         and cut2000_mean >= 0.20
398:         and all(g > 0 for g in cut2000_gaps)
399:     ):
400:         verdict = "PASS_EARLY_WINDOW"
401:     else:
402:         if not replication_gate and max(init_mean, anchor_full_mean, cut2000_mean) < 0.10:
403:             verdict = "FAIL_REPLICATION"
404:         elif late_mean < 0.10 and cut2000_mean < 0.10:
405:             verdict = "FAIL_TIMING_AMBIGUOUS"
406:         elif any(g <= 0 for g in cut2000_gaps) and any(g > 0 for g in cut2000_gaps):
407:             verdict = "FAIL_NOISY"
408:         else:
409:             verdict = "FAIL"
410: 
411:     return {
412:         "status": "complete",
413:         "verdict": verdict,
414:         "replication_gate_passed": replication_gate,
415:         "init_only_mean_gain": init_mean,
416:         "init_only_per_seed": init_gaps,
417:         "anchor_only_full_mean_gain": anchor_full_mean,
418:         "anchor_only_full_per_seed": anchor_full_gaps,
419:         "init_anchor_full_mean_gain": init_anchor_mean,
420:         "init_anchor_full_per_seed": init_anchor_gaps,
421:         "cutoff_50_mean_gain": cut50_mean,
422:         "cutoff_50_per_seed": cut50_gaps,
423:         "cutoff_500_mean_gain": cut500_mean,
424:         "cutoff_500_per_seed": cut500_gaps,
425:         "cutoff_2000_mean_gain": cut2000_mean,
426:         "cutoff_2000_per_seed": cut2000_gaps,
427:         "residue_fraction_2000": residue_fraction_2000,
428:         "late_anchor_only_2000_mean_gain": late_mean,
429:         "late_anchor_only_2000_per_seed": late_gaps,
430:         "orthogonal_scaffold_full_mean_gain": ortho_mean,
431:         "orthogonal_scaffold_full_per_seed": ortho_gaps,
432:         "cov_scaffold_full_mean_gain": cov_mean,
433:         "cov_scaffold_full_per_seed": cov_gaps,
434:         "control_max_gain": control_max_gain,
435:         "scaffold_alt": scaffold_alt,
436:     }
437: 
438: 
439: # ---------- Main ----------
440: 
441: def main() -> None:
442:     parser = argparse.ArgumentParser()
443:     parser.add_argument("--smoke", action="store_true")
444:     parser.add_argument("--no-resume", action="store_true")
445:     parser.add_argument("--surface", type=str, default=None,
446:                         choices=["input", "output", "both", "tied"],
447:                         help="Override g195 surface: input/output/both/tied")
448:     parser.add_argument("--tied", action="store_true", default=False,
449:                         help="Force tied fallback branch")
450:     args = parser.parse_args()
451: 
452:     smoke = args.smoke
453:     n_steps = 50 if smoke else TRAIN_STEPS
454:     seeds = [42] if smoke else SEEDS
455:     run_out_path = OUT_PATH.with_name(OUT_PATH.stem + "_smoke.json") if smoke else OUT_PATH
456: 
457:     # Determine surface from g195 result or CLI override
458:     g195_path = ROOT / "results" / "genome_195_untied_input_output_factorial.json"
459:     surface = args.surface
460:     use_tied = args.tied
461:     if surface == "tied":
462:         use_tied = True
463: 
464:     if surface is None and not use_tied:
465:         if g195_path.exists():
466:             g195 = json.loads(g195_path.read_text(encoding="utf-8"))
467:             g195_verdict = g195.get("verdict", "INCOMPLETE")
468:             print_flush(f"  g195 verdict: {g195_verdict}")
469: 
470:             verdict_to_surface = {
471:                 "PASS_INPUT": "input",
472:                 "PASS_INPUT_DOMINANT": "input",
473:                 "PASS_OUTPUT": "output",
474:                 "PASS_OUTPUT_DOMINANT": "output",
475:                 "PASS_BOTH_NEEDED": "both",
476:             }
477:             if g195_verdict in verdict_to_surface:
478:                 surface = verdict_to_surface[g195_verdict]
479:                 use_tied = False
480:             elif g195_verdict in ("AMBIGUOUS_POSITIVE", "FAIL"):
481:                 tied_gain = g195.get("summary", {}).get("tied_mean_gain", 0)
482:                 if tied_gain >= 0.30:
483:                     print_flush(f"  g195 ambiguous but tied_reference gain={tied_gain:+.3f} >= 0.30 => tied fallback")
484:                     use_tied = True
485:                     surface = "tied"
486:                 else:
487:                     print_flush(f"  g195 FAIL and tied_reference gain={tied_gain:+.3f} < 0.30 => ABORT g196")
488:                     print_flush("  Cannot launch g196 without g195 signal. Exiting.")
489:                     return
490:             else:
491:                 print_flush(f"  g195 verdict={g195_verdict} not recognized. Waiting for g195 to complete.")
492:                 return
493:         else:
494:             print_flush("  g195 result not found. Cannot determine surface. Use --surface or --tied.")
495:             return
496: 
497:     if surface is None:
498:         surface = "tied"
499:     if use_tied:
500:         surface = "tied"
501: 
502:     print_flush(f"=== g196 Anchor-Residue Factorial ===")
503:     print_flush(f"  smoke={smoke}, steps={n_steps}, seeds={seeds}")
504:     print_flush(f"  surface={surface}, tied={use_tied}")
505: 
506:     from transformers import AutoTokenizer, AutoModelForCausalLM
507:     tok_qwen = AutoTokenizer.from_pretrained(g188.QWEN_MODEL_ID)
508:     tok_gpt2 = AutoTokenizer.from_pretrained(g188.GPT2_MODEL_ID)
509:     if tok_gpt2.pad_token is None:
510:         tok_gpt2.pad_token = tok_gpt2.eos_token
511: 
512:     print_flush("\n--- Loading data ---")
513:     train_ids, train_mask, _ = g167.load_c4_windows(
514:         tok_gpt2, split="train", seed=g188.C4_TRAIN_SEED, n_windows=g188.N_TRAIN_WINDOWS,
515:     )
516:     train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
517:     val_ids, val_mask, _ = g167.load_c4_windows(
518:         tok_gpt2, split="train", seed=g188.C4_VAL_SEED, n_windows=g188.N_C4_VAL_WINDOWS,
519:         forbidden_hashes=train_hashes,
520:     )
521:     print_flush(f"  Train: {train_ids.shape}, Val: {val_ids.shape}")
522: 
523:     print_flush("\n--- Loading Qwen3 trained embeddings ---")
524:     qwen_model = AutoModelForCausalLM.from_pretrained(g188.QWEN_MODEL_ID, torch_dtype=torch.float32)
525:     trained_embed = qwen_model.model.embed_tokens.weight.detach().cpu().numpy()
526:     trained_fro = float(np.linalg.norm(trained_embed, "fro"))
527:     del qwen_model
528:     cleanup_cuda()
529:     print_flush(f"  Trained embed: {trained_embed.shape}, Fro={trained_fro:.1f}")
530: 
531:     print_flush("\n--- Building string-match base embeddings ---")
532:     gpt2_vocab = len(tok_gpt2)
533:     embed_dim = trained_embed.shape[1]
534:     full_embed, matched_mask = g191.build_string_match_with_mask(
535:         tok_qwen, tok_gpt2, trained_embed, gpt2_vocab, embed_dim,
536:     )
537:     full_embed = g188.normalize_to_fro_norm(full_embed, trained_fro)
538: 
539:     n_matched = int(matched_mask.sum())
540:     matched_fro = float(np.linalg.norm(full_embed[matched_mask], "fro"))
541:     print_flush(f"  Matched: {n_matched}, matched_fro={matched_fro:.2f}")
542: 
543:     # g194 direction-only target: correct directions, uniform norm, Fro-normalized
544:     from genome_194_scalar_direction_factorial import decompose_rows, build_correct_dir_uniform_norm
545:     norms, unit_dirs = decompose_rows(full_embed, matched_mask)
546:     primary_target = build_correct_dir_uniform_norm(unit_dirs, norms, matched_mask)
547:     primary_target = g188.normalize_to_fro_norm(primary_target, matched_fro)
548:     print_flush(f"  Primary target (cd_un) Fro={float(np.linalg.norm(primary_target[matched_mask], 'fro')):.2f}")
549: 
550:     # Scaffold controls
551:     print_flush("\n--- Building scaffold controls ---")
552:     ortho_target = build_orthogonal_scaffold(primary_target, matched_mask)
553:     ortho_target = g188.normalize_to_fro_norm(ortho_target, matched_fro)
554:     print_flush(f"  Orthogonal scaffold Fro={float(np.linalg.norm(ortho_target[matched_mask], 'fro')):.2f}")
555: 
556:     cov_target = build_covariance_scaffold(primary_target, matched_mask)
557:     print_flush(f"  Covariance scaffold Fro={float(np.linalg.norm(cov_target[matched_mask], 'fro')):.2f}")
558: 
559:     # Arm configurations
560:     # embed_init / lm_head_init: what to inject at step 0
561:     # anchor_target: the target matrix for the anchor loss term
562:     # anchor_mask: which rows to apply anchor to
563:     # For cutoff/late arms, the schedule is handled dynamically by get_anchor_lambda()
564:     def init_for_surface(target):
565:         if surface == "input":
566:             return target, None
567:         elif surface == "output":
568:             return None, target
569:         elif surface == "both":
570:             return target, target
571:         else:  # tied
572:             return target, None
573: 
574:     primary_embed_init, primary_lm_head_init = init_for_surface(primary_target)
575: 
576:     arm_configs = {
577:         "scratch": {
578:             "embed_init": None, "lm_head_init": None,
579:             "anchor_target": None, "anchor_mask": None,
580:         },
581:         "init_only": {
582:             "embed_init": primary_embed_init, "lm_head_init": primary_lm_head_init,
583:             "anchor_target": None, "anchor_mask": matched_mask,
584:         },
585:         "anchor_only_full": {
586:             "embed_init": None, "lm_head_init": None,
587:             "anchor_target": primary_target, "anchor_mask": matched_mask,
588:         },
589:         "init_anchor_full": {
590:             "embed_init": primary_embed_init, "lm_head_init": primary_lm_head_init,
591:             "anchor_target": primary_target, "anchor_mask": matched_mask,
592:         },
593:         "cutoff_50": {
594:             "embed_init": None, "lm_head_init": None,
595:             "anchor_target": primary_target, "anchor_mask": matched_mask,
596:         },
597:         "cutoff_500": {
598:             "embed_init": None, "lm_head_init": None,
599:             "anchor_target": primary_target, "anchor_mask": matched_mask,
600:         },
601:         "cutoff_2000": {
602:             "embed_init": None, "lm_head_init": None,
603:             "anchor_target": primary_target, "anchor_mask": matched_mask,
604:         },
605:         "late_anchor_only_2000": {
606:             "embed_init": None, "lm_head_init": None,
607:             "anchor_target": primary_target, "anchor_mask": matched_mask,
608:         },
609:         "orthogonal_scaffold_full": {
610:             "embed_init": None, "lm_head_init": None,
611:             "anchor_target": ortho_target, "anchor_mask": matched_mask,
612:         },
613:         "cov_scaffold_full": {
614:             "embed_init": None, "lm_head_init": None,
615:             "anchor_target": cov_target, "anchor_mask": matched_mask,
616:         },
617:     }
618: 
619:     if not args.no_resume and run_out_path.exists():
620:         payload = json.loads(run_out_path.read_text(encoding="utf-8"))
621:         prev_cfg = payload.get("config", {})
622:         if (prev_cfg.get("surface") != surface
623:                 or prev_cfg.get("tied") != use_tied
624:                 or prev_cfg.get("train_steps") != n_steps
625:                 or prev_cfg.get("seeds") != seeds):
626:             print_flush(f"  WARNING: resume config mismatch (prev surface={prev_cfg.get('surface')}, "
627:                         f"tied={prev_cfg.get('tied')}, steps={prev_cfg.get('train_steps')}). "
628:                         f"Starting fresh.")
629:             payload = None
630:     else:
631:         payload = None
632: 
633:     if payload is None:
634:         payload = {
635:             "genome": 196,
636:             "name": "anchor_residue_factorial",
637:             "timestamp_utc_started": now_utc(),
638:             "config": {
639:                 "train_steps": n_steps,
640:                 "seeds": seeds,
641:                 "anchor_lambda": ANCHOR_LAMBDA,
642:                 "n_matched": n_matched,
643:                 "trained_fro": trained_fro,
644:                 "matched_fro": matched_fro,
645:                 "surface": surface,
646:                 "tied": use_tied,
647:                 "scaffold_seed_ortho": SCAFFOLD_SEED_ORTHO,
648:                 "scaffold_seed_cov": SCAFFOLD_SEED_COV,
649:             },
650:             "results": {},
651:             "summary": {},
652:             "verdict": "INCOMPLETE",
653:         }
654: 
655:     t_start = time.time()
656: 
657:     def save():
658:         payload["timestamp_utc_last_write"] = now_utc()
659:         payload["elapsed_s"] = time.time() - t_start
660:         tmp = run_out_path.with_suffix(run_out_path.suffix + ".tmp")
661:         tmp.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")
662:         os.replace(tmp, run_out_path)
663: 
664:     for arm_label in ARMS:
665:         payload["results"].setdefault(arm_label, {})
666:         cfg = arm_configs[arm_label]
667: 
668:         for seed in seeds:
669:             key = str(seed)
670:             if key in payload["results"][arm_label] and not args.no_resume:
671:                 cell = payload["results"][arm_label][key]
672:                 if isinstance(cell, dict) and "final_val_nll" in cell:
673:                     print_flush(f"\n  Skipping {arm_label}/seed={seed} (done)")
674:                     continue
675: 
676:             print_flush(f"\n  === {arm_label} seed={seed} ===")
677:             result = train_cell(
678:                 arm_label=arm_label,
679:                 seed=seed,
680:                 tok_gpt2=tok_gpt2,
681:                 embed_init=cfg["embed_init"],
682:                 lm_head_init=cfg["lm_head_init"],
683:                 anchor_target=cfg["anchor_target"],
684:                 anchor_mask=cfg["anchor_mask"],
685:                 train_ids=train_ids,
686:                 train_mask=train_mask,
687:                 val_ids=val_ids,
688:                 val_mask=val_mask,
689:                 n_steps=n_steps,
690:                 tied=use_tied,
691:                 surface=surface,
692:             )
693:             payload["results"][arm_label][key] = result
694:             save()
695:             print_flush(f"  {arm_label} seed={seed} nll={result['final_val_nll']:.4f} ({result['wallclock_s']:.0f}s)")
696: 
697:     summary = compute_verdict(payload)
698:     payload["summary"] = summary
699:     payload["verdict"] = summary.get("verdict", "INCOMPLETE")
700:     save()
701: 
702:     print_flush(f"\n*** g196 VERDICT: {summary.get('verdict', '?')} ***")
703:     for key, val in summary.items():
704:         if key.endswith("_mean_gain"):
705:             print_flush(f"  {key}: {val:+.4f}")
706:     if "residue_fraction_2000" in summary:
707:         print_flush(f"  residue_fraction_2000: {summary['residue_fraction_2000']:.3f}")
708:     print_flush(f"  replication_gate_passed: {summary.get('replication_gate_passed')}")
709:     print_flush(f"  scaffold_alt: {summary.get('scaffold_alt')}")
710: 
711: 
712: if __name__ == "__main__":
713:     main()

codex
The docs establish the live framing: post-g194 score 5.2/10, g195 currently drives g192/g196 gating, and g196 is explicitly post-SEV-fix. I’m checking the implementation-level risks now: resume validity, mask/surface semantics, tied/untied behavior, NaN serialization, and whether g192 fits the advertised full-depth configuration.
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "python -m py_compile code\\genome_192_28layer_replication.py code\\genome_195_untied_input_output_factorial.py code\\genome_196_anchor_residue_factorial.py" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "@'
from pathlib import Path
for p in [Path('code/genome_192_28layer_replication.py'), Path('code/genome_195_untied_input_output_factorial.py'), Path('code/genome_196_anchor_residue_factorial.py')]:
    data = p.read_bytes()
    try:
        data.decode('ascii')
        print(f'{p}: ASCII OK')
    except UnicodeDecodeError as e:
        print(f'{p}: NON-ASCII at byte {e.start}')
'@ | python -" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"def make_gpt2_qwen3_model|QWEN_MODEL_ID|GPT2_MODEL_ID|BATCH_SIZE|FORWARD_DTYPE|SEQ_LEN|LR|GRAD_CLIP|def normalize_to_fro_norm|def _eval_nll\" code\\genome_188_tokenizer_flow_bridge.py; rg -n \"def build_string_match_with_mask|def build_row_shuffled_matched\" code\\genome_191_string_match_decomposition.py; rg -n \"def decompose_rows|def build_correct_dir_uniform_norm\" code\\genome_194_scalar_direction_factorial.py" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines = Get-Content C:'"\\Users\\devan\\.codex\\memories\\MEMORY.md; for ("'$i=314; $i -le 348; $i++) { '"'{0}: {1}' -f "'$i,$lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "git diff -- code\\genome_196_anchor_residue_factorial.py code\\genome_192_28layer_replication.py code\\genome_195_untied_input_output_factorial.py; git status --short" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
 succeeded in 326ms:
 succeeded in 321ms:
45:SEQ_LEN = g165.SEQ_LEN
46:BATCH_SIZE = g165.BATCH_SIZE
50:LR = g165.LR
53:GRAD_CLIP = 1.0
61:FORWARD_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
69:QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"
70:GPT2_MODEL_ID = "openai-community/gpt2"
301:def normalize_to_fro_norm(embeddings: np.ndarray, target_norm: float) -> np.ndarray:
311:def make_gpt2_qwen3_model(tok_gpt2, seed: int):
324:        max_position_embeddings=SEQ_LEN + 64,
397:        model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
405:        idx = torch.randint(0, n_train, (BATCH_SIZE,))
409:        with torch.amp.autocast("cuda", dtype=FORWARD_DTYPE):
421:        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
453:def _eval_nll(model, val_ids, val_mask, batch_size=16) -> float:
461:        with torch.amp.autocast("cuda", dtype=FORWARD_DTYPE):
595:            "source_model_id": QWEN_MODEL_ID,
596:            "target_tokenizer_id": GPT2_MODEL_ID,
627:    tok_qwen = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
628:    tok_gpt2 = AutoTokenizer.from_pretrained(GPT2_MODEL_ID)
83:def build_string_match_with_mask(
125:def build_row_shuffled_matched(full_embed: np.ndarray, matched_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
81:def decompose_rows(embed: np.ndarray, matched_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
127:def build_correct_dir_uniform_norm(

 succeeded in 317ms:
314: # Task Group: moonshot-llm-genome repo assessment and genome-forecast direction
315: scope: candid assessment of the current repo state plus the concrete "DeepSeek moment" subproject direction that emerged from the review
316: applies_to: cwd=C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome; reuse_rule=safe for this repo family while the trained-text activation-spectrum line remains active; re-check docs and claim map if the narrative shifts again
317: 
318: ## Task 1: Review `moonshot-llm-genome` honestly and inspect the newest untracked `genome_109` work
319: 
320: ### rollout_summary_files
321: 
322: - rollout_summaries/2026-04-23T08-16-20-m9E8-honest_assessment_and_deepseek_style_genome_forecast_subproj.md (cwd=\\?\C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome, rollout_path=C:\Users\devan\.codex\sessions\2026\04\23\rollout-2026-04-23T04-16-20-019db969-52ed-7ea2-8a05-5e42846ceb8c.jsonl, updated_at=2026-04-23T08:41:03+00:00, thread_id=019db969-52ed-7ea2-8a05-5e42846ceb8c, success; narrowed the strongest surviving claim)
323: 
324: ### keywords
325: 
326: - honest conversation, activation-spectrum, trained-text, WIKI.md, PAPER.md, CLAIM_EVIDENCE_MAP.md, genome_109_functional_depth.py, functional_depth.json, prereg_validator.py, normalized_depth
327: 
328: ## Task 2: Define a practical "DeepSeek moment" subproject for this repo
329: 
330: ### rollout_summary_files
331: 
332: - rollout_summaries/2026-04-23T08-16-20-m9E8-honest_assessment_and_deepseek_style_genome_forecast_subproj.md (cwd=\\?\C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome, rollout_path=C:\Users\devan\.codex\sessions\2026\04\23\rollout-2026-04-23T04-16-20-019db969-52ed-7ea2-8a05-5e42846ceb8c.jsonl, updated_at=2026-04-23T08:41:03+00:00, thread_id=019db969-52ed-7ea2-8a05-5e42846ceb8c, success; proposed "Genome Forecast" as the first practical wedge)
333: 
334: ### keywords
335: 
336: - DeepSeek moment, Genome Forecast, early-training prediction, checkpoint cadence, bad-run detection, compute savings, genome_forecast_probe.py, genome_forecast_predict.py, AUROC, early loss alone
337: 
338: ## User preferences
339: 
340: - when reviewing research repos, the user asked for "an honest conversation" and to "take things up a notch" by thinking through "what's working what's not working" -> default to candid critique and strategic analysis, not promotional framing [Task 1]
341: - when the user said they wanted a "deep seek moment" and later asked "How would we implement that in practice" -> pivot from theory to a concrete, economically meaningful subproject with an execution shape and acceptance criteria [Task 2]
342: 
343: ## Reusable knowledge
344: 
345: - The current strongest surviving claim is the trained-text activation-spectrum attractor: `sqrt(er) * alpha = f(normalized_depth)` in the mid-band under natural-text-like probes. The repo's strongest evidence now lives in the adversarial-correction arc around `genome_097` through `genome_108` [Task 1]
346: - Current repo-story mismatch: `README.md` and `PAPER.md` still emphasize older kNN/biology/Level-1 framing, while `WIKI.md` has shifted to the narrower trained-text spectral story. `research/CLAIM_EVIDENCE_MAP.md` being stale is a credibility leak [Task 1]
347: - `genome_109_functional_depth.py` compiled and `results/gate2/functional_depth.json` was valid JSON, but the artifact was not audit-complete because it persisted raw arrays without a full verdict/ledger entry [Task 1]
348: - The strongest practical next wedge is "Genome Forecast": use early activation-spectrum geometry to predict final run quality, bad-run risk, or compute-stopping decisions better than early loss alone [Task 2]

 succeeded in 317ms:
 M experiments/heartbeat_cycle.txt
?? results/genome_195_untied_input_output_factorial.json

 succeeded in 355ms:
code\genome_192_28layer_replication.py: ASCII OK
code\genome_195_untied_input_output_factorial.py: ASCII OK
code\genome_196_anchor_residue_factorial.py: ASCII OK

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content code'"\\genome_188_tokenizer_flow_bridge.py; foreach("'$range in @(@(301,340),@(453,476))) { for($i=$range[0]; $i -le $range[1]; $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] } }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content code'"\\genome_191_string_match_decomposition.py; foreach("'$range in @(@(83,153))) { for($i=$range[0]; $i -le $range[1]; $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] } }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content code'"\\genome_194_scalar_direction_factorial.py; foreach("'$range in @(@(81,151))) { for($i=$range[0]; $i -le $range[1]; $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] } }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "if (Test-Path results\\genome_195_untied_input_output_factorial.json) { "'$json=Get-Content -Raw results'"\\genome_195_untied_input_output_factorial.json | ConvertFrom-Json; "'$json.verdict; $json.summary | ConvertTo-Json -Depth 4; foreach($arm in $json.results.PSObject.Properties.Name){ $count=($json.results.$arm.PSObject.Properties | Measure-Object).Count; "$arm $count" } } else { '"'no g195 result' }" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"Qwen3-0.6B|28-layer|28layer|492M|tie_word|head_dim|num_hidden_layers|intermediate_size|num_key_value_heads\" research\\prereg\\genome_192_28layer_replication_2026-04-30.md research\\prereg\\genome_192_28layer_string_match_2026-04-30.md COMPUTE.md code\\genome_192_28layer_replication.py" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
 succeeded in 280ms:
 succeeded in 284ms:
 succeeded in 314ms:
301: def normalize_to_fro_norm(embeddings: np.ndarray, target_norm: float) -> np.ndarray:
302:     """Scale embedding matrix to match target Frobenius norm."""
303:     current = np.linalg.norm(embeddings)
304:     if current < 1e-8:
305:         return embeddings
306:     return embeddings * (target_norm / current)
307: 
308: 
309: # ---------- Training ----------
310: 
311: def make_gpt2_qwen3_model(tok_gpt2, seed: int):
312:     """Create a Qwen3-arch model with GPT-2 vocab size, matching Qwen3-0.6B hidden_size."""
313:     from transformers import Qwen3ForCausalLM
314:     from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
315: 
316:     torch.manual_seed(seed)
317:     cfg = Qwen3Config(
318:         vocab_size=len(tok_gpt2),
319:         hidden_size=1024,
320:         num_hidden_layers=8,
321:         num_attention_heads=16,
322:         num_key_value_heads=4,
323:         intermediate_size=2816,
324:         max_position_embeddings=SEQ_LEN + 64,
325:         rms_norm_eps=1e-6,
326:         tie_word_embeddings=True,
327:         head_dim=64,
328:         rope_theta=10000.0,
329:         use_cache=False,
330:         bos_token_id=tok_gpt2.bos_token_id if tok_gpt2.bos_token_id is not None else tok_gpt2.eos_token_id,
331:         eos_token_id=tok_gpt2.eos_token_id,
332:         pad_token_id=tok_gpt2.pad_token_id if tok_gpt2.pad_token_id is not None else tok_gpt2.eos_token_id,
333:     )
334:     if hasattr(cfg, "_attn_implementation"):
335:         cfg._attn_implementation = "eager"
336:     model = Qwen3ForCausalLM(cfg)
337:     model.tie_weights()
338:     model.to(DEVICE)
339:     if hasattr(model.config, "use_cache"):
340:         model.config.use_cache = False
453: def _eval_nll(model, val_ids, val_mask, batch_size=16) -> float:
454:     """Evaluate mean NLL on validation set."""
455:     total_loss = 0.0
456:     total_tokens = 0
457:     n = val_ids.shape[0]
458:     for i in range(0, n, batch_size):
459:         b_ids = val_ids[i:i+batch_size].to(DEVICE)
460:         b_mask = val_mask[i:i+batch_size].to(DEVICE)
461:         with torch.amp.autocast("cuda", dtype=FORWARD_DTYPE):
462:             out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_ids)
463:         n_tok = b_mask.sum().item()
464:         total_loss += out.loss.item() * n_tok
465:         total_tokens += n_tok
466:     return total_loss / max(total_tokens, 1)
467: 
468: 
469: # ---------- Verdict ----------
470: 
471: def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
472:     """Compute experiment verdict from results."""
473:     results = payload["results"]
474:     stage_a_done = all(
475:         str(s) in results.get(arm, {}) for arm in STAGE_A_ARMS for s in SEEDS
476:     )

 succeeded in 261ms:
code\genome_192_28layer_replication.py:2:genome_192_28layer_replication.py
code\genome_192_28layer_replication.py:5:at full 28-layer depth (vs 8-layer shell). Resolves adversarial A16 #3.
code\genome_192_28layer_replication.py:35:OUT_PATH = ROOT / "results" / "genome_192_28layer_replication.json"
code\genome_192_28layer_replication.py:75:def make_28layer_model(tok_gpt2, seed: int):
code\genome_192_28layer_replication.py:83:        num_hidden_layers=NUM_LAYERS_28,
code\genome_192_28layer_replication.py:85:        num_key_value_heads=8,
code\genome_192_28layer_replication.py:86:        intermediate_size=3072,
code\genome_192_28layer_replication.py:89:        tie_word_embeddings=True,
code\genome_192_28layer_replication.py:90:        head_dim=128,
code\genome_192_28layer_replication.py:99:def train_cell_28layer(
code\genome_192_28layer_replication.py:118:    model = make_28layer_model(tok_gpt2, seed)
code\genome_192_28layer_replication.py:131:            if hasattr(model, "lm_head") and not model.config.tie_word_embeddings:
code\genome_192_28layer_replication.py:315:            "name": "28layer_replication",
code\genome_192_28layer_replication.py:352:            result = train_cell_28layer(
COMPUTE.md:51:- Qwen3-0.6B at FP16: ~1.3 GB
research\prereg\genome_192_28layer_replication_2026-04-30.md:9:This experiment tests whether the embedding init effect persists at full 28-layer Qwen3-0.6B depth.
research\prereg\genome_192_28layer_replication_2026-04-30.md:27:All arms: **28-layer** Qwen3-arch with GPT-2 tokenizer, 5000 steps, same data/eval as g191. Anchor lambda=0.01, masked to matched rows only. Same training hyperparameters (lr, betas, weight_decay, batch_size, grad_clip).
research\prereg\genome_192_28layer_replication_2026-04-30.md:31:`Qwen3Config(vocab_size=50257, hidden_size=1024, num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, intermediate_size=3072, max_position_embeddings=320, tie_word_embeddings=True, head_dim=128, rope_theta=1000000.0, use_cache=False)`
research\prereg\genome_192_28layer_replication_2026-04-30.md:33:Matches actual Qwen3-0.6B config (except vocab_size=50257 for GPT-2 tokenizer). Estimated VRAM: ~4.0 GB (well within 22 GB envelope).
research\prereg\genome_192_28layer_replication_2026-04-30.md:49:- 28-layer Qwen3-arch, GPT-2 tokenizer, ~3.2 GB VRAM per cell
research\prereg\genome_192_28layer_string_match_2026-04-30.md:11:The g191 PASS_CONTENT finding (matched-row content drives the +0.478 cross-tokenizer signal) persists at full model scale (28-layer Qwen3-0.6B architecture), not just the 8-layer shell. This resolves Codex adversarial A16 attack #3 (SEV-8: shallow-init regime).
research\prereg\genome_192_28layer_string_match_2026-04-30.md:16:- Codex cycle 155 adversarial: "8-layer/5000-step may be a shallow-init regime. The result may vanish or shrink in the full 28-layer model where deeper dynamics dominate."
research\prereg\genome_192_28layer_string_match_2026-04-30.md:18:- If 28-layer PASS: §0.1 moves from ~4.3 to ~5.0+ (depth-robust cross-tokenizer bridge)
research\prereg\genome_192_28layer_string_match_2026-04-30.md:19:- If 28-layer FAIL: effect is shell-dependent, not architecture-general
research\prereg\genome_192_28layer_string_match_2026-04-30.md:21:## Arms (3 total, 28-layer Qwen3-0.6B with GPT-2 tokenizer)
research\prereg\genome_192_28layer_string_match_2026-04-30.md:38:- 28-layer Qwen3-0.6B config (full model, NOT 8-layer shell)
research\prereg\genome_192_28layer_string_match_2026-04-30.md:54:- [ ] Max VRAM: ~5-6 GB (full 28-layer Qwen3-0.6B). Under 22 GB.
research\prereg\genome_192_28layer_string_match_2026-04-30.md:63:- Code: `code/genome_192_28layer_string_match.py` (to be written after g191 verdict)
research\prereg\genome_192_28layer_string_match_2026-04-30.md:64:- Results: `results/genome_192_28layer_string_match.json`

 succeeded in 393ms:
INCOMPLETE
{

}
scratch_untied 3
input_inject_anchor 3
output_inject_anchor 3
both_inject_anchor 3
tied_reference 1

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content code'"\\genome_191_string_match_decomposition.py; "'$count=$lines.Count; "count=$count"; for($i=70; $i -le [Math]::Min(160,$count); $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content code'"\\genome_194_scalar_direction_factorial.py; "'$count=$lines.Count; "count=$count"; for($i=70; $i -le [Math]::Min(160,$count); $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "@'
# exact param estimate for Qwen3 config in g192, tied embeddings, ignoring zero-param activations
vocab=50257; h=1024; layers=28; heads=16; kv=8; head_dim=128; inter=3072
embed = vocab*h
q = h*(heads*head_dim)
k = h*(kv*head_dim)
v = h*(kv*head_dim)
o = (heads*head_dim)*h
mlp = h*inter*2 + inter*h
norms = layers*2*h + h # input/post norms rough
per_layer = q+k+v+o+mlp + 4 # biases? Qwen usually no bias, negligible
# Qwen3 decoder also q_norm/k_norm per layer? small: heads*head_dim + kv*head_dim maybe
attn_norms = layers*((heads*head_dim)+(kv*head_dim))
total = embed + layers*per_layer + norms + attn_norms
print('embed', embed)
print('per_layer', per_layer)
print('total', total, 'M', total/1e6)
print('bf16 weights GiB', total*2/1024**3)
print('AdamW states+grads approx GiB fp32 states bf16 weights/grads', (total*2 + total*2 + total*4*2)/1024**3)
'@ | python -" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"torch\\.amp\\.autocast\\(\\\"cuda\\\"|allow_nan=False|json\\.loads|read_text\\(|write_text\\(|multivariate_normal|np\\.cov|np\\.linalg|torch\\.from_numpy|tie_weights|tie_word_embeddings\" code\\genome_192_28layer_replication.py code\\genome_195_untied_input_output_factorial.py code\\genome_196_anchor_residue_factorial.py" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
 succeeded in 284ms:
count=359
70:     return datetime.now(timezone.utc).isoformat()
71: 
72: 
73: def cleanup_cuda() -> None:
74:     gc.collect()
75:     if torch.cuda.is_available():
76:         torch.cuda.empty_cache()
77: 
78: 
79: # ---------- Decomposition helpers ----------
80: 
81: def decompose_rows(embed: np.ndarray, matched_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
82:     """Decompose matched rows into norms (r_t) and unit directions (u_t)."""
83:     norms = np.linalg.norm(embed, axis=1, keepdims=True)
84:     norms = np.maximum(norms, 1e-8)
85:     unit_dirs = embed / norms
86:     return norms.squeeze(1), unit_dirs
87: 
88: 
89: def build_correct_dir_shuffled_norm(
90:     unit_dirs: np.ndarray, norms: np.ndarray, matched_mask: np.ndarray, rng: np.random.Generator,
91: ) -> np.ndarray:
92:     """Correct directions, shuffled norms among matched tokens."""
93:     out = np.zeros_like(unit_dirs)
94:     matched_ids = np.where(matched_mask)[0]
95:     perm = rng.permutation(len(matched_ids))
96:     shuffled_norms = norms[matched_ids][perm]
97:     out[matched_ids] = unit_dirs[matched_ids] * shuffled_norms[:, None]
98:     return out
99: 
100: 
101: def build_shuffled_dir_correct_norm(
102:     unit_dirs: np.ndarray, norms: np.ndarray, matched_mask: np.ndarray, rng: np.random.Generator,
103: ) -> np.ndarray:
104:     """Shuffled directions, correct norms per token."""
105:     out = np.zeros_like(unit_dirs)
106:     matched_ids = np.where(matched_mask)[0]
107:     perm = rng.permutation(len(matched_ids))
108:     shuffled_dirs = unit_dirs[matched_ids][perm]
109:     out[matched_ids] = shuffled_dirs * norms[matched_ids, None]
110:     return out
111: 
112: 
113: def build_random_dir_correct_norm(
114:     norms: np.ndarray, matched_mask: np.ndarray, embed_dim: int, rng: np.random.Generator,
115: ) -> np.ndarray:
116:     """Random unit directions, correct norms."""
117:     out = np.zeros((len(matched_mask), embed_dim), dtype=np.float32)
118:     matched_ids = np.where(matched_mask)[0]
119:     random_vecs = rng.standard_normal((len(matched_ids), embed_dim)).astype(np.float32)
120:     random_norms = np.linalg.norm(random_vecs, axis=1, keepdims=True)
121:     random_norms = np.maximum(random_norms, 1e-8)
122:     random_unit = random_vecs / random_norms
123:     out[matched_ids] = random_unit * norms[matched_ids, None]
124:     return out
125: 
126: 
127: def build_correct_dir_uniform_norm(
128:     unit_dirs: np.ndarray, norms: np.ndarray, matched_mask: np.ndarray,
129: ) -> np.ndarray:
130:     """Correct directions, uniform (mean) norm."""
131:     out = np.zeros_like(unit_dirs)
132:     matched_ids = np.where(matched_mask)[0]
133:     mean_norm = float(norms[matched_ids].mean())
134:     out[matched_ids] = unit_dirs[matched_ids] * mean_norm
135:     return out
136: 
137: 
138: # ---------- Main ----------
139: 
140: def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
141:     results = payload.get("results", {})
142:     required = ["scratch_ce", "full_match", "correct_dir_shuffled_norm",
143:                  "shuffled_dir_correct_norm", "random_dir_correct_norm",
144:                  "correct_dir_uniform_norm"]
145:     if not all(arm in results and len(results[arm]) >= len(SEEDS) for arm in required):
146:         return {"status": "incomplete"}
147: 
148:     scratch_nlls = {str(s): float(results["scratch_ce"][str(s)]["final_val_nll"]) for s in SEEDS}
149: 
150:     def arm_stats(arm_name):
151:         nlls = {str(s): float(results[arm_name][str(s)]["final_val_nll"]) for s in SEEDS}
152:         gaps = [scratch_nlls[str(s)] - nlls[str(s)] for s in SEEDS]
153:         return float(np.mean(gaps)), gaps
154: 
155:     full_mean, full_gaps = arm_stats("full_match")
156:     cd_sn_mean, cd_sn_gaps = arm_stats("correct_dir_shuffled_norm")
157:     sd_cn_mean, sd_cn_gaps = arm_stats("shuffled_dir_correct_norm")
158:     rd_cn_mean, rd_cn_gaps = arm_stats("random_dir_correct_norm")
159:     cd_un_mean, cd_un_gaps = arm_stats("correct_dir_uniform_norm")
160: 

 succeeded in 292ms:
count=474
70: 
71: def now_utc() -> str:
72:     return datetime.now(timezone.utc).isoformat()
73: 
74: 
75: def cleanup_cuda() -> None:
76:     gc.collect()
77:     if torch.cuda.is_available():
78:         torch.cuda.empty_cache()
79: 
80: 
81: # ---------- Embedding construction ----------
82: 
83: def build_string_match_with_mask(
84:     tok_src, tok_tgt, src_embeddings: np.ndarray, tgt_vocab_size: int, tgt_embed_dim: int,
85: ) -> tuple[np.ndarray, np.ndarray]:
86:     """Like g188's direct_string_match_embeddings but also returns matched_mask."""
87:     src_vocab = tok_src.get_vocab()
88:     tgt_vocab = tok_tgt.get_vocab()
89: 
90:     result = np.zeros((tgt_vocab_size, tgt_embed_dim), dtype=np.float32)
91:     matched = np.zeros(tgt_vocab_size, dtype=bool)
92: 
93:     for token_str, tgt_id in tgt_vocab.items():
94:         if tgt_id >= tgt_vocab_size:
95:             continue
96:         if token_str in src_vocab:
97:             src_id = src_vocab[token_str]
98:             if src_id < src_embeddings.shape[0]:
99:                 result[tgt_id] = src_embeddings[src_id]
100:                 matched[tgt_id] = True
101: 
102:     if matched.any():
103:         mean_emb = result[matched].mean(axis=0)
104:         result[~matched] = mean_emb
105: 
106:     n_matched = int(matched.sum())
107:     print_flush(f"  String match: {n_matched}/{tgt_vocab_size} tokens matched ({100*n_matched/tgt_vocab_size:.1f}%)")
108:     return result, matched
109: 
110: 
111: def build_matched_rows_only(full_embed: np.ndarray, matched_mask: np.ndarray) -> np.ndarray:
112:     """Keep matched rows from full_embed; zero-fill unmatched."""
113:     out = np.zeros_like(full_embed)
114:     out[matched_mask] = full_embed[matched_mask]
115:     return out
116: 
117: 
118: def build_unmatched_rows_only(full_embed: np.ndarray, matched_mask: np.ndarray) -> np.ndarray:
119:     """Keep unmatched rows (mean-filled in full_embed); zero matched."""
120:     out = np.zeros_like(full_embed)
121:     out[~matched_mask] = full_embed[~matched_mask]
122:     return out
123: 
124: 
125: def build_row_shuffled_matched(full_embed: np.ndarray, matched_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
126:     """Permute rows among matched tokens only, preserving norms/spectrum."""
127:     out = full_embed.copy()
128:     matched_ids = np.where(matched_mask)[0]
129:     matched_rows = out[matched_ids].copy()
130:     perm = rng.permutation(len(matched_ids))
131:     out[matched_ids] = matched_rows[perm]
132:     return out
133: 
134: 
135: def build_frequency_bucket_shuffle(
136:     full_embed: np.ndarray, matched_mask: np.ndarray, tok_tgt, train_ids: torch.Tensor, rng: np.random.Generator,
137: ) -> np.ndarray:
138:     """Shuffle matched rows within frequency quintiles."""
139:     token_counts = Counter(train_ids.reshape(-1).tolist())
140:     matched_ids = np.where(matched_mask)[0]
141:     freqs = np.array([token_counts.get(int(tid), 0) for tid in matched_ids])
142:     quintiles = np.quantile(freqs, [0.2, 0.4, 0.6, 0.8])
143: 
144:     out = full_embed.copy()
145:     for lo, hi in zip(
146:         [0.0] + quintiles.tolist(),
147:         quintiles.tolist() + [float("inf")],
148:     ):
149:         bucket = matched_ids[(freqs >= lo) & (freqs < hi)]
150:         if len(bucket) > 1:
151:             rows = out[bucket].copy()
152:             perm = rng.permutation(len(bucket))
153:             out[bucket] = rows[perm]
154:     return out
155: 
156: 
157: # ---------- Training with row-wise anchor masking ----------
158: 
159: def train_cell(
160:     arm_label: str,

2026-04-30T19:22:19.847482Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.2 seconds
Output:
rg: regex parse error:
    (?:torch\.amp\.autocast\(\)
    ^
error: unclosed group

 exited 1 in 244ms:
rg: regex parse error:
    (?:torch\.amp\.autocast\(\)
    ^
error: unclosed group

 succeeded in 292ms:
embed 51463168
per_layer 15728644
total 492009584 M 492.009584
bf16 weights GiB 0.9164392650127411
AdamW states+grads approx GiB fp32 states bf16 weights/grads 5.4986355900764465

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"SEQ_LEN|BATCH_SIZE|N_TRAIN_WINDOWS|N_C4_VAL_WINDOWS|LR|BETAS|WEIGHT_DECAY|DEVICE\" code\\genome_165_annealed_donor.py code\\genome_167_kd_canonical.py code\\genome_188_tokenizer_flow_bridge.py" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"Qwen3ForCausalLM|Qwen3Config|_attn_implementation|autocast|FORWARD_DTYPE|torch_dtype|trust_remote_code\" code\\genome_192_28layer_replication.py code\\genome_195_untied_input_output_factorial.py code\\genome_196_anchor_residue_factorial.py code\\genome_188_tokenizer_flow_bridge.py" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Get-Content -Raw results\\genome_195_untied_input_output_factorial.json | Select-Object -First 1" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
 succeeded in 251ms:
code\genome_195_untied_input_output_factorial.py:79:    from transformers import Qwen3ForCausalLM
code\genome_195_untied_input_output_factorial.py:80:    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
code\genome_195_untied_input_output_factorial.py:83:    cfg = Qwen3Config(
code\genome_195_untied_input_output_factorial.py:97:    model = Qwen3ForCausalLM(cfg).to(device=DEVICE)
code\genome_195_untied_input_output_factorial.py:181:        with torch.amp.autocast("cuda", dtype=g188.FORWARD_DTYPE):
code\genome_195_untied_input_output_factorial.py:327:    qwen_model = AutoModelForCausalLM.from_pretrained(g188.QWEN_MODEL_ID, torch_dtype=torch.float32)
code\genome_188_tokenizer_flow_bridge.py:61:FORWARD_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
code\genome_188_tokenizer_flow_bridge.py:313:    from transformers import Qwen3ForCausalLM
code\genome_188_tokenizer_flow_bridge.py:314:    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
code\genome_188_tokenizer_flow_bridge.py:317:    cfg = Qwen3Config(
code\genome_188_tokenizer_flow_bridge.py:334:    if hasattr(cfg, "_attn_implementation"):
code\genome_188_tokenizer_flow_bridge.py:335:        cfg._attn_implementation = "eager"
code\genome_188_tokenizer_flow_bridge.py:336:    model = Qwen3ForCausalLM(cfg)
code\genome_188_tokenizer_flow_bridge.py:409:        with torch.amp.autocast("cuda", dtype=FORWARD_DTYPE):
code\genome_188_tokenizer_flow_bridge.py:461:        with torch.amp.autocast("cuda", dtype=FORWARD_DTYPE):
code\genome_188_tokenizer_flow_bridge.py:627:    tok_qwen = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
code\genome_188_tokenizer_flow_bridge.py:667:        ds = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
code\genome_196_anchor_residue_factorial.py:184:        from transformers import Qwen3ForCausalLM
code\genome_196_anchor_residue_factorial.py:185:        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
code\genome_196_anchor_residue_factorial.py:187:        cfg = Qwen3Config(
code\genome_196_anchor_residue_factorial.py:201:        model = Qwen3ForCausalLM(cfg).to(device=DEVICE)
code\genome_196_anchor_residue_factorial.py:261:        with torch.amp.autocast("cuda", dtype=g188.FORWARD_DTYPE):
code\genome_196_anchor_residue_factorial.py:524:    qwen_model = AutoModelForCausalLM.from_pretrained(g188.QWEN_MODEL_ID, torch_dtype=torch.float32)
code\genome_192_28layer_replication.py:76:    from transformers import Qwen3ForCausalLM
code\genome_192_28layer_replication.py:77:    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
code\genome_192_28layer_replication.py:80:    cfg = Qwen3Config(
code\genome_192_28layer_replication.py:94:    model = Qwen3ForCausalLM(cfg).to(dtype=g188.FORWARD_DTYPE, device=DEVICE)
code\genome_192_28layer_replication.py:160:        with torch.amp.autocast("cuda", dtype=g188.FORWARD_DTYPE):
code\genome_192_28layer_replication.py:282:    qwen_model = AutoModelForCausalLM.from_pretrained(g188.QWEN_MODEL_ID, torch_dtype=torch.float32)

 succeeded in 258ms:
code\genome_188_tokenizer_flow_bridge.py:45:SEQ_LEN = g165.SEQ_LEN
code\genome_188_tokenizer_flow_bridge.py:46:BATCH_SIZE = g165.BATCH_SIZE
code\genome_188_tokenizer_flow_bridge.py:48:N_TRAIN_WINDOWS = 16384
code\genome_188_tokenizer_flow_bridge.py:49:N_C4_VAL_WINDOWS = 256
code\genome_188_tokenizer_flow_bridge.py:50:LR = g165.LR
code\genome_188_tokenizer_flow_bridge.py:51:BETAS = (0.9, 0.95)
code\genome_188_tokenizer_flow_bridge.py:52:WEIGHT_DECAY = 0.01
code\genome_188_tokenizer_flow_bridge.py:60:DEVICE = g165.DEVICE
code\genome_188_tokenizer_flow_bridge.py:324:        max_position_embeddings=SEQ_LEN + 64,
code\genome_188_tokenizer_flow_bridge.py:338:    model.to(DEVICE)
code\genome_188_tokenizer_flow_bridge.py:392:                    (param_dict[param_name], target_tensor.to(DEVICE, dtype=param_dict[param_name].dtype))
code\genome_188_tokenizer_flow_bridge.py:397:        model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
code\genome_188_tokenizer_flow_bridge.py:405:        idx = torch.randint(0, n_train, (BATCH_SIZE,))
code\genome_188_tokenizer_flow_bridge.py:406:        batch_ids = train_ids[idx].to(DEVICE)
code\genome_188_tokenizer_flow_bridge.py:407:        batch_mask = train_mask[idx].to(DEVICE)
code\genome_188_tokenizer_flow_bridge.py:459:        b_ids = val_ids[i:i+batch_size].to(DEVICE)
code\genome_188_tokenizer_flow_bridge.py:460:        b_mask = val_mask[i:i+batch_size].to(DEVICE)
code\genome_188_tokenizer_flow_bridge.py:597:            "device": str(DEVICE),
code\genome_188_tokenizer_flow_bridge.py:758:        tok_gpt2, split="train", seed=C4_TRAIN_SEED, n_windows=N_TRAIN_WINDOWS,
code\genome_188_tokenizer_flow_bridge.py:762:        tok_gpt2, split="train", seed=C4_VAL_SEED, n_windows=N_C4_VAL_WINDOWS,
code\genome_165_annealed_donor.py:58:SEQ_LEN = 256
code\genome_165_annealed_donor.py:59:N_TRAIN_TOKENS = 500 * 8 * SEQ_LEN  # 500 steps * batch 8 * seq_len
code\genome_165_annealed_donor.py:60:N_VAL_TOKENS = 64 * SEQ_LEN
code\genome_165_annealed_donor.py:61:LR = 3e-4
code\genome_165_annealed_donor.py:62:BATCH_SIZE = 8
code\genome_165_annealed_donor.py:81:DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
code\genome_165_annealed_donor.py:137:    model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, dtype=torch.bfloat16).to(DEVICE).eval()
code\genome_165_annealed_donor.py:147:    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).to(DEVICE)
code\genome_165_annealed_donor.py:169:    total = torch.zeros((), device=DEVICE, dtype=torch.float32)
code\genome_165_annealed_donor.py:191:    for i in range(0, val_ids.shape[0], BATCH_SIZE):
code\genome_165_annealed_donor.py:192:        ids = val_ids[i:i + BATCH_SIZE].to(DEVICE)
code\genome_165_annealed_donor.py:193:        mask = val_mask[i:i + BATCH_SIZE].to(DEVICE)
code\genome_165_annealed_donor.py:213:    optimizer = torch.optim.AdamW(recipient.parameters(), lr=LR, betas=(0.9, 0.95))
code\genome_165_annealed_donor.py:245:        idx = perm[(step * BATCH_SIZE) % n_train : (step * BATCH_SIZE) % n_train + BATCH_SIZE]
code\genome_165_annealed_donor.py:246:        if len(idx) < BATCH_SIZE:
code\genome_165_annealed_donor.py:247:            idx = perm[:BATCH_SIZE]
code\genome_165_annealed_donor.py:248:        ids = train_ids[idx].to(DEVICE)
code\genome_165_annealed_donor.py:249:        mask = train_mask[idx].to(DEVICE)
code\genome_165_annealed_donor.py:304:    print(f"  donor: {_MODEL_ID}  device: {DEVICE}")
code\genome_165_annealed_donor.py:305:    print(f"  seeds: {SEEDS}  steps: {N_STEPS}  batch: {BATCH_SIZE}")
code\genome_165_annealed_donor.py:318:        donor_params[k] = donor_params[k].to(DEVICE)
code\genome_165_annealed_donor.py:323:    train_ids, train_mask = tokenize_block(tok, train_texts, SEQ_LEN)
code\genome_165_annealed_donor.py:324:    val_ids, val_mask = tokenize_block(tok, val_texts, SEQ_LEN)
code\genome_165_annealed_donor.py:396:            "batch_size": BATCH_SIZE,
code\genome_165_annealed_donor.py:397:            "lr": LR,
code\genome_167_kd_canonical.py:81:SEQ_LEN = 256
code\genome_167_kd_canonical.py:82:TRAIN_BATCH_SIZE = 8
code\genome_167_kd_canonical.py:83:EVAL_BATCH_SIZE = 8
code\genome_167_kd_canonical.py:85:LR = 3e-4
code\genome_167_kd_canonical.py:86:LR_WARMUP_STEPS = 200
code\genome_167_kd_canonical.py:87:BETAS = (0.9, 0.95)
code\genome_167_kd_canonical.py:88:WEIGHT_DECAY = 0.1
code\genome_167_kd_canonical.py:92:N_TRAIN_WINDOWS = 8192
code\genome_167_kd_canonical.py:93:N_C4_VAL_WINDOWS = 1000
code\genome_167_kd_canonical.py:113:DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
code\genome_167_kd_canonical.py:191:    ).to(DEVICE).eval()
code\genome_167_kd_canonical.py:207:        max_position_embeddings=SEQ_LEN + 64,
code\genome_167_kd_canonical.py:219:    model = model.to(DEVICE)
code\genome_167_kd_canonical.py:443:        seq_len=SEQ_LEN,
code\genome_167_kd_canonical.py:456:        seq_len=SEQ_LEN,
code\genome_167_kd_canonical.py:479:    for start in range(0, eval_ids.shape[0], EVAL_BATCH_SIZE):
code\genome_167_kd_canonical.py:480:        ids = eval_ids[start : start + EVAL_BATCH_SIZE].to(DEVICE)
code\genome_167_kd_canonical.py:481:        mask = eval_mask[start : start + EVAL_BATCH_SIZE].to(DEVICE)
code\genome_167_kd_canonical.py:568:        for start in range(0, n_windows, TRAIN_BATCH_SIZE):
code\genome_167_kd_canonical.py:569:            ids = train_ids[start : start + TRAIN_BATCH_SIZE].to(DEVICE)
code\genome_167_kd_canonical.py:570:            mask = train_mask[start : start + TRAIN_BATCH_SIZE].to(DEVICE)
code\genome_167_kd_canonical.py:578:            if start == 0 or (start // TRAIN_BATCH_SIZE) % 200 == 0:
code\genome_167_kd_canonical.py:604:    return rng.integers(0, n_examples, size=(TRAIN_STEPS, TRAIN_BATCH_SIZE), dtype=np.int64)
code\genome_167_kd_canonical.py:638:        lr=LR,
code\genome_167_kd_canonical.py:639:        betas=BETAS,
code\genome_167_kd_canonical.py:640:        weight_decay=WEIGHT_DECAY,
code\genome_167_kd_canonical.py:653:        current_lr = warmup_lr(step - 1, LR, LR_WARMUP_STEPS)
code\genome_167_kd_canonical.py:659:        ids = train_ids[batch_index_tensor].to(DEVICE)
code\genome_167_kd_canonical.py:660:        mask = train_mask[batch_index_tensor].to(DEVICE)
code\genome_167_kd_canonical.py:671:                    DEVICE,
code\genome_167_kd_canonical.py:675:                    DEVICE,
code\genome_167_kd_canonical.py:914:    print(f"  device={DEVICE} forward_dtype={FORWARD_DTYPE} recipient_dtype=torch.float32")
code\genome_167_kd_canonical.py:915:    print(f"  seeds={SEEDS} steps={TRAIN_STEPS} batch={TRAIN_BATCH_SIZE}")
code\genome_167_kd_canonical.py:917:        f"  train_windows={N_TRAIN_WINDOWS} c4_val_windows={N_C4_VAL_WINDOWS} "
code\genome_167_kd_canonical.py:918:        f"wikitext_val_windows={N_WIKITEXT_VAL_WINDOWS} seq_len={SEQ_LEN}"
code\genome_167_kd_canonical.py:930:        n_windows=N_TRAIN_WINDOWS,
code\genome_167_kd_canonical.py:939:        n_windows=N_C4_VAL_WINDOWS,
code\genome_167_kd_canonical.py:1031:        "device": DEVICE,
code\genome_167_kd_canonical.py:1036:            "seq_len": SEQ_LEN,
code\genome_167_kd_canonical.py:1037:            "train_batch_size": TRAIN_BATCH_SIZE,
code\genome_167_kd_canonical.py:1038:            "eval_batch_size": EVAL_BATCH_SIZE,
code\genome_167_kd_canonical.py:1040:            "lr": LR,
code\genome_167_kd_canonical.py:1041:            "lr_warmup_steps": LR_WARMUP_STEPS,
code\genome_167_kd_canonical.py:1042:            "betas": list(BETAS),
code\genome_167_kd_canonical.py:1043:            "weight_decay": WEIGHT_DECAY,
code\genome_167_kd_canonical.py:1045:            "n_train_windows": N_TRAIN_WINDOWS,
code\genome_167_kd_canonical.py:1046:            "n_c4_val_windows": N_C4_VAL_WINDOWS,

 succeeded in 330ms:
{
  "genome": 195,
  "name": "untied_input_output_factorial",
  "timestamp_utc_started": "2026-04-30T16:42:30.518072+00:00",
  "config": {
    "train_steps": 5000,
    "seeds": [
      42,
      7,
      13
    ],
    "anchor_lambda": 0.01,
    "n_matched": 42257,
    "trained_fro": 363.26251220703125
  },
  "results": {
    "scratch_untied": {
      "42": {
        "arm_label": "scratch_untied",
        "seed": 42,
        "anchor_lambda": 0.0,
        "has_row_mask": false,
        "tied": false,
        "final_val_nll": 5.916087567806244,
        "trajectory": {
          "500": 6.577693045139313,
          "1000": 6.338981062173843,
          "1500": 6.190574675798416,
          "2000": 6.088173896074295,
          "2500": 6.028635501861572,
          "3000": 5.978825509548187,
          "3500": 5.929877817630768,
          "4000": 5.919655919075012,
          "4500": 5.926703304052353,
          "5000": 5.916087567806244
        },
        "wallclock_s": 441.8697397708893
      },
      "7": {
        "arm_label": "scratch_untied",
        "seed": 7,
        "anchor_lambda": 0.0,
        "has_row_mask": false,
        "tied": false,
        "final_val_nll": 5.936517000198364,
        "trajectory": {
          "500": 6.590441942214966,
          "1000": 6.34215247631073,
          "1500": 6.194429814815521,
          "2000": 6.108573019504547,
          "2500": 6.026343256235123,
          "3000": 5.9695262014865875,
          "3500": 5.934748888015747,
          "4000": 5.917667716741562,
          "4500": 5.909896790981293,
          "5000": 5.936517000198364
        },
        "wallclock_s": 442.33400774002075
      },
      "13": {
        "arm_label": "scratch_untied",
        "seed": 13,
        "anchor_lambda": 0.0,
        "has_row_mask": false,
        "tied": false,
        "final_val_nll": 5.915413916110992,
        "trajectory": {
          "500": 6.592167645692825,
          "1000": 6.348989367485046,
          "1500": 6.1985728442668915,
          "2000": 6.102028846740723,
          "2500": 6.004428178071976,
          "3000": 5.969106018543243,
          "3500": 5.936587363481522,
          "4000": 5.907089412212372,
          "4500": 5.917324960231781,
          "5000": 5.915413916110992
        },
        "wallclock_s": 442.3874309062958
      }
    },
    "input_inject_anchor": {
      "42": {
        "arm_label": "input_inject_anchor",
        "seed": 42,
        "anchor_lambda": 0.01,
        "has_row_mask": true,
        "tied": false,
        "final_val_nll": 5.739678591489792,
        "trajectory": {
          "500": 6.465795129537582,
          "1000": 6.204754561185837,
          "1500": 6.0550402998924255,
          "2000": 5.94249352812767,
          "2500": 5.8712553679943085,
          "3000": 5.8206928968429565,
          "3500": 5.770403504371643,
          "4000": 5.743799090385437,
          "4500": 5.740637719631195,
          "5000": 5.739678591489792
        },
        "wallclock_s": 458.9399878978729
      },
      "7": {
        "arm_label": "input_inject_anchor",
        "seed": 7,
        "anchor_lambda": 0.01,
        "has_row_mask": true,
        "tied": false,
        "final_val_nll": 5.743973910808563,
        "trajectory": {
          "500": 6.47830405831337,
          "1000": 6.210770070552826,
          "1500": 6.059934020042419,
          "2000": 5.966650754213333,
          "2500": 5.871767312288284,
          "3000": 5.8222969472408295,
          "3500": 5.778055667877197,
          "4000": 5.742737531661987,
          "4500": 5.730993390083313,
          "5000": 5.743973910808563
        },
        "wallclock_s": 459.06923842430115
      },
      "13": {
        "arm_label": "input_inject_anchor",
        "seed": 13,
        "anchor_lambda": 0.01,
        "has_row_mask": true,
        "tied": false,
        "final_val_nll": 5.7142486572265625,
        "trajectory": {
          "500": 6.467074483633041,
          "1000": 6.208632856607437,
          "1500": 6.047432035207748,
          "2000": 5.949491113424301,
          "2500": 5.851046293973923,
          "3000": 5.804028868675232,
          "3500": 5.756656736135483,
          "4000": 5.7338137328624725,
          "4500": 5.7301119565963745,
          "5000": 5.7142486572265625
        },
        "wallclock_s": 460.0550947189331
      }
    },
    "output_inject_anchor": {
      "42": {
        "arm_label": "output_inject_anchor",
        "seed": 42,
        "anchor_lambda": 0.01,
        "has_row_mask": true,
        "tied": false,
        "final_val_nll": 5.552595794200897,
        "trajectory": {
          "500": 6.259870648384094,
          "1000": 5.982435971498489,
          "1500": 5.804299682378769,
          "2000": 5.698489725589752,
          "2500": 5.638236165046692,
          "3000": 5.5905358493328094,
          "3500": 5.55367386341095,
          "4000": 5.537701278924942,
          "4500": 5.5661574602127075,
          "5000": 5.552595794200897
        },
        "wallclock_s": 458.74715399742126
      },
      "7": {
        "arm_label": "output_inject_anchor",
        "seed": 7,
        "anchor_lambda": 0.01,
        "has_row_mask": true,
        "tied": false,
        "final_val_nll": 5.57350018620491,
        "trajectory": {
          "500": 6.260403037071228,
          "1000": 5.977453589439392,
          "1500": 5.82775953412056,
          "2000": 5.716177761554718,
          "2500": 5.642973691225052,
          "3000": 5.594294846057892,
          "3500": 5.567190557718277,
          "4000": 5.5347417294979095,
          "4500": 5.533882707357407,
          "5000": 5.57350018620491
        },
        "wallclock_s": 655.3376989364624
      },
      "13": {
        "arm_label": "output_inject_anchor",
        "seed": 13,
        "anchor_lambda": 0.01,
        "has_row_mask": true,
        "tied": false,
        "final_val_nll": 5.555659115314484,
        "trajectory": {
          "500": 6.260916620492935,
          "1000": 5.981352120637894,
          "1500": 5.8083935379981995,
          "2000": 5.720544755458832,
          "2500": 5.626605778932571,
          "3000": 5.582149535417557,
          "3500": 5.56158059835434,
          "4000": 5.527446299791336,
          "4500": 5.5490908324718475,
          "5000": 5.555659115314484
        },
        "wallclock_s": 3351.116280555725
      }
    },
    "both_inject_anchor": {
      "42": {
        "arm_label": "both_inject_anchor",
        "seed": 42,
        "anchor_lambda": 0.01,
        "has_row_mask": true,
        "tied": false,
        "final_val_nll": 5.382154941558838,
        "trajectory": {
          "500": 6.049713909626007,
          "1000": 5.770394593477249,
          "1500": 5.595649063587189,
          "2000": 5.48610582947731,
          "2500": 5.433129549026489,
          "3000": 5.387698799371719,
          "3500": 5.370301455259323,
          "4000": 5.343516945838928,
          "4500": 5.375373184680939,
          "5000": 5.382154941558838
        },
        "wallclock_s": 525.504469871521
      },
      "7": {
        "arm_label": "both_inject_anchor",
        "seed": 7,
        "anchor_lambda": 0.01,
        "has_row_mask": true,
        "tied": false,
        "final_val_nll": 5.379207909107208,
        "trajectory": {
          "500": 6.053512454032898,
          "1000": 5.754784017801285,
          "1500": 5.60670605301857,
          "2000": 5.491317898035049,
          "2500": 5.420968234539032,
          "3000": 5.385938346385956,
          "3500": 5.357205957174301,
          "4000": 5.331331104040146,
          "4500": 5.322307705879211,
          "5000": 5.379207909107208
        },
        "wallclock_s": 562.7575595378876
      },
      "13": {
        "arm_label": "both_inject_anchor",
        "seed": 13,
        "anchor_lambda": 0.01,
        "has_row_mask": true,
        "tied": false,
        "final_val_nll": 5.361799776554108,
        "trajectory": {
          "500": 6.063318848609924,
          "1000": 5.771550476551056,
          "1500": 5.595634549856186,
          "2000": 5.503022134304047,
          "2500": 5.4181210696697235,
          "3000": 5.379931718111038,
          "3500": 5.354439169168472,
          "4000": 5.327908486127853,
          "4500": 5.355692684650421,
          "5000": 5.361799776554108
        },
        "wallclock_s": 560.7665469646454
      }
    },
    "tied_reference": {
      "42": {
        "arm_label": "tied_reference",
        "seed": 42,
        "anchor_lambda": 0.01,
        "has_row_mask": true,
        "tied": true,
        "final_val_nll": 5.361393749713898,
        "trajectory": {
          "500": 6.044264018535614,
          "1000": 5.766648977994919,
          "1500": 5.591025292873383,
          "2000": 5.48857581615448,
          "2500": 5.425168752670288,
          "3000": 5.377177447080612,
          "3500": 5.360908508300781,
          "4000": 5.331420660018921,
          "4500": 5.362398207187653,
          "5000": 5.361393749713898
        },
        "wallclock_s": 518.7009630203247
      }
    }
  },
  "summary": {},
  "verdict": "INCOMPLETE",
  "timestamp_utc_last_write": "2026-04-30T19:18:31.044220+00:00",
  "elapsed_s": 9360.526156187057
}

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "@'
import torch
for dtype in [torch.bfloat16, torch.float16, torch.float32]:
    p = torch.zeros((2,2), dtype=dtype)
    g = torch.ones((2,2), dtype=torch.float32)
    try:
        p.add_(g)
        print(dtype, 'ok', p.dtype)
    except Exception as e:
        print(dtype, type(e).__name__, str(e))
'@ | python -" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "@'
import torch
p = torch.nn.Parameter(torch.zeros((2,2), dtype=torch.bfloat16))
(p.sum()).backward()
print('grad dtype', p.grad.dtype)
grad_add = (p.detach().to(torch.float32) - torch.ones((2,2), dtype=torch.float32)) * 0.1
try:
    p.grad.add_(grad_add)
    print('param.grad.add_ ok')
except Exception as e:
    print('param.grad.add_ error', type(e).__name__, str(e))
'@ | python -" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
 succeeded in 2774ms:
torch.bfloat16 ok torch.bfloat16
torch.float16 ok torch.float16
torch.float32 ok torch.float32

 succeeded in 2791ms:
grad dtype torch.bfloat16
param.grad.add_ ok

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "@'
try:
    from transformers import Qwen3ForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    cfg = Qwen3Config(vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, intermediate_size=32, max_position_embeddings=64, tie_word_embeddings=True, head_dim=4, use_cache=False)
    m = Qwen3ForCausalLM(cfg)
    print('tie flag', m.config.tie_word_embeddings)
    print('same object', m.model.embed_tokens.weight is m.lm_head.weight)
    print('same ptr', m.model.embed_tokens.weight.data_ptr() == m.lm_head.weight.data_ptr())
except Exception as e:
    print(type(e).__name__, e)
'@ | python -" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "@'
try:
    import inspect
    from transformers import Qwen3ForCausalLM
    print(inspect.getsource(Qwen3ForCausalLM.__init__))
except Exception as e:
    print(type(e).__name__, e)
'@ | python -" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
 succeeded in 8487ms:
tie flag True
same object True
same ptr True

 succeeded in 8536ms:
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Test-Path results\\genome_192_28layer_replication.json; Test-Path results\\genome_192_28layer_replication_smoke.json; if (Test-Path results\\genome_192_28layer_replication.json) { Get-Content -Raw results\\genome_192_28layer_replication.json | Select-Object -First 1 }" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"genome_192_28layer\" WIKI.md experiments\\EXPERIMENTS.md experiments\\ledger.jsonl research\\prereg\\genome_192_28layer*.md" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content research'"\\prereg\\genome_192_28layer_replication_2026-04-30.md; for("'$i=1; $i -le [Math]::Min(80,$lines.Count); $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content research'"\\prereg\\genome_196_anchor_residue_factorial_2026-04-30.md; for("'$i=1; $i -le [Math]::Min(120,$lines.Count); $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
2026-04-30T19:23:49.340153Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.3 seconds
Output:
experiments\EXPERIMENTS.md:7:## 2026-04-30 — genome_192_28layer_replication — PRE-STAGED (full-depth Qwen3 replication)
experiments\EXPERIMENTS.md:15:Source: `code/genome_192_28layer_replication.py`, `research/prereg/genome_192_28layer_replication_2026-04-30.md` (DRAFT).
rg: research\prereg\genome_192_28layer*.md: The filename, directory name, or volume label syntax is incorrect. (os error 123)

 exited 1 in 250ms:
experiments\EXPERIMENTS.md:7:## 2026-04-30 — genome_192_28layer_replication — PRE-STAGED (full-depth Qwen3 replication)
experiments\EXPERIMENTS.md:15:Source: `code/genome_192_28layer_replication.py`, `research/prereg/genome_192_28layer_replication_2026-04-30.md` (DRAFT).
rg: research\prereg\genome_192_28layer*.md: The filename, directory name, or volume label syntax is incorrect. (os error 123)

 succeeded in 288ms:
False
False

 succeeded in 287ms:
1: # Pre-registration: g192 28-Layer String-Match Replication
2: 
3: **Status:** DRAFT (g194 PASS_DIRECTION confirmed cycle 180; gated on g195 per Codex Â§B cycle 168 â€” g192 inherits tied-head confound until g195 resolves)
4: 
5: ## Motivation
6: 
7: g191 PASS_CONTENT and g194 (pending) establish that trained embedding row content â€” specifically directional content â€” carries the +0.465 nats signal. But all experiments so far use an 8-layer Qwen3-arch shell. Adversarial A16 #3 (SEV-8) flags this as a scope limit: the signal may be a shallow-init regime artifact that disappears at full depth.
8: 
9: This experiment tests whether the embedding init effect persists at full 28-layer Qwen3-0.6B depth.
10: 
11: ## Hypothesis
12: 
13: **H1 (Persistence):** The matched_rows_only init + anchor effect is >= +0.20 nats at 28 layers (at least 43% of the 8-layer effect).
14: 
15: **H2 (Attenuation):** The effect attenuates to < +0.10 nats at 28 layers (deeper model washes out embedding signal).
16: 
17: ## Design
18: 
19: ### Arms (3 arms x 3 seeds = 9 cells)
20: 
21: | Arm | Init embedding | Anchor | Tests |
22: |-----|---------------|--------|-------|
23: | `scratch_ce` | Random (Qwen3 init) | None | Baseline |
24: | `matched_rows_only` | Matched trained rows at correct positions | Same, masked to matched rows | Reference (expects +0.465 at 8-layer; how much survives at 28?) |
25: | `row_shuffled` | Permuted matched rows | Same (permuted), masked | Negative control (expects harmful at both depths) |
26: 
27: All arms: **28-layer** Qwen3-arch with GPT-2 tokenizer, 5000 steps, same data/eval as g191. Anchor lambda=0.01, masked to matched rows only. Same training hyperparameters (lr, betas, weight_decay, batch_size, grad_clip).
28: 
29: ### Model specification
30: 
31: `Qwen3Config(vocab_size=50257, hidden_size=1024, num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, intermediate_size=3072, max_position_embeddings=320, tie_word_embeddings=True, head_dim=128, rope_theta=1000000.0, use_cache=False)`
32: 
33: Matches actual Qwen3-0.6B config (except vocab_size=50257 for GPT-2 tokenizer). Estimated VRAM: ~4.0 GB (well within 22 GB envelope).
34: 
35: ## Pass/Fail Criteria
36: 
37: **PASS_PERSISTENCE:** matched_rows_only mean gain >= +0.20 nats vs scratch AND 3/3 seeds AND row_shuffled mean gain <= 0.0 nats (still harmful).
38: 
39: **PASS_ATTENUATION:** matched_rows_only mean gain >= +0.10 but < +0.20 nats AND matched_mean > shuffled_mean + 0.05 nats. Effect exists but is weaker at depth.
40: 
41: **FAIL:** matched_rows_only mean gain < +0.10 nats. Embedding init effect is a shallow artifact.
42: 
43: ## Universality Level
44: 
45: Level-3 (architecture-specific, within Qwen3 family). Scale robustness test.
46: 
47: ## Compute Envelope
48: 
49: - 28-layer Qwen3-arch, GPT-2 tokenizer, ~3.2 GB VRAM per cell
50: - 9 cells x ~17 min = ~2.6 hours total
51: - Within COMPUTE.md 4h envelope
52: 
53: ## What a null result means
54: 
55: If FAIL: the embedding init effect is only relevant for shallow models. The "interface codebook" finding (g191, g194) is architecturally interesting but does not scale to practical depths. Training-health diagnostics would need to account for depth-dependent attenuation.

 succeeded in 291ms:
1: # Pre-registration: g196 Anchor-Residue Factorial
2: 
3: **Status:** DRAFT (launch gated on g195 final verdict; designed 2026-04-30 after g194 PASS_DIRECTION)
4: 
5: ## Motivation
6: 
7: g191/g194 established that exact-string matched trained token-row directions carry the within-family gain. g194 specifically resolved the scalar-vs-direction confound: correct trained unit directions with shuffled or uniform norms recover 95-97% of the full +0.465 nats signal, while wrong/random directions are harmful.
8: 
9: A18 SEV-10 #2 remains unresolved: g191 found `direct_anchor_only` recovers 98% of the signal, while `direct_init_only` recovers only 19%. The live question is whether the trained row directions change the recipient optimization basin, leaving a persistent residue after the tether is removed, or whether they only help as an active regularizer while the anchor term remains in the loss.
10: 
11: A18 SEV-10 #3 also remains live: "direction" could mean a well-conditioned angular scaffold, not token-specific trained row content. g196 therefore includes angular-scaffold controls, not just anchor cutoff schedules.
12: 
13: ## Launch Gate
14: 
15: Run g196 only after g195 completes.
16: 
17: - If g195 returns `PASS_INPUT`, `PASS_INPUT_DOMINANT`, `PASS_OUTPUT`, `PASS_OUTPUT_DOMINANT`, or `PASS_BOTH_NEEDED`, run the **untied primary branch** below using the g195-winning intervention surface.
18: - If g195 is ambiguous but `tied_reference` cleanly reproduces the g191/g194 signal while untied decomposition is not interpretable, run the **tied fallback branch** and label the result as resolving regularization only for the tied interface.
19: - If g195 `FAIL`s with no arm >= +0.10 nats and no tied-reference replication, do not launch g196. First diagnose why the known g191/g194 effect failed to replicate.
20: 
21: ## Hypotheses
22: 
23: **H1: PASS_RESIDUE.** Correct trained row directions alter the optimization basin. Early anchor exposure leaves a final-step gain after the anchor is removed for thousands of SGD steps.
24: 
25: **H2: PASS_REGULARIZATION.** Correct trained row directions help only as an active tether. Once the anchor is removed, gains decay back to the no-anchor/init-only regime.
26: 
27: **H3: PASS_SCAFFOLD_ALT.** The apparent direction effect is not token-specific trained content. A geometry-preserving or covariance-conditioned scaffold produces a comparable active-anchor gain.
28: 
29: ## Common Protocol
30: 
31: - Architecture: 8-layer Qwen3-architecture shell with GPT-2 tokenizer, same base setup as g191/g194/g195.
32: - Training: 5000 steps, C4 train windows and C4 validation windows identical to g191/g194/g195.
33: - Seeds: `[42, 7, 13]`.
34: - Eval cadence: every 500 steps plus final.
35: - Anchor lambda: `0.01` whenever the schedule says anchor is active.
36: - Anchor mask: exact-string matched GPT-2/Qwen3 token rows only.
37: - Primary target matrix: g194 direction-only target, `correct_dir_uniform_norm`, not full trained norms. Matched rows use trained unit directions with uniform matched-row mean norm, Frobenius-normalized to the g194 matched-row Fro norm. Unmatched rows are not anchored.
38: - Score: seed-matched final validation NLL gain vs scratch: `gain_arm_seed = scratch_seed_final_nll - arm_seed_final_nll`. Positive is better.
39: 
40: ## Branch Selection From g195
41: 
42: ### Untied Primary Branch
43: 
44: Use `tie_word_embeddings=False`. The intervention surface is determined by g195:
45: 
46: | g195 verdict | g196 intervention surface |
47: |---|---|
48: | `PASS_INPUT` or `PASS_INPUT_DOMINANT` | Apply init/anchor schedules only to `model.model.embed_tokens.weight`. |
49: | `PASS_OUTPUT` or `PASS_OUTPUT_DOMINANT` | Apply init/anchor schedules only to `model.lm_head.weight`. |
50: | `PASS_BOTH_NEEDED` | Apply the same target and schedule to both `embed_tokens` and `lm_head`. |
51: 
52: The scratch baseline is always `scratch_untied`.
53: 
54: ### Tied Fallback Branch
55: 
56: Use `tie_word_embeddings=True`, same as g191/g194. The single tied matrix receives the schedules below. This branch cannot distinguish input embedding from output classifier geometry; it only decides persistence vs active regularization for the tied interface prior.
57: 
58: ## Arms
59: 
60: Primary run: 10 arms x 3 seeds = 30 cells.
61: 
62: | Arm | Step-0 init | Anchor schedule | Purpose |
63: |---|---|---|---|
64: | `scratch` | None | lambda=0 for steps 1-5000 | Seed-matched baseline |
65: | `init_only` | Target rows injected into selected surface | lambda=0 for steps 1-5000 | Measures pure initialization residue |
66: | `anchor_only_full` | None | lambda=0.01 for steps 1-5000 | Active-regularization positive reference |
67: | `init_anchor_full` | Target rows injected into selected surface | lambda=0.01 for steps 1-5000 | g191/g194-style positive reference |
68: | `cutoff_50` | None | lambda=0.01 for steps 1-50, then 0 for steps 51-5000 | Tests whether a tiny early tether changes basin |
69: | `cutoff_500` | None | lambda=0.01 for steps 1-500, then 0 for steps 501-5000 | Tests early-training basin residue |
70: | `cutoff_2000` | None | lambda=0.01 for steps 1-2000, then 0 for steps 2001-5000 | Primary persistence test: 3000 post-cutoff steps |
71: | `late_anchor_only_2000` | None | lambda=0 for steps 1-2000, then 0.01 for steps 2001-5000 | Tests whether active tether helps even when introduced late |
72: | `orthogonal_scaffold_full` | None | lambda=0.01 for steps 1-5000 to orthogonally rotated target | Preserves all trained row-row angles; destroys trained coordinate basis |
73: | `cov_scaffold_full` | None | lambda=0.01 for steps 1-5000 to covariance-matched random target | Preserves second-order conditioning; destroys token-specific row content |
74: 
75: `cutoff_*` arms deliberately use no step-0 injection. They test whether the active anchor, starting from random init, leaves a basin residue. `init_only` and `init_anchor_full` separately measure whether step-0 placement matters.
76: 
77: ## Scaffold Control Construction
78: 
79: All scaffold controls use the same matched-row mask and same uniform row norm/Frobenius normalization as the primary target.
80: 
81: ### `orthogonal_scaffold_full`
82: 
83: Let `T` be the primary matched-row target. Draw a fixed random orthogonal matrix `Q` from QR decomposition of a standard normal `(d, d)` matrix using scaffold seed `19601`. Use `T_rot = T @ Q` on matched rows.
84: 
85: This preserves every pairwise cosine between token rows exactly, including the global angular graph, while destroying the trained coordinate basis.
86: 
87: ### `cov_scaffold_full`
88: 
89: Let `T_m` be the matched-row target matrix. Estimate its empirical mean and covariance. Draw `X_m ~ N(mean(T_m), cov(T_m) + eps I)` using scaffold seed `19602`, then row-normalize to the same uniform norm and Frobenius-normalize to matched Fro.
90: 
91: This preserves the row-cloud conditioning and coordinate covariance structure but destroys token-specific trained row identity and most pairwise row relations.
92: 
93: ## Metrics
94: 
95: For each arm:
96: 
97: - `final_gain_mean`: mean seed-matched gain at step 5000.
98: - `final_gain_per_seed`: seed-matched gains for seeds 42, 7, 13.
99: - `active_reference_gain`: `anchor_only_full.final_gain_mean`.
100: - `init_anchor_reference_gain`: `init_anchor_full.final_gain_mean`.
101: - `residue_fraction_2000`: `cutoff_2000.final_gain_mean / anchor_only_full.final_gain_mean`.
102: - `control_max_gain`: max of `orthogonal_scaffold_full.final_gain_mean`, `cov_scaffold_full.final_gain_mean`.
103: - Drift diagnostics for cutoff arms: gain at cutoff eval step, gain at 5000, and post-cutoff decay.
104: 
105: ## Pass/Fail Criteria
106: 
107: ### Anchor Effect Replication Gate
108: 
109: Before interpreting residue:
110: 
111: - `anchor_only_full.final_gain_mean >= +0.30` and 3/3 seeds positive, OR
112: - `init_anchor_full.final_gain_mean >= +0.30` and 3/3 seeds positive.
113: 
114: If neither condition holds, verdict is `FAIL_REPLICATION`; do not interpret residue.
115: 
116: ### PASS_RESIDUE
117: 
118: All must hold:
119: 
120: 1. Anchor Effect Replication Gate passes.

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content research'"\\prereg\\genome_196_anchor_residue_factorial_2026-04-30.md; for("'$i=116; $i -le [Math]::Min(190,$lines.Count); $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content experiments'"\\EXPERIMENTS.md; for("'$i=1; $i -le [Math]::Min(40,$lines.Count); $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content WIKI.md; for($i=29; $i -le 39; $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
 succeeded in 285ms:
116: ### PASS_RESIDUE
117: 
118: All must hold:
119: 
120: 1. Anchor Effect Replication Gate passes.
121: 2. `cutoff_2000.final_gain_mean >= +0.20`.
122: 3. `cutoff_2000` is positive in 3/3 seeds.
123: 4. `residue_fraction_2000 >= 0.45`.
124: 5. `cutoff_2000.final_gain_mean >= init_only.final_gain_mean + 0.10`.
125: 6. `control_max_gain <= +0.15` and `anchor_only_full.final_gain_mean - control_max_gain >= +0.20`.
126: 
127: Interpretation: correct trained directions leave a persistent basin/content residue after 3000 steps without active tether. This resolves A18 #2 in favor of basin change and eliminates A18 #3 for this run.
128: 
129: ### PASS_PARTIAL_RESIDUE
130: 
131: All must hold:
132: 
133: 1. Anchor Effect Replication Gate passes.
134: 2. `cutoff_2000.final_gain_mean >= +0.12` and positive in at least 2/3 seeds.
135: 3. `residue_fraction_2000 >= 0.25`.
136: 4. Scaffold controls do not pass `PASS_SCAFFOLD_ALT`.
137: 
138: Interpretation: weak basin residue exists, but not enough to support the strong +6.5 section-0.1 move without replication or full-depth follow-up.
139: 
140: ### PASS_REGULARIZATION
141: 
142: All must hold:
143: 
144: 1. Anchor Effect Replication Gate passes.
145: 2. `cutoff_500.final_gain_mean < +0.12`.
146: 3. `cutoff_2000.final_gain_mean < +0.12`.
147: 4. `residue_fraction_2000 < 0.25`.
148: 5. `late_anchor_only_2000.final_gain_mean >= +0.20` and at least 50% of `anchor_only_full.final_gain_mean`.
149: 
150: Interpretation: directions help mainly while actively tethering the selected rows. The anchor is a regularizer, not evidence of a persistent content/basin residue.
151: 
152: ### PASS_EARLY_WINDOW
153: 
154: All must hold:
155: 
156: 1. Anchor Effect Replication Gate passes.
157: 2. `cutoff_50.final_gain_mean < +0.10`.
158: 3. `cutoff_500.final_gain_mean < +0.15`.
159: 4. `cutoff_2000.final_gain_mean >= +0.20` and 3/3 seeds positive.
160: 
161: Interpretation: not pure regularization; the basin residue exists but requires a long early tether. This is weaker than PASS_RESIDUE only if retention is below 45%.
162: 
163: ### PASS_SCAFFOLD_ALT
164: 
165: Either scaffold control has:
166: 
167: - `final_gain_mean >= +0.20`, OR
168: - `final_gain_mean >= 0.50 * anchor_only_full.final_gain_mean`.
169: 
170: Interpretation: A18 #3 remains alive. A well-conditioned angular/covariance scaffold is sufficient to explain a substantial part of the benefit. Even if cutoff arms retain gain, the result cannot be claimed as token-specific trained-row content.
171: 
172: ### FAIL
173: 
174: Use `FAIL` if no above verdict applies. Report the closest pattern explicitly:
175: 
176: - active effect absent: `FAIL_REPLICATION`;
177: - active effect present but no residue and late anchor fails: `FAIL_TIMING_AMBIGUOUS`;
178: - controls positive: `PASS_SCAFFOLD_ALT`;
179: - mixed seed behavior: `FAIL_NOISY`.
180: 
181: ## Compute Envelope
182: 
183: Expected runtime from g194/g195:
184: 
185: - 8-layer tied cells: about 6.9 min/cell.
186: - 8-layer untied cells: about 7.3 min/cell from current g195 scratch cells.
187: - 30 cells x 7.3 min = about 219 min = 3.65 hours.
188: - Smoke test: all 10 arms, seed 42, 50 steps, non-verdict only.
189: 
190: The main run stays under the COMPUTE.md 4h target on the 8-layer shell. Do not run this exact 10-arm design at 28 layers: 30 cells x about 17 min would exceed 8h. If g192 passes and full-depth residue becomes necessary, use a separate compressed g196b with 5 arms (`scratch`, `anchor_only_full`, `cutoff_500`, `cutoff_2000`, `best_scaffold_full`) x 2-3 seeds.

 succeeded in 295ms:
1: # EXPERIMENTS â€” Neural Genome
2: 
3: *Reverse-chronological log of every experiment. One entry per run. Only Codex-validated conclusions appear here. Raw ledger lives in `ledger.jsonl`.*
4: 
5: ---
6: 
7: ## 2026-04-30 â€” genome_192_28layer_replication â€” PRE-STAGED (full-depth Qwen3 replication)
8: 
9: **Purpose.** Test whether the 8-layer trained-token-row direction signal persists at full 28-layer Qwen3-0.6B depth (~492M params). If it does, the finding generalizes beyond shallow shells and Â§0.1 score moves from 5.8 to ~6.1.
10: 
11: **Config:** 28 layers, hidden=1024, heads=16, kv_heads=8, intermediate=3072, head_dim=128, rope_theta=1e6, tied weights. Estimated peak VRAM ~8-10 GiB (within 22 GB envelope). 
12: 
13: **Status:** PRE-STAGED (code ready cycle 189, Codex Â§A confirmed launch-ready, prereg DRAFT â€” gated on g195).
14: 
15: Source: `code/genome_192_28layer_replication.py`, `research/prereg/genome_192_28layer_replication_2026-04-30.md` (DRAFT).
16: 
17: ---
18: 
19: ## 2026-04-30 â€” genome_196_anchor_residue_factorial â€” PRE-STAGED (anchor persistence vs regularization)
20: 
21: **Purpose.** Resolve A18 SEV-10 #2: does the trained-row direction signal create a persistent optimization basin (residue), or does it only help as an active regularizer (anchor term in loss)? 10 arms x 3 seeds x 5000 steps = 30 cells. Includes cutoff (anchor removed at step 2000), late-start (anchor added at step 2000), scaffold controls (orthogonal QR / covariance noise), and init_only (no anchor, just initialization).
22: 
23: **Arms:** scratch, init_only, anchor_only_full, init_anchor_full, cutoff_50, cutoff_500, cutoff_2000, late_anchor_only_2000, orthogonal_scaffold_full, cov_scaffold_full.
24: 
25: **Pass criteria.** PASS_RESIDUE: cutoff arms retain >= 80% of full-run signal at final step. FAIL_RESIDUE: cutoff arms lose > 50%. Scaffold controls must stay below +0.10 to confirm non-trivial geometry drives the signal.
26: 
27: **Status:** PRE-STAGED (code complete cycle 186, all Codex SEVs fixed cycle 190, prereg DRAFT â€” locks when g195 determines surface).
28: 
29: Source: `code/genome_196_anchor_residue_factorial.py`, `research/prereg/genome_196_anchor_residue_factorial_2026-04-30.md` (DRAFT), `results/genome_196_anchor_residue_factorial.json` (not yet created).
30: 
31: ---
32: 
33: ## 2026-04-30 â€” genome_195_untied_input_output_factorial â€” RUNNING (untied input/output mechanism isolation)
34: 
35: **Purpose.** Resolve A18 SEV-10 #1: with tie_word_embeddings=True, embed_tokens IS lm_head. The +0.465 nats signal may be output-logit geometry, not input embedding. This experiment unties the weights and tests each side independently. 5 arms x 3 seeds x 5000 steps = 15 cells.
36: 
37: **Arms:** scratch_untied (baseline), input_inject_anchor (trained embed_tokens only), output_inject_anchor (trained lm_head only), both_inject_anchor (both untied), tied_reference (tied comparison).
38: 
39: **Pass criteria.** PASS_INPUT: input >= +0.30 AND output < +0.15. PASS_OUTPUT: output >= +0.30 AND input < +0.15. PASS_BOTH_NEEDED: both >= +0.30 AND neither alone > 80%. FAIL: max gain < +0.10.
40: 

 succeeded in 286ms:
29: ## CURRENT STATUS (2026-04-30, cycle 190)
30: 
31: **Â§0.1 honest score: 5.2/10** (post-g194 PASS_DIRECTION, per Codex Â§B cycle 174). **g194 PASS_DIRECTION (18/18 cells, cycle 180).** Direction carries 95-97% of signal; norms irrelevant. cd_sn +0.442, cd_un +0.451, sd_cn -0.662, rd_cn -1.019. Resolves A17 SEV-10.
32: 
33: **g195 untied input/output factorial RUNNING (8/15 cells done, cycle 190).** Output dominance emerging: output_inject_anchor mean gain +0.364 (78% of tied), input_inject_anchor mean gain +0.190 (41%). Heading toward PASS_OUTPUT_DOMINANT. Prereg LOCKED.
34: 
35: **g196 anchor-residue factorial IMPLEMENTATION COMPLETE (cycle 190).** Prereg DRAFT (gated on g195 â€” locks when g195 verdict determines surface). Code at `code/genome_196_anchor_residue_factorial.py`. 10 arms x 3 seeds = 30 cells. Codex Â§A cycle 186+189 reviewed, all SEVs fixed (SEV-8 init_only mask, SEV-6 verdict guard, SEV-6 resume validation, SEV-5 cutoff eval steps, SEV-5 CLI footgun).
36: 
37: **A18 remaining:** (1) tied lm_head confound â†’ g195 resolving. (2) anchor dominance = regularization â†’ g196 anchor-residue factorial. **g192 28-layer replication PRE-STAGED** (config matches actual Qwen3-0.6B). Ceiling ~6.0 if g195+g192 pass; ~6.5 if g196 anchor-residue shows persistence.
38: 
39: **Queue:** g195 (RUNNING 8/15) -> g192 (28-layer, gated on g195) -> g196 (anchor-residue, gated on g195) -> g190 (DEFERRED).

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content code'"\\genome_194_scalar_direction_factorial.py; for("'$i=200; $i -le [Math]::Min(320,$lines.Count); $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content code'"\\genome_191_string_match_decomposition.py; for("'$i=300; $i -le [Math]::Min(420,$lines.Count); $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
 succeeded in 305ms:
200: 
201: 
202: def main() -> None:
203:     parser = argparse.ArgumentParser()
204:     parser.add_argument("--smoke", action="store_true")
205:     parser.add_argument("--no-resume", action="store_true")
206:     args = parser.parse_args()
207: 
208:     smoke = args.smoke
209:     n_steps = 50 if smoke else TRAIN_STEPS
210:     seeds = [42] if smoke else SEEDS
211:     run_out_path = OUT_PATH.with_name(OUT_PATH.stem + "_smoke.json") if smoke else OUT_PATH
212: 
213:     print_flush(f"=== g194 Scalar/Direction Factorial ===")
214:     print_flush(f"  smoke={smoke}, steps={n_steps}, seeds={seeds}")
215: 
216:     from transformers import AutoTokenizer, AutoModelForCausalLM
217:     tok_qwen = AutoTokenizer.from_pretrained(g188.QWEN_MODEL_ID)
218:     tok_gpt2 = AutoTokenizer.from_pretrained(g188.GPT2_MODEL_ID)
219:     if tok_gpt2.pad_token is None:
220:         tok_gpt2.pad_token = tok_gpt2.eos_token
221: 
222:     print_flush("\n--- Loading data ---")
223:     train_ids, train_mask, _ = g167.load_c4_windows(
224:         tok_gpt2, split="train", seed=g188.C4_TRAIN_SEED, n_windows=g188.N_TRAIN_WINDOWS,
225:     )
226:     train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
227:     val_ids, val_mask, _ = g167.load_c4_windows(
228:         tok_gpt2, split="train", seed=g188.C4_VAL_SEED, n_windows=g188.N_C4_VAL_WINDOWS,
229:         forbidden_hashes=train_hashes,
230:     )
231:     print_flush(f"  Train: {train_ids.shape}, Val: {val_ids.shape}")
232: 
233:     print_flush("\n--- Loading Qwen3 trained embeddings ---")
234:     qwen_model = AutoModelForCausalLM.from_pretrained(g188.QWEN_MODEL_ID, torch_dtype=torch.float32)
235:     trained_embed = qwen_model.model.embed_tokens.weight.detach().cpu().numpy()
236:     trained_fro = float(np.linalg.norm(trained_embed, "fro"))
237:     del qwen_model
238:     cleanup_cuda()
239:     print_flush(f"  Trained embed: {trained_embed.shape}, Fro={trained_fro:.1f}")
240: 
241:     print_flush("\n--- Building string-match base embeddings ---")
242:     gpt2_vocab = len(tok_gpt2)
243:     embed_dim = trained_embed.shape[1]
244:     full_embed, matched_mask = g191.build_string_match_with_mask(
245:         tok_qwen, tok_gpt2, trained_embed, gpt2_vocab, embed_dim,
246:     )
247:     full_embed = g188.normalize_to_fro_norm(full_embed, trained_fro)
248: 
249:     norms, unit_dirs = decompose_rows(full_embed, matched_mask)
250:     n_matched = int(matched_mask.sum())
251:     matched_norms = norms[matched_mask]
252:     matched_fro = float(np.linalg.norm(full_embed[matched_mask], "fro"))
253:     print_flush(f"  Matched: {n_matched}, norm range: [{matched_norms.min():.4f}, {matched_norms.max():.4f}], mean={matched_norms.mean():.4f}")
254:     print_flush(f"  matched_fro={matched_fro:.2f}, trained_fro={trained_fro:.1f}")
255: 
256:     rng = np.random.default_rng(PERM_SEED)
257: 
258:     matched_only = g191.build_matched_rows_only(full_embed, matched_mask)
259: 
260:     cd_sn = build_correct_dir_shuffled_norm(unit_dirs, norms, matched_mask, rng)
261:     cd_sn = g188.normalize_to_fro_norm(cd_sn, matched_fro)
262: 
263:     rng2 = np.random.default_rng(PERM_SEED + 1)
264:     sd_cn = build_shuffled_dir_correct_norm(unit_dirs, norms, matched_mask, rng2)
265:     sd_cn = g188.normalize_to_fro_norm(sd_cn, matched_fro)
266: 
267:     rng3 = np.random.default_rng(PERM_SEED + 2)
268:     rd_cn = build_random_dir_correct_norm(norms, matched_mask, embed_dim, rng3)
269:     rd_cn = g188.normalize_to_fro_norm(rd_cn, matched_fro)
270: 
271:     cd_un = build_correct_dir_uniform_norm(unit_dirs, norms, matched_mask)
272:     cd_un = g188.normalize_to_fro_norm(cd_un, matched_fro)
273: 
274:     arm_configs = {
275:         "scratch_ce":                {"custom_embed": None,       "anchor_embed": None,       "anchor_mask": None},
276:         "full_match":                {"custom_embed": full_embed, "anchor_embed": full_embed, "anchor_mask": matched_mask, "custom_mask": matched_mask},
277:         "correct_dir_shuffled_norm": {"custom_embed": cd_sn,      "anchor_embed": cd_sn,      "anchor_mask": matched_mask, "custom_mask": matched_mask},
278:         "shuffled_dir_correct_norm": {"custom_embed": sd_cn,      "anchor_embed": sd_cn,      "anchor_mask": matched_mask, "custom_mask": matched_mask},
279:         "random_dir_correct_norm":   {"custom_embed": rd_cn,      "anchor_embed": rd_cn,      "anchor_mask": matched_mask, "custom_mask": matched_mask},
280:         "correct_dir_uniform_norm":  {"custom_embed": cd_un,      "anchor_embed": cd_un,      "anchor_mask": matched_mask, "custom_mask": matched_mask},
281:     }
282: 
283:     if not args.no_resume and run_out_path.exists():
284:         payload = json.loads(run_out_path.read_text(encoding="utf-8"))
285:     else:
286:         payload = {
287:             "genome": 194,
288:             "name": "scalar_direction_factorial",
289:             "timestamp_utc_started": now_utc(),
290:             "config": {
291:                 "train_steps": n_steps,
292:                 "seeds": seeds,
293:                 "anchor_lambda": ANCHOR_LAMBDA,
294:                 "perm_seed": PERM_SEED,
295:                 "n_matched": n_matched,
296:                 "trained_fro": trained_fro,
297:                 "matched_fro": matched_fro,
298:                 "matched_norm_mean": float(matched_norms.mean()),
299:                 "matched_norm_std": float(matched_norms.std()),
300:             },
301:             "results": {},
302:             "summary": {},
303:             "verdict": "INCOMPLETE",
304:         }
305: 
306:     t_start = time.time()
307: 
308:     def save():
309:         payload["timestamp_utc_last_write"] = now_utc()
310:         payload["elapsed_s"] = time.time() - t_start
311:         tmp = run_out_path.with_suffix(run_out_path.suffix + ".tmp")
312:         tmp.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")
313:         os.replace(tmp, run_out_path)
314: 
315:     for arm_label in ARMS:
316:         payload["results"].setdefault(arm_label, {})
317:         cfg = arm_configs[arm_label]
318: 
319:         for seed in seeds:
320:             key = str(seed)

 succeeded in 288ms:
300:         and shuffled_mean <= 0.10
301:         and freq_mean <= 0.10
302:         and (matched_mean - shuffled_mean) >= 0.25
303:     )
304:     format_pass = shuffled_mean >= 0.25
305: 
306:     if content_pass:
307:         verdict = "PASS_CONTENT"
308:     elif format_pass:
309:         verdict = "PASS_FORMAT"
310:     elif matched_mean >= 0.15:
311:         verdict = "PARTIAL"
312:     else:
313:         verdict = "FAIL"
314: 
315:     return {
316:         "status": "complete",
317:         "verdict": verdict,
318:         "matched_rows_mean_gain": matched_mean,
319:         "matched_rows_per_seed": matched_gaps,
320:         "unmatched_rows_mean_gain": unmatched_mean,
321:         "row_shuffled_mean_gain": shuffled_mean,
322:         "freq_bucket_mean_gain": freq_mean,
323:         "init_only_mean_gain": init_only_mean,
324:         "anchor_only_mean_gain": anchor_only_mean,
325:     }
326: 
327: 
328: # ---------- Main ----------
329: 
330: def main() -> None:
331:     parser = argparse.ArgumentParser()
332:     parser.add_argument("--smoke", action="store_true")
333:     parser.add_argument("--no-resume", action="store_true")
334:     args = parser.parse_args()
335: 
336:     smoke = args.smoke
337:     n_steps = 50 if smoke else TRAIN_STEPS
338:     seeds = [42] if smoke else SEEDS
339:     run_out_path = OUT_PATH.with_name(OUT_PATH.stem + "_smoke.json") if smoke else OUT_PATH
340: 
341:     print_flush(f"=== g191 String-Match Decomposition ===")
342:     print_flush(f"  smoke={smoke}, steps={n_steps}, seeds={seeds}")
343: 
344:     from transformers import AutoTokenizer, AutoModelForCausalLM
345:     tok_qwen = AutoTokenizer.from_pretrained(g188.QWEN_MODEL_ID)
346:     tok_gpt2 = AutoTokenizer.from_pretrained(g188.GPT2_MODEL_ID)
347:     if tok_gpt2.pad_token is None:
348:         tok_gpt2.pad_token = tok_gpt2.eos_token
349: 
350:     print_flush("\n--- Loading data ---")
351:     train_ids, train_mask, _ = g167.load_c4_windows(
352:         tok_gpt2, split="train", seed=g188.C4_TRAIN_SEED, n_windows=g188.N_TRAIN_WINDOWS,
353:     )
354:     train_hashes = g167.collect_13gram_hashes(train_ids, train_mask)
355:     val_ids, val_mask, _ = g167.load_c4_windows(
356:         tok_gpt2, split="train", seed=g188.C4_VAL_SEED, n_windows=g188.N_C4_VAL_WINDOWS,
357:         forbidden_hashes=train_hashes,
358:     )
359:     print_flush(f"  Train: {train_ids.shape}, Val: {val_ids.shape}")
360: 
361:     print_flush("\n--- Loading Qwen3 trained embeddings ---")
362:     qwen_model = AutoModelForCausalLM.from_pretrained(g188.QWEN_MODEL_ID, torch_dtype=torch.float32)
363:     trained_embed = qwen_model.model.embed_tokens.weight.detach().cpu().numpy()
364:     trained_fro = float(np.linalg.norm(trained_embed, "fro"))
365:     del qwen_model
366:     cleanup_cuda()
367:     print_flush(f"  Trained embed: {trained_embed.shape}, Fro={trained_fro:.1f}")
368: 
369:     print_flush("\n--- Building string-match embeddings ---")
370:     gpt2_vocab = len(tok_gpt2)
371:     embed_dim = trained_embed.shape[1]
372:     full_embed, matched_mask = build_string_match_with_mask(
373:         tok_qwen, tok_gpt2, trained_embed, gpt2_vocab, embed_dim,
374:     )
375:     full_embed = g188.normalize_to_fro_norm(full_embed, trained_fro)
376: 
377:     rng = np.random.default_rng(191)
378:     matched_only = build_matched_rows_only(full_embed, matched_mask)
379:     unmatched_only = build_unmatched_rows_only(full_embed, matched_mask)
380:     shuffled = build_row_shuffled_matched(full_embed, matched_mask, rng)
381:     shuffled = g188.normalize_to_fro_norm(shuffled, trained_fro)
382:     freq_shuf = build_frequency_bucket_shuffle(full_embed, matched_mask, tok_gpt2, train_ids, rng)
383:     freq_shuf = g188.normalize_to_fro_norm(freq_shuf, trained_fro)
384: 
385:     n_matched = int(matched_mask.sum())
386:     n_unmatched = int((~matched_mask).sum())
387:     print_flush(f"  Matched: {n_matched}, Unmatched: {n_unmatched}")
388: 
389:     arm_configs = {
390:         "scratch_ce":              {"custom_embed": None,          "anchor_embed": None,          "anchor_mask": None},
391:         "direct_init_only":        {"custom_embed": full_embed,    "anchor_embed": None,          "anchor_mask": None},
392:         "direct_anchor_only":      {"custom_embed": None,          "anchor_embed": full_embed,    "anchor_mask": None},
393:         "matched_rows_only":       {"custom_embed": full_embed,    "anchor_embed": full_embed,    "anchor_mask": matched_mask, "custom_mask": matched_mask},
394:         "unmatched_rows_only":     {"custom_embed": None,          "anchor_embed": unmatched_only, "anchor_mask": ~matched_mask},
395:         "row_shuffled_matched":    {"custom_embed": shuffled,      "anchor_embed": shuffled,      "anchor_mask": matched_mask, "custom_mask": matched_mask},
396:         "frequency_bucket_shuffle":{"custom_embed": freq_shuf,     "anchor_embed": freq_shuf,     "anchor_mask": matched_mask, "custom_mask": matched_mask},
397:     }
398: 
399:     if not args.no_resume and run_out_path.exists():
400:         payload = json.loads(run_out_path.read_text(encoding="utf-8"))
401:     else:
402:         payload = {
403:             "genome": 191,
404:             "name": "string_match_decomposition",
405:             "timestamp_utc_started": now_utc(),
406:             "config": {
407:                 "train_steps": n_steps,
408:                 "seeds": seeds,
409:                 "anchor_lambda": ANCHOR_LAMBDA,
410:                 "n_matched": n_matched,
411:                 "n_unmatched": n_unmatched,
412:                 "trained_fro": trained_fro,
413:             },
414:             "results": {},
415:             "summary": {},
416:             "verdict": "INCOMPLETE",
417:         }
418: 
419:     t_start = time.time()
420: 

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "@'
import json
from pathlib import Path
p=Path('results/genome_195_untied_input_output_factorial.json')
if p.exists():
    d=json.loads(p.read_text())
    seeds=['42','7','13']
    scratch={s:d['results']['scratch_untied'][s]['final_val_nll'] for s in seeds}
    for arm in ['input_inject_anchor','output_inject_anchor','both_inject_anchor']:
        gaps=[scratch[s]-d['results'][arm][s]['final_val_nll'] for s in seeds]
        print(arm, sum(gaps)/len(gaps), gaps)
    # tied currently one seed
    gaps=[scratch[s]-d['results']['tied_reference'][s]['final_val_nll'] for s in d['results']['tied_reference']]
    print('tied partial', sum(gaps)/len(gaps), gaps)
    input_mean=sum([scratch[s]-d['results']['input_inject_anchor'][s]['final_val_nll'] for s in seeds])/3
    output_mean=sum([scratch[s]-d['results']['output_inject_anchor'][s]['final_val_nll'] for s in seeds])/3
    both_mean=sum([scratch[s]-d['results']['both_inject_anchor'][s]['final_val_nll'] for s in seeds])/3
    print('logic:', 'input_dom', input_mean>=.30 and output_mean<.15, 'output_dom', output_mean>=.30 and input_mean<.15, 'both_needed', both_mean>=.30 and input_mean<.8*both_mean and output_mean<.8*both_mean)
'@ | python -" in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$lines=Get-Content research'"\\prereg\\genome_195_untied_input_output_factorial_2026-04-30.md; for("'$i=1; $i -le [Math]::Min(140,$lines.Count); $i++){ '"'{0}: {1}' -f "'$i,$lines[$i-1] }' in C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome
 succeeded in 329ms:
input_inject_anchor 0.19003910819689432 [0.17640897631645203, 0.19254308938980103, 0.20116525888442993]
output_inject_anchor 0.36208779613176983 [0.3634917736053467, 0.363016813993454, 0.3597548007965088]
both_inject_anchor 0.5482852856318156 [0.533932626247406, 0.557309091091156, 0.5536141395568848]
tied partial 0.5546938180923462 [0.5546938180923462]
logic: input_dom False output_dom False both_needed True

 succeeded in 318ms:
1: # Pre-registration: g195 Untied Input/Output Factorial
2: 
3: **Status:** LOCKED (g194 PASS_DIRECTION confirmed cycle 180; resolves A18 SEV-10 #1 tied lm_head confound)
4: 
5: ## Motivation
6: 
7: All g191/g194 experiments use `tie_word_embeddings=True`. With tying, injecting trained embeddings into `embed_tokens` simultaneously sets the output classifier basis (`lm_head`). The +0.465 nats signal may be OUTPUT-logit class-vector prior, not INPUT embedding/interface geometry. This experiment untie the weights and tests each side independently.
8: 
9: ## Hypothesis
10: 
11: **H1 (Input dominates):** input_inject_anchor gain >= +0.30 nats AND output_inject_anchor < +0.15 nats. The signal is genuinely input embedding geometry.
12: 
13: **H2 (Output dominates):** output_inject_anchor gain >= +0.30 nats AND input_inject_anchor < +0.15 nats. The signal is output logit geometry. "Embedding interface" framing is wrong.
14: 
15: **H3 (Both needed):** both_inject_anchor >= +0.30 but neither alone > 80% of combined. The mechanism requires input-output coherence.
16: 
17: ## Design
18: 
19: ### Arms (5 arms x 3 seeds = 15 cells)
20: 
21: | Arm | embed_tokens init/anchor | lm_head init/anchor | tie_word_embeddings | Tests |
22: |-----|--------------------------|---------------------|---------------------|-------|
23: | `scratch_untied` | Random | Random | False | Baseline |
24: | `input_inject_anchor` | Trained matched rows | Random | False | Input-only |
25: | `output_inject_anchor` | Random | Trained matched rows | False | Output-only |
26: | `both_inject_anchor` | Trained matched rows | Trained matched rows | False | Both (untied) |
27: | `tied_reference` | Trained matched rows | (tied) | True | Tied comparison |
28: 
29: All arms: 8-layer Qwen3-arch, GPT-2 tokenizer, 5000 steps, same data/eval as g191. Anchor lambda=0.01, masked to matched rows only. String-matched embeddings normalized to trained_fro.
30: 
31: ### Smoke test (50 steps, seed=42)
32: 
33: | Arm | NLL | Gap vs scratch |
34: |-----|-----|---------------|
35: | scratch_untied | 7.605 | â€” |
36: | input_inject_anchor | 7.458 | +0.147 |
37: | output_inject_anchor | 7.422 | +0.183 |
38: | both_inject_anchor | 7.198 | +0.407 |
39: | tied_reference | 7.200 | +0.405 |
40: 
41: both_inject_anchor matches tied_reference (design validation). Neither side alone reaches combined effect at 50 steps.
42: 
43: ## Pass/Fail Criteria
44: 
45: **PASS_INPUT:** input_inject_anchor mean gain >= +0.30 AND output_inject_anchor < +0.15. Signal is input embedding geometry.
46: 
47: **PASS_OUTPUT:** output_inject_anchor mean gain >= +0.30 AND input_inject_anchor < +0.15. Signal is output logit geometry. Kills "embedding interface" framing.
48: 
49: **PASS_INPUT_DOMINANT:** input_inject_anchor >= +0.20 AND input > output. Input carries more.
50: 
51: **PASS_OUTPUT_DOMINANT:** output_inject_anchor >= +0.20 AND output > input. Output carries more.
52: 
53: **PASS_BOTH_NEEDED:** both_inject_anchor >= +0.30 AND neither alone > 80% of combined. Coherence required.
54: 
55: **FAIL:** No arm reaches +0.10 nats gain.
56: 
57: ## Universality Level
58: 
59: Level-3 (architecture-specific, within Qwen3 family). Mechanism isolation test.
60: 
61: ## Compute Envelope
62: 
63: - 8-layer Qwen3-arch, GPT-2 tokenizer, ~4.3 GB VRAM peak (untied both-anchor worst case)
64: - 15 cells x ~7 min = ~1.75 hours total
65: - Within COMPUTE.md 4h envelope
66: 
67: ## What a null result means
68: 
69: If PASS_OUTPUT: the entire "embedding interface" narrative is wrong. What we've been calling "trained embedding content" is actually "trained logit class vectors." The signal is output geometry, not input geometry. The training-health diagnostic framing survives but the mechanism shifts.
70: 
71: If PASS_BOTH_NEEDED: the signal requires input-output coherence (consistent codebook). Tying is mechanistically important, not just a confound.

codex
Not clean.

**SEV-7: g195 verdict priority can send g196 down the wrong surface.** Current 13/15 numbers satisfy `PASS_OUTPUT_DOMINANT` by the prereg text, but [genome_195...py](</c:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_195_untied_input_output_factorial.py:266>) checks `both_needed` before output/input dominant, so final likely becomes `PASS_BOTH_NEEDED`; then [g196](</c:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:470>) will choose `surface="both"`, not `output`.

Exact change:
```python
elif input_mean >= 0.20 and input_mean > output_mean:
    verdict = "PASS_INPUT_DOMINANT"
elif output_mean >= 0.20 and output_mean > input_mean:
    verdict = "PASS_OUTPUT_DOMINANT"
elif both_needed:
    verdict = "PASS_BOTH_NEEDED"
```
Move those dominant clauses above `both_needed`, or explicitly decide that "both-needed" overrides the current output-dominance interpretation and update WIKI/launch notes.

g196 post-fix code otherwise looks clean: init-only masking is present, verdict guard is fail-closed on incomplete arms, resume config checks surface/tied/steps/seeds, cutoff eval steps include 50/500/2000, JSON uses `allow_nan=False`, and source is ASCII.

g192 is launch-ready mechanically: config is 28 layers, ~492.0M params by formula, tied weights are enabled, no existing result file to accidentally resume, source compiles, ASCII OK, expected optimizer/grad memory is well under 22GB. Minor doc entropy: stale duplicate prereg `genome_192_28layer_string_match...` still points to a non-existent script, but canonical `genome_192_28layer_replication...` matches the code.


2026-04-30T19:25:47.065107Z ERROR codex_core::session: failed to record rollout items: thread 019ddfd6-2163-7b70-87dd-07d5fc0c608e not found
tokens used
104,749
Not clean.

**SEV-7: g195 verdict priority can send g196 down the wrong surface.** Current 13/15 numbers satisfy `PASS_OUTPUT_DOMINANT` by the prereg text, but [genome_195...py](</c:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_195_untied_input_output_factorial.py:266>) checks `both_needed` before output/input dominant, so final likely becomes `PASS_BOTH_NEEDED`; then [g196](</c:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:470>) will choose `surface="both"`, not `output`.

Exact change:
```python
elif input_mean >= 0.20 and input_mean > output_mean:
    verdict = "PASS_INPUT_DOMINANT"
elif output_mean >= 0.20 and output_mean > input_mean:
    verdict = "PASS_OUTPUT_DOMINANT"
elif both_needed:
    verdict = "PASS_BOTH_NEEDED"
```
Move those dominant clauses above `both_needed`, or explicitly decide that "both-needed" overrides the current output-dominance interpretation and update WIKI/launch notes.

g196 post-fix code otherwise looks clean: init-only masking is present, verdict guard is fail-closed on incomplete arms, resume config checks surface/tied/steps/seeds, cutoff eval steps include 50/500/2000, JSON uses `allow_nan=False`, and source is ASCII.

g192 is launch-ready mechanically: config is 28 layers, ~492.0M params by formula, tied weights are enabled, no existing result file to accidentally resume, source compiles, ASCII OK, expected optimizer/grad memory is well under 22GB. Minor doc entropy: stale duplicate prereg `genome_192_28layer_string_match...` still points to a non-existent script, but canonical `genome_192_28layer_replication...` matches the code.


