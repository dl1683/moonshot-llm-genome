**Q1.** For the project right now, a clean `g172 FAIL/washout` is slightly better than a PASS. It keeps the law tight: donor information must remain in the objective near convergence, not merely appear early.

A `g172 PASS` is more exciting for future engineering because it suggests cheaper warmup-only KD can seed capability. Trap: do not call that zero-step transfer or “no retraining”; it is still 2000 KD steps plus CE continuation.

A `g172 FAIL` is cleaner theory. Trap: do not overclaim if the CI crosses zero. That is tied/inconclusive, not confirmation. Only a CI entirely below the washout bar supports the continuous-constraint law.

**Q2.** Keep `g173` bundled. It is not overloading because the three validations are mutually dependent:

- cross-arch alone = credibility row, not flagship
- FLOP accounting alone = weak without end-task capability
- C3_macro alone = just another KD result unless it transfers across architecture with compute savings

The right change is not splitting. The right change is decomposed verdicting inside one experiment: separately report `cross_arch_transfer`, `end_task_gain`, and `matched_capability_FLOP_ratio`, then have a strict combined flagship verdict. That prevents a partial result from becoming narrative fog.

**Q3.** No procurement progress found. Repo and memory state still say wall-power is blocked. `g155` remains invalid as a headline run without an external AC meter. Practical buy remains a logging smart plug class device; Yokogawa WT310E is gold standard.

**Q4.** Tighter theoretical statement:

Let recipient SGD minimize  
`L_t(theta; D) = L_data(theta) + lambda_t R(T(theta), Phi_D)`  
where `Phi_D` is donor information and `T(theta)` is the recipient axis: weights, logits, attention maps, etc.

Persistent transfer requires nonzero donor-conditioned gradient information during the basin-selection/convergence phase:

`Integral lambda_t I(grad_theta R_t ; Phi_D | theta_t, batch_t) dt` must remain order-one through late training.

Optimization form: SGD is an SDE  
`dtheta = -grad L_data dt - lambda_t grad R_D dt + noise`.  
With constant `lambda`, the stationary distribution is donor-tilted:  
`p_lambda(theta|D) proportional exp(-(L_data + lambda R_D)/tau)`.  
If `lambda -> 0` early and enough CE-only mixing remains, `p(theta_T|D)` relaxes toward scratch. That predicts: constant KD/weight anchors work; zero-step and decayed scaffolds wash out; cross-arch weight anchors fail without shared coordinates; cross-arch KD can work through shared output space.

**Queue score.**

`g173`: 8.1-8.5 ceiling, highest runnable.  
`g172`: 7.9-8.0, mechanism clarifier only.  
`g155`: 8.5 ceiling, hardware-blocked.  
`g171`: ~7.2, demoted unless `g172` creates ambiguity.  
`g162/g166`: ≤6.8, do not prioritize.

**One concrete change:** keep `g173`, but add explicit three-axis sub-verdicts before launch so the single experiment consolidates without hiding failure modes.