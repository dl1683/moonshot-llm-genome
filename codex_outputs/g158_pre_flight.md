Pre-flight audit is written to [g158_pre_flight.md](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/codex_outputs/g158_pre_flight.md).

Verdict is `NO-GO` as written. The blockers are:
- Severity 10: exact-FLOP matching is not implemented; current schedule is token-budget matched and under-trains the minimal arm by about `5.2x` to `5.5x` relative to the locked spec.
- Severity 9: C4 train, C4 eval, and LR-selection data are all sliced from the same shuffled C4 train stream, with no 13-token dedup audit.
- Severity 8: NaN handling can still yield a prereg verdict on partial data, and failed cells would serialize as non-standard JSON `NaN`.

I also recorded the clean checks: the script’s runtime strings are ASCII-safe for Windows `cp1252`, and Wikitext OOD correctly uses the validation split. Honest compute estimate in the report: current invalid script is probably `35-50 min`; a spec-compliant launch is closer to `1.6-2.0 hr`.