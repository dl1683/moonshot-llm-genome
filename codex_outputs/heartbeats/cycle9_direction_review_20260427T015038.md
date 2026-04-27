1. `g159 INCOMPLETE` is not enough to declare class-extension dead. It is enough to declare the **rank-32 lesion assay** suspect if Falcon shows the same failure mode. The locked `g159b` prereg from **2026-04-27** exists exactly for this case: salvage only when the local lesion does not bite. If Falcon instead gives a clean, bite-capable result and still no transport/local asymmetry, then call the class-extension claim dead and do not rank-sweep.

2. Do **not** run `g158` and `g160` in parallel. `g158` is the cheapest remaining high-discrimination theory test, while `g160` is lower on the §0.1 axis and is not launch-clean anyway: its pre-flight still flags evaluator/FLOP-match/dedup/CtQ issues, and concurrent running would muddy throughput-sensitive measurements.

3. Minimum chain that maximizes §0.1 from the current **6/10**:
- Finish Falcon on `g159`.
- Run `g158` PILOT next.
- If `g158` is `PILOT_KILL_158`: skip `g159b`, pivot immediately to `g160 -> g155`.
- If `g158` is directional/weak and Falcon matches the same lesion-underbite pathology: run one **bounded** `g159b` rank-sweep, then `g160`.
- If `g158` is directional/weak but Falcon is a clean `g159 KILL`: skip `g159b`, go straight to `g160`.

**Concrete sequencing decision:** finish the current `g159` Falcon leg, then launch `g158` as the only next GPU job; defer both `g159b` and `g160` until the `g158` verdict.