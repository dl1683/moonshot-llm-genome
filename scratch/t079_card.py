t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T077 — E131:"
t079 = """## T079 — the credit-assignment law: keys strengthen in proportion to their INVARIANCE across error-bearing windows (2026-09-28 ~07:12Z)

The new frame's sharpest internal tension (handed to the R44
critic, worked here in parallel): e109's own data says locked
matched-mass replay — MORE error at ONE address — consolidated
WORSE through deletion than jitter. If consolidation follows error
placement (T076), why does concentrating the error fail? Because
error placement decides WHERE THE FACT IS TAUGHT; a second
variable decides WHICH KEY EARNS THE GROWTH: the invariant
features across the error-bearing windows. Under jitter, the
fact's position varies window-to-window, so the only stable
predictors of the fact's presence are CONTENT and ROW-0
PARTICIPATION (the sink attends in every window regardless of
offset) — the gradient's cheapest strengthening lands on the
invariant key, and the fact re-keys to row 0. Under locked
replay, position 129 is perfectly predictive — the cheapest
descent strengthens the 129-key, and the fact digs in. Under
erasure, there is no fact-error at all; recovery re-strengthens
whatever predicts the recovering expression — the address again.
One law, three roads: JITTER makes content+sink invariant (key
migrates to row 0); LOCK makes position invariant (key stays);
ERASE makes necessity point at the address (key tightens). This
also REABILITATES position diversity in precise form: diversity
was never an ingredient of consolidation — it is the CONDITION
UNDER WHICH THE POSITIONAL KEY LOSES THE CREDIT COMPETITION. And
it absorbs the splice result: all splice windows put the fact at
183 -> 183 invariant -> site-locked (W011's predicted (a)).

ALTERNATIVE EXPLANATIONS: (1) GRADIENT-VOLUME: row 0 grows simply
because it is most-attended (sink receives the most attention
mass, hence the most gradient) regardless of invariance — but
then LOCKED replay should grow row 0 equally (its windows also
contain row 0 with similar attention), and L should be as
migrated as R; L brakes -0.509 like E, which contradicts this
unless braking and keying dissociate. (2) MASS-TRANSFER: error
flows to row 0 in proportion to attention mass during ANY fact
training — same prediction as (1), same contradiction. (3)
TWO-FACTOR LUCK: R's row-0 growth was a seed-promotion accident
(W011's (c)) not requiring invariance at all.

DISCRIMINATING OBSERVATION (already queued as e140, eval-only on
saved nets — no new compute needed): row-0 content strength
across twin-start / L@150 / R@150 / R@300. CREDIT-ASSIGNMENT
predicts R@150 >> L@150 (jitter's invariance competition vs
locked's) and R@300 > R@150 (dose-monotone). GRADIENT-VOLUME and
MASS-TRANSFER predict R@150 ~= L@150 (equal steps, equal sink
attention). TWO-FACTOR-LUCK has no dose signature. One number
pair (R vs L row-0 strength) separates all three. REGISTERED
PREDICTION: R@150/L@150 row-0 strength ratio >= 2 with L
install-level-flat => credit-assignment; ratio < 1.3 =>
gradient-volume inherits and the law dies.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t079, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# strengthen e140's queue row with the L-cell emphasis
q = open("QUEUE.md", encoding="utf-8").read()
o_q = "Bars: R-ROW0-MONOTONE fires if row-0 strength rises with jitter dose and not with erase cycles; E-NEVER-KEYS fires if all E checkpoints row-0-null (address-dig-in confirmed at the key level); PARTIAL if E keys late |"
n_q = "Bars: R-ROW0-MONOTONE fires if row-0 strength rises with jitter dose and not with erase cycles; E-NEVER-KEYS fires if all E checkpoints row-0-null (address-dig-in confirmed at the key level); PARTIAL if E keys late. [T079 adjudication rides the same cells: R@150/L@150 row-0 ratio >=2 with L flat => CREDIT-ASSIGNMENT (keys strengthen by invariance across error windows); ratio <1.3 => GRADIENT-VOLUME (sink grows under any training) and the law dies] |"
assert o_q in q
q = q.replace(o_q, n_q, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)
print("T079 in; e140 row carries the adjudication")
