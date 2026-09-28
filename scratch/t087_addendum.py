import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

t = open("THINKING.md", encoding="utf-8").read()
o = """FOR THE PAPER: claim 2's switch upgrades from 'error-position
variance' to 'a binary cliff at zero-vs-any variance' — simpler
to state, stronger to show (one figure, two rungs). The
mechanism hunt for WHAT variance does (why does one moved
window cancel the address key?) reopens at the head level
(e133's L0H3 class) with the P-b cell (one net, both types) as
the taxonomy's last structural confound — dispatched next."""
n = """MECHANISM NOTE (the negative-posterior hypothesis, worked on
paper ~09:38Z): why does variance make the address key go
NEGATIVE (a brake) rather than merely shrink? Because once the
fact appears at other rows, the address key's firing ANTI-
correlates with the fact's presence there — when the fact sits
at 121, a read keyed on 129 carries other content; the address
becomes a below-prior predictor, and suppressing it sharpens the
attention budget. The pure Bayesian form predicts brake
MAGNITUDE tracks the miss-rate (w=1 misses 2/3; w=64 misses
~127/128 -> much stronger brake) — but the data show a plateau
(-0.03..-0.13, no width trend): suppression SATURATES once the
key is unreliable at all. The cliff again, now inside the brake:
demotion is all-or-none, not graded. Signature already visible
in e147's A-column; no new compute needed to see it.

READING FORK FOR e151 (pre-registered before its report): if
TWO-DOOR-ADDITION fires, the cliff is PER-MEMORY — each fact's
type is decided by its own training variance, and one net holds
mixed types. If ROUTE-OVERWRITES fires, the cliff is PER-NET —
any zero-variance teaching collapses geometry-generalization
globally, implying a shared substrate the locked re-teach
destroys. SITE-REJECTED would mean the geometry-general
structure absorbs zero-variance teaching without growing a site
— the cliff ran once and cannot run again on this net.

FOR THE PAPER: claim 2's switch upgrades from 'error-position
variance' to 'a binary cliff at zero-vs-any variance' — simpler
to state, stronger to show (one figure, two rungs). The
mechanism hunt for WHAT variance does (why does one moved
window cancel the address key?) reopens at the head level
(e133's L0H3 class) with the P-b cell (one net, both types) as
the taxonomy's last structural confound — dispatched next."""
assert o in t
t = t.replace(o, n, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("T087 mechanism note + e151 fork in")
