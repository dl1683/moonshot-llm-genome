import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T085 — E142:"
card = """## READING MAP — registered before e147/e150 report (2026-09-28 ~09:08Z; both mid-compute, no data read)

The two running experiments carry the frame's two load-bearing
questions. Their outcomes are pre-mapped so the folds are shaped
before numbers exist:

**e147 (width ladder) — what each bar would MEAN:**
- INVARIANCE-CAUSAL (A(w) monotone decreasing, NR(w) onsetting
  within one bin of the same w*): the credit-competition law
  returns AS A DOSE-RESPONSE LAW — the taxonomy gets its switch
  variable, T079 is redeemed from its dial-death, and the second
  paper's claim 2 becomes quantitative (a critical width w*, not
  a binary).
- DEAD-AGAIN (A flat, or routing onsets while A >= +0.15): two
  independent switches — the taxonomy survives as TYPES but its
  mechanism dies for good; W013's protagonist loses its key-
  selection rule permanently.
- SEED-COVERAGE (routing peaks at +-8, collapses at +-32/64):
  W010's ghost wins POSTHUMOUSLY — seeds have finite spatial
  reach, and 'omnipresence' acquires a spatial constant (the
  seed-reach radius). A new number either way.

**e150 (flat-CE route test) — the stakes, and the reframe if it
goes negative:**
- FLAT-CE-ROUTE: the route is isolable from wreckage; T081/T082
  unbound; the day's noun stands.
- ALL-KILLS-WRECK: T081/T082 HARD-BOUNDED — but the honest
  reframe is already worth savoring: 'the fact's readout
  requires the net's most load-bearing coordinate' is itself a
  finding — CONSOLIDATION AS A MOVEMENT FROM THE REMOVABLE TO
  THE IRREMOVABLE. The graft (row 129) was surgically deletable;
  the native organ (row 0) is unremovable without organism
  damage. What the lab called migration would then be: the
  memory moving from editable tissue into essential tissue —
  the OPPOSITE of surgical memory, and arguably the point of
  consolidation. The bio-echo sharpens: childhood memories
  resist erasure partly because they live in early, load-bearing
  circuitry. If ALL-KILLS-WRECK fires, this reframe — not a
  retreat — is the fold.
- DIRECTION-CONSULTED (fact-at-position-0 scramble kills): W014's
  tenant dies; direction-independence was an artifact of the
  fact never living at position 0.
- PRESENCE-AT-NOVEL (perm@g-12 spares >=80%): presence-only
  extends to novel geometry — the strongest single supporting
  cell the frame can add.

Adjudication on the registered bars verbatim; no shopping.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, card, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("reading map registered")
