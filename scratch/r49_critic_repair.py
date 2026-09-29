import json, datetime, re
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
t = open("THINKING.md", encoding="utf-8").read()
o1 = "## T099 — E166: zero of the closure — the third great asymmetry, and the fork narrows honestly (2026-09-28 ~13:50Z)"
n1 = "## T099 \u2014 [INVALID-BY-INSTRUMENT per R49 critic \u2014 the +0.0000 was a PROMPT-GEOMETRY TAUTOLOGY: the door battery's prompts span positions 0-141; the surgery edits wpe rows 183-189; a causal transformer never reads those rows for those prompts. g-12 was bit-identical to 17 figures EVEN ON THE ROOT'S OPEN DOOR \u2014 the dial was structurally blind. DOOR-STAYS-SHUT fired as a foregone conclusion; 'the graft rows carry zero of the closure' is UNSUPPORTED by e166; 'unreopenable-by-surgery' keeps only e153's sub-bar transplant arm (n=1). RULE 12's founding bite, a third time, at the same coordinate 183. The licensed graft-not-closer support remains T097's home-graft counterexample \u2014 itself the straddling cell. e166's long-window rerun rides e173's corrected design.] E166: the inverse event (2026-09-28 ~13:50Z)\n\nWHAT SURVIVES OF THE RUN: the site-read cells (surgery visible there: 0.998 -> 0.599) and the head-ablation cells (which DO move the door \u2014 reader3-zero 0.0041) are real; the ROW cells and the headline are void. The third-asymmetry noun is WITHDRAWN to a single-arm suggestion pending licensed evidence."
assert o1 in t, "T099"
t = t.replace(o1, n1, 1)
o2 = "## T098 — E164: access severed, substance intact"
n2 = "## T098 \u2014 [R49 BOUND: the kill was 71%, not 100% \u2014 the residual is a 29%-ALIVE readout; the saturated organ dials cannot distinguish storage-support from access-support (same-circuit-at-29%-amplitude predicts the same cells); weight-level intactness is construction-tautological; profile rank 0.587 is moderate. The DECISIVE cell queued as e175: few-step fact-replay on the killed net \u2014 fast recovery => thin access lesion; full-budget => substance degraded with access] E164: access severed, substance intact (bounded)"
assert o2 in t, "T098"
t = t.replace(o2, n2, 1)
o3 = "## T100 — E154: overwrite, not share — and the anchors may have done it (2026-09-28 ~14:00Z)"
n3 = "## T100 \u2014 [R49: the headline noun OVERWRITE-NOT-SHARE STRUCK \u2014 it asserts the capacity mechanism the run's own text says it cannot separate (the anchor-contradiction confound); the verdict TEXTURE was correct-by-registration; the F1-side rider readings are floor-ratio artifacts (base 0.0017, leak ~5e-4) \u2014 only the F2-side riders stand (N2 spares F2; F2 diffuse)] E154: F1 annihilated under an anchor-confounded protocol (2026-09-28 ~14:00Z)"
assert o3 in t, "T100"
t = t.replace(o3, n3, 1)
print('part1 written')