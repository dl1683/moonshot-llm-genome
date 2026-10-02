# -*- coding: utf-8 -*-
"""Fold g1c-root: NOTES, T181, ledger C6 stamp, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g1c-root — the wall's root redraw: ROOT-WALL-HOLDS — the 2.74M protection is ROOT-ROBUST at n=2 roots; the fresh draw barely dips; THE ANATOMY DIFFERS, THE WALL DOESN'T (protection without anatomical fidelity) (2026-10-02 ~15:00Z) — DONE

WHAT WE DID: the C6/R58 scope debt paid — the g1bR machinery at a
FRESH ROOT (install gen 24314 the only stochastic delta; cons 10901
and wash 10902 held); all 15 gates PASS; one disclosed gate
amendment (G-INST re-anchored same-instrument; G-CONS read on the
final root; the first TEXTURE pass re-executed bit-identical per
the precedent).

WHAT WE SAW (T181): THE FRESH ROOT LANDED STRONG (ruler 0.9289;
g-12 0.9026 vs the locked 0.9156; L2 18.6 away — no coin-flip
deviation). W1 HELD FLAT 0.82 -> 0.94 through +300 (the flat-phase
min 0.9265; the strict 0.9eq co-report ALSO holds) while C died by
+1 under bit-identical inputs. THE CLAIM'S NEW STAMP: "THE WALL AT
R=0.7 HOLDS THE BATTERY CHANNEL (n=3 WASH DRAWS AND n=2 ROOT
DRAWS)". THE CROSS-ROOT GEM: THE +2 DIP NEARLY VANISHED on the
stronger draw (min-at-+1 0.8214 vs the reference family's +2 dips
0.746-0.803) — T178's STRENGTH-SOFTENS-THE-SHOCK at the reference
scale; held30 IMPROVED under the wall (0.653 -> 0.758). THE
ANATOMY FINDING: the fresh root writes a DIFFERENT anatomy
(d183-independent g-12; negative A129) with IDENTICAL wall
behavior — THE PROTECTION IS NOT ANATOMY-SPECIFIC: the wall guards
the function, not the wiring. THE HONEST OPEN RUNGS: base-seed and
cons-seed redraws; the draw variance itself unmeasured (2-of-2
passing); W2/W3/noise stay n=1 textures.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T180 —"
card = """## T181 — g1c-root: the wall guards the function, not the wiring (2026-10-02 ~15:00Z)

The last carried wall debt pays: ROOT-WALL-HOLDS — the protection
replicates on a genuinely different root draw (L2 18.6 away), the
fresh root barely dipping where the reference family dipped, and
the strict form holding too. TWO FINDINGS BEYOND THE REPLICATION:
(1) T178's cross-draw law (strength softens the shock) now
confirmed at the reference scale — the dip-vs-strength relation
holds at 2.74M and 10M alike; (2) THE ANATOMY DISSOCIATION — the
fresh root's fact runs on DIFFERENT WIRING (d183-independent,
A129-negative) yet the wall's behavior is IDENTICAL: THE WALL
GUARDS THE FUNCTION, NOT THE WIRING — protection is
anatomy-independent, which is exactly what an architectural (not
anatomical) claim wanted. THE WALL'S LEDGER, FINAL: battery-channel
protection at n=3 wash draws + n=2 root draws + 2 scales (the 10M
verdict: direction-only); the tax priced at both scales; the
mechanism (a displacement budget) arithmetic-bounded; and now the
provenance-independent protection. THE PROGRAM'S LAW ONE MORE
TIME: the protection (a shape) replicates across every axis tried;
the dip and the tax (heights) vary with the draw.

"""
assert anchor in t and "## T181" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

c = "scratch/claims_ledger.md"
s = io.open(c, encoding="utf-8").read()
old = "licensed n=3 wash-draws ONE root"
assert old in s, "c6 stamp"
s = s.replace(old, "licensed n=3 wash-draws x n=2 root draws (g1c-root: ROOT-WALL-HOLDS; the anatomy differs, the protection identical — the wall guards the function, not the wiring)", 1)
io.open(c, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
i = q.index("| g1c-root |") if "| g1c-root |" in q else q.index("| g1c_root |")
row = q[i:]; row = row[:row.index("\n")]
q = q.replace(row, "| g1c-root | THE WALL'S ROOT REDRAW | DONE 15:00Z (T181: ROOT-WALL-HOLDS — the protection root-robust at n=2; the fresh draw barely dips (strength-softens-the-shock at 2.74M); THE ANATOMY DIFFERS, THE WALL DOESN'T — the wall guards the function, not the wiring; C6's stamp upgraded to n=3 wash x n=2 root) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T15:01:00Z"
st["current_experiment"] = ("g1c-root FOLDED (T181: ROOT-WALL-HOLDS — the wall guards the function, not the wiring). "
                            "Fleet: e211 (CPU, the walled-band question) + e182c-2 DISPATCHING (GPU freed: the "
                            "template-locus + fresh corpus draws)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g1c-root folded")
