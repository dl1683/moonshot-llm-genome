import json, datetime, re

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- QUEUE: e158 row + e157r one-liner + e152R status ----------
q = open("QUEUE.md", encoding="utf-8").read()
m = re.search(r"^\| e158 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e158 | the 2x2 completion | DONE — COMMITTED PASS-2 14:00Z update: TEXTURE (two-pass disclosure; jitter@183 OPEN 0.789, locked@band MID 0.458 straddling); the CONJUNCTION (closure requires novelty x zero-variance) robust across passes; mechanism NOT graft-formation (R48: home graft + open door); the memory emigrates (pass-2 texture) |\n" + q[m.end():]
q = q.replace("re-measures the razor-thin 0.505", "re-measures the straddling locked@band cell (0.458 GPU / 0.546 CPU)", 1)
q = q.replace("| e152R | DWELL RE-SEEDS (3 seeds + 10/12/14-step insert) | QUEUED |", "| e152R | DWELL RE-SEEDS (3 seeds + insert; also the straddle settler) | RUNNING ~13:20Z (GPU) |", 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- paper: the three stale clauses + abstract rescope ----------
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
fixes = [
 ("[e158 RESOLVED: SITE-INDEPENDENT — closure requires novelty AND zero-variance together (jitter@novel open, locked@home open, locked@novel shut); the door's closure ACCOMPANIES novel-site teaching",
  "[e158 COMMITTED PASS-2: TEXTURE — closure requires the CONJUNCTION novelty x zero-variance (jitter@novel OPEN 0.789, locked@home MID 0.458 straddling, locked@novel SHUT 0.102); the door's closure ACCOMPANIES novel-site teaching"),
 ("the 2x2 [e158 DONE: SITE-INDEPENDENT]", "the 2x2 [e158 DONE: TEXTURE-pass-2 — the conjunction]"),
 ("the site-stored (locked-in) memory has NO kill\n    set at ANY CE (92 cells, two sites, both modes; disjoint\n    fact-head populations; saturating redundant ladder) — the memory that\n    generalizes is the memory you can remove.",
  "the site-stored (locked-in) memory has NO kill set at\n    flat CE on ANY surface (92 head-coordinate cells across two sites and\n    both modes; the MLP plane's only kills are organism-priced — mlp_l5\n    92.9% at CE +0.79 — e164) — the memory that generalizes is the memory\n    you can remove at organism-tolerable cost."),
 ("RESOLVED MIXED by e162: BOTH channels kill, each sufficient — functional dependence on the sink's dual role (supplier + guarantor)",
  "RESOLVED MIXED by e162: healthy content fully rescues; total-dose absorption on a flattened profile kills (the per-layer distribution untested, e167 queued) — functional dependence on the sink's dual role, condition-qualified"),
 ("bidirectionally switchable at fixed architecture (both directions are 300 trained steps)",
  "bidirectionally switchable at fixed architecture (both directions are 300 trained steps; conversions demonstrated across the lineage — same-net reversibility is e155R's cell)"),
]
for old, new in fixes:
    if old in p:
        p = p.replace(old, new, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

# ---------- THINKING: T097 residual body pass-1 labels + number fix ----------
t = open("THINKING.md", encoding="utf-8").read()
o1 = "THE HONEST RESTATEMENT: home-site and novel-site grafts dissociate"
n1 = "[PASS-1 RESIDUE LABELED per R49: the verdict label and razor-thin paragraphs below predate the committed pass-2 — the current form is in the PASS-2 UPDATE above; committed: TEXTURE, a=0.789 clean-OPEN, b=0.458 MID-straddling. Number fix: the home-graft content is +0.072 (old-census), not +0.057 (a pass-1 value).]\nTHE HONEST RESTATEMENT: home-site and novel-site grafts dissociate"
assert o1 in t
t = t.replace(o1, n1, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- STATE ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("R49 audit repairs applied")
