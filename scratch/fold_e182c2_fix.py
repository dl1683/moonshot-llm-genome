# -*- coding: utf-8 -*-
"""Fix the e182c2 fold: dedupe NOTES (run 2 doubled the entry), add the QUEUE row, stamp STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
marker = "\n## e182c2 — phase 2:"
first = n.index(marker)
second = n.find(marker, first + 1)
if second >= 0:
    # the duplicate block spans from the second marker to the following separator
    end = n.index("\n---\n", second)
    n = n[:second] + n[end:]
    io.open("NOTES.md", "w", encoding="utf-8").write(n)
    print("NOTES deduped")
else:
    print("NOTES already single")

q = io.open("QUEUE.md", encoding="utf-8").read()
if "e182c2" not in q:
    j = q.index("\n", q.index("| e182c |")) + 1
    q = q[:j] + "| e182c2 | THE TEMPLATE LOCUS + FRESH DRAW | DONE 16:05Z (T183: TEMPLATE-GENERAL + DRAW-REPLICATES — the locus the few-shot-following faculty; the graded order ctrl < template < nearrel; THE CROSS-WASH FREE FIND: one battery's decline identical across two washes to three decimals — path-independence, the e188 echo at 124M) |\n" + q[j:]
    io.open("QUEUE.md", "w", encoding="utf-8").write(q)
    print("QUEUE row added")
else:
    print("QUEUE already has row")

s = json.load(io.open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = "2026-10-02T16:06:00Z"
s["current_experiment"] = ("e182c2 FOLDED (T183: the few-shot locus at n=2 draws + 2 forms; the cross-wash "
                            "stability the new open object). Fleet: e212 (CPU, the pristine band) + the GPU free."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(s, indent=2, ensure_ascii=False))
print("STATE stamped")
