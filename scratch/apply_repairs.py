import json, datetime, re

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

reps = json.load(open("scratch/r49_repairs.json", encoding="utf-8"))
applied = 0
for path, pairs in reps.items():
    s = open(path, encoding="utf-8").read()
    for pair in pairs:
        old, new = pair["old"], pair["new"]
        if old in s:
            s = s.replace(old, new, 1)
            applied += 1
        else:
            print(f"MISS in {path}: {old[:60]}")
    open(path, "w", encoding="utf-8").write(s)

# queue: e175 + e166 invalidation
q = open("QUEUE.md", encoding="utf-8").read()
o_q = "| e173 | THE CLOSURE PARTITION"
if "e175 | THE RECOVERY" not in q:
    row = ("| e175 | THE RECOVERY-KINETICS CELL (R49 critic — the decisive SUBSTANCE-SURVIVES discriminator) | READY (CPU; one short training) | "
           "few-step fact-replay (10-30 steps) on the N2-killed net: FAST-RECOVERY = g0 restores to ~0.78 in <=30 steps (thin access lesion — T098's strong reading licensed); "
           "FULL-PRICE = ~300 steps needed (substance degraded with access — the strong reading dies) |\n")
    q = q.replace(o_q, row + o_q, 1)
m = re.search(r"^\| e166 \|[^\n]*\n", q, re.M)
if m and "INVALID" not in q[m.start():m.end()]:
    q = q[:m.start()] + ("| e166 | the inverse event | INVALID-BY-INSTRUMENT (R49: the door battery never reads rows 183-189 — the zero was a tautology; "
                         "site/head cells real; long-window rerun rides e173's corrected design) |\n") + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print(f"applied {applied} replacements + queue updates")
