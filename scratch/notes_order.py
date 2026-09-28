import re, json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

n = open("NOTES.md", encoding="utf-8").read()

def extract_block(text, header_start):
    end = text.find("\n## E", header_start + 1)
    if end == -1:
        end = len(text)
    return text[header_start:end], header_start, end

# Move E091 (2026-09-26) from above the 09-28 block to above E089 (its 09-26 home)
h91 = n.find("## E091 — reverse-transplant")
assert h91 != -1
block91, s91, e91 = extract_block(n, h91)
n = n[:s91] + n[e91:]
h89 = n.find("## E089 — mass-response curve")
assert h89 != -1
n = n[:h89] + block91 + n[h89:]

# Move E119 (07:20Z) to its correct newest-first slot: between E133 (07:45) and E131 (07:05)
h119 = n.find("## E119 — migration head-to-head")
assert h119 != -1
block119, s119, e119 = extract_block(n, h119)
n = n[:s119] + n[e119:]
h131 = n.find("## E131 — the re-keying census")
assert h131 != -1
n = n[:h131] + block119 + n[h131:]

open("NOTES.md", "w", encoding="utf-8").write(n)

# verify order
heads = [l for l in n.splitlines() if l.startswith("## E") and "(date)" not in l]
print("top 12 order:")
for h in heads[:12]:
    print("  ", h[:80])
