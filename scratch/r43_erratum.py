import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- REVIEWS.md: erratum in R43 auditor section ----------
r = open("REVIEWS.md", encoding="utf-8").read()
o = """- NEW audit finding: NO model weights are persisted anywhere in the repo
  (only an unrelated e055 scratch cache). The critic's "eval-only on existing
  checkpoints" discriminator is therefore NOT free — nets must be regenerated
  (deterministic recipes exist). Process fix adopted: every future experiment
  saves phase-boundary checkpoints under runs/eNNN/ckpts/ (small, loadable)."""
n = """- AUDIT FINDING, CORRECTED BY ERRATUM (~07:25Z): the lead's original claim
  "NO model weights are persisted anywhere" was WRONG — the search was
  under-scoped (find -maxdepth 2 + never reading .gitignore). In fact
  runs/checkpoints/ holds 102 phase checkpoints (convention: eNNN_<phase>.pt,
  gitignored so commits never show them). The OPERATIVE finding stands,
  narrower: the consolidation arc (e109–e121) saved NO nets — the newest
  checkpoint is e117's, and no e109/e113/e120/e121 arm net exists. So the
  critic's discriminator still required regenerating the arc's fine-tunes,
  but from existing roots (e120's cited base e082_b43_install.pt IS there;
  e044 zephyra installs for the older line). Process fix, reframed: restore
  the OLD convention (runs/checkpoints/eNNN_*.pt) for every future
  experiment — e131 was mid-dispatch and was corrected by message."""
assert o in r, "erratum anchor"
r = r.replace(o, n, 1)
open("REVIEWS.md", "w", encoding="utf-8").write(r)

# ---------- STATE.json ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("erratum in")
