import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()

# Finding 1: compass now causal
o1 = """1. **The fact consolidates where its error is placed (T076; the
   E120 inversion).** E120's splice arms "failed to consolidate" —
   except they hadn't: e131's 183-geometry read found them at
   0.989/0.988, consolidated at full strength at the address the
   battery never read. The R43 critic's error-location theory,
   promoted to T076, is the lab's consolidation-law candidate:
   error placement is the compass. The position-diversity
   "ingredient" retired (resolved cleanly by e139/T082); what
   diversity actually does belongs to finding 5."""
n1 = """1. **The fact consolidates where its error is placed — now
   CAUSAL (T076 -> T084; the E120 inversion, then e143).** E120's
   splice arms "failed to consolidate" — except they hadn't:
   e131's 183-geometry read found them at 0.989/0.988, at the
   address the battery never read (the R43 critic's error-location
   theory, caught before the proving data existed). e143 then made
   the compass causal: parking the fact's error at positions 5-13
   (zero diversity, sink-adjacent) built a site-store AT 5-13 with
   row-0 at baseline — choose the error's site, choose the store's
   site; proximity piggybacking dead. The position-diversity
   "ingredient" retired; what diversity actually does (choose
   routed vs site-stored) belongs to finding 5."""
assert o1 in r, "f1"
r = r.replace(o1, n1, 1)

# Finding 2: add e142 ROW-0-ALWAYS + law promotion + flat-CE bound
o2 = """The migration wrote nothing in the
   destination row — everything lives in readout weights (e133:
   fact-specific residue 84.5% heads (locality-filtered, report-only table))."""
n2 = """The migration wrote nothing in the
   destination row — everything lives in readout weights (e133:
   fact-specific residue 84.5% heads (locality-filtered, report-only table)).
   And e142's census made it HISTORY: ROW-0-ALWAYS — 13/13 nets,
   every dose — there was never an address-only phase, W011's
   savor promoted to LAW (consolidation = share-growth of the
   largest seed), and the ADDRESS itself was protocol-made
   (direct/natural-exposure installs carry row 129 at NULL; the
   five-day address story was a property of the masked-replay
   protocol). BOUND (R45 critic, held open): every fact-killing
   intervention in the row-0 plane sits at CE +0.70 to +4.44 —
   no flat-CE fact-kill exists yet, so 'routed' vs 'dies when the
   net dies' is not fully separated until e150's cells land; the
   install-restore probe was also a no-op by norm (the presence
   conclusion rests on the perm/halfnorm/mean riders)."""
assert o2 in r, "f2"
r = r.replace(o2, n2, 1)

# Finding 3: add lineage-confound bound
o3 = """3. **Two memory types: ROUTED vs SITE-STORED (T082; e139).** Row 0
   routes; the site stores."""
n3 = """3. **Two memory types: ROUTED vs SITE-STORED (T082; e139) —
   bounded twice since.** Row 0 routes; the site stores — but the
   discriminator crosses net lineages and trained-vs-novel status
   (no single net holds both types yet), and T085's history
   rewrite reframes the site itself as protocol-sculpted (locked
   replay grows a site; natural placement grows row 0). The type
   claim stands on e139's cells and e141's novel-geometry collapse,
   pending e147's width ladder and e150's flat-CE test."""
assert o3 in r, "f3"
r = r.replace(o3, n3, 1)

# Finding 5: e147/e150 in flight
o4 = """   L +0.33); and L-CYCLED retired "erasure digs in" outright
   (locked cycles thin identically — cycle damage, not erasure).
   e143 carries invariance's last causal stand."""
n4 = """   L +0.33); and L-CYCLED retired "erasure digs in" outright
   (locked cycles thin identically — cycle damage, not erasure).
   e143 then WON invariance's last causal stand (COMPASS-CAUSAL),
   and the width ladder (e147) is measuring the law's
   dose-response — whether address-key death and route birth
   CO-ONSET at a critical jitter width — as this report's final
   open cell, alongside e150's flat-CE verdict."""
assert o4 in r, "f5"
r = r.replace(o4, n4, 1)

# Coverage line update
o5 = "Covers the 2026-09-28 session 05:00–08:10Z+ (E119-E121 folds, E131,"
n5 = "Covers the 2026-09-28 session 05:00–09:45Z+ (E119-E121 folds, E131-E143, E140-E142,"
assert o5 in r
r = r.replace(o5, n5, 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("report updated")
