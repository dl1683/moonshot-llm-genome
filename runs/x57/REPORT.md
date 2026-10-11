# X57 — THE KEY'S CHANNEL (the role split at full dose)

**VERDICT: TARGET-CARRIES** — tonly PROTECTS, conly EXECUTED

Date (datetime.now(UTC)): 2026-10-11T08:49:12Z | smoke: False | CPU desk only, threads 4, zero CUDA (sibling x58 live, never touched)

## The classification scheme (disclosed exactly, with counts)

- The canon cons batch (32 windows/step): **16 name-bearing** jittered-pool windows ([130-char authored pre][7-char NAME][119-char post], j in {-8,-4,0,4,8}) + **8 paired-original anchors + 8 randoms** (nameless).
- The canon loss's two role terms: **TARGET channel** = the 7 name-EMISSION positions per name window (G1's `nm` — the name as the attended target; 16x7 = 112 supervised tokens/step); **CONTEXT channel** = the nameless windows' full supervision (G1's `cm`; 16x255 = 4080/step). The canon union (the k=16 ruler's loss) = 4192/step.
- **tonly** = the target channel ALONE (112/step; the context windows neutralized — drawn, out of the loss). **conly** = the context channel MAXIMALLY EXTENDED: the exact complement on the name windows (the name as mid-window background — reading through it; 16x248) + the nameless windows (16x255) = 8048/step; the emission supervision neutralized.
- Bitwise (G_MASKUNION): tonly and conly are disjoint and cover the full batch (8160); the canon mask == tonly + conly's nameless part. Both arms all-16-t keyed (112 TAVIREN tokens in stream — the k=16 dose), draws bit-equal (G_DRAWS).

## The two arms' fates

| arm | s25 | s125 | s300 | verdict | s1 (armor) |
|---|---|---|---|---|---|
| tonly | 0.999955 (ALIVE) | 0.999994 (ALIVE) | 1.000000 (ALIVE) | **PROTECTS** | 0.984782 |
| conly | 0.003532 (DEAD) | 0.002916 (DEAD) | 0.000798 (DEAD) | **EXECUTED** | 0.002382 |

The committed poles: k=0 (e345, all-Z canon) EXECUTED 0.003601/0.000473/0.003064; k=16 ruler SURVIVED 0.684341/0.309011/0.715929 (s1 0.7490 vs k0 0.0068).

## The reads beyond the primary

- Dominance (tonly - conly): s25=+0.9964 (tonly); s125=+0.9971 (tonly); s300=+0.9992 (tonly).
- The guest channel (p(Z), co-read): tonly 5.50e-08/1.65e-08/2.84e-09; conly 1.58e-04/8.75e-05/4.43e-06 — dead-class in both (zero Z-dose; the committed poles: k0 RISE 0.528/0.580/0.635, k16 corner DEAD 1.7e-05/2.4e-04/9.7e-05).
- The rider: s1: |W| 9.11/9.10, dW 0.01; s25: |W| 12.84/11.70, dW 1.14; s125: |W| 13.93/15.50, dW 1.56; s300: |W| 14.17/19.59, dW 5.42 — WITHIN-ENVELOPE (co-report only).

## The registration, adjudicated

- BARS (frozen at birth): TARGET-CARRIES / CONTEXT-CARRIES / ROLE-SYMMETRIC.
- PARITY: LAB LEAN TARGET-CARRIES weakly; COUNTER ROLE-SYMMETRIC (both carried verbatim in metrics.registered).
- P-x57a (mine, registered BEFORE compute): TARGET-CARRIES weakly-to-moderately — scored HIT.

## Gates

| gate | pass |
|---|---|
| G_MD5 | PASS |
| G_QUOTES | PASS |
| G_NAMEFREE | PASS |
| G_SPLICE | PASS |
| G_BATTERY | PASS |
| G_POOL | PASS |
| G_POOLTWIN | PASS |
| G_ROOM | PASS |
| G_STREAMIDENT_Z | PASS |
| G_STREAMIDENT_T | PASS |
| G_ROLEMASK | PASS |
| G_MASKUNION | PASS |
| G_DRAWS | PASS |
| G_ENDPOINTS | PASS |
| G_RIGCONST | PASS |

## The honest caveats

- If conly protects: the mechanism decomposes (nameless-cm rehearsal vs read-through) beyond two arms — the named successor (cm-only 4080 / read-through-only 3968).
- The supervised-token asymmetry (112 vs 8048/step) is the role cut's own shape; if conly wins, supervision mass is a named confound.
- n=1 seed per arm, one lineage, one subject; the s125 knife (the ruler's own 8% ALIVE margin) is the fragile margin — the dominance co-report keeps it legible.

## Post-adjudication reading notes (wording-only; no bar, scoring or metric touched)

- **tonly EXCEEDS the ruler itself** at every fate step (0.999955/0.999994/1.000000
  vs the canon union's 0.684341/0.309011/0.715929): the target channel ALONE is
  not merely sufficient — the canon's context channel (cm, 4080 tokens/step)
  DILUTES the maintenance it does not carry (the thinner 112-token loss
  concentrates the clip-normalized gradient on the read's own coordinates; no
  s125 dip at all appears in tonly).
- **The armor observable is target-channel-carried**: tonly's s1 0.984782 sits
  ABOVE even the canon's 0.748952; conly's 0.002382 is k0-class (0.006756).
- **conly's write approaches the ruler's while its read is dead** (|W| 19.59 vs
  the ruler's 18.36 at s300, in_room 0.391 vs 0.408): the full-dose
  stream-training write mass without emission supervision maintains nothing —
  the write's magnitude is role-blind while the read's survival is not (the
  standing name-blindness rider, sharpened by the role cut: cross-arm dW grows
  0.01 -> 5.42 through the run).
- **The name's presence as input maintains nothing**: conly carries the same 112
  TAVIREN tokens/step in stream (the full k=16 dose) as background input and
  executes as dead as k=0's all-Z arm (0.0035/0.0029/0.0008 vs
  0.0036/0.0005/0.0031) — maintenance is supervision-side (the attended
  emission), not stimulus-side.
