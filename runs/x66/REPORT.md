# x66 — THE ORACLE, CALIBRATED — REPORT
(written 2026-10-12T15:28:20.679927+00:00; smoke=False)

## The four verdicts (against the frozen bars)
- REGRESSION: **REGRESSION-SILENT** (Q2_full None; bar 0.5; n 19; IQR 0.017841)
- ABLATION: **UNSCORED-SILENT-FULL** (Q2_no-bio None; margin 0.1)
- PRELUDE: **PRELUDE-SILENT** 
- RIDER: **BARRIERS-VANISH-AT-k16** (0 dead interior cells; both-alive window 1/15)
- RIDERC-EXT (R88's registered P-NARROWS-TO-S200): **NARROWS-TO-S200** (fires: ['read@s200'])

## P_x66a (the executor's registered reads)
- read1: guess REGRESSION-SILENT | verdict REGRESSION-SILENT | HIT
- read2: guess ABLATION-HOLDS | verdict UNSCORED-SILENT-FULL | MISS
- read3: guess PRELUDE-SILENT | verdict PRELUDE-SILENT | HIT
- read4: guess BARRIERS-PERSIST | verdict BARRIERS-VANISH-AT-k16 | MISS
- read5: guess NARROWS-TO-S200 | verdict NARROWS-TO-S200 | HIT
- primary compound: 2/4
- lab-lean hits: 1/4 | counter hits: 2/4

## The prelude table (K + the winners)
- s_s4: K 0.320080
- s_s1: K 0.269025
- 24316: K 0.259131
- s_s2: K 0.245983
- s_s3: K 0.214787
- 24314b: K 0.204653
- 24314a: K 0.196489
- s_s5: K 0.146190
- 24315: K 0.144955

## The sweep table (retention + K per variant)
- V01 disk_a: host Q seed 24314 menu authored dose 400 | t0 0.4024 -> end 0.004968 (retention 0.012347) | K —  | powered
- V02 disk_b: host N seed 24314 menu authored dose 400 | t0 0.3681 -> end 0.001713 (retention 0.004654) | K —  | powered
- V03 disk_a2: host Q seed 24315 menu authored dose 400 | t0 0.6879 -> end 0.003360 (retention 0.004884) | K —  | powered
- V04 disk_a3: host Q seed 24316 menu authored dose 400 | t0 0.4393 -> end 0.011899 (retention 0.027087) | K —  | powered
- V05 disk_s1: host Q seed 429496730 menu authored dose 400 | t0 0.6153 -> end 0.004057 (retention 0.006594) | K —  | powered
- V06 disk_s2: host Q seed 1288490189 menu authored dose 400 | t0 0.4570 -> end 0.012950 (retention 0.028336) | K —  | powered
- V07 disk_s3: host Q seed 2147483648 menu authored dose 400 | t0 0.3950 -> end 0.014068 (retention 0.035612) | K —  | powered
- V08 disk_s4: host Q seed 3006477407 menu authored dose 400 | t0 0.5521 -> end 0.001226 (retention 0.002220) | K —  | powered
- V09 disk_s5: host Q seed 3865470566 menu authored dose 400 | t0 0.4354 -> end 0.005914 (retention 0.013584) | K —  | powered
- V10 disk_b2: host N seed 24315 menu authored dose 400 | t0 0.5739 -> end 0.000651 (retention 0.001135) | K —  | powered
- V11 m_jit: host Q seed 24314 menu jittered dose 400 | t0 0.1443 -> end 0.002807 (retention 0.019450) | K 0.460698 KNIFE | powered
- V12 m_rnd: host Q seed 24314 menu random dose 400 | t0 0.0002 -> end 0.000038 (retention 0.216772) | K —  | DIE-MODE
- V13 m_half: host Q seed 24314 menu other dose 400 | t0 0.1806 -> end 0.003260 (retention 0.018053) | K 0.391346 KNIFE | powered
- V14 m_coo: host Q seed 24314 menu other dose 400 | t0 0.1743 -> end 0.004491 (retention 0.025771) | K 0.358807 CARRY | powered
- V15 d_100: host Q seed 24314 menu authored dose 100 | t0 0.5799 -> end 0.002098 (retention 0.003618) | K 0.157217 KNIFE | powered
- V16 d_200: host Q seed 24314 menu authored dose 200 | t0 0.6845 -> end 0.011646 (retention 0.017015) | K 0.310099 KNIFE | powered
- V17 d_800: host Q seed 24314 menu authored dose 800 | t0 0.5089 -> end 0.007540 (retention 0.014818) | K 0.347526 CARRY | powered
- V18 js_15: host Q seed 24315 menu jittered dose 400 | t0 0.3021 -> end 0.001248 (retention 0.004132) | K 0.393944 CARRY | powered
- V19 rs_15: host Q seed 24315 menu random dose 400 | t0 0.0001 -> end 0.000119 (retention 1.965763) | K —  | DIE-MODE
- V20 js_16: host Q seed 24316 menu jittered dose 400 | t0 0.1064 -> end 0.003049 (retention 0.028642) | K 0.552715 CARRY | powered
- V21 jd_800: host Q seed 24314 menu jittered dose 800 | t0 0.1833 -> end 0.001185 (retention 0.006463) | K 0.44114 CARRY | powered

## The rider grid
- alpha 0.000: host 0.715929 | guest 0.000097
- alpha 0.062: host 0.685436 | guest 0.000171
- alpha 0.125: host 0.645623 | guest 0.000357
- alpha 0.188: host 0.592977 | guest 0.000892
- alpha 0.250: host 0.522159 | guest 0.002640
- alpha 0.312: host 0.429617 | guest 0.008583
- alpha 0.375: host 0.319738 | guest 0.027471
- alpha 0.438: host 0.206601 | guest 0.076747
- alpha 0.500: host 0.113225 | guest 0.168074
- alpha 0.562: host 0.057862 | guest 0.280420
- alpha 0.625: host 0.033112 | guest 0.378456
- alpha 0.688: host 0.022138 | guest 0.446159
- alpha 0.750: host 0.015859 | guest 0.490682
- alpha 0.812: host 0.011299 | guest 0.526322
- alpha 0.875: host 0.007705 | guest 0.562377
- alpha 0.938: host 0.004975 | guest 0.599989
- alpha 1.000: host 0.003064 | guest 0.635227

## The mid-band co-occupants (co-report)
- k7r (k=7): host 0.2946 guest 0.3997 | coords (a_T16 0.3573, b_Z16 0.4925) resid 0.6458 -> OWN-BASIN
- k9 (k=9): host 0.3889 guest 0.3244 | coords (a_T16 0.3847, b_Z16 0.468) resid 0.6403 -> OWN-BASIN
- k12 (k=12): host 0.5300 guest 0.1450 | coords (a_T16 0.4779, b_Z16 0.361) resid 0.6587 -> OWN-BASIN
- k14 (k=14): host 0.6538 guest 0.1012 | coords (a_T16 0.5782, b_Z16 0.2789) resid 0.6253 -> OWN-BASIN

## Gates
- 61/61 gate families PASS (the bind table's 45 md5 cells inside G_MD5).

## Deviations
- {'t': '2026-10-12T01:22:14.027592+00:00', 'note': 'restart: progressive metrics merged; legs resume from their own checkpoints'}
- {'t': '2026-10-12T01:46:34.772683+00:00', 'note': 'restart: progressive metrics merged; legs resume from their own checkpoints'}
- {'t': '2026-10-12T01:56:19.591946+00:00', 'note': 'restart: progressive metrics merged; legs resume from their own checkpoints'}
- {'t': '2026-10-12T02:13:16.538643+00:00', 'note': 'restart: progressive metrics merged; legs resume from their own checkpoints'}
- {'t': '2026-10-12T02:18:03.350438+00:00', 'note': 'restart: progressive metrics merged; legs resume from their own checkpoints'}
- {'t': '2026-10-12T03:18:09.898382+00:00', 'note': 'restart: progressive metrics merged; legs resume from their own checkpoints'}
- {'t': '2026-10-12T04:18:54.278980+00:00', 'note': 'restart: progressive metrics merged; legs resume from their own checkpoints'}
- {'t': '2026-10-12T05:19:47.648938+00:00', 'note': 'restart: progressive metrics merged; legs resume from their own checkpoints'}
- {'t': '2026-10-12T06:19:47.083674+00:00', 'note': 'restart: progressive metrics merged; legs resume from their own checkpoints'}
- {'t': '2026-10-12T13:21:43.974218+00:00', 'note': 'restart: progressive metrics merged; legs resume from their own checkpoints'}
- {'t': '2026-10-12T14:22:04.185156+00:00', 'note': 'restart: progressive metrics merged; legs resume from their own checkpoints'}
- {'t': '2026-10-12T15:22:11.736581+00:00', 'note': 'restart: progressive metrics merged; legs resume from their own checkpoints'}