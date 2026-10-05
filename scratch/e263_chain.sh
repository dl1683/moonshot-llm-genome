#!/usr/bin/env bash
# E263 ORCHESTRATION CHAIN (detached; dispatched by the e263 executor
# 2026-10-05): e264_middle_rungs owns the GPU lane -> WAIT for the lane
# (no lab training process + GPU cool/idle, double-polled) -> SMOKE
# FIRST (the e246 discipline) -> the FULL cell -> commit + push each
# phase. A failed smoke STOPS the chain (the autopsy record commits);
# the full run's own G_COORD re-checks the lane in-process and stands
# down honestly if a new elder appeared. 3 h cap. v2: logs as .txt (the .log gitignore silently ate the v1 autopsy commit — diagnosed and fixed).
set -u
REPO="C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome"
cd "$REPO/lab" || exit 9
LOG="$REPO/scratch/e263_chain_log.txt"
PY=python

log() { echo "[chain $(date -u +%H:%M:%S)] $*" >> "$LOG"; }

lab_procs() {
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object {\$_.CommandLine -match '(e|g)[0-9][^ ]*\\.py' -and \$_.CommandLine -notmatch 'e263'} | Select-Object -ExpandProperty ProcessId" 2>/dev/null | tr -d '\r' | grep -v '^$'
}

gpu_idle() {
  local u t
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' %\r')
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr -d ' C\r')
  [ -n "$u" ] && [ -n "$t" ] && [ "$u" -le 10 ] && [ "$t" -le 70 ]
}

log "chain start: waiting for the GPU lane (e264 owns it)"
deadline=$(( $(date +%s) + 10800 ))
lane_free=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  p1=$(lab_procs); sleep 20; p2=$(lab_procs)
  if [ -z "$p1" ] && [ -z "$p2" ] && gpu_idle; then
    sleep 10
    p3=$(lab_procs)
    if [ -z "$p3" ] && gpu_idle; then lane_free=1; break; fi
  fi
  sleep 25
done
if [ "$lane_free" -ne 1 ]; then
  log "LANE NEVER FREED within 3h — chain exits (the registration stands; compute deferred to the next dispatch)"
  exit 3
fi
log "LANE FREE — running the SMOKE first (e246's discipline)"

E263_SMOKE=1 $PY e263_counterfeit_self.py > "$REPO/scratch/e263_chain_smoke_log.txt" 2>&1
src=$?
log "smoke exit $src"
if [ "$src" -ne 0 ]; then
  log "SMOKE FAILED — chain STOPS (the autopsy record); tail:"
  tail -25 "$REPO/scratch/e263_chain_smoke_log.txt" >> "$LOG"
  cd "$REPO" && git add scratch/e263_chain_log.txt scratch/e263_chain_smoke_log.txt \
    && git commit -q -m "e263 smoke FAILED (caught by the chain, pre-full-run): the autopsy record — the compute stands down until fixed" \
    && git push -q origin main
  exit 4
fi
log "smoke PASSED — tail:"
tail -8 "$REPO/scratch/e263_chain_smoke_log.txt" >> "$LOG"
cd "$REPO" && git add scratch/e263_chain_log.txt scratch/e263_chain_smoke_log.txt \
  && git commit -q -m "e263 smoke PASSED (end-to-end, all phases, SMOKE-stamped): the shakedown cleared — the full cell launches when the lane holds" \
  && git push -q origin main
cd "$REPO/lab" || exit 9

log "running the FULL cell"
$PY e263_counterfeit_self.py > "$REPO/scratch/e263_chain_full_log.txt" 2>&1
frc=$?
log "full run exit $frc"
cd "$REPO" || exit 9
git add runs/e263/metrics.json runs/e263/*.png scratch/e263_chain_log.txt \
  scratch/e263_chain_full_log.txt 2>/dev/null
if [ "$frc" -eq 2 ]; then
  git commit -q -m "e263 STOOD DOWN at G_COORD (a new elder cell owned the lane at launch): the certified P0-P4 instrument record + the registered stand-down; compute deferred" && git push -q origin main
  exit 5
fi
git commit -q -m "e263 DONE (the chain's full run): THE COUNTERFEIT SELF adjudicated per the frozen bars — see runs/e263/metrics.json + the draft NOTES entry inside" && git push -q origin main
log "chain complete"
exit 0
