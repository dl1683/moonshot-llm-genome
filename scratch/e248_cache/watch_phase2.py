"""e248 recovery-executor phase-2 babysitter (SESSION-SCOPED, not a lab
automation; the beat/heartbeat system is untouched). The predecessor's
train invocation (PID 39968, pre-burst-cap-fix code) is running; when it
exits at its per-invocation 60-burst cap it will stamp organism.done
prematurely — this watcher clears that stamp and re-invokes
`python lab/e248_organism_replicate.py train` (checkpoint-resumable).
Exits (notifying the executor) when phase 2 is REALLY done (step cap
reached OR the registered U-turn guard fired), or on hang/crash-loop."""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs" / "e248"
JOURNAL = RUN / "journal.jsonl"
METRICS = RUN / "metrics.json"
LOG = REPO / "scratch" / "e248_cache" / "watch_phase2.log"
STEPS = 4000


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def train_pid():
    """PID of any live `e248_organism_replicate.py train` process."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object {$_.CommandLine -like '*e248_organism_replicate.py train*'} | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    pids = [int(x) for x in out.split() if x.strip().isdigit()]
    return pids[0] if pids else None


def journal_mtime():
    try:
        return JOURNAL.stat().st_mtime
    except OSError:
        return 0.0


def journal_tail(n=8):
    try:
        lines = [l for l in JOURNAL.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        return [json.loads(l).get("event", "") for l in lines[-n:]]
    except Exception:
        return []


def read_metrics():
    try:
        return json.loads(METRICS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def relaunch():
    subprocess.Popen(
        [sys.executable, "lab/e248_organism_replicate.py", "train"],
        cwd=str(REPO),
        stdout=open(REPO / "scratch" / "e248_cache" / "train_relaunch.log",
                    "ab"),
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )


def main():
    relaunches = 0
    last_progress_mtime = journal_mtime()
    while True:
        time.sleep(90)
        pid = train_pid()
        jm = journal_mtime()
        if jm > last_progress_mtime + 1:
            last_progress_mtime = jm
        if pid is not None:
            if time.time() - last_progress_mtime > 1800:
                log(f"HANG-SUSPECT: train pid {pid} alive but journal "
                    f"stale {time.time() - last_progress_mtime:.0f}s — "
                    "exiting for executor inspection")
                sys.exit(3)
            continue
        # no train process: classify the exit
        m = read_metrics()
        org = m.get("organism", {})
        tail = journal_tail()
        last = tail[-1] if tail else ""
        if org.get("done"):
            if org.get("final_step", 0) >= STEPS or \
                    "train_uturn_stop" in tail:
                log(f"PHASE2 COMPLETE: final_step={org.get('final_step')} "
                    f"best_val={org.get('best_val')} gate_pass="
                    f"{org.get('G_ORGANISM_pass')} best_step="
                    f"{org.get('best_step')} (last events: {tail[-3:]})")
                sys.exit(0)
            # premature stamp from the pre-fix invocation: clear + relaunch
            log(f"CAP-EXIT with premature done stamp (final_step="
                f"{org.get('final_step')} < {STEPS}; tail {tail[-3:]}) — "
                "clearing stamp, relaunching")
            m["organism"]["done"] = False
            m["organism"]["premature_stamp_cleared_by"] = "watch_phase2"
            METRICS.write_text(json.dumps(m, indent=2), encoding="utf-8")
        # cap-exit (new code logs train_burst_cap_reached) or a crash:
        # relaunch, guard the crash-loop
        if relaunches >= 6:
            log(f"CRASH-LOOP GUARD: {relaunches} relaunches exhausted "
                f"(tail {tail[-3:]}) — exiting for executor inspection")
            sys.exit(2)
        relaunches += 1
        log(f"train not running (tail {tail[-3:]}) — relaunch #{relaunches}")
        relaunch()
        # give it time to make progress before re-checking; a relaunch
        # that dies instantly still burns one relaunch token
        time.sleep(300)
        if journal_mtime() <= last_progress_mtime + 1:
            log("relaunch made no journal progress in 300s "
                "(may be in the gpu_wait/thermal gate — one more cycle)")
            time.sleep(300)
            if journal_mtime() <= last_progress_mtime + 1:
                log("relaunch silent for 600s — exiting for inspection")
                sys.exit(5)
        # progress resumed: reset the counter
        relaunches = 0
        last_progress_mtime = journal_mtime()


if __name__ == "__main__":
    log("watcher started")
    main()
