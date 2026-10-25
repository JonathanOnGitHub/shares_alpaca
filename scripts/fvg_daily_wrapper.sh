#!/bin/bash
# fvg_daily_wrapper.sh — run inside a cron no_agent job
#
# The intraday FVG strategy is a 5-minute-bar strategy, so the signal set
# changes throughout the session. Two daemons are needed:
#
#   1. the SCANNER  — polls for new signals every 5 minutes until 14:00 ET
#                     and keeps signals.json current
#   2. the EXECUTOR — polls signals.json every 60 s and places the orders
#
# Both are detached with setsid so they survive the cron runner's exit
# and do not hold the runner's captured stdout pipe open.
#
# TIMING: this job must fire BEFORE the US open (14:00 BST = 09:00 ET).
# The previous 21:30 BST schedule was 16:30 ET — after the 14:00 ET flat
# time and after the close — so the scanner's first tick was already past
# its `--until-et` cutoff and it exited immediately, and the executor had
# no session left to trade in.
#
# STALE-DAEMON HANDLING: a daemon that is alive but wedged is WORSE than
# no daemon at all. On 2026-10-23 the executor stayed alive across the
# close because a pending position (whose order had been cancelled) never
# reached the "session over and flat" state. The next day's wrapper then
# saw a live PID, printed "executor already running" and skipped the
# start — so the new session inherited yesterday's in-memory state and
# the strategy stopped trading. This wrapper now checks that a running
# daemon actually belongs to TODAY's session before reusing it, and
# kills it if it does not.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PY="$REPO_DIR/.venv/bin/python"

LOGFILE="$REPO_DIR/executor.log"
PIDFILE="$REPO_DIR/executor.pid"
SCAN_PIDFILE="$REPO_DIR/fvg_scanner.pid"

cd "$REPO_DIR"

# ── 0. Retire any daemon that belongs to a previous session ────────────────
# A daemon is considered STALE if it was started before today's 00:00
# host-local time. Anything older belongs to a session that is over.
reap_stale_daemon() {
    local pidfile="$1" label="$2"
    [ -f "$pidfile" ] || return 0
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    [ -n "$pid" ] || { rm -f "$pidfile"; return 0; }
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "$(date -Iseconds) — $label PID $pid is not running — clearing pidfile" >> "$LOGFILE"
        rm -f "$pidfile"
        return 0
    fi
    # Guard against PID reuse: if the PID now belongs to some unrelated
    # process, do not kill it — just clear the stale pidfile.
    local cmdline
    cmdline="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    case "$cmdline" in
        *run_fvg_paper.py*) ;;
        *)
            echo "$(date -Iseconds) — $label PID $pid is not the FVG daemon ('$cmdline') — clearing pidfile, not killing" >> "$LOGFILE"
            rm -f "$pidfile"
            return 0
            ;;
    esac
    local started
    started="$(ps -o lstart= -p "$pid" 2>/dev/null | sed 's/^ *//')"
    local start_epoch
    start_epoch="$(date -d "$started" +%s 2>/dev/null || echo 0)"
    local today_epoch
    today_epoch="$(date -d "$(date +%Y-%m-%d) 00:00:00" +%s)"
    if [ "$start_epoch" -lt "$today_epoch" ]; then
        echo "$(date -Iseconds) — $label PID $pid started '$started' (before today) — killing stale daemon" >> "$LOGFILE"
        kill "$pid" 2>/dev/null || true
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi
        rm -f "$pidfile"
        echo "$(date -Iseconds) — stale $label PID $pid reaped" >> "$LOGFILE"
    else
        echo "$(date -Iseconds) — $label PID $pid is from today's session (started '$started') — reusing" >> "$LOGFILE"
    fi
}

reap_stale_daemon "$SCAN_PIDFILE" "FVG scanner"
reap_stale_daemon "$PIDFILE" "FVG executor"

# ── 1. Start the signal scanner daemon ───────────────────────────────────────
# Rescans for signals every 5 minutes until 14:00 ET. A single daily scan
# would trade yesterday's setup and miss most of the day's opportunities.
if [ -f "$SCAN_PIDFILE" ] && kill -0 "$(cat "$SCAN_PIDFILE")" 2>/dev/null; then
    echo "$(date -Iseconds) — FVG scanner already running (PID $(cat "$SCAN_PIDFILE"))" >> "$LOGFILE"
else
    setsid "$VENV_PY" "$SCRIPT_DIR/run_fvg_paper.py" --poll --poll-seconds 300 --until-et 840 \
        >> "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$SCAN_PIDFILE"
    echo "$(date -Iseconds) — FVG scanner started PID $!" >> "$LOGFILE"
fi

# ── 2. Do one immediate scan so signals.json is populated now ───────────────
# Otherwise the executor starts against a stale (or missing) file and sits
# idle until the scanner's first tick.
"$VENV_PY" "$SCRIPT_DIR/run_fvg_paper.py" --signals-only 2>&1 | tee -a "$LOGFILE"

# ── 3. Start the executor daemon ─────────────────────────────────────────────
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "$(date -Iseconds) — executor already running (PID $(cat "$PIDFILE"))" >> "$LOGFILE"
else
    setsid "$VENV_PY" "$SCRIPT_DIR/run_fvg_paper.py" >> "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "$(date -Iseconds) — executor started PID $!" >> "$LOGFILE"
fi

# ── 4. P&L report ────────────────────────────────────────────────────────────
# Printed at the START of the run, so the delivered message reflects the
# account as it stood when the session opened. The end-of-day figures are
# produced by the separate end-of-day report job.
"$VENV_PY" "$SCRIPT_DIR/fvg_pnl_report.py" 2>&1 | tee -a "$LOGFILE"
