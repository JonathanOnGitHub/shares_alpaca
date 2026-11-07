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
# ── WHY THIS WRAPPER NOW DELEGATES TO SYSTEMD ─────────────────────────────
# This wrapper used to start both daemons itself with `setsid`. Two
# problems came out of that:
#
#   1. DUPLICATE DAEMONS. A PID file cannot prevent a second daemon,
#      because this script OVERWRITES the PID file on every run. The
#      daemon started by yesterday's run became invisible to today's
#      "is it already running?" check, so a second daemon started
#      alongside it. On 2026-11-07 TWO run_engulfing_paper.py daemons
#      were running at once, and TWO FVG executors were spawned by
#      systemd units whose ExecStart used an unsupported --executor-only
#      flag. Each daemon keeps its own _acted_signals set and its own
#      position file, so they can double-enter a setup and overwrite
#      each other's view of an open position, leaving that position
#      unmanaged at the broker with no stop and no target.
#
#   2. THE CGROUP PROBLEM. setsid detaches a process from its
#      terminal and session, but NOT from the systemd cgroup it was
#      launched in, so the daemon stayed inside hermes-gateway.service's
#      cgroup. When the kernel OOM killer hit the gateway (7.4G peak on
#      28 and 29 Oct), systemd tore the whole cgroup down with SIGKILL
#      and the daemon died with it.
#
# The systemd units now own the daemons: they run in their own cgroups
# and the runner's flock (src/trading/process_lock.py) guarantees only
# one daemon per strategy can ever hold the strategy's signals file.
# This wrapper therefore only:
#   * does one immediate scan so signals.json is populated before the
#     executor starts polling it,
#   * asks systemd to (re)start the scanner and executor units,
#   * prints the P&L report.
#
# If systemd is unavailable (no user session bus), it falls back to the
# old setsid path — the flock still prevents a duplicate daemon, so
# the fallback is safe.
#
# TIMING: this job must fire BEFORE the US open (14:00 BST = 09:00 ET).
# The previous 21:30 BST schedule was 16:30 ET — after the 14:00 ET flat
# time and after the close — so the scanner's first tick was already past
# its `--until-et` cutoff and it exited immediately, and the executor had
# no session left to trade in.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PY="$REPO_DIR/.venv/bin/python"

LOGFILE="$REPO_DIR/executor.log"
PIDFILE="$REPO_DIR/executor.pid"
SCAN_PIDFILE="$REPO_DIR/fvg_scanner.pid"

cd "$REPO_DIR"

SYSTEMD_OK=0
if systemctl --user is-system-running >/dev/null 2>&1 || \
   systemctl --user list-unit-files >/dev/null 2>&1; then
    SYSTEMD_OK=1
fi

if [ "$SYSTEMD_OK" = "1" ]; then
    # ── Retire daemons left over from a previous session ───────────────────
    # A daemon that is alive but wedged is WORSE than no daemon at all. On
    # 2026-10-23 the executor stayed alive across the close because a
    # pending position (whose order had been cancelled) never reached
    # the "session over and flat" state. The next day's wrapper then saw
    # a live PID, printed "executor already running" and skipped the
    # start — so the new session inherited yesterday's in-memory state
    # and the strategy stopped trading.
    #
    # `systemctl restart` handles this: it stops whatever is running
    # (including a wedged daemon) and starts a fresh process, so the new
    # session begins with clean in-memory state.
    echo "=== $(date -Iseconds) — FVG daily run (systemd mode) ===" >> "$LOGFILE"

    # ── 1. One immediate scan so signals.json is populated now ─────────────
    # Otherwise the executor starts against a stale (or missing) file and
    # sits idle until the scanner's first tick.
    "$VENV_PY" "$SCRIPT_DIR/run_fvg_paper.py" --signals-only 2>&1 | tee -a "$LOGFILE"

    # ── 2. Restart the scanner and executor units ──────────────────────────
    # `restart` (not `start`) so a daemon left over from a previous
    # session is replaced rather than joined.
    systemctl --user restart shares-alpaca-fvg-scanner.service  >> "$LOGFILE" 2>&1
    systemctl --user restart shares-alpaca-fvg-executor.service >> "$LOGFILE" 2>&1
    echo "$(date -Iseconds) — systemd units restarted (scanner + executor)" >> "$LOGFILE"

    # Record the unit's main PID for the end-of-day report.
    systemctl --user show -p MainPID --value shares-alpaca-fvg-executor.service > "$PIDFILE" 2>/dev/null || true
    systemctl --user show -p MainPID --value shares-alpaca-fvg-scanner.service > "$SCAN_PIDFILE" 2>/dev/null || true
else
    # ── Fallback: no user session bus available ─────────────────────────────
    echo "=== $(date -Iseconds) — FVG daily run (setsid fallback) ===" >> "$LOGFILE"
    "$VENV_PY" "$SCRIPT_DIR/run_fvg_paper.py" --signals-only 2>&1 | tee -a "$LOGFILE"
    setsid "$VENV_PY" "$SCRIPT_DIR/run_fvg_paper.py" --poll --poll-seconds 300 --until-et 840 \
        >> "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$SCAN_PIDFILE"
    setsid "$VENV_PY" "$SCRIPT_DIR/run_fvg_paper.py" --executor-only \
        >> "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "$(date -Iseconds) — fallback daemons started (scanner $(cat "$SCAN_PIDFILE"), executor $(cat "$PIDFILE"))" >> "$LOGFILE"
fi

# ── 3. P&L report ────────────────────────────────────────────────────────────
# Printed at the START of the run, so the delivered message reflects the
# account as it stood when the session opened. The end-of-day figures are
# produced by the separate end-of-day report job.
"$VENV_PY" "$SCRIPT_DIR/fvg_pnl_report.py" 2>&1 | tee -a "$LOGFILE"
