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
# WHY systemd instead of setsid + a PID file:
# On 28 and 29 Oct 2026 the kernel OOM killer hit hermes-gateway.service
# (7.4G memory peak) and systemd tore the whole gateway cgroup down with
# SIGKILL. Both FVG daemons had been started by this very cron job via
# `setsid`, and setsid detaches a process from the terminal and the
# session — but NOT from the systemd cgroup it was started in. So both
# daemons were killed with the gateway, mid-session, losing their open
# positions and their trade log.
#
# The daemons now run in their own units (shares-alpaca-fvg-scanner.service
# and shares-alpaca-fvg-executor.service), which are NOT in the gateway's
# cgroup, so an OOM kill or a gateway restart leaves the strategy alone.
# This wrapper only:
#   1. makes sure both units are installed, enabled and running,
#   2. does one immediate scan so signals.json is populated now,
#   3. prints the P&L report.
#
# TIMING: the daily job fires at 14:00 UTC = 15:00 BST = 10:00 ET, i.e.
# shortly after the 09:30 ET open. The scanner then polls every 5 minutes
# until its --until-et cutoff of 14:00 ET (the strategy's hard flat
# time), and the executor trades from the open until 16:05 ET.
#
# STALE-DAEMON HANDLING: the old wrapper skipped the start when the PID
# file named a live process. A daemon that is alive but WEDGED (stuck in
# a stale session, holding a phantom pending position whose order was
# cancelled) satisfies that check, so the next day's wrapper declared it
# "already running" and reused the stale process — with stale in-memory
# state and stale code. The session then traded nothing, the wrapper
# exited 0, and cron reported success. Observed 23 Oct 2026.
#
# The systemd units are now restarted unconditionally, which avoids
# inheriting a stale process. Restarting is safe because the daemon
# re-reads .positions_fvg.json and .acted_fvg.json on startup, so it
# re-adopts any position it already owns and never re-enters a signal it
# has already acted on.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PY="$REPO_DIR/.venv/bin/python"

LOGFILE="$REPO_DIR/executor.log"
SCAN_UNIT="shares-alpaca-fvg-scanner.service"
EXEC_UNIT="shares-alpaca-fvg-executor.service"

cd "$REPO_DIR"

# ── 0. Make sure the user systemd manager is reachable ──────────────────────
# The Hermes cron runner inherits the gateway's environment, which does
# carry XDG_RUNTIME_DIR/DBUS_SESSION_BUS_ADDRESS. But this wrapper is also
# run by hand from plain shells, where those may be missing. `systemctl
# --user` needs them to find the user manager, so set them explicitly.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"

systemctl_user() {
    systemctl --user "$@"
}

# Install or refresh a unit if the source file has changed.
install_unit() {
    local unit="$1"
    local src="$REPO_DIR/deploy/$unit"
    local dst="$HOME/.config/systemd/user/$unit"
    mkdir -p "$HOME/.config/systemd/user"
    if [ ! -f "$dst" ] || ! cmp -s "$src" "$dst"; then
        cp "$src" "$dst"
        systemctl_user daemon-reload
        echo "$(date -Iseconds) — installed/refreshed $unit" >> "$LOGFILE"
    fi
}

install_unit "$SCAN_UNIT"
install_unit "$EXEC_UNIT"

# ── 1. Reap any LEGACY setsid daemon ───────────────────────────────────────
# Before the systemd units existed, both daemons were started by this
# same wrapper with setsid, which left them inside the gateway's cgroup.
# Leaving them running would give two scanners and two executors watching
# the same signals file, and both would open the same position.
for PID in $(pgrep -f "run_fvg_paper\.py" 2>/dev/null || true); do
    if [ "$PID" = "$$" ]; then
        continue
    fi
    # Only kill it if it is NOT one of the systemd units' own main processes.
    KEEP=""
    for unit in "$SCAN_UNIT" "$EXEC_UNIT"; do
        unit_pid="$(systemctl_user show -p MainPID --value "$unit" 2>/dev/null || echo 0)"
        if [ "$unit_pid" != "0" ] && [ "$PID" = "$unit_pid" ]; then
            KEEP="yes"
        fi
    done
    if [ -z "$KEEP" ]; then
        echo "$(date -Iseconds) — reaping legacy setsid FVG daemon PID $PID" >> "$LOGFILE"
        kill "$PID" 2>/dev/null || true
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null || true
        fi
    fi
done
rm -f "$REPO_DIR/executor.pid" "$REPO_DIR/fvg_scanner.pid"

# ── 2. Do one immediate scan so signals.json is populated now ───────────────
# Otherwise the executor starts against a stale (or missing) file and sits
# idle until the scanner's first tick.
echo "=== $(date -Iseconds) — FVG daily run ===" >> "$LOGFILE"
"$VENV_PY" "$SCRIPT_DIR/run_fvg_paper.py" --signals-only 2>&1 | tee -a "$LOGFILE"

# ── 3. Start the scanner and executor under systemd ─────────────────────────
# `enable` makes them survive a reboot; `restart` guarantees a fresh
# daemon that has just read the signal file written above.
#
# The two are restarted independently: the scanner exits NORMALLY at
# 14:00 ET by design, so its unit uses Restart=on-failure and will simply
# sit in "exited (success)" until tomorrow's wrapper restarts it. The
# executor exits NORMALLY at 16:05 ET (close_mode="intraday" — it must
# be flat by 14:00 ET and must not linger overnight), and is likewise
# restarted by the next day's wrapper.
for unit in "$SCAN_UNIT" "$EXEC_UNIT"; do
    systemctl_user enable "$unit" >> "$LOGFILE" 2>&1
    systemctl_user restart "$unit" >> "$LOGFILE" 2>&1
done
sleep 2

for unit in "$SCAN_UNIT" "$EXEC_UNIT"; do
    if systemctl_user is-active --quiet "$unit"; then
        MAIN_PID="$(systemctl_user show -p MainPID --value "$unit" 2>/dev/null || echo '?')"
        echo "$(date -Iseconds) — $unit active (PID $MAIN_PID)" >> "$LOGFILE"
    else
        echo "$(date -Iseconds) — WARNING: $unit is NOT active after restart" >> "$LOGFILE"
        systemctl_user status "$unit" --no-pager >> "$LOGFILE" 2>&1 || true
    fi
done

# ── 4. P&L report ────────────────────────────────────────────────────────────
# Printed at the START of the run, so the delivered message reflects the
# account as it stood when the session opened. The end-of-day figures are
# produced by the separate end-of-day report job.
"$VENV_PY" "$SCRIPT_DIR/fvg_pnl_report.py" 2>&1 | tee -a "$LOGFILE"
