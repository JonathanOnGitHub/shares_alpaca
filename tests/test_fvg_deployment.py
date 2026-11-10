"""
Tests for the FVG deployment supervision fix (2026-10-30 audit).

The bug these cover:

  The two FVG daemons (scanner and executor) were started by the Hermes
  cron job via `setsid`. setsid detaches a process from its terminal and
  session but NOT from the systemd cgroup it was launched in, so both
  daemons stayed inside hermes-gateway.service's cgroup. On 28 and 29 Oct
  the kernel OOM killer hit the gateway (7.4G peak); systemd tore the
  whole cgroup down with SIGKILL and both FVG daemons died mid-session,
  losing their open positions and their trade log.

The fix moves both daemons into their own user systemd units, so:
  - they are not in the gateway's cgroup,
  - they are restarted automatically if they CRASH,
  - SIGTERM produces a graceful shutdown (positions flattened, trade
    log written) instead of a SIGKILL.

These tests lock in the properties that make the deployment work. They
are static checks — they do not start or stop any service.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

SCAN_UNIT_SRC = ROOT / "deploy" / "shares-alpaca-fvg-scanner.service"
EXEC_UNIT_SRC = ROOT / "deploy" / "shares-alpaca-fvg-executor.service"
WRAPPER = ROOT / "scripts" / "fvg_daily_wrapper.sh"
EOD_WRAPPER = ROOT / "scripts" / "fvg_eod_report.sh"
RUNNER = ROOT / "scripts" / "run_fvg_paper.py"

UNITS = [SCAN_UNIT_SRC, EXEC_UNIT_SRC]


def _read(path):
    return path.read_text()


def _strip_comments(path):
    """Text with shell comments removed.

    The deployment notes in these files explain WHY setsid and the
    gateway cgroup are a problem, so the words appear in the comments.
    A test that greps the raw file would match the explanation rather
    than the behaviour, so the checks below run against the code only.
    """
    out = []
    for line in _read(path).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0]
        out.append(line)
    return "\n".join(out)


def test_both_units_exist():
    for unit in UNITS:
        assert unit.exists(), f"missing systemd unit: {unit}"


def test_units_run_outside_the_gateway_cgroup():
    """The units must be independent units, not drop-ins of the gateway."""
    for unit in UNITS:
        txt = _strip_comments(unit)
        assert "hermes-gateway" not in txt, (
            f"{unit.name} references hermes-gateway; it must be an "
            "independent unit so an OOM kill of the gateway cannot take "
            "the trader with it"
        )
        assert "WantedBy=default.target" in txt


def test_units_restart_on_crash():
    """Restart policy must be present so a crash is recovered from."""
    for unit in UNITS:
        txt = _read(unit)
        assert re.search(r"^Restart=(always|on-failure)$", txt, re.M), \
            f"{unit.name} must restart the daemon if it dies"
        assert re.search(r"^RestartSec=\d+$", txt, re.M), \
            "RestartSec prevents a tight crash loop"


def test_fvg_units_do_not_hot_loop():
    """
    The FVG daemons must use Restart=on-failure, NOT Restart=always.

    Both FVG processes EXIT NORMALLY by design:
      - the scanner's poll loop returns at --until-et (14:00 ET),
      - the executor uses close_mode="intraday", so PaperExecutor.run()
        returns at 16:05 ET once the session is over.

    With Restart=always systemd would restart each daemon, it would
    exit again immediately, and the unit would hot-loop all afternoon.
    Only a CRASH should trigger a restart.

    The engulfing unit is deliberately different — its executor uses
    close_mode="overnight" and never self-exits — so it correctly uses
    Restart=always. This test guards the FVG units only.
    """
    for unit in UNITS:
        txt = _read(unit)
        assert re.search(r"^Restart=on-failure$", txt, re.M), (
            f"{unit.name} must use Restart=on-failure: this daemon exits "
            "normally at the end of the session, and Restart=always would "
            "hot-loop it all afternoon"
        )


def test_units_have_memory_ceiling():
    """A memory cap means this unit can never be the OOM killer's victim."""
    for unit in UNITS:
        txt = _read(unit)
        assert re.search(r"^MemoryMax=\d+[MG]$", txt, re.M), \
            f"{unit.name} needs a MemoryMax so it cannot trigger a global OOM"


def test_units_kill_the_whole_process_group():
    """KillMode=control-group: no orphaned child left holding broker state."""
    for unit in UNITS:
        txt = _read(unit)
        assert re.search(r"^KillMode=control-group$", txt, re.M)
        assert re.search(r"^KillSignal=SIGTERM$", txt, re.M), \
            "SIGTERM lets the executor close its positions and write the log"


def test_units_use_the_project_venv():
    for unit in UNITS:
        txt = _read(unit)
        assert ".venv/bin/python" in txt, f"{unit.name} must use the project virtualenv"
        assert f"WorkingDirectory={ROOT}" in txt


def test_scanner_unit_polls():
    """The scanner must poll, not scan once — the strategy is intraday."""
    txt = _read(SCAN_UNIT_SRC)
    assert "--poll" in txt
    assert "--until-et" in txt, "the scanner must stop at the 14:00 ET flat time"


def test_executor_unit_does_not_regenerate_signals():
    """
    The executor must consume signals, not regenerate them.

    The scanner owns signals.json. If the executor regenerated them it
    would overwrite the file mid-session while the executor is reading
    it, and would load the whole yfinance universe into a second
    long-lived process.
    """
    txt = _read(EXEC_UNIT_SRC)
    assert "--executor-only" in txt
    assert "--signals-only" not in txt
    assert "--poll" not in txt, "the executor must not run a scan loop"


def test_runner_supports_executor_only():
    txt = _read(RUNNER)
    assert '"--executor-only"' in txt
    assert re.search(r"if args\.executor_only:", txt)


def test_runner_handles_sigterm():
    """
    systemd stops the service with SIGTERM. Without a handler the process
    dies outright and the trade log + open positions are lost — exactly
    the failure mode the OOM kill caused.
    """
    txt = _read(RUNNER)
    assert "signal.SIGTERM" in txt, "run_fvg_paper.py must handle SIGTERM"
    assert "signal.SIGINT" in txt
    assert re.search(r"def _sig_handler.*?executor\.stop\(\)", txt, re.S), \
        "the SIGTERM handler must call executor.stop()"


def test_wrapper_uses_systemd_not_setsid():
    """
    The wrapper must hand the daemons to systemd. `setsid` is what
    caused the failure: it does not leave the gateway's cgroup.
    """
    txt = _strip_comments(WRAPPER)
    assert not re.search(r"(^|[;&|]\s*|&&|\|\|)\s*setsid\s", txt, re.M), (
        "the wrapper must not start the daemons with setsid — that is the bug"
    )
    assert "systemctl --user" in txt or "systemctl_user" in txt


def test_wrapper_reaps_legacy_setsid_daemons():
    """
    Daemons left over from the old setsid deployment would run alongside
    the systemd ones. Two executors watching the same signals file would
    each open the position — double the intended exposure.
    """
    txt = _read(WRAPPER)
    assert "pgrep" in txt, "the wrapper must reap legacy setsid daemons"
    assert "run_fvg_paper" in txt


def test_wrapper_preserves_the_units_own_main_pid():
    """
    The legacy reaper must not kill the systemd units' own processes.

    If it did, the wrapper would kill the very daemon it just started.
    """
    txt = _read(WRAPPER)
    assert "MainPID" in txt, \
        "the legacy reaper must skip the systemd units' own MainPID"


def test_wrapper_sets_the_systemd_runtime_env():
    """
    `systemctl --user` needs XDG_RUNTIME_DIR/DBUS_SESSION_BUS_ADDRESS to
    find the user manager. The cron runner inherits them from the gateway
    today, but the wrapper is also run by hand, so it must not rely on it.
    """
    for path in (WRAPPER, EOD_WRAPPER):
        txt = _read(path)
        assert "XDG_RUNTIME_DIR" in txt, f"{path.name} must set XDG_RUNTIME_DIR"
        assert "DBUS_SESSION_BUS_ADDRESS" in txt, \
            f"{path.name} must set DBUS_SESSION_BUS_ADDRESS"


def test_wrapper_still_generates_signals():
    """
    One immediate scan must stay in the wrapper so signals.json is
    populated before the executor starts, rather than the executor
    sitting idle until the scanner's first tick.
    """
    txt = _read(WRAPPER)
    assert "--signals-only" in txt


def test_wrapper_enables_and_restarts_both_units():
    txt = _read(WRAPPER)
    assert re.search(r"systemctl_user enable", txt), \
        "the units must be enabled so they survive a reboot"
    assert re.search(r"systemctl_user restart", txt), (
        "the daemons must be restarted unconditionally — a wedged daemon "
        "that still holds a PID is worse than no daemon at all"
    )
    assert "shares-alpaca-fvg-scanner.service" in txt
    assert "shares-alpaca-fvg-executor.service" in txt


def test_eod_report_stops_the_units():
    """
    The end-of-day job must stop the units, so no daemon is left writing
    signals.json or holding a position overnight.
    """
    txt = _read(EOD_WRAPPER)
    assert "shares-alpaca-fvg-executor.service" in txt
    assert "shares-alpaca-fvg-scanner.service" in txt
    assert re.search(r"systemctl --user stop", txt)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll FVG deployment tests passed.")
