"""
Tests for the engulfing deployment supervision fix (2026-10-30 audit).

The bug these cover:

  The engulfing paper executor was started by the Hermes cron job via
  `setsid`. setsid detaches a process from its terminal and session but
  NOT from the systemd cgroup it was launched in, so the daemon stayed
  inside hermes-gateway.service's cgroup. On 28 and 29 Oct the kernel
  OOM killer hit the gateway (7.4G peak); systemd tore the whole cgroup
  down with SIGKILL and the trading daemon died with it, before it could
  act on its signal. The strategy then recorded zero trades for days
  while every log looked "healthy".

The fix moves the daemon into its own user systemd unit, so:
  - it is not in the gateway's cgroup,
  - it is restarted automatically if it dies,
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

UNIT_SRC = ROOT / "deploy" / "shares-alpaca-engulfing-paper.service"
WRAPPER = ROOT / "scripts" / "engulfing_daily_wrapper.sh"
RUNNER = ROOT / "scripts" / "run_engulfing_paper.py"


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
        # Remove trailing shell comments (respecting nothing exotic --
        # these files have no '#' inside quoted strings).
        if "#" in line:
            line = line.split("#", 1)[0]
        out.append(line)
    return "\n".join(out)


def test_unit_exists():
    assert UNIT_SRC.exists(), f"missing systemd unit: {UNIT_SRC}"


def test_unit_runs_outside_the_gateway_cgroup():
    """The unit must be its own unit — not a drop-in of the gateway."""
    txt = _strip_comments(UNIT_SRC)
    assert "hermes-gateway" not in txt, (
        "the unit references hermes-gateway; it must be an independent "
        "unit so an OOM kill of the gateway cannot take the trader with it"
    )
    # A dedicated unit gets its own cgroup path.
    assert "WantedBy=default.target" in txt


def test_unit_restarts_on_failure():
    txt = _read(UNIT_SRC)
    assert re.search(r"^Restart=always$", txt, re.M), \
        "the unit must restart the daemon if it dies"
    assert re.search(r"^RestartSec=\d+$", txt, re.M), \
        "RestartSec prevents a tight crash loop"


def test_unit_has_memory_ceiling():
    """A memory cap means this unit can never be the OOM killer's victim."""
    txt = _read(UNIT_SRC)
    assert re.search(r"^MemoryMax=\d+[MG]$", txt, re.M), \
        "the unit needs a MemoryMax so it cannot trigger a global OOM"


def test_unit_kills_the_whole_process_group():
    """KillMode=control-group: no orphaned child left holding broker state."""
    txt = _read(UNIT_SRC)
    assert re.search(r"^KillMode=control-group$", txt, re.M)
    assert re.search(r"^KillSignal=SIGTERM$", txt, re.M), \
        "SIGTERM lets the executor close its positions and write the log"


def test_unit_uses_the_project_venv():
    txt = _read(UNIT_SRC)
    assert ".venv/bin/python" in txt, "the unit must use the project virtualenv"
    assert f"WorkingDirectory={ROOT}" in txt


def test_unit_does_not_regenerate_signals():
    """
    The daemon must consume signals, not regenerate them.

    Regenerating loads the whole yfinance universe into memory — the
    daemon was the largest process in the gateway cgroup and the OOM
    killer's target — and it would overwrite signals_engulfing.json
    mid-session while the executor is reading it.
    """
    txt = _read(UNIT_SRC)
    assert "--executor-only" in txt
    assert "--signals-only" not in txt


def test_runner_supports_executor_only():
    txt = _read(RUNNER)
    assert '"--executor-only"' in txt
    # The flag must short-circuit signal generation.
    assert re.search(r"if args\.executor_only:", txt)


def test_runner_handles_sigterm():
    """
    systemd stops with SIGTERM. Without a handler the process dies
    outright and the trade log + open positions are lost — exactly the
    failure mode the OOM kill caused.
    """
    txt = _read(RUNNER)
    assert "signal.SIGTERM" in txt, "run_engulfing_paper.py must handle SIGTERM"
    assert "signal.SIGINT" in txt
    # The handler must actually stop the executor, not just log.
    assert re.search(r"def _sig_handler.*?executor\.stop\(\)", txt, re.S), \
        "the SIGTERM handler must call executor.stop()"


def test_wrapper_uses_systemd_not_setsid():
    """
    The wrapper must hand the daemon to systemd. `setsid` is what caused
    the failure: it does not leave the gateway's cgroup.
    """
    txt = _strip_comments(WRAPPER)
    # "setsid" as a COMMAND, not as a word in a log message.
    assert not re.search(r"(^|[;&|]\s*|&&|\|\|)\s*setsid\s", txt, re.M), (
        "the wrapper must not start the daemon with setsid — that is the bug"
    )
    assert "systemctl --user" in txt or "systemctl_user" in txt


def test_wrapper_reaps_legacy_setsid_daemon():
    """
    A daemon left over from the old setsid deployment would run alongside
    the systemd one. Both would read the same signal file and both would
    open the position — double the intended exposure.
    """
    txt = _read(WRAPPER)
    assert "pgrep" in txt, "the wrapper must reap legacy setsid daemons"
    assert "run_engulfing_paper" in txt


def test_wrapper_sets_the_systemd_runtime_env():
    """
    `systemctl --user` needs XDG_RUNTIME_DIR/DBUS_SESSION_BUS_ADDRESS to
    find the user manager. The cron runner inherits them from the gateway
    today, but the wrapper is also run by hand, so it must not rely on it.
    """
    txt = _read(WRAPPER)
    assert "XDG_RUNTIME_DIR" in txt
    assert "DBUS_SESSION_BUS_ADDRESS" in txt


def test_wrapper_still_generates_signals():
    """
    Signal generation must stay in the wrapper: it is a one-shot
    post-close job, not something the long-running daemon should redo.
    """
    txt = _read(WRAPPER)
    assert "--signals-only" in txt


def test_wrapper_enables_and_restarts_the_unit():
    txt = _read(WRAPPER)
    assert re.search(r"systemctl_user enable", txt), \
        "the unit must be enabled so it survives a reboot"
    assert re.search(r"systemctl_user restart", txt), (
        "the daemon must be restarted unconditionally — a wedged daemon "
        "that still holds a PID is worse than no daemon at all"
    )


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll engulfing deployment tests passed.")
