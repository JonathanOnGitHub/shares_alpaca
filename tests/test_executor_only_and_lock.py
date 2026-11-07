"""
Tests for the single-instance lock and the --executor-only flag.

Covers the defects found in the 2026-11-07 audit:

  1. `--executor-only` was passed by the systemd units but was not
     defined by either runner's argparse, so argparse exited with
     status 2 (INVALIDARGUMENT) and the unit crash-looped ~12,400
     times. The daemons therefore never ran, and the positions they
     should have managed were orphaned at the broker.
  2. Two daemons for the same strategy could run at once, because the
     cron wrapper overwrites its PID file on every run. Two daemons
     keep separate `_acted_signals` sets and separate position files,
     so they can double-enter a setup and erase each other's view of
     an open position.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.trading.process_lock import acquire_lock, lock_holder_pid  # noqa: E402


# ── 1. the lock is exclusive ────────────────────────────────────────────────

def test_lock_is_exclusive(tmp_path):
    """A second acquire on the same path must fail while the first holds it."""
    lock_path = Path(tmp_path) / ".executor.lock"
    first = acquire_lock(lock_path)
    assert first is not None, "first acquire failed"
    second = acquire_lock(lock_path)
    assert second is None, "second acquire succeeded — the lock is not exclusive"


def test_lock_released_on_process_exit(tmp_path):
    """The lock must be released when the holder dies.

    This is the property a PID file cannot provide: a PID file left
    behind by a SIGKILLed daemon blocks the next start forever, unless
    someone manually cleans it up. flock is released by the kernel
    when the fd is closed, which happens on process exit regardless
    of the signal.
    """
    lock_path = Path(tmp_path) / ".executor.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; sys.path.insert(0, %r);"
         "from src.trading.process_lock import acquire_lock;"
         "from pathlib import Path;"
         "f = acquire_lock(Path(%r));"
         "print('held' if f else 'failed', flush=True);"
         "time.sleep(300)" % (str(ROOT), str(lock_path))],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout is not None
        out = holder.stdout.readline().strip()
        assert out == "held", f"child did not take the lock: {out!r}"
        # While the child holds it, we must not be able to take it.
        assert acquire_lock(lock_path) is None, "lock was not exclusive across processes"
        # Kill it hard — SIGKILL gives the process no chance to clean up.
        holder.kill()
        holder.wait(timeout=10)
        # Now the lock must be available again.
        again = acquire_lock(lock_path)
        assert again is not None, "lock was still held after the holder was SIGKILLed"
    finally:
        if holder.poll() is None:
            holder.kill()


def test_lock_records_holder_pid(tmp_path):
    lock_path = Path(tmp_path) / ".executor.lock"
    lock = acquire_lock(lock_path)
    assert lock is not None
    assert lock_holder_pid(lock_path) == os.getpid()


# ── 2. the --executor-only flag exists in both runners ─────────────────────

def _runner_accepts_flag(script):
    """Return True if `script --executor-only` parses without a usage error.

    The bug was an argparse usage error (exit code 2), so the check is
    that the flag is NOT rejected as unknown. We only inspect argparse's
    behaviour, not whether the daemon then runs, so the test does not
    need the market open or broker credentials.
    """
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, f"{script} --help failed: {res.stderr[-500:]}"
    return "--executor-only" in res.stdout


def test_fvg_runner_accepts_executor_only(tmp_path):
    assert _runner_accepts_flag("run_fvg_paper.py"), \
        "run_fvg_paper.py does not accept --executor-only — the systemd " \
        "unit shares-alpaca-fvg-executor.service will crash-loop with " \
        "status=2/INVALIDARGUMENT"


def test_engulfing_runner_accepts_executor_only(tmp_path):
    assert _runner_accepts_flag("run_engulfing_paper.py"), \
        "run_engulfing_paper.py does not accept --executor-only — the " \
        "systemd unit shares-alpaca-engulfing-paper.service will " \
        "crash-loop with status=2/INVALIDARGUMENT"


def test_systemd_units_use_flags_the_runners_accept(tmp_path):
    """Every ExecStart flag must be understood by the script it calls.

    This is the general form of the bug: a unit passes a flag the
    runner does not define, argparse exits 2, and the unit hot-loops.
    """
    import re
    unit_dir = Path.home() / ".config/systemd/user"
    pairs = [
        ("shares-alpaca-fvg-executor.service", "run_fvg_paper.py"),
        ("shares-alpaca-fvg-scanner.service", "run_fvg_paper.py"),
        ("shares-alpaca-engulfing-paper.service", "run_engulfing_paper.py"),
    ]
    for unit_name, script in pairs:
        unit = unit_dir / unit_name
        if not unit.exists():
            continue
        text = unit.read_text()
        m = re.search(r"^ExecStart=.*" + re.escape(script) + r"(.*)$",
                      text, re.MULTILINE)
        assert m, f"{unit_name}: no ExecStart line for {script}"
        flags = re.findall(r"--[a-z0-9-]+", m.group(1))
        help_res = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            capture_output=True, text=True, timeout=120,
        )
        assert help_res.returncode == 0, help_res.stderr[-500:]
        for flag in flags:
            assert flag in help_res.stdout, \
                f"{unit_name} passes {flag} but {script} does not accept it"


if __name__ == "__main__":
    failures = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(td)
                print(f"PASS {fn.__name__}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {fn.__name__}: {e}")
            except Exception as e:
                failures += 1
                print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
