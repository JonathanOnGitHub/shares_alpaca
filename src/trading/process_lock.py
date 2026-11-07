"""
process_lock.py — single-instance guard for the paper-trading daemons.

Why this exists
---------------
On 2026-11-07 the audit found TWO `run_engulfing_paper.py` daemons
running at the same time (PIDs 113311 and 537265), plus a third spawned
by a systemd unit that was crash-looping on an unsupported CLI flag.

Two daemons for one strategy is not merely untidy: both poll the same
signals file, both keep their OWN in-memory `_acted_signals` set and
their OWN `.positions_<strategy>.json`, and both submit orders to the
same shared Alpaca paper account. Each therefore believes a setup is
un-traded and can enter it a second time — and whichever daemon writes
the position file last erases the other's view of an open position, so
the position ends up unmanaged at the broker with no stop and no
target.

A PID file cannot prevent this: the wrapper overwrites the PID file
every time it starts a daemon, so a daemon started by an earlier run
becomes invisible to the next run's "is it already running?" check.

An advisory file lock (flock) is the right primitive here, because it
is released automatically by the kernel when the holding process dies
for ANY reason — including SIGKILL by the OOM killer. There is no
stale lock file to clean up.

Usage
-----
    from src.trading.process_lock import acquire_lock

    lock = acquire_lock(Path("/path/to/.executor_engulfing.lock"))
    if lock is None:
        raise SystemExit("another daemon already holds the lock")
    # keep `lock` referenced for the lifetime of the process; the lock
    # is released when the object is garbage collected or the process
    # exits.
"""
import fcntl
import os
from pathlib import Path
from typing import Optional


def acquire_lock(lock_path: Path, timeout: float = 0.0) -> Optional[object]:
    """
    Try to take an exclusive advisory lock on `lock_path`.

    Returns the open file object holding the lock on success, or None
    if another live process already holds it. The caller MUST keep the
    returned object alive for as long as it wants the lock held.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None

    f = os.fdopen(fd, "r+")
    try:
        f.seek(0)
        f.truncate()
        f.write(f"{os.getpid()}\n")
        f.flush()
    except Exception:
        pass
    return f


def lock_holder_pid(lock_path: Path) -> Optional[int]:
    """Return the PID recorded in the lock file, if any.

    This is only a hint: the recorded PID may have died without
    releasing the lock cleanly. Use it for logging, not for decisions.
    """
    try:
        txt = Path(lock_path).read_text().strip()
        return int(txt) if txt else None
    except Exception:
        return None
