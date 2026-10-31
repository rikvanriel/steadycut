"""Render subprocess owned by the GUI: process-group TERM, then KILL.

The CLI spawns ffmpeg children, so killing the parent PID alone strands
them: the process starts in its own session and cancel signals the whole
group. Qt-free; the app polls `proc.poll()` on a QTimer.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


def start_job(argv: list[str], log_path: str | Path) -> tuple:
    """Launch argv with output to log_path; returns (proc, log_fh).

    `start_new_session` puts the child (and its ffmpeg) in a fresh
    process group the GUI owns for cancel. The caller keeps log_fh open
    until the process exits, then closes it.
    """
    log_fh = open(log_path, "wb")
    proc = subprocess.Popen(argv, stdout=log_fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc, log_fh


def cancel(proc: subprocess.Popen, grace_s: float = 5.0) -> str:
    """Stop a job: TERM the group, KILL it past the grace period.

    Returns "done" (already exited), "term", or "kill". Never raises
    for a racy exit: a process that dies mid-cancel counts as done.
    """
    if proc.poll() is not None:
        return "done"
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return "done"
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return "term"
        time.sleep(0.1)
    if proc.poll() is not None:
        return "term"
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return "done"
    proc.wait()
    return "kill"


def group_pids(pid: int) -> list[int]:
    """PIDs still alive in `pid`'s process group (Linux /proc scan)."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return []
    alive = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[2]) == pgid:  # pgrp field
                alive.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return alive
