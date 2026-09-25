"""Locate the footage without hard-coding a path into every script.

The development scripts were written against one machine's download directory. That
is fine locally and wrong for anything shared, so the directory now comes from the
environment instead, defaulting to the working directory so the tools run from
wherever the footage happens to be:

    STEADYCUT_FOOTAGE=/path/to/footage python3 compare_policies.py
    cd /path/to/footage && python3 ...            # equivalent

Only the directory is configured, not individual files, because the scripts
reference different clips and modes of the same footage.
"""
import os
from pathlib import Path

ENV_VAR = "STEADYCUT_FOOTAGE"


def footage_root() -> Path:
    """Directory holding the .insv files, from the environment."""
    return Path(os.environ.get(ENV_VAR, "."))


def clip(*parts: str) -> Path:
    """A path under the footage directory, e.g. clip("reel", "a.insv")."""
    return footage_root().joinpath(*parts)