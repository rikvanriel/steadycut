"""The package must not carry one person's recordings or one machine's paths.

A recording name or a home directory in shipped code is one of two things:
private material leaking into a repository other people read, or a default that
quietly measures one ride while looking general. Neither shows up in review, so
they are checked here.

Footage belongs in an argument -- a command line, an environment variable, a
fixture -- never in a module constant. The placeholder user names that
documentation is allowed to use are listed rather than imported, so the check
does not need to know whose machine it runs on.
"""

import pathlib
import re

import steadycut

ROOT = pathlib.Path(steadycut.__file__).parent

RECORDING = re.compile(r"VID_\d{6,}")
HOME_PATH = re.compile(r"/(?:home|Users)/([\w.-]+)")
PLACEHOLDER_USERS = {"example-user", "user", "username", "you"}


def shipped_modules():
    """Every module that ships, i.e. everything outside tests and caches."""
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        yield path


def lines(pattern):
    """(path, line number, line) for every line of shipped code matching."""
    for path in shipped_modules():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                yield path.relative_to(ROOT), n, line


def where(path, n, line):
    return f"{path}:{n}: {line.strip()[:70]}"


def test_no_module_names_a_recording():
    """Which recording a measurement ran on is not a property of the code."""
    found = [where(p, n, line) for p, n, line in lines(RECORDING)]
    assert not found, found


def test_no_module_hardcodes_a_home_directory():
    """A path into one person's home is not usable by anyone else."""
    found = [where(p, n, line) for p, n, line in lines(HOME_PATH)
             if any(u not in PLACEHOLDER_USERS for u in HOME_PATH.findall(line))]
    assert not found, found
