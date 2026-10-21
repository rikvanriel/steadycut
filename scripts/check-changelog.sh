#!/bin/bash
# commit-msg hook: reject upstream changelogs that leak internal
# context (local paths, infra names, process references). Upstream
# readers have no board, no homedir, no scratch disk.
# Install: cp scripts/check-changelog.sh .git/hooks/commit-msg
# Usage:   git log -1 --pretty=%B | scripts/check-changelog.sh --stdin
#          (the hook passes the message file as $1)
set -u
SRC="${1:-/dev/stdin}"
if [ "$SRC" = "--stdin" ]; then SRC=/dev/stdin; fi
PAT='(/backup/|/data/|/home/|~/|/tmp/|/source/|fangorn|glamdring|lemmalog|sashiko|semcode|zoekt|kanban|[Rr]eviewer|[Vv]erdict|([Rr]ound [0-9])|researcher|subagent|\bopus\b|t_[0-9]{4})'
HITS=$(grep -inE "$PAT" "$SRC" || true)
if [ -n "$HITS" ]; then
    echo "commit-msg rejected: internal context in changelog:" >&2
    echo "$HITS" >&2
    echo "(see docs/sources.md for how to cite public sources instead)" >&2
    exit 1
fi
exit 0
