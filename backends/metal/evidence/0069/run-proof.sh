#!/bin/sh
set -eu

EVIDENCE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/task0069-proof.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM
LEAN=/Users/phujka/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean

cd "$EVIDENCE"
"$LEAN" -o "$TMP/Task0069Binary32.olean" proof/Task0069Binary32.lean
LEAN_PATH="$TMP" "$LEAN" proof/Task0069ReductionTree.lean
python3 verify-source-certificate.py
"$EVIDENCE/run-compiled-msl-audit.sh"
