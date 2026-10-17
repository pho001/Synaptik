#!/bin/sh
set -eu

EVIDENCE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/task0071-proof.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM
LEAN=/Users/phujka/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean

(
  cd "$EVIDENCE/../0069"
  "$LEAN" -o "$TMP/Task0069Binary32.olean" proof/Task0069Binary32.lean
)
cd "$EVIDENCE"
LEAN_PATH="$TMP" "$LEAN" proof/Task0071AnchorEpilogue.lean
python3 verify-source-certificate.py
"$EVIDENCE/run-compiled-anchor-audit.sh"
