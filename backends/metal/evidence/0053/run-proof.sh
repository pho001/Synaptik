#!/bin/sh
set -eu

EVIDENCE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$EVIDENCE/../../../.." && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/task0053-proof.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

cd "$EVIDENCE"
python3 tools/verify_toolchain.py

clang -std=c17 -O2 -Wall -Wextra -Werror tools/generate_constants.c \
  -I/opt/homebrew/Cellar/mpfr/4.2.2/include \
  -I/opt/homebrew/Cellar/gmp/6.3.0/include \
  -L/opt/homebrew/Cellar/mpfr/4.2.2/lib \
  -L/opt/homebrew/Cellar/gmp/6.3.0/lib -lmpfr -lgmp \
  -o "$TMP/generate_constants"
"$TMP/generate_constants" \
  "$TMP/task0053_constants.h" "$TMP/constants.json" "$TMP/directed-certificates.json"
cmp generated/task0053_constants.h "$TMP/task0053_constants.h"
cmp generated/constants.json "$TMP/constants.json"
cmp generated/directed-certificates.json "$TMP/directed-certificates.json"
python3 tools/generate_partitions.py --check

python3 "$ROOT/native/metal-macos-arm64/generate-task0053-header.py" --check
python3 tools/audit_source.py
python3 tools/generate_lean_constants.py --check
python3 tools/generate_lean_partitions.py --check
LEAN=/Users/phujka/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean
"$LEAN" -o "$TMP/Task0053Generated.olean" proof/Task0053Generated.lean
LEAN_PATH="$TMP" "$LEAN" -o "$TMP/Task0053Integer.olean" proof/Task0053Integer.lean
LEAN_PATH="$TMP" "$LEAN" proof/Task0053Partitions.lean
"$LEAN" proof/Task0053Structural.lean
/opt/homebrew/bin/sollya proof/polynomial.sollya > "$TMP/polynomial-report.txt"
cmp generated/polynomial-report.txt "$TMP/polynomial-report.txt"

clang -std=c17 -O3 -Wall -Wextra -Werror \
  checker/task0053_all_words.c model/task0053_model.c \
  -I/opt/homebrew/Cellar/mpfr/4.2.2/include \
  -I/opt/homebrew/Cellar/gmp/6.3.0/include \
  -L/opt/homebrew/Cellar/mpfr/4.2.2/lib \
  -L/opt/homebrew/Cellar/gmp/6.3.0/lib -lmpfr -lgmp -pthread \
  -o "$TMP/task0053_all_words"
python3 checker/run_all_words.py \
  "$TMP/task0053_all_words" 64 "$TMP/all-words-report.json"
cmp generated/all-words-report.json "$TMP/all-words-report.json"

python3 tools/write_manifest.py --check
python3 tools/assert_proof_gate.py
