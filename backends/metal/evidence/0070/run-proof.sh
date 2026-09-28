#!/bin/sh
set -eu

EVIDENCE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$EVIDENCE/../../../.." && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/task0070-proof.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM
LEAN=/Users/phujka/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean

cd "$EVIDENCE"
"$LEAN" -R "$ROOT" -o "$TMP/Task0069Binary32.olean" \
  "$ROOT/backends/metal/evidence/0069/proof/Task0069Binary32.lean"
for proof in Floor Ceil Sign Relu; do
  LEAN_PATH="$TMP" "$LEAN" "proof/Task0070${proof}.lean"
done
python3 verify-source-certificate.py
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun --sdk macosx clang -fobjc-arc -Wall -Wextra -Werror \
    -I "$ROOT/native/metal-macos-arm64/src" -framework Foundation \
    "$EVIDENCE/pointwise-cap-fixture.m" -o "$TMP/pointwise-cap-fixture"
"$TMP/pointwise-cap-fixture"
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun --sdk macosx clang -fobjc-arc -Wall -Wextra -Werror \
    -I "$ROOT/native/metal-macos-arm64/src" -framework Foundation \
    "$EVIDENCE/pointwise-count-fixture.m" -o "$TMP/pointwise-count-fixture"
"$TMP/pointwise-count-fixture"
"$EVIDENCE/run-compiled-generated-msl-audit.sh"
"$EVIDENCE/run-dispatch-observer.sh"
"$ROOT/native/metal-macos-arm64/build.sh"
cd "$ROOT"
env SYNAPTIK_METAL_TEST_LIBRARY="$ROOT/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  "$ROOT/gradlew" :testing:integration-tests:test --rerun-tasks --no-build-cache \
    --no-configuration-cache \
    --tests io.github.pho001.synaptik.testing.integration.EngineExplicitCompositionMetalIntegrationTest.cpuFreeMetalEngineRunsGeneratedPointwiseChainThroughOnePartition

TEST_CLASS="$ROOT/testing/integration-tests/build/classes/java/test/io/github/pho001/synaptik/testing/integration/EngineExplicitCompositionMetalIntegrationTest.class"
TEST_SOURCE="$ROOT/testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineExplicitCompositionMetalIntegrationTest.java"
[ -f "$TEST_CLASS" ] || { printf '%s\n' "checkout-local public proof class is absent: $TEST_CLASS" >&2; exit 1; }
[ "$TEST_CLASS" -nt "$TEST_SOURCE" ] || { printf '%s\n' "public proof class is not current for this checkout: $TEST_CLASS" >&2; exit 1; }
