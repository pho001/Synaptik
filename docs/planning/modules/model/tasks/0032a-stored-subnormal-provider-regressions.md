# Task 0032A: Stored-subnormal provider regressions

## Status

Complete — the sole serial Model test frontier before Draft 0032. Model 0030/0031, CPU 0017,
and Metal 0063 are Complete in the current plans. The existing CPU generated and Metal
custom-program routes are available; a verified local Metal native package and temporary
CPU/Metal probes passed on this machine. At launch, recheck the package against current
source and ensure no conflicting provider or native edits are running.

## Change class

Class A — two backend-local regression classes only; no change to numerical semantics,
capability, generated-code invariants, native application binary interface (ABI), or behavior.

## Goal

Certify the *current providers' realization* of stored `FLOAT32` minimum positive subnormal
(`0x00000001`) versus positive zero (`0x00000000`). These observations do not strengthen
Model's contract: `ACCELERATOR` may denormals-are-zero (DAZ)-normalize comparisons and
treat extrema as tied while returning an original candidate.

## Scope

- Add one new CPU backend-local JUnit class. Under both `STRICT_IEEE` and `ACCELERATOR`,
  execute generated scalar and Vector API `FLOAT32` greater-than (`GT`) and equality
  (`EQ`), tensor/scalar `MIN` and `MAX`, and generated `ARG_MAX`. Force at least one
  full Vector chunk and a scalar tail; test both input orders and raw result bits.
- Add one new Metal backend-local native JUnit class. Under `ACCELERATOR`, execute the
  corresponding `FLOAT32` cases on the actual `CUSTOM_PROGRAM` route. Assert raw
  predicate/index/result bytes and unchanged input bytes. For `[+0, minSubnormal, +0]`,
  both `ARG_MAX` FIRST and LAST must choose unique winner index 1; also use true-equal
  inputs to prove FIRST and LAST select different tie indices.
- Unconfigured CI may skip native tests only with a clear missing-package/device reason.
  A configured verified native package must execute the Metal cases with `skipped=0`.

## Non-goals

No production, architecture, provider-capability, native, existing-test, or 0032 edits;
no MPSGraph substitution, low-precision expansion, or general numerical suite.

## Contracts

- [Root core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts):
  Model owns profile-indexed allowed results; provider tests do not redefine them.
- [Model numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles),
  especially the [exact and discrete floor](../../../../architecture/contracts/foundational-modules.md#exact-and-discrete-floor):
  `ACCELERATOR` comparison/extrema DAZ freedom remains authoritative.
- [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend):
  generated CPU and fixed custom Metal execution remain backend-owned.

If current source or preflight contradicts the observed expectation, report it; do not
modify production or assert a stricter Model rule.

## Dependencies and integration

- Depends on: Model 0030/0031, CPU 0017, Metal 0063 (Complete); verified native package
  and available Metal device for the configured local gate.
- Conflicts with: Model 0032 execution and concurrent CPU/Metal pointwise, `ARG_MAX`,
  numerical-authority, or native-source edits; serialize these scopes.
- Parallel group: None.
- Common base revision: N/A (serial task; preserve the dirty worktree).
- Integration order: finish 0032A before considering the 0032 cutover.
- Integration validation: focused CPU and packaged-native Metal JUnit, Metal XML
  `tests>0`, `failures=0`, `errors=0`, `skipped=0`, and scoped whitespace/link checks.
- Shared-document integration owner: Model planning coordinator; the test worker changes
  only the two new backend-local classes and records the result in this brief.

## Files and symbols

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuStoredSubnormalProviderRegressionTest.java`
  — generated pointwise scalar/Vector and `ARG_MAX` checks.
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalStoredSubnormalProviderNativeTest.java`
  — native `CUSTOM_PROGRAM` pointwise and arg-extrema checks.

## Acceptance criteria

- Both CPU profiles retain `minSubnormal > +0`, `minSubnormal != +0`, raw `MIN=+0`,
  raw `MAX=minSubnormal`, and `ARG_MAX` index 1 in scalar and required Vector/tail cases.
- Configured Metal `ACCELERATOR` custom execution shows the same distinction,
  original input bytes, unique-winner FIRST/LAST index 1, and correct true-equal
  FIRST/LAST control indices. No configured case skips or changes route.
- The implementation touches only the two new JUnit classes and this brief's Result;
  existing dirty files remain intact.

## Validation

Use the [Metal local package instructions](../../../../../native/metal-macos-arm64/README.md#build-prerequisites-and-local-package)
to rebuild/sign/package if source changed, then verify the package immediately before the
configured test. Point `SYNAPTIK_METAL_TEST_LIBRARY` at its packaged dylib, not a loose build.

```bash
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
./gradlew :backends:cpu:test --tests '*CpuStoredSubnormalProviderRegressionTest'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test --tests '*MetalStoredSubnormalProviderNativeTest' --rerun-tasks
git diff --check
```

Inspect the focused Metal JUnit XML for the configured counts above. Record package
verification, commands, counts, and skips in Result. Unconfigured CI skips do not satisfy
the configured native acceptance gate. No repository-wide suite is needed for this
test-only task.

## Documentation and review impact

The implementer reviews affected test/Javadoc context, backend documentation, relevant
glossary terms, links, and final diff. Expected documentation impact: none; these tests
observe existing behavior without changing a public, implementation, or architecture
contract. No independent documentation context is required for Class A.

## Result

Added only `CpuStoredSubnormalProviderRegressionTest.java` and
`MetalStoredSubnormalProviderNativeTest.java`; all other dirty worktree files were preserved.
The CPU class runs generated scalar and preferred-species Vector API pointwise cases under both
profiles, checks a full lane chunk and scalar tail, both operand orders, raw predicate/extrema
bits, unchanged inputs, and generated scalar `ARG_MAX`. The Metal class runs the real-device
`CUSTOM_PROGRAM` route under `ACCELERATOR`, checks raw output bytes and unchanged input bytes,
and distinguishes the unique arg-max winner from genuine FIRST/LAST ties. These assertions
remain backend-local observations, not a stronger Model guarantee.

Validation: `./native/metal-macos-arm64/verify-package.sh
native/metal-macos-arm64/build/package-v1/macos-arm64` passed immediately before the final
configured Metal run. The packaged dylib is newer than current native source, so no rebuild was
needed. `./gradlew :backends:cpu:test --tests '*CpuStoredSubnormalProviderRegressionTest'`
passed: JUnit XML `tests=2, failures=0, errors=0, skipped=0`. The first CPU run exposed a
test-fixture schema mismatch; the new class alone was corrected to use schema 61 for its Vector
BOOL-mask boundary and schema 52 otherwise, then the focused run passed. With
`SYNAPTIK_METAL_TEST_LIBRARY` set to the verified packaged dylib,
`./gradlew :backends:metal:test --tests '*MetalStoredSubnormalProviderNativeTest'
--rerun-tasks` passed: configured JUnit XML `tests=2, failures=0, errors=0, skipped=0`.
`git diff --check` passed; scoped no-index whitespace checks of both new files reported no
errors. The brief's contract links and exact headings were checked during execution.

Documentation impact: none; the change only records existing provider behavior, and the two
new test-class Javadocs state that it does not narrow Model's `ACCELERATOR` freedom. Relevant
backend-guide and glossary terms were reviewed with targeted searches; no existing Javadoc,
example, or glossary definition changed. Architecture, backend-conformance, and integration
test modules need no changes because no behavior or module boundary changed, while direct
backend-route internal access requires these backend-local tests. Unresolved issues: none.
Follow-up required: none.

Status: Complete
