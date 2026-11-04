# Task 0032A: Stored-[subnormal](../../../../glossary.md#term-subnormal) provider regressions

**Editorial reading key (current context, not a status update).**

This historical task records backend-local observations, not a stronger Model contract or a current
execution report. Model defines allowed results; the test caller exercises CPU generated
scalar/Vector routes and Metal’s `CUSTOM_PROGRAM` callee, which realize supported routes rather than
redefine semantics. Here, stored means input values whose raw bits are inspected: `0x00000001` is
the smallest positive `FLOAT32` subnormal; `0x00000000` is positive zero. In the recorded
three-element example, `ARG_MAX` selects index 1; FIRST/LAST choose among genuinely tied candidates,
not different unique winners.

Keep the historical `ACCELERATOR` DAZ allowance distinct from the current [data-type
contract](../../../../reference/tensor/data-types.md#data-types), which describes stored comparisons
as exact; DAZ treats permitted inputs as zero, whereas FTZ concerns results. Follow [exact typed
scalar values](../../../../reference/tensor/data-types.md#exact-typed-scalar-values) to distinguish
a tensor/scalar operation’s exact-bit parameter from Tensor storage. Tensor expressions carry
descriptions, not executed results; provider execution consumes actual inputs and produces
predicate, extremum or index values. Model’s semantic authority does not imply ownership of those
buffers.

## Status

Complete — the sole serial [Model](../../../../glossary.md#term-model) test frontier before Draft 0032. Model 0030/0031, [CPU](../../../../glossary.md#term-cpu) 0017,
and [Metal](../../../../glossary.md#term-metal) 0063 are Complete in the current plans. The existing CPU generated and Metal
custom-program routes are available; a verified local Metal native package and temporary
CPU/Metal probes passed on this machine. At launch, recheck the package against current
source and ensure no conflicting provider or native edits are running.

## Change class

Class A — two [backend](../../../../glossary.md#backend)-local [regression](../../../../glossary.md#term-regression) classes only; no change to numerical semantics,
capability, generated-code invariants, native [application binary interface](../../../../glossary.md#term-application-binary-interface-abi) ([ABI](../../../../glossary.md#application-binary-interface-and-native-bridge)), or behavior.

## Goal

Certify the *current providers' realization* of stored [`FLOAT32`](../../../../glossary.md#term-float32) minimum positive [subnormal](../../../../glossary.md#term-subnormal)
(`0x00000001`) versus [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) (`0x00000000`). These observations do not strengthen
[Model](../../../../glossary.md#term-model)'s contract: [`ACCELERATOR`](../../../../glossary.md#term-accelerator) may [denormals-are-zero](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow) ([DAZ](../../../../glossary.md#term-daz))-normalize comparisons and
treat [extrema](../../../../glossary.md#saturation-pole-and-extremum) as tied while returning an original candidate.

## [Scope](../../../../glossary.md#term-scope)

- Add one new [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend)-local [JUnit](../../../../glossary.md#term-junit) class. Under both `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator),
  execute generated scalar and Vector API <span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api))</span>[`FLOAT32`](../../../../glossary.md#term-float32) greater-than (`GT`) and equality
  (`EQ`), [tensor](../../../../glossary.md#tensor)/[scalar](../../../../glossary.md#term-scalar) `MIN` and `MAX`, and generated `ARG_MAX`. Force at least one
  full Vector chunk and a scalar tail; test both input orders and raw result bits.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [scalar tail glossary](../../../../glossary.md#term-scalar-tail))</span>
- Add one new [Metal](../../../../glossary.md#term-metal) backend-local native JUnit class. Under `ACCELERATOR`, execute the
  corresponding `FLOAT32` cases on the actual `CUSTOM_PROGRAM` route. Assert raw
  predicate/index/result bytes and unchanged input bytes. For `[+0, minSubnormal, +0]`,
  both `ARG_MAX` FIRST and LAST must choose unique winner index 1; also use true-equal
  inputs to prove FIRST and LAST select different tie indices.
- Unconfigured CI may skip native tests only with a clear missing-package/device reason.
  A configured verified native package must execute the Metal cases with <span data-glossary-navigation="true"> ([CI glossary](../../../../glossary.md#term-continuous-integration-ci); [device glossary](../../../../glossary.md#term-device))</span>`skipped=0`.

## Non-goals

No production, architecture, provider-capability, native, existing-test, or 0032 edits;
no [MPSGraph](../../../../glossary.md#term-mpsgraph) substitution, low-[precision](../../../../glossary.md#term-precision) expansion, or general numerical suite.

## Contracts

- [Root core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>:
  Model owns profile-indexed allowed results; provider tests do not redefine them.
- [Model numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles),
  especially the [exact and discrete floor](../../../../architecture/contracts/foundational-modules.md#exact-and-discrete-floor):
  [`ACCELERATOR`](../../../../glossary.md#term-accelerator) comparison/[extrema](../../../../glossary.md#saturation-pole-and-extremum) [DAZ](../../../../glossary.md#term-daz) freedom remains authoritative.
- [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>:
  generated CPU and fixed custom Metal execution remain backend-owned.

If current source or [preflight](../../../../glossary.md#term-preflight) contradicts the observed expectation, report it; do not
modify production or assert a stricter [Model](../../../../glossary.md#term-model) rule.

## Dependencies and integration

- Depends on: Model 0030/0031, CPU 0017, Metal 0063 (Complete); verified native package
  and available Metal device for the configured local gate.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- Conflicts with: [Model](../../../../glossary.md#term-model) 0032 execution and concurrent [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) [pointwise](../../../../glossary.md#term-pointwise), `ARG_MAX`,
  numerical-authority, or native-source edits; serialize these scopes.
- Parallel group: None.
- Common base revision: N/A (serial task; preserve the dirty worktree).
- Integration order: finish 0032A before considering the 0032 cutover.
- Integration [validation](../../../../glossary.md#term-validation): focused CPU and packaged-native Metal [JUnit](../../../../glossary.md#term-junit), Metal XML
  `tests>0`, `failures=0`, `errors=0`, `skipped=0`, and scoped whitespace/link checks.
- Shared-document integration owner: Model planning coordinator; the test worker changes
  only the two new backend-local classes and records the result in this brief.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend))</span>

## Files and symbols

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuStoredSubnormalProviderRegressionTest.java`
  — generated pointwise scalar/Vector and <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Vector glossary](../../../../glossary.md#term-vector))</span>`ARG_MAX` checks.
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalStoredSubnormalProviderNativeTest.java`
  — native `CUSTOM_PROGRAM` [pointwise](../../../../glossary.md#term-pointwise) and arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum) checks.

## Acceptance criteria

- Both [CPU](../../../../glossary.md#term-cpu) profiles retain `minSubnormal > +0`, `minSubnormal != +0`, raw `MIN=+0`,
  raw `MAX=minSubnormal`, and `ARG_MAX` index 1 in [scalar](../../../../glossary.md#term-scalar) and required [Vector](../../../../glossary.md#term-vector)/tail cases.
- Configured [Metal](../../../../glossary.md#term-metal) [`ACCELERATOR`](../../../../glossary.md#term-accelerator) custom execution shows the same distinction,
  original input bytes, unique-winner FIRST/LAST index 1, and correct true-equal
  FIRST/LAST control indices. No configured case skips or changes route.
- The implementation touches only the two new JUnit classes and this brief's Result;
  existing dirty files remain intact.<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit))</span>

## [Validation](../../../../glossary.md#term-validation)

Use the [Metal local package instructions](../../../../../modules/backend/metal/macos-arm64/README.md#build-prerequisites-and-local-package)
to rebuild/sign/package if source changed, then verify the package immediately before the
configured test. Point `SYNAPTIK_METAL_TEST_LIBRARY` at its packaged [dylib](../../../../glossary.md#term-dylib), not a loose build.

```bash
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
./gradlew :backends:cpu:test --tests '*CpuStoredSubnormalProviderRegressionTest'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test --tests '*MetalStoredSubnormalProviderNativeTest' --rerun-tasks
git diff --check
```

Inspect the focused [Metal](../../../../glossary.md#term-metal) [JUnit](../../../../glossary.md#term-junit) XML for the configured counts above. Record package
verification, commands, counts, and skips in Result. Unconfigured [CI](../../../../glossary.md#term-continuous-integration-ci) skips do not satisfy
the configured native acceptance gate. No repository-wide suite is needed for this
test-only task.

## Documentation and review impact

The implementer reviews affected test/[Javadoc](../../../../glossary.md#term-javadoc) context, [backend](../../../../glossary.md#backend) documentation, relevant
glossary terms, links, and final diff. Expected documentation impact: none; these tests
observe existing behavior without changing a public, implementation, or [architecture
contract](../../../../glossary.md#architecture-contract). No independent documentation context is required for Class A.

## Result

Added only `CpuStoredSubnormalProviderRegressionTest.java` and
`MetalStoredSubnormalProviderNativeTest.java`; all other dirty worktree files were preserved.
The [CPU](../../../../glossary.md#term-cpu) class runs generated [scalar](../../../../glossary.md#term-scalar) and preferred-species [Vector API](../../../../glossary.md#term-vector-api) [pointwise](../../../../glossary.md#term-pointwise) cases under both
profiles, checks a full [lane](../../../../glossary.md#term-lane) chunk and [scalar tail](../../../../glossary.md#term-scalar-tail), both operand orders, raw predicate/[extrema](../../../../glossary.md#saturation-pole-and-extremum)
bits, unchanged inputs, and generated scalar `ARG_MAX`. The [Metal](../../../../glossary.md#term-metal) class runs the real-[device](../../../../glossary.md#term-device)
`CUSTOM_PROGRAM` route under [`ACCELERATOR`](../../../../glossary.md#term-accelerator), checks raw output bytes and unchanged input bytes,
and distinguishes the unique arg-max winner from genuine FIRST/LAST ties. These assertions
remain [backend](../../../../glossary.md#backend)-local observations, not a stronger [Model](../../../../glossary.md#term-model) guarantee.

[Validation](../../../../glossary.md#term-validation): `./native/metal-macos-arm64/verify-package.sh
native/metal-macos-arm64/build/package-v1/macos-arm64` passed immediately before the final
configured Metal [run](../../../../glossary.md#term-run). The packaged [dylib](../../../../glossary.md#term-dylib) is newer than current native source, so no rebuild was
needed. `./gradlew :backends:cpu:test --tests '*CpuStoredSubnormalProviderRegressionTest'`
passed: [JUnit](../../../../glossary.md#term-junit) XML `tests=2, failures=0, errors=0, skipped=0`. The first CPU run exposed a
test-[fixture](../../../../glossary.md#term-fixture) [schema](../../../../glossary.md#term-schema) mismatch; the new class alone was corrected to use schema 61 for its [Vector](../../../../glossary.md#term-vector)
[BOOL](../../../../glossary.md#data-type--datatype)-[mask](../../../../glossary.md#term-boolean-mask) boundary and schema 52 otherwise, then the focused run passed. With
`SYNAPTIK_METAL_TEST_LIBRARY` set to the verified packaged dylib,
`./gradlew :backends:metal:test --tests '*MetalStoredSubnormalProviderNativeTest'
--rerun-tasks` passed: configured JUnit XML `tests=2, failures=0, errors=0, skipped=0`.
`git diff --check` passed; scoped no-index whitespace checks of both new files reported no
errors. The brief's contract links and exact headings were checked during execution.

Documentation impact: none; the change only records existing provider behavior, and the two
new test-class Javadocs state that it does not narrow Model's `ACCELERATOR` freedom. Relevant
backend-guide and glossary terms were reviewed with targeted searches; no existing [Javadoc](../../../../glossary.md#term-javadoc),
example, or glossary definition changed. Architecture, backend-[conformance](../../../../glossary.md#term-conformance), and [integration
test](../../../../glossary.md#term-integration-test) [modules](../../../../glossary.md#term-module) need no changes because no behavior or module boundary changed, while direct
backend-route internal access requires these backend-local tests. Unresolved issues: none.
Follow-up required: none.

Status: Complete
