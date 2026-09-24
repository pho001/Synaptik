# Task 0015: ACCELERATOR FLOAT32 Tensor Binary Arithmetic

## Status

Ready

Readiness verification: Metal 0014, Model 0027, Config 0006, and Engine 0018 are Complete. Engine
0018 landed at implementation `ce7a7dfa` plus documentation/Javadoc remediation `07a01b9c` with
required validation and independent Class C approval. Its integrated version-five Metal workload,
candidate, compatibility, route-policy, and codec identities already include the graph profile.
Metal 0015 and CPU 0017 have disjoint backend semantics and write scopes after that stable spine;
shared status and explanatory documents are reserved to the integration owner.

## Change class

Class C — this restores profile-qualified public capability through a native MPSGraph schema,
whole-partition lowering, persistent executable lifecycle, backend cache compatibility, and real
public Engine execution. Fresh operation-by-operation device evidence and independent review are
mandatory.

## Goal

Under `NumericalProfile.ACCELERATOR` only, restore tensor `FLOAT32` `ADD`, `SUB`, `MUL`, and `DIV`
for the measured MPSGraph DAZ/FTZ result envelope permitted by the Model contract. Keep the strict
binary matrix fail-closed, preserve existing exact NEG/affine/`CONTIGUOUS` support under
`STRICT_IEEE`, and reject every unproved scalar, unary, reduction, MATMUL, comparison, extrema,
scan, and other successor under `ACCELERATOR`.

## Exact capability and topology

- `STRICT_IEEE` remains unchanged: exact existing NEG, five affine transforms, and `CONTIGUOUS` are
  supported; all tensor binary arithmetic remains false and forced preparation rejects before a
  native resource is created.
- `ACCELERATOR` supports only `BinaryArithmeticKind.ADD`, `SUB`, `MUL`, and `DIV` with
  `NoOperationAttrs.INSTANCE`; existing NEG, affine, and `CONTIGUOUS` rows remain unsupported.
- Each binary occurrence has two `FLOAT32` inputs and one `FLOAT32` output; fully static positive
  rank `1..16`; resolved canonical dense-contiguous, zero-offset, non-view descriptors; equal
  `requiresGrad`; checked element/byte geometry; and exact right-aligned broadcasting. Aligned
  extents are equal or one, missing leading axes are implicit ones, and output extents are the
  aligned maxima. Operand order is immutable for `SUB` and `DIV`.
- ACCELERATOR partitions may compose only the four binary kinds. Every input is canonical and every
  output is canonical; an affine view must be canonicalized by another eligible backend before
  transfer. A binary output may feed another binary or publication. Fan-out, repeated operands,
  multiple feeds/targets, splat tensor feeds, and published intermediates retain stable order. No
  implicit canonicalization, baseline-node admission, or cross-node algebraic fusion.
- This is forward Metal execution only. Compiler-generated backward nodes may be CPU-owned in a
  mixed engine only when independently eligible; no Metal-only backward/training claim is added.

## Bounded numerical contract

For each binary operation, every produced lane must belong to the sole Model `ACCELERATOR` result
set: each declared subnormal input may independently become same-signed zero before that operation;
ordinary `FLOAT32` arithmetic then rounds once; a finite subnormal result may become zero with
either sign. Exact-zero `ADD`/`SUB` may use either sign. `MUL`/`DIV` otherwise retain the sign
required by the selected operands. Current NaN freedoms remain operation-owned.
No tolerance, ULP budget, host rewrite, value-dependent fallback, reciprocal substitution, term
loss, reduced precision, cross-node contraction, or special-case dispatch is permitted.
`maxFinite / maxFinite -> 0` and `+infinity / maxFinite -> NaN` remain invalid, as do ordinary
finite mismatches outside the enumerated DAZ/FTZ set. Scalar arithmetic remains blocked, including
the gross failures recorded by Metal 0011.

## Mandatory disposable real-device probe

Before production edits, build a disposable Objective-C/MPSGraph probe at
`/tmp/synaptik-metal-0015-binary-probe.m`; remove it before completion. Run each of `ADD`, `SUB`,
`MUL`, and `DIV` as a separately reported gate at optimization levels `0` and `1`, reduced-
precision fast math disabled, three independently compiled executables per operation/level/Shape
case, and at least eight executions per executable. Include equal-Shape, row broadcast, column
broadcast, scalar-shaped tensor broadcast, repeated operand, fan-out, and ordered `SUB`/`DIV`
cases. Bind direct target buffers, permute feed/target order, preserve inputs, surround every target
with canaries, and repeat in two independently opened contexts.
Use a fixed raw-bit corpus covering both zeros; positive/negative minimum, ordinary, and maximum
subnormals; minimum normals and adjacent values; `0.5`, `1`, `2`; opposite ordinary values;
maximum finite values; both infinities; and quiet/signaling NaNs of both signs with multiple
payloads. Include directed FTZ producers (`minimum-normal * 0.5`, signed variants, and
`minimum-normal / 2`), DAZ-sensitive pairs, exact cancellation, division by zeros, and the explicit
gross-error controls above.
The oracle must enumerate the finite allowed bit set from all independent DAZ choices, one
operation, and optional FTZ; apply the exact zero-sign rules; and use current family rules for NaN
classification/payload freedom. Record totals and mismatches separately by operation, optimization
level, Shape case, and semantic bucket. Require every output to be in the enumerated set and every
control, Shape, target order, input-preservation, canary, repetition, and context-independence gate
to pass. Any out-of-set result blocks 0015 with no production/capability/schema change; do not
shrink the corpus or reinterpret the contract.

## Native schema, identity, and lifecycle

- Keep native ABI version `4`, statuses, create signature, and exactly thirteen exports. Advance the
  fixed 160-byte MPSGraph node schema `4 -> 5`; retain wires `NEG=1`, affine/contiguous `6..11`, and
  restore `ADD=2`, `SUB=3`, `MUL=4`, `DIV=5` with attribute kind `NONE`, two ordered input indices,
  zero attribute payload/reserved cells, canonical input states, exact broadcast output, and
  canonical output state.
- Java capability and preparation are the profile authority. Java native-create preflight takes the
  plan profile and enforces the closed matrix: strict permits only retained baseline wires and
  accelerator permits only wires `2..5`; every mismatch fails before downcall. The private native
  v5 schema validates topology/Shape/state/wires, not Model policy; do not change the ABI merely to
  duplicate the already authenticated cold profile.
- Advance workload-signature, exact-policy, candidate, compatibility, route-policy, and codec
  versions `5 -> 6`. Include profile, node schema 5, ordered binary kinds/edges, descriptors,
  states, targets, geometry, ABI, and splats in workload identity. Version-five, cross-profile,
  malformed, corrupt, unknown-wire, and stale-session decisions reject or miss safely.
- Preserve strict-only custom singleton NEG, MPSGraph whole-partition selection, transactional
  finalization, context leases, rollback/suppressed failures, per-run buffers/address workspace,
  direct assigned output publication, synchronous submission, reuse, concurrency, close rejection,
  and canonical-only cross-owner transfer. No route/profile lookup enters Runtime.

## Non-goals

No scalar arithmetic; binary `POW`, `MIN`, or `MAX`; comparison; unary/transcendental; SUM/MEAN or
other reduction; MATMUL/convolution; cumulative scan; BOOL/WHERE; FLOAT16/BFLOAT16/FLOAT64; custom
binary kernel; transfer/ingress widening; alias promise; backward/training; packaging/discovery; or
performance claim. Metal 0006–0007 and 0009–0013 stay Blocked. Do not authorize a successor for any
of them, and preserve Model 0026 as Draft.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — profile
  semantics remain Model-owned and profile selection remains cold.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — sole DAZ/FTZ, zero-sign, and gross-error authority.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) — Metal
  owns truthful occurrence capability, MPSGraph lowering, native integration, and resources.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) —
  route/resources are fixed before direct hot execution.
If the probe, bounded oracle, or profile guard cannot be implemented without a shared contract,
semantic rewrite, fallback, or ABI signature change, stop and report the blocker.

## Dependencies and integration

- Depends on: Metal 0014; Model 0027; Config 0006; Engine 0018
- Conflicts with: every Metal capability/preparation/native schema/candidate/codec/materialization
  scope and public Metal Engine integration
- Parallel group: `numerical-profile-backends`
- Common base revision: `07a01b9c13c2ceea0922b9af82da3e6e08375306`
- Integration order: Any relative to CPU 0017; both executable commits precede shared-document
  reconciliation
- Integration validation: numerical-profile backend integration checkpoint
- Shared-document integration owner: Main planner
The Metal worker owns Metal/native production/tests, `EngineExplicitCompositionMetalIntegrationTest`,
this brief, Metal package Javadocs, `docs/backend-guide/metal-backend.md`, and the native README. It
must not edit roadmap/master plans, root/scoped architecture, shared API/user/glossary/capability/
preparer documents, CPU paths, or `EngineNumericalProfileIntegrationTest`.

## Exact files and symbols

- `MetalCapabilityProvider.java`, `MetalNegPartitionPreparer.java`, `MetalNegPreparationPlan.java`
  — profile-qualified capability, topology/state validation, ordered binary lowering, retained plan.
- `MetalMpsGraphProgram.java`, `MetalNativeApi.java`, `MetalDeviceContext.java` — schema v5, two-input
  Java preflight, profile guard before downcall, and stable ABI-v4 marshalling.
- `MetalNegRouteCandidateGenerator.java`, `MetalNegTuningBatch.java`,
  `MetalNegTuningCodec.java`, `MetalNegTuningDecision.java` — version-six identity/compatibility.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` and
  `native/metal-macos-arm64/README.md` — v5 validation/lowering with ABI v4/export stability.
- `MetalCapabilityProviderTest.java`, `MetalMpsGraphAffineSchemaTest.java`,
  `MetalMpsGraphRawAbiNativeTest.java`, `MetalNegPreparedExecutionTest.java`,
  `MetalNegRouteCandidateGeneratorTest.java`, and new `MetalMpsGraphBinaryNativeTest.java` — positive
  and malformed schema/native/profile/lifecycle/codec coverage.
- `testing/backend-conformance/.../MetalNegCapabilityPartitionConformanceTest.java` and
  `testing/integration-tests/.../EngineExplicitCompositionMetalIntegrationTest.java` — public
  profile matrix, maximal closure, real accelerator execution, and unchanged strict rejection.
- `backends/metal/.../package-info.java`, affected type Javadocs, and
  `docs/backend-guide/metal-backend.md` — exact current scope and exclusions.

## Tests and public smoke

Tests must cover each operation separately plus binary-only chains, all broadcast directions,
operand order, repeated operands, fan-out, splats, direct intermediate/final targets,
canonical-state transitions, and malformed attrs/types/ranks/layouts/broadcasts/second-input
indices. The profile matrix must reject all baseline nodes under accelerator and all binary nodes
under strict. Raw ABI tests must reject schema 4, unknown wires, profile-incompatible preflight,
bad states/shapes/sentinels/padding, and malformed ordering before resource creation. Codec tests
must prove deterministic v6 encoding and v5/cross-profile/corrupt/stale rejection.

The real CPU-free public Engine smoke selects `ACCELERATOR`, executes all four operations with
ordinary, DAZ, FTZ, signed-zero, infinity, and broadcast cases, materializes direct publications,
reuses one session, opens an independent session, and verifies close/rejection behavior. A separate
strict Metal-only binary compile must still fail; with CPU registered it may be CPU-owned. No test
may assert only that execution does not throw.

## Acceptance criteria

1. The disposable probe passes every per-operation allowed-set and control gate before production
   edits; its source/binary is absent from the final tree.
2. Strict binary stays false/rejected. Accelerator adds exactly four tensor binary kinds; strict
   retains the exact baseline while accelerator unary/layout/scalar and all blocked families stay
   false.
3. Java/native schema v5 enforces ordered two-input canonical topology and exact broadcast Shapes;
   ABI v4 retains exactly thirteen exports. Only accelerator plans can reach a binary downcall.
4. Every real result is inside the bounded Model set; gross scalar-style errors remain rejected.
5. Cache/workload/candidate/compatibility/route/codec versions are 6 and reject v5/cross-profile
   reuse. Lifecycle, transfer, Runtime/Trace absence, and existing exact routes are unchanged.
6. Focused native/backend/conformance/public Engine/Javadoc/docs checks and independent Class C
   review pass. No blocked successor or Model 0026 is promoted.

## Validation

Worker validation:

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test :backends:metal:javadoc
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
git diff --check
```

Inspect per-operation probe summaries, exact exports/schema, generated Metal Javadocs, and public
materialized raw bits. The integration owner runs combined backend, architecture/documentation, and
full-build checks once after both parallel branches land.

## Documentation and review impact

The Metal worker updates Metal Javadocs/guide/native README only; the integration owner updates
shared status/explanations. Independent Class C review covers the oracle, gross exclusions, strict
fail-closed path, all four capability rows, schema/codec evolution, lifecycle, real Engine evidence,
and proof that blocked successors remain unauthorized.

## Result


Empty until execution.
