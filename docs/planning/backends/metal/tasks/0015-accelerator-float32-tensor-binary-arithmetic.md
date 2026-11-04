# Task 0015: [ACCELERATOR](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) [Tensor](../../../../glossary.md#tensor) [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) Arithmetic

## Reader context

ACCELERATOR and STRICT_IEEE below are historical profile identities, not current configuration
advice. This task restored four tensor binary operations under a Model-authorized set of
permitted represented results. Denormals-are-zero (DAZ) allowed specified subnormal inputs to
act as signed zero; flush-to-zero (FTZ) allowed specified subnormal results to become signed
zero. Neither meant an arbitrary error tolerance.

Shape and layout still described exact broadcasting, while actual buffer values determined
which permitted result applied. Metal checked profile policy during preparation; Runtime did
not select a profile during execution. A broadcast tensor was not the embedded scalar attribute
of task 0011.
The [current profile-free numerical implementation](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
is the present reading path, not a reinterpretation of these old profile matrices.

## Status

Complete

Readiness verification: [Metal](../../../../glossary.md#term-metal) 0014, [Model](../../../../glossary.md#term-model) 0027, Config 0006, and [Engine](../../../../glossary.md#term-engine) 0018 are Complete. Engine
0018 landed at implementation `ce7a7dfa` plus documentation/[Javadoc](../../../../glossary.md#term-javadoc) remediation `07a01b9c` with
required [validation](../../../../glossary.md#term-validation) and independent Class C approval. Its integrated version-five Metal workload,
candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) already include the [graph](../../../../glossary.md#graph) profile.
Metal 0015 and [CPU](../../../../glossary.md#term-cpu) 0017 have disjoint [backend](../../../../glossary.md#backend) semantics and write scopes after that stable spine;
shared status and explanatory documents are reserved to the integration [owner](../../../../glossary.md#term-owner).

## Change class

Class C — this restores profile-qualified public capability through a native [MPSGraph](../../../../glossary.md#term-mpsgraph) [schema](../../../../glossary.md#term-schema),
whole-[partition](../../../../glossary.md#partition) [lowering](../../../../glossary.md#lowering), persistent executable [lifecycle](../../../../glossary.md#lifecycle), [backend](../../../../glossary.md#backend) [cache](../../../../glossary.md#term-cache) compatibility, and real
public [Engine](../../../../glossary.md#term-engine) execution. Fresh [operation](../../../../glossary.md#operation)-by-operation [device](../../../../glossary.md#term-device) evidence and independent review are
mandatory.

## Goal

Under `NumericalProfile.ACCELERATOR` only, restore [tensor](../../../../glossary.md#tensor) [`FLOAT32`](../../../../glossary.md#term-float32) `ADD`, `SUB`, `MUL`, and `DIV`
for the measured [MPSGraph](../../../../glossary.md#term-mpsgraph) [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) result envelope permitted by the [Model](../../../../glossary.md#term-model) contract. Keep the strict
[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [matrix](../../../../glossary.md#term-matrix-meanings) [fail-closed](../../../../glossary.md#term-fail-closed), preserve existing exact NEG/affine/[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) support under
`STRICT_IEEE`, and reject every unproved [scalar](../../../../glossary.md#term-scalar), [unary](../../../../glossary.md#term-unary), reduction, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), comparison, [extrema](../../../../glossary.md#saturation-pole-and-extremum),
[scan](../../../../glossary.md#term-cumulative-scan), and other successor under [`ACCELERATOR`](../../../../glossary.md#term-accelerator).

## Exact capability and [topology](../../../../glossary.md#term-topology)

- `STRICT_IEEE` remains unchanged: exact existing NEG, five affine transforms, and [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) are
  supported; all tensor binary arithmetic remains false and forced preparation rejects before a
  native resource is created.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>
- [`ACCELERATOR`](../../../../glossary.md#term-accelerator) supports only `BinaryArithmeticKind.ADD`, `SUB`, `MUL`, and `DIV` with
  `NoOperationAttrs.INSTANCE`; existing NEG, affine, and `CONTIGUOUS` rows remain unsupported.
- Each [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [occurrence](../../../../glossary.md#term-occurrence) has two [`FLOAT32`](../../../../glossary.md#term-float32) inputs and one `FLOAT32` output; fully static positive
  rank <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>`1..16`; resolved [canonical](../../../../glossary.md#term-canonical-form) dense-contiguous, zero-offset, non-[view](../../../../glossary.md#term-view-meanings) [descriptors](../../../../glossary.md#term-descriptor); equal
  `requiresGrad`; checked element/byte geometry; and exact right-aligned broadcasting. Aligned
  extents are equal or one, missing leading axes are implicit ones, and output extents are the
  aligned maxima. Operand order is immutable for <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [broadcasting glossary](../../../../glossary.md#broadcasting); [axes glossary](../../../../glossary.md#term-axis); [immutable glossary](../../../../glossary.md#term-immutable))</span>`SUB` and `DIV`.
- ACCELERATOR partitions may compose only the four binary kinds. Every input is canonical and every
  output is canonical; an affine view must be canonicalized by another eligible backend before
  transfer. A binary output may feed another binary or publication. Fan-out, repeated operands,
  multiple feeds/targets, splat tensor feeds, and published intermediates retain stable order. No
  implicit canonicalization, baseline-node admission, or cross-node algebraic fusion.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [publication glossary](../../../../glossary.md#term-publication); [Fan-out glossary](../../../../glossary.md#term-fan-out); [targets glossary](../../../../glossary.md#term-target); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [node glossary](../../../../glossary.md#node); [admission glossary](../../../../glossary.md#term-admission); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- This is forward Metal execution only. Compiler-generated backward nodes may be CPU-owned in a
  mixed engine only when independently eligible; no Metal-only backward/training claim is added.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [engine glossary](../../../../glossary.md#term-engine))</span>

## Bounded numerical contract

For each [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [operation](../../../../glossary.md#operation), every produced [lane](../../../../glossary.md#term-lane) must belong to the sole [Model](../../../../glossary.md#term-model) [`ACCELERATOR`](../../../../glossary.md#term-accelerator) result
set: each declared [subnormal](../../../../glossary.md#term-subnormal) input may independently become same-[signed zero](../../../../glossary.md#term-signed-zero) before that operation;
ordinary [`FLOAT32`](../../../../glossary.md#term-float32) arithmetic then rounds once; a [finite](../../../../glossary.md#term-finite) subnormal result may become zero with
either sign. Exact-zero `ADD`/`SUB` may use either sign. `MUL`/`DIV` otherwise retain the sign
required by the selected operands. Current [NaN](../../../../glossary.md#term-nan) freedoms remain operation-owned.
No [tolerance](../../../../glossary.md#term-tolerance), [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) budget, [host](../../../../glossary.md#term-host) rewrite, value-dependent fallback, reciprocal substitution, term
loss, reduced [precision](../../../../glossary.md#term-precision), cross-[node](../../../../glossary.md#node) contraction, or special-case [dispatch](../../../../glossary.md#term-dispatch) is permitted.
`maxFinite / maxFinite -> 0` and `+infinity / maxFinite -> NaN` remain invalid, as do ordinary
finite mismatches outside the enumerated [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) set. [Scalar](../../../../glossary.md#term-scalar) arithmetic remains blocked, including
the gross failures recorded by [Metal](../../../../glossary.md#term-metal) 0011.

## Mandatory disposable real-[device](../../../../glossary.md#term-device) probe

Before production edits, build a disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary)/[MPSGraph](../../../../glossary.md#term-mpsgraph) probe at
`/tmp/synaptik-metal-0015-binary-probe.m`; remove it before completion. [Run](../../../../glossary.md#term-run) each of `ADD`, `SUB`,
`MUL`, and `DIV` as a separately reported gate at optimization levels `0` and `1`, reduced-
[precision](../../../../glossary.md#term-precision) fast math disabled, three independently compiled executables per [operation](../../../../glossary.md#operation)/level/[Shape](../../../../glossary.md#shape)
case, and at least eight executions per executable. Include equal-Shape, row broadcast, column
broadcast, [scalar](../../../../glossary.md#term-scalar)-shaped [tensor](../../../../glossary.md#tensor) broadcast, repeated operand, [fan-out](../../../../glossary.md#term-fan-out), and ordered `SUB`/`DIV`
cases. Bind direct [target](../../../../glossary.md#term-target) [buffers](../../../../glossary.md#term-buffer), permute feed/target order, preserve inputs, surround every target
with canaries, and repeat in two independently opened contexts.
Use a fixed raw-bit corpus covering both zeros; positive/negative minimum, ordinary, and maximum
[subnormals](../../../../glossary.md#term-subnormal); minimum normals and adjacent values; `0.5`, `1`, `2`; opposite ordinary values;
maximum [finite](../../../../glossary.md#term-finite) values; both [infinities](../../../../glossary.md#nan-infinity-and-signed-zero); and quiet/signaling NaNs of both signs with multiple
payloads. Include directed [FTZ](../../../../glossary.md#term-ftz) [producers](../../../../glossary.md#term-producer) (`minimum-normal * 0.5`, signed variants, and
`minimum-normal / 2`), [DAZ](../../../../glossary.md#term-daz)-sensitive pairs, exact [cancellation](../../../../glossary.md#term-cancellation), division by zeros, and the explicit
gross-error controls above.
The [oracle](../../../../glossary.md#term-test-oracle) must enumerate the finite allowed bit set from all independent DAZ choices, one
operation, and optional FTZ; apply the exact zero-sign rules; and use current family rules for [NaN](../../../../glossary.md#term-nan)
classification/payload freedom. Record totals and mismatches separately by operation, optimization
level, Shape case, and semantic bucket. Require every output to be in the enumerated set and every
control, Shape, target order, input-preservation, [canary](../../../../glossary.md#term-canary), repetition, and context-independence gate
to pass. Any out-of-set result blocks 0015 with no production/capability/[schema](../../../../glossary.md#term-schema) change; do not
shrink the corpus or reinterpret the contract.

## Native [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), and [lifecycle](../../../../glossary.md#lifecycle)

- Keep native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version `4`, statuses, create signature, and exactly thirteen exports. Advance the
  fixed 160-byte MPSGraph node schema <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [node glossary](../../../../glossary.md#node))</span>`4 -> 5`; retain wires `NEG=1`, affine/[contiguous](../../../../glossary.md#term-contiguous) `6..11`, and
  restore `ADD=2`, `SUB=3`, `MUL=4`, `DIV=5` with attribute kind `NONE`, two ordered input indices,
  zero attribute payload/reserved cells, canonical input states, exact broadcast output, and
  canonical output state.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Java capability and preparation are the profile authority. Java native-create preflight takes the
  plan profile and enforces the closed matrix: strict permits only retained baseline wires and
  accelerator permits only wires <span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [accelerator glossary](../../../../glossary.md#term-accelerator))</span>`2..5`; every mismatch fails before downcall. The private native
  v5 schema validates topology/Shape/state/wires, not Model policy; do not change the ABI merely to
  duplicate the already authenticated cold profile.<span data-glossary-navigation="true"> ([downcall glossary](../../../../glossary.md#term-downcall); [topology glossary](../../../../glossary.md#term-topology); [Shape glossary](../../../../glossary.md#shape); [Model glossary](../../../../glossary.md#term-model))</span>
- Advance workload-signature, exact-policy, candidate, compatibility, route-policy, and codec
  versions <span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>`5 -> 6`. Include profile, node schema 5, ordered binary kinds/edges, descriptors,
  states, targets, geometry, ABI, and splats in workload identity. Version-five, cross-profile,
  malformed, corrupt, unknown-wire, and stale-session decisions reject or miss safely.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [descriptors glossary](../../../../glossary.md#term-descriptor); [targets glossary](../../../../glossary.md#term-target))</span>
- Preserve strict-only custom singleton NEG, MPSGraph whole-partition selection, transactional
  finalization, context leases, rollback/suppressed failures, per-run buffers/address workspace,
  direct assigned output publication, synchronous submission, reuse, concurrency, close rejection,
  and canonical-only cross-owner transfer. No route/profile lookup enters Runtime.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [partition glossary](../../../../glossary.md#partition); [leases glossary](../../../../glossary.md#term-lease); [rollback glossary](../../../../glossary.md#term-rollback); [run glossary](../../../../glossary.md#term-run); [buffers glossary](../../../../glossary.md#term-buffer); [publication glossary](../../../../glossary.md#term-publication); [synchronous glossary](../../../../glossary.md#term-synchronous); [concurrency glossary](../../../../glossary.md#term-concurrency); [owner glossary](../../../../glossary.md#term-owner); [lookup glossary](../../../../glossary.md#term-lookup); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Non-goals

No [scalar](../../../../glossary.md#term-scalar) arithmetic; [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `POW`, `MIN`, or `MAX`; comparison; [unary](../../../../glossary.md#term-unary)/transcendental; SUM/[MEAN](../../../../glossary.md#term-mean) or
other reduction; [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/convolution; [cumulative scan](../../../../glossary.md#cumulative-scan); [BOOL](../../../../glossary.md#data-type--datatype)/WHERE; [FLOAT16](../../../../glossary.md#term-float16)/[BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT64](../../../../glossary.md#term-float64); custom
binary [kernel](../../../../glossary.md#kernel); transfer/ingress [widening](../../../../glossary.md#term-widening); [alias](../../../../glossary.md#aliasing-and-arena) promise; backward/training; packaging/discovery; or
performance claim. [Metal](../../../../glossary.md#term-metal) 0006–0007 and 0009–0013 stay Blocked within 0015; this task does not
authorize a successor for any of them and preserves [Model](../../../../glossary.md#term-model) 0026 as Draft.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — profile
  semantics remain Model-owned and profile selection remains cold.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  — sole DAZ/FTZ, zero-sign, and gross-error authority.<span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span> — Metal
  owns truthful occurrence capability, MPSGraph lowering, native integration, and resources.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [lowering glossary](../../../../glossary.md#lowering))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span> —
  route/resources are fixed before direct hot execution.
If the probe, bounded oracle, or profile guard cannot be implemented without a shared contract,
semantic rewrite, fallback, or ABI signature change, stop and report the blocker.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [guard glossary](../../../../glossary.md#term-guard); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [signature glossary](../../../../glossary.md#term-signature))</span>

## Dependencies and integration

- Depends on: [Metal](../../../../glossary.md#term-metal) 0014; [Model](../../../../glossary.md#term-model) 0027; Config 0006; [Engine](../../../../glossary.md#term-engine) 0018
- Conflicts with: every Metal capability/preparation/native schema/candidate/codec/materialization
  scope and public Metal Engine integration<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: `numerical-profile-backends`
- Common base revision: `07a01b9c13c2ceea0922b9af82da3e6e08375306`
- Integration order: Any relative to CPU 0017; both executable commits precede shared-document
  reconciliation<span data-glossary-navigation="true"> ([commits glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Integration [validation](../../../../glossary.md#term-validation): numerical-profile [backend](../../../../glossary.md#backend) integration checkpoint
- Shared-document integration [owner](../../../../glossary.md#term-owner): Main planner
The Metal worker owns Metal/native production/tests, `EngineExplicitCompositionMetalIntegrationTest`,
this brief, Metal package Javadocs, `docs/backend-guide/metal-backend.md`, and the native README. It
must not edit roadmap/master plans, root/scoped architecture, shared [API](../../../../glossary.md#term-application-programming-interface-api)/user/glossary/capability/
preparer documents, [CPU](../../../../glossary.md#term-cpu) paths, or `EngineNumericalProfileIntegrationTest`.

## Exact files and symbols

- `MetalCapabilityProvider.java`, `MetalNegPartitionPreparer.java`, `MetalNegPreparationPlan.java`
  — profile-qualified capability, topology/state validation, ordered binary lowering, retained plan.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- `MetalMpsGraphProgram.java`, `MetalNativeApi.java`, `MetalDeviceContext.java` — schema v5, two-input
  Java preflight, profile guard before downcall, and stable ABI-v4 marshalling.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [preflight glossary](../../../../glossary.md#term-preflight); [guard glossary](../../../../glossary.md#term-guard); [downcall glossary](../../../../glossary.md#term-downcall); [marshalling glossary](../../../../glossary.md#term-marshalling))</span>
- `MetalNegRouteCandidateGenerator.java`, `MetalNegTuningBatch.java`,
  `MetalNegTuningCodec.java`, `MetalNegTuningDecision.java` — version-six [identity](../../../../glossary.md#term-identity-meanings)/compatibility.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` and
  `native/metal-macos-arm64/README.md` — v5 [validation](../../../../glossary.md#term-validation)/[lowering](../../../../glossary.md#lowering) with [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4/export stability.
- `MetalCapabilityProviderTest.java`, `MetalMpsGraphAffineSchemaTest.java`,
  `MetalMpsGraphRawAbiNativeTest.java`, `MetalNegPreparedExecutionTest.java`,
  `MetalNegRouteCandidateGeneratorTest.java`, and new `MetalMpsGraphBinaryNativeTest.java` — positive
  and malformed schema/native/profile/lifecycle/codec coverage.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [codec glossary](../../../../glossary.md#term-codec))</span>
- `testing/backend-conformance/.../MetalNegCapabilityPartitionConformanceTest.java` and
  `testing/integration-tests/.../EngineExplicitCompositionMetalIntegrationTest.java` — public
  profile matrix, maximal closure, real accelerator execution, and unchanged strict rejection.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [accelerator glossary](../../../../glossary.md#term-accelerator))</span>
- `backends/metal/.../package-info.java`, affected type Javadocs, and
  `docs/backend-guide/metal-backend.md` — exact current [scope](../../../../glossary.md#term-scope) and exclusions.

## Tests and public [smoke](../../../../glossary.md#term-smoke-test)

Tests must cover each [operation](../../../../glossary.md#operation) separately plus [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)-only chains, all broadcast directions,
operand order, repeated operands, [fan-out](../../../../glossary.md#term-fan-out), splats, direct intermediate/final [targets](../../../../glossary.md#term-target),
[canonical](../../../../glossary.md#term-canonical-form)-state transitions, and malformed attrs/types/ranks/[layouts](../../../../glossary.md#layout)/broadcasts/second-input
indices. The profile [matrix](../../../../glossary.md#term-matrix-meanings) must reject all baseline [nodes](../../../../glossary.md#node) under [accelerator](../../../../glossary.md#term-accelerator) and all binary nodes
under strict. Raw [ABI](../../../../glossary.md#term-application-binary-interface-abi) tests must reject [schema](../../../../glossary.md#term-schema) 4, unknown wires, profile-incompatible [preflight](../../../../glossary.md#term-preflight),
bad states/[shapes](../../../../glossary.md#shape)/sentinels/padding, and malformed ordering before resource creation. [Codec](../../../../glossary.md#term-codec) tests
must prove deterministic v6 encoding and v5/cross-profile/corrupt/stale rejection.

The real [CPU](../../../../glossary.md#term-cpu)-free public [Engine](../../../../glossary.md#term-engine) smoke selects `ACCELERATOR`, executes all four operations with
ordinary, [DAZ](../../../../glossary.md#term-daz), [FTZ](../../../../glossary.md#term-ftz), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), [infinity](../../../../glossary.md#term-infinity), and broadcast cases, materializes direct publications,
reuses one session, opens an independent session, and verifies close/rejection behavior. A separate
strict [Metal](../../../../glossary.md#term-metal)-only binary [compile](../../../../glossary.md#compile) must still fail; with CPU registered it may be CPU-owned. No test
may assert only that execution does not throw.

## Acceptance criteria

1. The disposable probe passes every per-operation allowed-set and control gate before production
   edits; its source/binary is absent from the final tree.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
2. Strict binary stays false/rejected. Accelerator adds exactly four tensor binary kinds; strict
   retains the exact baseline while accelerator unary/layout/scalar and all blocked families stay
   false.<span data-glossary-navigation="true"> ([Accelerator glossary](../../../../glossary.md#term-accelerator); [tensor glossary](../../../../glossary.md#tensor); [unary glossary](../../../../glossary.md#term-unary); [layout glossary](../../../../glossary.md#layout))</span>
3. Java/native schema v5 enforces ordered two-input canonical topology and exact broadcast Shapes;
   ABI v4 retains exactly thirteen exports. Only accelerator plans can reach a binary downcall.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [canonical glossary](../../../../glossary.md#term-canonical-form); [topology glossary](../../../../glossary.md#term-topology); [Shapes glossary](../../../../glossary.md#shape); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [downcall glossary](../../../../glossary.md#term-downcall))</span>
4. Every real result is inside the bounded [Model](../../../../glossary.md#term-model) set; gross [scalar](../../../../glossary.md#term-scalar)-style errors remain rejected.
5. Cache/workload/candidate/compatibility/route/codec versions are 6 and reject v5/cross-profile
   reuse. Lifecycle, transfer, Runtime/Trace absence, and existing exact routes are unchanged.<span data-glossary-navigation="true"> ([Cache glossary](../../../../glossary.md#term-cache); [codec glossary](../../../../glossary.md#term-codec); [Lifecycle glossary](../../../../glossary.md#lifecycle); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace))</span>
6. Focused native/backend/conformance/public Engine/Javadoc/docs checks and independent Class C
   review pass. No blocked successor or Model 0026 is promoted.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test :backends:metal:javadoc
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
git diff --check
```

Inspect per-[operation](../../../../glossary.md#operation) probe summaries, exact exports/[schema](../../../../glossary.md#term-schema), generated [Metal](../../../../glossary.md#term-metal) Javadocs, and public
materialized [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity). The integration [owner](../../../../glossary.md#term-owner) runs combined [backend](../../../../glossary.md#backend), architecture/documentation, and
full-build checks once after both parallel branches land.

## Documentation and review impact

The [Metal](../../../../glossary.md#term-metal) worker updates Metal Javadocs/guide/native README only; the integration [owner](../../../../glossary.md#term-owner) updates
shared status/explanations. Independent Class C review covers the [oracle](../../../../glossary.md#term-test-oracle), gross exclusions, strict
[fail-closed](../../../../glossary.md#term-fail-closed) path, all four capability rows, [schema](../../../../glossary.md#term-schema)/[codec](../../../../glossary.md#term-codec) evolution, [lifecycle](../../../../glossary.md#lifecycle), real [Engine](../../../../glossary.md#term-engine) evidence,
and proof that blocked successors remain unauthorized.

## Result

Completed on Apple M3 Max. Before production edits, the disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary)/[MPSGraph](../../../../glossary.md#term-mpsgraph) probe ran
`ADD`, `SUB`, `MUL`, and `DIV` as separate gates at optimization levels 0 and 1 with
reduced-[precision](../../../../glossary.md#term-precision) fast math disabled. Each [operation](../../../../glossary.md#operation)/level/[Shape](../../../../glossary.md#shape)/context combination used three
independently compiled executables and eight repetitions across two fresh contexts. Equal, row,
column, [scalar](../../../../glossary.md#term-scalar)-[tensor](../../../../glossary.md#tensor), repeated-operand, [fan-out](../../../../glossary.md#term-fan-out), ordered, direct-[target](../../../../glossary.md#term-target), target/feed [permutation](../../../../glossary.md#permutation),
input-preservation, and [canary](../../../../glossary.md#term-canary) controls all passed. Every operation reported zero
bounded-[oracle](../../../../glossary.md#term-test-oracle) mismatches and zero control failures. The `/tmp` probe source and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) were
removed.

[Metal](../../../../glossary.md#term-metal) now advertises only tensor [`FLOAT32`](../../../../glossary.md#term-float32) `ADD`, `SUB`, `MUL`, and `DIV` under [`ACCELERATOR`](../../../../glossary.md#term-accelerator);
`STRICT_IEEE` retains the unchanged NEG/affine/[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) domain and rejects binary arithmetic.
Whole-[partition](../../../../glossary.md#partition) [lowering](../../../../glossary.md#lowering) preserves ordered operands, exact [broadcasting](../../../../glossary.md#broadcasting), [canonical](../../../../glossary.md#term-canonical-form) value states,
fan-out, repeated operands, splat tensor feeds, publications, and binary chains. Java
profile-qualified [preflight](../../../../glossary.md#term-preflight) rejects strict/binary and accelerator/baseline programs before the
native call. Native [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) 5 restores wires `2..5`, validates ordered two-input [topology](../../../../glossary.md#term-topology) and
exact broadcast Shapes, lowers to the matching MPSGraph selectors, and explicitly disables
reduced-precision fast math where available. [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 4, all statuses, the create [signature](../../../../glossary.md#term-signature),
and exactly thirteen exports remain unchanged.

Workload-signature, exact-policy, candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) are
version 6. Workload [identity](../../../../glossary.md#term-identity-meanings) includes profile, schema, ordered topology, [descriptors](../../../../glossary.md#term-descriptor), value
states, targets, checked geometry, ABI, and splats. Version-five codec data, cross-profile
decisions, malformed/corrupt bytes, changed workloads, foreign sessions, and unknown candidates
[fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

[Backend](../../../../glossary.md#backend)-native coverage executes each binary operation separately against an independent exact
raw-bit [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz)/[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)/[NaN](../../../../glossary.md#term-nan)-classification oracle with row, column, and scalar-tensor
broadcasts, operand reversal, chains, fan-out, repeated operands, direct intermediate/final
targets, repeated runs, and unchanged inputs. Public [CPU](../../../../glossary.md#term-cpu)-free [Engine](../../../../glossary.md#term-engine) coverage selects
`ACCELERATOR`, runs all four operations, includes a published-and-consumed chain plus reversed
`SUB`/`DIV`, reuses one session, opens an independent session, materializes every [publication](../../../../glossary.md#term-publication), and
proves close rejection. Existing strict Metal-only binary rejection and explicit CPU [ownership](../../../../glossary.md#term-ownership-meanings)
remain covered.

[Validation](../../../../glossary.md#term-validation) passed:

- `./native/metal-macos-arm64/build.sh`
- `nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib` (exactly thirteen
  expected exports)
- `./gradlew :backends:metal:test :backends:metal:javadoc` with the real [dylib](../../../../glossary.md#term-dylib) configured
- `./gradlew :testing:backend-conformance:test --tests '*Metal*'`
- real-[device](../../../../glossary.md#term-device) `EngineExplicitCompositionMetalIntegrationTest`
- `git diff --check`
- `./gradlew :backends:metal:test :backends:metal:javadoc` passed 84 tests with zero
  failures/errors/skips against the real dylib; focused Metal conformance passed four tests, and
  the real Metal Engine integration passed seven tests.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- The serial combined checkpoint passed the native arm64 build and exact thirteen-export audit,
  nine architecture tests, the full 87-task build, Runtime/Trace profile-free inspection, and
  baseline-aware documentation, Javadoc, status, and diff checks.<span data-glossary-navigation="true"> ([arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Implementation [commit](../../../../glossary.md#term-commit) `42c4cfbf` plus evidence-wording remediation `fb102a46` received final
  independent Class C `APPROVE` with zero findings. No limitation or unresolved issue remains.

Status: Complete
