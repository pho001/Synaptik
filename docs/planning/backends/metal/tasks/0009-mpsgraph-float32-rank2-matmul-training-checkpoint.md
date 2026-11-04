# Task 0009: [MPSGraph](../../../../glossary.md#term-mpsgraph) [FLOAT32](../../../../glossary.md#term-float32) [Rank-Two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) Training Checkpoint

## Reader context

“Training checkpoint” names the intended proof, not delivered training support. The mandatory
matrix-multiplication probe failed before the planned implementation: one contraction term
multiplied by one lost a nonzero subnormal, so rearranging a sum could not explain the result.

Compiler would have constructed derivative graphs from an explicit caller-supplied cotangent
seed, meaning the output sensitivity to propagate to each operand. Metal would have executed
those ordinary graphs. That is narrower than an implicit scalar loss or a training session.
The proposed local transpose rule authenticated a producer edge and logical operand order,
not arbitrary external view storage.
The [current descriptor admission and matrix-multiplication boundary](../../../../backend-guide/metal-backend.md#capability-whole-partition-analysis-and-route-selection)
provides present navigation without reopening this blocked qualification.

## Status

Blocked

The mandatory real probe is complete and [fail-closed](../../../../glossary.md#term-fail-closed). No production code, [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture), tests, or
probe [artifacts](../../../../glossary.md#term-artifact) were changed or committed.

## Blocking evidence

On Apple M3 Max, [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [FLOAT32](../../../../glossary.md#term-float32) [MPSGraph](../../../../glossary.md#term-mpsgraph) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) was tested at optimization levels `0` and `1`
with three independently compiled executables per level, eight runs per case, caller-supplied
[canary](../../../../glossary.md#term-canary)-filled direct [targets](../../../../glossary.md#term-target), exact [Shapes](../../../../glossary.md#shape), and direct/left-[transpose](../../../../glossary.md#term-transpose)/right-transpose
compositions. Selector availability, direct [binding](../../../../glossary.md#term-binding), complete target writes, Shape reporting,
permutations, repetition, independence, and both transpose orientations passed.

For `K=1`, multiplication by exact `1.0f` flushed all four tested [subnormal](../../../../glossary.md#term-subnormal) inputs:

- minimum positive/negative subnormal bits `0x00000001`/`0x80000001`; and
- ordinary positive/negative subnormal bits `0x00400000`/`0x80400000`.

The observed results were signed or unsigned zero. Minimum positive/negative normals
`0x00800000`/`0x80800000` were preserved. Strict [host](../../../../glossary.md#term-host) multiplication, addition from `+0.0f`, and
`fmaf` all preserve the minimum subnormal. Because `K=1` has no alternative association and every
[Model](../../../../glossary.md#term-model)-permitted multiply/add or [fused multiply-add](../../../../glossary.md#term-fused-multiply-add-fma) realization preserves that nonzero result, no
permitted FLOAT32 association/[FMA](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor) outcome is zero. This is an exact [precision](../../../../glossary.md#term-precision) failure, not
[tolerance](../../../../glossary.md#term-tolerance) noise, target [publication](../../../../glossary.md#term-publication) error, or a selector/Shape/[topology](../../../../glossary.md#term-topology) failure.

[Signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) behavior was investigated and is not a blocker: [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) is permitted by the
[accumulator](../../../../glossary.md#accumulator-type) [identity](../../../../glossary.md#term-identity-meanings)/FMA policy. The independent Class C verdict is `APPROVE-BLOCKER`: the probe
supports marking 0009 Blocked with no production change. Unblocking requires an exact replacement
route, such as separately authorized custom [kernels](../../../../glossary.md#kernel), or an explicit Model numerical-contract
change; neither is authorized here.

## Change class

Class C. This extends [Metal](../../../../glossary.md#term-metal) [occurrence](../../../../glossary.md#term-occurrence) capability, private native [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema), whole-[partition](../../../../glossary.md#partition)
[topology](../../../../glossary.md#term-topology), prepared compatibility, and real [Engine](../../../../glossary.md#term-engine) forward/backward execution without changing
[Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) semantics, public [API](../../../../glossary.md#term-application-programming-interface-api), shared contracts, dependencies, or cross-[owner](../../../../glossary.md#term-owner) transfer.

## Goal

Add the exact positive [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), same-type [`FLOAT32`](../../../../glossary.md#term-float32) subset of `MatmulKind.MATMUL` to the [ABI](../../../../glossary.md#term-application-binary-interface-abi)-v4
whole-[partition](../../../../glossary.md#partition) [MPSGraph](../../../../glossary.md#term-mpsgraph) route. Prove [Metal](../../../../glossary.md#term-metal)-only forward execution and [Compiler](../../../../glossary.md#term-compiler)-generated,
explicitly seeded [gradients](../../../../glossary.md#term-gradient) for both operands without claiming [scalar](../../../../glossary.md#term-scalar)-[loss](../../../../glossary.md#loss) training,
[`TrainingSession`](../../../../glossary.md#training-session--trainingsession), [optimizer](../../../../glossary.md#term-optimizer) execution, [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) unbroadcasting, or blocked [unary](../../../../glossary.md#term-unary)/reduction semantics.

## Exact capability domain

An [occurrence](../../../../glossary.md#term-occurrence) is eligible only when all of the following hold:

1. The [operation](../../../../glossary.md#operation) is exactly `MatmulKind.MATMUL` with `NoOperationAttrs.INSTANCE`, two ordered
   inputs, and one output.
2. Inputs/output are same-type [`FLOAT32`](../../../../glossary.md#term-float32), fully static rank two, and strictly positive. Shapes are
   exactly <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>`[M,K]`, `[K,N]`, and `[M,N]`; all extents, element counts, byte sizes, and native
   conversions pass checked arithmetic.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
3. The output is [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous), zero-offset, and non-[view](../../../../glossary.md#term-view-meanings). Its `requiresGrad` is the
   logical OR of input flags, matching Model/Compiler inference; input flags need not be equal.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
4. Each input is canonical dense-contiguous, zero-offset, and non-view, or is the exact rank-two
   descriptor produced by an in-partition `PERMUTE [1,0]` from a canonical source. A noncanonical
   input must have that unique earlier producer in the same maximal Metal partition and cannot be
   a feed, caller input, constant, transfer result, or another affine form.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [Metal glossary](../../../../glossary.md#term-metal); [constant glossary](../../../../glossary.md#term-constant))</span>

The fourth rule is the terminal/view decision: task 0009 narrowly permits an existing [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
[transpose](../../../../glossary.md#term-transpose) result to feed [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) because its authenticated [MPSGraph](../../../../glossary.md#term-mpsgraph) [tensor](../../../../glossary.md#tensor) is already dense in
represented logical order. `RESHAPE`, `EXPAND`, `EXPAND_DIMS`, and `SQUEEZE` remain terminal, and
all affine results remain ordinary logical views in [graph](../../../../glossary.md#graph) [metadata](../../../../glossary.md#term-metadata). No view ingress,
[materialization](../../../../glossary.md#materialization) workspace, [descriptor](../../../../glossary.md#term-descriptor) rewrite, [aliasing](../../../../glossary.md#term-aliasing), or transfer [widening](../../../../glossary.md#term-widening) is authorized.

One maximal [partition](../../../../glossary.md#partition) may mix the current 0005 operations, 0008 terminal affine [targets](../../../../glossary.md#term-target), rank-two
`PERMUTE [1,0] -> MATMUL`, and canonical MATMUL outputs consumed by already-supported operations.
Capability and analysis [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) for [vectors](../../../../glossary.md#term-vector), [scalars](../../../../glossary.md#term-scalar), [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [dimensions](../../../../glossary.md#dimension), zero extents, dynamic
[Shapes](../../../../glossary.md#shape), other data types or promotions, wrong attrs, wrong output geometry, arbitrary view inputs,
view partition feeds, malformed [topology](../../../../glossary.md#term-topology), or unchecked geometry.

## [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), and training boundary

Model [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) contracts the final left [axis](../../../../glossary.md#term-axis) with the penultimate right axis, uses
`NoOperationAttrs.INSTANCE`, and here produces `[M,N]`. Products and accumulation are [`FLOAT32`](../../../../glossary.md#term-float32);
[reassociation](../../../../glossary.md#term-reassociation)/[FMA](../../../../glossary.md#term-fused-multiply-add-fma) are permitted, but reduced [precision](../../../../glossary.md#term-precision) is not. Compiler closes the materialized
output to [canonical](../../../../glossary.md#term-canonical-form) logical [layout](../../../../glossary.md#layout).

Compiler's [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) rules are exactly `g @ permute(right,[1,0])` and
`permute(left,[1,0]) @ g`. With a canonical `[M,N]` [cotangent](../../../../glossary.md#cotangent), both results match their operands,
so no `SUM_TO_SHAPE` is generated. Inspect and execute both [targets](../../../../glossary.md#term-target) through public seeded
`Engine.compile(...)`.

This is training-enabling, not end-to-end training. [`TrainingSession`](../../../../glossary.md#training-session--trainingsession) requires a [scalar objective](../../../../glossary.md#loss)
and ordinary model/[loss](../../../../glossary.md#loss) [graph](../../../../glossary.md#graph); broad MATMUL ranks, [scalar](../../../../glossary.md#term-scalar) ingress, losses, and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-unbroadcast
[gradients](../../../../glossary.md#term-gradient) are excluded. [Metal](../../../../glossary.md#term-metal) 0007 blocks exact `SUM`/[`MEAN`](../../../../glossary.md#term-mean)/`SUM_TO_SHAPE`, and 0006 blocks its
recorded [unary](../../../../glossary.md#term-unary) algebra. No scalar-loss, [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd), convergence, or general backward claim is allowed.

## [MPSGraph](../../../../glossary.md#term-mpsgraph) and numerical gates

Use only the [SDK](../../../../glossary.md#term-sdk) selector
`matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:`. Before production edits, [compile](../../../../glossary.md#compile)
and [run](../../../../glossary.md#term-run) a disposable real-[device](../../../../glossary.md#term-device) probe using the production-style placeholders, executable, and
caller-supplied direct output [buffers](../../../../glossary.md#term-buffer). It must prove:

- non-square `[M,K] @ [K,N]`, `M/K/N = 1` boundaries, and multiple positive `K` values;
- direct inputs and left/right `transposeTensor:permutation:name:` compositions, including the
  linear-projection <span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings))</span>`PERMUTE [1,0] -> MATMUL` orientation;
- exact reported `[M,N]` Shapes, complete direct-target writes, stable target/feed permutation,
  repeated runs, and independent executables;<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [permutation glossary](../../../../glossary.md#permutation))</span>
- raw-bit equality for cases whose current Model policy has one result under every permitted
  association/FMA choice, including exact finite integers, multiplication by one, signed-zero and
  subnormal preservation where the IEEE result is unique, and invariant infinity cases;<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [finite glossary](../../../../glossary.md#term-finite); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [subnormal glossary](../../../../glossary.md#term-subnormal); [IEEE glossary](../../../../glossary.md#term-ieee); [invariant glossary](../../../../glossary.md#term-invariant); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- membership in an independent bounded [oracle](../../../../glossary.md#term-test-oracle)'s complete set of permitted [`FLOAT32`](../../../../glossary.md#term-float32)
  association/FMA outcomes for cancellation-sensitive finite cases, plus the exact IEEE result
  class for NaN-producing cases. A generic epsilon, CPU-output comparison, relaxed type, or
  undocumented fast-math assumption is forbidden.<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [NaN glossary](../../../../glossary.md#term-nan); [epsilon glossary](../../../../glossary.md#epsilon))</span>

If selector availability, direct [binding](../../../../glossary.md#term-binding), [Shape](../../../../glossary.md#shape), [precision](../../../../glossary.md#term-precision), exceptional-value behavior, or either
[transpose](../../../../glossary.md#term-transpose) composition fails, remove the probe and mark 0009 Blocked. Do not substitute a custom
[kernel](../../../../glossary.md#kernel), [CPU](../../../../glossary.md#term-cpu) fallback, [host](../../../../glossary.md#term-host) transpose/[GEMM](../../../../glossary.md#general-matrix-multiplication--gemm), relaxed contract, or undocumented selector.

## Native [schema](../../../../glossary.md#term-schema), candidates, and [lifecycle](../../../../glossary.md#lifecycle)

Keep native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 4, statuses, and exactly thirteen exported symbols unchanged. Bump the
pointed-to fixed 160-byte [node](../../../../glossary.md#node) schema from version 2 to version 3 and add only `MATMUL=11` with the
existing no-attribute discriminant, two ordered inputs, required absent-[axis](../../../../glossary.md#term-axis) sentinel, zero count,
reserved word, and payload. Java and native [validation](../../../../glossary.md#term-validation) must agree on [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [Shapes](../../../../glossary.md#shape), [contraction](../../../../glossary.md#contraction-dimension),
[canonical](../../../../glossary.md#term-canonical-form) output, local [transpose](../../../../glossary.md#term-transpose) [topology](../../../../glossary.md#term-topology), value [identity](../../../../glossary.md#term-identity-meanings), and bounds.

Bump candidate schema, compatibility/workload schema, route-policy version, and [backend](../../../../glossary.md#backend)-local [codec](../../../../glossary.md#term-codec)
version monotonically. Workload identity includes [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) kind/attrs, ordered Shapes and [descriptors](../../../../glossary.md#term-descriptor),
local transpose [provenance](../../../../glossary.md#provenance), [gradient](../../../../glossary.md#term-gradient) flags, exact numerical policy, node schema, ABI, topology,
and live [target](../../../../glossary.md#term-target) identity. Old decisions [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). The sole safe MATMUL candidate remains the
whole-[partition](../../../../glossary.md#partition) [MPSGraph](../../../../glossary.md#term-mpsgraph) route; the custom [singleton](../../../../glossary.md#term-singleton)-NEG route is unchanged.

Analysis declares boundary [buffers](../../../../glossary.md#term-buffer)/address workspace before assignment; finalization creates one
reusable Shape-specialized executable transactionally. Cold [binding](../../../../glossary.md#term-binding) validates typed
representations once; hot execution makes one [synchronous](../../../../glossary.md#term-synchronous) submission into [run](../../../../glossary.md#term-run)-owned targets.
[Rollback](../../../../glossary.md#term-rollback), closure, repeated/concurrent runs, and two sessions retain current [ownership](../../../../glossary.md#term-ownership-meanings). No retry,
fallback, per-node call, hidden copy, late selection, async, pooling, packaging, or tuning is added.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects Metal ownership; Metal owns truthful lowering and Runtime invokes prepared work.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Metal glossary](../../../../glossary.md#term-metal); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Concrete backend modules and Metal backend](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend))</span>
  — Metal owns its native schema, MPSGraph lowering, storage, and materialization; cross-owner
  transfer remains canonical positive rank <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [materialization glossary](../../../../glossary.md#materialization); [owner glossary](../../../../glossary.md#term-owner); [canonical glossary](../../../../glossary.md#term-canonical-form); [rank glossary](../../../../glossary.md#term-rank))</span>`1..16` [`FLOAT32`](../../../../glossary.md#term-float32) only.
- [Compiler-owned automatic differentiation and compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd); [compile glossary](../../../../glossary.md#compile); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  — Compiler owns the rank-two gradient graph and maximal same-owner partitioning.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph))</span>
- [Prepare lifecycle and run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run))</span>
  — all resources and route facts are fixed before finalization and hot execution.
- [Metal backend guide](../../../../backend-guide/metal-backend.md),
  [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>, and
  [Compile API](../../../../api/compile-api.md) — current lifecycle, MATMUL, and seeded-gradient
  behavior.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>

If these contracts cannot express the local-[transpose](../../../../glossary.md#term-transpose) input rule without a shared representation,
capability-query, or transfer change, stop and mark 0009 Blocked pending architecture authorization.

## Dependencies and integration

- **Depends on:** Metal 0008; current Model rank-two MATMUL semantics; Compiler MATMUL layout and
  first-order rules; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016. Metal
  0006/0007 are explicitly not dependencies.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Model glossary](../../../../glossary.md#term-model); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Compiler glossary](../../../../glossary.md#term-compiler); [layout glossary](../../../../glossary.md#layout); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- **Conflicts with:** Metal 0006/0007 and any concurrent Metal capability, preparation, native
  ABI/schema, candidate/codec, materialization, or Engine Metal integration work.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization))</span>
- **Parallel group:** None.
- **Common base revision:** `aa42ed711a0d3120f97e4241181b557130255e97`.
- **Integration order:** disposable selector/numerical probe → schema/validation → capability and
  topology analysis → finalization/runtime → forward and seeded-backward Engine proof → docs/review.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>
- **Integration [validation](../../../../glossary.md#term-validation):** native build/export audit; focused Metal, conformance, real Engine,
  Compiler-contract, architecture, Javadoc, Markdown, and whitespace checks.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- **Shared-document integration [owner](../../../../glossary.md#term-owner):** Main planner.

## Files and symbols

- `backends/metal/src/main/.../metal/MetalCapabilityProvider.java` — exact [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [occurrence](../../../../glossary.md#term-occurrence) truth.
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalNegPartitionPreparer`, and tuning batch/codec —
  schema, topology, lowering, compatibility, and fail-closed decoding.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [codec glossary](../../../../glossary.md#term-codec); [topology glossary](../../../../glossary.md#term-topology); [lowering glossary](../../../../glossary.md#lowering); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — [schema](../../../../glossary.md#term-schema)-v3 [validation](../../../../glossary.md#term-validation) and selector.
- Focused [Metal](../../../../glossary.md#term-metal) tests, `testing/backend-conformance`, and
  `EngineExplicitCompositionMetalIntegrationTest` — fake/raw/real and public proofs.
- Metal package Javadocs, backend/native guides, glossary, task, master plan, and roadmap — exact
  current scope and limitations.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Capability and preparation admit exactly the stated rank-two FLOAT32 domain and local transpose
  topology while preserving all existing domains and transfer restrictions.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [transpose glossary](../../../../glossary.md#term-transpose); [topology glossary](../../../../glossary.md#term-topology))</span>
- The disposable probe passes every selector/direct-target/numerical gate before production edits;
  retained native tests reproduce the accepted matrix.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 still has exactly thirteen exports; [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) is v3 with `MATMUL=11`; malformed raw ABI
  records and stale candidate/codec decisions reject before resource publication.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>
- Fake and real tests cover Shape/contraction/layout/gradient-flag boundaries, wrong attrs/types,
  view-feed rejection, overflow, rollback, close, repetition, concurrency, and independent sessions.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [contraction glossary](../../../../glossary.md#contraction-dimension); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [view glossary](../../../../glossary.md#term-view-meanings); [overflow glossary](../../../../glossary.md#term-overflow); [rollback glossary](../../../../glossary.md#term-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- A [Metal](../../../../glossary.md#term-metal)-only public [Engine](../../../../glossary.md#term-engine) forward runs bare [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) and [`linear`](../../../../glossary.md#term-linear-meanings)'s
  `PERMUTE [1,0] -> MATMUL`, with direct [canonical](../../../../glossary.md#term-canonical-form) [publication](../../../../glossary.md#term-publication) and no [CPU](../../../../glossary.md#term-cpu) [owner](../../../../glossary.md#term-owner).
- A Metal-only public explicitly seeded backward run publishes correct left and right gradients,
  and inspection proves the generated graph contains only the expected PERMUTE/MATMUL formulas
  with no reduction, unary algebra, hidden copy, or CPU partition.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [graph glossary](../../../../glossary.md#graph); [unary glossary](../../../../glossary.md#term-unary); [partition glossary](../../../../glossary.md#partition))</span>
- Documentation/Javadocs state that [scalar](../../../../glossary.md#term-scalar)-[loss](../../../../glossary.md#loss) [`TrainingSession`](../../../../glossary.md#training-session--trainingsession), batch/vector MATMUL, reduction-
  requiring gradients, custom kernels, and broader transfers remain unsupported. Independent
  Class C review returns APPROVE before completion.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [vector glossary](../../../../glossary.md#term-vector); [kernels glossary](../../../../glossary.md#kernel))</span>

## [Validation](../../../../glossary.md#term-validation)

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*Matmul*' --tests '*Metal*MpsGraph*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :backends:metal:javadoc :testing:architecture-tests:test
git diff --check
```

Also verify balanced Markdown fences, final newlines, repository-local links/anchors, exactly
thirteen native exports, no production [dependency](../../../../glossary.md#term-dependency) change, no custom [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [kernel](../../../../glossary.md#kernel), no [view](../../../../glossary.md#term-view-meanings) ingress
or transfer [widening](../../../../glossary.md#term-widening), and no [`TrainingSession`](../../../../glossary.md#training-session--trainingsession)/[optimizer](../../../../glossary.md#term-optimizer) claim.

## Documentation and review impact

Update [Metal](../../../../glossary.md#term-metal) Javadocs, package documentation, [backend](../../../../glossary.md#backend)/native guides, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) glossary status, and
[planning](../../../../glossary.md#term-planning) after evidence passes. [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) semantics and architecture contracts remain
unchanged. Independent Class C review inspects capability truth, local-[view](../../../../glossary.md#term-view-meanings) [authentication](../../../../glossary.md#term-authentication),
numerics, raw [schema](../../../../glossary.md#term-schema), compatibility invalidation, [lifecycle](../../../../glossary.md#lifecycle), [Engine](../../../../glossary.md#term-engine) evidence, and exclusions.

## Result

Blocked. The mandatory Apple M3 Max probe passed selector, direct-[target](../../../../glossary.md#term-target), [Shape](../../../../glossary.md#shape), [permutation](../../../../glossary.md#permutation),
[transpose](../../../../glossary.md#term-transpose), repetition, and independent-executable gates but failed exact [FLOAT32](../../../../glossary.md#term-float32) [precision](../../../../glossary.md#term-precision):
`K=1`, multiplication by one flushed minimum and ordinary positive/negative [subnormals](../../../../glossary.md#term-subnormal) to zero
while minimum normals survived. [Host](../../../../glossary.md#term-host) strict multiply, `+0.0f` accumulation, and `fmaf` preserve
the minimum subnormal, so zero is outside every permitted association/[FMA](../../../../glossary.md#term-fused-multiply-add-fma) result. [Signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) was
not a blocker. Independent review returned `APPROVE-BLOCKER`; no production, native, test, probe,
or [commit](../../../../glossary.md#term-commit) change exists.
