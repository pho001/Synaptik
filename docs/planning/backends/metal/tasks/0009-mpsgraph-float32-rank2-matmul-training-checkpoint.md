# Task 0009: MPSGraph FLOAT32 Rank-Two MATMUL Training Checkpoint

## Status

Ready

This is the sole authorized Metal frontier. Metal 0008 is Complete; 0006 and 0007 remain Blocked.

## Change class

Class C. This extends Metal occurrence capability, private native node schema, whole-partition
topology, prepared compatibility, and real Engine forward/backward execution without changing
Model/Compiler semantics, public API, shared contracts, dependencies, or cross-owner transfer.

## Goal

Add the exact positive rank-two, same-type `FLOAT32` subset of `MatmulKind.MATMUL` to the ABI-v4
whole-partition MPSGraph route. Prove Metal-only forward execution and Compiler-generated,
explicitly seeded gradients for both operands without claiming scalar-loss training,
`TrainingSession`, optimizer execution, batch unbroadcasting, or blocked unary/reduction semantics.

## Exact capability domain

An occurrence is eligible only when all of the following hold:

1. The operation is exactly `MatmulKind.MATMUL` with `NoOperationAttrs.INSTANCE`, two ordered
   inputs, and one output.
2. Inputs/output are same-type `FLOAT32`, fully static rank two, and strictly positive. Shapes are
   exactly `[M,K]`, `[K,N]`, and `[M,N]`; all extents, element counts, byte sizes, and native
   conversions pass checked arithmetic.
3. The output is canonical dense-contiguous, zero-offset, and non-view. Its `requiresGrad` is the
   logical OR of input flags, matching Model/Compiler inference; input flags need not be equal.
4. Each input is canonical dense-contiguous, zero-offset, and non-view, or is the exact rank-two
   descriptor produced by an in-partition `PERMUTE [1,0]` from a canonical source. A noncanonical
   input must have that unique earlier producer in the same maximal Metal partition and cannot be
   a feed, caller input, constant, transfer result, or another affine form.

The fourth rule is the terminal/view decision: task 0009 narrowly permits an existing rank-two
transpose result to feed MATMUL because its authenticated MPSGraph tensor is already dense in
represented logical order. `RESHAPE`, `EXPAND`, `EXPAND_DIMS`, and `SQUEEZE` remain terminal, and
all affine results remain ordinary logical views in graph metadata. No view ingress,
materialization workspace, descriptor rewrite, aliasing, or transfer widening is authorized.

One maximal partition may mix the current 0005 operations, 0008 terminal affine targets, rank-two
`PERMUTE [1,0] -> MATMUL`, and canonical MATMUL outputs consumed by already-supported operations.
Capability and analysis fail closed for vectors, scalars, batch dimensions, zero extents, dynamic
Shapes, other data types or promotions, wrong attrs, wrong output geometry, arbitrary view inputs,
view partition feeds, malformed topology, or unchecked geometry.

## Model, Compiler, and training boundary

Model MATMUL contracts the final left axis with the penultimate right axis, uses
`NoOperationAttrs.INSTANCE`, and here produces `[M,N]`. Products and accumulation are `FLOAT32`;
reassociation/FMA are permitted, but reduced precision is not. Compiler closes the materialized
output to canonical logical layout.

Compiler's rank-two rules are exactly `g @ permute(right,[1,0])` and
`permute(left,[1,0]) @ g`. With a canonical `[M,N]` cotangent, both results match their operands,
so no `SUM_TO_SHAPE` is generated. Inspect and execute both targets through public seeded
`Engine.compile(...)`.

This is training-enabling, not end-to-end training. `TrainingSession` requires a scalar objective
and ordinary model/loss graph; broad MATMUL ranks, scalar ingress, losses, and batch-unbroadcast
gradients are excluded. Metal 0007 blocks exact `SUM`/`MEAN`/`SUM_TO_SHAPE`, and 0006 blocks its
recorded unary algebra. No scalar-loss, SGD, convergence, or general backward claim is allowed.

## MPSGraph and numerical gates

Use only the SDK selector
`matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:`. Before production edits, compile
and run a disposable real-device probe using the production-style placeholders, executable, and
caller-supplied direct output buffers. It must prove:

- non-square `[M,K] @ [K,N]`, `M/K/N = 1` boundaries, and multiple positive `K` values;
- direct inputs and left/right `transposeTensor:permutation:name:` compositions, including the
  linear-projection `PERMUTE [1,0] -> MATMUL` orientation;
- exact reported `[M,N]` Shapes, complete direct-target writes, stable target/feed permutation,
  repeated runs, and independent executables;
- raw-bit equality for cases whose current Model policy has one result under every permitted
  association/FMA choice, including exact finite integers, multiplication by one, signed-zero and
  subnormal preservation where the IEEE result is unique, and invariant infinity cases;
- membership in an independent bounded oracle's complete set of permitted `FLOAT32`
  association/FMA outcomes for cancellation-sensitive finite cases, plus the exact IEEE result
  class for NaN-producing cases. A generic epsilon, CPU-output comparison, relaxed type, or
  undocumented fast-math assumption is forbidden.

If selector availability, direct binding, Shape, precision, exceptional-value behavior, or either
transpose composition fails, remove the probe and mark 0009 Blocked. Do not substitute a custom
kernel, CPU fallback, host transpose/GEMM, relaxed contract, or undocumented selector.

## Native schema, candidates, and lifecycle

Keep native ABI version 4, statuses, and exactly thirteen exported symbols unchanged. Bump the
pointed-to fixed 160-byte node schema from version 2 to version 3 and add only `MATMUL=11` with the
existing no-attribute discriminant, two ordered inputs, required absent-axis sentinel, zero count,
reserved word, and payload. Java and native validation must agree on rank-two Shapes, contraction,
canonical output, local transpose topology, value identity, and bounds.

Bump candidate schema, compatibility/workload schema, route-policy version, and backend-local codec
version monotonically. Workload identity includes MATMUL kind/attrs, ordered Shapes and descriptors,
local transpose provenance, gradient flags, exact numerical policy, node schema, ABI, topology,
and live target identity. Old decisions fail closed. The sole safe MATMUL candidate remains the
whole-partition MPSGraph route; the custom singleton-NEG route is unchanged.

Analysis declares boundary buffers/address workspace before assignment; finalization creates one
reusable Shape-specialized executable transactionally. Cold binding validates typed
representations once; hot execution makes one synchronous submission into run-owned targets.
Rollback, closure, repeated/concurrent runs, and two sessions retain current ownership. No retry,
fallback, per-node call, hidden copy, late selection, async, pooling, packaging, or tuning is added.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects Metal ownership; Metal owns truthful lowering and Runtime invokes prepared work.
- [Concrete backend modules and Metal backend](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  — Metal owns its native schema, MPSGraph lowering, storage, and materialization; cross-owner
  transfer remains canonical positive rank `1..16` `FLOAT32` only.
- [Compiler-owned automatic differentiation and compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  — Compiler owns the rank-two gradient graph and maximal same-owner partitioning.
- [Prepare lifecycle and run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  — all resources and route facts are fixed before finalization and hot execution.
- [Metal backend guide](../../../../backend-guide/metal-backend.md),
  [Tensor API](../../../../api/tensor-api.md), and
  [Compile API](../../../../api/compile-api.md) — current lifecycle, MATMUL, and seeded-gradient
  behavior.

If these contracts cannot express the local-transpose input rule without a shared representation,
capability-query, or transfer change, stop and mark 0009 Blocked pending architecture authorization.

## Dependencies and integration

- **Depends on:** Metal 0008; current Model rank-two MATMUL semantics; Compiler MATMUL layout and
  first-order rules; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016. Metal
  0006/0007 are explicitly not dependencies.
- **Conflicts with:** Metal 0006/0007 and any concurrent Metal capability, preparation, native
  ABI/schema, candidate/codec, materialization, or Engine Metal integration work.
- **Parallel group:** None.
- **Common base revision:** `aa42ed711a0d3120f97e4241181b557130255e97`.
- **Integration order:** disposable selector/numerical probe → schema/validation → capability and
  topology analysis → finalization/runtime → forward and seeded-backward Engine proof → docs/review.
- **Integration validation:** native build/export audit; focused Metal, conformance, real Engine,
  Compiler-contract, architecture, Javadoc, Markdown, and whitespace checks.
- **Shared-document integration owner:** Main planner.

## Files and symbols

- `backends/metal/src/main/.../metal/MetalCapabilityProvider.java` — exact MATMUL occurrence truth.
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalNegPartitionPreparer`, and tuning batch/codec —
  schema, topology, lowering, compatibility, and fail-closed decoding.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — schema-v3 validation and selector.
- Focused Metal tests, `testing/backend-conformance`, and
  `EngineExplicitCompositionMetalIntegrationTest` — fake/raw/real and public proofs.
- Metal package Javadocs, backend/native guides, glossary, task, master plan, and roadmap — exact
  current scope and limitations.

## Acceptance criteria

- Capability and preparation admit exactly the stated rank-two FLOAT32 domain and local transpose
  topology while preserving all existing domains and transfer restrictions.
- The disposable probe passes every selector/direct-target/numerical gate before production edits;
  retained native tests reproduce the accepted matrix.
- ABI v4 still has exactly thirteen exports; node schema is v3 with `MATMUL=11`; malformed raw ABI
  records and stale candidate/codec decisions reject before resource publication.
- Fake and real tests cover Shape/contraction/layout/gradient-flag boundaries, wrong attrs/types,
  view-feed rejection, overflow, rollback, close, repetition, concurrency, and independent sessions.
- A Metal-only public Engine forward runs bare rank-two MATMUL and `linear`'s
  `PERMUTE [1,0] -> MATMUL`, with direct canonical publication and no CPU owner.
- A Metal-only public explicitly seeded backward run publishes correct left and right gradients,
  and inspection proves the generated graph contains only the expected PERMUTE/MATMUL formulas
  with no reduction, unary algebra, hidden copy, or CPU partition.
- Documentation/Javadocs state that scalar-loss `TrainingSession`, batch/vector MATMUL, reduction-
  requiring gradients, custom kernels, and broader transfers remain unsupported. Independent
  Class C review returns APPROVE before completion.

## Validation

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
thirteen native exports, no production dependency change, no custom MATMUL kernel, no view ingress
or transfer widening, and no `TrainingSession`/optimizer claim.

## Documentation and review impact

Update Metal Javadocs, package documentation, backend/native guides, MATMUL glossary status, and
planning after evidence passes. Tensor/Compile API semantics and architecture contracts remain
unchanged. Independent Class C review inspects capability truth, local-view authentication,
numerics, raw schema, compatibility invalidation, lifecycle, Engine evidence, and exclusions.

## Result

Empty until execution.
