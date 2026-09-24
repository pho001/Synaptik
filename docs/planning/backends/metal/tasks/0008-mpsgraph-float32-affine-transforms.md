# Task 0008: MPSGraph FLOAT32 Affine Transforms

## Status

Complete

Metal 0006 and 0007 remain independently Blocked; this task neither depended on them nor changed
their numerical contracts. Metal 0009 is the separately authorized successor.

## Change class

Class C. This extends Metal capability, the typed native node schema, backend-private physical
representation semantics for logical affine-view results, prepared-resource lifecycle, and public
Engine materialization. It changes no shared architecture, Model semantics, cross-owner transfer
contract, or public API.

## Goal

Extend the existing ABI-v4 whole-partition MPSGraph route with exactly five FLOAT32 affine
transforms:

- `ShapeTransformKind.RESHAPE` with `TargetShapeAttrs`;
- `ShapeTransformKind.EXPAND` with `TargetShapeAttrs`;
- `AxisTransformKind.PERMUTE` with `PermutationAttrs`;
- `AxisTransformKind.EXPAND_DIMS` with `AxisTransformAttrs`; and
- `AxisTransformKind.SQUEEZE` with `AxisTransformAttrs`.

Preserve the current `NEG` and binary `ADD/SUB/MUL/DIV` domain and the custom singleton-NEG route.
Produce exact logical values and Shapes while retaining the exact Model/Compiler affine-view
layout descriptors. A Metal MPSGraph target for an affine result is a backend-private dense
physical materialization in canonical logical coordinate order; it does not claim shared storage
aliasing with the source and does not rewrite the logical descriptor.

## Independence decision

This work is independent of blocked 0006 and 0007. The five operations only reindex, repeat,
insert, or remove logical coordinates and must preserve represented FLOAT32 bits; none invokes
unary algebra, reduction, accumulation, division, or a reduction-derived count. The Model operation
and typed-attribute records already exist, Compiler inference already validates their exact Shapes
and layouts. Current architecture assigns lowering, physical materialization, and MPSGraph
resource ownership to Metal.

`EXPAND` first-order backward uses `SUM_TO_SHAPE`, so blocked 0007 prevents a complete Metal-only
backward claim for this family. Task 0008 is therefore a forward capability. It must build and
inspect Compiler-generated gradient graphs as a contract gate, but it must not claim executable
Metal-only backward. The later MATMUL checkpoint is a separate successor and is not authorized by
this brief.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects Metal ownership; Metal owns truthful lowering and Runtime invokes prepared work.
- [`ARCHITECTURE.md` — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) —
  Metal owns MPSGraph lowering, storage, native integration, and Metal-specific materialization.
- [`ARCHITECTURE.md` — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  — analysis declares all resources before assignment and finalization cannot reselect a route.
- [`ARCHITECTURE.md` — Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)
  — cold binding creates typed direct references and hot execution performs no lookup or selection.
- [Metal backend guide](../../../../backend-guide/metal-backend.md),
  [Tensor API](../../../../api/tensor-api.md), and
  [Compile API](../../../../api/compile-api.md) — current executable, affine-layout, and Compiler
  contracts.

If implementation requires a shared aliasing contract, descriptor rewrite, broader CPU/Metal
transfer, custom kernel, CPU fallback, or public API change, stop and mark the task Blocked rather
than expanding scope.

## Exact capability domain

An affine occurrence is eligible only when all of the following hold:

1. It has exactly one input, one output, and the exact typed attributes paired with its operation
   kind. No generic attributes, strings, reflection, or inferred operation names are allowed.
2. Input and output use `FLOAT32`, have equal `requiresGrad` flags, are fully static, have strictly
   positive extents, and have ranks in `1..16`. Rank-zero and zero-extent cases remain unsupported.
3. The input descriptor is exactly canonical dense-contiguous, zero-offset, and non-view. This
   deliberately excludes caller or partition feeds whose logical coordinates require gathering,
   permutation, offset handling, or broadcast-stride interpretation.
4. The output descriptor exactly equals the Model/Compiler result: exact Shape, zero storage
   offset, `isView = true`, and exact derived strides. `RESHAPE` uses canonical target strides;
   `EXPAND` uses right-aligned preserved or zero strides; `PERMUTE` reorders input strides;
   `EXPAND_DIMS` inserts the exact derived singleton stride; and `SQUEEZE` removes the selected
   singleton stride.
5. Target Shapes, axes, permutations, element counts, referenced spans, dense physical element
   counts, byte extents, and all fixed-width native conversions pass checked arithmetic and schema
   bounds.

This input rule makes an affine operation a terminal leaf of the currently admitted Metal
operation domain. One maximal partition may contain any already-supported 0005 topology followed
by one or more affine leaves with stable fan-out and multiple targets. Task 0008 does not admit an
affine result as the input of another affine or existing elementwise occurrence, because that input
is a logical view rather than canonical non-view geometry. A later MATMUL task must state and prove
any view-input rule it needs; this brief does not pre-authorize it.

Capability and analysis must fail closed for `CONTIGUOUS`, slice/select/pad/tile/composition,
MATMUL, every other operation family, wrong attributes, wrong type, dynamic or zero dimensions,
rank outside `1..16`, noncanonical input, incorrect output Shape/layout, malformed topology, or
unchecked geometry.

## Model and Compiler contract readout

- `RESHAPE` preserves ordered logical elements and requires equal element counts. For the admitted
  canonical input, its exact static result is a same-offset view with canonical target strides.
- `EXPAND` right-aligns source and target. A source extent must equal the target or be one; target-
  only leading axes are implicit source singletons. Repeated axes have output stride zero.
- `PERMUTE` carries a complete output-to-input permutation whose length equals rank and whose axes
  are unique and in range. Rank-two transpose is `PERMUTE [1, 0]`, not another operation kind.
- `EXPAND_DIMS` inserts one extent-one output axis at the normalized position. `SQUEEZE` removes one
  selected input axis only when its static extent is exactly one.
- Compiler inference retains the exact affine view geometry and deliberately excludes these results
  from canonical logical-layout closure. Metal therefore must validate, not replace, that metadata.
- Compiler gradient rules use inverse reshape/permutation/rank edits and use `SUM_TO_SHAPE` for
  `EXPAND`. Gradient graph construction and descriptor inspection are required; execution is not.

## Native schema and MPSGraph lowering

Keep native ABI version 4 and its exact thirteen exported symbols unchanged. Evolve the pointed-to
node schema to a documented version-2 typed discriminated record or union. It must represent only:

- the existing no-attribute unary/binary variants;
- target-Shape attributes with bounded rank and dimensions;
- a complete bounded permutation; and
- one bounded normalized axis.

Every unused field must have a required zero/sentinel value. Java and native validation must agree
on operation/attribute pairing, rank, target dimensions, axes, permutation completeness, ordered
inputs, one output, topological value identity, and schema size. No variable object graph, generic
integer payload, raw string, serialized map, hidden pointer, or dimensions repurposed as attrs may
cross the ABI.

Use the current SDK typed selectors:

- `reshapeTensor:withShape:name:` for `RESHAPE`;
- `broadcastTensor:toShape:name:` for `EXPAND`;
- `transposeTensor:permutation:name:` for `PERMUTE`;
- `expandDimsOfTensor:axis:name:` for `EXPAND_DIMS`; and
- `squeezeTensor:axis:name:` for `SQUEEZE`.

Before production edits, compile and run a disposable real probe against those selectors and the
exact direct-output-buffer binding used by production. If a selector is unavailable, reports a
wrong Shape, does not write the supplied target directly, or fails the exact bit/shape matrix below,
revert the probe and mark the task Blocked. Do not substitute an undocumented selector, custom
kernel, host rewrite, or relaxed gate.

Bump the Java/native node-schema version, candidate schema, workload fingerprint schema, and route-
policy version monotonically. Compatibility identity includes operation kind, typed attrs, exact
input/output descriptor and logical layout, dense represented-order geometry, ordered topology,
numerical policy, native node schema, ABI version, and live Metal target identity. Old decisions
must fail closed.

## Physical materialization and transfer boundary

An affine graph target receives a Metal buffer sized to the full positive logical element count,
not merely the affine descriptor's referenced element span. MPSGraph writes that target in
canonical logical coordinate order. This is a backend-private physical representation choice:
logical zero strides, permuted strides, and `isView` remain unchanged in graph metadata, and no two
logical graph values share a Runtime representation or slot.

Extend Metal publication materialization only enough to turn this authenticated dense represented
order into detached canonical host bytes for an exact admitted affine publication. Validate the
prepared route identity, descriptor, context, openness, current-thread access, full logical byte
extent, and maximum-byte limit before downloading. Raw FLOAT32 bits must be retained.

Do not widen CPU-to-Metal or Metal-to-CPU transfer. Cross-owner transfer remains restricted to
positive rank-`1..16` static canonical-contiguous non-view FLOAT32. An affine logical view at a
CPU/Metal boundary must still fail during transfer preflight before backend analysis. Do not add
canonical-byte transfer, descriptor rewriting, implicit contiguous nodes, host gather/scatter,
shared aliasing, or hidden fallback.

## Exact numerical and shape gates

Affine transforms perform no floating arithmetic. Every output coordinate must preserve the exact
32-bit payload selected from its input. The real probe and production evidence compare raw bits,
not ULP distance, for:

- positive and negative zero;
- minimum and ordinary subnormals;
- minimum and maximum finite values;
- positive and negative infinity;
- multiple quiet-NaN payloads and signs; and
- ordinary distinct finite lane markers that expose wrong coordinate order or repetition.

The matrix covers identity and shape-changing `RESHAPE`; leading, singleton, multi-axis, and
same-shape `EXPAND`; identity, rank-two transpose, and nontrivial rank-3/rank-16 `PERMUTE`;
front/middle/end `EXPAND_DIMS`; and matching `SQUEEZE`. Each case checks exact output Shape, logical
layout, full dense target byte extent, canonical publication order, and repeated-run stability.
Malformed attrs, mismatched inferred layout, overflow, rank-zero, zero extent, and unsupported view
inputs reject deterministically before native resource creation.

## Lifecycle and implementation boundaries

Analysis validates the complete maximal Metal partition, assigns stable node/value/feed/target
order, fixes the MPSGraph or existing custom-singleton route, and declares exact boundary buffers
plus the MPSGraph address workspace. Affine nodes never select the custom route. Finalization
creates one reusable shape-specialized executable transactionally and transfers ownership through
Prepare. Runtime cold binding validates direct typed representations once; hot execution makes one
synchronous native submission into assigned targets. Repeated and concurrent runs use isolated
run-owned buffers/workspaces and share only immutable prepared resources.

No CPU fallback, route retry, graph inspection, per-node native invocation, hidden output copy,
Engine lookup, async execution, pooling, packaging, tuning integration, or custom Metal kernel is
allowed. Existing custom NEG behavior, resource cleanup, context child leases, rollback,
close/rejection, and exact ABI exports remain unchanged.

## Acceptance gates

- Capability admits exactly the five affine kinds under the stated typed, static, rank, input-
  layout, output-layout, and topology constraints while preserving the exact 0005 domain.
- Java/native schema and validation agree; ABI remains version 4 with exactly thirteen exports;
  candidate/fingerprint/policy versions bump and stale decisions reject.
- Real disposable probe and retained native tests prove every selector, direct supplied targets,
  exact Shapes, and raw-bit preservation before the capability is advertised.
- Fake-native tests cover every mapping and typed attr form, malformed schema records, rank/axis/
  permutation/target bounds, declaration geometry, rollback, close/rejection, and repeated binding.
- Real Metal tests cover each operation, adversarial bits, rank boundaries, fan-out, multiple
  targets, mixed 0005-prefix/affine-leaf partitions, repeated runs, concurrent runs where current
  lifecycle tests support them, and two independent prepared sessions.
- A real Metal-only public `Engine.builder()` scenario publishes and materializes each affine kind
  from direct Metal output buffers with exact canonical bytes and no CPU partition or
  representation. A separate negative scenario proves affine view cross-owner transfer still
  rejects and existing canonical rank-`1..16` transfer remains unchanged.
- Compiler-generated forward and first-order graphs are built and their affine/reduction operation
  identities, Shapes, layouts, and attrs inspected. No Metal-only backward claim is made.
- Documentation, Javadocs, native schema/export audit, focused architecture checks, and an
  independent Class C review pass before completion.

## Dependencies and integration

- **Depends on:** Metal 0005; current Model affine operation/attribute and layout contracts;
  Compiler layout inference/autograd; Engine 0017; Compiler 0006B7 and 0006B11; Prepare 0008;
  Runtime 0016. Metal 0006 and 0007 are explicitly not dependencies.
- **Conflicts with:** Metal capability, preparation, native ABI/schema, candidate compatibility,
  Engine Metal materialization, and Metal documentation scopes; any shared aliasing or transfer
  work.
- **Parallel group:** None.
- **Common base revision:** `0aafa92d1135f5588ce9c28983b1b9a46a5c15a1`.
- **Integration order:** real selector/bit probe → typed schema and validation → capability and
  analysis → finalization/runtime/materialization → fake/native tests → public Engine proof → docs
  and independent review.
- **Integration validation:** native build/export audit; focused Metal, backend-conformance, and
  real Engine suites; architecture tests; documentation checks; `git diff --check`.
- **Shared-document integration owner:** Main planner.

## Validation

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*Affine*' --tests '*Metal*MpsGraph*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :backends:metal:javadoc
./gradlew :testing:architecture-tests:test
git diff --check
```

Also verify balanced Markdown fences, final newlines, local links/anchors, exact thirteen native
exports, no changed production dependency direction, no custom affine kernel, and no broadened
CPU/Metal transfer predicate.

## Documentation and review impact

Update the Metal backend/native guides, package/Javadocs, master plan, roadmap, and glossary/API
status only after executable evidence passes. Architecture and Model/Compiler semantic documents
should remain unchanged because this task realizes existing contracts. A clean independent Class C
review must inspect logical-versus-physical layout truth, publication authentication, transfer
non-widening, schema bounds, selector availability, direct-output behavior, bit preservation,
rollback, and absence of claims for 0006, 0007, backward reduction, or MATMUL.

## Result

Implemented in `7e39f705`, remediated in `9713e528` and
`aa42ed711a0d3120f97e4241181b557130255e97`, and independently approved at Class C with zero
findings. The mandatory disposable selector/direct-target/raw-bit probe passed
16 cases twice and was removed. Production retains ABI v4 with exactly thirteen exports, uses typed
node schema 2, implements only the five bounded terminal affine occurrences, authenticates dense
represented-order publication, and leaves canonical-only CPU/Metal transfer unchanged.

Final evidence: 46 focused Metal tests passed with zero failures, errors, or skips (raw ABI 6,
prepared 34, candidate 6); five real Engine Metal integration tests, two conformance tests, and
nine architecture tests also passed with zero failures, errors, or skips. Native build/export,
Javadoc, and diff checks passed; the reviewed implementation tree was clean. No Metal-only
backward, blocked 0006/0007, custom affine kernel, transfer widening, or MATMUL claim is made.

Status: Complete
