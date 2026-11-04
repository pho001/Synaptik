# Task 0008: [MPSGraph](../../../../glossary.md#term-mpsgraph) [FLOAT32](../../../../glossary.md#term-float32) Affine Transforms

## Reader context

An affine descriptor explains logical coordinates, not necessarily the physical arrangement of
this backend's output bytes. Here MPSGraph wrote a separate dense buffer in logical coordinate
order while the result retained its Model/Compiler view description. It did not alias or borrow
the source's storage.

A terminal leaf in this task means a result with no consuming operation inside the partition;
it is not the application's input leaf Tensor. Metal authenticated that private result
representation before publication could copy detached canonical host bytes. The completed slice
did not thereby admit arbitrary view consumers or widen cross-owner transfer.
The [current lowering and physical-geometry explanation](../../../../explanation/packages/metal.md#2-lower-once-declare-resources-then-compile)
explains why logical view ancestry alone is still not proof of a physical buffer's geometry.

## Status

Complete

[Metal](../../../../glossary.md#term-metal) 0006 and 0007 remain independently Blocked; this task neither depended on them nor changed
their numerical contracts. Metal 0009 is the separately authorized successor.

## Change class

Class C. This extends [Metal](../../../../glossary.md#term-metal) capability, the typed native [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema), [backend](../../../../glossary.md#backend)-private physical
representation semantics for logical affine-[view](../../../../glossary.md#term-view-meanings) results, prepared-resource [lifecycle](../../../../glossary.md#lifecycle), and public
[Engine](../../../../glossary.md#term-engine) [materialization](../../../../glossary.md#materialization). It changes no shared architecture, [Model](../../../../glossary.md#term-model) semantics, cross-[owner](../../../../glossary.md#term-owner) transfer
contract, or public [API](../../../../glossary.md#term-application-programming-interface-api).

## Goal

Extend the existing [ABI](../../../../glossary.md#term-application-binary-interface-abi)-v4 whole-[partition](../../../../glossary.md#partition) [MPSGraph](../../../../glossary.md#term-mpsgraph) route with exactly five [FLOAT32](../../../../glossary.md#term-float32) affine
transforms:

- `ShapeTransformKind.RESHAPE` with `TargetShapeAttrs`;
- `ShapeTransformKind.EXPAND` with `TargetShapeAttrs`;
- `AxisTransformKind.PERMUTE` with `PermutationAttrs`;
- `AxisTransformKind.EXPAND_DIMS` with `AxisTransformAttrs`; and
- `AxisTransformKind.SQUEEZE` with `AxisTransformAttrs`.

Preserve the current `NEG` and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `ADD/SUB/MUL/DIV` domain and the custom [singleton](../../../../glossary.md#term-singleton)-NEG route.
Produce exact logical values and [Shapes](../../../../glossary.md#shape) while retaining the exact [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) affine-[view](../../../../glossary.md#term-view-meanings)
[layout](../../../../glossary.md#layout) [descriptors](../../../../glossary.md#term-descriptor). A [Metal](../../../../glossary.md#term-metal) MPSGraph [target](../../../../glossary.md#term-target) for an affine result is a [backend](../../../../glossary.md#backend)-private dense
physical [materialization](../../../../glossary.md#materialization) in [canonical](../../../../glossary.md#term-canonical-form) logical [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) order; it does not claim shared storage
[aliasing](../../../../glossary.md#term-aliasing) with the source and does not rewrite the logical descriptor.

## Independence decision

This work is independent of blocked 0006 and 0007. The five operations only reindex, repeat,
insert, or remove logical [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and must preserve represented [FLOAT32](../../../../glossary.md#term-float32) bits; none invokes
[unary](../../../../glossary.md#term-unary) algebra, reduction, accumulation, division, or a reduction-derived count. The [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation)
and typed-attribute records already exist, [Compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference) already validates their exact [Shapes](../../../../glossary.md#shape)
and [layouts](../../../../glossary.md#layout). Current architecture assigns [lowering](../../../../glossary.md#lowering), physical [materialization](../../../../glossary.md#materialization), and [MPSGraph](../../../../glossary.md#term-mpsgraph)
resource [ownership](../../../../glossary.md#term-ownership) to [Metal](../../../../glossary.md#term-metal).

`EXPAND` first-order backward uses `SUM_TO_SHAPE`, so blocked 0007 prevents a complete Metal-only
backward claim for this family. Task 0008 is therefore a forward capability. It must build and
inspect Compiler-generated [gradient](../../../../glossary.md#term-gradient) [graphs](../../../../glossary.md#graph) as a contract gate, but it must not claim executable
Metal-only backward. The later [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [checkpoint](../../../../glossary.md#term-checkpoint) is a separate successor and is not authorized by
this brief.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects Metal ownership; Metal owns truthful lowering and Runtime invokes prepared work.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`ARCHITECTURE.md` — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span> —
  Metal owns MPSGraph lowering, storage, native integration, and Metal-specific materialization.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [materialization glossary](../../../../glossary.md#materialization))</span>
- [`ARCHITECTURE.md` — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  — analysis declares all resources before assignment and finalization cannot reselect a route.
- [`ARCHITECTURE.md` — Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>
  — cold binding creates typed direct references and hot execution performs no lookup or selection.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [Metal backend guide](../../../../backend-guide/metal-backend.md),
  [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>, and
  [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span> — current executable, affine-layout, and Compiler
  contracts.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [Compiler glossary](../../../../glossary.md#term-compiler))</span>

If implementation requires a shared [aliasing](../../../../glossary.md#term-aliasing) contract, [descriptor](../../../../glossary.md#term-descriptor) rewrite, broader [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal)
transfer, custom [kernel](../../../../glossary.md#kernel), CPU fallback, or public [API](../../../../glossary.md#term-application-programming-interface-api) change, stop and mark the task Blocked rather
than expanding [scope](../../../../glossary.md#term-scope).

## Exact capability domain

An affine [occurrence](../../../../glossary.md#term-occurrence) is eligible only when all of the following hold:

1. It has exactly one input, one output, and the exact typed attributes paired with its operation
   kind. No generic attributes, strings, reflection, or inferred operation names are allowed.
2. Input and output use [`FLOAT32`](../../../../glossary.md#term-float32), have equal `requiresGrad` flags, are fully static, have strictly
   positive extents, and have ranks in `1..16`. [Rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and zero-[extent](../../../../glossary.md#term-extent) cases remain unsupported.
3. The input descriptor is exactly canonical dense-contiguous, zero-offset, and non-view. This
   deliberately excludes caller or partition feeds whose logical coordinates require gathering,
   permutation, offset handling, or broadcast-stride interpretation.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [permutation glossary](../../../../glossary.md#permutation))</span>
4. The output descriptor exactly equals the Model/Compiler result: exact Shape, zero storage
   offset, <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [storage offset glossary](../../../../glossary.md#term-storage-offset))</span>`isView = true`, and exact derived [strides](../../../../glossary.md#term-stride). `RESHAPE` uses [canonical](../../../../glossary.md#term-canonical-form) [target](../../../../glossary.md#term-target) strides;
   `EXPAND` uses right-aligned preserved or zero strides; `PERMUTE` reorders input strides;
   `EXPAND_DIMS` inserts the exact derived [singleton](../../../../glossary.md#term-singleton) stride; and `SQUEEZE` removes the selected
   singleton stride.
5. Target Shapes, axes, permutations, element counts, referenced spans, dense physical element
   counts, byte extents, and all fixed-width native conversions pass checked arithmetic and schema
   bounds.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [element glossary](../../../../glossary.md#term-element); [schema glossary](../../../../glossary.md#term-schema))</span>

This input rule makes an affine [operation](../../../../glossary.md#operation) a terminal [leaf](../../../../glossary.md#term-leaf) of the currently admitted [Metal](../../../../glossary.md#term-metal)
operation domain. One maximal [partition](../../../../glossary.md#partition) may contain any already-supported 0005 [topology](../../../../glossary.md#term-topology) followed
by one or more affine leaves with stable [fan-out](../../../../glossary.md#term-fan-out) and multiple targets. Task 0008 does not admit an
affine result as the input of another affine or existing [elementwise](../../../../glossary.md#term-elementwise) occurrence, because that input
is a logical [view](../../../../glossary.md#term-view-meanings) rather than canonical non-view geometry. A later [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) task must state and prove
any view-input rule it needs; this brief does not pre-authorize it.

Capability and analysis must [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) for [`CONTIGUOUS`](../../../../glossary.md#term-contiguous), [slice](../../../../glossary.md#slice)/select/pad/tile/composition,
MATMUL, every other operation family, wrong attributes, wrong type, dynamic or zero [dimensions](../../../../glossary.md#dimension),
[rank](../../../../glossary.md#term-rank) outside `1..16`, noncanonical input, incorrect output [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout), malformed topology, or
unchecked geometry.

## [Model](../../../../glossary.md#term-model) and [Compiler](../../../../glossary.md#term-compiler) contract readout

- `RESHAPE` preserves ordered logical elements and requires equal element counts. For the admitted
  canonical input, its exact static result is a same-offset view with canonical target strides.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element); [canonical glossary](../../../../glossary.md#term-canonical-form); [view glossary](../../../../glossary.md#term-view-meanings); [target glossary](../../../../glossary.md#term-target); [strides glossary](../../../../glossary.md#term-stride))</span>
- `EXPAND` right-aligns source and target. A source extent must equal the target or be one; target-
  only leading axes are implicit source singletons. Repeated axes have output stride zero.
- `PERMUTE` carries a complete output-to-input permutation whose length equals rank and whose axes
  are unique and in range. Rank-two transpose is <span data-glossary-navigation="true"> ([Rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [transpose glossary](../../../../glossary.md#term-transpose))</span>`PERMUTE [1, 0]`, not another [operation](../../../../glossary.md#operation) kind.
- `EXPAND_DIMS` inserts one [extent](../../../../glossary.md#term-extent)-one output [axis](../../../../glossary.md#term-axis) at the normalized position. `SQUEEZE` removes one
  selected input axis only when its static extent is exactly one.
- Compiler inference retains the exact affine view geometry and deliberately excludes these results
  from canonical logical-layout closure. Metal therefore must validate, not replace, that metadata.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [layout glossary](../../../../glossary.md#layout); [Metal glossary](../../../../glossary.md#term-metal); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Compiler [gradient](../../../../glossary.md#term-gradient) rules use inverse reshape/[permutation](../../../../glossary.md#permutation)/[rank](../../../../glossary.md#term-rank) edits and use `SUM_TO_SHAPE` for
  `EXPAND`. Gradient [graph](../../../../glossary.md#graph) construction and [descriptor](../../../../glossary.md#term-descriptor) inspection are required; execution is not.

## Native [schema](../../../../glossary.md#term-schema) and [MPSGraph](../../../../glossary.md#term-mpsgraph) [lowering](../../../../glossary.md#lowering)

Keep native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 4 and its exact thirteen exported symbols unchanged. Evolve the pointed-to
[node](../../../../glossary.md#node) schema to a documented version-2 typed discriminated record or union. It must represent only:

- the existing no-attribute [unary](../../../../glossary.md#term-unary)/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) variants;
- [target](../../../../glossary.md#term-target)-[Shape](../../../../glossary.md#shape) attributes with bounded [rank](../../../../glossary.md#term-rank) and [dimensions](../../../../glossary.md#dimension);
- a complete bounded [permutation](../../../../glossary.md#permutation); and
- one bounded [normalized axis](../../../../glossary.md#normalized-axis).

Every unused field must have a required zero/sentinel value. Java and native [validation](../../../../glossary.md#term-validation) must agree
on [operation](../../../../glossary.md#operation)/attribute pairing, rank, target dimensions, [axes](../../../../glossary.md#term-axis), permutation completeness, ordered
inputs, one output, [topological](../../../../glossary.md#term-topological-order) value [identity](../../../../glossary.md#term-identity-meanings), and schema size. No variable object [graph](../../../../glossary.md#graph), generic
integer [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), raw string, serialized map, hidden pointer, or dimensions repurposed as attrs may
cross the ABI.

Use the current [SDK](../../../../glossary.md#term-sdk) typed selectors:

- `reshapeTensor:withShape:name:` for `RESHAPE`;
- `broadcastTensor:toShape:name:` for `EXPAND`;
- `transposeTensor:permutation:name:` for `PERMUTE`;
- `expandDimsOfTensor:axis:name:` for `EXPAND_DIMS`; and
- `squeezeTensor:axis:name:` for `SQUEEZE`.

Before production edits, [compile](../../../../glossary.md#compile) and [run](../../../../glossary.md#term-run) a disposable real probe against those selectors and the
exact direct-output-[buffer](../../../../glossary.md#term-buffer) [binding](../../../../glossary.md#term-binding) used by production. If a selector is unavailable, reports a
wrong Shape, does not write the supplied target directly, or fails the exact bit/shape [matrix](../../../../glossary.md#term-matrix-meanings) below,
revert the probe and mark the task Blocked. Do not substitute an undocumented selector, custom
[kernel](../../../../glossary.md#kernel), [host](../../../../glossary.md#term-host) rewrite, or relaxed gate.

Bump the Java/native node-schema version, candidate schema, workload [fingerprint](../../../../glossary.md#term-fingerprint) schema, and route-
policy version monotonically. [Compatibility identity](../../../../glossary.md#identity-meanings-and-algebraic-identities) includes operation kind, typed attrs, exact
input/output [descriptor](../../../../glossary.md#term-descriptor) and logical [layout](../../../../glossary.md#layout), dense represented-order geometry, ordered [topology](../../../../glossary.md#term-topology),
numerical policy, native node schema, ABI version, and live [Metal](../../../../glossary.md#term-metal) target identity. Old decisions
must [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

## Physical [materialization](../../../../glossary.md#materialization) and transfer boundary

An affine [graph](../../../../glossary.md#graph) [target](../../../../glossary.md#term-target) receives a [Metal](../../../../glossary.md#term-metal) [buffer](../../../../glossary.md#term-buffer) sized to the full positive logical [element](../../../../glossary.md#term-element) count,
not merely the affine [descriptor](../../../../glossary.md#term-descriptor)'s [referenced element span](../../../../glossary.md#referenced-element-span). [MPSGraph](../../../../glossary.md#term-mpsgraph) writes that target in
[canonical](../../../../glossary.md#term-canonical-form) logical [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) order. This is a [backend](../../../../glossary.md#backend)-private physical representation choice:
logical zero [strides](../../../../glossary.md#term-stride), permuted strides, and `isView` remain unchanged in graph [metadata](../../../../glossary.md#term-metadata), and no two
logical graph values share a [Runtime](../../../../glossary.md#term-runtime) representation or slot.

Extend Metal [publication](../../../../glossary.md#term-publication) materialization only enough to turn this authenticated dense represented
order into [detached](../../../../glossary.md#detached-model-results) canonical [host](../../../../glossary.md#term-host) bytes for an exact admitted affine publication. Validate the
prepared route [identity](../../../../glossary.md#term-identity-meanings), descriptor, context, openness, current-thread access, full logical byte
[extent](../../../../glossary.md#term-extent), and maximum-byte limit before downloading. Raw [FLOAT32](../../../../glossary.md#term-float32) bits must be retained.

Do not widen [CPU](../../../../glossary.md#term-cpu)-to-Metal or Metal-to-CPU transfer. Cross-[owner](../../../../glossary.md#term-owner) transfer remains restricted to
positive [rank](../../../../glossary.md#term-rank)-`1..16` static canonical-[contiguous](../../../../glossary.md#term-contiguous) non-[view](../../../../glossary.md#term-view-meanings) FLOAT32. An affine logical view at a
CPU/Metal boundary must still fail during transfer [preflight](../../../../glossary.md#term-preflight) before backend analysis. Do not add
canonical-byte transfer, descriptor rewriting, implicit contiguous [nodes](../../../../glossary.md#node), host [gather](../../../../glossary.md#gather)/scatter,
shared [aliasing](../../../../glossary.md#term-aliasing), or hidden fallback.

## Exact numerical and [shape](../../../../glossary.md#shape) gates

Affine transforms perform no floating arithmetic. Every output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) must preserve the exact
32-bit [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) selected from its input. The real probe and production evidence compare [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity),
not [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) distance, for:

- positive and [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero);
- minimum and ordinary [subnormals](../../../../glossary.md#term-subnormal);
- minimum and maximum [finite](../../../../glossary.md#term-finite) values;
- positive and [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero);
- multiple quiet-[NaN](../../../../glossary.md#term-nan) payloads and signs; and
- ordinary distinct finite [lane](../../../../glossary.md#term-lane) markers that expose wrong coordinate order or repetition.

The [matrix](../../../../glossary.md#term-matrix-meanings) covers [identity](../../../../glossary.md#term-identity-meanings) and shape-changing `RESHAPE`; leading, [singleton](../../../../glossary.md#term-singleton), multi-[axis](../../../../glossary.md#term-axis), and
same-shape `EXPAND`; identity, [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [transpose](../../../../glossary.md#term-transpose), and nontrivial [rank](../../../../glossary.md#term-rank)-3/rank-16 `PERMUTE`;
front/middle/end `EXPAND_DIMS`; and matching `SQUEEZE`. Each case checks exact output Shape, logical
[layout](../../../../glossary.md#layout), full [dense target](../../../../glossary.md#dense-target) byte [extent](../../../../glossary.md#term-extent), [canonical](../../../../glossary.md#term-canonical-form) [publication](../../../../glossary.md#term-publication) order, and repeated-[run](../../../../glossary.md#term-run) stability.
Malformed attrs, mismatched inferred layout, [overflow](../../../../glossary.md#term-overflow), [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), zero extent, and unsupported [view](../../../../glossary.md#term-view-meanings)
inputs reject deterministically before native resource creation.

## [Lifecycle](../../../../glossary.md#lifecycle) and implementation boundaries

Analysis validates the complete maximal [Metal](../../../../glossary.md#term-metal) [partition](../../../../glossary.md#partition), assigns stable [node](../../../../glossary.md#node)/value/feed/[target](../../../../glossary.md#term-target)
order, fixes the [MPSGraph](../../../../glossary.md#term-mpsgraph) or existing custom-[singleton](../../../../glossary.md#term-singleton) route, and declares exact boundary [buffers](../../../../glossary.md#term-buffer)
plus the MPSGraph address workspace. Affine nodes never select the custom route. Finalization
creates one reusable [shape](../../../../glossary.md#shape)-specialized executable transactionally and transfers [ownership](../../../../glossary.md#term-ownership-meanings) through
[Prepare](../../../../glossary.md#prepare). [Runtime](../../../../glossary.md#term-runtime) cold [binding](../../../../glossary.md#term-binding) validates direct typed representations once; hot execution makes one
[synchronous](../../../../glossary.md#term-synchronous) native submission into assigned targets. Repeated and concurrent runs use isolated
[run](../../../../glossary.md#term-run)-owned buffers/workspaces and share only [immutable](../../../../glossary.md#term-immutable) prepared resources.

No [CPU](../../../../glossary.md#term-cpu) fallback, route retry, [graph](../../../../glossary.md#graph) inspection, per-node native invocation, hidden output copy,
[Engine](../../../../glossary.md#term-engine) [lookup](../../../../glossary.md#term-lookup), async execution, [pooling](../../../../glossary.md#term-pooling), packaging, tuning integration, or custom Metal [kernel](../../../../glossary.md#kernel) is
allowed. Existing custom NEG behavior, resource cleanup, context child [leases](../../../../glossary.md#term-lease), [rollback](../../../../glossary.md#term-rollback),
close/rejection, and exact [ABI](../../../../glossary.md#term-application-binary-interface-abi) exports remain unchanged.

## Acceptance gates

- Capability admits exactly the five affine kinds under the stated typed, static, rank, input-
  layout, output-layout, and topology constraints while preserving the exact 0005 domain.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [layout glossary](../../../../glossary.md#layout); [topology glossary](../../../../glossary.md#term-topology))</span>
- Java/native schema and validation agree; ABI remains version 4 with exactly thirteen exports;
  candidate/fingerprint/policy versions bump and stale decisions reject.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [validation glossary](../../../../glossary.md#term-validation); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- Real disposable probe and retained native tests prove every selector, direct supplied targets,
  exact Shapes, and raw-bit preservation before the capability is advertised.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [Shapes glossary](../../../../glossary.md#shape))</span>
- Fake-native tests cover every mapping and typed attr form, malformed schema records, rank/axis/
  permutation/target bounds, declaration geometry, rollback, close/rejection, and repeated binding.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [permutation glossary](../../../../glossary.md#permutation); [rollback glossary](../../../../glossary.md#term-rollback); [binding glossary](../../../../glossary.md#term-binding))</span>
- Real Metal tests cover each operation, adversarial bits, rank boundaries, fan-out, multiple
  targets, mixed 0005-prefix/affine-leaf partitions, repeated runs, concurrent runs where current
  lifecycle tests support them, and two independent prepared sessions.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [fan-out glossary](../../../../glossary.md#term-fan-out); [leaf glossary](../../../../glossary.md#term-leaf); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A real [Metal](../../../../glossary.md#term-metal)-only public `Engine.builder()` scenario publishes and materializes each affine kind
  from direct Metal output buffers with exact canonical bytes and no CPU partition or
  representation. A separate negative scenario proves affine view cross-owner transfer still
  rejects and existing canonical rank-<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [canonical glossary](../../../../glossary.md#term-canonical-form); [CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [view glossary](../../../../glossary.md#term-view-meanings); [owner glossary](../../../../glossary.md#term-owner))</span>`1..16` transfer remains unchanged.
- Compiler-generated forward and first-order graphs are built and their affine/reduction operation
  identities, Shapes, layouts, and attrs inspected. No Metal-only backward claim is made.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graphs glossary](../../../../glossary.md#graph))</span>
- Documentation, Javadocs, native schema/export audit, focused architecture checks, and an
  independent Class C review pass before completion.

## Dependencies and integration

- **Depends on:** Metal 0005; current Model affine operation/attribute and layout contracts;
  Compiler layout inference/autograd; Engine 0017; Compiler 0006B7 and 0006B11; Prepare 0008;
  Runtime 0016. Metal 0006 and 0007 are explicitly not dependencies.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [layout glossary](../../../../glossary.md#layout); [Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [autograd glossary](../../../../glossary.md#autograd); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- **Conflicts with:** Metal capability, preparation, native ABI/schema, candidate compatibility,
  Engine Metal materialization, and Metal documentation scopes; any shared aliasing or transfer
  work.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [materialization glossary](../../../../glossary.md#materialization); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- **Parallel group:** None.
- **Common base revision:** `0aafa92d1135f5588ce9c28983b1b9a46a5c15a1`.
- **Integration order:** real selector/bit probe → typed schema and validation → capability and
  analysis → finalization/runtime/materialization → fake/native tests → public Engine proof → docs
  and independent review.
- **Integration [validation](../../../../glossary.md#term-validation):** native build/export audit; focused Metal, backend-conformance, and
  real Engine suites; architecture tests; documentation checks; <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>`git diff --check`.
- **Shared-document integration [owner](../../../../glossary.md#term-owner):** Main planner.

## [Validation](../../../../glossary.md#term-validation)

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
exports, no changed production [dependency direction](../../../../glossary.md#java-platform-and-build-tools), no custom affine [kernel](../../../../glossary.md#kernel), and no broadened
[CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) transfer predicate.

## Documentation and review impact

Update the [Metal](../../../../glossary.md#term-metal) [backend](../../../../glossary.md#backend)/native guides, package/Javadocs, master plan, roadmap, and glossary/[API](../../../../glossary.md#term-application-programming-interface-api)
status only after executable evidence passes. Architecture and [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) semantic documents
should remain unchanged because this task realizes existing contracts. A clean independent Class C
review must inspect logical-versus-physical [layout](../../../../glossary.md#layout) truth, [publication](../../../../glossary.md#term-publication) [authentication](../../../../glossary.md#term-authentication), transfer
non-[widening](../../../../glossary.md#term-widening), [schema](../../../../glossary.md#term-schema) bounds, selector availability, direct-output behavior, bit preservation,
[rollback](../../../../glossary.md#term-rollback), and absence of claims for 0006, 0007, backward reduction, or [MATMUL](../../../../glossary.md#matrix-multiplication--matmul).

## Result

Implemented in `7e39f705`, remediated in `9713e528` and
`aa42ed711a0d3120f97e4241181b557130255e97`, and independently approved at Class C with zero
findings. The mandatory disposable selector/direct-[target](../../../../glossary.md#term-target)/raw-bit probe passed
16 cases twice and was removed. Production retains [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 with exactly thirteen exports, uses typed
[node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) 2, implements only the five bounded terminal affine occurrences, authenticates dense
represented-order [publication](../../../../glossary.md#term-publication), and leaves [canonical](../../../../glossary.md#term-canonical-form)-only [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) transfer unchanged.

Final evidence: 46 focused Metal tests passed with zero failures, errors, or skips (raw ABI 6,
prepared 34, candidate 6); five real [Engine](../../../../glossary.md#term-engine) Metal integration tests, two [conformance](../../../../glossary.md#term-conformance) tests, and
nine architecture tests also passed with zero failures, errors, or skips. Native build/export,
[Javadoc](../../../../glossary.md#term-javadoc), and diff checks passed; the reviewed implementation tree was clean. No Metal-only
backward, blocked 0006/0007, custom affine [kernel](../../../../glossary.md#kernel), transfer [widening](../../../../glossary.md#term-widening), or [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) claim is made.

Status: Complete
