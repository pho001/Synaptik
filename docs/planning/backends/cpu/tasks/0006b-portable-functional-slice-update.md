# Task 0006B: Portable Functional Slice Update

## Status

Complete

## Goal

Add the first executable CPU frontier after completed CPU 0006A2: exactly one fully static,
resolved-layout `SliceKind.SLICE_UPDATE` occurrence using either current `SliceAttrs` signed
finite-coordinate semantics or current `CropToShapeAttrs` target-relative placement semantics.

The CPU result is functional, not in-place. Every logical result coordinate reads either the
corresponding base value or the uniquely mapped update value, writes that represented value once,
and leaves both inputs unchanged. All six current Model data types are represented-bit movement
rows. This task extends the existing CPU data-movement IR, compact geometry, generated emitter,
preparation, binding, reference, and artifact seams without adding a shared contract or another
route.

The former broad CPU 0006B frontier is dependency-split because its operations do not share one
bounded execution contract:

```text
0006B  SLICE_UPDATE
  value-blind output-domain choice between base and update
  -> existing movement family, injective one-write output, no value validation

0006B1 SCATTER_ELEMENTS / SCATTER_ADD / SCATTER_ND
  index-value bounds and duplicate policy plus replacement or reduction
  -> separate pre-write validation and functional-scatter execution design

0006B2 FOLD_AXIS / FOLD2D
  zero-initialized result plus overlap accumulation and 2D padding exclusion
  -> separate accumulation execution design
```

Only this first slice receives a detailed specification. CPU 0006B1 and 0006B2 remain concise
`Draft` master-plan rows without task files.

## Scope

- Admit exactly one CPU-owned node whose kind is `SliceKind.SLICE_UPDATE`, whose ordered logical
  inputs are `[base, update]`, and whose sole output has the exact base Shape and data type.
- Support both current family-owned signatures:
  - `SliceAttrs` for normalized distinct axes, non-negative starts and lengths, and signed
    non-zero steps; and
  - `CropToShapeAttrs` for an exact update-region Shape beginning after the exact per-axis prefix
    Shape.
- Require every base, update, and output Shape to be fully static and every layout to be resolved
  before CPU capability or lowering succeeds.
- Preserve represented bits for FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and canonical BOOL.
- Require an injective materialized output layout. The output value and its physical accessed span
  remain distinct and non-overlapping with every unique input at cold binding.
- Extend `CpuDataMovementIr` with one nested `SliceUpdatePlan` and
  `CpuNonAffineMovementLowering.Geometry` with one nested `SliceUpdate` variant. Add no top-level
  production type or package.
- Normalize both attribute forms during lowering to one compact rank-sized sequence geometry:
  per output axis, retain an inclusive start, finite length, and signed non-zero step. An
  unselected `SliceAttrs` axis normalizes to start zero, full base extent, and step one;
  `CropToShapeAttrs` normalizes every axis to prefix extent, update extent, and step one.
- Generate one direct output-domain scalar body. At each output coordinate, choose update only
  when every normalized axis is on its finite sequence; otherwise choose base. Derive the update
  coordinate from the sequence ordinal on each axis.
- Seed arbitrary range starts on the cold geometry path. The generated loop must advance output
  coordinates and per-axis sequence cursors through primitive carry/reset state, with no
  per-element division, modulo, allocation, table lookup, Model interpretation, or semantic
  dispatch.
- Preserve semantic input occurrence order while declaring each distinct input `ValueId` once in
  first-occurrence order. The plan's occurrence map is `[baseBoundary, updateBoundary]`; when
  base and update are the same `ValueId`, it is `[0, 0]` and only one input buffer is declared.
- Select scalar or parallel-scalar orchestration through the completed movement path. Parallel
  chunks are deterministic disjoint output ranges because each output coordinate is written once.
- Declare unique input buffers followed by the output buffer, one execution unit, no workspace,
  no materialization, and one generated artifact.
- Advance generated compatibility exactly from schema 14 to schema 15 because the new movement
  family changes emitted bytecode. Preserve current-only miss/regeneration behavior.
- Extend the independent scalar reference and focused generated/reference tests for both
  attribute forms, signed steps, arbitrary ranges, layouts, carriers, zero extents, scalar Shape,
  and represented-bit parity.
- After executable Java and tests stabilize, use a distinct clean documentation-focused context
  to finalize affected Javadocs, package summaries, the CPU guide, glossary, task evidence, CPU
  master plan, and roadmap.

## Current Model contract and future CPU realization

### Shared signature, Shape, and type boundary

Current Model semantics are already complete:

- `SLICE_UPDATE + SliceAttrs` and `SLICE_UPDATE + CropToShapeAttrs` each have exactly two ordered
  inputs and one output.
- Input zero is the unchanged base; input one is the unchanged update.
- Base and update have the same exact data type and rank. The result has the exact base Shape and
  type. Model construction leaves result layout unresolved.
- All six current data types participate. There is no promotion, conversion, accumulation type,
  reduction attribute, padding scalar, or update scalar separate from input one.
- Result gradient eligibility in Model is the logical OR of base and update eligibility. CPU
  execution consumes only the projected descriptors and values; it does not inspect or change
  gradient metadata.

This task adds only the concrete CPU realization after shared lifecycle work supplies fully static
Shapes and resolved layouts. Dynamic or unresolved fit obligations remain unsupported by CPU
0006B and fail closed.

### Signed finite-coordinate placement

For `SliceAttrs`, entry `i` identifies normalized base axis `axes[i]` and maps update coordinate
`k` on that axis to:

```text
baseCoordinate = starts[i] + k * steps[i]
0 <= k < lengths[i]
```

The update extent on a selected axis equals `lengths[i]`. Every unselected update Dimension equals
the corresponding base Dimension and keeps the same coordinate. Distinct axes and non-zero steps
make the complete update-to-base coordinate mapping injective, so no two update coordinates target
the same result coordinate. Positive and negative steps are equally valid. A length-one entry
also permits `Long.MIN_VALUE` as its step because no second coordinate or absolute-step arithmetic
is required.

If any selected length is zero, the Cartesian update target is empty and every result coordinate
comes from base. Four empty attribute lists select no axes: update Shape must equal base Shape and
every result coordinate comes from update. Rank-zero base/update/output with empty lists is the
scalar form of that complete replacement.

Compact mapping example:

- Inputs: base `[10, 11, 12, 13, 14]`; update `[90, 80]`.
- Transformation: `starts=[4]`, `lengths=[2]`, `axes=[0]`, `steps=[-2]`, so update coordinates
  zero and one target base coordinates four and two.
- Result: `[10, 11, 80, 13, 90]`.
- Interpretation: values outside target coordinates two and four retain their exact base
  representations; this is replacement, not addition or mutation.

### Target-relative placement

For `CropToShapeAttrs`, the exact `targetShape` is the update Shape, not the result Shape. On each
axis `a`, the target interval is:

```text
prefixShape[a] <= baseCoordinate < prefixShape[a] + updateShape[a]
updateCoordinate = baseCoordinate - prefixShape[a]
```

All axes participate. The CPU static boundary requires exact equality between `targetShape` and
the update Shape, equal base/update/output ranks, output Shape equal to base Shape, and checked
`prefix + update <= base` on every axis before capability/lowering succeeds. A zero update extent
on any axis makes the complete target empty. For rank zero, the per-axis conditions are vacuous
and the scalar update replaces the scalar base.

Compact mapping example:

- Inputs: base `[[1, 2, 3, 4], [5, 6, 7, 8]]`; update
  `[[9, 10], [11, 12]]`.
- Transformation: target/update Shape `[2, 2]` with prefix Shape `[0, 1]` places the update after
  zero rows and one column.
- Result: `[[1, 9, 10, 4], [5, 11, 12, 8]]`.
- Interpretation: the prefix is a logical target-relative offset, not a storage address or a
  Tensor input; the result still has base Shape `[2, 4]`.

### Replacement, duplicates, order, and validation

`SLICE_UPDATE` has no reduction and no duplicate-target policy because its coordinate mapping is
injective. Every target receives exactly one update value, and every non-target receives exactly
one base value. There is no first-write/last-write rule, base participation at a replaced target,
reduction identity, accumulation order, or overlap accumulation.

All geometry validation is static/cold. Unlike Gather and future scatter, this operation has no
index tensor and needs no execution-time index-value pass. Existing cold binding still validates
exact carriers, spans, alignment, mutability, lifetime, output/input non-overlap, and canonical
BOOL bytes for every unique BOOL input before any generated call or worker submission. Invalid
structural geometry is rejected before artifact realization; invalid binding mutates no output.

### Deferred scatter and fold semantics

Current Model semantics for the later Draft slices remain unchanged:

- Every scatter has ordered `[data, indices, updates]` inputs and a fresh output with the exact
  data Shape and type. Indices are exact INT32 or INT64; updates have the exact data type. `NONE`
  accepts all six current data types, while arithmetic reductions and intrinsic scatter-add
  accept the five floating or signed-integral types and reject BOOL.
- `SCATTER_ELEMENTS` uses `ScatterElementsAttrs(axis, reduction)`. Indices and updates have equal
  same-rank Shapes, that rank equals data rank, and every non-selected Dimension equals data; the
  selected extent may differ. Because an axis must exist, its updates Tensor cannot be rank-zero.
- Gather-compatible `SCATTER_ADD` uses `IndexAxisAttrs`. Its updates Shape is the data prefix
  before the selected axis, followed by the complete indices Shape, followed by the data suffix.
  Rank-one data with rank-zero indices therefore has a canonical scalar updates Tensor. Addition
  is intrinsic and duplicate indices accumulate.
- `SCATTER_ND` uses tuple depth from the final indices Dimension plus
  `ScatterNdAttrs(batchDimensions, reduction)`. Its updates Shape is the indices prefix without
  tuple depth followed by the untouched data suffix after the indexed axes. A tuple spanning all
  data axes with no indices prefix therefore uses a canonical scalar updates Tensor; otherwise
  each tuple contributes its complete suffix slice scalar by scalar.
- `NONE` replaces an addressed base with its single exact update and requires unique targets;
  duplicates are invalid rather than first-write or last-write ordered. Every arithmetic
  reduction includes the base exactly once and every addressed update exactly once, including
  duplicates. An empty update group preserves the exact base representation instead of applying
  a separate reduction identity.
- Integral `ADD` and `MUL` are fixed-width modular; floating `ADD` is reassociable without a
  bitwise-order guarantee. Floating `MUL` has the current abstract-product special-value and
  final-format-rounding contract. Floating `MIN`/`MAX` propagate NaN and select negative/positive
  zero respectively; integral extrema use signed order. The latter three are defined independently
  of encounter/layout/tree order. CPU 0006B1 must preserve those represented-value targets rather
  than invent one universal update traversal guarantee.
- Scatter index values below zero or at least the indexed extent are invalid and never wrap or
  clamp. Model construction does not read values. Future CPU 0006B1 must resolve complete
  deterministic pre-write bounds validation and complete `NONE` duplicate validation before it
  becomes `Ready`; invalid input must expose no partial result.
- `FOLD_AXIS` has one numeric rank-two-or-greater window input, no data base or scalar-update
  input, and a rank-one-smaller result formed by removing the final window-size Dimension and
  restoring the selected target extent. It starts from conceptual numeric zero, maps final-window
  coordinate `offset` from `windowIndex` to `windowIndex * step + offset`, adds overlaps, and
  leaves uncovered result positions zero.
- `FOLD2D` has one floating rank-three canonical-column input and an explicit rank-four NCHW
  result Shape, with no data base, update Tensor, or padding scalar input. It overlap-adds columns;
  samples mapped outside the unpadded output because of symmetric padding or a ceil-mode tail
  contribute nowhere. Uncovered output positions remain zero and no overlap averaging occurs.
- The current fold contract fixes zero initialization, coordinate participation, and overlap
  summation but does not make the configurable `UNFOLD2D` padding value a fold contribution.
  CPU 0006B2 must resolve its exact type-specific accumulation/determinism realization before it
  becomes `Ready`; CPU 0006B must not guess or consume that later decision.

Those index-dependent reduction and base-free accumulation contracts do not enter CPU 0006B
source, tests, capability, generated identity, or detailed scope.

## Out of scope

- `SCATTER_ELEMENTS`, `SCATTER_ADD`, `SCATTER_ND`, `FOLD_AXIS`, `FOLD2D`, or any detailed 0006B1,
  0006B2, or later task specification
- changing `SliceKind`, `SliceAttrs`, `CropToShapeAttrs`, `TensorSlicePlacementExpressions`, any
  Model validation, Tensor method, signature, Shape rule, descriptor, provenance, gradient rule,
  or public API
- dynamic or symbolic CPU Shapes, unresolved layouts, deferred binding constraints, negative
  storage strides, non-injective output layouts, in-place output, or input/output overlap
- mutable slice assignment, partial update visibility, base/update conversion, promotion,
  reduction, arithmetic, padding, duplicate policy, or update ordering
- adding a new top-level IR, lowerer, emitter, route, registry, manager, utility package, selector
  table, per-output address table, per-element object, or Runtime semantic interpreter
- pointwise fusion, mixed movement chains, broad partition directed acyclic graph (DAG)
  decomposition, cross-node scheduling, multi-output units, or later safe split fallback work
- vectorizing the movement body, masked/gather vector access, native/vendor routes, OpenBLAS,
  tuning, benchmarking, relaxed numerics, or a performance claim
- workspace, materialization, a new resource kind, shared Prepare changes, Runtime changes, Engine
  composition, Planning changes, Trace changes, Config changes, or dependency changes
- `ARCHITECTURE.md`, focused architecture documents, ADRs, architecture tests, Model source/tests,
  Gradle, dependency declarations, backend-conformance tests, integration tests, vendor code, or
  unrelated cleanup
- repository-wide validation outside CPU 0009 and CI

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU 0005A atomic partition-kernel reset](0005a-atomic-partition-kernel-architecture-reset.md)
- [CPU 0006 portable static affine views](0006-portable-static-affine-views-and-boundary-materialization.md)
- [CPU 0006A portable static movement](0006a-portable-pad-tile-and-tensor-composition-movement.md)
- [CPU 0006A1 portable static window extraction](0006a1-portable-static-window-extraction.md)
- [CPU 0006A2 portable Gather and one-hot](0006a2-portable-gather-and-one-hot-indexing.md)
- [Model 0023C slice update and target-relative crop](../../../modules/model/tasks/0023c-slice-update-and-target-relative-crop.md)
- [Model 0025D dynamic-extent slice extraction and symbolic placement](../../../modules/model/tasks/0025d-dynamic-extent-slice-extraction-and-symbolic-slice-placement.md)
- [Model 0025C functional-scatter reduction semantics](../../../modules/model/tasks/0025c-portable-functional-scatter-reduction-semantics.md)

## Architecture constraints

- `ARCHITECTURE.md` remains authoritative. This plan is explanatory and cannot create or override
  a shared contract.
- Model owns the two exact `SLICE_UPDATE` meanings. CPU analysis may validate projected static
  descriptors and lower those meanings, but it must not change them or add a public semantic API.
- Planning has already selected CPU ownership. CPU analysis owns occurrence capability,
  fail-closed lowering, code-shaping identity, route realization, strategy selection, and exact
  resource declarations before shared assignment.
- Shared Prepare remains CPU-opaque. It assigns only the exact input/output buffers declared by
  CPU analysis and receives no slice geometry, operation kind, route, or generated artifact.
- CPU finalization verifies assignments and worker availability before exactly one artifact-store
  lookup. It may not revise capability, lowering, boundaries, route, strategy, or resources.
- Runtime receives one prepared executable and invokes only direct bound work. No `Operation`,
  `CompiledNode`, Tensor metadata, route choice, cache lookup, or slice interpretation may reach
  the generated hot loop or Runtime execution path.
- The existing portable route remains the sole route. The scalar reference is independent test
  evidence, not a selectable Runtime fallback or Model evaluator.
- Exact buffer declarations remain unique inputs in first-occurrence order followed by the sole
  materialized output. No workspace or materialization may be introduced after assignment.
- Output writes are one-to-one and disjoint, permitting current deterministic external parallel
  chunking. Generated code performs no scheduling, synchronization, atomics, or shared reduction.
- If implementation requires a shared/public type, another module, dependency/build change,
  architecture rule, Runtime semantic branch, new resource contract, or non-injective/in-place
  output, stop and report the exact conflict instead of planning around it.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — truthful occurrence capability.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — closed structural movement identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — exact one-node lowering and compact
  static geometry.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct generated output-domain
  represented-bit loop.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent scalar mapping oracle.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema and structural compatibility.
- Existing `internal.prepare` and `internal.executable` contracts are exercised without production
  changes for declarations, scalar/parallel-scalar selection, binding, overlap, canonical BOOL,
  range geometry, and artifact ownership.

Packages added, moved, or removed:

- None.

Type placement:

- Add nested `CpuDataMovementIr.SliceUpdatePlan` to the existing closed movement-plan vocabulary.
  Its structural fields are output rank and the two semantic-occurrence boundary positions.
- Add nested `CpuNonAffineMovementLowering.Geometry.SliceUpdate` to the existing closed cold
  geometry vocabulary. It owns only rank-sized normalized starts, lengths, and signed steps.
- Add no top-level type. A new functional-update IR/lowerer/emitter would duplicate current
  value-blind movement ownership for no current need and is outside scope.

Tests continue to mirror the owning existing packages. No generic `slice`, `index`, `update`,
`util`, registry, manager, or shared abstraction package is introduced.

## Affected files

Authorized CPU production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuDataMovementIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuNonAffineMovementLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`

Authorized CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuDataMovementIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuNonAffineMovementLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Authorized explanatory and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0006b-portable-functional-slice-update.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized. Listed paths change only when their actual implementation contract,
Javadoc, package summary, regression ownership, or synchronized status is affected. The two clean
handoffs must record review-only no-change conclusions for unused authorized paths.

Review without modification:

- `SliceKind`, `SliceAttrs`, `CropToShapeAttrs`, `TensorSlicePlacementExpressions`, `Tensor`,
  operation/signature/Shape/layout/data-type contracts, and all Model tests remain semantic input
  evidence only.
- `CpuPortableKernelIr`, `CpuKernelSpecialization`, `CpuClassFileKernelGenerator`,
  `CpuCarrierEmitter`, `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`,
  `CpuPartitionFinalizer`, `CpuPreparedExecutable`, `CpuPortableRoutePlan`, their package summaries
  not listed above, and current memory/worker contracts already handle another movement variant
  generically. If Java changes are required in one of them, stop and revise this allowlist rather
  than expanding it during implementation.
- Architecture, ADR, architecture-test, backend-conformance, integration, Gradle, dependency,
  Model, shared-module, vendor, and later-task paths remain unchanged.

## Maximum scope

| Category | Maximum | Accounting |
|---|---:|---|
| CPU production/package | 13 | Existing movement, capability, reference, and schema owners; no new top-level type |
| CPU tests | 10 | Existing focused regression owners |
| Explanatory documentation | 2 | CPU guide and glossary |
| Planning/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **28** | **13 + 10 + 2 + 3** |

This ceiling covers one complete generated capability while staying smaller than the earlier
movement and value-dependent indexing tasks. Do not spend unused scope on cleanup. Stop and
request a revised planning decision before a 29th path or any path outside the exact allowlist.

## Acceptance criteria

- Capability returns `true` only for exactly one fully static, resolved-layout
  `SLICE_UPDATE + SliceAttrs` or `SLICE_UPDATE + CropToShapeAttrs` node satisfying the locked
  type, rank, Shape, bounds, and injective-output matrix. Every excluded form returns `false`.
- Lowering independently revalidates the complete occurrence and fails closed before route or
  artifact work. It accepts both attribute forms, all six exact data types, positive/negative
  steps, selected/unselected axes, rank zero, zero extents, and arbitrary resolved input/output
  layouts within scope.
- `SliceAttrs` output matches the exact mapping `start + k * step`; selected update extents equal
  declared lengths, unselected update Dimensions equal base, every non-empty endpoint is inside
  base, and empty/identity forms follow the current Model contract.
- `CropToShapeAttrs` target Shape equals update Shape exactly, prefix and target ranks match base,
  output Shape equals base, and every checked static `prefix + update` fits base.
- Every logical output coordinate receives exactly one represented-bit write: update at the
  unique target coordinate and base everywhere else. Base and update remain unchanged. Scalar
  replacement, empty targets, and zero-element outputs are explicit tests.
- FLOAT64/FLOAT32 signed zero and adversarial NaN bits, opaque BFLOAT16 bits, signed integral edge
  bits, and canonical BOOL values agree bit-for-bit between generated output and independent
  reference for both attribute forms.
- Compact geometry is O(rank plus unique boundaries), contains no per-output selector/address
  table, and snapshots all arrays. Arbitrary range starts are decomposed cold; generated hot loops
  use primitive sequence cursors and coordinate carry/reset with no per-element division, modulo,
  allocation, reflection, map lookup, Model access, or operation/attribute dispatch.
- Base/update occurrences are mapped to unique declarations in first-occurrence order, including
  `[0, 0]` when both semantic inputs use the same `ValueId`. Boundary declarations are exactly
  unique inputs followed by output.
- Input read-zero strides, offset/strided input layouts, injective offset/strided output layouts,
  heap arrays, `MemorySegment`, mixed carriers, arbitrary output ranges, and parallel chunks work
  without changing represented values.
- Cold binding rejects wrong type/carrier, size, alignment, access, lifetime, noncanonical BOOL,
  output non-injectivity, and every output/input accessed-span overlap before any output write.
  Input/input overlap is permitted because both inputs are read-only.
- Analysis selects one unit, scalar or parallel-scalar compute, no vector body, no materialization,
  no workspace, exact buffer declarations, and one artifact. Finalization performs exactly one
  artifact lookup after assignments and any worker checks.
- Structural identity includes schema 15, movement family, output rank, occurrence-to-boundary
  map, exact ordered data types, structural access plans, carrier pattern, and scalar compute.
  It excludes concrete extents, start/length/axis/step values, attribute form, prefix/target
  extents, stride magnitudes, offsets, addresses, values, slots, range/chunk/worker facts, and run
  identity. Compatible signed-slice and target-relative occurrences may share bytes only when all
  included facts match.
- Schema advances exactly 14 to 15 because the emitter gains a new family body. A schema-14
  envelope is an incompatible miss with deterministic regeneration; no migration or legacy
  reader is added.
- Existing pointwise, affine, movement/window, indexing, carrier, parallel, persistence,
  validation, and failure tests remain green.
- The clean implementation handoff stabilizes Java/tests first. A distinct clean documentation
  handoff then finalizes every affected Javadoc/package/doc/status contract and records reasoned
  no-change conclusions without repeating successful Java tests unless it changes executable
  Java or identifies concrete stale evidence.
- CPU 0006B becomes `Complete` only after all implementation, documentation, scope, and validation
  gates pass. CPU 0006B1 and 0006B2 remain `Draft` without task files; CPU 0006C and later rows are
  not advanced.
- No forbidden path, shared/public/build/architecture/conformance/integration/vendor/later-family
  change is present, and the final changed-path set stays within the exact 28-path ceiling.

## Tests / validation

Clean implementation-focused matrix:

```bash
./gradlew :backends:cpu:test --tests 'io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIrTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuDataMovementGeneratedKernelTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest'
./gradlew :backends:cpu:test
```

The focused matrix must cover both exact attribute signatures; every represented type; positive,
negative, non-unit, and length-one `Long.MIN_VALUE` steps; multiple selected axes; empty lists;
zero selected length; zero-element output; rank-zero scalar replacement; target-relative prefix
placement; exact-boundary and invalid-bound failures; arbitrary resolved layouts; read-zero
strides; injective strided output; unique/deduplicated inputs; heap/segment/mixed carriers;
arbitrary flattened ranges; parallel chunks; canonical/noncanonical BOOL; output/input overlap;
one unit/declarations/no workspace/no materialization/one artifact; reference parity; hot bytecode
dependency inspection; and schema/key inclusion/exclusion/miss behavior.

Run the final CPU module suite exactly once after executable Java stabilizes. Record suite, test,
skip, failure, and error totals plus exact Java 26 Runtime/VM identity. Do not rerun that successful
suite in the documentation context unless executable Java changes or a concrete stale-evidence
risk is documented.

The distinct clean documentation-focused handoff then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

It must also validate canonical task headings; repository-local Markdown paths and heading
anchors; balanced fences; final newlines; trailing whitespace; exact allowlist and 28-path ceiling;
schema/status/dependency synchronization; current Model names and both `SLICE_UPDATE` signatures;
absence of 0006B1/0006B2 task files; and absence of forbidden paths or Java changes after the
recorded final suite. If `/tmp/validate_synaptik_markdown.py` is absent, create an equivalent
validator outside the repository.

Repository-wide tests are deferred to CPU 0009 and CI because this is one CPU-private capability
with no shared, dependency, build, or architecture change. Architecture tests remain unchanged
because module boundaries and dependency rules do not change. Backend-conformance and integration
tests remain unchanged because current public Engine composition/output access is outside this
slice and the focused generated-versus-independent-reference matrix owns its CPU-private evidence.
If implementation makes any of those conclusions false, stop and replan before changing another
validation tier.

## Dependencies

- [CPU 0006A2](0006a2-portable-gather-and-one-hot-indexing.md) is `Complete` and supplies the
  current schema-14 generated artifact, compact geometry, typed carriers, complete pre-write
  binding checks, one-unit preparation/finalization, scalar/parallel-scalar, and independent
  reference seams.
- [CPU 0006A](0006a-portable-pad-tile-and-tensor-composition-movement.md) and
  [CPU 0006A1](0006a1-portable-static-window-extraction.md) are `Complete` and supply the exact
  `CpuDataMovementIr`, compact range geometry, generated emitter, represented-bit, and movement
  lifecycle contracts extended here.
- [CPU 0006](0006-portable-static-affine-views-and-boundary-materialization.md) is `Complete` and
  supplies current static `SLICE`/target-relative crop interpretation evidence, resolved-layout
  validation, and boundary materialization/overlap contracts.
- [Model 0023C](../../../modules/model/tasks/0023c-slice-update-and-target-relative-crop.md) is
  `Complete` and owns signed finite-coordinate functional update.
- [Model 0025D](../../../modules/model/tasks/0025d-dynamic-extent-slice-extraction-and-symbolic-slice-placement.md)
  is `Complete` and owns target-relative `SLICE_UPDATE + CropToShapeAttrs` construction.
- Current `SliceKind`, `SliceAttrs`, `CropToShapeAttrs`, `TensorSlicePlacementExpressions`,
  `OperationSignature`, `Shape`, `LayoutDescriptor`, data-type, CPU carrier, worker, artifact,
  preparation, finalization, binding, and reference contracts.

## Follow-up tasks

- CPU 0006B1 — `Draft`, depends on CPU 0006B, and owns the separate detailed planning frontier for
  `SCATTER_ELEMENTS`, Gather-compatible `SCATTER_ADD`, and `SCATTER_ND`. It must resolve exact
  index bounds, pre-write validation, `NONE` duplicate rejection, base participation, configurable
  reductions, represented-value conformance, deterministic execution, resources, and safe split
  behavior before a task file can become `Ready`.
- CPU 0006B2 — `Draft`, depends on CPU 0006B1, and owns the later separate detailed frontier for
  `FOLD_AXIS` and `FOLD2D`. It must resolve zero initialization, uncovered positions, overlap-add,
  numeric accumulation, deterministic execution, NCHW column mapping, padding/ceil-tail exclusion,
  resources, and safe split behavior before a task file can become `Ready`.
- CPU 0006C — remains `Draft`, now depends on CPU 0006B2, and owns stable ordering/selection.
- CPU 0008A remains `Draft` for general partition-DAG decomposition/fusion. CPU 0009 remains the
  repository/conformance closure checkpoint.

Do not create any follow-up specification or advance a later status during this task.

## Architecture impact

Expected impact: None.

The architecture already assigns operation meaning to Model, backend-owned lowering/strategy and
resource declaration to CPU analysis, assignment to shared Prepare, post-assignment artifact
realization to CPU finalization, and direct prepared invocation to Runtime. A nested movement
variant and compact geometry are CPU-private implementation details within those boundaries.

If implementation requires a change to `ARCHITECTURE.md`, another architecture document, an ADR,
an architecture test, a dependency edge, or a shared/public lifecycle contract, stop and report
the exact conflict rather than implementing it under this task.

## Implementation prompt

Use this prompt for the mandatory clean implementation handoff:

```text
You are the isolated implementation agent for Synaptik CPU task 0006B. Do not commit or push, and
do not use any GSD skill or workflow.

Read in full AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md, the
focused architecture documents referenced by this task, docs/planning/planning-guide.md,
docs/planning/roadmap.md, the CPU master plan, this task specification, its completed CPU/Model
dependencies, and the current Model and CPU contracts it names. Implement this specification
exactly within its 28-path allowlist and run its focused command followed by the sole final CPU
module suite. Stop and report any architecture conflict, unresolved semantic decision,
forbidden-path need, or scope-ceiling breach instead of inventing or broadening the design.

After executable Java and tests stabilize, hand the same uncommitted diff and recorded evidence
to a distinct clean documentation-focused context. That context must follow the documentation
rules and matching General, API/Javadoc, Backend Guide, Planning, and Example profiles; finalize
affected Javadocs, package summaries, explanatory docs, task evidence, and synchronized status;
run the documentation and planning gates in this specification; and avoid repeating successful
Java tests absent executable changes or a recorded stale-evidence reason. Mark the task Complete
only after both clean handoffs and every acceptance gate pass, using the AGENTS.md completion
summary format.
```

## Local decisions

- Split the former broad 0006B into ordered 0006B slice update, 0006B1 functional scatter, and
  0006B2 overlap fold. Only 0006B is detailed and is now `Complete`.
- Extend the existing value-blind movement family. Do not create a functional-update top-level IR,
  lowerer, emitter, package, or route for one current operation.
- Normalize both attribute forms to one rank-sized `(start, length, signed step)` sequence per
  output axis. This makes attribute form and concrete placement geometry cold facts.
- Iterate the result domain once and select exactly one source per output coordinate. Do not copy
  the whole base and then patch the update region; the one-pass form preserves independent output
  ranges and current parallel-scalar orchestration without barriers.
- Seed signed-sequence state cold and advance it with primitive hot cursor carry/reset. A
  length-one `Long.MIN_VALUE` step must not be negated or assigned a synthetic absolute stride.
- Support all six exact represented types. This is copying/selection only, not BFLOAT16 arithmetic,
  BOOL truth-table work, conversion, or promotion.
- Declare each unique input once, then output; permit input/input alias; reject output/input
  overlap before writes. No workspace or materialization is selected.
- Structural identity includes code-shaping rank/map/access/type/carrier facts and excludes exact
  placement/extent/layout magnitudes. Both attribute forms may reuse one artifact when included
  facts match.
- Advance to schema 15 because a new emitted family body changes bytecode. There is no migration.
- Use focused CPU-private generated/reference evidence now. Defer repo-wide and public end-to-end
  evidence to their recorded checkpoints.

## Known limitations

- Only one fully static, resolved-layout `SLICE_UPDATE` occurrence is executable. Dynamic or
  symbolic fit obligations remain unsupported.
- The output must be a distinct injective materialization and must not overlap either input.
- Inputs may use positive or zero layout strides; negative storage strides remain outside current
  layout contracts. Signed slice steps affect logical placement, not storage-stride support.
- The generated movement realization is scalar or parallel-scalar only. No vector gather/mask,
  native provider, tuning, benchmark, or performance result is added.
- No scatter bounds/duplicate/reduction execution and no fold initialization/overlap accumulation
  exists until separately planned CPU 0006B1 and 0006B2.
- Tensor expressions normally leave layout unresolved. Existing lifecycle work must provide exact
  static descriptors before CPU capability can truthfully admit the occurrence.

## Validation evidence

Planning context `/root/cpu_frontier_planning` read the governing architecture and focused
architecture documents, documentation rules and required profiles, planning guide/roadmap, CPU
master plan, completed predecessor CPU tasks, required completed Model tasks, current Model source
and Javadocs, and current affected/review-only CPU production and test contracts. It found no
architecture conflict, missing shared contract, or unresolved Model semantic decision.

Planning-stage validation passed after the three planning paths were finalized: the full diff and
exact path inventory were inspected; the task-template headings, one-Ready-frontier status,
0006B/0006B1/0006B2/0006C dependency chain, schema-14-to-15 boundary, and unchanged Complete
0006A2 row were checked; an external validator for repository-local links and anchors, balanced
fences, final newlines, and trailing whitespace passed all three Markdown files;
`git diff --check` passed; and the inventory
contained exactly this new task, the CPU master plan, and the roadmap. No 0006B1 or 0006B2 task
file and no source, test, architecture, guide, glossary, build, dependency, or other forbidden
path is present. No Java test, Javadoc, or build command ran for this planning-only change.

Clean implementation context `019ff156-d822-7442-b8a6-20f7e5e55d4d` stabilized the allowlisted
executable Java and focused tests.
After the first documentation pass, mandatory independent audit/fix context
`019ff188-85cb-7d50-be34-c5c3038b5634` found no production defect and changed tests only. Its
final exact ten-class command passed 10 suites and 91 tests with 0 skipped, 0 failures, and 0
errors. Its final unfiltered `./gradlew :backends:cpu:test` run passed 35 suites and 198 tests with
1 skipped, 0 failures, and 0 errors on Oracle OpenJDK Runtime Environment `26.0.1+8-34`, OpenJDK
64-Bit Server VM `26.0.1+8-34`. No repository-wide, architecture, backend-conformance,
integration, or post-audit Javadoc command ran; their documented deferrals remain true.

The audit added direct SLICE_UPDATE evidence for generated arbitrary layouts, ranges, and multiple
axes; parallel mixed heap/MemorySegment execution; all-MemorySegment execution; all-heap BOOL
execution plus complete binding, overlap, and input-alias rules; one unit, exact declarations,
deduplication, scalar/parallel choice, zero output, no workspace, and no materialization during
preparation; exact assignments, one artifact, and no workspace during finalization; exact and
invalid bounds; non-injective output rejection; and both attribute forms with multiple axes in the
independent reference. These checks now live across all ten authorized test owners rather than
relying on generic movement, preparation, binding, and regression coverage alone.

Clean documentation context `019ff15f-a57f-7b62-aadc-40d3335d4a96` independently reviewed the
complete source/test diff and finalized every affected Java contract, six package summaries, the
CPU guide, glossary, task record, CPU master plan, and roadmap. The guide and glossary now explain
both attribute forms, copy-base-then-replace selected-position semantics, the one-pass generated
realization, signed/non-unit steps, all six represented types, range/carrier/alias rules, schema
15, and the explicit Draft 0006B1/0006B2 boundary without broadening Model or shared semantics.

The first CPU Javadoc build succeeded but reported the two expected incubating-Vector warnings
and three missing-comment warnings on the new defensive-copy slice-geometry accessors. After a
Javadoc-only correction, the second build succeeded with only the two expected incubating-Vector
warnings. A final completeness review then corrected only the capability-provider overview, and
the final post-stability build again succeeded with those same two expected warnings. Generated-
page inspection confirmed the rendered capability, schema, movement emitter/IR/lowering/reference,
slice-geometry accessor, and package-summary contracts. No Java test was rerun because this
context changed no executable statement or test.

This final clean documentation synchronization reviewed the later test-only diff against the
final production contracts, Javadocs, six package summaries, CPU guide, and glossary. The audit
changed no production Java or Java contract after the previous Javadoc pass, so those artifacts
remain accurate and no Java test or Javadoc command was repeated. Only this task, the CPU master
plan, and the roadmap required evidence synchronization.

## Implementation notes

Implementation context `019ff156-d822-7442-b8a6-20f7e5e55d4d` added nested
`CpuDataMovementIr.SliceUpdatePlan` and nested
`CpuNonAffineMovementLowering.Geometry.SliceUpdate`. Lowering normalizes both attribute forms to
rank-sized start, length, and signed-step arrays while unique boundaries retain first-occurrence
order. Range packing cold-seeds ascending target coordinates and update ordinals. Generated code
selects base or update in one output-domain pass and advances primitive per-axis target/ordinal
cursors with the output odometer; the legal length-one `Long.MIN_VALUE` step is exhausted before
any negation or subtraction. The independent reference derives the finite mapping directly.

Production implementation changes are confined to seven allowlisted production paths. The final
test diff uses all ten authorized test owners. Direct focused tests cover capability and exact
bounds, lowering and non-injective output rejection, structural identity, deduplicated inputs,
both attribute forms, multiple axes, arbitrary layouts and ranges, all six represented types,
scalar replacement, empty targets and outputs, legal length-one `Long.MIN_VALUE`, heap/segment/
mixed carriers, parallel execution, BOOL validation, alias and overlap rules, exact preparation
resources, one-artifact finalization, reference parity, and schema 15/current-only misses. No
review-only shared, Runtime, Prepare, Model, build, dependency, architecture, conformance,
integration, or vendor path changed.

## Completion summary

- Completed changes: implemented and documented portable functional SLICE_UPDATE for both current
  attribute forms through the existing represented-bit movement route, with truthful capability,
  lowering, compact geometry, generated/reference realization, schema 15, and synchronized
  planning status.
- Files changed or created: the final combined change contains exactly 28 allowlisted paths—13
  CPU production/package paths, ten CPU test paths, two explanatory documents, and three
  planning/status documents. This final synchronization changed exactly this task, the CPU master
  plan, and the roadmap; it changed no Java source, test, guide, or glossary path.
- Tests and validation: reused audit/fix context
  `019ff188-85cb-7d50-be34-c5c3038b5634`'s exact focused 10-suite/91-test pass and final CPU
  35-suite/198-test pass with one expected skip and zero failures/errors; no Java test or Javadoc
  was rerun after the test-only audit. The prior final CPU Javadoc and generated-page inspection
  remain valid. Markdown links/anchors/
  fences/newlines/whitespace, exact allowlist/ceiling, schema/status/dependency/current-versus-
  planned checks, absent 0006B1/0006B2 specs, and `git diff --check` passed.
- Documentation impact: affected Javadocs and package summaries, CPU guide, glossary, task,
  master plan, and roadmap are current. The final test-only diff required evidence changes only in
  the three planning/status documents. Review found no change necessary to
  `CpuPortableKernelIr`, `CpuKernelSpecialization`, `CpuClassFileKernelGenerator`,
  `CpuCarrierEmitter`, the preparation plan/preparer/finalizer, prepared executable, portable
  route plan, memory/worker contracts, any affected Javadoc or package summary, the CPU guide, or
  the glossary because the audit changed tests only and exposed no contract or behavior defect.
- Architecture impact: None. Architecture documents/tests, dependencies/build, shared/public
  APIs, other modules, backend conformance, and integration tests remain unchanged because the
  implementation consumes established Model and staged CPU-private contracts.
- Documentation-agent review: clean context `019ff15f-a57f-7b62-aadc-40d3335d4a96` performed the
  original finalization; this mandatory final clean synchronization incorporated audit/fix
  context `019ff188-85cb-7d50-be34-c5c3038b5634` without changing Java source or tests. This
  synchronization context's ID was not exposed to the agent.
- Unresolved issues: None.
- Required follow-up: None for CPU 0006B. CPU 0006B1 functional scatter and CPU 0006B2 overlap
  fold remain ordered Draft work without detailed task specifications.

Status: Complete
