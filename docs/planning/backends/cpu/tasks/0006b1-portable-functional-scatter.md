# Task 0006B1: Portable Functional Scatter

## Status

Complete

## Goal

Add the next executable CPU frontier after completed CPU 0006B: exactly one fully static,
resolved-layout current Model occurrence of `SCATTER_ELEMENTS`, Gather-compatible `SCATTER_ADD`,
or `SCATTER_ND` through the existing whole-partition portable generated-kernel architecture.

The result is functional and has the exact base Shape and type. CPU validates the complete logical
index domain, and `NONE` target uniqueness where applicable, on the invoking thread before any
output write or worker submission. Successful execution writes each result coordinate once from
its base and complete target group. Output-coordinate ownership makes scalar and parallel-scalar
execution deterministic and race-free. A declared per-range scratch workspace is used only for
floating `MUL`, whose exact abstract product must round once to the unchanged format; all other
rows require no workspace.

This task adds one CPU-private scatter IR/lowering/emitter family. It does not extend Model,
Compiler, shared semantics, Runtime interpretation, native routes, or the later overlap-fold
frontier.

## Scope

- Admit exactly one CPU-owned node with ordered logical inputs `[data, indices, updates]`, one
  output, and one of these current identities:
  - `AxisScatterKind.SCATTER_ELEMENTS + ScatterElementsAttrs(axis, reduction)`;
  - `AxisScatterKind.SCATTER_ADD + IndexAxisAttrs(axis)`, with intrinsic `ADD`; or
  - `ScatterNdKind.SCATTER_ND + ScatterNdAttrs(batchDimensions, reduction)`.
- Require fully static Shapes, resolved non-negative-offset/non-negative-stride layouts, exact
  current Model signature and Shape rules, and one distinct writable injective output.
- Support both exact INT32 and INT64 index carriers through heap arrays, native-order
  `MemorySegment`, and compatible mixed carrier patterns.
- Support FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and canonical BOOL for `NONE`; support the five
  non-BOOL types for `ADD`, `MUL`, `MIN`, and `MAX`. `SCATTER_ADD` is numeric and fixed-add only.
- Add distinct `CpuScatterIr`, `CpuScatterLowering`, and `CpuScatterEmitter` owners. Scatter is not
  a gather/indexing extension: it has a functional base, three semantic inputs, multi-pass
  pre-write validation, target grouping, configurable reductions, and optional exact-product
  scratch.
- Preserve semantic input occurrences while declaring each distinct input `ValueId` once in
  first-occurrence order, followed by one separate output. Retain the three-entry occurrence map
  so equal data/indices/updates graph values deduplicate without changing roles.
- Validate all index scalars in deterministic logical row-major order before duplicate checking,
  generated invocation, worker submission, or output mutation. Negative values are invalid; no
  normalization, wrapping, clamping, ignoring, padding row, or default target exists.
- For `SCATTER_ELEMENTS + NONE` and `SCATTER_ND + NONE`, perform a second complete deterministic
  allocation-free uniqueness pass after all bounds succeed. Reject the first later logical update
  or tuple that repeats an earlier complete target. No first-write or last-write rule exists.
- Generate an output-domain scalar body. Each disjoint output range loads the exact base, scans
  logical update contributions in row-major order, selects those targeting that output coordinate,
  and writes once. The generated writer contains no bounds or duplicate-failure branch because
  complete validation already succeeded.
- Use scalar or parallel-scalar orchestration only. Parallel chunks own disjoint output
  coordinates, share read-only inputs, use disjoint scratch slices when required, and use no
  atomics, locks, barriers, cross-range partials, or reduction merge.
- Declare one workspace only for floating `MUL` with a non-empty possible contribution domain and
  non-empty output. It contains one fixed-capacity exact-product accumulator slice per selected
  range. All other scatter plans declare no workspace. No input materialization is selected.
- Extend preparation, finalization, generated-entry signature, binding, and workspace validation
  only as needed to pass that exact scratch segment and one cold slice offset to each range.
- Advance generated compatibility from schema 15 to schema 16. Older entries are incompatible
  misses; there is no migration reader.
- Extend the independent scalar reference with a separately implemented scatter oracle. Its
  floating-product oracle may use `BigInteger` on the test/reference path; generated execution
  must use the declared primitive-limb workspace and may not delegate to the oracle.
- After Java and focused tests stabilize, hand the uncommitted diff to a distinct clean
  documentation-focused context to finalize Javadocs, affected package summaries, the CPU guide,
  glossary, task evidence, CPU master plan, and roadmap.

## Current Model contract and future CPU realization

### Shared functional contract

- All three operations consume exact ordered `[data, indices, updates]` inputs and produce a fresh
  value with the exact data Shape and type. CPU never mutates any input.
- Indices are exactly INT32 or INT64. Updates have exactly the data type. There is no promotion,
  broadcasting, conversion, saturation, alternate index carrier, or implicit zero base.
- `NONE` replaces an addressed target with its sole exact update and accepts all six represented
  types. Arithmetic reductions reject BOOL. Every non-replacement target group contains the base
  exactly once and every addressed update exactly once; duplicate positions remain distinct
  contributions.
- A result coordinate addressed by no update preserves the exact base representation. It is not
  combined with an identity, widened and narrowed, or canonicalized.
- Current source is authoritative over historical plans. In particular, current
  `AxisScatterKind` contains only `SCATTER_ELEMENTS` and Gather-compatible `SCATTER_ADD`.
  Historical `SCATTER_AXIS_ADD` and the superseded reduced-rank `SCATTER_ADD` shape must not be
  treated as current by capability, lowering, tests, explanatory documentation, or cache identity;
  preserved historical planning records do not reopen either contract.

### `SCATTER_ELEMENTS`

Let data rank be `R`, normalized selected axis be `a`, and let indices and updates share Shape
`I`. Model requires `R > 0`, `rank(I) = R`, and `I[d] == data[d]` for every `d != a`; `I[a]` may
differ. For update coordinate `u`, read `v = indices[u]` and address:

```text
target(u)[d] = u[d]  when d != a
target(u)[a] = v
0 <= v < data[a]
```

The validation ordinal is the flattened row-major ordinal in the complete indices/updates Shape.
For `NONE`, uniqueness means equality of the complete target coordinate above, not merely equality
of index values. Because non-axis coordinates participate, equal index values in different
non-axis positions need not be duplicates.

### Gather-compatible `SCATTER_ADD`

Let normalized data axis be `a` and indices Shape be `I`. Updates have exactly ordinary Gather's
result Shape:

```text
updates.shape = data.shape[0:a] + I + data.shape[a+1:R]
```

For update coordinate `[before..., i..., after...]`, read `v = indices[i...]` and address
`[before..., v, after...]`, requiring `0 <= v < data[a]`. Scalar indices insert no dimensions.
Every result target begins with its base and adds all contributions, including duplicates.
`SCATTER_ADD` has no `NONE` mode and therefore no duplicate rejection.

Bounds validation visits the indices tensor once in its own logical row-major order even when the
same index value is reused across many data-prefix/data-suffix coordinates. Generated output may
reload validated index values while matching contributions.

### `SCATTER_ND`

Let data rank be `R`, indices rank be `Q`, shared batch count be `B`, and tuple depth be the static
positive final indices extent `K`. Model requires `B < min(R, Q)`, `K <= R - B`, structurally
equal data/indices prefixes `[0, B)`, and:

```text
updates.shape = indices.shape[0:Q-1] + data.shape[B+K:R]
```

For one tuple-prefix coordinate `t`, tuple component `k` addresses data axis `B + k` and must
satisfy:

```text
0 <= indices[t..., k] < data[B + k]
```

The complete target is the shared batch prefix, the `K` indexed components, and the update suffix
coordinate. Each tuple contributes its entire suffix slice scalar by scalar. Bounds validation
uses flattened row-major scalar order, including tuple components. `NONE` uniqueness is equality
of the target tuple within the same shared-batch prefix; equal tuples in different batches are not
duplicates. A duplicate tuple is invalid even when the suffix slice is empty, because Model makes
tuple uniqueness the replacement precondition.

### Reduction realization

For every result coordinate `c`, define `U(c)` as the logical multiset of addressed update scalars.

- `NONE`: `U(c)` has size zero or one after validation. Size zero returns exact base bits; size one
  returns exact update bits. Base does not participate at an addressed target.
- `ADD`: include base once and every member of `U(c)` once. INT32 and INT64 use fixed-width
  two's-complement modular addition. FLOAT32 and FLOAT64 use deterministic row-major sequential
  same-format addition for this CPU route. BFLOAT16 expands each exact represented operand to
  binary32, adds it to the represented current BFLOAT16 accumulator, and rounds back to BFLOAT16
  after every logical contribution.
  These are CPU choices within Model's reassociation permission; scalar and parallel-scalar must
  agree bitwise for the same bound values and layouts.
- `MUL`: INT32 and INT64 use fixed-width modular multiplication. Floating rows implement Model's
  one abstract unchanged-format product whenever `U(c)` is non-empty: include base and every
  update once, classify every represented factor, propagate NaN, make zero with infinity NaN,
  otherwise derive infinity or zero with complete negative-factor sign parity, and for all-finite
  non-zero factors multiply exact significands and exponent state then round once using
  round-to-nearest, ties-to-even to FLOAT64, FLOAT32, or BFLOAT16. Overflow, normal, subnormal,
  underflow, and signed-zero results follow that target. The result promises no NaN payload,
  source, signaling state, or sign. Empty `U(c)` copies exact base bits without entering the
  accumulator.
- `MIN` and `MAX`: include base once and every member of `U(c)` once. INT32/INT64 use signed order.
  Floating rows propagate any NaN; ordinary represented numeric order includes infinities; if
  both zero signs occur, `MIN` yields negative zero and `MAX` yields positive zero. No NaN payload,
  source, signaling state, or sign is promised. Empty `U(c)` copies exact base bits.

The output-domain scan makes `MUL`, `MIN`, and `MAX` independent of physical layout, worker
scheduling, atomics, and tree shape, as Model requires. The selected deterministic `ADD` order is
a CPU realization, not a new Model guarantee.

### Exact floating-product workspace

Floating `MUL` uses one declared run-owned `CpuContiguousWorkspace`, generalized from its current
materialization-only wording to generic aligned CPU scratch. The plan computes an exact
per-range slice size from the maximum number of factors that can address one result coordinate,
including the base, and the precision of the unchanged result format: 53 bits for FLOAT64, 24 for
FLOAT32, and 8 for BFLOAT16. The slice contains fixed metadata plus enough unsigned 64-bit limbs
for the checked worst-case significand product. Slice size and `selectedRangeCount * sliceSize`
use exact arithmetic, align to eight bytes, and fail closed on overflow.

Let `m` be the maximum updates in one target group. If the complete updates domain is empty,
`m = 0`; otherwise `m = indicesExtent[a]` for `SCATTER_ELEMENTS`, the checked product of every
indices extent for Gather-compatible `SCATTER_ADD`, or the checked product of indices extents
`[B, Q - 1)` for one `SCATTER_ND` batch. The maximum factor count is `1 + m`. Scratch format
version one reserves three eight-byte words for flags/sign/special-value state, saturated exponent
state, and used-limb count, followed by
`ceil(precision * (1 + m) / 64)` little-endian unsigned 64-bit limbs. The per-range slice size is
therefore `24 + 8 * limbCount` bytes; every multiplication/addition used to derive these values is
checked.

One workspace declaration with requirement ID zero covers all slices. A scalar call uses slice
zero; parallel range `i` uses slice `i`. The generated entry adds a `MemorySegment` scratch
parameter only for the floating-`MUL` specialization, and cold packed geometry carries its slice
offset/capacity. Each output resets and reuses its range's limbs. Exponent state may saturate only
beyond proven final overflow/underflow decision thresholds; sign and special-value state remain
exact. The hot path performs no `BigInteger`, collection, array, cursor, table, per-factor, or
per-output allocation.

The workspace is absent when no generated floating-product calculation can occur, including an
empty output or empty contribution domain. Scratch presence/format is explicit specialization
compatibility metadata. It is mutually exclusive with materialization, which scatter never
selects.

## Validation order and failure-before-write contract

Cold analysis and lowering first reject unsupported kind/attrs/signature/type/Shape/static/layout
or arithmetic overflow. Finalization verifies every exact buffer/workspace assignment and worker
requirement before its sole artifact lookup. Binding then checks, in order:

1. representation count, exact type/carrier, byte size, alignment, lifetime/accessibility, output
   writability, and exact scratch size/alignment/accessibility when present;
2. full referenced spans, injective output, and output non-overlap with every unique input;
3. canonical byte `0`/`1` for every logical BOOL data/update input occurrence;
4. complete logical row-major bounds validation of every index scalar;
5. for `NONE`, complete logical row-major target-uniqueness validation.

Only after all five stages succeed may generated execution begin or any worker be submitted. Any
failure leaves every output byte unchanged. Bounds always precede duplicates, so mixed-invalid
input reports the first out-of-bounds scalar rather than a duplicate.

Exact runtime failures are:

```text
SCATTER_ELEMENTS index at logical position <p> for data axis <a> is out of bounds: value=<v>, extent=<e>
SCATTER_ADD index at logical position <p> for data axis <a> is out of bounds: value=<v>, extent=<e>
SCATTER_ND index at logical position <p> for data axis <a> is out of bounds: value=<v>, extent=<e>
SCATTER_ELEMENTS duplicate target at logical update position <later>; first addressed at logical update position <earlier>
SCATTER_ND duplicate target tuple at logical tuple position <later>; first addressed at logical tuple position <earlier>
```

Bounds throw `IndexOutOfBoundsException`; duplicates throw `IllegalArgumentException`. Duplicate
search compares each later logical update/tuple with earlier positions in ascending order, so the
first later duplicate and its earliest predecessor are stable. It uses reusable binding-created
primitive coordinate state and direct carrier loads, not a hash table or execute-time allocation.

Validation and output have independent zero domains. An empty index domain succeeds without a
load. A zero selected extent rejects every encountered index. A zero output caused by another
extent still validates all indices and `NONE` tuples, then makes no generated call and submits no
worker work. `SCATTER_ELEMENTS` with an empty selected update extent, or any valid empty
contribution domain, copies base when output is non-empty. `SCATTER_ND + NONE` still checks tuple
uniqueness when its suffix slice is empty.

## Layouts, carriers, ranges, aliasing, and strategy

- Inputs may have resolved arbitrary non-negative offsets and strides, including read-zero
  strides and repeated physical reads. Output must have a resolved injective layout.
- Heap carriers are exactly `double[]`, `float[]`, raw BFLOAT16 `short[]`, `int[]`, `long[]`, and
  canonical BOOL `byte[]`; native carriers are native-order `MemorySegment`. Every ordered mixed
  pattern admitted by current carrier rules is supported.
- Input/input physical overlap is allowed, including exact `ValueId` deduplication. Output may not
  overlap the referenced span of any unique input, even though output-domain writing would often
  appear safe; this preserves the current functional boundary and prevents parallel read/write
  races. Empty referenced spans follow the current non-overlap convention.
- Logical row-major order is defined by Shape coordinates and is independent of physical address
  order, strides, aliases, or carrier form.
- Each generated range owns `[start, end)` output ordinals. Cold binding allocates and packs one
  primitive geometry array per selected call, seeds output coordinates, update-scan cursors, and
  optional scratch offset, and reuses it for every invocation. The generated hot loop advances
  odometers without division/modulo, Model inspection, semantic lookup, or allocation.
- Scalar preference selects scalar. Vector preference deliberately falls back to scalar compute.
  Existing cold policy may select parallel orchestration by output count; zero output selects no
  call. A range never writes outside its output interval and never shares mutable scratch with
  another range.

## Resources, preparation, finalization, cache, and schema

- Lowering produces one unit, one `CpuScatterIr`, one `CpuScatterLowering.Geometry`, unique input
  boundaries in semantic first-use order, then one distinct output, and output extents/count.
- Analysis declares exactly two through four buffers depending on input `ValueId` deduplication.
  Inputs are read-only and output is write-only. No scatter materialization is allowed.
- Non-floating-`MUL` plans declare zero workspaces. An eligible floating-`MUL` plan declares the
  single exact scratch workspace described above. No resource is inferred or added after shared
  assignment.
- `CpuPartitionPreparationPlan` must distinguish materialization workspace use from scatter
  product scratch while retaining the current maximum of one workspace. Finalization verifies
  exact identity/size/alignment and one matching assignment before one artifact-store call.
- `CpuPreparedExecutable` retains only prepared scatter geometry, validation state, the optional
  workspace selection, direct buffers, one artifact, and optional borrowed workers. Runtime sees
  no Operation, node, attrs, reduction enum, Tensor, cache choice, or fallback.
- Structural IR/artifact identity includes scatter family, reduction, data/index types, ranks,
  occurrence map, normalized structural access forms, output store shape, and scratch signature/
  format. Specialization includes exact ordered carrier pattern and whether the generated entry
  accepts scratch. Axis, batch count, tuple depth, extents, offsets, stride magnitudes, element
  counts, range bounds/count, workspace byte count/offset, slots, addresses, run identity, and
  worker identity remain cold compatible facts.
- Schema becomes 16. Persistence treats every older schema as a miss and regenerates through the
  existing atomic protocol; no compatibility reader or migration is added.

## Out of scope

- `FOLD_AXIS`, `FOLD2D`, CPU 0006B2, stable ordering/top-K CPU 0006C, explicit-state random,
  native/vendor routes, tuning, benchmarking, or performance claims
- changing any Model operation kind, attrs, reduction, Tensor construction, Shape rule,
  represented-value meaning, Javadoc, test, or public API
- changing Compiler capture, canonicalization, autograd, publication, decomposition, or shared
  semantics; CPU consumes the already current scatter occurrences
- dynamic/symbolic CPU Shapes, unresolved layouts, negative storage strides, deferred layout
  binding, non-injective/in-place output, or output/input overlap
- negative-index normalization, ignored indices, clamping, padding targets, first/last duplicate
  replacement, atomic scatter, unordered reduction, update-centric parallelism, or partial output
- general multi-node indexing/scatter fusion, mixed-family partition DAGs, safe split fallback,
  multi-output units, a Runtime semantic interpreter, or a selectable reference fallback
- vector scatter/gather, SIMD reduction, relaxed numerics, cross-type conversion/promotion,
  FLOAT16, alternate BFLOAT16 storage, or BOOL arithmetic
- a per-index/per-target address table, hash-based duplicate table, hidden mutable global/thread-
  local state, undeclared scratch, execute-time `BigInteger`, or avoidable hot-path allocation
- `ARCHITECTURE.md`, architecture explanations, ADRs, architecture tests, shared Prepare/Runtime
  contracts, Gradle/dependencies, backend-conformance, integration, vendor code, or unrelated
  refactors
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
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Backend guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU 0005E represented types/carriers](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
- [CPU 0005F floating division/power](0005f-floating-division-and-exact-scalar-power-realization.md)
- [CPU 0005G represented extrema](0005g-extrema-clamp-tensor-power-and-logical-coverage.md)
- [CPU 0005H unary closure](0005h-portable-unary-transcendental-and-activation-closure.md)
- [CPU 0005I FLOAT32 vector parity](0005i-float32-vector-parity-and-vector-emission-boundary.md)
- [CPU 0005J bounded parity hardening](0005j-bounded-pointwise-coverage-and-parity-hardening.md)
- [CPU 0006A2 Gather and one-hot](0006a2-portable-gather-and-one-hot-indexing.md)
- [CPU 0006B functional slice update](0006b-portable-functional-slice-update.md)
- [Model 0018G axis-scatter semantics](../../../modules/model/tasks/0018g-axis-scatter-semantics.md)
- [Model 0018H axis-scatter expressions](../../../modules/model/tasks/0018h-axis-scatter-tensor-expressions.md)
- [Model 0018I Scatter-ND semantics](../../../modules/model/tasks/0018i-scatter-nd-semantics.md)
- [Model 0018J Scatter-ND expressions](../../../modules/model/tasks/0018j-scatter-nd-tensor-expression.md)
- [Model 0018O indexing normalization](../../../modules/model/tasks/0018o-indexing-taxonomy-and-unstack-normalization.md)
- [Model 0023B Gather-compatible Scatter Add](../../../modules/model/tasks/0023b-gather-compatible-scatter-add.md)
- [Model 0025C functional-scatter reductions](../../../modules/model/tasks/0025c-portable-functional-scatter-reduction-semantics.md)
- [Tensor API](../../../../api/tensor-api.md#axis-scatter-semantic-kinds-reduction-and-attributes)
- [Compile API scatter contract](../../../../api/compile-api.md)
- [Functional-scatter target group](../../../../glossary.md#functional-scatter-target-group)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative. Model owns scatter meaning. CPU owns exact occurrence
  capability, lowering, code-shaping identity, strategy, resource declarations, artifact
  realization, validation, and direct execution without changing shared meaning.
- Planning has already assigned the occurrence to CPU. Shared Prepare receives only exact opaque
  buffer/workspace declarations. CPU finalization verifies assignments and workers before its
  single artifact lookup and may not revise lowering, route, strategy, or resources.
- Runtime receives one prepared executable and direct resource selections. No Operation,
  `CompiledNode`, attrs, Tensor metadata, reduction selection, cache lookup, or fallback enters
  Runtime execution.
- The portable generated route remains the sole route. The scalar reference is test evidence only.
- All value-dependent failure occurs before output mutation. No worker may observe or expose a
  partially validated scatter.
- Exact product scratch is one explicitly declared backend-private workspace, not a new shared
  resource kind or implicit allocator. Its use must remain compatible with the current workspace
  lifecycle.
- If implementation needs a shared/public type, another module, dependency/build change,
  architecture rule, Runtime semantic branch, second workspace, non-injective/in-place output, or
  any semantic assumption not fixed here, stop and report the conflict rather than broaden scope.

## Package impact and type placement

Existing packages used:

- `backend.cpu` for truthful capability.
- `internal.ir` for the closed scatter structural identity.
- `internal.lowering` for one-node semantic revalidation and compact geometry.
- `internal.codegen.emit` for the generated bridge, output scan, typed carrier access, reductions,
  and primitive-limb exact product.
- `internal.prepare` and `internal.executable` for exact scratch declaration/selection, complete
  pre-write validation, cold range packing, and direct invocation.
- `internal.memory` for generic aligned run-owned CPU scratch.
- `internal.cache` and `internal.route.portable` for schema-16 identity and artifact realization.
- `internal.reference` for an independent scalar oracle.

Packages added, moved, or removed: none.

New top-level production types:

- `CpuScatterIr` in `internal.ir`, because scatter structural identity is not gather indexing or
  value-blind movement.
- `CpuScatterLowering` in `internal.lowering`, including sealed/nested Axis and ND cold geometry.
- `CpuScatterEmitter` in `internal.codegen.emit`, owning generated bridge and execution helpers.

Exact-product limb arithmetic remains a focused private/nested implementation under
`CpuScatterEmitter`; do not add a generic math utility or shared numerical abstraction.

## Affected files

Authorized CPU production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuScatterIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuScatterLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuContiguousWorkspace.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`

Authorized CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuShapePolymorphicArtifactTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuScatterIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuScatterLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Authorized explanatory and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0006b1-portable-functional-scatter.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized. Listed paths change only when their implementation contract,
Javadoc, package summary, regression ownership, or synchronized status is affected. Both clean
handoffs must record review-only no-change conclusions for unused authorized paths.

Review without modification:

- Model, Compiler, Tensor/API documentation, architecture, ADR, architecture-test,
  backend-conformance, integration, shared-module, Gradle/dependency, and vendor paths remain
  evidence only. Current API and compiler guides already state the Model scatter contracts and do
  not promise CPU coverage; change them only after revising this allowlist.
- Existing gather/indexing and movement IR/lowering/emitter source and tests remain unchanged.
  Scatter uses their lifecycle patterns without weakening or merging their contracts.

## Maximum scope

| Category | Maximum | Accounting |
|---|---:|---|
| CPU production/package | 26 | 3 new owners plus affected capability, generator, lifecycle, scratch, cache, route, reference, and package contracts |
| CPU tests | 12 | Focused new family plus existing lifecycle/cache/reference regression owners |
| Explanatory documentation | 2 | CPU guide and glossary |
| Planning/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **43** | **26 + 12 + 2 + 3** |

Do not spend unused scope on cleanup. Stop and request a revised planning decision before a 44th
path or any path outside the exact allowlist.

## Acceptance criteria

- Capability and lowering admit only the three exact current scatter forms and independently
  reject every wrong kind/attrs/input count/output count/type/index type/Shape/static/layout/
  output-injectivity combination.
- Current `SCATTER_ADD` uses the exact Gather-compatible updates Shape. No historical
  `SCATTER_AXIS_ADD` or superseded reduced-rank shape is implemented.
- Both index types, all permitted data types/reductions, scalar Shapes where legal, zero extents,
  non-dense/read-zero input layouts, injective output layouts, heap/segment/mixed carriers, and
  input deduplication are covered.
- Bounds and `NONE` duplicate validation follow the exact mappings, order, exception types, and
  messages above. Tests prefill output sentinels and prove every invalid case fails before any
  write or worker submission, including zero-output cases.
- `NONE` copies exact update bits at unique targets and exact base bits elsewhere. Duplicate
  element targets and duplicate ND tuples are rejected; same index/tuple in distinct non-target
  coordinates or batches remains valid.
- Every arithmetic target includes base once and every update once; duplicate contributions
  accumulate. Empty groups preserve exact base bits.
- INT32/INT64 ADD/MUL are modular; extrema use signed order. FLOAT64/FLOAT32/BFLOAT16 cover normal,
  subnormal, overflow/underflow, signed zero, infinity, NaN, ties-to-even, zero-times-infinity,
  negative-sign parity, and multiple-factor cases against the independent oracle.
- Floating `MUL` exactly matches the once-rounded abstract product and uses one exact declared
  per-range-sliced workspace with no hot-path allocation. Every other row has no workspace.
- Scalar and parallel-scalar produce identical results and failures. Parallel tests cover multiple
  chunks, mixed carriers, disjoint scratch slices, repeat invocation, and no races/atomics/merge.
- Output/input overlap is rejected before validation/writes; input/input overlap and exact input
  deduplication work. Inputs remain byte-for-byte unchanged.
- Arbitrary output ranges write only their interval. Packed geometry contains no per-index or
  per-output table, and generated execution performs no per-element allocation, Model dispatch,
  bounds validation, division, or modulo.
- Analysis declares exact buffers, optional workspace, one unit, one artifact, and correct
  scalar/parallel-scalar strategy. Finalization accepts only exact assignments and performs one
  schema-16 artifact lookup after assignment validation.
- Structural cache tests prove compatible extent/layout/range/workspace-size changes reuse bytes,
  while family/reduction/type/rank/access/carrier/scratch-signature changes separate artifacts;
  schema 15 is an incompatible miss.
- Independent differential tests do not call generated helpers or share the primitive-limb
  implementation. Expected represented values come from a separately implemented oracle.
- Every changed Java API/implementation contract has meaningful complete Javadoc; affected
  package summaries, CPU guide, glossary, task evidence, master plan, and roadmap are finalized in
  the mandatory clean documentation pass.
- Architecture contracts, Model/Compiler/shared source, dependencies, and later tasks remain
  unchanged. Any need to change them makes this task incomplete pending replanning.

## Tests and validation

Focused implementation command, run after production and focused tests stabilize:

```bash
./gradlew :backends:cpu:test --tests '*CpuCapabilityProviderTest' --tests '*CpuScatterIrTest' --tests '*CpuScatterLoweringTest' --tests '*CpuScatterGeneratedKernelTest' --tests '*CpuPartitionPreparerTest' --tests '*CpuPartitionFinalizerTest' --tests '*CpuPreparedExecutableTest' --tests '*CpuGeneratedKernelArtifactStoreTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuShapePolymorphicArtifactTest' --tests '*CpuReferenceDifferentialTest' --tests '*CpuInternalPackageInventoryTest'
```

Final affected-module validation, run once after the focused command passes:

```bash
./gradlew :backends:cpu:test
./gradlew :backends:cpu:javadoc
```

The clean documentation context runs link/anchor/fence/newline/trailing-whitespace checks for the
changed Markdown/Javadoc surface, inspects generated Javadoc pages, verifies exact changed paths
and status/dependency synchronization, then runs:

```bash
git diff --check
```

This is a normal single-module task. Repository-wide, architecture, backend-conformance, and
integration suites are deferred to CPU 0009 and CI because no dependency/module boundary, shared
contract, or current public Engine path changes. If implementation makes that conclusion false,
stop and replan before changing another tier. The documentation context does not repeat successful
Java tests unless it changes executable Java or records a concrete stale-evidence reason.

## Dependencies

- [CPU 0006B](0006b-portable-functional-slice-update.md) is `Complete` and supplies current
  schema-15 generated movement, output-domain range, alias, carrier, and lifecycle evidence.
- [CPU 0006A2](0006a2-portable-gather-and-one-hot-indexing.md) is `Complete` and supplies mixed
  typed indexing carriers, compact geometry, deterministic complete pre-write validation,
  scalar/parallel-scalar output generation, and independent reference seams.
- CPU 0005E through 0005J are `Complete` and supply represented carriers, exact signed integral
  arithmetic, floating classification/extrema/BFLOAT16 conventions, generated specialization,
  range strategy, cache, and parity foundations.
- [Model 0023B](../../../modules/model/tasks/0023b-gather-compatible-scatter-add.md) is `Complete`
  and owns the current fixed-add Gather-compatible shape.
- [Model 0025C](../../../modules/model/tasks/0025c-portable-functional-scatter-reduction-semantics.md)
  is `Complete` and owns current base/target-group, represented floating product/extrema, and
  order-independence semantics, superseding incomplete historical prose.
- Current Model scatter kinds/attrs/reduction, Tensor construction, operation signatures, Shape,
  layout, carrier, CPU preparation/finalization/executable, workspace, worker, artifact, and
  reference contracts named above.

## Follow-up tasks

- CPU 0006B2 remains `Draft`, depends on CPU 0006B1, and owns `FOLD_AXIS`/`FOLD2D` zero
  initialization, uncovered positions, overlap addition, NCHW column mapping, padding/ceil-tail
  exclusion, resources, and safe split behavior. Do not create its task file here.
- CPU 0006C remains `Draft`, depends on CPU 0006B2, and owns stable ordering/selection.
- Later ordering, explicit-state random, native, tuning, dynamic layout, and general partition-DAG
  work retain their existing roadmap ownership and status.

## Architecture impact

Expected impact: None.

The architecture already assigns semantic meaning to Model, CPU route/lowering/resource choices
to backend analysis, opaque assignment to Prepare, generated artifact realization to CPU
finalization, and direct invocation to Runtime. A CPU-private scatter family and one explicitly
declared workspace reuse those boundaries.

If implementation requires an architecture contract, focused architecture document, ADR,
architecture test, dependency edge, shared/public lifecycle contract, or another module to
change, stop and report the exact conflict rather than implement it under this task.

## Implementation prompt

Use this prompt for the mandatory clean implementation handoff:

```text
You are the isolated implementation agent for Synaptik CPU task 0006B1. Work on the current
uncommitted planning diff. Do not commit or push, and do not use any GSD skill or workflow.

Read in full AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md, the
focused architecture and documentation rules referenced by this task, planning guide/roadmap,
CPU master plan, this task specification, its completed CPU/Model dependencies, and every current
Model/CPU contract named here. Implement this specification exactly within its 43-path allowlist.
Current source wins over historical planning prose: implement only SCATTER_ELEMENTS,
Gather-compatible SCATTER_ADD, and SCATTER_ND.

Run the focused twelve-owner command followed once by the CPU module suite. Stop and report any
architecture conflict, unresolved semantic decision, forbidden-path need, scope-ceiling breach,
shared-contract need, or inability to implement exact floating MUL with declared allocation-free
hot-path scratch instead of inventing or broadening the design.

After executable Java and tests stabilize, hand the same uncommitted diff and evidence to a
distinct clean documentation-focused context. That context must follow General, API/Javadoc,
Backend Guide, Planning, and Example profiles; independently review/finalize affected Javadocs,
package summaries, CPU guide, glossary, task evidence, CPU master plan, and roadmap; run the
documentation/planning gates; and avoid repeating successful Java tests absent executable change
or a recorded stale-evidence reason. Mark Complete only after both clean handoffs and every gate
pass, using the AGENTS.md completion summary.
```

## Documentation prompt

Use this prompt after implementation and tests stabilize:

```text
You are the mandatory distinct clean documentation-focused agent for Synaptik CPU task 0006B1.
Work on the existing uncommitted implementation diff. Do not commit or push, do not use GSD, and
do not change executable Java except to report a blocking contract defect to the implementation
agent.

Read AGENTS.md, the documentation rules and General/API-Javadoc/Backend Guide/Planning/Example
profiles, architecture boundary, task 0006B1, final source/tests, current Model scatter contracts,
CPU guide, glossary, CPU master plan, and roadmap. Independently verify and finalize all affected
Javadocs and package summaries plus the CPU guide, glossary, task evidence/completion, master plan,
and roadmap. Explain current operations/types, exact bounds and NONE duplicate validation,
failure-before-write, base/target groups, represented reductions, exact floating-MUL scratch,
layouts/carriers/aliasing/ranges, scalar/parallel behavior, resources, and schema 16 without
claiming broader Model/Compiler/Runtime/native/dynamic support or performance.

Inspect generated Javadoc, links, anchors, fences, whitespace/newlines, exact changed paths,
status/dependencies, and git diff --check. Reuse recorded successful Java evidence unless this
pass changes executable Java or identifies a concrete stale-evidence reason. Record explicit
no-change conclusions for API/Compiler docs, architecture/tests, shared modules,
backend-conformance/integration, build/dependencies, and later tasks. Mark the task Complete only
when documentation and every acceptance gate pass.
```

## Local decisions

- Use a distinct scatter family rather than extending gather indexing or value-blind movement.
- Produce by disjoint output coordinates and scan logical contributions. This preserves functional
  base participation, one final write, deterministic parallelism, and zero atomic/merge state.
- Validate every bound first, then `NONE` uniqueness, before any generated work. Use deterministic
  allocation-free comparison rather than a target hash table.
- Use a deterministic CPU row-major order for ADD while preserving Model's broader reassociation
  permission. Do not export that order as Model semantics.
- Use explicitly declared per-range primitive-limb scratch for floating MUL so the abstract exact
  product rounds once without hot-path arbitrary-precision allocation.
- Permit input/input alias and ValueId deduplication; reject every output/input overlap.
- Keep extents/layout magnitudes/ranges/workspace sizes cold; include family/reduction/type/access/
  carrier/scratch signature in generated compatibility; advance to schema 16.
- Use a separately implemented arbitrary-precision reference oracle for differential evidence,
  never as Runtime fallback.

## Known limitations

- Only one fully static resolved-layout scatter occurrence is executable. Tensor expressions
  normally remain unresolved until existing lifecycle work supplies concrete descriptors.
- The output must be distinct, writable, injective, and physically non-overlapping with inputs.
- The algorithm favors semantic closure and deterministic independent output ranges; it makes no
  complexity or performance claim and may rescan the update domain for each output.
- Vector compute, update-centric atomics, native routes, tuning, dynamic binding, multi-node
  fusion, and safe split remain planned elsewhere.
- Floating ADD has a deterministic CPU realization but no Model bitwise-order promise. Floating
  MUL is once-rounded exact; MIN/MAX preserve only the Model-promised NaN and signed-zero result,
  not NaN payload/source/sign.
- Floating-MUL workspace can be large for large contribution groups. Exact checked declaration or
  shared allocation failure rejects preparation/run; there is no hidden heap fallback.

## Validation evidence

Planning context read the governing repository/architecture/planning contracts, CPU master plan
and predecessor tasks, current CPU capability/IR/lowering/preparation/finalization/generated/
execution/reference/cache owners, current Model scatter kinds/attrs/reduction/Tensor contracts,
completed Model correction/reduction tasks, and current API/CPU-guide/glossary statements. Current
source resolved the stale historical scatter taxonomy and Gather-compatible shape. No architecture
conflict or unresolved semantic decision remains.

Planning-stage validation passed on exactly this new task, the CPU master plan, and the roadmap.
Repository-local Markdown paths and heading anchors, balanced fences, final newlines, trailing
whitespace, canonical headings, exact changed-path inventory, the sole-Ready 0006B1 status,
0006B -> 0006B1 -> 0006B2 -> 0006C dependency chain, and `git diff --check` all passed. No Java,
test, build, architecture, API/guide/glossary, or later-task file changed, and no Java test or
Javadoc command ran for this planning-only change.

Implementation context `019ff230-109c-73a3-933f-611ee7f6143d` implemented the CPU-private
scatter family and its focused regression matrix. Independent audit/fix context
`019ff248-a9e4-7150-8fbb-db2730d7cc1b` reviewed the complete uncommitted implementation and
repaired four production defects plus one reference-oracle defect. After the final executable/test
edit, the prescribed twelve-owner gate passed 12 suites and 103 tests with 0 skipped, failures,
or errors. The sole final `./gradlew :backends:cpu:test` passed 38 suites and 230 tests with one
expected existing skip and 0 failures or errors on Java/OpenJDK 26.0.1+8-34 and Gradle 9.6.1.
No executable Java statement or test changed after that evidence.

The mandatory clean documentation-focused context ID was
`019ff4fb-94c7-7921-af13-28a7395c3ae7`.
This context independently reviewed the final production/test diff, current Model scatter
contracts and completed tasks, affected public/generated contract owners, package summaries, CPU
guide, glossary, and planning records under the General, API/Javadoc, Backend Guide, Planning, and
Example profiles. It finalized twelve Java/package documentation paths plus the two explanatory
and three planning/status documents. The first `./gradlew :backends:cpu:javadoc` run succeeded but
reported 23 actionable missing-parameter/description warnings plus the two expected incubating-
Vector-module warnings. Javadoc-only corrections were applied. The second run succeeded with only
the two incubating-module warnings. Generated pages for `CpuScatterIr`, `CpuScatterLowering`,
`CpuScatterEmitter`, `CpuPreparedExecutable`, `CpuKernelSpecialization`,
`CpuScalarReferenceKernel`, and all affected package summaries were inspected.

Final focused checks confirmed local Markdown file targets and heading anchors, balanced fences,
final newlines, trailing whitespace, schema 16/current-operation wording, absence of historical
`SCATTER_AXIS_ADD` as a current operation, synchronized 0006B1 `Complete` and next-frontier 0006B2
`Draft` status/dependencies, absence of an unauthorized 0006B2 task specification, exact allowlist
membership, a changed-path union no greater than 43, and `git diff --check`. Java tests were not
repeated because this context changed only Javadocs, package summaries, explanatory docs, and
planning records.

## Implementation notes

- Added one distinct `CpuScatterIr`/`CpuScatterLowering`/`CpuScatterEmitter` family and connected
  it to capability, preparation, finalization, generated invocation, execution binding, reference,
  and schema-16 artifact compatibility owners without changing shared modules.
- The independent audit corrected exponent clamping under opposing exponent cancellation, packed
  coordinate reuse across repeated partial ranges, base reads when data/output layouts differ,
  insufficient IR/geometry validation, and FLOAT32/BFLOAT16 MIN/MAX oracle boxed-result typing.
- Direct regression coverage spans the complete type/reduction matrix, both index types, scalar
  and zero domains, arbitrary layouts, mixed carriers, modular overflow, floating special values
  and rounding, exponent cancellation, parallel scratch/repeat, validation before workers,
  overlap/deduplication, cache compatibility, and malformed finalization.
- Documentation preserves the current-source correction: `AxisScatterKind` contains only
  `SCATTER_ELEMENTS` and Gather-compatible `SCATTER_ADD`; historical `SCATTER_AXIS_ADD` and its
  superseded shape remain historical evidence only.

## Completion summary

- Completed changes: implemented and documented current portable functional scatter for exactly
  one fully static resolved-layout CPU occurrence, including deterministic validation, represented
  reductions, disjoint output ownership, optional exact floating-product scratch, and schema 16.
- Files changed or created: the exact allowlisted CPU production/test paths recorded by the final
  changed-path inventory plus, in this documentation context, `CpuKernelSpecialization.java`,
  `CpuPartitionLowering.java`, `CpuScatterLowering.java`; package summaries for cache,
  codegen/emit, executable, IR, lowering, memory, prepare, reference, and route/portable;
  `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan, and the
  roadmap. No file was created by this documentation context.
- Tests and validation: reused the stabilized 12-suite/103-test focused and 38-suite/230-test CPU
  evidence above; ran CPU Javadoc twice as recorded; inspected generated pages; passed final
  Markdown/status/scope/schema checks and `git diff --check`.
- Documentation-agent review: complete in context `019ff4fb-94c7-7921-af13-28a7395c3ae7`.
- Documentation impact: the CPU guide now explains exact current scatter coverage and boundaries;
  API/Compiler guides remain accurate because they already describe Model semantics without
  promising CPU coverage.
- Javadoc review: every changed implementation contract was reviewed; actionable warnings were
  corrected, and rendered IR/lowering/emitter plus lifecycle/cache/reference/package contracts
  were inspected.
- Glossary impact: current CPU portable-route, Scatter Add, Scatter Elements, Scatter-ND, slice
  update, and kernel-specialization entries now distinguish schema-16 CPU execution from Model
  construction and future routes.
- No-change conclusions: Model, Compiler, shared Prepare/Runtime, public Tensor/API, architecture,
  ADRs, architecture tests, backend conformance, integration tests, Gradle/dependencies,
  vendor/native code, existing gather/movement families, and later task specifications required no
  change because this task adds only a CPU-private route within existing ownership and lifecycle
  contracts. CPU 0006B2 remains Draft without a detailed specification.
- Unresolved issues: None.
- Follow-up required: None; CPU 0006B2 is the next Draft planning frontier.

Status: Complete
