# Task 0006A2: Portable [Gather](../../../../glossary.md#gather) and One-Hot Indexing

## Status

Complete

## Goal

Add the first value-dependent portable [CPU](../../../../glossary.md#term-cpu) indexing family for exactly one fully static,
resolved-[layout](../../../../glossary.md#layout) [`GATHER`](../../../../glossary.md#gather), `GATHER_ELEMENTS`, `GATHER_ND`, or `ONE_HOT` [occurrence](../../../../glossary.md#term-occurrence). Preserve the
current [Model](../../../../glossary.md#term-model) [signatures](../../../../glossary.md#term-signature), [Shapes](../../../../glossary.md#shape), types, attributes, bounds, and strict invalid-index meaning
without adding or changing any shared semantic contract.

The CPU implementation has two execution phases:

```text
bound direct index carrier
  -> deterministic scalar validation of every logical index value
  -> only after complete success, generated scalar or parallel-scalar output writes
```

The [validation](../../../../glossary.md#term-validation) phase performs no physical output-[carrier](../../../../glossary.md#term-storage-carrier) write. It reports the first invalid
[logical index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) deterministically and prevents every generated write call and worker submission.
The generated phase may then trust all values to be in bounds. CPU analysis owns [lowering](../../../../glossary.md#lowering),
compact geometry, strategy, declarations, and [specialization](../../../../glossary.md#term-specialization); finalization realizes one
[schema](../../../../glossary.md#term-schema)-14 [artifact](../../../../glossary.md#term-artifact) after shared slot assignment; [Runtime](../../../../glossary.md#term-runtime) receives only the [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable)
and direct bound carriers.

## [Scope](../../../../glossary.md#term-scope)

### Exact [operation](../../../../glossary.md#operation) [matrix](../../../../glossary.md#term-matrix-meanings)

| Kind | Exact attributes and [signature](../../../../glossary.md#term-signature) | Admitted boundary types | Exact result mapping |
|---|---|---|---|
| `AxisGatherKind.GATHER` | `IndexAxisAttrs(axis)`; ordered inputs `[data, indices]`; one output | data/output: any current [Model](../../../../glossary.md#term-model) type; indices: [INT32](../../../../glossary.md#data-type--datatype) or [INT64](../../../../glossary.md#data-type--datatype) | replace data [axis](../../../../glossary.md#term-axis) `a` with the complete indices [Shape](../../../../glossary.md#shape) |
| `AxisGatherKind.GATHER_ELEMENTS` | `IndexAxisAttrs(axis)`; ordered inputs `[data, indices]`; one output | data/output: any current Model type; indices: INT32 or INT64 | output is exact indices Shape; non-axis [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) align with data |
| `GatherNdKind.GATHER_ND` | `GatherNdAttrs(batchDimensions)`; ordered inputs `[data, indices]`; one output | data/output: any current Model type; indices: INT32 or INT64 | indices prefix without [tuple depth](../../../../glossary.md#term-tuple-depth) plus the unindexed data suffix |
| `OneHotKind.ONE_HOT` | `OneHotAttrs(depth)`; ordered input `[indices]`; one output | indices: INT32 or INT64; output: [BOOL](../../../../glossary.md#data-type--datatype) | append depth and emit exact false/true indicator values |

The six current Model data types are [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), `INT32`, `INT64`, and
`BOOL`. [Gather](../../../../glossary.md#gather)-family execution copies [represented bits](../../../../glossary.md#term-represented-bits) without conversion. BFLOAT16 remains raw
16-bit movement, and a BOOL gather input remains subject to the current [canonical](../../../../glossary.md#term-canonical-form) `0`/`1` cold
[binding](../../../../glossary.md#term-binding) check. ONE_HOT emits canonical BOOL byte `0` for false and byte `1` for true. It has no
configurable on/off values, output type, axis, [ignore index](../../../../glossary.md#ignore-index), sparse form, or default row.

Only one semantic [node](../../../../glossary.md#node) and one distinct output are admitted. GATHER-family input occurrences are
deduplicated by exact [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) in first-[occurrence](../../../../glossary.md#term-occurrence) order, so a legal request that uses one value
for both data and indices declares it once while retaining occurrence map `[0, 0]`. Otherwise the
map is `[0, 1]`. ONE_HOT has map `[0]`. The output is always a separate final boundary.

### [Canonical](../../../../glossary.md#term-canonical-form) [GATHER](../../../../glossary.md#gather) mapping

Let data [rank](../../../../glossary.md#term-rank) be `R`, [normalized axis](../../../../glossary.md#normalized-axis) be `a`, indices rank be `Q`, data extents be `D`, and
indices extents be `I`. Analysis requires the current [Model](../../../../glossary.md#term-model) result formula exactly:

```text
output.shape = D[0:a] + I[0:Q] + D[a+1:R]
```

For output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `y`, the indices coordinate is `y[a:a+Q]`. Its [scalar](../../../../glossary.md#term-scalar) value `v` selects:

```text
data coordinate = y[0:a] + [v] + y[a+Q:outputRank]
```

Scalar indices have `Q == 0` and remove the selected [axis](../../../../glossary.md#term-axis). Every logical indices value is
validated once in [row-major](../../../../glossary.md#term-row-major) logical encounter order against `0 <= v < D[a]`, independently of
the number of data prefix/suffix combinations that reuse it.

### [Canonical](../../../../glossary.md#term-canonical-form) GATHER_ELEMENTS mapping

Data, indices, and output have the same [rank](../../../../glossary.md#term-rank) `R`; output retains the exact indices [Shape](../../../../glossary.md#shape). The
current [Model](../../../../glossary.md#term-model) contract has already proved equal non-[axis](../../../../glossary.md#term-axis) [Dimensions](../../../../glossary.md#dimension). For output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `y`,
read `v = indices[y]` and then:

```text
data coordinate = y with coordinate a replaced by v
```

Validate every logical indices value once in [row-major](../../../../glossary.md#term-row-major) logical encounter order against
`0 <= v < D[a]`. The selected indices [extent](../../../../glossary.md#term-extent) may differ from the selected data extent.

### [Canonical](../../../../glossary.md#term-canonical-form) GATHER_ND mapping

Let data [rank](../../../../glossary.md#term-rank) be `R`, indices rank be `Q`, [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count be `B`, and the final indices [extent](../../../../glossary.md#term-extent) be the
positive static [tuple depth](../../../../glossary.md#term-tuple-depth) `K`. Analysis preserves the exact [Model](../../../../glossary.md#term-model) prerequisites:

```text
0 <= B < min(R, Q)
1 <= K <= R - B
indices.shape[0:B] == data.shape[0:B]
output.shape = indices.shape[0:Q-1] + data.shape[B+K:R]
```

For each [row-major](../../../../glossary.md#term-row-major) indices-prefix [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `p`, validate tuple components `k = 0..K-1` in
increasing order. Component `k` supplies data [axis](../../../../glossary.md#term-axis) `B + k` and must satisfy:

```text
0 <= indices[p..., k] < data.extent[B + k]
```

The deterministic [scalar](../../../../glossary.md#term-scalar)-index ordinal is the flattened row-major ordinal in the complete
indices [tensor](../../../../glossary.md#tensor), so tuple order and component order form one total encounter order. For output
coordinate `y`, leading batch coordinates come from `y[0:B]`, indexed coordinates come from the
tuple at the output's indices-prefix coordinate, and the remaining output suffix addresses data
axes `[B+K, R)` unchanged.

### [Canonical](../../../../glossary.md#term-canonical-form) ONE_HOT mapping

Let indices [rank](../../../../glossary.md#term-rank) be `Q`, input [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) be `p`, and positive static depth be `D`. Analysis
requires output [Shape](../../../../glossary.md#shape) `indices.shape + [D]`. Validate each [row-major](../../../../glossary.md#term-row-major) logical input value `v`
against `0 <= v < D`. After complete [validation](../../../../glossary.md#term-validation), output coordinate `[p..., j]` receives:

```text
output[p..., j] = (v == j) ? 1 : 0
```

The generated writer emits exact canonical [BOOL](../../../../glossary.md#data-type--datatype) bytes. Invalid negative or out-of-range values
never wrap, clamp, select a default, or produce an all-false row.

### Deterministic complete pre-write [validation](../../../../glossary.md#term-validation)

Structural validation remains cold: capability and [lowering](../../../../glossary.md#lowering) validate kind/attributes, input and
output counts, exact data types, static [Shapes](../../../../glossary.md#shape), resolved [layouts](../../../../glossary.md#layout), Shape formulas, ranks, [axes](../../../../glossary.md#term-axis),
[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) prefix, [tuple depth](../../../../glossary.md#term-tuple-depth), checked counts/spans, and output [injectivity](../../../../glossary.md#storage-mapping-contiguity-and-injectivity). Cold [binding](../../../../glossary.md#term-binding) validates
[carrier](../../../../glossary.md#term-storage-carrier) type, accessibility, size, [alignment](../../../../glossary.md#term-alignment), output writability, [canonical](../../../../glossary.md#term-canonical-form) gathered [BOOL](../../../../glossary.md#data-type--datatype) input,
and output/input non-overlap.

Index-value validation is necessarily execution-time because caller inputs may change between
runs. `CpuPreparedExecutable` cold binding creates one [CPU](../../../../glossary.md#term-cpu)-private typed validation action with
direct [INT32](../../../../glossary.md#data-type--datatype) or [INT64](../../../../glossary.md#data-type--datatype) carrier access and compact [immutable](../../../../glossary.md#term-immutable) geometry. On every execution it scans
the complete [logical index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) domain on the invoking thread, without allocation, reflection, map
[lookup](../../../../glossary.md#term-lookup), semantic inspection, virtual per-[element](../../../../glossary.md#term-element) [dispatch](../../../../glossary.md#term-dispatch), output write, or worker submission.
The carrier form is selected once during binding; the inner [scan](../../../../glossary.md#term-cumulative-scan) uses direct primitive-array or
native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment) loads and a resolved-layout odometer with no per-element division or
modulo.

The first invalid [scalar](../../../../glossary.md#term-scalar) in the family order above throws `IndexOutOfBoundsException` with one of
these exact messages:

```text
GATHER index at logical position <ordinal> for data axis <axis> is out of bounds: value=<value>, extent=<extent>
GATHER_ELEMENTS index at logical position <ordinal> for data axis <axis> is out of bounds: value=<value>, extent=<extent>
GATHER_ND index at logical position <ordinal> for data axis <axis> is out of bounds: value=<value>, extent=<extent>
ONE_HOT index at logical position <ordinal> is out of bounds: value=<value>, depth=<depth>
```

`value`, extents, and depth are represented as signed `long` diagnostic values; INT32 values are
sign-extended. An invalid [run](../../../../glossary.md#term-run) performs no physical output-carrier mutation. [Runtime](../../../../glossary.md#term-runtime) may already
have invalidated logical output validity under its existing runner contract; this task changes no
Runtime state rule and promises only that CPU writes no partial output bytes.

### Zero-work behavior

All structural, [layout](../../../../glossary.md#layout), count, span, [carrier](../../../../glossary.md#term-storage-carrier), [injectivity](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), and overlap checks still [run](../../../../glossary.md#term-run).
[Validation](../../../../glossary.md#term-validation) visits the [logical index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) domain even when the output [element](../../../../glossary.md#term-element) count is zero because an
unrelated zero data suffix does not make supplied indices valid. If the logical index domain is
empty, validation succeeds without a load. After successful validation, zero output elements
submit no generated range and touch no output carrier.

Consequences include:

- empty [GATHER](../../../../glossary.md#gather) or GATHER_ELEMENTS indices validate no values and produce zero work;
- an empty ONE_HOT input validates no values and writes no rows;
- GATHER_ND with an empty tuple-prefix domain validates no tuple values;
- GATHER_ND with non-empty tuples and a zero unindexed data suffix still validates every tuple;
- a zero selected data extent accepts only an empty relevant index domain; every encountered
  value is invalid because <span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>`[0, 0)` is empty; and
- output-count or span [overflow](../../../../glossary.md#term-overflow) fails cold instead of attempting [materialization](../../../../glossary.md#materialization).

### [Layout](../../../../glossary.md#layout), carriers, [injectivity](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), and overlap

All input and output [Shapes](../../../../glossary.md#shape) are fully static and layouts resolved. Data and index inputs honor
non-negative offsets and [strides](../../../../glossary.md#term-stride), including read-zero strides. Index encounter order is logical
[row-major](../../../../glossary.md#term-row-major) order, independent of physical address order or repeated addresses. Outputs honor
resolved offsets and non-negative strides only after the existing complete bounded injectivity
decision succeeds.

The existing seven [carrier](../../../../glossary.md#term-storage-carrier) forms remain sufficient: `DOUBLE_ARRAY`, `FLOAT_ARRAY`,
`SHORT_ARRAY`, `INT_ARRAY`, `LONG_ARRAY`, `BYTE_ARRAY`, and `MEMORY_SEGMENT`. Each boundary uses
the form compatible with its own type; indexing is the first movement family with mixed boundary
types. Input/input physical overlap is allowed, including a deduplicated exact input. The complete
output accessed span must be distinct from and non-overlapping with every unique input span.
No in-place or partially overlapping [gather](../../../../glossary.md#gather)/one-hot route is admitted.

### [CPU](../../../../glossary.md#term-cpu)-private [IR](../../../../glossary.md#term-intermediate-representation-ir) and compact geometry

Add one [sealed](../../../../glossary.md#term-sealed)-family implementation `CpuIndexingIr` to `CpuPortableKernelIr`; do not overload
`CpuDataMovementIr`, whose [invariant](../../../../glossary.md#term-invariant) is one represented type shared by every boundary and no
value-dependent failure. `CpuIndexingIr` owns exactly four closed plan variants for [GATHER](../../../../glossary.md#gather),
GATHER_ELEMENTS, GATHER_ND, and ONE_HOT. It records only code-shaping facts:

- family;
- data, indices, and output ranks as applicable;
- ordered [occurrence](../../../../glossary.md#term-occurrence)-to-boundary mapping;
- each boundary's [data type](../../../../glossary.md#data-type--datatype) and structural access plan;
- one [injective](../../../../glossary.md#term-injective) output store; and
- the universal generated output `start`/`end` loop.

Add `CpuIndexingLowering` with one compact [immutable](../../../../glossary.md#term-immutable) `Geometry` and four closed variants. Geometry
retains exact static extents, offsets, [strides](../../../../glossary.md#term-stride), [normalized axis](../../../../glossary.md#normalized-axis) or [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count, [tuple depth](../../../../glossary.md#term-tuple-depth),
bounds, and range-start odometer state needed by [validation](../../../../glossary.md#term-validation) and output mapping. It must not retain
one selector/address per output, one bounds entry per index, a [Tensor](../../../../glossary.md#tensor), [graph](../../../../glossary.md#graph) [node](../../../../glossary.md#node), [`Operation`](../../../../glossary.md#operation),
[Runtime](../../../../glossary.md#term-runtime) slot, [carrier](../../../../glossary.md#term-storage-carrier), or [run](../../../../glossary.md#term-run) object.

Cold geometry may use division/remainder to decompose an arbitrary validation or output range
start. Both the bound validation loop and generated write loop then use primitive [carry](../../../../glossary.md#term-carry)/reset
odometers. Gather writers may reload already validated indices but perform no repeated bounds
branch. ONE_HOT may compare `j` with its already validated index. Hot output loops perform no
allocation, reflection, map [lookup](../../../../glossary.md#term-lookup), string [dispatch](../../../../glossary.md#term-dispatch), graph/semantic inspection, route choice,
per-[element](../../../../glossary.md#term-element) division/modulo, or generic cursor call.

`CpuPartitionLowering` routes only an exact one-node indexing occurrence to the new lowerer.
`CpuClassFileKernelGenerator` dispatches the new structural family to a focused
`CpuIndexingEmitter`. `CpuScalarReferenceKernel` independently evaluates the same formulas and
strict validation order for differential tests; it is not a Runtime fallback.

### Strategy, declarations, workspace, and [artifact](../../../../glossary.md#term-artifact) count

Index [validation](../../../../glossary.md#term-validation) is always single-thread [scalar](../../../../glossary.md#term-scalar) for deterministic first-invalid selection. Output
generation uses scalar compute with either one range or the completed deterministic disjoint
parallel orchestration. [Vector](../../../../glossary.md#term-vector) preference falls back to scalar. Parallel selection depends on
output [element](../../../../glossary.md#term-element) count only and workers start only after the full validation pass succeeds.

Resource and artifact counts are exact:

- [GATHER](../../../../glossary.md#gather)/GATHER_ELEMENTS/GATHER_ND declare each distinct input [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) once in semantic
  first-occurrence order, then the distinct output: two or three buffer declarations;<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- ONE_HOT declares indices then output: exactly two [buffer](../../../../glossary.md#term-buffer) declarations;
- every family declares zero workspaces and selects no [materialization](../../../../glossary.md#materialization);
- every family lowers to one execution unit, one specialization, one generated class artifact,
  one prepared executable, and one bound invocation; and<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
- finalization performs exactly one artifact-store call after all assignments and worker
  prerequisites validate.

The generated artifact owns only the output-writing pass. Execution-time value validation remains
in the bound [CPU](../../../../glossary.md#term-cpu) executable because it consumes [run](../../../../glossary.md#term-run)-bound carriers and has a different logical
iteration domain from output generation. It is neither [prepare](../../../../glossary.md#prepare)-time value inspection nor a
[Runtime](../../../../glossary.md#term-runtime) semantic interpreter.

### Generated [schema](../../../../glossary.md#term-schema) and compatibility

Generated [bytecode](../../../../glossary.md#term-bytecode) changes, so advance `CpuGeneratorSchema.CURRENT_VERSION` exactly `13 -> 14`.
There is no migration reader; schema 13 and older envelopes are incompatible safe misses.

The generated compatibility [identity](../../../../glossary.md#term-identity-meanings) includes schema 14, indexing family, structural input and
output ranks/access plans, [occurrence](../../../../glossary.md#term-occurrence) map, ordered boundary types, [carrier](../../../../glossary.md#term-storage-carrier)/access pattern, [scalar](../../../../glossary.md#term-scalar)
compute form, materialized position `-1`, entry [descriptor](../../../../glossary.md#term-descriptor), and existing Java/Class-File
compatibility facts. It excludes concrete extents and counts, [axis](../../../../glossary.md#term-axis), [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count, [tuple depth](../../../../glossary.md#term-tuple-depth),
one-hot depth, [layout](../../../../glossary.md#layout) offsets/[stride](../../../../glossary.md#term-stride) magnitudes, bound values, [validation](../../../../glossary.md#term-validation) results, range/chunk
sizes, worker identity, carrier objects/byte offsets, slots, addresses, [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), [graph](../../../../glossary.md#graph)/[run](../../../../glossary.md#term-run)
identity, and [artifact](../../../../glossary.md#term-artifact)-root state. Tests must prove compatible cold geometry reuses exact identity
and class bytes while family, [rank](../../../../glossary.md#term-rank)/access structure, occurrence map, boundary type/carrier, or
schema changes do not.

### Failure behavior and stop conditions

Capability returns false and [lowering](../../../../glossary.md#lowering) fails closed for a wrong [signature](../../../../glossary.md#term-signature), attribute class,
input/output count, type, [rank](../../../../glossary.md#term-rank), [Shape](../../../../glossary.md#shape) formula, [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) prefix, [tuple depth](../../../../glossary.md#term-tuple-depth), dynamic Shape,
unresolved [layout](../../../../glossary.md#layout), non-[injective](../../../../glossary.md#term-injective) output, unsupported multi-[node](../../../../glossary.md#node)/mixed [partition](../../../../glossary.md#partition), or checked
count/span/geometry [overflow](../../../../glossary.md#term-overflow). [Binding](../../../../glossary.md#term-binding) fails before execution for an incompatible [carrier](../../../../glossary.md#term-storage-carrier),
undersized/misaligned/inaccessible storage, read-only output, noncanonical gathered [BOOL](../../../../glossary.md#data-type--datatype) data, or
output/input overlap.

The execution-time failure is only the strict first-invalid-index exception specified above.
Valid execution has no partial-failure policy because all generated writes are independent and
in bounds. Stop before editing outside this task if implementation needs [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler),
[Planning](../../../../glossary.md#term-planning), shared [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), Config, [Trace](../../../../glossary.md#trace), [Engine](../../../../glossary.md#term-engine), another [backend](../../../../glossary.md#backend), [dependency](../../../../glossary.md#term-dependency), [Gradle](../../../../glossary.md#term-gradle),
architecture, architecture-test, backend-[conformance](../../../../glossary.md#term-conformance), integration, public [API](../../../../glossary.md#term-application-programming-interface-api), or 0006B planning
changes; a workspace or second [artifact](../../../../glossary.md#term-artifact); generated [validation](../../../../glossary.md#term-validation); a per-[element](../../../../glossary.md#term-element) prepared table; or
more than the authorized maximum. Report the exact missing contract instead of inventing shared
architecture or broadening Model semantics.

## Out of [scope](../../../../glossary.md#term-scope)

- Scalar SELECT, embedding-specific behavior, reduced-rank historical gather names, aliases, or
  any new Model kind/method/attribute.<span data-glossary-navigation="true"> ([Scalar SELECT glossary](../../../../glossary.md#scalar-select); [embedding glossary](../../../../glossary.md#embedding); [rank glossary](../../../../glossary.md#term-rank); [gather glossary](../../../../glossary.md#gather); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [Model glossary](../../../../glossary.md#term-model))</span>
- SCATTER_ELEMENTS, SCATTER_ADD, SCATTER_ND, SLICE_UPDATE, FOLD_AXIS, FOLD2D, duplicate-target
  policy, overlap accumulation, or task 0006B implementation/specification.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Negative-index normalization, wrapping, clamping, padding/default rows, ignored indices,
  all-false invalid one-hot rows, sparse one-hot, or configurable on/off values.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [padding glossary](../../../../glossary.md#padding))</span>
- Dynamic [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout) [binding](../../../../glossary.md#term-binding), unresolved layout selection, in-place or overlapping output.
- Vector gather, vectorized one-hot, native/vendor routes, tuning, benchmarks, fixed-shape or
  unrolled variants, and performance claims.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span>
- Multi-node or mixed-family partitions, general DAG decomposition/fusion, multiple outputs, or
  generated validation entry points.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [validation glossary](../../../../glossary.md#term-validation))</span>
- Shared/public/build/architecture/[conformance](../../../../glossary.md#term-conformance)/integration changes, [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), or pushes.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`docs/architecture/current-architecture-plan.md`](../../../../architecture/current-architecture-plan.md)
- [`docs/architecture/module-boundaries.md`](../../../../architecture/module-boundaries.md)
- [`docs/architecture/dependency-rules.md`](../../../../architecture/dependency-rules.md)
- [`docs/architecture/lifecycle.md`](../../../../architecture/lifecycle.md)
- [`docs/architecture/runtime-prepare-backend-boundary.md`](../../../../architecture/runtime-prepare-backend-boundary.md)
- [`docs/planning/planning-guide.md`](../../../planning-guide.md)
- [`docs/developer-guide/documentation-rules.md`](../../../../developer-guide/documentation-rules.md)
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0005A atomic partition-kernel reset](0005a-atomic-partition-kernel-architecture-reset.md)<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [partition glossary](../../../../glossary.md#partition); [kernel glossary](../../../../glossary.md#kernel))</span>
- [CPU 0006 portable static affine views](0006-portable-static-affine-views-and-boundary-materialization.md)<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings))</span>
- [CPU 0006A portable static movement](0006a-portable-pad-tile-and-tensor-composition-movement.md)
- [CPU 0006A1 portable static window extraction](0006a1-portable-static-window-extraction.md)
- [Model 0018O canonical indexing taxonomy](../../../modules/model/tasks/0018o-indexing-taxonomy-and-unstack-normalization.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- [Model 0019A2 one-hot encoding](../../../modules/model/tasks/0019a2-one-hot-encoding.md)<span data-glossary-navigation="true"> ([one-hot encoding glossary](../../../../glossary.md#one-hot-encoding))</span>

## Architecture constraints

- Model owns the exact operation meaning, signature, attributes, Shapes, admitted semantic types,
  and strict invalid-index policy. CPU consumes those current contracts without modifying them.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [signature glossary](../../../../glossary.md#term-signature); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Planning selects only CPU ownership. CPU analysis/lowering owns compact IR/geometry, route and
  strategy selection, exact resource declarations, and compatibility identity.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lowering glossary](../../../../glossary.md#lowering); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Shared Prepare assigns declared buffers opaquely. CPU finalization realizes one selected
  artifact only after assignment and cannot change route or add a resource.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [buffers glossary](../../../../glossary.md#term-buffer); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- [Runtime](../../../../glossary.md#term-runtime) sees and invokes only the [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable) and direct bound actions. It never sees
  [`Operation`](../../../../glossary.md#operation), `CompiledNode`, indexing [IR](../../../../glossary.md#term-intermediate-representation-ir), [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), or index semantics.
- Each active run performs complete CPU-owned value validation before any physical output write.
  The output is distinct, injective, and non-overlapping with every input.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [validation glossary](../../../../glossary.md#term-validation); [injective glossary](../../../../glossary.md#term-injective))</span>
- Unsupported, dynamic, unresolved, overflowing, malformed, or overlapping cases [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — truthful [occurrence](../../../../glossary.md#term-occurrence) capability.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — route-independent generated structural
  identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — Model-to-CPU lowering and compact
  geometry.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — generated output loops.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — deterministic analysis and
  post-assignment finalization handoff.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — typed cold binding, execution-time
  validation, scalar/parallel output orchestration, and overlap checks.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent [scalar](../../../../glossary.md#term-scalar) [test oracle](../../../../glossary.md#term-test-oracle).
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema) and structural compatibility.

Packages added or changed:

- No package is added or moved. Three focused unsupported internal types are added to existing
  owning packages; no supported public type is added.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr` — closed code-shaping indexing
  family distinct from same-typed value-blind movement IR.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLowering` — exact one-node
  semantic lowering and compact validation/write geometry.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuIndexingEmitter` — focused
  allocation-free generated output writer for already validated indices.

Tests mirror these packages for package-private inspection. No generic `index`, `util`, manager,
[registry](../../../../glossary.md#term-registry), route, selector table, or shared abstraction package is introduced.

## Affected files

Authorized [CPU](../../../../glossary.md#term-cpu) production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuIndexingEmitter.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuIndexingIr.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuIndexingLowering.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`

Authorized CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuIndexingGeneratedKernelTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuIndexingIrTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuIndexingLoweringTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Authorized explanatory and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0006a2-portable-gather-and-one-hot-indexing.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized. Listed paths change only when their actual contract, [Javadoc](../../../../glossary.md#term-javadoc), package
summary, or [regression](../../../../glossary.md#term-regression) [ownership](../../../../glossary.md#term-ownership-meanings) is affected; the implementation and documentation summaries
must record review-only no-change conclusions.

## Maximum [scope](../../../../glossary.md#term-scope)

| Category | Maximum | Accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production/package | 24 | Existing [lifecycle](../../../../glossary.md#lifecycle) [owners](../../../../glossary.md#term-owner) plus three focused internal family types |
| CPU tests | 12 | Existing [regression](../../../../glossary.md#term-regression) owners plus three focused family tests |
| Explanatory documentation | 2 | CPU guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, master plan, and roadmap |
| **Total** | **41** | **24 + 12 + 2 + 3** |

The exception is justified because the first value-dependent generated family atomically crosses
truthful capability, mixed-type [IR](../../../../glossary.md#term-intermediate-representation-ir), compact geometry, generated output emission, typed [run](../../../../glossary.md#term-run)-bound
prevalidation, preparation/finalization, direct [binding](../../../../glossary.md#term-binding), reference parity, [schema](../../../../glossary.md#term-schema) compatibility,
Javadocs, and explanatory documentation. Splitting these seams would either advertise an
unexecutable [operation](../../../../glossary.md#operation) or permit partial output before strict bounds [validation](../../../../glossary.md#term-validation). Do not spend
unused scope on refactoring; stop and revise the task before a 42nd path.

## Acceptance criteria

- Capability and lowering admit exactly the four operation/signature/type/Shape rows specified
  here and fail closed for every excluded case.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [Shape glossary](../../../../glossary.md#shape); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- GATHER, GATHER_ELEMENTS, GATHER_ND, and ONE_HOT generated/reference outputs match exact Model
  coordinate mappings and represented bits across all admitted types and both index types.<span data-glossary-navigation="true"> ([GATHER glossary](../../../../glossary.md#gather); [Model glossary](../../../../glossary.md#term-model); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [represented bits glossary](../../../../glossary.md#term-represented-bits))</span>
- Every logical index value is validated in the locked deterministic order before any physical
  output write or worker submission; the first invalid value throws the exact exception/message,
  and sentinel output bytes remain unchanged.<span data-glossary-navigation="true"> ([logical index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Empty-index and zero-output cases follow the locked independent validation/write domains,
  including non-empty GATHER_ND tuples with a zero result suffix.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- All resolved input layouts, read-zero strides, offset/strided injective outputs, array/segment/
  mixed carriers, and deduplicated same-<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) inputs work as specified.
- Binding rejects wrong carriers, span/alignment/access failures, noncanonical gathered BOOL data,
  non-injective output, and every output/input overlap before execution.<span data-glossary-navigation="true"> ([Binding glossary](../../../../glossary.md#term-binding); [alignment glossary](../../../../glossary.md#term-alignment); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Index validation is scalar and allocation-free; output execution is scalar or parallel-scalar,
  with vector preference falling back and no generated bounds branch or per-element
  division/modulo/allocation/semantic inspection.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [element glossary](../../../../glossary.md#term-element))</span>
- Analysis declares exactly the unique input boundaries then output, no workspace, no
  materialization, one unit, and one artifact; finalization performs one artifact lookup only
  after all assignments and worker checks.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- `CpuIndexingIr` and compact geometry contain only the locked structural/cold facts and no
  per-index/per-output table, graph/Runtime identity, carrier, or run state.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime); [identity glossary](../../../../glossary.md#term-identity-meanings); [carrier glossary](../../../../glossary.md#term-storage-carrier); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- Schema advances exactly 13 to 14; schema 13 safely misses; key inclusion/exclusion and compatible
  cold-geometry reuse are tested.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema))</span>
- Existing pointwise, affine, movement/window, parallel, carrier, persistence, and failure tests
  remain green.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- A separate clean documentation pass finalizes affected Javadocs, package summaries, CPU guide,
  glossary, task/master/roadmap evidence, and no-change conclusions after executable stability.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- No Model/shared/public/build/architecture/conformance/integration/0006B specification or
  unrelated change is present, and the final changed-path set stays within the exact allowlist and
  41-path ceiling.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused [matrix](../../../../glossary.md#term-matrix-meanings):

```bash
./gradlew :backends:cpu:test --tests 'io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest' --tests 'io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIrTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuIndexingGeneratedKernelTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGeneratorTest'
./gradlew :backends:cpu:test
```

The focused matrix covers every family, admitted data/index type, [rank](../../../../glossary.md#term-rank)/formula, [scalar](../../../../glossary.md#term-scalar) indices,
[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count and [tuple depth](../../../../glossary.md#term-tuple-depth), exact [BOOL](../../../../glossary.md#data-type--datatype) on/off bytes, negative and upper-bound failures, first-
invalid order, unchanged sentinel output, zero validation/write domains, arbitrary resolved
[layouts](../../../../glossary.md#layout), zero [strides](../../../../glossary.md#term-stride), deduplicated input [identity](../../../../glossary.md#term-identity-meanings), [heap](../../../../glossary.md#term-heap)/segment/mixed carriers, overlap,
parallel-after-validation behavior, declarations, zero workspace, one [artifact](../../../../glossary.md#term-artifact), reference parity,
and [schema](../../../../glossary.md#term-schema)/key behavior. [Run](../../../../glossary.md#term-run) the final [CPU](../../../../glossary.md#term-cpu) suite exactly once after executable Java stabilizes and
record suite/test/skip totals plus exact Java 26 identity.

The separate documentation-focused pass reuses that Java evidence unless executable behavior
changes or a concrete stale-evidence risk is recorded, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates [canonical](../../../../glossary.md#term-canonical-form) task headings, every local Markdown path and heading anchor, balanced
fences, final newlines, trailing whitespace, exact allowlist/41-path ceiling, schema/status
[synchronization](../../../../glossary.md#term-synchronization), current canonical [Model](../../../../glossary.md#term-model) indexing names with no removed historical [alias](../../../../glossary.md#aliasing-and-arena), unchanged
0006B-and-later Draft statuses, and absence of a 0006B specification.

Repository-wide tests are deferred to CPU 0009 and [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes only the CPU
[module](../../../../glossary.md#term-module) and its documentation. Architecture tests are unchanged because no [dependency](../../../../glossary.md#term-dependency) or hot-path
boundary changes. [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) and integration tests remain unchanged because [Engine](../../../../glossary.md#term-engine)
composition/public output access are not current and the focused generated-versus-independent-
reference matrix owns this CPU-private capability evidence. If implementation changes a shared
contract or makes either conclusion false, stop and replan instead of silently broadening the
validation tier.

## Dependencies

- [CPU 0006A1](0006a1-portable-static-window-extraction.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> is `Complete` and supplies the current
  schema-13 movement, compact geometry, scalar/parallel-scalar, carrier, binding, and artifact
  seams.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [scalar glossary](../../../../glossary.md#term-scalar); [carrier glossary](../../../../glossary.md#term-storage-carrier); [binding glossary](../../../../glossary.md#term-binding); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- [Model 0018O](../../../modules/model/tasks/0018o-indexing-taxonomy-and-unstack-normalization.md)
  is `Complete` and owns the [canonical](../../../../glossary.md#term-canonical-form) [`GATHER`](../../../../glossary.md#gather)/`GATHER_ELEMENTS` names and exact [Shape](../../../../glossary.md#shape) mappings.
- [Model](../../../../glossary.md#term-model) tasks [0018E](../../../modules/model/tasks/0018e-gather-nd-semantics.md) and
  [0018F](../../../modules/model/tasks/0018f-gather-nd-tensor-expressions.md) are `Complete` and own
  GATHER_ND signature, batch/tuple, type, and Shape contracts.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- [Model 0019A2](../../../modules/model/tasks/0019a2-one-hot-encoding.md) is `Complete` and owns
  ONE_HOT depth, BOOL result, exact on/off, and strict invalid-index semantics.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Current `OperationSignature`, `Shape`, `LayoutDescriptor`, CPU prepare/finalize/executable,
  typed carrier, artifact store, worker, and scalar-reference contracts.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0006B remains `Draft` without a detailed specification and owns functional update,
  scatter, and overlap-fold execution after this task is Complete.
- CPU 0006C remains `Draft` for stable ordering/selection; CPU 0006D remains `Draft` for explicit-
  state random/dropout execution.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- CPU 0008A remains `Draft` for general [partition](../../../../glossary.md#partition)-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [decomposition](../../../../glossary.md#term-decomposition)/[fusion](../../../../glossary.md#term-fusion).
- CPU 0009 owns portable generated-coverage closure plus repository/[conformance](../../../../glossary.md#term-conformance) [checkpoint](../../../../glossary.md#term-checkpoint).

Do not create a 0006B or later specification during this task.

## Architecture impact

Expected impact: None.

The architecture already assigns semantics to [Model](../../../../glossary.md#term-model), deterministic [lowering](../../../../glossary.md#lowering) and route choice to
[CPU](../../../../glossary.md#term-cpu) analysis, [opaque](../../../../glossary.md#term-opaque) assignment to shared [Prepare](../../../../glossary.md#prepare), [artifact](../../../../glossary.md#term-artifact) realization to CPU finalization, and
direct [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) to [Runtime](../../../../glossary.md#term-runtime). The new [IR](../../../../glossary.md#term-intermediate-representation-ir), compact geometry, and bound [validation](../../../../glossary.md#term-validation) action
are CPU-private. If implementation requires shared value validation, Runtime semantic
inspection, a second artifact/workspace, or another [ownership](../../../../glossary.md#term-ownership-meanings) rule, stop and report the exact gap.

## Implementation prompt

```text
You are the isolated implementation agent for Synaptik CPU task 0006A2. Do not commit or push.
Do not use any GSD skill, command, artifact, directory, or workflow.

Read in full AGENTS.md, ARCHITECTURE.md, the focused module/dependency/lifecycle/runtime-prepare-
backend architecture docs, planning guide/roadmap, CPU master plan, completed CPU 0005A through
0006A1 as relevant, this task, completed Model 0018O/0018E/0018F/0019A2, and every current source,
test, Javadoc, CPU-guide, glossary, artifact, prepare/finalize/executable/binding/reference
contract named by this specification.

Implement task 0006A2 exactly within its 41-path allowlist. Add the focused CPU-private indexing
IR, lowering/compact geometry, and generated output emitter. Preserve exact current GATHER,
GATHER_ELEMENTS, GATHER_ND, and ONE_HOT signatures, Shapes, types, bounds, encounter order,
zero-work rules, carriers/layouts, output injectivity/non-overlap, scalar/parallel-scalar strategy,
unique declarations, zero workspace, and one artifact. Perform a direct allocation-free scalar
run-bound validation pass before every physical output write or worker submission, throw the
locked first-invalid exception/message, and run generated output code only after full success.
Advance schema exactly 13 to 14 and enforce the locked key inclusions/exclusions.

Do not add or change Model/shared/public/build/architecture/conformance/integration contracts,
negative-index normalization, scatter/fold/order/random work, vector/native/tuning/general-DAG
work, generated validation, a workspace/second artifact/per-element table, a 0006B specification,
commit, or push. Stop on architecture uncertainty, missing contracts, or a 42nd path.

Run the focused twelve-class command and the sole final :backends:cpu:test after Java stabilizes.
Then hand the exact diff and evidence to a distinct clean documentation-focused agent in the same
overall change. That agent follows documentation rules and General/API-Javadoc/Backend-guide/
Planning/Example profiles, independently finalizes affected Javadocs/package summaries, CPU guide,
glossary, task/master/roadmap evidence, runs CPU Javadoc and documentation/scope/schema/status/
whitespace gates, and reuses Java evidence unless executable behavior changes or a concrete stale-
evidence risk is recorded.

Report exact completed changes, files, validation, documentation review, no-change conclusions,
unresolved issues, follow-up, and clean context IDs. Finish with `Status: Complete` only after all
implementation, test, documentation, and synchronization gates pass; otherwise use
`Status: Incomplete` and a specific `Follow-up required:` line.
```

## Local decisions

- Use a distinct `CpuIndexingIr` because current `CpuDataMovementIr` requires one represented
  type across every boundary and encodes value-blind movement; indexing has INT32/INT64 coordinate
  boundaries, mixed output types, and a value-dependent failure contract.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Keep structural and carrier checks cold, but perform value validation on every bound run.
  Prepare-time inspection would be incorrect for dynamic caller values, while Runtime semantic
  inspection would violate the hot-path boundary.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Keep validation in a CPU-owned direct bound action and generation in one output-writing
  artifact. This preserves one artifact and permits one deterministic validation domain followed
  by independently parallelizable output ranges.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Validate on one thread in row-major logical index order. Parallel validation would require a
  reduction/coordination contract merely to select the first failure and could allow writes to
  race ahead; it is not justified here.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [logical index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [race glossary](../../../../glossary.md#term-race))</span>
- Reload already validated indices during output generation instead of retaining a per-index
  table or workspace. This is bounded additional read work and preserves exact no-workspace,
  allocation-free, reusable preparation.
- Deduplicate exact input [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) occurrences in first-use order, matching completed static
  movement, while retaining semantic occurrence mapping in IR/geometry.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Treat zero output and zero index domains independently so strict index validity is not skipped
  merely because an unindexed suffix has zero elements.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>
- Advance generated compatibility to schema 14 with no migration. Concrete geometry remains cold
  when it changes neither bytecode nor entry compatibility.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>

## Known limitations

- Only one fully static, resolved-layout indexing occurrence is executable; mixed/multi-node
  partitions and unresolved/dynamic binding remain unsupported.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [occurrence glossary](../../../../glossary.md#term-occurrence); [node glossary](../../../../glossary.md#node); [binding glossary](../../../../glossary.md#term-binding))</span>
- Validation is deliberately scalar and reads index values a second time during generated output
  mapping. No vector/native/tuned or retained validated-index representation exists.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector))</span>
- Outputs must be distinct, [injective](../../../../glossary.md#term-injective), and non-overlapping with inputs. No in-place route exists.
- Model expression construction normally leaves gather/one-hot layout unresolved; existing
  lifecycle work must supply exact resolved descriptors before CPU capability/preparation.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [gather glossary](../../../../glossary.md#gather); [lifecycle glossary](../../../../glossary.md#lifecycle); [descriptors glossary](../../../../glossary.md#term-descriptor); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Public Engine composition and backend-conformance execution remain later work; this task uses
  independent CPU scalar-reference differential evidence.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Mandatory clean implementation audit/fix context
  `019ff0bc-5996-7c30-903e-6f32d1b53a36` independently inspected the complete indexing
  production path and found no executable semantic defect. It did find that the prior seven-file
  test increment did not substantiate the task's claimed focused matrix, so it added compact
  allowlisted coverage in nine existing/new indexing test owners without changing production
  Java.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- The audit coverage now exercises fail-closed capability rows; every Gather family with all six
  represented data types and both index widths; scalar, aligned-axis, batch, tuple-depth, and
  canonical one-hot mappings; exact negative/upper/first-invalid diagnostics and unchanged
  outputs; independent empty-index and zero-output domains; zero selected extents; resolved
  offsets, strides, zero-stride reads, injective strided writes, deduplicated input identity,
  heap/direct-segment/mixed carriers and nonzero carrier offsets; canonical gathered BOOL,
  overlap rejection, and parallel validation-before-write behavior; exact declarations, no
  workspace/materialization, one unit/artifact, reference parity, schema-13 safe miss, and
  structural-key inclusion/exclusion with cold-geometry byte reuse.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [Gather glossary](../../../../glossary.md#gather); [scalar glossary](../../../../glossary.md#term-scalar); [axis glossary](../../../../glossary.md#term-axis); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [canonical glossary](../../../../glossary.md#term-canonical-form); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective); [identity glossary](../../../../glossary.md#term-identity-meanings); [heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier); [BOOL glossary](../../../../glossary.md#data-type--datatype); [materialization glossary](../../../../glossary.md#materialization); [artifact glossary](../../../../glossary.md#term-artifact); [schema glossary](../../../../glossary.md#term-schema))</span>
- `./gradlew :backends:cpu:testClasses` passed. The first development-focused nine-class run
  exposed one incorrect test configuration (65 tests, one failure); after correcting its
  availability snapshot, the same 65 tests passed. After the final coverage increment, the
  nine-class development run passed 66 tests with no skips, failures, or errors.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- The exact required twelve-class focused command passed 12 suites and 71 tests with zero skips,
  failures, or errors.
- The one final exact `./gradlew :backends:cpu:test` module validation passed 35 suites and 183
  tests with one expected opt-in persistence-evidence skip and zero failures or errors on Oracle
  OpenJDK 26.0.1, Runtime and 64-Bit Server VM 26.0.1+8-34.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [persistence glossary](../../../../glossary.md#term-persistence); [Oracle glossary](../../../../glossary.md#term-test-oracle); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- The audit leaves 38 changed paths, all inside the exact allowlist and below the 41-path ceiling.
  CPU 0006B remains Draft and no 0006B specification exists.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Because this audit changed Java test sources after documentation context
  `019ff0af-4cbe-7e03-ad16-3176f117c78a`, that prior documentation pass is stale under the
  repository's clean-context rule. A distinct documentation-focused context must review the
  expanded executable evidence, refresh this task/master/roadmap status and totals, and rerun the
  final documentation/scope/whitespace gates before this task may return to <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`Complete`.

- Implementation context `019ff098-313c-7b53-8a03-df9f31fcf71f` stabilized executable Java after
  direct review removed a per-run `DataType.values()` allocation and made valid zero-output
  execution validate without generated calls or worker submission.
- Mandatory clean documentation context `019ff0c9-83cf-7d82-8f3c-e61be7a30269` independently
  reviewed the complete production/test diff, the final expanded tests, generated Javadocs,
  affected package summaries, CPU guide, glossary, task/master/roadmap state, and current Model
  GATHER/GATHER_ELEMENTS/GATHER_ND/ONE_HOT contracts. It applied General, API/Javadoc, Backend
  Guide, Planning, and Example Format profiles.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning))</span>
- The documentation review found the changed type/member Javadocs and package summaries accurate:
  they preserve Model ownership, describe the CPU-private validation/write split, direct-carrier
  lifecycle, geometry and schema boundaries, parameters, results, and failures without claiming
  dynamic layouts, other backends, Runtime semantic interpretation, or performance. No production
  Java or Java test source changed in this context.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [parameters glossary](../../../../glossary.md#term-parameter); [layouts glossary](../../../../glossary.md#layout))</span>
- Reused audit/fix evidence from `019ff0bc-5996-7c30-903e-6f32d1b53a36`: the exact required
  twelve-class command passed 12 suites/71 tests with zero skips, failures, or errors; the sole
  final `./gradlew :backends:cpu:test` passed 35 suites/183 tests with one expected opt-in
  persistence-evidence skip and zero failures/errors on Oracle OpenJDK 26.0.1, Runtime and 64-Bit
  Server VM 26.0.1+8-34. No executable Java changed afterward, so this context did not rerun Java
  tests.
- `./gradlew :backends:cpu:javadoc` passed once after the documentation stabilized: 11 actionable
  tasks were up to date, with no Javadoc warning or error. Generated-Javadoc inspection found 190
  files and confirmed the indexing IR, lowering, emitter, and prepared-executable contracts in
  their rendered pages.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering))</span>
- The five affected Markdown files passed local target/anchor validation for 481 file links and
  47 heading anchors. Balanced-fence, final-newline, trailing-whitespace, canonical-task-heading,
  task-structure, exact schema/status/name, 0006B-Draft/no-specification, and <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`git diff --check`
  gates passed.
- The final set contains exactly 38 task-allowlisted paths—22 CPU production/package paths,
  eleven CPU test paths, two explanatory documents, and three planning/status documents—within
  the 41-path ceiling. This documentation context changed exactly the task specification, CPU
  master plan, and roadmap to synchronize final counts, context IDs, evidence, and status.

## Implementation notes

- Added CPU-private indexing IR, compact lowering geometry, generated output emission, direct
  execution-time validation, capability/preparation/finalization integration, independent scalar
  reference evaluation, schema-14 identity, and focused regressions within the authorized CPU
  production/test paths.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation); [scalar glossary](../../../../glossary.md#term-scalar); [evaluation glossary](../../../../glossary.md#term-evaluation); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- The documentation pass finalized all changed indexing contracts, constructor/method component
  tags, and affected package summaries. The CPU guide and glossary now explain the four current
  indexing operations, complete pre-write validation, independent zero domains, one artifact/no
  workspace ownership, and current-versus-planned boundaries.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- No executable Java statement changed in the documentation context.

## Completion summary

- Completed changes: truthful one-node GATHER/GATHER_ELEMENTS/GATHER_ND/ONE_HOT capability,
  lowering, validation-before-write execution, generated/reference output mapping, schema 14,
  and an audited acceptance-criteria test matrix rather than the prior shallow evidence.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [GATHER glossary](../../../../glossary.md#gather); [lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation); [schema glossary](../../../../glossary.md#term-schema))</span>
- Files changed or created: 38 authorized paths—22 CPU production/package paths, eleven CPU test
  paths, two explanatory documents, and three planning/status documents.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: the final exact twelve-class command passed 12 suites/71 tests and the
  sole final CPU command passed 35 suites/183 tests with one expected skip and no failures/errors.
- Documentation-agent review: clean context `019ff0c9-83cf-7d82-8f3c-e61be7a30269` independently
  finalized the complete affected documentation set after the audit/fix test expansion.
- Documentation impact: CPU guide, glossary, package summaries, task, master plan, and roadmap are
  finalized in the same overall change; this context edited only the three planning/status files.
- Javadoc review: every affected public or package-private indexing type, constructor, record
  component, and method has meaningful parameter/result/failure coverage as applicable.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Glossary impact: current CPU static indexing and the Model-versus-CPU execution boundary are
  explicit.<span data-glossary-navigation="true"> ([CPU static indexing glossary](../../../../glossary.md#cpu-static-indexing))</span>
- Architecture impact: None. Existing [Model](../../../../glossary.md#term-model)/shared/public/build/[dependency](../../../../glossary.md#term-dependency) contracts are unchanged.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
