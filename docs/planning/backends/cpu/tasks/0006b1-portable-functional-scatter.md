# Task 0006B1: Portable Functional Scatter

## Reader context

Scatter reverses the direction of indexing: `[data, indices, updates]` describes how update
values contribute to a fresh data-shaped result. “Functional” means that neither base data nor
updates are modified. This checkpoint validated all indices, and replacement-target uniqueness
where required, before starting result work. A range owned complete output coordinates so
parallel workers could not race on one destination.

Floating MUL’s once-rounded exact-product scratch was a specific historical numerical choice,
not a rule to infer for today’s Model contract. Its per-range temporary storage was declared
before assignment and released with the run, while input storage remained borrowed. Compare the
current scatter reference for present reduction semantics and the executable explanation for
validation timing; preserve the old exact-product claims as history.

Current reading alongside this historical record:

- [Scatter Elements and current reduction semantics](../../../../reference/tensor/scatter.md#scatter-elements-expressions)
  — explains base/index/update roles, duplicate membership, and the present numerical contract.
- [CPU binding, validation, and ranges](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalexecutable)
  — identifies current-value validation, direct invocation state, and worker/resource lifetimes.

## Status

Complete

## Goal

Add the next executable [CPU](../../../../glossary.md#term-cpu) frontier after completed CPU 0006B: exactly one fully static,
resolved-[layout](../../../../glossary.md#layout) current [Model](../../../../glossary.md#term-model) [occurrence](../../../../glossary.md#term-occurrence) of `SCATTER_ELEMENTS`, [Gather](../../../../glossary.md#gather)-compatible `SCATTER_ADD`,
or `SCATTER_ND` through the existing whole-[partition](../../../../glossary.md#partition) portable generated-[kernel](../../../../glossary.md#kernel) architecture.

The result is functional and has the exact base [Shape](../../../../glossary.md#shape) and type. CPU validates the complete [logical
index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) domain, and `NONE` [target](../../../../glossary.md#term-target) uniqueness where applicable, on the invoking thread before any
output write or worker submission. Successful execution writes each result [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) once from
its base and complete target group. Output-coordinate [ownership](../../../../glossary.md#term-ownership-meanings) makes [scalar](../../../../glossary.md#term-scalar) and parallel-scalar
execution deterministic and [race](../../../../glossary.md#term-race)-free. A declared per-range scratch workspace is used only for
floating `MUL`, whose exact abstract product must round once to the unchanged format; all other
rows require no workspace.

This task adds one CPU-private scatter [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/emitter family. It does not extend Model,
[Compiler](../../../../glossary.md#term-compiler), shared semantics, [Runtime](../../../../glossary.md#term-runtime) interpretation, native routes, or the later overlap-fold
frontier.

## [Scope](../../../../glossary.md#term-scope)

- Admit exactly one [CPU](../../../../glossary.md#term-cpu)-owned [node](../../../../glossary.md#node) with ordered logical inputs `[data, indices, updates]`, one
  output, and one of these current identities:<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
  - `AxisScatterKind.SCATTER_ELEMENTS + ScatterElementsAttrs(axis, reduction)`;
  - `AxisScatterKind.SCATTER_ADD + IndexAxisAttrs(axis)`, with intrinsic `ADD`; or
  - `ScatterNdKind.SCATTER_ND + ScatterNdAttrs(batchDimensions, reduction)`.
- Require fully static Shapes, resolved non-negative-offset/non-negative-stride layouts, exact
  current Model signature and Shape rules, and one distinct writable injective output.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [stride glossary](../../../../glossary.md#term-stride); [layouts glossary](../../../../glossary.md#layout); [Model glossary](../../../../glossary.md#term-model); [signature glossary](../../../../glossary.md#term-signature); [injective glossary](../../../../glossary.md#term-injective))</span>
- Support both exact [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype) index carriers through [heap](../../../../glossary.md#term-heap) arrays, native-order
  [`MemorySegment`](../../../../glossary.md#term-memorysegment), and compatible mixed [carrier](../../../../glossary.md#term-storage-carrier) patterns.
- Support [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), INT32, INT64, and [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) for `NONE`; support the five
  non-BOOL types for `ADD`, `MUL`, `MIN`, and `MAX`. `SCATTER_ADD` is numeric and fixed-add only.
- Add distinct `CpuScatterIr`, `CpuScatterLowering`, and `CpuScatterEmitter` owners. Scatter is not
  a gather/indexing extension: it has a functional base, three semantic inputs, multi-pass
  pre-write validation, target grouping, configurable reductions, and optional exact-product
  scratch.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [validation glossary](../../../../glossary.md#term-validation); [target glossary](../../../../glossary.md#term-target))</span>
- Preserve semantic input occurrences while declaring each distinct input [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) once in
  first-occurrence order, followed by one separate output. Retain the three-entry occurrence map
  so equal data/indices/updates graph values deduplicate without changing roles.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [graph glossary](../../../../glossary.md#graph))</span>
- Validate all index scalars in deterministic logical row-major order before duplicate checking,
  generated invocation, worker submission, or output mutation. Negative values are invalid; no
  normalization, wrapping, clamping, ignoring, padding row, or default target exists.<span data-glossary-navigation="true"> ([scalars glossary](../../../../glossary.md#term-scalar); [row-major glossary](../../../../glossary.md#term-row-major); [normalization glossary](../../../../glossary.md#term-normalization); [padding glossary](../../../../glossary.md#padding))</span>
- For `SCATTER_ELEMENTS + NONE` and `SCATTER_ND + NONE`, perform a second complete deterministic
  allocation-free uniqueness pass after all bounds succeed. Reject the first later logical update
  or tuple that repeats an earlier complete target. No first-write or last-write rule exists.
- Generate an output-domain scalar body. Each disjoint output range loads the exact base, scans
  logical update contributions in row-major order, selects those targeting that output coordinate,
  and writes once. The generated writer contains no bounds or duplicate-failure branch because
  complete validation already succeeded.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Use scalar or parallel-scalar orchestration only. Parallel chunks own disjoint output
  coordinates, share read-only inputs, use disjoint scratch slices when required, and use no
  atomics, locks, barriers, cross-range partials, or reduction merge.
- Declare one workspace only for floating `MUL` with a non-empty possible contribution domain and
  non-empty output. It contains one fixed-capacity exact-product accumulator slice per selected
  range. All other scatter plans declare no workspace. No input materialization is selected.<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type); [slice glossary](../../../../glossary.md#slice); [materialization glossary](../../../../glossary.md#materialization))</span>
- Extend preparation, finalization, generated-entry signature, binding, and workspace validation
  only as needed to pass that exact scratch segment and one cold slice offset to each range.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Advance generated compatibility from schema 15 to schema 16. Older entries are incompatible
  misses; there is no migration reader.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Extend the independent scalar reference with a separately implemented scatter oracle. Its
  floating-product oracle may use <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>`BigInteger` on the test/reference path; generated execution
  must use the declared primitive-limb workspace and may not delegate to the oracle.<span data-glossary-navigation="true"> ([limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs))</span>
- After Java and focused tests stabilize, hand the uncommitted diff to a distinct clean
  documentation-focused context to finalize Javadocs, affected package summaries, the CPU guide,
  glossary, task evidence, CPU master plan, and roadmap.

## Current [Model](../../../../glossary.md#term-model) contract and future [CPU](../../../../glossary.md#term-cpu) realization

### Shared functional contract

- All three operations consume exact ordered `[data, indices, updates]` inputs and produce a fresh
  value with the exact data Shape and type. CPU never mutates any input.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Indices are exactly INT32 or INT64. Updates have exactly the data type. There is no promotion,
  broadcasting, conversion, saturation, alternate index carrier, or implicit zero base.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [data type glossary](../../../../glossary.md#data-type--datatype); [broadcasting glossary](../../../../glossary.md#broadcasting); [saturation glossary](../../../../glossary.md#term-saturation); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `NONE` replaces an addressed target with its sole exact update and accepts all six represented
  types. Arithmetic reductions reject BOOL. Every non-replacement target group contains the base
  exactly once and every addressed update exactly once; duplicate positions remain distinct
  contributions.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- A result coordinate addressed by no update preserves the exact base representation. It is not
  combined with an identity, widened and narrowed, or canonicalized.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Current source is authoritative over historical plans. In particular, current
  `AxisScatterKind` contains only `SCATTER_ELEMENTS` and [Gather](../../../../glossary.md#gather)-compatible `SCATTER_ADD`.
  Historical `SCATTER_AXIS_ADD` and the superseded reduced-[rank](../../../../glossary.md#term-rank) `SCATTER_ADD` shape must not be
  treated as current by capability, lowering, tests, explanatory documentation, or cache identity;
  preserved historical planning records do not reopen either contract.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [cache glossary](../../../../glossary.md#term-cache); [planning glossary](../../../../glossary.md#term-planning))</span>

### `SCATTER_ELEMENTS`

Let data [rank](../../../../glossary.md#term-rank) be `R`, normalized selected [axis](../../../../glossary.md#term-axis) be `a`, and let indices and updates share [Shape](../../../../glossary.md#shape)
`I`. [Model](../../../../glossary.md#term-model) requires `R > 0`, `rank(I) = R`, and `I[d] == data[d]` for every `d != a`; `I[a]` may
differ. For update [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `u`, read `v = indices[u]` and address:

```text
target(u)[d] = u[d]  when d != a
target(u)[a] = v
0 <= v < data[a]
```

The [validation](../../../../glossary.md#term-validation) ordinal is the flattened [row-major](../../../../glossary.md#term-row-major) ordinal in the complete indices/updates Shape.
For `NONE`, uniqueness means equality of the complete [target](../../../../glossary.md#term-target) coordinate above, not merely equality
of index values. Because non-axis coordinates participate, equal index values in different
non-axis positions need not be duplicates.

### [Gather](../../../../glossary.md#gather)-compatible `SCATTER_ADD`

Let normalized data [axis](../../../../glossary.md#term-axis) be `a` and indices [Shape](../../../../glossary.md#shape) be `I`. Updates have exactly ordinary Gather's
result Shape:

```text
updates.shape = data.shape[0:a] + I + data.shape[a+1:R]
```

For update [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `[before..., i..., after...]`, read `v = indices[i...]` and address
`[before..., v, after...]`, requiring `0 <= v < data[a]`. [Scalar](../../../../glossary.md#term-scalar) indices insert no [dimensions](../../../../glossary.md#dimension).
Every result [target](../../../../glossary.md#term-target) begins with its base and adds all contributions, including duplicates.
`SCATTER_ADD` has no `NONE` mode and therefore no duplicate rejection.

Bounds [validation](../../../../glossary.md#term-validation) visits the indices [tensor](../../../../glossary.md#tensor) once in its own logical [row-major](../../../../glossary.md#term-row-major) order even when the
same index value is reused across many data-prefix/data-suffix coordinates. Generated output may
reload validated index values while matching contributions.

### `SCATTER_ND`

Let data [rank](../../../../glossary.md#term-rank) be `R`, indices rank be `Q`, shared [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count be `B`, and [tuple depth](../../../../glossary.md#term-tuple-depth) be the static
positive final indices [extent](../../../../glossary.md#term-extent) `K`. [Model](../../../../glossary.md#term-model) requires `B < min(R, Q)`, `K <= R - B`, structurally
equal data/indices prefixes `[0, B)`, and:

```text
updates.shape = indices.shape[0:Q-1] + data.shape[B+K:R]
```

For one tuple-prefix [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `t`, tuple component `k` addresses data [axis](../../../../glossary.md#term-axis) `B + k` and must
satisfy:

```text
0 <= indices[t..., k] < data[B + k]
```

The complete [target](../../../../glossary.md#term-target) is the shared batch prefix, the `K` indexed components, and the update suffix
coordinate. Each tuple contributes its entire suffix [slice](../../../../glossary.md#slice) [scalar](../../../../glossary.md#term-scalar) by scalar. Bounds [validation](../../../../glossary.md#term-validation)
uses flattened [row-major](../../../../glossary.md#term-row-major) scalar order, including tuple components. `NONE` uniqueness is equality
of the target tuple within the same shared-batch prefix; equal tuples in different batches are not
duplicates. A duplicate tuple is invalid even when the suffix slice is empty, because Model makes
tuple uniqueness the replacement precondition.

### Reduction realization

For every result [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `c`, define `U(c)` as the logical multiset of addressed update [scalars](../../../../glossary.md#term-scalar).

- `NONE`: `U(c)` has size zero or one after validation. Size zero returns exact base bits; size one
  returns exact update bits. Base does not participate at an addressed target.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [target glossary](../../../../glossary.md#term-target))</span>
- `ADD`: include base once and every member of `U(c)` once. INT32 and INT64 use fixed-width
  two's-complement modular addition. FLOAT32 and FLOAT64 use deterministic row-major sequential
  same-format addition for this CPU route. BFLOAT16 expands each exact represented operand to
  binary32, adds it to the represented current BFLOAT16 accumulator, and rounds back to BFLOAT16
  after every logical contribution.
  These are CPU choices within Model's reassociation permission; scalar and parallel-scalar must
  agree bitwise for the same bound values and layouts.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [two's-complement glossary](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [row-major glossary](../../../../glossary.md#term-row-major); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [accumulator glossary](../../../../glossary.md#accumulator-type); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- `MUL`: INT32 and INT64 use fixed-width modular multiplication. Floating rows implement Model's
  one abstract unchanged-format product whenever `U(c)` is non-empty: include base and every
  update once, classify every represented factor, propagate NaN, make zero with infinity NaN,
  otherwise derive infinity or zero with complete negative-factor sign parity, and for all-finite
  non-zero factors multiply exact significands and exponent state then round once using
  round-to-nearest, ties-to-even to FLOAT64, FLOAT32, or BFLOAT16. Overflow, normal, subnormal,
  underflow, and signed-zero results follow that target. The result promises no NaN payload,
  source, signaling state, or sign. Empty <span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [finite glossary](../../../../glossary.md#term-finite); [exponent glossary](../../../../glossary.md#term-exponent); [round-to-nearest, ties-to-even glossary](../../../../glossary.md#floating-point-precision-and-rounding); [Overflow glossary](../../../../glossary.md#term-overflow); [subnormal glossary](../../../../glossary.md#term-subnormal); [underflow glossary](../../../../glossary.md#term-underflow); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`U(c)` copies exact base bits without entering the
  accumulator.
- `MIN` and `MAX`: include base once and every member of `U(c)` once. INT32/INT64 use signed order.
  Floating rows propagate any NaN; ordinary represented numeric order includes infinities; if
  both zero signs occur, <span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`MIN` yields [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) and `MAX` yields positive zero. No NaN payload,
  source, signaling state, or sign is promised. Empty <span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`U(c)` copies exact base bits.

The output-domain [scan](../../../../glossary.md#term-cumulative-scan) makes `MUL`, `MIN`, and `MAX` independent of physical [layout](../../../../glossary.md#layout), worker
scheduling, atomics, and tree [shape](../../../../glossary.md#shape), as [Model](../../../../glossary.md#term-model) requires. The selected deterministic `ADD` order is
a [CPU](../../../../glossary.md#term-cpu) realization, not a new Model guarantee.

### Exact floating-product workspace

Floating `MUL` uses one declared [run](../../../../glossary.md#term-run)-owned `CpuContiguousWorkspace`, generalized from its current
[materialization](../../../../glossary.md#materialization)-only wording to generic aligned [CPU](../../../../glossary.md#term-cpu) scratch. The plan computes an exact
per-range [slice](../../../../glossary.md#slice) size from the maximum number of factors that can address one result [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix),
including the base, and the [precision](../../../../glossary.md#term-precision) of the unchanged result format: 53 bits for [FLOAT64](../../../../glossary.md#term-float64), 24 for
[FLOAT32](../../../../glossary.md#term-float32), and 8 for [BFLOAT16](../../../../glossary.md#term-bfloat16). The slice contains fixed [metadata](../../../../glossary.md#term-metadata) plus enough unsigned 64-bit limbs
for the checked worst-case [significand](../../../../glossary.md#term-significand) product. Slice size and `selectedRangeCount * sliceSize`
use exact arithmetic, align to eight bytes, and [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) on [overflow](../../../../glossary.md#term-overflow).

Let `m` be the maximum updates in one [target](../../../../glossary.md#term-target) group. If the complete updates domain is empty,
`m = 0`; otherwise `m = indicesExtent[a]` for `SCATTER_ELEMENTS`, the checked product of every
indices [extent](../../../../glossary.md#term-extent) for [Gather](../../../../glossary.md#gather)-compatible `SCATTER_ADD`, or the checked product of indices extents
`[B, Q - 1)` for one `SCATTER_ND` [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters). The maximum factor count is `1 + m`. Scratch format
version one reserves three eight-byte words for flags/sign/special-value state, saturated [exponent](../../../../glossary.md#term-exponent)
state, and used-[limb](../../../../glossary.md#exact-superaccumulator-and-limbs) count, followed by
`ceil(precision * (1 + m) / 64)` [little-endian](../../../../glossary.md#text-encoding-and-file-boundaries) unsigned 64-bit limbs. The per-range slice size is
therefore `24 + 8 * limbCount` bytes; every multiplication/addition used to derive these values is
checked.

One workspace declaration with requirement ID zero covers all slices. A [scalar](../../../../glossary.md#term-scalar) call uses slice
zero; parallel range `i` uses slice `i`. The generated entry adds a [`MemorySegment`](../../../../glossary.md#term-memorysegment) scratch
[parameter](../../../../glossary.md#term-parameter) only for the floating-`MUL` [specialization](../../../../glossary.md#term-specialization), and cold packed geometry carries its slice
offset/capacity. Each output resets and reuses its range's limbs. Exponent state may saturate only
beyond proven final overflow/[underflow](../../../../glossary.md#term-underflow) decision thresholds; sign and special-value state remain
exact. The [hot path](../../../../glossary.md#term-hot-path) performs no `BigInteger`, collection, array, cursor, table, per-factor, or
per-output allocation.

The workspace is absent when no generated floating-product calculation can occur, including an
empty output or empty contribution domain. Scratch presence/format is explicit specialization
compatibility metadata. It is mutually exclusive with materialization, which scatter never
selects.

## [Validation](../../../../glossary.md#term-validation) order and failure-before-write contract

Cold analysis and [lowering](../../../../glossary.md#lowering) first reject unsupported kind/attrs/[signature](../../../../glossary.md#term-signature)/type/[Shape](../../../../glossary.md#shape)/static/[layout](../../../../glossary.md#layout)
or arithmetic [overflow](../../../../glossary.md#term-overflow). Finalization verifies every exact [buffer](../../../../glossary.md#term-buffer)/workspace assignment and worker
requirement before its sole [artifact](../../../../glossary.md#term-artifact) [lookup](../../../../glossary.md#term-lookup). [Binding](../../../../glossary.md#term-binding) then checks, in order:

1. representation count, exact type/carrier, byte size, alignment, lifetime/accessibility, output
   writability, and exact scratch size/alignment/accessibility when present;<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
2. full referenced spans, [injective](../../../../glossary.md#term-injective) output, and output non-overlap with every unique input;
3. [canonical](../../../../glossary.md#term-canonical-form) byte `0`/`1` for every logical [BOOL](../../../../glossary.md#data-type--datatype) data/update input [occurrence](../../../../glossary.md#term-occurrence);
4. complete logical [row-major](../../../../glossary.md#term-row-major) bounds validation of every index [scalar](../../../../glossary.md#term-scalar);
5. for `NONE`, complete logical row-major [target](../../../../glossary.md#term-target)-uniqueness validation.

Only after all five stages succeed may generated execution begin or any worker be submitted. Any
failure leaves every output byte unchanged. Bounds always precede duplicates, so mixed-invalid
input reports the first out-of-bounds scalar rather than a duplicate.

Exact [runtime](../../../../glossary.md#term-runtime) failures are:

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
primitive [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) state and direct [carrier](../../../../glossary.md#term-storage-carrier) loads, not a [hash](../../../../glossary.md#term-hash) table or execute-time allocation.

Validation and output have independent zero domains. An empty index domain succeeds without a
load. A zero selected [extent](../../../../glossary.md#term-extent) rejects every encountered index. A zero output caused by another
extent still validates all indices and `NONE` tuples, then makes no generated call and submits no
worker work. `SCATTER_ELEMENTS` with an empty selected update extent, or any valid empty
contribution domain, copies base when output is non-empty. `SCATTER_ND + NONE` still checks tuple
uniqueness when its suffix [slice](../../../../glossary.md#slice) is empty.

## [Layouts](../../../../glossary.md#layout), carriers, ranges, [aliasing](../../../../glossary.md#term-aliasing), and strategy

- Inputs may have resolved arbitrary non-negative offsets and strides, including read-zero
  strides and repeated physical reads. Output must have a resolved injective layout.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective))</span>
- [Heap](../../../../glossary.md#term-heap) carriers are exactly `double[]`, `float[]`, raw [BFLOAT16](../../../../glossary.md#term-bfloat16) `short[]`, `int[]`, `long[]`, and
  canonical BOOL <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`byte[]`; native carriers are native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment). Every ordered mixed
  pattern admitted by current carrier rules is supported.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Input/input physical overlap is allowed, including exact [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) deduplication. Output may not
  overlap the referenced span of any unique input, even though output-domain writing would often
  appear safe; this preserves the current functional boundary and prevents parallel read/write
  races. Empty referenced spans follow the current non-overlap convention.<span data-glossary-navigation="true"> ([referenced span glossary](../../../../glossary.md#referenced-element-span))</span>
- Logical row-major order is defined by Shape coordinates and is independent of physical address
  order, strides, aliases, or carrier form.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [Shape glossary](../../../../glossary.md#shape); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Each generated range owns `[start, end)` output ordinals. Cold binding allocates and packs one
  primitive geometry array per selected call, seeds output coordinates, update-scan cursors, and
  optional scratch offset, and reuses it for every invocation. The generated hot loop advances
  odometers without division/modulo, Model inspection, semantic lookup, or allocation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [scan glossary](../../../../glossary.md#term-cumulative-scan); [Model glossary](../../../../glossary.md#term-model); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Scalar preference selects scalar. Vector preference deliberately falls back to scalar compute.
  Existing cold policy may select parallel orchestration by output count; zero output selects no
  call. A range never writes outside its output interval and never shares mutable scratch with
  another range.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [Vector glossary](../../../../glossary.md#term-vector); [mutable glossary](../../../../glossary.md#term-mutable))</span>

## Resources, preparation, finalization, [cache](../../../../glossary.md#term-cache), and [schema](../../../../glossary.md#term-schema)

- [Lowering](../../../../glossary.md#lowering) produces one unit, one `CpuScatterIr`, one `CpuScatterLowering.Geometry`, unique input
  boundaries in semantic first-use order, then one distinct output, and output extents/count.
- Analysis declares exactly two through four [buffers](../../../../glossary.md#term-buffer) depending on input [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) deduplication.
  Inputs are read-only and output is write-only. No scatter materialization is allowed.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- Non-floating-`MUL` plans declare zero workspaces. An eligible floating-`MUL` plan declares the
  single exact scratch workspace described above. No resource is inferred or added after shared
  assignment.
- `CpuPartitionPreparationPlan` must distinguish materialization workspace use from scatter
  product scratch while retaining the current maximum of one workspace. Finalization verifies
  exact identity/size/alignment and one matching assignment before one artifact-store call.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [alignment glossary](../../../../glossary.md#term-alignment); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- `CpuPreparedExecutable` retains only prepared scatter geometry, validation state, the optional
  workspace selection, direct buffers, one artifact, and optional borrowed workers. Runtime sees
  no Operation, node, attrs, reduction enum, Tensor, cache choice, or fallback.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Runtime glossary](../../../../glossary.md#term-runtime); [Operation glossary](../../../../glossary.md#operation); [node glossary](../../../../glossary.md#node); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Structural IR/artifact identity includes scatter family, reduction, data/index types, ranks,
  occurrence map, normalized structural access forms, output store shape, and scratch signature/
  format. Specialization includes exact ordered carrier pattern and whether the generated entry
  accepts scratch. Axis, batch count, tuple depth, extents, offsets, stride magnitudes, element
  counts, range bounds/count, workspace byte count/offset, slots, addresses, run identity, and
  worker identity remain cold compatible facts.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [occurrence glossary](../../../../glossary.md#term-occurrence); [shape glossary](../../../../glossary.md#shape); [signature glossary](../../../../glossary.md#term-signature); [Specialization glossary](../../../../glossary.md#term-specialization); [carrier glossary](../../../../glossary.md#term-storage-carrier); [Axis glossary](../../../../glossary.md#term-axis); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [tuple depth glossary](../../../../glossary.md#term-tuple-depth); [stride glossary](../../../../glossary.md#term-stride); [element glossary](../../../../glossary.md#term-element); [run glossary](../../../../glossary.md#term-run))</span>
- Schema becomes 16. Persistence treats every older schema as a miss and regenerates through the
  existing atomic protocol; no compatibility reader or migration is added.<span data-glossary-navigation="true"> ([Persistence glossary](../../../../glossary.md#term-persistence); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- `FOLD_AXIS`, `FOLD2D`, CPU 0006B2, stable ordering/top-K CPU 0006C, explicit-state random,
  native/vendor routes, tuning, benchmarking, or performance claims<span data-glossary-navigation="true"> ([benchmarking glossary](../../../../glossary.md#term-benchmarking))</span>
- changing any Model operation kind, attrs, reduction, Tensor construction, Shape rule,
  represented-value meaning, Javadoc, test, or public API<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- changing Compiler capture, canonicalization, autograd, publication, decomposition, or shared
  semantics; CPU consumes the already current scatter occurrences<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- dynamic/symbolic CPU Shapes, unresolved layouts, negative storage strides, deferred layout
  binding, non-injective/in-place output, or output/input overlap<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [binding glossary](../../../../glossary.md#term-binding); [injective glossary](../../../../glossary.md#term-injective))</span>
- negative-index normalization, ignored indices, clamping, padding targets, first/last duplicate
  replacement, atomic scatter, unordered reduction, update-centric parallelism, or partial output<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [padding glossary](../../../../glossary.md#padding); [targets glossary](../../../../glossary.md#term-target); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- general multi-node indexing/scatter fusion, mixed-family partition DAGs, safe split fallback,
  multi-output units, a Runtime semantic interpreter, or a selectable reference fallback<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [fusion glossary](../../../../glossary.md#term-fusion); [partition glossary](../../../../glossary.md#partition); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- vector scatter/gather, SIMD reduction, relaxed numerics, cross-type conversion/promotion,
  FLOAT16, alternate BFLOAT16 storage, or BOOL arithmetic<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [gather glossary](../../../../glossary.md#gather); [SIMD glossary](../../../../glossary.md#simd); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- a per-index/per-target address table, hash-based duplicate table, hidden mutable global/thread-
  local state, undeclared scratch, execute-time <span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash); [mutable glossary](../../../../glossary.md#term-mutable))</span>`BigInteger`, or avoidable hot-path allocation
- `ARCHITECTURE.md`, architecture explanations, ADRs, architecture tests, shared Prepare/Runtime
  contracts, Gradle/dependencies, backend-conformance, integration, vendor code, or unrelated
  refactors<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- repository-wide [validation](../../../../glossary.md#term-validation) outside [CPU](../../../../glossary.md#term-cpu) 0009 and [CI](../../../../glossary.md#term-continuous-integration-ci)

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Backend guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0005E represented types/carriers](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
- [CPU 0005F floating division/power](0005f-floating-division-and-exact-scalar-power-realization.md)
- [CPU 0005G represented extrema](0005g-extrema-clamp-tensor-power-and-logical-coverage.md)<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- [CPU 0005H unary closure](0005h-portable-unary-transcendental-and-activation-closure.md)<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>
- [CPU 0005I FLOAT32 vector parity](0005i-float32-vector-parity-and-vector-emission-boundary.md)<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [vector glossary](../../../../glossary.md#term-vector))</span>
- [CPU 0005J bounded parity hardening](0005j-bounded-pointwise-coverage-and-parity-hardening.md)
- [CPU 0006A2 Gather and one-hot](0006a2-portable-gather-and-one-hot-indexing.md)<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather))</span>
- [CPU 0006B functional slice update](0006b-portable-functional-slice-update.md)<span data-glossary-navigation="true"> ([slice update glossary](../../../../glossary.md#slice-update))</span>
- [Model 0018G axis-scatter semantics](../../../modules/model/tasks/0018g-axis-scatter-semantics.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [axis glossary](../../../../glossary.md#term-axis))</span>
- [Model 0018H axis-scatter expressions](../../../modules/model/tasks/0018h-axis-scatter-tensor-expressions.md)
- [Model 0018I Scatter-ND semantics](../../../modules/model/tasks/0018i-scatter-nd-semantics.md)<span data-glossary-navigation="true"> ([Scatter-ND glossary](../../../../glossary.md#scatter-nd))</span>
- [Model 0018J Scatter-ND expressions](../../../modules/model/tasks/0018j-scatter-nd-tensor-expression.md)
- [Model 0018O indexing normalization](../../../modules/model/tasks/0018o-indexing-taxonomy-and-unstack-normalization.md)<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- [Model 0023B Gather-compatible Scatter Add](../../../modules/model/tasks/0023b-gather-compatible-scatter-add.md)<span data-glossary-navigation="true"> ([Scatter Add glossary](../../../../glossary.md#scatter-add))</span>
- [Model 0025C functional-scatter reductions](../../../modules/model/tasks/0025c-portable-functional-scatter-reduction-semantics.md)
- [Tensor API](../../../../api/tensor-api.md#axis-scatter-semantic-kinds-reduction-and-attributes)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compile API scatter contract](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Functional-scatter target group](../../../../glossary.md#functional-scatter-target-group)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative. Model owns scatter meaning. CPU owns exact occurrence
  capability, lowering, code-shaping identity, strategy, resource declarations, artifact
  realization, validation, and direct execution without changing shared meaning.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu); [occurrence glossary](../../../../glossary.md#term-occurrence); [lowering glossary](../../../../glossary.md#lowering); [identity glossary](../../../../glossary.md#term-identity-meanings); [artifact glossary](../../../../glossary.md#term-artifact); [validation glossary](../../../../glossary.md#term-validation))</span>
- Planning has already assigned the occurrence to CPU. Shared Prepare receives only exact opaque
  buffer/workspace declarations. CPU finalization verifies assignments and workers before its
  single artifact lookup and may not revise lowering, route, strategy, or resources.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque); [buffer glossary](../../../../glossary.md#term-buffer); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [Runtime](../../../../glossary.md#term-runtime) receives one [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable) and direct resource selections. No [Operation](../../../../glossary.md#operation),
  `CompiledNode`, attrs, Tensor metadata, reduction selection, cache lookup, or fallback enters
  Runtime execution.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [cache glossary](../../../../glossary.md#term-cache))</span>
- The portable generated route remains the sole route. The [scalar](../../../../glossary.md#term-scalar) reference is test evidence only.
- All value-dependent failure occurs before output mutation. No worker may observe or expose a
  partially validated scatter.
- Exact product scratch is one explicitly declared backend-private workspace, not a new shared
  resource kind or implicit allocator. Its use must remain compatible with the current workspace
  lifecycle.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- If implementation needs a shared/public type, another module, dependency/build change,
  architecture rule, Runtime semantic branch, second workspace, non-injective/in-place output, or
  any semantic assumption not fixed here, stop and report the conflict rather than broaden scope.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [injective glossary](../../../../glossary.md#term-injective); [scope glossary](../../../../glossary.md#term-scope))</span>

## Package impact and type placement

Existing packages used:

- `backend.cpu` for truthful capability.
- `internal.ir` for the closed scatter structural [identity](../../../../glossary.md#term-identity-meanings).
- `internal.lowering` for one-[node](../../../../glossary.md#node) semantic revalidation and compact geometry.
- `internal.codegen.emit` for the generated bridge, output scan, typed carrier access, reductions,
  and primitive-limb exact product.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `internal.prepare` and `internal.executable` for exact scratch declaration/selection, complete
  pre-write validation, cold range packing, and direct invocation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `internal.memory` for generic aligned [run](../../../../glossary.md#term-run)-owned [CPU](../../../../glossary.md#term-cpu) scratch.
- `internal.cache` and `internal.route.portable` for [schema](../../../../glossary.md#term-schema)-16 identity and [artifact](../../../../glossary.md#term-artifact) realization.
- `internal.reference` for an independent [scalar](../../../../glossary.md#term-scalar) [oracle](../../../../glossary.md#term-test-oracle).

Packages added, moved, or removed: none.

New top-level production types:

- `CpuScatterIr` in `internal.ir`, because scatter structural identity is not gather indexing or
  value-blind movement.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather))</span>
- `CpuScatterLowering` in `internal.lowering`, including [sealed](../../../../glossary.md#term-sealed)/nested [Axis](../../../../glossary.md#term-axis) and [ND](../../../../glossary.md#term-nd) cold geometry.
- `CpuScatterEmitter` in `internal.codegen.emit`, owning generated bridge and execution helpers.

Exact-product [limb](../../../../glossary.md#exact-superaccumulator-and-limbs) arithmetic remains a focused private/nested implementation under
`CpuScatterEmitter`; do not add a generic math utility or shared numerical abstraction.

## Affected files

Authorized [CPU](../../../../glossary.md#term-cpu) production/package paths:

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

Authorized explanatory and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0006b1-portable-functional-scatter.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized. Listed paths change only when their implementation contract,
[Javadoc](../../../../glossary.md#term-javadoc), package summary, [regression](../../../../glossary.md#term-regression) [ownership](../../../../glossary.md#term-ownership-meanings), or synchronized status is affected. Both clean
handoffs must record review-only no-change conclusions for unused authorized paths.

Review without modification:

- Model, Compiler, Tensor/API documentation, architecture, ADR, architecture-test,
  backend-conformance, integration, shared-module, Gradle/dependency, and vendor paths remain
  evidence only. Current API and compiler guides already state the Model scatter contracts and do
  not promise CPU coverage; change them only after revising this allowlist.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Existing gather/indexing and movement IR/lowering/emitter source and tests remain unchanged.
  Scatter uses their lifecycle patterns without weakening or merging their contracts.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

| Category | Maximum | Accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production/package | 26 | 3 new [owners](../../../../glossary.md#term-owner) plus affected capability, generator, [lifecycle](../../../../glossary.md#lifecycle), scratch, [cache](../../../../glossary.md#term-cache), route, reference, and package contracts |
| CPU tests | 12 | Focused new family plus existing lifecycle/cache/reference [regression](../../../../glossary.md#term-regression) owners |
| Explanatory documentation | 2 | CPU guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **43** | **26 + 12 + 2 + 3** |

Do not spend unused scope on cleanup. Stop and request a revised planning decision before a 44th
path or any path outside the exact allowlist.

## Acceptance criteria

- Capability and lowering admit only the three exact current scatter forms and independently
  reject every wrong kind/attrs/input count/output count/type/index type/Shape/static/layout/
  output-injectivity combination.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [layout glossary](../../../../glossary.md#layout); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Current `SCATTER_ADD` uses the exact [Gather](../../../../glossary.md#gather)-compatible updates [Shape](../../../../glossary.md#shape). No historical
  `SCATTER_AXIS_ADD` or superseded reduced-[rank](../../../../glossary.md#term-rank) shape is implemented.
- Both index types, all permitted data types/reductions, scalar Shapes where legal, zero extents,
  non-dense/read-zero input layouts, injective output layouts, heap/segment/mixed carriers, and
  input deduplication are covered.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [injective glossary](../../../../glossary.md#term-injective); [heap glossary](../../../../glossary.md#term-heap))</span>
- Bounds and `NONE` duplicate validation follow the exact mappings, order, exception types, and
  messages above. Tests prefill output sentinels and prove every invalid case fails before any
  write or worker submission, including zero-output cases.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `NONE` copies exact update bits at unique targets and exact base bits elsewhere. Duplicate
  element targets and duplicate ND tuples are rejected; same index/tuple in distinct non-target
  coordinates or batches remains valid.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [element glossary](../../../../glossary.md#term-element); [ND glossary](../../../../glossary.md#term-nd); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Every arithmetic target includes base once and every update once; duplicate contributions
  accumulate. Empty groups preserve exact base bits.
- INT32/INT64 ADD/MUL are modular; extrema use signed order. FLOAT64/FLOAT32/BFLOAT16 cover normal,
  subnormal, overflow/underflow, signed zero, infinity, NaN, ties-to-even, zero-times-infinity,
  negative-sign parity, and multiple-factor cases against the independent oracle.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [subnormal glossary](../../../../glossary.md#term-subnormal); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [signed zero glossary](../../../../glossary.md#term-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Floating `MUL` exactly matches the once-rounded abstract product and uses one exact declared
  per-range-sliced workspace with no hot-path allocation. Every other row has no workspace.
- Scalar and parallel-scalar produce identical results and failures. Parallel tests cover multiple
  chunks, mixed carriers, disjoint scratch slices, repeat invocation, and no races/atomics/merge.
- Output/input overlap is rejected before validation/writes; input/input overlap and exact input
  deduplication work. Inputs remain byte-for-byte unchanged.
- Arbitrary output ranges write only their interval. Packed geometry contains no per-index or
  per-output table, and generated execution performs no per-element allocation, Model dispatch,
  bounds validation, division, or modulo.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Analysis declares exact buffers, optional workspace, one unit, one artifact, and correct
  scalar/parallel-scalar strategy. Finalization accepts only exact assignments and performs one
  schema-16 artifact lookup after assignment validation.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [artifact glossary](../../../../glossary.md#term-artifact); [schema glossary](../../../../glossary.md#term-schema); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Structural cache tests prove compatible extent/layout/range/workspace-size changes reuse bytes,
  while family/reduction/type/rank/access/carrier/scratch-signature changes separate artifacts;
  schema 15 is an incompatible miss.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [extent glossary](../../../../glossary.md#term-extent); [carrier glossary](../../../../glossary.md#term-storage-carrier); [signature glossary](../../../../glossary.md#term-signature))</span>
- Independent differential tests do not call generated helpers or share the primitive-limb
  implementation. Expected represented values come from a separately implemented oracle.<span data-glossary-navigation="true"> ([limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs))</span>
- Every changed Java API/implementation contract has meaningful complete Javadoc; affected
  package summaries, CPU guide, glossary, task evidence, master plan, and roadmap are finalized in
  the mandatory clean documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Architecture contracts, Model/Compiler/shared source, dependencies, and later tasks remain
  unchanged. Any need to change them makes this task incomplete pending replanning.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>

## Tests and [validation](../../../../glossary.md#term-validation)

Focused implementation command, [run](../../../../glossary.md#term-run) after production and focused tests stabilize:

```bash
./gradlew :backends:cpu:test --tests '*CpuCapabilityProviderTest' --tests '*CpuScatterIrTest' --tests '*CpuScatterLoweringTest' --tests '*CpuScatterGeneratedKernelTest' --tests '*CpuPartitionPreparerTest' --tests '*CpuPartitionFinalizerTest' --tests '*CpuPreparedExecutableTest' --tests '*CpuGeneratedKernelArtifactStoreTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuShapePolymorphicArtifactTest' --tests '*CpuReferenceDifferentialTest' --tests '*CpuInternalPackageInventoryTest'
```

Final affected-[module](../../../../glossary.md#term-module) validation, run once after the focused command passes:

```bash
./gradlew :backends:cpu:test
./gradlew :backends:cpu:javadoc
```

The clean documentation context runs link/anchor/fence/newline/trailing-whitespace checks for the
changed Markdown/[Javadoc](../../../../glossary.md#term-javadoc) surface, inspects generated Javadoc pages, verifies exact changed paths
and status/[dependency](../../../../glossary.md#term-dependency) [synchronization](../../../../glossary.md#term-synchronization), then runs:

```bash
git diff --check
```

This is a normal single-module task. Repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and
integration suites are deferred to [CPU](../../../../glossary.md#term-cpu) 0009 and [CI](../../../../glossary.md#term-continuous-integration-ci) because no dependency/module boundary, shared
contract, or current public [Engine](../../../../glossary.md#term-engine) path changes. If implementation makes that conclusion false,
stop and replan before changing another tier. The documentation context does not repeat successful
Java tests unless it changes executable Java or records a concrete stale-evidence reason.

## Dependencies

- [CPU 0006B](0006b-portable-functional-slice-update.md) is `Complete` and supplies current
  schema-15 generated movement, output-domain range, alias, carrier, and lifecycle evidence.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [alias glossary](../../../../glossary.md#aliasing-and-arena); [carrier glossary](../../../../glossary.md#term-storage-carrier); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [CPU 0006A2](0006a2-portable-gather-and-one-hot-indexing.md) is `Complete` and supplies mixed
  typed indexing carriers, compact geometry, deterministic complete pre-write validation,
  scalar/parallel-scalar output generation, and independent reference seams.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- [CPU](../../../../glossary.md#term-cpu) 0005E through 0005J are `Complete` and supply represented carriers, exact signed integral
  arithmetic, floating classification/extrema/BFLOAT16 conventions, generated specialization,
  range strategy, cache, and parity foundations.<span data-glossary-navigation="true"> ([floating classification glossary](../../../../glossary.md#floating-classification); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [specialization glossary](../../../../glossary.md#term-specialization); [cache glossary](../../../../glossary.md#term-cache))</span>
- [Model 0023B](../../../modules/model/tasks/0023b-gather-compatible-scatter-add.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span> is `Complete`
  and owns the current fixed-add Gather-compatible shape.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [shape glossary](../../../../glossary.md#shape))</span>
- [Model 0025C](../../../modules/model/tasks/0025c-portable-functional-scatter-reduction-semantics.md)
  is `Complete` and owns current base/target-group, represented floating product/extrema, and
  order-independence semantics, superseding incomplete historical prose.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Current Model scatter kinds/attrs/reduction, Tensor construction, operation signatures, Shape,
  layout, carrier, CPU preparation/finalization/executable, workspace, worker, artifact, and
  reference contracts named above.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature); [layout glossary](../../../../glossary.md#layout); [artifact glossary](../../../../glossary.md#term-artifact))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0006B2 remains `Draft`, depends on CPU 0006B1, and owns `FOLD_AXIS`/`FOLD2D` zero
  initialization, uncovered positions, overlap addition, NCHW column mapping, padding/ceil-tail
  exclusion, resources, and safe split behavior. Do not create its task file here.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [padding glossary](../../../../glossary.md#padding))</span>
- CPU 0006C remains `Draft`, depends on CPU 0006B2, and owns stable ordering/selection.
- Later ordering, explicit-state random, native, tuning, dynamic layout, and general partition-DAG
  work retain their existing roadmap ownership and status.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Architecture impact

Expected impact: None.

The architecture already assigns semantic meaning to [Model](../../../../glossary.md#term-model), [CPU](../../../../glossary.md#term-cpu) route/[lowering](../../../../glossary.md#lowering)/resource choices
to [backend](../../../../glossary.md#backend) analysis, [opaque](../../../../glossary.md#term-opaque) assignment to [Prepare](../../../../glossary.md#prepare), generated [artifact](../../../../glossary.md#term-artifact) realization to CPU
finalization, and direct invocation to [Runtime](../../../../glossary.md#term-runtime). A CPU-private scatter family and one explicitly
declared workspace reuse those boundaries.

If implementation requires an [architecture contract](../../../../glossary.md#architecture-contract), focused architecture document, [ADR](../../../../glossary.md#term-architecture-decision-record-adr),
[architecture test](../../../../glossary.md#term-architecture-test), [dependency edge](../../../../glossary.md#term-dependency-edge), shared/public [lifecycle](../../../../glossary.md#lifecycle) contract, or another [module](../../../../glossary.md#term-module) to
change, stop and report the exact conflict rather than implement it under this task.

## Implementation prompt

Use this prompt for the mandatory clean implementation [handoff](../../../../glossary.md#term-handoff):

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

- Use a distinct scatter family rather than extending [gather](../../../../glossary.md#gather) indexing or value-blind movement.
- Produce by disjoint output coordinates and scan logical contributions. This preserves functional
  base participation, one final write, deterministic parallelism, and zero atomic/merge state.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [scan glossary](../../../../glossary.md#term-cumulative-scan); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Validate every bound first, then `NONE` uniqueness, before any generated work. Use deterministic
  allocation-free comparison rather than a target hash table.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [hash glossary](../../../../glossary.md#term-hash))</span>
- Use a deterministic CPU row-major order for ADD while preserving Model's broader reassociation
  permission. Do not export that order as Model semantics.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [row-major glossary](../../../../glossary.md#term-row-major); [Model glossary](../../../../glossary.md#term-model); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Use explicitly declared per-range primitive-limb scratch for floating MUL so the abstract exact
  product rounds once without hot-path arbitrary-precision allocation.<span data-glossary-navigation="true"> ([limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [precision glossary](../../../../glossary.md#term-precision))</span>
- Permit input/input [alias](../../../../glossary.md#aliasing-and-arena) and [ValueId](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) deduplication; reject every output/input overlap.
- Keep extents/layout magnitudes/ranges/workspace sizes cold; include family/reduction/type/access/
  carrier/scratch signature in generated compatibility; advance to schema 16.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier); [signature glossary](../../../../glossary.md#term-signature); [schema glossary](../../../../glossary.md#term-schema))</span>
- Use a separately implemented arbitrary-precision reference oracle for differential evidence,
  never as Runtime fallback.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Known limitations

- Only one fully static resolved-layout scatter occurrence is executable. Tensor expressions
  normally remain unresolved until existing lifecycle work supplies concrete descriptors.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [occurrence glossary](../../../../glossary.md#term-occurrence); [Tensor glossary](../../../../glossary.md#tensor); [lifecycle glossary](../../../../glossary.md#lifecycle); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- The output must be distinct, writable, [injective](../../../../glossary.md#term-injective), and physically non-overlapping with inputs.
- The algorithm favors semantic closure and deterministic independent output ranges; it makes no
  complexity or performance claim and may rescan the update domain for each output.
- Vector compute, update-centric atomics, native routes, tuning, dynamic binding, multi-node
  fusion, and safe split remain planned elsewhere.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [binding glossary](../../../../glossary.md#term-binding); [node glossary](../../../../glossary.md#node); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Floating ADD has a deterministic CPU realization but no Model bitwise-order promise. Floating
  MUL is once-rounded exact; MIN/MAX preserve only the Model-promised NaN and signed-zero result,
  not NaN payload/source/sign.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Floating-MUL workspace can be large for large contribution groups. Exact checked declaration or
  shared allocation failure rejects preparation/run; there is no hidden heap fallback.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [heap glossary](../../../../glossary.md#term-heap))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context read the governing repository/architecture/planning contracts, [CPU](../../../../glossary.md#term-cpu) master plan
and predecessor tasks, current CPU capability/[IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/preparation/finalization/generated/
execution/reference/[cache](../../../../glossary.md#term-cache) [owners](../../../../glossary.md#term-owner), current [Model](../../../../glossary.md#term-model) scatter kinds/attrs/reduction/[Tensor](../../../../glossary.md#tensor) contracts,
completed Model correction/reduction tasks, and current [API](../../../../glossary.md#term-application-programming-interface-api)/CPU-guide/glossary statements. Current
source resolved the stale historical scatter taxonomy and [Gather](../../../../glossary.md#gather)-compatible [shape](../../../../glossary.md#shape). No architecture
conflict or unresolved semantic decision remains.

Planning-stage validation passed on exactly this new task, the CPU master plan, and the roadmap.
Repository-local Markdown paths and heading anchors, balanced fences, final newlines, trailing
whitespace, [canonical](../../../../glossary.md#term-canonical-form) headings, exact changed-path inventory, the sole-[Ready](../../../../glossary.md#term-ready) 0006B1 status,
0006B -> 0006B1 -> 0006B2 -> 0006C [dependency](../../../../glossary.md#term-dependency) chain, and `git diff --check` all passed. No Java,
test, build, architecture, API/guide/glossary, or later-task file changed, and no Java test or
[Javadoc](../../../../glossary.md#term-javadoc) command ran for this planning-only change.

Implementation context `019ff230-109c-73a3-933f-611ee7f6143d` implemented the CPU-private
scatter family and its focused [regression](../../../../glossary.md#term-regression) [matrix](../../../../glossary.md#term-matrix-meanings). Independent audit/fix context
`019ff248-a9e4-7150-8fbb-db2730d7cc1b` reviewed the complete uncommitted implementation and
repaired four production defects plus one reference-[oracle](../../../../glossary.md#term-test-oracle) defect. After the final executable/test
edit, the prescribed twelve-[owner](../../../../glossary.md#term-owner) gate passed 12 suites and 103 tests with 0 skipped, failures,
or errors. The sole final `./gradlew :backends:cpu:test` passed 38 suites and 230 tests with one
expected existing skip and 0 failures or errors on Java/OpenJDK 26.0.1+8-34 and [Gradle](../../../../glossary.md#term-gradle) 9.6.1.
No executable Java statement or test changed after that evidence.

The mandatory clean documentation-focused context ID was
`019ff4fb-94c7-7921-af13-28a7395c3ae7`.
This context independently reviewed the final production/test diff, current Model scatter
contracts and completed tasks, affected public/generated contract owners, package summaries, CPU
guide, glossary, and planning records under the General, API/Javadoc, [Backend](../../../../glossary.md#backend) Guide, Planning, and
Example profiles. It finalized twelve Java/package documentation paths plus the two explanatory
and three planning/status documents. The first `./gradlew :backends:cpu:javadoc` [run](../../../../glossary.md#term-run) succeeded but
reported 23 actionable missing-[parameter](../../../../glossary.md#term-parameter)/description warnings plus the two expected incubating-
[Vector](../../../../glossary.md#term-vector)-[module](../../../../glossary.md#term-module) warnings. Javadoc-only corrections were applied. The second run succeeded with only
the two incubating-module warnings. Generated pages for `CpuScatterIr`, `CpuScatterLowering`,
`CpuScatterEmitter`, `CpuPreparedExecutable`, `CpuKernelSpecialization`,
`CpuScalarReferenceKernel`, and all affected package summaries were inspected.

Final focused checks confirmed local Markdown file [targets](../../../../glossary.md#term-target) and heading anchors, balanced fences,
final newlines, trailing whitespace, [schema](../../../../glossary.md#term-schema) 16/current-[operation](../../../../glossary.md#operation) wording, absence of historical
`SCATTER_AXIS_ADD` as a current operation, synchronized 0006B1 `Complete` and next-frontier 0006B2
`Draft` status/dependencies, absence of an unauthorized 0006B2 task specification, exact allowlist
membership, a changed-path union no greater than 43, and `git diff --check`. Java tests were not
repeated because this context changed only Javadocs, package summaries, explanatory docs, and
planning records.

## Implementation notes

- Added one distinct `CpuScatterIr`/`CpuScatterLowering`/`CpuScatterEmitter` family and connected
  it to capability, preparation, finalization, generated invocation, execution binding, reference,
  and schema-16 artifact compatibility owners without changing shared modules.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [schema glossary](../../../../glossary.md#term-schema); [artifact glossary](../../../../glossary.md#term-artifact); [modules glossary](../../../../glossary.md#term-module))</span>
- The independent audit corrected exponent clamping under opposing exponent cancellation, packed
  coordinate reuse across repeated partial ranges, base reads when data/output layouts differ,
  insufficient IR/geometry validation, and FLOAT32/BFLOAT16 MIN/MAX oracle boxed-result typing.<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent); [cancellation glossary](../../../../glossary.md#term-cancellation); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [layouts glossary](../../../../glossary.md#layout); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [validation glossary](../../../../glossary.md#term-validation); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Direct regression coverage spans the complete type/reduction matrix, both index types, scalar
  and zero domains, arbitrary layouts, mixed carriers, modular overflow, floating special values
  and rounding, exponent cancellation, parallel scratch/repeat, validation before workers,
  overlap/deduplication, cache compatibility, and malformed finalization.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [overflow glossary](../../../../glossary.md#term-overflow); [rounding glossary](../../../../glossary.md#term-rounding); [cache glossary](../../../../glossary.md#term-cache))</span>
- Documentation preserves the current-source correction: `AxisScatterKind` contains only
  `SCATTER_ELEMENTS` and [Gather](../../../../glossary.md#gather)-compatible `SCATTER_ADD`; historical `SCATTER_AXIS_ADD` and its
  superseded shape remain historical evidence only.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>

## Completion summary

- Completed changes: implemented and documented current portable functional scatter for exactly
  one fully static resolved-layout CPU occurrence, including deterministic validation, represented
  reductions, disjoint output ownership, optional exact floating-product scratch, and schema 16.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [occurrence glossary](../../../../glossary.md#term-occurrence); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [schema glossary](../../../../glossary.md#term-schema))</span>
- Files changed or created: the exact allowlisted CPU production/test paths recorded by the final
  changed-path inventory plus, in this documentation context, `CpuKernelSpecialization.java`,
  `CpuPartitionLowering.java`, `CpuScatterLowering.java`; package summaries for cache,
  codegen/emit, executable, IR, lowering, memory, prepare, reference, and route/portable;<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [prepare glossary](../../../../glossary.md#prepare))</span>
  `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan, and the
  roadmap. No file was created by this documentation context.
- Tests and validation: reused the stabilized 12-suite/103-test focused and 38-suite/230-test CPU
  evidence above; ran CPU Javadoc twice as recorded; inspected generated pages; passed final
  Markdown/status/scope/schema checks and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.
- Documentation-agent review: complete in context `019ff4fb-94c7-7921-af13-28a7395c3ae7`.
- Documentation impact: the CPU guide now explains exact current scatter coverage and boundaries;
  API/Compiler guides remain accurate because they already describe Model semantics without
  promising CPU coverage.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model))</span>
- Javadoc review: every changed implementation contract was reviewed; actionable warnings were
  corrected, and rendered IR/lowering/emitter plus lifecycle/cache/reference/package contracts
  were inspected.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Glossary impact: current CPU portable-route, Scatter Add, Scatter Elements, Scatter-ND, slice
  update, and kernel-specialization entries now distinguish schema-16 CPU execution from Model
  construction and future routes.<span data-glossary-navigation="true"> ([Scatter Add glossary](../../../../glossary.md#scatter-add); [Scatter Elements glossary](../../../../glossary.md#scatter-elements); [Scatter-ND glossary](../../../../glossary.md#scatter-nd); [slice update glossary](../../../../glossary.md#slice-update); [kernel glossary](../../../../glossary.md#kernel); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- No-change conclusions: Model, Compiler, shared Prepare/Runtime, public Tensor/API, architecture,
  ADRs, architecture tests, backend conformance, integration tests, Gradle/dependencies,
  vendor/native code, existing gather/movement families, and later task specifications required no
  change because this task adds only a CPU-private route within existing ownership and lifecycle
  contracts. CPU 0006B2 remains Draft without a detailed specification.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Tensor glossary](../../../../glossary.md#tensor); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [gather glossary](../../../../glossary.md#gather))</span>
- Unresolved issues: None.
- Follow-up required: None; [CPU](../../../../glossary.md#term-cpu) 0006B2 is the next Draft [planning](../../../../glossary.md#term-planning) frontier.

Status: Complete
