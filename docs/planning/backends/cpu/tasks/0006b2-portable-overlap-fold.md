# Task 0006B2: Portable Overlap Fold

## Status

Complete

## Goal

Add the next executable [CPU](../../../../glossary.md#term-cpu) frontier after completed CPU 0006B1: exactly one fully static,
resolved-[layout](../../../../glossary.md#layout) current [Model](../../../../glossary.md#term-model) `FOLD_AXIS` or `FOLD2D` [occurrence](../../../../glossary.md#term-occurrence) through the existing whole-
[partition](../../../../glossary.md#partition) portable generated-[kernel](../../../../glossary.md#kernel) architecture.

The result is a fresh, zero-initialized [Tensor](../../../../glossary.md#tensor). Every logical input occurrence that maps to one
result [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) is accumulated exactly once, in [canonical](../../../../glossary.md#term-canonical-form) logical input [row-major](../../../../glossary.md#term-row-major) order, using
the represented result type's addition semantics. `FOLD2D` excludes every column position whose
padded/dilated coordinate lies outside the unpadded [NCHW](../../../../glossary.md#term-nchw) output. Generated [scalar](../../../../glossary.md#term-scalar) and parallel-
scalar execution must agree bitwise; parallel work is legal only as disjoint output-coordinate
ranges, with scalar fallback when that proof cannot be retained.

This task adds one CPU-private fold [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/emitter family. It does not extend Model,
[Compiler](../../../../glossary.md#term-compiler), shared semantics, [Runtime](../../../../glossary.md#term-runtime) interpretation, native routes, [fusion](../../../../glossary.md#term-fusion), or the later ordering,
reduction, [scan](../../../../glossary.md#term-cumulative-scan), and dynamic-[binding](../../../../glossary.md#term-binding) frontiers.

## [Scope](../../../../glossary.md#term-scope)

- Admit exactly one CPU-owned node with one logical input, one output, and one of these current
  identities:<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [node glossary](../../../../glossary.md#node))</span>
  - `WindowTransformKind.FOLD_AXIS + FoldAxisAttrs(axis, outputSize, step)`; or
  - `WindowTransformKind.FOLD2D + Fold2dAttrs(outputShape, window)`.
- Require fully static Shapes, resolved non-negative-offset/non-negative-stride layouts, exact
  current Model signature and Shape rules, one distinct writable injective output, and exact
  input/output DataType agreement.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [stride glossary](../../../../glossary.md#term-stride); [layouts glossary](../../../../glossary.md#layout); [Model glossary](../../../../glossary.md#term-model); [signature glossary](../../../../glossary.md#term-signature); [injective glossary](../../../../glossary.md#term-injective); [DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- Support `FOLD_AXIS` for [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT32](../../../../glossary.md#data-type--datatype), and [INT64](../../../../glossary.md#data-type--datatype). Support `FOLD2D` for
  FLOAT64, FLOAT32, and BFLOAT16. BOOL, implicit conversion, promotion, and saturation remain
  unsupported.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [saturation glossary](../../../../glossary.md#term-saturation))</span>
- Add distinct `CpuFoldIr`, `CpuFoldLowering`, and `CpuFoldEmitter` owners. Fold is neither
  represented-bit movement nor functional scatter: it has no base Tensor or indices, initializes
  every result from zero, performs overlap addition, and discards out-of-domain 2D positions.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Preserve the existing one-input then output boundary order. No input deduplication map,
  materialization, workspace, hidden temporary result, atomic accumulator, or Runtime slot beyond
  those exact boundaries is required.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [accumulator glossary](../../../../glossary.md#accumulator-type); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Generate an output-domain scalar body. Each disjoint result range initializes its coordinates,
  visits contributing logical input positions in canonical row-major input order, performs the
  selected represented-type addition, and writes each result coordinate exactly once.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Permit parallel-scalar orchestration only when every chunk owns a disjoint half-open flattened
  output interval and independently preserves the same per-output contribution order as scalar
  execution. Select scalar execution if analysis cannot prove those facts. Do not split the input
  contribution domain, merge partial sums, use atomics, or make scheduling observable.<span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval))</span>
- Honor arbitrary supported resolved input and output layouts through compact cold geometry,
  including non-dense positive strides, input zero strides, rank-minimal and singleton domains,
  zero extents, offsets, heap arrays, native-order <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton); [heap glossary](../../../../glossary.md#term-heap))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment), and compatible mixed carrier
  patterns. Repeated logical positions in a zero-stride input remain distinct contributions.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Reject every physical input/output overlap before any output write or worker submission. Input
  is read-only; output is distinct, writable, and injective. Successful repeated and concurrent
  invocations use only invocation-owned carriers and existing worker orchestration.
- Keep all Shape, attribute, layout, carrier, range, alias, and checked-geometry decisions cold.
  Generated execution receives compact primitive geometry and direct carriers and performs no
  Model operation dispatch, reflection, map lookup, synchronization, boxing, or per-element
  allocation.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch); [lookup glossary](../../../../glossary.md#term-lookup); [synchronization glossary](../../../../glossary.md#term-synchronization); [element glossary](../../../../glossary.md#term-element))</span>
- Extend capability, lowering, preparation, assignment validation, finalization, generated entry,
  binding, independent reference, cache compatibility, and package contracts only as required by
  this exact family.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation); [binding glossary](../../../../glossary.md#term-binding); [cache glossary](../../../../glossary.md#term-cache))</span>
- Advance generated compatibility from schema 16 to schema 17. Older entries are incompatible
  misses; there is no migration reader.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Extend the independent scalar reference with a separately implemented fold oracle. Expected
  results must not call <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>`CpuFoldEmitter`, generated helpers, or share the generated coordinate
  traversal implementation.
- After executable Java and focused tests stabilize, hand the uncommitted diff to a distinct clean
  documentation-focused context to finalize affected Javadocs, package summaries, the CPU guide,
  glossary, task evidence, CPU master plan, and roadmap.

## Current [Model](../../../../glossary.md#term-model) contract and [CPU](../../../../glossary.md#term-cpu) realization

### Shared fold contract

- Both operations consume exactly one Tensor and produce one fresh Tensor of the same represented
  type. They do not mutate, alias, or use an implicit base Tensor.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Every result [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) begins at represented [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero): integer zero, [FLOAT64](../../../../glossary.md#term-float64) `+0.0`,
  FLOAT32 <span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32))</span>`+0.0f`, or BFLOAT16 positive-zero bits. A coordinate with no contributions retains that
  exact zero.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Every logical input scalar is considered once. Multiple positions that map to the same result
  are distinct contributions, even when their physical input address is equal because of a
  zero-stride layout.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [stride glossary](../../../../glossary.md#term-stride); [layout glossary](../../../../glossary.md#layout))</span>
- This CPU route fixes canonical flattened row-major input order as its deterministic addition
  order. It is a CPU realization within Model's overlap-sum meaning, not a new backend-neutral
  promise. Scalar and parallel-scalar execution must be bitwise identical for the same values and
  layouts.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [Model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend))</span>
- INT32 and INT64 addition is fixed-width two's-complement modular addition. FLOAT32 and FLOAT64
  use sequential same-format IEEE addition in canonical contribution order. BFLOAT16 expands the
  represented current accumulator and next operand to binary32, adds them, and rounds back to
  BFLOAT16 after every logical contribution. Floating NaN payload, source, signaling state, and
  sign are not promised beyond the represented result produced by this fixed route.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [two's-complement glossary](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks); [IEEE glossary](../../../../glossary.md#term-ieee); [accumulator glossary](../../../../glossary.md#accumulator-type); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- There is no widening accumulator, pairwise/tree summation, compensation, tolerance result,
  averaging, saturation, or once-rounded abstract sum.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [tolerance glossary](../../../../glossary.md#term-tolerance); [saturation glossary](../../../../glossary.md#term-saturation))</span>

### `FOLD_AXIS`

Let input [rank](../../../../glossary.md#term-rank) be `R + 1`, [target](../../../../glossary.md#term-target) rank be `R`, normalized target [axis](../../../../glossary.md#term-axis) be `a`, final input [extent](../../../../glossary.md#term-extent) be
window size `K`, selected input extent be window count `W`, step be `S`, and restored target extent
be `L`. Current [Model](../../../../glossary.md#term-model) construction requires numeric input, `R >= 1`, `K > 0`, `S > 0`, and:

```text
L == 0  => W == 0
L > 0   => K <= L and W == ((L - K) / S) + 1
```

For logical input [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `x[0:R+1]`, with `w = x[a]` and `k = x[R]`, the result coordinate is
the first `R` input coordinates with:

```text
result[a] = w * S + k
```

All arithmetic and range checks are exact. The final window-size [dimension](../../../../glossary.md#dimension) is removed; unaffected
dimensions retain their exact extents. Overlapping windows add. Positions not covered because of
step gaps or the trailing floor-window remainder remain represented [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero). A zero output
has a zero selected window count and causes no generated invocation or worker submission.

### `FOLD2D`

Input is [canonical](../../../../glossary.md#term-canonical-form) [rank](../../../../glossary.md#term-rank)-three columns `[N, Q, P]`; output is exact rank-four [NCHW](../../../../glossary.md#term-nchw) `[N, C, H, W]`.
For [kernel](../../../../glossary.md#kernel) `(KH, KW)`, [stride](../../../../glossary.md#term-stride) `(SH, SW)`, symmetric [padding](../../../../glossary.md#padding) `(PH, PW)`, [dilation](../../../../glossary.md#term-dilation) `(DH, DW)`, and
the current floor/ceil-mode formulas:

```text
Q = C * KH * KW
P = OH * OW
q = ((c * KH) + kh) * KW + kw
p = oh * OW + ow
ih = oh * SH - PH + kh * DH
iw = ow * SW - PW + kw * DW
```

The logical column value `[n, q, p]` contributes to `[n, c, ih, iw]` exactly when:

```text
0 <= ih < H and 0 <= iw < W
```

Every other column value is excluded, including leading/trailing symmetric-padding positions and
ceil-mode terminal positions whose window starts or dilated [elements](../../../../glossary.md#term-element) lie outside the unpadded
output. Exclusion is independent of the input value: there is no padding [scalar](../../../../glossary.md#term-scalar) to compare, add,
or preserve. Valid positions that overlap add; uncovered output positions remain represented
[positive zero](../../../../glossary.md#nan-infinity-and-signed-zero). Checked long arithmetic derives effective [kernels](../../../../glossary.md#kernel), padded fit, `OH`, `OW`, `Q`,
`P`, element counts, and physical spans before finalization.

### Generated range and [lifecycle](../../../../glossary.md#lifecycle) contract

- The flattened iteration domain is the output [element](../../../../glossary.md#term-element) count. A range `[start, end)` owns exactly
  those logical output ordinals and writes no other coordinate.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- For each owned output ordinal, the generated body derives the output coordinate, visits exactly
  the contributing input ordinals in increasing canonical flattened input order, accumulates, and
  performs one final output write. It may derive only matching contributors instead of scanning
  the entire input when equivalence of order and membership is proved.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Parallel ranges share only the read-only input and immutable cold geometry. They have disjoint
  output coordinates and no cross-range state. Empty output performs no generated call and no
  worker work; non-empty output with no contributors writes exact represented zeros.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Preparation declares exactly two buffers, zero workspaces, one computation unit, and one
  generated artifact. Finalization validates exact assigned slots, buffer roles, types, spans,
  carrier pattern, zero workspace, selected strategy, structural IR, and schema before loading or
  generating bytes.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [artifact glossary](../../../../glossary.md#term-artifact); [carrier glossary](../../../../glossary.md#term-storage-carrier); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [schema glossary](../../../../glossary.md#term-schema))</span>
- Binding revalidates carrier liveness/thread access, byte sizes, writability, represented address
  spans, output injectivity facts retained by lowering, and physical non-overlap before execution.
  The hot invocation receives direct carrier arguments plus packed primitive layout/geometry and
  range bounds; it does not receive <span data-glossary-navigation="true"> ([Binding glossary](../../../../glossary.md#term-binding); [liveness glossary](../../../../glossary.md#term-liveness); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [lowering glossary](../../../../glossary.md#lowering); [arguments glossary](../../../../glossary.md#term-argument); [layout glossary](../../../../glossary.md#layout))</span>[`Operation`](../../../../glossary.md#operation), `CompiledNode`, `GraphValue`, or [descriptor](../../../../glossary.md#term-descriptor) state.
- Cache identity includes schema 17, fold family, represented type, boundary ranks and structural
  access plans, carrier pattern, execution mode, and an explicit addition-policy signature.
  Concrete extents, offsets, positive stride magnitudes, axis, step, window geometry, ranges,
  worker count, and carrier instances remain cold when emitted bytes do not depend on them.
  FOLD_AXIS and FOLD2D, different types, access regimes, carriers, modes, or addition-policy
  signatures never collide.<span data-glossary-navigation="true"> ([Cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [signature glossary](../../../../glossary.md#term-signature); [stride glossary](../../../../glossary.md#term-stride); [axis glossary](../../../../glossary.md#term-axis))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- `SCATTER_ELEMENTS`, `SCATTER_ADD`, `SCATTER_ND`, any new scatter form, index validation,
  replacement, base participation, configurable reduction, or exact-product scratch<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `UNFOLD_AXIS`, either `UNFOLD2D` signature, padding-value materialization, or changes to the
  existing movement IR/lowering/emitter<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [padding glossary](../../../../glossary.md#padding); [materialization glossary](../../../../glossary.md#materialization); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering))</span>
- SORT, ARGSORT, TOP_K, RNG, DROPOUT, aggregate/reduction families, arg reductions, scans,
  softmax/log-softmax, statistics, normalization, convolution, pooling, attention, or losses<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [DROPOUT glossary](../../../../glossary.md#dropout--inverted-dropout); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [losses glossary](../../../../glossary.md#loss))</span>
- changing contribution order, adding reduction reassociation choices, relaxed numerical modes,
  tolerance-based fold results, vector fold, input-domain parallel partials, atomics, or merges<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [tolerance glossary](../../../../glossary.md#term-tolerance); [vector glossary](../../../../glossary.md#term-vector))</span>
- multiple fold nodes, mixed fold/pointwise partitions, general partition-DAG decomposition,
  fusion, native routes, vendor libraries, tuning, persistence-policy changes, or route priority<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [pointwise glossary](../../../../glossary.md#term-pointwise); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- dynamic Shape binding, unresolved layouts, negative offsets/strides, non-injective outputs,
  output/input alias execution, device carriers, transfer, publication, or Engine integration<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding); [layouts glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective); [alias glossary](../../../../glossary.md#aliasing-and-arena); [device glossary](../../../../glossary.md#term-device); [publication glossary](../../../../glossary.md#term-publication); [Engine glossary](../../../../glossary.md#term-engine))</span>
- new Model kinds/attrs/Tensor APIs, compiler gradients or rewrites, shared Prepare/Runtime
  contracts, public configuration, tracing, training, architecture rules, dependencies, or builds<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- copying legacy implementation structure or code; `legacy/pre-rewrite` remains read-only behavior
  evidence only
- architecture/ADR/architecture-test, backend-conformance, integration-test, API-guide, or later-
  task changes unless a discovered contract conflict first causes this task to stop and replan<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0006A1 static window extraction](0006a1-portable-static-window-extraction.md)
- [CPU 0006B functional slice update](0006b-portable-functional-slice-update.md)<span data-glossary-navigation="true"> ([slice update glossary](../../../../glossary.md#slice-update))</span>
- [CPU 0006B1 functional scatter](0006b1-portable-functional-scatter.md)
- [Model 0017M unfold/fold semantics](../../../modules/model/tasks/0017m-unfold-and-fold-semantics.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model 0017N unfold/fold expressions](../../../modules/model/tasks/0017n-unfold-and-fold-tensor-expressions.md)
- [Model 0023D public fold axis and dynamic windows](../../../modules/model/tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md)<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- [Compiler 0005C window gradient completion](../../../modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Model remains the sole owner of fold identity, attributes, Shape rules, and backend-neutral
  overlap-sum meaning. CPU revalidates eligibility and owns only its private realization.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [identity glossary](../../../../glossary.md#term-identity-meanings); [Shape glossary](../../../../glossary.md#shape); [backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- CPU analysis performs Shape/layout/type/alias/range/resource decisions before shared assignment.
  Prepare assigns opaque declared resources; CPU finalization validates those assignments and
  realizes one generated artifact afterward.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [alias glossary](../../../../glossary.md#aliasing-and-arena); [Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- [Runtime](../../../../glossary.md#term-runtime) receives a prepared direct invocation and no [graph](../../../../glossary.md#graph) or semantic object.
- The implementation remains in `backends/cpu` and adds no dependency edge. Existing package
  direction and the generated JVM-bytecode portable route remain unchanged.<span data-glossary-navigation="true"> ([dependency edge glossary](../../../../glossary.md#term-dependency-edge); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Capability advertisement must be truthful and no broader than lowering, finalization, binding,
  carriers, and generated execution can complete. Capability and lowering independently reject
  every unsupported signature or geometry.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding); [signature glossary](../../../../glossary.md#term-signature))</span>
- Current source is authoritative over stale planning or legacy names. Stop if current Model
  source contradicts the mappings or type matrix in this task.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Stop before editing if exact behavior requires an architecture change, another module, a shared
  contract, a build/dependency change, an unlisted path, a 42nd changed path, hidden workspace,
  input-domain merge, or a numerical decision not resolved here.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

No package is added, removed, or moved.

Existing packages used or changed:

- `io.github.pho001.synaptik.backend.cpu` — truthful public [backend capability](../../../../glossary.md#backend-capability) only.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — structural portable fold [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — Model-to-CPU eligibility and compact
  geometry.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — generated fold body.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — declaration and post-assignment
  validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold carrier/alias binding and
  direct ranged invocation.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [alias glossary](../../../../glossary.md#aliasing-and-arena); [binding glossary](../../../../glossary.md#term-binding))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema)-17 structural compatibility.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — existing one-[artifact](../../../../glossary.md#term-artifact) route.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent [test oracle](../../../../glossary.md#term-test-oracle).

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr` — immutable family/type/access/
  addition-policy identity because these facts shape generated bytes and cache compatibility.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [cache glossary](../../../../glossary.md#term-cache))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLowering` — one-node Model
  revalidation, exact axis/NCHW geometry, layout normalization, range safety, and zero-resource
  analysis because CPU lowering owns route-private eligibility and cold facts.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [axis glossary](../../../../glossary.md#term-axis); [NCHW glossary](../../../../glossary.md#term-nchw); [normalization glossary](../../../../glossary.md#term-normalization); [lowering glossary](../../../../glossary.md#lowering))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuFoldEmitter` — Class-File API
  scalar fold emission and represented-type addition because generated mechanics belong to the
  selected portable route leaf.<span data-glossary-navigation="true"> ([Class-File API glossary](../../../../glossary.md#term-class-file-api); [scalar glossary](../../../../glossary.md#term-scalar); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- Nested [immutable](../../../../glossary.md#term-immutable) `CpuFoldLowering.Geometry`, `AxisGeometry`, `TwoDimensionalGeometry`, and
  [`Layout`](../../../../glossary.md#layout) values remain with the lowerer; they are CPU-private cold facts, not public Model or
  shared Prepare types.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>

## Affected files

Authorized [CPU](../../../../glossary.md#term-cpu) production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFoldEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuFoldIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuFoldLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
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
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFoldGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuNonAffineMovementLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuFoldIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuFoldLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Authorized explanatory and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0006b2-portable-overlap-fold.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized. Listed paths change only when their implementation contract,
[Javadoc](../../../../glossary.md#term-javadoc), package summary, [regression](../../../../glossary.md#term-regression) [ownership](../../../../glossary.md#term-ownership-meanings), or synchronized status is affected. Both clean
handoffs must record review-only no-change conclusions for unused authorized paths.

Review without modification:

- Model, Compiler, Tensor/API documentation, architecture, ADR, architecture-test, backend-
  conformance, integration, shared-module, Gradle/dependency, vendor, and legacy paths remain
  evidence only. Current Tensor and Compile API guides already explain the Model fold expressions
  and do not promise CPU execution.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [Compile glossary](../../../../glossary.md#compile))</span>
- Existing movement, indexing, scatter, workspace, and worker source/tests remain unchanged unless
  an explicitly listed generic owner requires a fold-aware branch. Do not merge fold semantics
  into <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>`CpuDataMovementIr`, `CpuNonAffineMovementLowering`, `CpuDataMovementEmitter`,
  `CpuScatterIr`, `CpuScatterLowering`, or `CpuScatterEmitter`.

## Maximum [scope](../../../../glossary.md#term-scope)

| Category | Maximum | Accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production/package | 24 | 3 new [owners](../../../../glossary.md#term-owner) plus affected capability, generator, [lifecycle](../../../../glossary.md#lifecycle), [cache](../../../../glossary.md#term-cache), route, reference, and package contracts |
| CPU tests | 12 | Focused new family plus existing lifecycle/cache/reference [regression](../../../../glossary.md#term-regression) owners |
| Explanatory documentation | 2 | CPU guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **41** | **24 + 12 + 2 + 3** |

Do not spend unused scope on cleanup. Stop and request a revised planning decision before a 42nd
path, any path outside the exact allowlist, any need for workspace/[materialization](../../../../glossary.md#materialization), or any need to
change an existing family-specific [owner](../../../../glossary.md#term-owner) excluded above.

## Acceptance criteria

- Capability and [lowering](../../../../glossary.md#lowering) admit exactly one fully static resolved-[layout](../../../../glossary.md#layout) current `FOLD_AXIS` or
  `FOLD2D` occurrence with the exact kind/attrs/signature/type/Shape matrix and independently
  reject every wrong kind, attrs, arity, type, Shape, staticity, layout, or output-injectivity row.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [signature glossary](../../../../glossary.md#term-signature); [Shape glossary](../../../../glossary.md#shape); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [arity glossary](../../../../glossary.md#term-arity); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- `FOLD_AXIS` implements the exact target-axis normalization, count/window/step compatibility,
  final-dimension removal, mapping, overlap addition, gaps/trailing uncovered zeros, rank-minimal
  and singleton domains, and zero-output behavior defined above.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [dimension glossary](../../../../glossary.md#dimension); [rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- `FOLD2D` implements exact [canonical](../../../../glossary.md#term-canonical-form) `[N,C*KH*KW,OH*OW]` to NCHW mapping with checked floor/ceil,
  stride, symmetric padding, and dilation geometry. Every and only out-of-unpadded-output column
  position is excluded, including ceil-tail cases.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation))</span>
- Every output starts from represented positive zero and every contributing logical input
  occurrence participates exactly once in canonical row-major input order. Input zero strides do
  not collapse logical occurrences.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- INT32/INT64 modular overflow, FLOAT64/FLOAT32 sequential same-format rounding, and BFLOAT16
  expand/add/round-after-each-contribution semantics match an independently implemented oracle.
  Tests cover finite normals, subnormals, overflow, underflow, cancellation, signed zeros,
  infinities, NaNs, rounding-sensitive order, and BFLOAT16 ties or boundary rounding.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [overflow glossary](../../../../glossary.md#term-overflow); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [rounding glossary](../../../../glossary.md#term-rounding); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [oracle glossary](../../../../glossary.md#term-test-oracle); [finite glossary](../../../../glossary.md#term-finite); [subnormals glossary](../../../../glossary.md#term-subnormal); [underflow glossary](../../../../glossary.md#term-underflow); [cancellation glossary](../../../../glossary.md#term-cancellation); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- FLOAT64/FLOAT32/BFLOAT16 are covered for both families; INT32/INT64 are additionally covered for
  FOLD_AXIS; BOOL and integral FOLD2D fail closed.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Scalar Shapes where Model permits them, zero extents, empty contributions, uncovered output,
  non-dense/read-zero layouts, injective strided output, non-zero offsets, heap/segment/mixed
  carriers, and arbitrary partial output ranges are covered.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [Model glossary](../../../../glossary.md#term-model); [injective glossary](../../../../glossary.md#term-injective); [heap glossary](../../../../glossary.md#term-heap))</span>
- Parallel-scalar ranges own disjoint outputs and are bitwise identical to scalar execution.
  Tests cover multiple chunks, repeated invocation, simultaneous independent invocations, worker
  reuse, zero-output no-submission, and no races, atomics, cross-range merge, or hidden scratch.
- Output/input overlap is rejected before writes or worker submission; input remains byte-for-byte
  unchanged. Disjoint views in one carrier are accepted when exact represented spans prove them
  disjoint.<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Generated ranges write only their interval and write each coordinate once. Packed geometry is
  compact and contains no per-input, per-output, or contribution table. Generated hot loops use
  no allocation, boxing, reflection, maps, synchronization, Model dispatch, or arbitrary-
  precision fallback.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [synchronization glossary](../../../../glossary.md#term-synchronization); [dispatch glossary](../../../../glossary.md#term-dispatch); [precision glossary](../../../../glossary.md#term-precision))</span>
- Analysis declares exactly two buffers, zero workspaces/materializations, one unit, one artifact,
  and a valid scalar or parallel-scalar strategy. Any geometry whose safe split proof cannot be
  retained selects scalar execution rather than unsafe parallelism.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Finalization rejects missing/extra/reordered/wrong-role/wrong-type/wrong-span assignments,
  non-zero workspace assignments, mismatched fold family/geometry/access/carriers/ranges, stale
  schema, and malformed structural IR before artifact invocation.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Structural cache tests prove compatible concrete extent/offset/stride/range/worker/axis/window
  changes reuse bytes when code shape is unchanged, while family/type/rank/access/carrier/mode/
  addition-policy changes separate artifacts. Schema 16 is an incompatible miss under schema 17.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [extent glossary](../../../../glossary.md#term-extent))</span>
- The generated kernel and independent reference oracle are compared across the semantic matrix;
  neither delegates fold evaluation to the other or shares a coordinate-walk helper.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Existing movement, indexing, scatter, pointwise, cache, preparation, finalization, execution,
  carrier, and worker regression suites remain green. Capability advertisement is updated only
  after end-to-end preparation/finalization/execution coverage exists.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [regression glossary](../../../../glossary.md#term-regression))</span>
- Every changed Java API/implementation contract has meaningful complete Javadoc; affected
  package summaries, CPU guide, glossary, task evidence, master plan, and roadmap are finalized in
  the mandatory distinct clean documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Architecture contracts, Model/Compiler/shared source, dependencies, builds, explanatory API
  guides, architecture/conformance/integration tests, and later task specifications remain
  unchanged. Any need to change them makes this task incomplete pending replanning.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Focused implementation command, [run](../../../../glossary.md#term-run) after production and focused tests stabilize:

```bash
./gradlew :backends:cpu:test --tests '*CpuCapabilityProviderTest' --tests '*CpuFoldIrTest' --tests '*CpuFoldLoweringTest' --tests '*CpuFoldGeneratedKernelTest' --tests '*CpuPartitionPreparerTest' --tests '*CpuPartitionFinalizerTest' --tests '*CpuPreparedExecutableTest' --tests '*CpuGeneratedKernelArtifactStoreTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuShapePolymorphicArtifactTest' --tests '*CpuReferenceDifferentialTest' --tests '*CpuInternalPackageInventoryTest'
```

The focused [matrix](../../../../glossary.md#term-matrix-meanings) must include direct generated-code evidence and independent-[oracle](../../../../glossary.md#term-test-oracle) comparison
for both families, every supported type/[carrier](../../../../glossary.md#term-storage-carrier) form, [axis](../../../../glossary.md#term-axis) positions, [padding](../../../../glossary.md#padding)/[stride](../../../../glossary.md#term-stride)/[dilation](../../../../glossary.md#term-dilation)/
floor/ceil geometry, overlap and uncovered [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), [scalar](../../../../glossary.md#term-scalar)/zero domains, arbitrary [layouts](../../../../glossary.md#layout) and
ranges, [alias](../../../../glossary.md#aliasing-and-arena) rejection, malformed finalization, [cache](../../../../glossary.md#term-cache) compatibility, numeric edges, and repeated/
parallel invocation.

Final affected-[module](../../../../glossary.md#term-module) validation, run once after the focused command passes:

```bash
./gradlew :backends:cpu:test
./gradlew :backends:cpu:javadoc
```

The clean documentation context runs repository-local link and heading-anchor checks, balanced-
fence, final-newline, trailing-whitespace, generated-[Javadoc](../../../../glossary.md#term-javadoc) inspection, exact allowlist/ceiling,
[schema](../../../../glossary.md#term-schema)/status/[dependency](../../../../glossary.md#term-dependency) [synchronization](../../../../glossary.md#term-synchronization), absence of a [CPU](../../../../glossary.md#term-cpu) 0006C detailed specification, and:

```bash
git diff --check
```

This is a normal single-module task. Repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and
integration suites are deferred to CPU 0009 and [CI](../../../../glossary.md#term-continuous-integration-ci) because no dependency/module boundary, shared
contract, or current public [Engine](../../../../glossary.md#term-engine) path changes. If implementation makes that conclusion false,
stop and replan before changing another tier. The documentation context reuses successful Java
evidence unless it changes executable Java or records a concrete stale-evidence reason.

## Dependencies

- [CPU 0006B1](0006b1-portable-functional-scatter.md) is `Complete` and supplies current schema-16
  generated lifecycle, represented-type addition choices, disjoint output ownership, arbitrary-
  layout/carrier/alias handling, malformed-finalization tests, and independent reference seams.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- [CPU 0006A1](0006a1-portable-static-window-extraction.md) is `Complete` and supplies current
  axis/NCHW window geometry, canonical column order, floor/ceil, stride/padding/dilation, carrier,
  compact-layout, and schema compatibility foundations.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [NCHW glossary](../../../../glossary.md#term-nchw); [canonical glossary](../../../../glossary.md#term-canonical-form); [stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation))</span>
- [CPU 0006B](0006b-portable-functional-slice-update.md) and [CPU](../../../../glossary.md#term-cpu) 0006A2 are `Complete` and supply
  current one-node movement/indexing preparation and direct ranged execution patterns.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>
- CPU 0005E through 0005J are `Complete` and supply represented carriers, fixed-width integral
  arithmetic, FLOAT64/FLOAT32/BFLOAT16 conventions, specialization, range strategy, cache, and
  scalar/parallel parity foundations.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [specialization glossary](../../../../glossary.md#term-specialization); [cache glossary](../../../../glossary.md#term-cache); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Model 0017M](../../../modules/model/tasks/0017m-unfold-and-fold-semantics.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>,
  [Model 0017N](../../../modules/model/tasks/0017n-unfold-and-fold-tensor-expressions.md), and
  [Model 0023D](../../../modules/model/tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md)
  are `Complete` and own the current identities, attributes, type/Shape rules, canonical mappings,
  dynamic-expression surface, and overlap-sum meaning. CPU admits only their fully static resolved
  subset.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- [Compiler 0005C](../../../modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
  is `Complete` and supplies current adjoint construction evidence without making compiler changes
  part of this task.<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0006C remains `Draft`, depends on CPU 0006B2, and owns stable SORT/ARGSORT and two-output
  TOP_K. Do not create its detailed task specification here.
- CPU 0006D remains `Draft`, depends on CPU 0006C, and owns explicit-state [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/[dropout](../../../../glossary.md#dropout--inverted-dropout).
- CPU 0007 owns reduction, scan, statistics, and normalization families. Fold does not preempt or
  generalize that accumulation framework.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Later native, tuning, dynamic-layout, general partition-DAG, fusion, conformance-checkpoint, and
  Engine work retain their existing roadmap ownership and status.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [fusion glossary](../../../../glossary.md#term-fusion); [conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Engine glossary](../../../../glossary.md#term-engine); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Architecture impact

Expected impact: None.

The architecture already assigns semantic meaning to [Model](../../../../glossary.md#term-model), [CPU](../../../../glossary.md#term-cpu) route/[lowering](../../../../glossary.md#lowering)/resource choices
to [backend](../../../../glossary.md#backend) analysis, [opaque](../../../../glossary.md#term-opaque) assignment to [Prepare](../../../../glossary.md#prepare), generated [artifact](../../../../glossary.md#term-artifact) realization to CPU
finalization, and direct invocation to [Runtime](../../../../glossary.md#term-runtime). One CPU-private zero-workspace fold family reuses
those boundaries.

If implementation requires an [architecture contract](../../../../glossary.md#architecture-contract), focused architecture document, [ADR](../../../../glossary.md#term-architecture-decision-record-adr),
[architecture test](../../../../glossary.md#term-architecture-test), [dependency edge](../../../../glossary.md#term-dependency-edge), shared/public [lifecycle](../../../../glossary.md#lifecycle) contract, or another [module](../../../../glossary.md#term-module) to change,
stop and report the exact conflict rather than implement it under this task.

## Implementation prompt

Use this prompt for the mandatory isolated implementation [handoff](../../../../glossary.md#term-handoff):

```text
You are the isolated implementation agent for Synaptik CPU task 0006B2. Work on the current
uncommitted diff, preserving the complete CPU 0006B1 predecessor baseline. Do not commit or push,
and do not use any GSD skill or workflow.

Read in full AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0006b2-portable-overlap-fold.md. Read every dependency and current
Model/CPU owner named by the task as needed. Implement that task exactly within its 41-path
allowlist, using current source rather than legacy naming.

Run the specified focused twelve-owner command followed once by the CPU module suite. Stop and
report any architecture conflict, unresolved semantic decision, forbidden-path need, scope-
ceiling breach, shared-contract need, or hidden-resource need instead of inventing or broadening
the design.

After executable Java and tests stabilize, hand the same uncommitted diff and recorded evidence to
a distinct clean documentation-focused context using the task's documentation prompt. Mark the
task Complete only after both clean contexts and every specified gate pass, and return the
AGENTS.md completion summary.
```

## Documentation prompt

Use this prompt only after implementation and tests stabilize:

```text
You are the mandatory distinct clean documentation-focused agent for Synaptik CPU task 0006B2.
Work on the existing uncommitted implementation diff, preserving the inherited completed 0006B1
baseline. Do not commit or push, do not use GSD, and do not change executable Java except to report
a blocking contract defect to the implementation agent.

Read AGENTS.md; ARCHITECTURE.md; docs/developer-guide/documentation-rules.md and the General,
API/Javadoc, Backend Guide, Planning, and Example profiles; task 0006B2; final affected source and
tests; current Model fold contracts; CPU guide; glossary; CPU master plan; and roadmap. Independently
verify and finalize every affected Javadoc and package summary plus the CPU guide, glossary, task
evidence/completion, master plan, and roadmap.

Document exactly current FOLD_AXIS/FOLD2D CPU coverage: zero initialization, canonical overlap
addition, represented type rules, padding/ceil-tail exclusion, layouts/carriers/aliasing/ranges,
scalar/parallel behavior, zero resources, lifecycle, and schema 17. Do not claim broader Model,
Compiler, Runtime, native, vector, dynamic-layout, fusion, reduction-family, performance, or Engine
coverage.

Inspect generated Javadoc and run repository-local links/anchors, fences, whitespace/newlines,
exact changed-path/41-path ceiling, status/dependency, no-0006C-spec, schema, and git diff --check
gates. Reuse recorded successful Java evidence unless this pass changes executable Java or finds a
concrete stale-evidence reason. Record explicit no-change conclusions for Tensor/Compile/Training
API docs, Model/Compiler/shared modules, architecture/ADRs/tests, backend conformance/integration,
Gradle/dependencies, legacy/vendor/native code, existing family owners, and later tasks. Mark the
task Complete only when documentation and every acceptance gate pass, then return the mandatory
AGENTS.md completion summary and context ID if available.
```

## Decisions made

- Use a distinct fold family rather than movement or scatter because fold has zero initialization,
  no base or indices, overlap addition, and 2D exclusion semantics.
- Produce by disjoint output coordinates and preserve canonical logical input row-major
  contribution order independently inside every range. This gives one final write, deterministic
  bit parity, and safe parallelism without atomics or merges.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Use represented positive zero and the current CPU scatter-ADD type policies: modular INT32/
  INT64, sequential same-format FLOAT32/FLOAT64, and BFLOAT16 rounding after every addition.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [CPU glossary](../../../../glossary.md#term-cpu); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Exclude FOLD2D padded and ceil-tail column positions geometrically. Never infer exclusion from a
  value or add a conceptual padding scalar.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Permit logical input repetition through zero strides, but require a distinct injective output
  and reject every physical input/output overlap before work.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective))</span>
- Declare no workspace or materialization. Keep concrete geometry cold and version generated
  identity with schema 17 plus an explicit fold-addition policy signature.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [identity glossary](../../../../glossary.md#term-identity-meanings); [schema glossary](../../../../glossary.md#term-schema); [signature glossary](../../../../glossary.md#term-signature))</span>
- Use scalar fallback whenever disjoint output ownership and identical contribution order cannot
  be proved; never introduce input-range partial reductions as an implicit fallback.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Use a separately implemented scalar reference oracle for differential evidence, never as a
  Runtime or generated-code fallback.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Replace the unused authorized `CpuShapePolymorphicArtifactTest` path with
  `CpuNonAffineMovementLoweringTest`. The latter owns a stale whole-partition dispatch assertion
  that rejects the exact canonical FOLD2D row this task now supports. Correcting that assertion is
  in-scope regression maintenance, not movement-family implementation, architecture change, or
  scope expansion; the focused command may still run the former test read-only.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [dispatch glossary](../../../../glossary.md#term-dispatch); [scope glossary](../../../../glossary.md#term-scope); [regression glossary](../../../../glossary.md#term-regression); [run glossary](../../../../glossary.md#term-run))</span>

## Known limitations

- Only one fully static resolved-layout fold occurrence is executable. Public Tensor expressions
  may retain dynamic Shapes or unresolved layouts until later lifecycle work supplies concrete
  descriptors.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [occurrence glossary](../../../../glossary.md#term-occurrence); [Tensor glossary](../../../../glossary.md#tensor); [Shapes glossary](../../../../glossary.md#shape); [lifecycle glossary](../../../../glossary.md#lifecycle); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- The output must be distinct, writable, [injective](../../../../glossary.md#term-injective), and physically non-overlapping with input.
- The output-domain algorithm favors semantic closure and deterministic independent ranges. It
  makes no asymptotic or throughput claim and may revisit contribution geometry for each output.<span data-glossary-navigation="true"> ([throughput glossary](../../../../glossary.md#term-throughput))</span>
- Floating results are deterministic for this CPU route but are not a new cross-backend bitwise
  Model promise; NaN payload/source/sign and signaling state are not preserved promises.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [Model glossary](../../../../glossary.md#term-model); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Vector fold, native routes, tuning, dynamic binding, multi-node fusion, input-domain partials,
  and reduction-framework reuse remain outside this task.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [binding glossary](../../../../glossary.md#term-binding); [node glossary](../../../../glossary.md#node); [fusion glossary](../../../../glossary.md#term-fusion))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `019ff539-81ff-7013-a8a6-d2bc5138f522` read the governing AGENTS, architecture,
current-architecture, planning-guide, roadmap, and [CPU](../../../../glossary.md#term-cpu) master-plan contracts in full; completed CPU
0006A1/0006B/0006B1 and [Model](../../../../glossary.md#term-model) 0017M/0017N/0023D specifications; current Model fold kinds,
attributes, [Tensor](../../../../glossary.md#tensor) [Shape](../../../../glossary.md#shape) construction, and [compiler](../../../../glossary.md#term-compiler) [gradient](../../../../glossary.md#term-gradient) use; and current CPU capability, [IR](../../../../glossary.md#term-intermediate-representation-ir),
[lowering](../../../../glossary.md#lowering), preparation, finalization, generated-code, execution, range/worker, [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout),
reference, [cache](../../../../glossary.md#term-cache), tests, guide, [API](../../../../glossary.md#term-application-programming-interface-api), glossary, and inherited uncommitted diff [owners](../../../../glossary.md#term-owner). Repository
search found only the root `AGENTS.md` for affected paths. Current source resolved the mappings,
type [matrix](../../../../glossary.md#term-matrix-meanings), represented addition, [padding](../../../../glossary.md#padding) exclusion, [lifecycle](../../../../glossary.md#lifecycle), and safe output-range design; no
architecture conflict or unresolved semantic question remained. Legacy source was not used or
modified.

Before editing, the context saved the inherited dirty baseline status plus byte-for-byte snapshots
of the CPU master plan and roadmap under `/tmp/synaptik-0006b2-planning.zOmnIA`. Diffing those
snapshots after planning isolates exactly this new task plus the frontier/status edits in the CPU
master plan and roadmap. The inherited CPU 0006B1 production, test, guide, glossary, and planning
changes remain otherwise untouched.

Planning-stage validation passed:

- a repository-local Ruby Markdown check resolved every local file target and heading anchor in
  the three planning-delta files;<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- shell/Ruby checks confirmed every required heading, balanced code fences, final newlines, and no
  trailing whitespace;
- status checks confirmed exactly one [`Ready`](../../../../glossary.md#term-ready) CPU master-plan row, linked 0006B2 status in both
  planning indexes, preserved 0006B1 `Complete`, preserved 0006C `Draft` depending on 0006B2, and
  confirmed no `0006C` detailed task file;
- saved-baseline `diff -u`, `git status --short`, and the baseline/current status comparison
  confirmed the planning-context delta is exactly this new task, `../master-plan.md`, and
  `../../../roadmap.md`; no inherited predecessor path was reverted or rewritten by this context;
- `git diff --check` passed.

No Java, test, [Javadoc](../../../../glossary.md#term-javadoc), build, architecture, explanatory-guide, API, glossary, or legacy command
was [run](../../../../glossary.md#term-run) or changed because this is a planning-only Ready transition. Implementation and
documentation contexts must append their commands, results, test counts, generated-Javadoc
review, semantic evidence, reused evidence, and no-change conclusions before changing status to
`Complete`.

The same planning context later reviewed the implementation blocker in full and confirmed that
`CpuNonAffineMovementLoweringTest.rejectsExcludedWindowSignaturesTypesLayoutsAndOverflow`
retains one obsolete assertion requiring a [canonical](../../../../glossary.md#term-canonical-form) fully static resolved-layout FOLD2D row to
fail through `CpuPartitionLowering`. That assertion is now incompatible with this task's required
and implemented [dispatch](../../../../glossary.md#term-dispatch). The authorized test list therefore replaces the unchanged
`CpuShapePolymorphicArtifactTest` [owner](../../../../glossary.md#term-owner) with `CpuNonAffineMovementLoweringTest`; the focused
command remains unchanged and may continue executing the former owner read-only. The authorized
test count remains 12 and the total ceiling remains 41. This is a correction of an in-[scope](../../../../glossary.md#term-scope) stale
[regression](../../../../glossary.md#term-regression) owner, with no new semantic behavior, architecture impact, production owner, test
category, or scope expansion.

Planning-correction validation confirmed valid local Markdown paths, balanced fences, final
newline, no trailing whitespace, task `Ready` status synchronized with the unchanged planning
indexes, 24 production/package paths plus 12 test paths plus five explanatory/planning paths, the
unchanged focused command retaining read-only `CpuShapePolymorphicArtifactTest`, removal of stale
blocked-status wording, and `git diff --check`. No Java test or Javadoc command was rerun, and no
claim is made that the newly authorized stale assertion has already been corrected or validated.

## Implementation notes

- Implementation context `019ff549-d477-7140-921d-8404d10a2c7e` saved the inherited dirty-path
  inventory and planning-file snapshots under<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
  `/tmp/synaptik-cpu-0006b2-implementation.3YWeYs` before editing.
- The context added the distinct [CPU](../../../../glossary.md#term-cpu)-private `CpuFoldIr`, `CpuFoldLowering`, and `CpuFoldEmitter`
  family and integrated current fully-static resolved-layout <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>`FOLD_AXIS`/`FOLD2D` capability,
  lowering, generated scalar mechanics, preparation/finalization, disjoint output-range execution,
  reference evaluation, cache schema 17, and focused coverage without workspace,
  materialization, atomics, or input-domain partials.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [scalar glossary](../../../../glossary.md#term-scalar); [evaluation glossary](../../../../glossary.md#term-evaluation); [cache glossary](../../../../glossary.md#term-cache); [schema glossary](../../../../glossary.md#term-schema); [materialization glossary](../../../../glossary.md#materialization))</span>
- The specified focused twelve-owner command passed 98 tests in 12 suites with zero skips,
  failures, or errors after the planning correction. The final CPU module suite then passed 245
  tests in 41 suites with one skip and zero failures or errors.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [module glossary](../../../../glossary.md#term-module))</span>
- The corrected non-affine movement [regression](../../../../glossary.md#term-regression) now proves that [canonical](../../../../glossary.md#term-canonical-form) `FOLD2D` dispatches to
  `CpuFoldIr` through `CpuPartitionLowering` while a direct
  `CpuNonAffineMovementLowering.lower(...)` call still rejects it. No movement implementation was
  changed, and every existing movement exclusion in the owner remains covered.
- Final implementation review found no obvious contract defect in fold IR, cold geometry,
  generated mechanics, represented addition, preparation/finalization, range ownership,
  reference evaluation, cache identity, or capability reporting. The Java/Gradle identity remains
  OpenJDK 26.0.1+8-34 (64-bit Server VM) and Gradle 9.6.1 on macOS 26.5.2 aarch64.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [Gradle glossary](../../../../glossary.md#term-gradle); [aarch64 glossary](../../../../glossary.md#term-aarch64))</span>

## Documentation review evidence

Clean documentation context `019ff565-3991-72a1-a47a-78f63ae600ec` independently reviewed all 24
authorized production/package paths, all 12 authorized tests, the final implementation diff, the
current [Model](../../../../glossary.md#term-model) and [Tensor](../../../../glossary.md#tensor) fold contracts, directly relevant [CPU](../../../../glossary.md#term-cpu) [lifecycle](../../../../glossary.md#lifecycle) contracts, and the five
authorized explanatory/[planning](../../../../glossary.md#term-planning) documents. It changed no executable Java behavior. It finalized
four affected Java documentation [owners](../../../../glossary.md#term-owner), the CPU [backend](../../../../glossary.md#backend) guide, glossary, this task, the CPU
master plan, and the roadmap; the remaining affected Javadocs and package summaries were already
accurate and complete for their owning contracts.

The review documents only the current CPU realization: one fully static resolved-[layout](../../../../glossary.md#layout)
`FOLD_AXIS` or `FOLD2D`; fresh represented-positive-zero output; [canonical](../../../../glossary.md#term-canonical-form) logical input
[row-major](../../../../glossary.md#term-row-major) represented addition; [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16) support for both families and
[INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) support for [axis](../../../../glossary.md#term-axis) fold only; BFLOAT16 [rounding](../../../../glossary.md#term-rounding) after every addition; fixed-width
modular integral addition; exact `FOLD2D` [padding](../../../../glossary.md#padding) and ceil-tail exclusion; arbitrary supported
layouts and carriers; a distinct [injective](../../../../glossary.md#term-injective) non-overlapping output; direct [scalar](../../../../glossary.md#term-scalar) execution or
parallel-scalar disjoint output ranges; two boundary [buffers](../../../../glossary.md#term-buffer), one unit, one [artifact](../../../../glossary.md#term-artifact), and no
workspace, [materialization](../../../../glossary.md#materialization), atomics, partials, or merge. Generated compatibility is [schema](../../../../glossary.md#term-schema) 17.
No wider Model, [Compiler](../../../../glossary.md#term-compiler), [Runtime](../../../../glossary.md#term-runtime), native, [vector](../../../../glossary.md#term-vector), [fusion](../../../../glossary.md#term-fusion), dynamic-layout, reduction-framework,
performance, [Engine](../../../../glossary.md#term-engine), [gradient](../../../../glossary.md#term-gradient), or cross-backend bitwise support is claimed.

The exact implementation Java evidence was reused because the documentation context changed no
executable behavior and found no stale-evidence reason: focused 12 suites / 98 tests / 0 skipped /
0 failures / 0 errors; final CPU 41 suites / 245 tests / 1 skipped / 0 failures / 0 errors;
OpenJDK 26.0.1+8-34; [Gradle](../../../../glossary.md#term-gradle) 9.6.1. After documentation stabilized,
`./gradlew :backends:cpu:javadoc` passed. The generated pages for the affected fold [IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering),
emitter, capability [owner](../../../../glossary.md#term-owner), and package summaries were inspected; the final build emitted only the
two expected incubating `jdk.incubator.vector` warnings.

Repository-local Markdown file/heading-anchor checks, balanced-fence, final-newline, trailing-
whitespace, schema/status/[dependency](../../../../glossary.md#term-dependency) [synchronization](../../../../glossary.md#term-synchronization), exact 24-production/package + 12-test +
five-document allowlist/category-ceiling, inherited-dirty preservation, no-0006C-detailed-spec,
and `git diff --check` gates passed. CPU 0006B1 remains `Complete`; CPU 0006B2 is `Complete` in
this task, the CPU master plan, and roadmap; CPU 0006C remains `Draft`, depends on CPU 0006B2, and
has no detailed specification.

No Tensor, [Compile](../../../../glossary.md#compile), or Training [API](../../../../glossary.md#term-application-programming-interface-api) guide change is needed because those guides remain backend-
neutral and already describe the public fold construction or applicable lifecycle boundary
without claiming CPU execution. No Model, Compiler, shared [module](../../../../glossary.md#term-module), [architecture contract](../../../../glossary.md#architecture-contract),
focused architecture document, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [architecture test](../../../../glossary.md#term-architecture-test), backend-[conformance](../../../../glossary.md#term-conformance) test, [integration
test](../../../../glossary.md#term-integration-test), Gradle/dependency, legacy/vendor/[native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture), or later task-specification change is needed:
CPU 0006B2 is a portable CPU-private realization within existing boundaries and the later
[checkpoints](../../../../glossary.md#term-checkpoint) retain their owners.

## Completion summary

- Completed changes: Delivered and documented the exact CPU-private FOLD_AXIS/FOLD2D family,
  represented deterministic overlap addition, padding exclusion, disjoint output ownership,
  reference evidence, lifecycle integration, and schema 17.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [padding glossary](../../../../glossary.md#padding); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [schema glossary](../../../../glossary.md#term-schema))</span>
- Files changed or created by the documentation context: changed `CpuCapabilityProvider`,
  `CpuFoldIr`, `CpuFoldLowering`, and `CpuFoldEmitter` Javadocs; the CPU backend guide; glossary;
  this task; CPU master plan; and roadmap. It created no files and changed no executable Java.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Tests and validation: Reused the final implementation evidence exactly as recorded above. CPU
  Javadoc and all documentation, scope, preservation, synchronization, no-0006C-spec, and diff
  gates passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Javadoc and documentation impact: All affected Java/Javadoc and package summaries were reviewed;
  generated Javadoc was inspected; the guide and glossary now state the exact current CPU fold
  coverage and boundaries.
- No-change conclusions: Tensor/Compile/Training guides, Model/Compiler/shared modules,
  architecture/ADRs/tests, conformance/integration, Gradle/dependencies, legacy/vendor/native
  code, existing unaffected family owners, and later task specifications require no change for
  the reasons recorded above.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [modules glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture))</span>
- Unresolved issues: None.
- Follow-up required: None for CPU 0006B2. CPU 0006C remains the next Draft CPU frontier and must
  receive its own detailed specification before implementation.

Status: Complete
