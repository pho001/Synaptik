# Task 0006: Portable Static Affine [Views](../../../../glossary.md#term-view-meanings) and Boundary [Materialization](../../../../glossary.md#materialization)

## Status

Complete

## Goal

Add the first bounded [layout](../../../../glossary.md#layout)/indexing [slice](../../../../glossary.md#slice) after completed [CPU](../../../../glossary.md#term-cpu) 0005J. CPU preparation will
compose a connected straight-line chain of fully static, resolved-layout affine [view](../../../../glossary.md#term-view-meanings) operations
into one source-to-result mapping. Intermediate view values remain [graph](../../../../glossary.md#graph) values but require no
computation, [buffer](../../../../glossary.md#term-buffer) declaration, [Runtime](../../../../glossary.md#term-runtime) slot, or generated store when they have no [publication](../../../../glossary.md#term-publication),
[fan-out](../../../../glossary.md#term-fan-out), or [partition](../../../../glossary.md#partition)-boundary obligation. When the final result must cross the unit boundary,
the portable route generates one exact [scalar](../../../../glossary.md#term-scalar) or parallel-scalar copy into the result's resolved
layout.

The admitted semantic set is deliberately closed:

```text
CONTIGUOUS
RESHAPE | EXPAND
PERMUTE | EXPAND_DIMS | SQUEEZE
SELECT
SLICE with SliceAttrs or CropToShapeAttrs
```

Every admitted [occurrence](../../../../glossary.md#term-occurrence) must have fully static input/output [Shapes](../../../../glossary.md#shape) and resolved input/output
layouts, and every Shape-valued attribute needed for mapping must also be fully static. All six
current [Model](../../../../glossary.md#term-model) data types are copied as [represented bits](../../../../glossary.md#term-represented-bits). [BFLOAT16](../../../../glossary.md#term-bfloat16) support is limited to raw
two-byte movement through the existing `CpuBufferArgument.Shorts` representation and a new
`CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY` form; it does not advertise BFLOAT16
[pointwise](../../../../glossary.md#term-pointwise) arithmetic, conversion, numerical semantics, or [Vector API](../../../../glossary.md#term-vector-api) support. The task adds no
numerical [operation](../../../../glossary.md#operation), index [Tensor](../../../../glossary.md#tensor), duplicate-[target](../../../../glossary.md#term-target) behavior, ordering algorithm, random
algorithm, public [API](../../../../glossary.md#term-application-programming-interface-api), shared physical-layout contract, or general partition-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [decomposition](../../../../glossary.md#term-decomposition).

[Mental model](../../../../glossary.md#term-mental-model):

```text
validated static affine chain
  -> compose exact logical-coordinate mappings during CPU analysis
  -> internal view values: descriptor/IR facts only, no work and no slot
  -> final boundary: one generated represented-bit copy into exact resolved output geometry
  -> one partition executable
```

This is the largest [dependency](../../../../glossary.md#term-dependency)-correct normal slice of the former broad 0006 row. Non-affine
movement/[gather](../../../../glossary.md#gather), functional scatter/fold, ordering, and explicit-state random work remain four
ordered Draft follow-ups in the CPU master plan and have no detailed specifications.

## [Scope](../../../../glossary.md#term-scope)

### Fresh current-[operation](../../../../glossary.md#operation) inventory

The inventory below is derived from the current [Model](../../../../glossary.md#term-model) [`OperationKind`](../../../../glossary.md#operationkind) families and [signatures](../../../../glossary.md#term-signature),
not from the former broad [CPU](../../../../glossary.md#term-cpu) row. A classification describes the implementation role of an
eligible [occurrence](../../../../glossary.md#term-occurrence); a kind can have both a zero-work internal case and a materialized boundary
case.

| Current Model operation | Classification | CPU 0006 disposition |
|---|---|---|
| `ContiguousKind.CONTIGUOUS` | [descriptor](../../../../glossary.md#term-descriptor) request when its input representation already satisfies the result; otherwise executable data movement | Admit only the resolved-[layout](../../../../glossary.md#layout) static one-input/one-output occurrence; the current no-[alias](../../../../glossary.md#aliasing-and-arena) shared resource model means its final boundary is an exact copy. |
| `ShapeTransformKind.RESHAPE` | [metadata](../../../../glossary.md#term-metadata)-only affine [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) reinterpretation internally; boundary copy when a distinct result slot is required | Admit resolved-layout static occurrences and compose ordered-logical-[element](../../../../glossary.md#term-element) mapping. |
| `ShapeTransformKind.EXPAND` | metadata-only zero-[stride](../../../../glossary.md#term-stride) affine [view](../../../../glossary.md#term-view-meanings) internally; boundary [materialization](../../../../glossary.md#materialization) over distinct output addresses | Admit resolved-layout static occurrences; repeated logical positions must not cause parallel duplicate writes. |
| `AxisTransformKind.PERMUTE` | metadata-only stride/[axis](../../../../glossary.md#term-axis) [permutation](../../../../glossary.md#permutation) internally; boundary copy when required | Admit resolved-layout static occurrences. |
| `AxisTransformKind.EXPAND_DIMS` | metadata-only [extent](../../../../glossary.md#term-extent)-one axis insertion internally; boundary copy when required | Admit resolved-layout static occurrences. |
| `AxisTransformKind.SQUEEZE` | metadata-only extent-one axis removal internally; boundary copy when required | Admit resolved-layout static occurrences. |
| `SelectKind.SELECT` | metadata-only fixed-coordinate view internally; executable data movement at a boundary | Admit resolved-layout static occurrences with the already-normalized intrinsic coordinate. |
| `SliceKind.SLICE` with `SliceAttrs` | metadata-only affine view when the current result layout is resolved; otherwise executable extraction requiring a selected physical layout | Admit only fully static resolved-layout occurrences. Negative-step results are currently layout-unresolved and therefore remain [fail-closed](../../../../glossary.md#term-fail-closed). |
| `SliceKind.SLICE` with `CropToShapeAttrs` | metadata-only [target](../../../../glossary.md#term-target)-relative affine view after exact attributes are static; otherwise requires unresolved [binding](../../../../glossary.md#term-binding)/layout support | Admit only when target and prefix [Shapes](../../../../glossary.md#shape) and both descriptors are fully static and resolved. |
| `PadKind.PAD`, `TileKind.TILE`, `TensorCompositionKind.CONCAT`/`STACK` | executable non-affine data movement | Defer to Draft 0006A. |
| `WindowTransformKind.UNFOLD_AXIS`/`UNFOLD2D` | executable non-affine window data movement, including explicit [padding](../../../../glossary.md#padding) for `Unfold2dAttrs` | Defer to Draft 0006A. |
| `AxisGatherKind.GATHER`/`GATHER_ELEMENTS`, `GatherNdKind.GATHER_ND`, `OneHotKind.ONE_HOT` | executable value-dependent indexing/data generation with execution-time index bounds | Defer to Draft 0006A. |
| `SliceKind.SLICE_UPDATE` | functional base copy plus indexed replacement | Defer to Draft 0006B. |
| `AxisScatterKind.SCATTER_ELEMENTS`/`SCATTER_ADD`, `ScatterNdKind.SCATTER_ND` | executable functional scatter; `SCATTER_ELEMENTS`/`SCATTER_ND` admit `ScatterReduction.NONE`, `ScatterReduction.ADD`, `ScatterReduction.MUL`, `ScatterReduction.MAX`, and `ScatterReduction.MIN`, while `SCATTER_ADD` fixes ADD; duplicate semantics follow that exact variant | Defer to Draft 0006B. |
| `WindowTransformKind.FOLD_AXIS`/`FOLD2D` | executable overlap scatter-add | Defer to Draft 0006B rather than pulling reduction work into 0006 or general reductions into CPU 0007. |
| `OrderingKind.SORT`/`ARGSORT` | stable full-ordering algorithm with exact tie, [NaN](../../../../glossary.md#term-nan), and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) rules | Defer to Draft 0006C. |
| `TopKKind.TOP_K` | deterministic selection algorithm with two outputs | Defer to Draft 0006C. |
| `GraphRngKind.INITIAL_STATE` | explicit-state descriptor/value initialization, not hidden [Runtime](../../../../glossary.md#term-runtime) state | Defer to Draft 0006D. |
| `DropoutKind.DROPOUT` | explicit-state random work with three outputs: value, [auxiliary mask](../../../../glossary.md#auxiliary-mask), and next state | Defer to Draft 0006D. |

Unsupported occurrences remain unadvertised and fail before [artifact](../../../../glossary.md#term-artifact) [lookup](../../../../glossary.md#term-lookup). The deferred rows are
not authorization to implement their algorithms in this task.

### Affine [occurrence](../../../../glossary.md#term-occurrence) and chain eligibility

- Extend `CpuCapabilityProvider` only for the exact admitted occurrence-local matrix. Require the
  operation's exact current attributes class and signature, fully static Shapes, resolved layouts,
  same input/output data type, and exact Model Shape/layout relationship. Do not reinterpret or
  repair Model descriptors.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [data type glossary](../../../../glossary.md#data-type--datatype); [Model glossary](../../../../glossary.md#term-model); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Accept every current [`DataType`](../../../../glossary.md#data-type--datatype): FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and BOOL. Copy raw
  represented values. Preserve floating NaN payload bits, infinities, both signed zeros, signed
  integral bit patterns, BFLOAT16 raw bits, and canonical BOOL bytes without conversion or
  arithmetic.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Extend `CpuKernelSpecialization.CarrierAccess` from its current six forms to exactly seven:
  `DOUBLE_ARRAY`, `FLOAT_ARRAY`, `SHORT_ARRAY`, `INT_ARRAY`, `LONG_ARRAY`, `BYTE_ARRAY`, and
  `MEMORY_SEGMENT`. The exact [heap](../../../../glossary.md#term-heap) mapping is [FLOAT64](../../../../glossary.md#term-float64)→`DOUBLE_ARRAY`, [FLOAT32](../../../../glossary.md#term-float32)→`FLOAT_ARRAY`,
  BFLOAT16→`SHORT_ARRAY`, [INT32](../../../../glossary.md#data-type--datatype)→`INT_ARRAY`, [INT64](../../../../glossary.md#data-type--datatype)→`LONG_ARRAY`, and [BOOL](../../../../glossary.md#data-type--datatype)→`BYTE_ARRAY`; every
  data type may instead use `MEMORY_SEGMENT` when the matching observable heap [carrier](../../../../glossary.md#term-storage-carrier) is absent.
  `SHORT_ARRAY` binds only the already-existing `CpuBufferArgument.Shorts` raw-bit carrier.
- Keep the existing pointwise capability and lowering rejection for BFLOAT16 unchanged. Affine
  copy eligibility is a separate operation-family route whose load/store treats each BFLOAT16
  element as one opaque 16-bit payload; it performs no widening, narrowing, rounding, arithmetic,
  comparison, NaN handling, or vector-lane interpretation.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [lowering glossary](../../../../glossary.md#lowering); [element glossary](../../../../glossary.md#term-element); [opaque glossary](../../../../glossary.md#term-opaque); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [widening glossary](../../../../glossary.md#term-widening); [narrowing glossary](../../../../glossary.md#term-narrowing); [rounding glossary](../../../../glossary.md#term-rounding); [NaN glossary](../../../../glossary.md#term-nan); [vector glossary](../../../../glossary.md#term-vector); [lane glossary](../../../../glossary.md#term-lane))</span>
- Lower one through eight nodes only when they form one connected straight-line affine chain with
  exactly one external data input and one final output. Every non-final output must feed the next
  node exactly once, have no other consumer, not be a graph output, and remain inside the same
  planned partition.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [graph glossary](../../../../glossary.md#graph); [partition glossary](../../../../glossary.md#partition))</span>
- Permit [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) anywhere in the chain as an explicit materialization barrier in Model
  semantics, while still composing the logical mapping. It does not authorize alias reuse or
  elimination across a boundary.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Reject mixed pointwise/affine chains, branches, joins, multiple external data inputs, multiple
  final outputs, repeated intermediate uses, and disconnected nodes. CPU 0008A remains the owner
  of general partition-DAG decomposition and bounded horizontal or general vertical fusion.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Consume normalized axes, coordinates, permutations, lengths, signed steps, exact target Shapes,
  and prefix Shapes as captured. CPU validates them against projected descriptors but does not
  normalize caller syntax or invent compiler semantics.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [target glossary](../../../../glossary.md#term-target); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Require every `TargetShapeAttrs.targetShape`, `CropToShapeAttrs.targetShape`, and
  `CropToShapeAttrs.prefixShape` used by the admitted mapping to be fully static. Current Prepare
  supplies no binding substitution for symbolic attribute values.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [binding glossary](../../../../glossary.md#term-binding))</span>

### [Metadata](../../../../glossary.md#term-metadata)-only and physical-boundary rules

- Represent each internal affine result as a topology-local metadata value. Compose its mapping
  into the final source load and do not emit an instruction, class call, buffer declaration,
  workspace, Runtime slot, or store for that value.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [buffer glossary](../../../../glossary.md#term-buffer); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Retain every internal value and [descriptor](../../../../glossary.md#term-descriptor) in the [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) and `LogicalMemoryPlan`.
  CPU-private virtuality changes only CPU declarations and lowering.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering))</span>
- Declare exactly the source and final result buffers for an eligible chain. A final result always
  needs a declaration because current shared preparation has no value-alias assignment and
  publication/cross-partition consumers address a result <span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [publication glossary](../../../../glossary.md#term-publication); [partition glossary](../../../../glossary.md#partition))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), not an input representation.
- Write the final result according to its exact resolved `LayoutDescriptor`, including offset,
  positive strides, zero strides, referenced element span, and view classification. Do not silently
  replace it with canonical dense geometry.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [referenced element span glossary](../../../../glossary.md#referenced-element-span); [view glossary](../../../../glossary.md#term-view-meanings); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- For an injective result layout, the generated range domain is the logical result element range.
  For a zero-stride/non-injective result layout, derive a deterministic distinct-address domain and
  write each referenced result address at most once; allow parallel orchestration only when those
  writes are disjoint. Every omitted repeated logical coordinate must map to the same source value,
  or lowering fails closed.<span data-glossary-navigation="true"> ([injective glossary](../../../../glossary.md#term-injective); [layout glossary](../../../../glossary.md#layout); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Preserve exact source/output accessed spans and reject any source/result overlap that cannot be
  proved safe. Do not add in-place view mutation or copy-direction heuristics.
- Zero-[element](../../../../glossary.md#term-element) results perform no [carrier](../../../../glossary.md#term-storage-carrier) access. [Scalar](../../../../glossary.md#term-scalar) results use one logical element.

### [CPU](../../../../glossary.md#term-cpu)-private [IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering), and generated execution

- Add one small sealed CPU-private portable-kernel-IR role shared only by the existing pointwise
  IR and one new affine-copy IR. The existing pointwise structure and semantic inventory remain
  unchanged.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [kernel glossary](../../../../glossary.md#kernel); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- The affine-copy IR records only topology-local type, the exact composed structural mapping,
  source and result access-plan forms, iteration/write-domain form, and one ordered store. It does
  not retain <span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>[`Operation`](../../../../glossary.md#operation), `CompiledNode`, graph or Runtime identities, concrete carriers, slots,
  addresses, route, worker, or artifact-store state.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Keep family interpretation in a dedicated affine lowerer. `CpuPartitionLowering` dispatches
  cold analysis to the existing pointwise lowerer path or the new affine path; it must not become
  a monolithic switch containing both families' algorithms.
- Add a family-specific affine emitter using Java 26 Class-File API generation. Reuse the existing
  carrier load/store and primitive odometer facilities where their contracts match; do not
  duplicate one implementation per carrier or data type.<span data-glossary-navigation="true"> ([Class-File API glossary](../../../../glossary.md#term-class-file-api); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Generated entry descriptors remain specialized to the exact ordered source/result carrier
  pattern. Admit the six exact primitive-array forms plus <span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment), producing exactly seven
  final `CarrierAccess` forms. For each logical [data type](../../../../glossary.md#data-type--datatype), admit its matching array form and
  `MemorySegment`, including all-array, all-segment, and both mixed two-boundary patterns.
- Generated [BFLOAT16](../../../../glossary.md#term-bfloat16) affine loads and stores use [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) `short` array operations or exact native-order
  two-byte `MemorySegment` access and preserve the 16 represented bits unchanged. These carrier
  primitives are available only to the affine copy emitter/reference path in this task; they do
  not make BFLOAT16 a supported pointwise scalar or vector lane type.<span data-glossary-navigation="true"> ([represented bits glossary](../../../../glossary.md#term-represented-bits); [scalar glossary](../../../../glossary.md#term-scalar); [vector lane glossary](../../../../glossary.md#simd))</span>
- Generated affine hot paths receive direct primitive carriers, cold geometry, and primitive
  `start`/`end` bounds. They perform no operation/graph/shape/layout/route/carrier switch, map
  lookup, reflection, allocation, virtual cursor call, or per-element division/modulo.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [lookup glossary](../../../../glossary.md#term-lookup); [element glossary](../../../../glossary.md#term-element))</span>
- Preserve exactly the existing scalar/vector compute axis and single/parallel orchestration
  vocabulary. Affine-copy bodies are scalar-compute in this task. <span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [axis glossary](../../../../glossary.md#term-axis))</span>`VECTOR_IF_ELIGIBLE` therefore
  selects deterministic scalar fallback, while eligible disjoint ranges may select
  parallel-scalar orchestration. Do not claim vector, gather, scatter, or masked-tail execution.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather))</span>
- Preserve the four-complete-candidate, one-realized-artifact, zero-fixed-shape, and zero-unroll
  ceilings. An affine chain forms one direct candidate and one artifact; it creates no optional
  materialization workspace because its boundary copy is the semantic realization itself.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- Advance the CPU generator schema exactly once. Include IR family, mapping topology, data type,
  source/result access structure, carrier pattern, write-domain form, execution compute form, and
  every other bytecode/compatibility-changing fact. Retain existing exclusions for compatible
  extents, offsets/stride magnitudes within one structure, slots, carriers, addresses, workers,
  graph/run identity, and artifact root.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [bytecode glossary](../../../../glossary.md#term-bytecode); [stride glossary](../../../../glossary.md#term-stride); [run glossary](../../../../glossary.md#term-run); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Keep optional generated-class persistence disabled by default and retain one current schema
  with no migration reader. Finalization realizes or reuses the selected artifact only after exact
  shared assignments are present.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
- Extend the scalar reference path with the same composed mapping for differential evidence. It
  remains a test/conformance reference, never a Runtime operation or IR interpreter.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>

### [Lifecycle](../../../../glossary.md#lifecycle) and failure behavior

- CPU analysis performs Model/graph inspection, chain validation, mapping composition, strategy
  selection, specialization, and exact two-buffer declaration. Shared Prepare sees only the
  opaque selected plan and declarations.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation); [specialization glossary](../../../../glossary.md#term-specialization); [buffer glossary](../../../../glossary.md#term-buffer); [Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- CPU finalization validates both assigned buffer positions before artifact access and constructs
  one immutable partition executable. It cannot change mapping, strategy, route, or resources.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [immutable glossary](../../../../glossary.md#term-immutable); [partition glossary](../../../../glossary.md#partition))</span>
- Cold binding validates exact type, carrier, size, alignment, accessibility, read-only output
  rejection, canonical BOOL inputs, and span non-overlap once. The bound invocation retains direct
  typed references.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alignment glossary](../../../../glossary.md#term-alignment); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
- [Runtime](../../../../glossary.md#term-runtime) executes only the prepared invocation. It does not receive [`Operation`](../../../../glossary.md#operation), `CompiledNode`,
  Shape, layout, affine IR, or route choice and does not interpret a view.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Generated/reference failures are deterministic and fail closed. No partial success changes the
  existing Runtime output-validity rules.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>

### Stop conditions

Stop implementation and report the exact conflict before broadening the diff if any of these is
required:

- assigning two distinct [graph](../../../../glossary.md#graph) values to one shared [Runtime](../../../../glossary.md#term-runtime) slot or representation;
- attaching a [CPU](../../../../glossary.md#term-cpu)-selected physical [layout](../../../../glossary.md#layout) to an unresolved value through a new shared contract;
- [binding](../../../../glossary.md#term-binding) a dynamic/symbolic [Shape](../../../../glossary.md#shape) or Shape-valued attribute during CPU preparation;
- changing Model, Compiler, Planning, shared Prepare, Runtime, public API, architecture, module
  dependencies, or Gradle;<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- accepting a multi-output, random, ordering, scatter/fold, mixed-family, or general-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) unit;
- adding a second route, workspace, artifact per analysis, package, or path outside the exact map;
  or<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- discovering that the current captured [descriptors](../../../../glossary.md#term-descriptor) do not determine one exact admitted mapping.

## Out of [scope](../../../../glossary.md#term-scope)

- PAD, TILE, CONCAT, STACK, UNFOLD_AXIS, UNFOLD2D, GATHER, GATHER_ELEMENTS, GATHER_ND, ONE_HOT,
  SLICE_UPDATE, every scatter, FOLD_AXIS, FOLD2D, SORT, ARGSORT, TOP_K, INITIAL_STATE, or DROPOUT.<span data-glossary-navigation="true"> ([PAD glossary](../../../../glossary.md#padding); [GATHER glossary](../../../../glossary.md#gather); [DROPOUT glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- Any dynamic or symbolic Shape execution, binding of Shape-valued attributes, negative-step
  layout invention, or a shared prepared physical-layout/alias contract.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding); [layout glossary](../../../../glossary.md#layout); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Multi-output CPU execution, state outputs, random algorithms, ordering workspaces, scatter
  duplicate handling, overlap accumulation, atomics, locks, or reductions.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- General DAG decomposition, horizontal fusion, general vertical fusion, fan-out materialization,
  or recognition/profitability work owned by CPU 0008A–0008C.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [fan-out glossary](../../../../glossary.md#term-fan-out); [materialization glossary](../../../../glossary.md#materialization))</span>
- CPU 0007 reductions/statistics/normalization; CPU 0008 matrix multiplication, convolution,
  pooling, attention, or loss; native/vendor routes; tuning; benchmarks; or relaxed math.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [matrix multiplication glossary](../../../../glossary.md#matrix-multiplication--matmul); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [loss glossary](../../../../glossary.md#loss))</span>
- Vector affine loads/stores, gather/scatter Vector API operations, masked tails, fixed-shape
  specialization, unrolling, or broadening CPU 0005D materialization.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [Vector API glossary](../../../../glossary.md#term-vector-api); [specialization glossary](../../../../glossary.md#term-specialization); [unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- Model, Tensor API, Compiler, Config, Planning, shared Prepare, Runtime, Backend Contract, Trace,
  Engine, OpenBLAS provider, another backend, public API, architecture, dependency, Gradle,
  backend-conformance, or integration changes.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- A detailed specification for 0006A or later, a [commit](../../../../glossary.md#term-commit), or a push.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Runtime API](../../../../api/runtime-api.md)
- [Public API](../../../../api/public-api.md)
- [Glossary](../../../../glossary.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [Completed CPU 0005A architecture reset](0005a-atomic-partition-kernel-architecture-reset.md)
- [Completed CPU 0005B access and broadcasting](0005b-universal-access-plans-and-right-aligned-broadcasting.md)<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- [Completed CPU 0005D materialization gate](0005d-materialization-specialization-and-persistence-evidence-gate.md)<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- [Completed CPU 0005E family expansion](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
- [Completed CPU 0005J parity hardening](0005j-bounded-pointwise-coverage-and-parity-hardening.md)
- [Model capability baseline](../../../modules/model/capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../../../modules/model/master-plan.md)

## Architecture constraints

- Model owns every operation meaning and descriptor. CPU consumes validated captured semantics; it
  does not normalize Tensor API requests, change provenance, infer a missing layout, or define
  public behavior.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [descriptor glossary](../../../../glossary.md#term-descriptor); [CPU glossary](../../../../glossary.md#term-cpu); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [provenance glossary](../../../../glossary.md#provenance); [layout glossary](../../../../glossary.md#layout))</span>
- [Planning](../../../../glossary.md#term-planning) selects only `BackendId("cpu")`. CPU analysis owns family lowering, metadata folding,
  route/strategy selection, specialization, and exact resource declarations.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [metadata glossary](../../../../glossary.md#term-metadata); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- `LogicalMemoryPlan` remains complete. Only CPU analysis decides that an internal view value needs
  no `PreparationResourceRequirement.Buffer`; shared [Prepare](../../../../glossary.md#prepare) does not learn [view](../../../../glossary.md#term-view-meanings) semantics.
- Current shared contracts permit multiple outputs and multiple executable write selections, but
  the current CPU pipeline does not. This task does not change that CPU-private boundary because
  every admitted node has one output. Draft 0006C/0006D must extend it before TOP_K/DROPOUT.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [DROPOUT glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- Current shared contracts do not assign two [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values to one physical slot or representation.
  Therefore only an internal same-unit view may be computation-free. A final/cross-partition/
  published view result is copied or rejected; the task must not fabricate shared aliasing.<span data-glossary-navigation="true"> ([aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- [Runtime](../../../../glossary.md#term-runtime) hot paths never see [`Operation`](../../../../glossary.md#operation), `CompiledNode`, graph IR, Shape, layout, route, or
  resource-selection policy. Generated hot paths never dispatch on any of them.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [Shape glossary](../../../../glossary.md#shape); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Portable Class-File bytecode remains the production baseline and the scalar reference remains
  conformance-only. Scalar, vector, parallel-scalar, and parallel-vector reporting must remain
  exact even though this family admits only scalar compute in 0006.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [scalar glossary](../../../../glossary.md#term-scalar); [conformance glossary](../../../../glossary.md#term-conformance); [vector glossary](../../../../glossary.md#term-vector))</span>
- One [partition](../../../../glossary.md#partition) still produces one [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable). No general [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [fusion](../../../../glossary.md#term-fusion) is authorized.
- Any need for a shared physical-layout contract, shared aliasing, another module change, or an
  architecture-rule change is a stop condition.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

### Architecture and no-change conclusions

- `ARCHITECTURE.md`, focused architecture explanations, ADRs, and architecture tests need no
  change because ownership, dependency direction, lifecycle placement, and hot-path rules remain
  unchanged.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Model and Tensor API need no change because all admitted kinds, attributes, Shapes, layouts,
  signatures, and provenance already exist and remain authoritative.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [signatures glossary](../../../../glossary.md#term-signature); [provenance glossary](../../../../glossary.md#provenance))</span>
- Compiler needs no change because capture/inference/validation already preserve the exact
  operations, descriptors, and output positions consumed through <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext).
- Planning and shared Prepare need no change because logical values remain complete, CPU analysis
  may omit same-unit buffer requirements, and shared assignment already treats the CPU plan
  opaquely.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Runtime needs no change because one executable with two selected buffers and explicit read/write
  access already fits its cold-binding, validity, schedule, and ownership contracts.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding))</span>
- Backend Contract, Config, Trace, Engine, OpenBLAS, other backends, backend conformance,
  integration tests, and Gradle need no change because the task adds one CPU-private portable
  family with no composition or cross-module behavior.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- The CPU guide, affected Javadocs/package summaries, glossary, task, master plan, and roadmap do
  require the targeted documentation pass because executable CPU capability and internal
  terminology change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — [occurrence](../../../../glossary.md#term-occurrence)-local truthful capability.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — portable kernel IR roles and structural
  access/mapping facts.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — family dispatch, affine chain
  validation, mapping composition, and boundary derivation.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — Class-File generation and direct
  primitive carrier/loop emission.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — deterministic strategy, declarations,
  opaque plan retention, and post-assignment finalization.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — selected portable IR and
  specialization facts.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema) and structural compatibility.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold typed binding and direct
  partition invocation.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — differential reference mapping.

Packages added, removed, or moved:

- None.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPortableKernelIr` — new sealed internal
  structural role implemented by the existing pointwise IR and the new affine IR; it exposes only
  deterministic structural identity needed by the portable route.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [pointwise glossary](../../../../glossary.md#term-pointwise); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr` — new immutable composed
  source-to-result mapping and distinct-write-domain contract.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAffineLayoutLowering` — new family
  lowerer because Model-kind interpretation and affine composition belong in lowering, not the
  generator or Runtime.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAffineCopyEmitter` — new
  family-specific bytecode emitter; shared carrier and loop primitives remain in their existing
  owners.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Existing `CpuKernelSpecialization.CarrierAccess` gains exactly `SHORT_ARRAY`, and existing
  `CpuPreparedExecutable` cold [binding](../../../../glossary.md#term-binding) recognizes `CpuBufferArgument.Shorts`; no new storage
  representation type is needed because task 0001 already established that exact raw carrier.
- Existing `CpuPartitionLowering` becomes only the small cold family dispatcher plus the retained
  pointwise path. It must not absorb the affine algorithm.

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineCopyEmitter.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLoopEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAccessPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAffineCopyIr.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAffineLayoutLowering.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineCopyGeneratedKernelTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAffineCopyIrTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAffineLayoutLoweringTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected explanatory documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0006-portable-static-affine-views-and-boundary-materialization.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only paths include [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), Config, Planning, shared [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Backend](../../../../glossary.md#backend)
Contract, [Trace](../../../../glossary.md#trace), public [API](../../../../glossary.md#term-application-programming-interface-api) guides, architecture/ADRs/tests, [Gradle](../../../../glossary.md#term-gradle), backend [conformance](../../../../glossary.md#term-conformance),
integration tests, native routes, and later CPU tasks.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 40 paths:

| Category | Maximum | Accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production/package | 22 | Exact eighteen existing paths plus four new focused internal types listed above |
| CPU tests | 13 | Exact ten existing tests plus three new focused tests listed above |
| Explanatory documentation | 2 | CPU [backend](../../../../glossary.md#backend) guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **40** | **22 + 13 + 2 + 3** |

The larger-than-normal path count is required because one new [kernel](../../../../glossary.md#kernel) family must remain coherent
across capability truth, family [lowering](../../../../glossary.md#lowering), route-neutral [IR](../../../../glossary.md#term-intermediate-representation-ir), Class-File emission, [specialization](../../../../glossary.md#term-specialization)
compatibility, staged finalization, cold [binding](../../../../glossary.md#term-binding), reference evidence, and documentation. The
semantic scope remains eight named one-output operations, one straight-line chain form, one source,
one result, one portable route, [scalar](../../../../glossary.md#term-scalar) compute, and no shared-[module](../../../../glossary.md#term-module) change. Unused authorized
paths remain unchanged. Adding `SHORT_ARRAY` changes only already-listed specialization, emitter,
executable, reference, and test paths. Advancing the generator [schema](../../../../glossary.md#term-schema) from 10 to 11 also requires
the existing `CpuGeneratedKernelArtifactStoreTest` current-schema assertion to advance, so the
exact accounting is 22/13/2/3 with a 40-path ceiling. If another path, package, module, [operation](../../../../glossary.md#operation),
workspace, output, or execution route is required, stop and revise the plan rather than consuming
the ceiling.

## Acceptance criteria

- `CpuCapabilityProvider` returns true only for the exact admitted static/resolved occurrence
  matrix and false for every adjacent kind, wrong attributes, wrong arity, type mismatch, dynamic
  Shape/attribute, unresolved layout, and deferred operation.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [arity glossary](../../../../glossary.md#term-arity); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [operation glossary](../../../../glossary.md#operation))</span>
- One through eight admitted nodes lower only as one exact straight-line one-input/one-output
  affine chain. Branches, joins, mixed pointwise nodes, fan-out, publication of an intermediate,
  cross-partition intermediate use, and disconnected nodes fail before artifact access.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [pointwise glossary](../../../../glossary.md#term-pointwise); [fan-out glossary](../../../../glossary.md#term-fan-out); [publication glossary](../../../../glossary.md#term-publication); [partition glossary](../../../../glossary.md#partition); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Every eligible internal affine result remains present in graph/logical-memory facts but has no
  CPU declaration, Runtime slot, workspace, computation instruction, or store.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- The final source and result are the only buffer declarations, in deterministic order. Final
  boundary work uses one affine IR, one generated class/artifact, one partition executable, and
  one bound invocation.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
- Reshape, expand, permutation, singleton-axis insertion/removal, select, every admitted normalized slice,
  target-relative crop, and contiguous mappings match the current Model contracts for every
  admitted resolved-layout case.<span data-glossary-navigation="true"> ([permutation glossary](../../../../glossary.md#permutation); [singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice); [target-relative crop glossary](../../../../glossary.md#target-relative-crop); [contiguous glossary](../../../../glossary.md#term-contiguous); [Model glossary](../../../../glossary.md#term-model))</span>
- FLOAT64/FLOAT32/BFLOAT16/INT32/INT64/BOOL results preserve exact represented bits across array,
  segment, and mixed source/result carriers. No conversion, promotion, arithmetic, NaN canonicalization,
  signed-zero change, or BOOL truthiness is introduced.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [represented bits glossary](../../../../glossary.md#term-represented-bits); [NaN glossary](../../../../glossary.md#term-nan); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- `CpuKernelSpecialization.CarrierAccess` has exactly seven forms after the task, including the new
  `SHORT_ARRAY`; its method [descriptor](../../../../glossary.md#term-descriptor) is `short[]`, its only matching logical heap type is
  BFLOAT16, and cold binding maps the existing <span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [binding glossary](../../../../glossary.md#term-binding))</span>`CpuBufferArgument.Shorts` variant to it rather than
  falling through to `MEMORY_SEGMENT`. Wrong type/[carrier](../../../../glossary.md#term-storage-carrier) pairs fail cold.
- [BFLOAT16](../../../../glossary.md#term-bfloat16) affine fixtures prove exact raw-bit preservation for `short[]`→`short[]`,
  `short[]`→segment, segment→`short[]`, and segment→segment paths, including payloads representing
  zeros, infinities, and NaNs. Existing BFLOAT16 pointwise capability/lowering rejection remains
  regression-covered; no BFLOAT16 arithmetic, conversion, comparison, numerical, or vector claim
  is introduced.<span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [lowering glossary](../../../../glossary.md#lowering); [regression glossary](../../../../glossary.md#term-regression); [vector glossary](../../../../glossary.md#term-vector))</span>
- Exact offsets, positive/zero strides, accessed spans, scalar and zero-element Shapes, and
  distinct-address output domains are correct. Non-injective output layouts never create parallel
  duplicate writes, and unsafe aliasing fails cold.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [element glossary](../../../../glossary.md#term-element); [injective glossary](../../../../glossary.md#term-injective); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- [Scalar](../../../../glossary.md#term-scalar) and parallel-scalar generated execution handle arbitrary legal `start`/`end` ranges.
  Vector preference falls back to scalar compute truthfully; no vector/gather/scatter claim appears.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather))</span>
- Generated hot code has direct typed arguments and contains no Model/graph/Shape/layout/route
  dispatch, map lookup, reflection, storage discovery, allocation, virtual cursor, or per-element
  division/modulo.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [dispatch glossary](../../../../glossary.md#term-dispatch); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- The new sealed IR seam is CPU-private and minimal. Existing pointwise IR, all forty-eight
  opcodes, schema-10 behavior before the planned version advance, four execution-strategy names,
  one-copy materialization policy, and pointwise validation remain regression-covered.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [materialization glossary](../../../../glossary.md#materialization); [validation glossary](../../../../glossary.md#term-validation))</span>
- `CpuGeneratorSchema.CURRENT_VERSION` advances exactly from 10 to 11, and the existing
  `CpuGeneratedKernelArtifactStoreTest` current-[schema](../../../../glossary.md#term-schema) assertion advances with it.
- Specialization and artifact identity contain every affine code-shaping compatibility fact and
  exclude all instance/runtime identities. Older schemas fail closed without migration; optional
  persistence stays disabled by default.<span data-glossary-navigation="true"> ([Specialization glossary](../../../../glossary.md#term-specialization); [identity glossary](../../../../glossary.md#term-identity-meanings); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- No shared contract, public API, architecture, dependency, Gradle, conformance, integration,
  native route, tuning, CPU 0007/0008 family, or CPU 0008A fusion work changes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- A separate clean documentation-focused pass finalizes affected Javadocs/package summaries, the
  CPU guide, glossary impact, this task evidence, CPU master plan, and roadmap. It reuses the
  implementation pass's successful Java evidence unless it changes executable behavior.
- [CPU](../../../../glossary.md#term-cpu) 0005A–0005J remain `Complete`; 0006 is `Complete` only after implementation and all
  validation passed; 0006A and later remain `Draft` without detailed specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused validation:

```bash
./gradlew :backends:cpu:test --tests 'io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIrTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAffineLayoutLoweringTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAffineCopyGeneratedKernelTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest'
./gradlew :backends:cpu:test
```

The final [CPU](../../../../glossary.md#term-cpu) command is the sole normal affected-[module](../../../../glossary.md#term-module) Java evidence after executable code
stabilizes. Record suite/test counts, failures, errors, skips, Java 26 [toolchain](../../../../glossary.md#term-toolchain) [identity](../../../../glossary.md#term-identity-meanings), and any
environmental limitation.

Required automated/manual checks include:

- exhaustive admitted/rejected [operation](../../../../glossary.md#operation)/attribute/type/[Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout) capability rows;
- one- and eight-node chains; intermediate no-declaration/no-slot proof; exactly two boundary
  declarations; and rejection of branch, join, fan-out, publication, mixed-family, and
  cross-partition intermediate cases;<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [fan-out glossary](../../../../glossary.md#term-fan-out); [publication glossary](../../../../glossary.md#term-publication); [partition glossary](../../../../glossary.md#partition))</span>
- all eight named semantics, both slice attribute variants where static/resolved, scalar and
  zero-element cases, offsets, positive/zero strides, injective and distinct-address write domains;<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective))</span>
- raw-bit fixtures for all six data types, every per-type array/segment pair, representative mixed
  carriers, arbitrary ranges, worker boundaries, and deterministic scalar fallback; BFLOAT16 must
  cover all four <span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>`SHORT_ARRAY`/`MEMORY_SEGMENT` source-result combinations and adversarial raw
  16-bit payloads without conversion;
- exact seven-form [carrier](../../../../glossary.md#term-storage-carrier) inventory, `SHORT_ARRAY`→`short[]` entry descriptors, specialization-key
  equality/difference, <span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [specialization glossary](../../../../glossary.md#term-specialization))</span>`CpuBufferArgument.Shorts` cold recognition, wrong BFLOAT16 carrier
  rejection, and unchanged BFLOAT16 pointwise rejection;<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- generated/reference differential results and cold [alias](../../../../glossary.md#aliasing-and-arena)/read-only/size/[alignment](../../../../glossary.md#term-alignment)/[liveness](../../../../glossary.md#term-liveness) checks;
- structural-key equality/difference, schema invalidation, no migration, one-artifact ceiling,
  the artifact-store current-schema assertion at 11, no default persistence change, and unchanged
  pointwise regression behavior;<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [artifact glossary](../../../../glossary.md#term-artifact); [persistence glossary](../../../../glossary.md#term-persistence); [regression glossary](../../../../glossary.md#term-regression))</span>
- generated-bytecode or automated source-shape checks for forbidden hot-path dependencies. Add a
  stable automated assertion rather than relying on a repeated manual <span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>`javap` check.

Documentation-focused pass after executable evidence:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Also validate local Markdown links and anchors, balanced fences, final newlines, trailing
whitespace, the exact changed-path set and 40-path ceiling, [canonical](../../../../glossary.md#term-canonical-form) task headings, `Complete`
status/dependencies, 0005A–0005J `Complete`, 0006A+ `Draft`, and absence of any later detailed task
specification.

Repository-wide validation is deferred to CPU 0009 portable generated-coverage closure and [CI](../../../../glossary.md#term-continuous-integration-ci).
No shared module, [dependency](../../../../glossary.md#term-dependency), build, architecture, [conformance](../../../../glossary.md#term-conformance), or integration contract changes in
this task, so a root `test` [run](../../../../glossary.md#term-run) would duplicate the focused CPU evidence without a recorded risk.

## Dependencies

- Completed CPU 0005A through 0005J, especially the partition reset, access/broadcast contracts,
  execution strategies, materialization/specialization budgets, typed carriers, and parity gates.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [materialization glossary](../../../../glossary.md#materialization); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Current [Model](../../../../glossary.md#term-model) [layout](../../../../glossary.md#layout)/index [operation](../../../../glossary.md#operation) [signatures](../../../../glossary.md#term-signature), static [`Shape`](../../../../glossary.md#shape), resolved `LayoutDescriptor`,
  typed scalar/carrier, and exact provenance contracts.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [carrier glossary](../../../../glossary.md#term-storage-carrier); [provenance glossary](../../../../glossary.md#provenance))</span>
- Current [Planning](../../../../glossary.md#term-planning) logical-memory and maximal same-[owner](../../../../glossary.md#term-owner) [partition](../../../../glossary.md#partition) contracts.
- Current staged [Prepare](../../../../glossary.md#prepare) projection/declaration/assignment/finalization contracts.
- Current Runtime multiple-selection access declarations, cold binding, validity, schedule, and
  per-run resource ownership contracts.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding); [run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0006A (Draft): non-affine [PAD](../../../../glossary.md#padding)/TILE/composition/unfold, [gather](../../../../glossary.md#gather) families, and one-hot.
- CPU 0006B (Draft): functional [slice update](../../../../glossary.md#slice-update), scatter families, and overlap-fold accumulation.
- CPU 0006C (Draft): stable SORT/ARGSORT and two-output TOP_K.
- CPU 0006D (Draft): INITIAL_STATE and three-output explicit-state [DROPOUT](../../../../glossary.md#dropout--inverted-dropout).
- CPU 0007 (Draft): reductions, scans, statistics, softmax/log-softmax, and normalization after
  0006D.<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- CPU 0008A (Draft): general [partition](../../../../glossary.md#partition)-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [decomposition](../../../../glossary.md#term-decomposition) and bounded [fusion](../../../../glossary.md#term-fusion) after family coverage.

These rows remain master-plan summaries. Do not create their detailed task files while 0006 is
the active implementation frontier.

## Architecture impact

Expected impact: None.

The current shared contracts are sufficient for this one-output [slice](../../../../glossary.md#slice) and, later, for multiple
outputs and state values. The current [CPU](../../../../glossary.md#term-cpu) implementation's single output/final-store assumption is
CPU-private and deliberately left for 0006C/0006D. The absence of shared physical [aliasing](../../../../glossary.md#term-aliasing) is not
silently repaired: it bounds computation-free [view](../../../../glossary.md#term-view-meanings) folding to same-unit internal values and
requires a final boundary copy or [fail-closed](../../../../glossary.md#term-fail-closed) result.

If implementation requires [Runtime](../../../../glossary.md#term-runtime) to inspect an [operation](../../../../glossary.md#operation)/[layout](../../../../glossary.md#layout), shared [Prepare](../../../../glossary.md#prepare) to interpret an
affine plan, two [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values to share one slot, a new public physical-layout contract, or any
other architecture change, stop and report the exact decision rather than editing architecture.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are a clean-context implementation agent in the Synaptik repository.

Read completely:
- AGENTS.md
- ARCHITECTURE.md
- docs/planning/planning-guide.md
- docs/planning/backends/cpu/tasks/0006-portable-static-affine-views-and-boundary-materialization.md
- the directly referenced completed CPU and current Model/Planning/Prepare/Runtime contracts

Implement CPU 0006 exactly as specified. Do not implement out-of-scope operations, general DAG
fusion, shared contracts, public APIs, architecture, dependencies, Gradle, or later task specs.
Add exactly the seventh `CarrierAccess` form `SHORT_ARRAY` and use it only for opaque BFLOAT16
affine copying through the existing `CpuBufferArgument.Shorts`; preserve the current BFLOAT16
pointwise capability/lowering rejection and make no BFLOAT16 arithmetic, conversion, numerical,
or vector claim. Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly from 10 to 11 and update the
authorized existing `CpuGeneratedKernelArtifactStoreTest` current-schema assertion accordingly.
Stop if the task requires a shared physical-layout/alias contract or exceeds the exact 40-path map,
including its 13 CPU test paths.
Do not commit or push.

After executable Java stabilizes, run the focused and sole final CPU validation and record exact
evidence. Then hand the task spec, final diff, affected APIs/behavior, architecture constraints,
expected CPU guide/glossary/Javadoc work, and test evidence to a separate clean documentation-
focused agent. That pass must follow docs/developer-guide/documentation-rules.md, finalize the
authorized documentation and Javadocs in the same overall change, reuse successful Java evidence
unless executable behavior changes, run CPU Javadoc and documentation checks, and update this task
with evidence, notes, completion summary, and final status. Mark 0006 Complete only after every
gate passes; keep 0006A and later Draft without specs.
```

## Local decisions

- The former broad 0006 family is split because affine view composition, non-affine/index-tensor
  movement, duplicate-target functional updates, ordering algorithms, and explicit-state random
  execution have different IR, workspace, output, determinism, and validation boundaries.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [tensor glossary](../../../../glossary.md#tensor); [target glossary](../../../../glossary.md#term-target); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [validation glossary](../../../../glossary.md#term-validation))</span>
- Resolved-layout static affine views are the first slice because current Prepare supplies static
  descriptors and current CPU access plans already model positive and zero strides. Unresolved
  physical layout selection is not inferred.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [Prepare glossary](../../../../glossary.md#prepare); [descriptors glossary](../../../../glossary.md#term-descriptor); [CPU glossary](../../../../glossary.md#term-cpu); [model glossary](../../../../glossary.md#term-model); [strides glossary](../../../../glossary.md#term-stride))</span>
- Raw [BFLOAT16](../../../../glossary.md#term-bfloat16) copying is safe within this [slice](../../../../glossary.md#slice) because current storage already represents it as
  `CpuBufferArgument.Shorts` or an exact typed segment. The task adds the missing `SHORT_ARRAY`
  generated carrier form and opaque two-byte load/store only. Storage availability does not make
  BFLOAT16 pointwise arithmetic executable, so all existing numerical capability gates remain
  fail-closed.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [opaque glossary](../../../../glossary.md#term-opaque); [pointwise glossary](../../../../glossary.md#term-pointwise); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Same-unit internal affine values are the only zero-work form. Current shared slot assignment has
  no alias contract for distinct values, so a final view value is materialized.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- The task admits scalar and parallel-scalar execution only. This preserves the existing four-name
  strategy model and deterministic scalar fallback without claiming Vector API gather behavior.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Vector API glossary](../../../../glossary.md#term-vector-api); [gather glossary](../../../../glossary.md#gather))</span>
- Shared multi-output and state contracts are adequate; later CPU tasks need CPU-private IR,
  executable, and emitter extensions rather than architecture changes.

## Known limitations

- Dynamic/symbolic [Shapes](../../../../glossary.md#shape) and Shape-valued attributes remain unsupported by current [Prepare](../../../../glossary.md#prepare).
- Negative-step slices normally have unresolved result layout and therefore remain unsupported in
  this task even though <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>`SliceAttrs` can represent them.
- A mixed affine/pointwise or branched CPU-owned partition may fail CPU preparation until CPU
  0008A generalizes partition decomposition. Occurrence capability does not promise that every
  combination of individually supported nodes is one eligible unit.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [decomposition glossary](../../../../glossary.md#term-decomposition); [Occurrence glossary](../../../../glossary.md#term-occurrence); [nodes glossary](../../../../glossary.md#node))</span>
- Boundary materialization preserves exact resolved layout geometry and may retain unused prefix
  or holes. This task adds no layout compaction, alias assignment, or representation pooling.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [alias glossary](../../../../glossary.md#aliasing-and-arena); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- No performance claim is made. Vector affine execution and broader materialization require later
  evidence and scope.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [scope glossary](../../../../glossary.md#term-scope))</span>
- BFLOAT16 support is representation-only affine movement. Pointwise arithmetic, conversion,
  comparison, reduction, ordering, random generation, and vector execution remain unsupported
  until their owning tasks establish exact numerical and determinism policies.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `/root`:

- Repository rules, architecture, focused lifecycle/boundary documentation, documentation General
  and Planning profiles, planning guide, roadmap, CPU master/completed-task history, current CPU
  source/test inventory, current Model operation/capability records, relevant public API/glossary
  contracts, and shared Planning/Prepare/Runtime contracts were reviewed before editing.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- A focused [carrier](../../../../glossary.md#term-storage-carrier) audit confirmed six current `CarrierAccess` forms, the existing
  `CpuBufferArgument.Shorts` and [BFLOAT16](../../../../glossary.md#term-bfloat16)→`short[]` representation mapping, the missing generated
  `SHORT_ARRAY` specialization/binding form, and the current explicit BFLOAT16 pointwise rejection.
  The plan therefore scopes one new seventh carrier form solely to opaque affine copying.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [binding glossary](../../../../glossary.md#term-binding); [pointwise glossary](../../../../glossary.md#term-pointwise); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- After the carrier correction, shell validation over the task, CPU master plan, roadmap, Git
  index, and untracked files passed: exact three-path scope; all twenty canonical headings in
  order; task/master <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>[`Ready`](../../../../glossary.md#term-ready) synchronization; exact 0006→0005J and 0007→0006D dependencies;
  0005A–0005J <span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>`Complete`; 0006A–0017 `Draft`; all twenty-nine inventoried layout/indexing/
  ordering/random operation names; exactly seven final carrier forms; no 0006A-or-later task spec;
  and absence of Java, Gradle, <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [Gradle glossary](../../../../glossary.md#term-gradle))</span>`ARCHITECTURE.md`, or `testing/` changes.
- A Ruby local-Markdown check resolved every relative file target and explicit heading anchor in
  the three changed planning files and passed balanced-fence, final-newline, and trailing-whitespace
  checks.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- `git diff --check` passed on the corrected final planning diff.
- Java tests and [Javadoc](../../../../glossary.md#term-javadoc) were not [run](../../../../glossary.md#term-run) because this planning change modifies no Java or Javadoc.

Implementation context `/root/cpu_0006_impl`:

- The prescribed focused task command passed after implementation stabilized.
- The sole final `./gradlew :backends:cpu:test` passed on OpenJDK 26.0.1+8-34. XML evidence
  recorded 29 suites, 140 tests, zero failures, zero errors, and one skipped opt-in persistence-
  evidence test. No environmental limitation was reported.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
- No executable Java changed after that final test evidence. The documentation pass therefore
  reused it as required instead of repeating the successful Java suites.

Documentation-focused context `/root/cpu_0006_docs`:

- Read the repository instructions and architecture contract; the current architecture index;
  planning guide and roadmap; documentation rules and General, API/Javadoc, Backend Guide,
  Planning, and Example profiles; CPU master plan and task 0006; completed CPU 0005A, 0005B,
  0005D, 0005E, and 0005J contracts; every final changed/created CPU production and test source;
  CPU guide and glossary; and directly relevant Model, Compiler, Planning, Prepare, Runtime,
  public API, architecture, build, conformance, integration, and native-route boundaries.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [Backend glossary](../../../../glossary.md#backend); [Compiler glossary](../../../../glossary.md#term-compiler); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Independently reviewed the final implementation and test diff. The pass finalized 17 authorized
  production/package Javadocs, the CPU guide, glossary, task record, CPU master plan, and roadmap.
  It documented the exact static/resolved affine chain, internal virtuality, final boundary copy,
  resolved offsets and positive/zero strides, distinct-address write domain, six-type raw-bit
  copying, BFLOAT16 representation-only <span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride))</span>`SHORT_ARRAY`, scalar/parallel-scalar behavior, vector
  fallback, schema 11, and cold lifecycle/failure boundaries without broadening supported scope.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [schema glossary](../../../../glossary.md#term-schema))</span>
- The first `./gradlew :backends:cpu:javadoc` attempt failed on a stale renamed `@param` and
  exposed incomplete new-type comments. After Javadoc-only correction, the required rerun passed
  with only the two expected incubating-Vector warnings. Inspection of generated pages confirmed
  rendered affine-copy, distinct-address, `SHORT_ARRAY`, portable-IR, route-plan, and executable
  contracts. No Java test was rerun because no executable behavior changed.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- The first three Ruby Markdown-check attempts failed in the checker itself: a Ruby regular-
  expression interpolation error, an incorrect `Dir.glob` invocation, and an incorrect GitHub
  duplicate-space slug rule. The corrected repository-local check passed 321 Markdown files,
  5,478 local links, 354 explicit heading
  anchors, balanced fences, final newlines, and trailing whitespace.
- Exact path-map validation passed for 34 changed paths within the exact 40-path map: 20 of 22
  authorized production/package paths, 9 of 13 authorized tests, both explanatory documents, and
  all 3 planning/status paths. Canonical headings, dependency/status synchronization, schema 11,
  exactly seven carrier forms, CPU 0005A–0005J <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>`Complete`, [CPU](../../../../glossary.md#term-cpu) 0006A–0017 `Draft`, and absence of
  any later detailed task specification passed.
- `git diff --check` passed on the final combined change.
- Architecture and focused architecture pages, ADRs, architecture tests, public Tensor/Compile/
  Runtime/Training APIs, Model capabilities/master plan, shared modules, Gradle, backend
  conformance, integration tests, OpenBLAS/native routes, and other backends remain unchanged.
  The reason is specific: CPU 0006 consumes existing Model semantics and staged shared contracts
  entirely inside one CPU-private route, changes no dependency or public/shared contract, and adds
  no composed Engine or cross-backend behavior.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [modules glossary](../../../../glossary.md#term-module); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Implementation notes

- Added a sealed CPU-private portable-IR role so the existing pointwise IR and the new affine-copy
  IR share only structural identity. Affine structure encodes mapping family and write-domain
  compatibility while exact composed addresses remain cold plan facts.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [CPU glossary](../../../../glossary.md#term-cpu); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Added a dedicated affine lowerer for one-through-eight connected static resolved-layout
  occurrences. Internal single-use values retain graph/logical-memory identity but create no CPU
  declaration or Runtime slot; source and final result are the only declared buffers.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- Final result materialization preserves the resolved layout. Repeated zero-stride coordinates are
  deduplicated into deterministic distinct-address writes only when they select one source value;
  ambiguous repetitions fail closed. Scalar and zero-element domains remain exact.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [stride glossary](../../../../glossary.md#term-stride); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [Scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element))</span>
- Added scalar generated/reference represented-bit copying for FLOAT64, FLOAT32, BFLOAT16, INT32,
  INT64, and BOOL across array, segment, and mixed carriers. <span data-glossary-navigation="true"> ([represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`SHORT_ARRAY` binds only existing raw
  BFLOAT16 `CpuBufferArgument.Shorts`; [pointwise](../../../../glossary.md#term-pointwise) [BFLOAT16](../../../../glossary.md#term-bfloat16) remains rejected.
- Preserved scalar/vector and single/parallel strategy vocabulary. Affine work selects scalar or
  parallel-scalar, and vector preference falls back deterministically without gather/scatter or
  masked-tail claims. Schema 11 invalidates older generated artifacts without migration.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [gather glossary](../../../../glossary.md#gather); [Schema glossary](../../../../glossary.md#term-schema); [artifacts glossary](../../../../glossary.md#term-artifact))</span>

## Completion summary

- Completed changes: implemented and documented bounded portable static affine view composition,
  CPU-private internal virtuality, and exact final boundary represented-bit materialization.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [materialization glossary](../../../../glossary.md#materialization))</span>
- Files changed or created: 34 authorized paths total—20 CPU production/package paths, 9 CPU test
  paths, 2 explanatory documentation paths, and 3 planning/status paths. The documentation pass
  finalized 17 of those production/package Javadocs plus all 5 authorized Markdown records.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the stabilized focused pass and sole final CPU 29-suite/140-test
  evidence; final CPU Javadoc, generated-page inspection, Markdown links/anchors/fences/newlines/
  whitespace, exact path map and ceiling, semantic/schema/status gates, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [schema glossary](../../../../glossary.md#term-schema))</span>`git diff --check`
  passed.
- Documentation-agent review: clean context `/root/cpu_0006_docs`; complete with no executable
  behavior or test changes.
- Documentation impact: CPU guide, glossary, task, CPU master plan, and roadmap now describe the
  implemented scope and boundary accurately.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Javadoc review: affected Javadocs and package summaries were finalized; authorized unchanged
  source paths retained accurate contracts and required no edit.
- Glossary impact: added the reusable CPU affine boundary-copy distinction and synchronized
  portable route, generated kernel, specialization, artifact, access, materialization, and
  virtual-intermediate terminology.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0006. CPU 0006A–0006D and later Draft rows retain all excluded
  non-affine/index/scatter/order/random, dynamic-layout, vector-affine, and general-DAG work.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [vector glossary](../../../../glossary.md#term-vector); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>

Status: Complete
