# [CPU](../../../../glossary.md#term-cpu) Task 0008G1: Portable [Pool1d Composition](../../../../glossary.md#pool1d-composition) [Validation](../../../../glossary.md#term-validation) and [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) Generated Execution

## Reader context

This task records two different pooling boundaries: exact NCW Pool1d composition recognition and
dedicated NCDHW Pool3d execution. `CpuPool1dCompositionLowering` accepts only the private
expand–Pool2d–squeeze chain and reuses schema-55 Pool2d code; component capability answers alone do
not establish recognition. Virtual rank edits describe validated views and introduce no intermediate
storage slots. Pool3d instead receives `CpuPool3dIr`, `CpuPool3dLowering`, `CpuPool3dEmitter`, and a
schema-56 identity.

Here, PartitionDag supplies structural topology, while `CpuPool3dReferenceKernel` supplies numerical
and performance comparisons. CPU preparation declares only external input/output buffers; shared
Prepare assigns them, and Runtime invokes bound geometry and carriers. Worker ranges own complete
output cells and their depth-height-width windows. No workspace, persistent model state, or generic
PoolNd runtime bridge is introduced.

Current-topic navigation (not a replacement for the historical record):

- [current visible Pool1d composition](../../../../reference/tensor/pooling.md#ncw-pool1d-composition) — Shows exact rank edits and the inherited singleton-height Pool2d semantics.
- [current rank-five pooling semantics](../../../../reference/tensor/pooling.md#ncdhw-pool3d-expressions) — Separates the first-class 3D window operation and literal-ceiling rules from composition recognition.

## Status

Complete.

## Goal

Validate and execute the exact visible [NCW](../../../../glossary.md#term-ncw) [Pool1d composition](../../../../glossary.md#pool1d-composition) without inventing a Pool1d
[operation](../../../../glossary.md#operation) kind, capability, [IR](../../../../glossary.md#term-intermediate-representation-ir) family, or generated body, and add first-class portable [CPU](../../../../glossary.md#term-cpu)
execution for the complete current static [NCDHW](../../../../glossary.md#conv3d--ncdhw) `MAX_POOL3D` and `AVERAGE_POOL3D` families.

The Pool1d path must recognize only
`EXPAND_DIMS(axis 2) -> POOL2D -> SQUEEZE(axis 2)`, keep both [rank](../../../../glossary.md#term-rank)-edit results virtual, and reuse
the existing [schema](../../../../glossary.md#term-schema)-55 Pool2d generated body. The [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) path must use a rank-five direct
complete-output-cell [scalar](../../../../glossary.md#term-scalar) body, with caller-owned parallelism only across disjoint output cells,
and prove that generated code has the same semantic algorithm, loop/[dataflow](../../../../glossary.md#term-dataflow) [shape](../../../../glossary.md#shape), and
avoidable-overhead profile as an optimal clean Java implementation of the same [specialization](../../../../glossary.md#term-specialization).

## [Scope](../../../../glossary.md#term-scope)

### Source-backed baseline

- Model task 0025I exposes Pool1d only as three ordinary graph occurrences. The middle Pool2d has
  height geometry <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>`(kernel, stride, padding, dilation) = (1, 1, 0, 1)` and the requested width
  geometry and literal ceil mode. There is no Pool1d operation kind or signature.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature))</span>
- [CPU](../../../../glossary.md#term-cpu) capability already reports the exact affine `EXPAND_DIMS` and `SQUEEZE` occurrences and the
  exact Pool2d occurrence independently. The current partition decomposer can seed the visible
  nodes separately, but <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [partition glossary](../../../../glossary.md#partition); [seed glossary](../../../../glossary.md#term-seed); [nodes glossary](../../../../glossary.md#node))</span>`CpuPartitionLowering` has no exact three-node Pool1d recognizer, so the
  views are not yet preserved as one Pool2d-backed executable unit.<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings))</span>
- [Model](../../../../glossary.md#term-model) task 0025J defines first-class [rank](../../../../glossary.md#term-rank)-five `MAX_POOL3D` and `AVERAGE_POOL3D`; Model 0025K
  defines the separate general three-dimensional unfold/fold algebra; Compiler 0006B1 and 0006B2
  capture, validate, and differentiate the Pool3d signatures. None of those completions advertises
  current CPU Pool3d ownership or execution.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- The current CPU provider reports Pool2d but not Pool3d. CPU preparation has a dedicated Pool2d
  IR/lowering/emitter/reference family and a rank-five Conv3d lowering/emitter that demonstrates
  checked NCDHW output decoding and depth-aware address arithmetic. Neither is a generic PoolNd
  bridge.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- Generator schema 55 belongs to Pool2d. Unchanged families project schema 52, MATMUL projects
  schema 54, and Pool2d projects schema 55. Pool3d needs a new identity projection; recognized
  Pool1d must retain the existing Pool2d projection.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

### Exact [Pool1d composition](../../../../glossary.md#pool1d-composition) [validation](../../../../glossary.md#term-validation)

- Recognize exactly three [nodes](../../../../glossary.md#node) in strict order: [rank](../../../../glossary.md#term-rank)-three [NCW](../../../../glossary.md#term-ncw) `EXPAND_DIMS` with normalized
  axis 2, matching rank-four Pool2d, then <span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis))</span>`SQUEEZE` with normalized axis 2. Both graph edges must be
  the exact producer-to-consumer values; both intermediate values must be private to this chain,
  unpublished, and single-use.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [producer glossary](../../../../glossary.md#term-producer))</span>
- Require the expanded [descriptor](../../../../glossary.md#term-descriptor) to be `[N,C,1,W]` with the exact affine view layout derived by
  the current CPU affine rules. Require the squeezed output to be <span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [layout glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu))</span>`[N,C,Wout]` with the exact view
  layout obtained by removing axis 2. Revalidate type, gradient eligibility, fully static Shape,
  resolved non-negative layout, and exact descriptor identity relationships at lowering.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [gradient glossary](../../../../glossary.md#term-gradient); [Shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings); [lowering glossary](../../../../glossary.md#lowering))</span>
- Accept only `MAX_POOL2D` with `MaxPool2dAttrs` or `AVERAGE_POOL2D` with
  `AveragePool2dAttrs`. Height attributes must be exactly `(1,1,0,1)`; width attributes and
  `ceilMode` remain exact. Reject every mismatch, extra edge, fan-out, publication, reordered node,
  extra affine node, different axis, or noncanonical height geometry.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [publication glossary](../../../../glossary.md#term-publication))</span>
- Lower the exact chain as one Pool2d-backed executable unit. The external rank-three input and
  output are the only declared boundary values. The expanded and squeezed values remain virtual
  views, with no slot, copy, workspace, materialization, generated invocation, or Runtime step.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Reuse `CpuPool2dIr`, `CpuPool2dLowering.Geometry`, `CpuPool2dEmitter`, and schema-55 class
  identity. Translate the validated external NCW layouts into the exact rank-four singleton-height
  input/output layouts required by the existing Pool2d lowering. Do not create a Pool1d IR,
  emitter, reference kernel, schema, cache key, route, or fallback.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [singleton glossary](../../../../glossary.md#term-singleton); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [kernel glossary](../../../../glossary.md#kernel); [cache key glossary](../../../../glossary.md#term-cache-key))</span>
- Capability reporting remains occurrence-local and truthful: query and assert support separately
  for the actual <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>`EXPAND_DIMS`, Pool2d, and `SQUEEZE` occurrences. Do not add a synthetic Pool1d
  query or capability. A supported three-component capability conjunction does not by itself
  guarantee that the topology recognizer accepts a malformed chain.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>
- If exact recognition fails, retain the established legal partition-DAG decomposition rather
  than silently treating a near match as Pool1d. The recognized fast path must not hide generic
  dispatch, an interpreted bridge, or an unreported operation.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [dispatch glossary](../../../../glossary.md#term-dispatch); [operation glossary](../../../../glossary.md#operation))</span>

### [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) admitted operations and geometry

- Admit exactly `MAX_POOL3D` paired with `MaxPool3dAttrs` and `AVERAGE_POOL3D` paired with
  `AveragePool3dAttrs`, one input and one output, and no other [pooling](../../../../glossary.md#term-pooling) or window kind.
- Admit only [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), and [FLOAT64](../../../../glossary.md#term-float64) [rank](../../../../glossary.md#term-rank)-five input `[N,C,D,H,W]` and same-type,
  same-gradient-eligibility output <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>`[N,C,Dout,Hout,Wout]`. Shapes must be fully static, layouts
  resolved with non-negative offsets and strides, and output layout injective. Read-only input may
  be non-injective.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective))</span>
- For each depth, height, and width [axis](../../../../glossary.md#term-axis), compute with checked `long` arithmetic:

  ```text
  effective = dilation * (kernel - 1) + 1
  numerator = inputExtent + 2 * padding - effective
  floorOut  = floor(numerator / stride) + 1
  ceilOut   = ceil(numerator / stride) + 1
  ```

  Reject negative numerator, overflow, or output-Shape disagreement. Literal ceil mode retains a
  terminal all-padding window; it does not apply a framework-specific tail-removal rule.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [padding glossary](../../../../glossary.md#padding))</span>
- Output `(n,c,od,oh,ow)` starts at
  `(od*strideDepth-paddingDepth, oh*strideHeight-paddingHeight,
  ow*strideWidth-paddingWidth)`. Visit kernel positions in increasing depth, then height, then
  width order, applying the corresponding dilation.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [dilation glossary](../../../../glossary.md#term-dilation))</span>
- Support static zero batch or channel as zero work. Zero input depth, height, or width is
  admissible only when the checked padding geometry produces the supplied positive spatial output;
  its windows follow the all-padding rules. Reject every unrepresentable flattened cell count,
  address span, byte offset, range endpoint, or invocation geometry.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../../../glossary.md#term-channel); [spatial glossary](../../../../glossary.md#term-spatial-axes))</span>
- The [Model](../../../../glossary.md#term-model) divisor is a mathematical product. [CPU](../../../../glossary.md#term-cpu) [lowering](../../../../glossary.md#lowering) may [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) if the exact positive
  `kernelDepth * kernelHeight * kernelWidth` cannot be represented by the primitive invocation
  boundary, but must not reinterpret, saturate, wrap, or reduce that divisor.

### Maximum [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) contract

- Exclude padding from selection. A window with no in-bounds sampled coordinate writes exact
  negative infinity in the result type.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Any eligible NaN wins over every non-NaN, and the first eligible NaN wins. Otherwise use ordinary
  numeric maximum, rank positive zero above negative zero, and retain the first equal winner in
  increasing depth-height-width order. Real negative infinity is distinct from the all-padding
  state.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [rank glossary](../../../../glossary.md#term-rank); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Preserve the selected represented non-NaN value. For BFLOAT16, compare decoded values but retain
  the selected BFLOAT16 bits; do not introduce an accumulator narrowing. NaN payload/sign and
  signaling preservation remain unspecified beyond NaN class and first-winner selection.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [accumulator glossary](../../../../glossary.md#accumulator-type); [narrowing glossary](../../../../glossary.md#term-narrowing); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Use one-pass state shared across all three kernel loops. Do not allocate or publish winner
  indices, masks, counts, partials, or workspace.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [masks glossary](../../../../glossary.md#term-boolean-mask))</span>

### Average [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) contract

- Divide by the fixed positive mathematical count-[padding](../../../../glossary.md#padding) divisor
  `kernelDepth * kernelHeight * kernelWidth`. Dilation changes sampled coordinates, not the
  divisor. Every logical kernel position counts; an out-of-bounds position contributes conceptual
  exact positive zero.<span data-glossary-navigation="true"> ([Dilation glossary](../../../../glossary.md#term-dilation); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [kernel glossary](../../../../glossary.md#kernel); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- BFLOAT16 and FLOAT32 accumulate and divide in FLOAT32; FLOAT64 accumulates and divides in
  FLOAT64. BFLOAT16 narrows once after the one final division. The direct and generated CPU forms
  use identical increasing depth-height-width accumulation order.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Preserve the Model exceptional classes: an eligible NaN produces NaN; opposing infinities
  produce NaN; otherwise a present infinity retains its sign. An exact finite zero is negative
  only when every divisor position is an in-bounds negative zero. Cancellation, any positive zero,
  any padding position, or an all-padding window produces positive zero.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [finite glossary](../../../../glossary.md#term-finite); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [Cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- Implement the signed-zero rule explicitly. Do not claim NaN payload/sign preservation,
  cross-backend finite bit identity, reassociation, a valid-sample divisor, or a divisor override.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [backend glossary](../../../../glossary.md#backend); [identity glossary](../../../../glossary.md#term-identity-meanings); [reassociation glossary](../../../../glossary.md#term-reassociation); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>

### [Lowering](../../../../glossary.md#lowering), generated route, and execution [ownership](../../../../glossary.md#term-ownership-meanings)

- Add focused [CPU](../../../../glossary.md#term-cpu)-private `CpuPool3dIr`, `CpuPool3dLowering`, `CpuPool3dEmitter`, and
  `CpuPool3dReferenceKernel`. The reference kernel is test/performance evidence only and must be
  unreachable from production Runtime execution.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Select exactly `DIRECT_SCALAR`. Flatten output in stable NCDHW order; each half-open worker range
  owns complete output cells and every cell's complete depth-height-width window. Parallelize only
  across those cells through the existing caller-owned worker group. Add no depth/kernel split,
  partial reduction, combine, atomics, locks, nested workers, or Pool3d-specific worker threshold.<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [half-open glossary](../../../../glossary.md#term-half-open-interval))</span>
- Reuse the proven Pool2d generated design for carrier handling, max/average state, all-padding,
  average signed-zero tracking, preparation, and direct-vs-generated evidence. The extra depth
  dimension requires rank-specific output decoding, <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [padding glossary](../../../../glossary.md#padding); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [dimension glossary](../../../../glossary.md#dimension); [rank glossary](../../../../glossary.md#term-rank))</span>`kd` loop, depth bounds, depth strides,
  address arithmetic, and max first-winner/average accumulator state spanning all three loops.
  Implement those in the Pool3d emitter/reference; do not route through Pool2d, Conv3d, a generic
  PoolNd Runtime helper, unfold/fold, reflection, or an operation switch.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [accumulator glossary](../../../../glossary.md#accumulator-type); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- Use one general checked-long rank-five address body for all admitted resolved layouts, matching
  the established Pool2d design. Do not add an unsupported dense-int, vector, tiled, native,
  packed, or materialized algorithm in this task. Access plans and carriers remain class-identity
  facts even though the emitted body has one address form.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Support array, native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment), and all four input/output carrier pairings for each
  type. Reject output/input physical overlap before any write or worker submission. Declare exactly
  one input read and one output write, and zero workspace, materialization, scratch, columns,
  indices, counts, winners, or padding buffers.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- Thread [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) geometry through `LoweredPartition`, portable route planning, preparation,
  finalization, generated dispatch, prepared executable binding, and cache identity. Runtime sees
  only prepared primitive geometry, carriers/slots, range, and generated handle; it must not see
  Model <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [dispatch glossary](../../../../glossary.md#term-dispatch); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [binding glossary](../../../../glossary.md#term-binding); [cache glossary](../../../../glossary.md#term-cache); [Model glossary](../../../../glossary.md#term-model))</span>[`Operation`](../../../../glossary.md#operation), `CompiledNode`, [graph](../../../../glossary.md#graph) [topology](../../../../glossary.md#term-topology), or [layout](../../../../glossary.md#layout) objects.
- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 55 to 56. Pool3d uses schema 56 with every
  code-shaping kind/type/access/carrier/algorithm fact. Pool1d-recognized bodies and ordinary
  Pool2d retain schema 55; MATMUL retains 54; unchanged families retain 52. Advance compatibility
  envelopes to current-only 56 and prove old envelopes safe-miss without changing the retained
  class projections or bytes of unchanged controls.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Generated classes remain final, field-free, and constructor-free, with one typed static entry.
  Their hot loops may contain primitive indexing, direct typed array/segment access, comparisons,
  accumulation, one average division, and one store. They must contain no allocation, boxing,
  reflection, collection/map lookup, string dispatch, graph/layout/operation lookup, cache/route
  selection, worker management, fallback call, or Synaptik-owned hot helper call.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>

### Capability and fallback boundaries

- Extend `CpuCapabilityProvider` only for exact Pool3d kind/attribute pairs and the static
  descriptor/geometry boundary above. A <span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>`true` answer means the ordinary generated portable route
  can prepare that occurrence; capability must fail closed for mismatches or unrepresentable CPU
  geometry.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [occurrence glossary](../../../../glossary.md#term-occurrence); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Pool1d capability remains the conjunction of the three visible component queries. No public
  Model API, shared capability result, or backend-contract change is permitted.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend))</span>
- Pool3d becomes an atomic numerical seed in the partition DAG. No pooling epilogue fusion,
  pointwise materialization candidate, external affine suffix, or general DAG reconstruction is
  added. Surrounding unsupported or separately executable work follows existing decomposition and
  preparation fallback rules.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [seed glossary](../../../../glossary.md#term-seed); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [pooling glossary](../../../../glossary.md#term-pooling); [epilogue glossary](../../../../glossary.md#term-epilogue); [fusion glossary](../../../../glossary.md#term-fusion); [pointwise glossary](../../../../glossary.md#term-pointwise); [materialization glossary](../../../../glossary.md#materialization); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- The generated portable Pool3d body is the production semantic baseline and safe CPU route.
  There is no interpreted reference fallback, generic bridge, Vector API route, OpenBLAS route,
  native peer, or Runtime operation dispatch.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [Runtime glossary](../../../../glossary.md#term-runtime); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>

## Exclusions

- Attention, CPU 0008H details, losses, Conv3d changes, Conv3d gradients, Pool3d gradients,
  unfold/fold execution, unpooling, saved indices, and training-specific CPU state.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [losses glossary](../../../../glossary.md#loss); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [gradients glossary](../../../../glossary.md#term-gradient); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling))</span>
- New public Model, Compiler, Planning, Prepare, Runtime, backend-contract, or capability APIs;
  new Pool1d/PoolNd operations; architecture or module-boundary changes.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [module glossary](../../../../glossary.md#term-module))</span>
- Pooling fusion, materialization, workspace, vector/native execution, autotuning, dynamic or
  symbolic execution Shapes, unresolved layouts, negative strides, in-place output, or
  cross-backend bitwise/performance guarantees.<span data-glossary-navigation="true"> ([Pooling glossary](../../../../glossary.md#term-pooling); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization); [vector glossary](../../../../glossary.md#term-vector); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride))</span>
- Unrelated refactors, [schema](../../../../glossary.md#term-schema) cleanup, generic window abstractions, or 0008H task specification.

## Architecture references and constraints

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) is authoritative, especially Model semantic
  ownership, Planning occurrence ownership, backend Prepare ownership, finalization after shared
  slot assignment, and Runtime's prepared-schedule-only boundary.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Planning glossary](../../../../glossary.md#term-planning); [occurrence glossary](../../../../glossary.md#term-occurrence); [backend glossary](../../../../glossary.md#backend); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`current-architecture-plan.md`](../../../../architecture/current-architecture-plan.md) explains
  the current module topology; this task changes no dependency direction or public ownership.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [topology glossary](../../../../glossary.md#term-topology); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- [`planning-guide.md`](../../../planning-guide.md) governs Ready state, clean implementation and
  documentation contexts, validation tiers, completion evidence, and path-ceiling escalation.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Model 0025I](../../../modules/model/tasks/0025i-ncw-max-average-pool1d-composition.md),
  [0025J](../../../modules/model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md),
  and [0025K](../../../modules/model/tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md)
  are the semantic contracts. [Compiler 0006B1](../../../modules/compiler/tasks/0006b1-pool3d-and-3d-window-forward-adoption-and-explicit-gradient-boundary.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
  and [0006B2](../../../modules/compiler/tasks/0006b2-pool3d-and-3d-window-gradient-closure.md)
  are the capture/inference/gradient contracts.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- [CPU 0008G](0008g-portable-max-average-pool2d-execution.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> is the implementation and evidence
  precedent. Reuse its proven private design without turning it into a shared PoolNd framework.
- Preserve generated-code discipline from `AGENTS.md`: the optimal clean Java specialized case is
  the design/review oracle, and any hot-loop/dataflow deviation needs an explicit technical reason
  and evidence.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>

## Affected files and package impact

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths are the following 17; the category ceiling is 18:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `.../internal/cache/CpuGeneratorSchema.java`
- `.../internal/cache/CpuKernelSpecialization.java`
- `.../internal/codegen/emit/CpuClassFileKernelGenerator.java`
- new `.../internal/codegen/emit/CpuPool3dEmitter.java`
- `.../internal/executable/CpuPreparedExecutable.java`
- new `.../internal/ir/CpuPool3dIr.java`
- `.../internal/ir/CpuPortableKernelIr.java`
- new `.../internal/lowering/CpuPool1dCompositionLowering.java`
- `.../internal/lowering/CpuPartitionDagDecomposer.java`
- `.../internal/lowering/CpuPartitionLowering.java`
- new `.../internal/lowering/CpuPool3dLowering.java`
- `.../internal/prepare/CpuPartitionFinalizer.java`
- `.../internal/prepare/CpuPartitionPreparationPlan.java`
- `.../internal/prepare/CpuPartitionPreparer.java`
- new `.../internal/reference/CpuPool3dReferenceKernel.java`
- `.../internal/route/portable/CpuPortableRoutePlan.java`

The one-path production reserve is only for unavoidable mechanical propagation through an
existing [CPU](../../../../glossary.md#term-cpu)-private [schema](../../../../glossary.md#term-schema)/generated [carrier](../../../../glossary.md#term-storage-carrier) discovered during implementation. It cannot hold a
new abstraction, route, public [API](../../../../glossary.md#term-application-programming-interface-api), algorithm, or [module](../../../../glossary.md#term-module) boundary, and its reason must be recorded.

Expected CPU test/evidence paths, maximum 21:

- update `CpuCapabilityProviderTest`, `CpuInternalPackageInventoryTest`,
  `CpuGeneratedKernelArtifactStoreTest`, `CpuKernelSpecializationTest`,
  `CpuClassFileKernelGeneratorTest`, `CpuPartitionDagDecomposerTest`,
  `CpuPartitionPreparerTest`, `CpuPartitionFinalizerTest`, `CpuPreparedExecutableTest`, and the
  schema/projection controls in `CpuBatchNormTrainingEvidenceTest`, `CpuConv2dEvidenceTest`,
  `CpuConv3dEvidenceTest`, `CpuPartitionDagGeneratedEvidenceTest`, and
  `CpuPointwiseLedgerEvidenceTest`;
- add focused `CpuPool1dCompositionLoweringTest`, `CpuPool3dIrTest`,
  `CpuPool3dLoweringTest`, `CpuPool3dReferenceTest`, `CpuPool3dGeneratedKernelTest`, and
  `CpuPool3dPerformanceTest`;
- use the final remaining path only for an existing shape-polymorphism or persistence control when
  implementation evidence shows that schema-56 projection reaches it.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [persistence glossary](../../../../glossary.md#term-persistence))</span>

Expected documentation/[planning](../../../../glossary.md#term-planning) paths, maximum 9:

- `docs/backend-guide/cpu-backend.md`, `docs/api/tensor-api.md`, `docs/api/compile-api.md`, and
  `docs/glossary.md` when the documentation-focused review confirms affected current/future text;
- this task, the CPU master plan, [Model](../../../../glossary.md#term-model) capabilities, Model master plan, and roadmap.

Maximum [scope](../../../../glossary.md#term-scope) is 48 changed paths: at most 18 production/Javadoc, 21 tests/evidence, and 9
documentation/planning paths. Generated evidence [artifacts](../../../../glossary.md#term-artifact) outside tracked source do not count.
Stop and return to planning before exceeding any category or the total; do not hide excess scope
as mechanical churn.

Package impact is confined to `backends:cpu` implementation/test packages and explanatory/planning
documentation. No module [dependency](../../../../glossary.md#term-dependency), exported package, service registration, or [architecture test](../../../../glossary.md#term-architecture-test)
rule changes are expected.

## Acceptance criteria

1. Exact valid Pool1d chains lower as one Pool2d-backed executable with only external NCW
   boundaries, virtual rank edits, schema-55 identity, and byte-identical generated Pool2d class
   bytes for matched specialization facts.<span data-glossary-navigation="true"> ([NCW glossary](../../../../glossary.md#term-ncw); [rank glossary](../../../../glossary.md#term-rank); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [specialization glossary](../../../../glossary.md#term-specialization))</span>
2. Every malformed or nonprivate Pool1d near-match is rejected by recognition and safely follows
   existing decomposition. Capability tests report only the actual expand, Pool2d, and squeeze
   components; no Pool1d kind/query/IR/schema appears.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
3. CPU capability and lowering accept exactly the stated static Pool3d matrix and independently
   reject kind/attrs, type, rank, gradient, Shape, layout, injectivity, geometry, overflow, and
   representability mismatches.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [overflow glossary](../../../../glossary.md#term-overflow))</span>
4. Direct reference and generated Pool3d results agree for both kinds, all three types, floor and
   ceil, asymmetric per-axis parameter values, padding, dilation, stride, dense/noninjective-input/
   general layouts, arrays, segments, mixed carriers, zero domains, all-padding windows, overlap
   rejection, scalar ranges, and parallel range subdivision.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [parameter glossary](../../../../glossary.md#term-parameter); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation); [stride glossary](../../../../glossary.md#term-stride); [scalar glossary](../../../../glossary.md#term-scalar))</span>
5. Maximum tests prove first depth-height-width ties, first NaN, positive-over-negative zero,
   infinities, real negative infinity versus all-padding, BFLOAT16 selected-bit retention, and no
   winner state. Average tests prove fixed divisor, padding participation, one final BFLOAT16
   narrowing, accumulation domain/order, NaN, opposing/single infinities, cancellation,
   all-negative zero, positive zero, padding-forced positive zero, and all-padding positive zero.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [narrowing glossary](../../../../glossary.md#term-narrowing); [cancellation glossary](../../../../glossary.md#term-cancellation); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
6. Pool3d uses zero workspace/materialization and disjoint complete-output-cell ranges. Prepared
   Runtime state contains no Model/Compiler graph object and production cannot call the reference
   kernel or a generic fallback bridge.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [Runtime glossary](../../../../glossary.md#term-runtime); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [kernel glossary](../../../../glossary.md#kernel))</span>
7. Schema 56 is current-only compatible; Pool3d identity includes all code-shaping facts. Existing
   schema-52, schema-54, and schema-55 controls keep their intended projections and generated
   bytes. Recognized Pool1d adds no projection.
8. Generated structural evidence enumerates all 24 body families (`2 kinds * 3 data types * 4
   input/output carrier pairings`) because Pool3d deliberately has one general-long access body.
   For every family, retain the full Class-File and complete `javap -c -v -p` output and verify one
   typed entry, depth-height-width hot loops, direct load/store, and the forbidden-structure list.
   Tests fail if production selection emits an unenumerated body family.
9. The generated code has no avoidable helper dispatch, allocation, boxing, reflection,
   map/collection/string dispatch, cache/route/fallback overhead, or Synaptik-owned hot call. Any
   generation-time source helper must disappear into direct bytecode; no hidden generic bridge is
   accepted.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [cache glossary](../../../../glossary.md#term-cache); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
10. Five isolated fixed-heap Java 26 forks compare generated and direct optimal clean Java using
    randomized generated/direct order, five warmup rounds, nine measured rounds, adaptive batches
    of at least 25 ms, and no retry/discard. Every fork and aggregate generated/direct ratio is<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [warmup glossary](../../../../glossary.md#term-warmup); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
    `<= 1.15x` for at least dense FLOAT64 max/average 3x3x3, padded+dilated+ceil FLOAT32
    max/average, and mixed-carrier BFLOAT16 max/average. Add a row only if implementation creates
    another emitted hot-loop family.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
11. Documentation accurately distinguishes current Pool1d composition optimization from component
    capability and current Pool3d CPU execution from Model/Compiler semantics. A clean
    documentation-focused agent finalizes affected Javadoc, CPU guide, Tensor/Compile API text,
    glossary impact, and reasoned no-change conclusions.<span data-glossary-navigation="true"> ([Pool1d composition glossary](../../../../glossary.md#pool1d-composition); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
12. The implementation stays within the path ceiling, contains no unrelated attention/Conv3d/
    gradient/API/architecture work, and all required validation and evidence pass before the task
    status becomes Complete.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [validation glossary](../../../../glossary.md#term-validation))</span>

## [Validation](../../../../glossary.md#term-validation)

### Tier 1: focused semantic, [lowering](../../../../glossary.md#lowering), preparation, and [schema](../../../../glossary.md#term-schema) [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) the focused [CPU](../../../../glossary.md#term-cpu) test [owners](../../../../glossary.md#term-owner) for capability, Pool1d recognition, [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) reference/[IR](../../../../glossary.md#term-intermediate-representation-ir)/lowering/
generation, [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [decomposition](../../../../glossary.md#term-decomposition), preparation/finalization/executable [binding](../../../../glossary.md#term-binding), [cache](../../../../glossary.md#term-cache)/schema controls,
[artifact](../../../../glossary.md#term-artifact) [persistence](../../../../glossary.md#term-persistence), and unchanged generated-family projections. Record the exact [Gradle](../../../../glossary.md#term-gradle)
`--tests` command and test count in implementation evidence.

### Tier 2: generated structure and performance

- Generate and retain all 24 [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) family Class-Files and full `javap -c -v -p` reports under an
  explicit evidence directory. Run both semantic bytecode inspection and independent forbidden
  constant-pool/instruction/call-owner scans for every file.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [constant glossary](../../../../glossary.md#term-constant); [owner glossary](../../../../glossary.md#term-owner))</span>
- [Run](../../../../glossary.md#term-run) `CpuPool3dPerformanceTest` only under an explicit opt-in environment flag and evidence-root
  property, in five fresh 1 GiB Java 26 forks with the methodology and gates above. Retain raw
  samples, medians, ratios, fork metadata, generated classes, and decompilation. Do not rerun a
  failed fork to manufacture acceptance.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [fork glossary](../../../../glossary.md#term-fork-meanings); [metadata glossary](../../../../glossary.md#term-metadata); [decompilation glossary](../../../../glossary.md#term-decompilation))</span>

### Tier 3: [module](../../../../glossary.md#term-module) [checkpoint](../../../../glossary.md#term-checkpoint)

[Run](../../../../glossary.md#term-run) `./gradlew :backends:cpu:test` once after focused [validation](../../../../glossary.md#term-validation) because this task changes [CPU](../../../../glossary.md#term-cpu)
capability breadth, [partition](../../../../glossary.md#partition) [lowering](../../../../glossary.md#lowering), preparation plumbing, and generator compatibility. Run
`./gradlew :backends:cpu:javadoc` after the documentation/[Javadoc](../../../../glossary.md#term-javadoc) pass. Run `git diff --check` and
the repository's Markdown link/anchor/fence validation commands.

### Tier 4: boundary conclusions

- Backend conformance and integration source modules are currently marker modules with no pooling
  harness. Do not add placeholder tests. Record a reasoned no-change conclusion unless the
  implementation introduces a callable shared harness or end-to-end path, in which case stop and
  replan the added scope.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module); [pooling glossary](../../../../glossary.md#term-pooling); [scope glossary](../../../../glossary.md#term-scope))</span>
- No architecture/module dependency changes are planned, so architecture-test changes and a full
  repository test are not required. If implementation changes a dependency boundary, public
  contract, shared build configuration, or multiple modules, stop and obtain a revised plan before
  running the corresponding higher tier.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- The documentation-focused agent must not repeat successful Java suites unless it changes
  executable Java behavior or finds a concrete reason. It owns documentation validation and may
  reuse implementation evidence.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

Before marking Complete, verify exactly one detailed [Ready](../../../../glossary.md#term-ready)/Complete [CPU](../../../../glossary.md#term-cpu) 0008G1 task exists, all
links resolve, CPU 0008G is Complete, [Model](../../../../glossary.md#term-model) 0025I/J/K and [Compiler](../../../../glossary.md#term-compiler) 0006B1/B2 remain Complete, CPU
0008H remains Draft without a detailed task file, and roadmap/master-plan order agrees.

## Dependencies

- CPU 0008G, Model 0025I, Model 0025J, Model 0025K, Compiler 0006B1, and Compiler 0006B2 are
  Complete.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Existing CPU 0008A–0008G partition-DAG, recognition, preparation, direct-representation,
  generated-code, evidence, and Pool2d contracts remain authoritative.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- Java 26 Class-File generation/decompilation, fixed-heap fork harness, artifact retention, typed
  carriers, and caller-owned worker infrastructure already exist.<span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation); [heap glossary](../../../../glossary.md#term-heap); [fork glossary](../../../../glossary.md#term-fork-meanings); [artifact glossary](../../../../glossary.md#term-artifact))</span>

## Follow-ups

- CPU 0008H remains Draft and owns scaled-dot-product attention only after this task completes. Do
  not create its detailed task specification here.
- [CPU](../../../../glossary.md#term-cpu) 0009 later closes the portable generated-coverage capability/[conformance](../../../../glossary.md#term-conformance) [checkpoint](../../../../glossary.md#term-checkpoint).
- Pooling fusion, vector/native Pool3d, materialization, dynamic execution, and any shared
  conformance harness require separately planned evidence-backed tasks.<span data-glossary-navigation="true"> ([Pooling glossary](../../../../glossary.md#term-pooling); [fusion glossary](../../../../glossary.md#term-fusion); [vector glossary](../../../../glossary.md#term-vector); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [materialization glossary](../../../../glossary.md#materialization))</span>

## Architecture impact

No architecture change is intended. The task fills an existing [CPU](../../../../glossary.md#term-cpu)-[backend](../../../../glossary.md#backend) implementation slot:
[Model](../../../../glossary.md#term-model) keeps semantic [ownership](../../../../glossary.md#term-ownership-meanings), [Planning](../../../../glossary.md#term-planning) keeps [occurrence](../../../../glossary.md#term-occurrence) ownership, CPU [Prepare](../../../../glossary.md#prepare) chooses and binds
the generated route, and [Runtime](../../../../glossary.md#term-runtime) receives only [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) state. Any need for a new shared
[API](../../../../glossary.md#term-application-programming-interface-api), [module](../../../../glossary.md#term-module) edge, Runtime interpreter, Pool1d [operation](../../../../glossary.md#operation), or generic bridge is an architectural
uncertainty and must stop implementation for clarification.

## Implementation prompt

Use a mandatory separate clean implementation context. Give it this task, `AGENTS.md`, the
authoritative architecture/[planning](../../../../glossary.md#term-planning) documents, [Model](../../../../glossary.md#term-model) 0025I/J/K, [Compiler](../../../../glossary.md#term-compiler) 0006B1/B2, [CPU](../../../../glossary.md#term-cpu) 0008G,
and the directly affected CPU sources/tests. Tell it to implement only the bounded Java/test work,
[run](../../../../glossary.md#term-run) Tiers 1–3 through the CPU [module](../../../../glossary.md#term-module) [checkpoint](../../../../glossary.md#term-checkpoint), retain all generated/performance evidence, avoid
documentation finalization, make no [commit](../../../../glossary.md#term-commit) or push unless separately authorized, and return the
required completion summary with exact changed paths, commands, results, unresolved issues, and
`Status: Complete` or `Status: Incomplete`.

After implementation succeeds, use a distinct mandatory clean documentation-focused context. Give
it the final diff, task, relevant documentation rules and General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/Planning/Example
profiles, CPU guide, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) API docs, glossary, and affected Javadoc. It must independently
review and finalize documentation/Javadoc, record explicit no-change conclusions, run CPU Javadoc
and documentation [validation](../../../../glossary.md#term-validation) without repeating successful Java tests absent cause, enforce the
48-path ceiling and task status gates, and return the same required completion-summary format.

## Local decisions

- Pool1d is exact [topology](../../../../glossary.md#term-topology) recognition plus Pool2d reuse, never a capability or [operation](../../../../glossary.md#operation) family.
- Pool3d uses a dedicated rank-five direct scalar generated/reference loop. Pool2d supplies the
  numerical/evidence pattern; Conv3d supplies evidence that rank-five decode and depth addressing
  need rank-specific code. Neither is called as a hot-path bridge.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [rank glossary](../../../../glossary.md#term-rank); [scalar glossary](../../../../glossary.md#term-scalar); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- Pool3d has one general-long address form. Therefore the exhaustive emitted-body inventory is 24
  kind/type/carrier families, while semantic tests independently cover dense and general layouts.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [layouts glossary](../../../../glossary.md#layout))</span>
- Deterministic CPU average accumulation is depth-major, then height, then width, matching the
  direct oracle. This is an implementation choice within the Model contract, not a new Model rule.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [oracle glossary](../../../../glossary.md#term-test-oracle); [Model glossary](../../../../glossary.md#term-model))</span>
- Direct representation, zero workspace/materialization, complete-output-cell parallelism, and
  schema 56 are the only selected route/resource/compatibility choices.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [schema glossary](../../../../glossary.md#term-schema))</span>

## Known limitations

- Only fully static, resolved non-negative-[layout](../../../../glossary.md#layout) [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) occurrences are admitted.
- Output/input overlap, unrepresentable geometry/divisor/address domains, and noninjective outputs
  fail closed even when abstract Model semantics are otherwise meaningful.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [Model glossary](../../../../glossary.md#term-model))</span>
- No vector/native/materialized Pool3d route, Pool3d fusion, saved winner state, or Pool1d-specific
  artifact exists. Performance evidence is bounded to retained representatives and environment.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [fusion glossary](../../../../glossary.md#term-fusion); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- The shared backend-conformance and integration modules have no applicable executable pooling
  harness at planning time.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module); [pooling glossary](../../../../glossary.md#term-pooling); [planning glossary](../../../../glossary.md#term-planning))</span>

## Evidence

[Planning](../../../../glossary.md#term-planning) evidence inspected the clean current worktree; authoritative architecture/planning
documents; completed [Model](../../../../glossary.md#term-model) 0025I/J/K and [Compiler](../../../../glossary.md#term-compiler) 0006B1/B2; completed [CPU](../../../../glossary.md#term-cpu) 0008G; CPU capability,
affine, [partition](../../../../glossary.md#partition)-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag), Pool2d, [Conv3d](../../../../glossary.md#conv3d--ncdhw), [schema](../../../../glossary.md#term-schema), preparation, generated [artifact](../../../../glossary.md#term-artifact), [decompilation](../../../../glossary.md#term-decompilation), and
benchmark sources/tests; [backend capability](../../../../glossary.md#backend-capability) documentation; CPU guide; [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) docs;
glossary; and the [conformance](../../../../glossary.md#term-conformance)/integration marker [modules](../../../../glossary.md#term-module). This evidence supports the bounded
algorithm and [ownership](../../../../glossary.md#term-ownership-meanings) choices above. It is not implementation evidence.

## Notes

- Preserve the exact source-backed names and status ordering. Do not broaden this task because a
  nearby Pool3d window, gradient, Conv3d, or attention facility exists.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [gradient glossary](../../../../glossary.md#term-gradient); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- If the final source proves that any stated 24-family body inventory or schema projection cannot
  be expressed without a materially different hot loop or public boundary, stop and amend this
  Ready task rather than silently changing the design.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [Ready glossary](../../../../glossary.md#term-ready))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation-context validation completed on Java 26.0.1/OpenJDK 64-Bit Server VM:

- Tier 1 used one focused `./gradlew :backends:cpu:test` invocation with the 20 exact [owners](../../../../glossary.md#term-owner)
  `CpuCapabilityProviderTest`, `CpuInternalPackageInventoryTest`,
  `CpuPool1dCompositionLoweringTest`, `CpuPool3dIrTest`, `CpuPool3dLoweringTest`,
  `CpuPool3dReferenceTest`, `CpuPool3dGeneratedKernelTest`,
  `CpuPartitionDagDecomposerTest`, `CpuPartitionDagResourceTest`, `CpuPartitionPreparerTest`,
  `CpuPartitionFinalizerTest`, `CpuPreparedExecutableTest`,
  `CpuGeneratedKernelArtifactStoreTest`, `CpuKernelSpecializationTest`,
  `CpuClassFileKernelGeneratorTest`, `CpuBatchNormTrainingEvidenceTest`,
  `CpuConv2dEvidenceTest`, `CpuConv3dEvidenceTest`,
  `CpuPartitionDagGeneratedEvidenceTest`, and `CpuPointwiseLedgerEvidenceTest`. Result: 20 suites,
  181 tests, 0 skipped, 0 failures, and 0 errors.
- Tier 2 used the opt-in properties `pool3dPerformance=true` and
  `pool3dEvidenceRoot=/tmp/synaptik-cpu-0008g1-pool3d-evidence-20260830c` through a temporary
  test-environment init script, with `:backends:cpu:test --tests
  io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPool3dPerformanceTest
  --rerun-tasks`. The temporary init script was removed after the [run](../../../../glossary.md#term-run). Exactly five fresh `-Xms1g
  -Xmx1g` [forks](../../../../glossary.md#term-fork-meanings) passed; every individual row was at most `1.002680589x`. Median-of-five aggregate
  ratios were dense FLOAT64 max <span data-glossary-navigation="true"> ([Median glossary](../../../../glossary.md#term-median))</span>`0.955486129x`, padded/dilated/ceil [FLOAT32](../../../../glossary.md#term-float32) max `0.866583071x`,
  mixed-carrier BFLOAT16 max `0.988159577x`, dense [FLOAT64](../../../../glossary.md#term-float64) average `1.001036519x`,
  padded/dilated/ceil FLOAT32 average `0.966994679x`, and mixed-[carrier](../../../../glossary.md#term-storage-carrier) [BFLOAT16](../../../../glossary.md#term-bfloat16) average
  `0.932421527x`.
- Two earlier development evidence roots failed the per-fork FLOAT32 max gate before the clean
  Java oracle was corrected to preserve the generated general-long address/dataflow shape. They
  were not retried, averaged, or included in the accepted root. The final retained run followed
  the source correction and used a fresh evidence root.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape))</span>
- The accepted root contains five raw fork CSV files, all 24 distinct generated Class-Files, 24
  complete `javap -c -v -p` reports, 24 member-reference scans, 24 specialization snapshots, the
  direct-oracle source, environment metadata, aggregate results, and a 104-entry SHA-256 manifest.
  The manifest digest is<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [metadata glossary](../../../../glossary.md#term-metadata); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [manifest glossary](../../../../glossary.md#term-manifest); [digest glossary](../../../../glossary.md#term-digest))</span>
  `80457c6c181e0ae80e6c33c264ddbc02202c79bd3f67e76cf017172d540cf85c`.
  All generated families were final, field-free, constructor-free, single-entry bodies with no
  Synaptik, collection/map, reflection, allocation, boxing, string-dispatch, route/cache, or
  fallback hot call. Segment families retain only their required direct <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [cache glossary](../../../../glossary.md#term-cache))</span>`MemorySegment.get/set`
  calls.
- Tier 3 ran the required final `./gradlew :backends:cpu:test` checkpoint once after focused
  validation: 120 suites, 618 tests, 5 intentional opt-in skips, 0 failures, and 0 errors.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Schema controls passed with 56 current-only, Pool3d at 56, Pool1d-recognized and ordinary Pool2d
  at 55, MATMUL at 54, and unchanged generated families at 52. The recognized Pool1d and direct
  Pool2d specializations produced byte-identical generated classes.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Implementation stayed at 42 changed paths: 17 production/Javadoc, 20 tests/evidence, and 5
  planning paths. `git diff --check` passed. No architecture/module dependency, shared build,
  backend-conformance, or integration harness changed, so architecture-test, conformance,
  integration, and repository-wide validation remain reasoned no-change conclusions.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- CPU Javadoc, explanatory/API/glossary review, Markdown links/anchors/fences, final path audit,
  status/order audit, and the overall Complete transition remain owned by the mandatory independent
  documentation-focused context.

Independent documentation context `01a053e7-8514-7622-b16c-44ef10c39460` used the General,
[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Planning](../../../../glossary.md#term-planning), [Backend](../../../../glossary.md#backend) Guide, and Example profiles. It reviewed all 17 affected production
contracts, all 20 affected tests/evidence owners, the [CPU](../../../../glossary.md#term-cpu) guide, [Tensor](../../../../glossary.md#tensor) and [Compile](../../../../glossary.md#compile) API references,
glossary, [Model](../../../../glossary.md#term-model) capability/master-plan records, CPU master plan, roadmap, task lineage, and status
gates. It changed no executable Java behavior and did not repeat the successful Java suites.

- The first `./gradlew :backends:cpu:javadoc` completed but reported missing record-component and
  compatibility-constructor documentation in the provisional Javadocs. The documentation context
  corrected those comments and one inaccurate reference-oracle overlap claim. The final
  `./gradlew :backends:cpu:javadoc --rerun-tasks` passed with exactly the two expected
  `jdk.incubator.vector` warnings and no documentation warnings. Rendered `CpuPool3dIr`,
  `CpuPool3dLowering.Geometry`, and `CpuPool3dReferenceKernel` pages exist and contain the
  finalized component and failure contracts.
- The CPU guide now documents exact Pool1d recognition/schema-55 reuse and the bounded direct
  Pool3d schema-56 route. Tensor and Compile API text preserves Model/Compiler ownership while
  identifying only the fully static resolved-layout non-gradient CPU subset. The glossary now
  distinguishes visible Pool1d component capability from CPU topology recognition and records the
  bounded Pool3d execution boundary. No new reusable term was introduced.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [topology glossary](../../../../glossary.md#term-topology))</span>
- A targeted Ruby checker validated local Markdown targets, GitHub-style heading anchors, and
  balanced backtick/tilde fences in all nine changed Markdown files. Early checker-development
  invocations had a Ruby interpolation error and then an incomplete/incorrect slug calculation;
  neither was accepted. The corrected final invocation passed all nine files.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- Final scope is 46 paths: 17 production/Javadoc, 20 tests/evidence, and 9 documentation/planning.
  This remains below the 48-path total and within every category ceiling. No retained evidence
  artifact entered the repository.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Final status/order inspection confirmed CPU 0008G, Model 0025I/J/K, and Compiler 0006B1/B2
  remain Complete; CPU 0008G1 is the sole detailed task at this position and is Complete; CPU
  0008H remains Draft with no detailed task file. CPU master plan, Model records, and roadmap agree.
- `git diff --check` passed after final content/status edits. Architecture, ADR, package-summary,
  Tensor/Compile Java API, Training API/documentation, backend-conformance, integration,
  architecture-test, Gradle, and other-module changes were not required: the implementation stays
  inside the existing CPU-private prepare/finalize/executable boundary and changes no shared or
  public Java contract.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [prepare glossary](../../../../glossary.md#prepare))</span>

## Implementation notes

The implementation context added strict three-[node](../../../../glossary.md#node) [Pool1d composition](../../../../glossary.md#pool1d-composition) recognition and reuses the
existing Pool2d [IR](../../../../glossary.md#term-intermediate-representation-ir), geometry, [schema](../../../../glossary.md#term-schema)-55 [specialization](../../../../glossary.md#term-specialization), and generated body with only external [NCW](../../../../glossary.md#term-ncw)
boundaries. Both [rank](../../../../glossary.md#term-rank)-four intermediates remain virtual; malformed [topology](../../../../glossary.md#term-topology), affine [descriptors](../../../../glossary.md#term-descriptor),
height geometry, or [publication](../../../../glossary.md#term-publication) obligations fail recognition, and a valid near-match decomposes
through the existing [partition](../../../../glossary.md#partition) [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag). No Pool1d kind, capability, IR, schema, or [artifact](../../../../glossary.md#term-artifact) was added.

[Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) now has dedicated [CPU](../../../../glossary.md#term-cpu)-private IR, [lowering](../../../../glossary.md#lowering), clean-Java reference, and Class-File emitter
[owners](../../../../glossary.md#term-owner). Capability and lowering admit only non-[gradient](../../../../glossary.md#term-gradient) static resolved [NCDHW](../../../../glossary.md#conv3d--ncdhw) [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/
[FLOAT64](../../../../glossary.md#term-float64) max and fixed-divisor average occurrences, including noninjective read-only inputs and
[injective](../../../../glossary.md#term-injective) outputs. Preparation, schema projection, [persistence](../../../../glossary.md#term-persistence), finalization, [scalar](../../../../glossary.md#term-scalar)/caller-parallel
execution, overlap [validation](../../../../glossary.md#term-validation), and complete-output-cell range [ownership](../../../../glossary.md#term-ownership-meanings) [carry](../../../../glossary.md#term-carry) checked rank-five
geometry with zero workspace or [materialization](../../../../glossary.md#materialization). Generated max and average bodies preserve
depth-height-width/width-fastest traversal, [padding](../../../../glossary.md#padding)/[dilation](../../../../glossary.md#term-dilation)/[stride](../../../../glossary.md#term-stride), special-value and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)
rules, direct typed array/segment access, and the general-long address [shape](../../../../glossary.md#shape) of the direct [oracle](../../../../glossary.md#term-test-oracle).

The independent documentation-focused context finalized all affected production Javadocs,
including record-component and compatibility-constructor contracts, and narrowed the reference
oracle's failure documentation to its actual checks. It finalized the CPU guide, [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api),
[Compile](../../../../glossary.md#compile) API, glossary, capabilities, master plans, roadmap, and this task without changing
executable Java behavior or tests.

## Completion summary

Completed exact visible [NCW](../../../../glossary.md#term-ncw) Pool1d [topology](../../../../glossary.md#term-topology) recognition with virtual [rank](../../../../glossary.md#term-rank) edits and byte-identical
[schema](../../../../glossary.md#term-schema)-55 Pool2d reuse, plus direct schema-56 [NCDHW](../../../../glossary.md#conv3d--ncdhw) max/average [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) generated execution for the
bounded static [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [matrix](../../../../glossary.md#term-matrix-meanings). Implementation evidence passed 181 focused tests,
the 618-test [CPU](../../../../glossary.md#term-cpu) [checkpoint](../../../../glossary.md#term-checkpoint), all 24 structural family scans, and all retained five-[fork](../../../../glossary.md#term-fork-meanings) performance
gates; the two earlier FLOAT32 maximum development-root failures remain explicitly pre-fix and are
not acceptance evidence.

The independent documentation context finalized four explanatory/[API](../../../../glossary.md#term-application-programming-interface-api)/glossary documents, five
[planning](../../../../glossary.md#term-planning) documents, and affected Javadocs; passed final CPU [Javadoc](../../../../glossary.md#term-javadoc), nine-file Markdown
link/anchor/fence [validation](../../../../glossary.md#term-validation), exact 46-path [scope](../../../../glossary.md#term-scope), status/order, and whitespace checks; and recorded
reasoned no-change conclusions for shared/public APIs, package summaries, architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/tests,
[backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), integration, [Gradle](../../../../glossary.md#term-gradle), Training documentation, and other [modules](../../../../glossary.md#term-module). No [commit](../../../../glossary.md#term-commit) or
push was made. Unresolved issues: None. Follow-up required for CPU 0008G1: None. Draft CPU 0008H
remains separate and has no detailed task.

Status: Complete
