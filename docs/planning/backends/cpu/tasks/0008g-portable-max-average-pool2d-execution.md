# [CPU](../../../../glossary.md#term-cpu) Task 0008G: Portable Max/Average Pool2d Execution

## Reader context

This task records portable generated execution for static NCHW max and fixed-divisor average Pool2d.
`CpuPool2dLowering` validates descriptors, geometry, layouts, and ranges; `CpuPool2dIr` captures
code-shaping facts; `CpuPool2dEmitter` produces the `DIRECT_SCALAR` body. `CpuPool2dReferenceKernel`
is the semantic and performance oracle, not a Runtime fallback. Max selects a represented value,
whereas average uses its specified accumulator and explicit signed-zero policy.

Here, PartitionDag supplies authoritative structure, not numerical expected results. CPU analysis
declares only input/output buffers before shared Prepare assigns slots; finalization binds the
selected artifact. Runtime invokes typed carriers and primitive geometry. The CPU executable borrows
its supplied workers to divide complete output cells, with no persistent model state, workspace,
pooling fusion, or input materialization.

Current-topic navigation (not a replacement for the historical record):

- [current fixed-divisor Pool2d meaning](../../../../reference/tensor/pooling.md#nchw-average-pooling-expressions) — Explains that all kernel positions, including padding, count in average pooling.
- [current CPU preparation and invocation handoffs](../../../../explanation/packages/cpu.md#read-the-pipeline-before-the-packages) — Names the current analysis, shared assignment, finalization, cold binding, and Runtime participants.

## Status

Complete.

## Goal

Implement the complete current static [Model](../../../../glossary.md#term-model) `MAX_POOL2D` and `AVERAGE_POOL2D` families on the [CPU](../../../../glossary.md#term-cpu)
portable generated-[bytecode](../../../../glossary.md#term-bytecode) route. Preserve the exact [NCHW](../../../../glossary.md#term-nchw) literal floor/ceiling grid, [padding](../../../../glossary.md#padding),
[dilation](../../../../glossary.md#term-dilation), [extrema](../../../../glossary.md#saturation-pole-and-extremum), fixed-divisor, [accumulator](../../../../glossary.md#accumulator-type), special-value, empty-domain, [layout](../../../../glossary.md#layout), [carrier](../../../../glossary.md#term-storage-carrier),
[alias](../../../../glossary.md#aliasing-and-arena), and [publication](../../../../glossary.md#term-publication) contracts already established by Model tasks 0020A and 0020A1 and adopted
by [Compiler](../../../../glossary.md#term-compiler) task 0005D.

The implementation must provide one generated [scalar](../../../../glossary.md#term-scalar) production form for every admitted case,
reuse the existing caller-owned CPU workers only across disjoint complete output cells, and prove
the generated body against an equally specialized optimal clean Java [oracle](../../../../glossary.md#term-test-oracle). It must not add
[pooling](../../../../glossary.md#term-pooling) [epilogue](../../../../glossary.md#term-epilogue) [fusion](../../../../glossary.md#term-fusion), a native route, a public/shared abstraction, hidden [materialization](../../../../glossary.md#materialization), or a
[Runtime](../../../../glossary.md#term-runtime) [operation](../../../../glossary.md#operation) interpreter.

## [Scope](../../../../glossary.md#term-scope)

### Current source-backed baseline

- `Pool2dKind` contains exactly `MAX_POOL2D` and `AVERAGE_POOL2D`. Each kind accepts only its own
  nine-component attribute record, exactly one input, and exactly one output.
- `MaxPool2dAttrs` and `AveragePool2dAttrs` contain positive kernel, stride, and dilation
  components; non-negative symmetric padding per side; and one literal <span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [padding glossary](../../../../glossary.md#padding))</span>`ceilMode` flag. There is
  no asymmetric padding, automatic padding, count-padding flag, divisor override, padding value,
  or runtime geometry input.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Model](../../../../glossary.md#term-model) accepts [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), and [FLOAT64](../../../../glossary.md#term-float64) [rank](../../../../glossary.md#term-rank)-four [NCHW](../../../../glossary.md#term-nchw) input `[N,C,H,W]`. Output preserves
  the exact input type, gradient eligibility, batch extent, and channel extent and has Shape<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [extent glossary](../../../../glossary.md#term-extent); [channel glossary](../../../../glossary.md#term-channel); [Shape glossary](../../../../glossary.md#shape))</span>
  `[N,C,Hout,Wout]`.
- Model construction may retain symbolic spatial extents. CPU execution remains narrower: this
  task admits only fully static Shapes and resolved non-negative layouts whose exact geometry is
  independently revalidated during CPU capability analysis and lowering.<span data-glossary-navigation="true"> ([spatial glossary](../../../../glossary.md#term-spatial-axes); [layouts glossary](../../../../glossary.md#layout); [lowering glossary](../../../../glossary.md#lowering))</span>
- Compiler structured inference revalidates the one-input descriptor, attributes, rank, type,
  exact output Shape, and gradient eligibility. Compiler 0005D owns the already completed
  pooling-gradient formulas; CPU executes the resulting ordinary forward operations and does not
  implement autograd.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [descriptor glossary](../../../../glossary.md#term-descriptor); [autograd glossary](../../../../glossary.md#autograd))</span>
- [CPU](../../../../glossary.md#term-cpu) capability currently rejects both [pooling](../../../../glossary.md#term-pooling) kinds. The shared [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) already carries
  authoritative partition-local topology. CPU 0008B treats numerical semantic families as atomic
  units and materializes legal splits around them. CPU 0008C explicitly excludes pooling
  recognition and pooling epilogues. CPU 0008D ranks only existing recognized topology choices,
  and CPU 0008E retains pointwise materializations as candidate-only while ordinary preparation
  remains direct.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [topology glossary](../../../../glossary.md#term-topology); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Generator compatibility is currently schema 54. Unchanged families retain schema-52 class
  identity, and MATMUL retains schema-54 class identity. A new Pool2d emitted form must not reuse
  either identity projection.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>

### Exact admitted operations and geometry

- Admit only `Pool2dKind.MAX_POOL2D` paired with `MaxPool2dAttrs` and
  `Pool2dKind.AVERAGE_POOL2D` paired with `AveragePool2dAttrs`. Reject a kind/attributes mismatch,
  input/output count mismatch, or any other pooling kind or attribute representation.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling))</span>
- Revalidate exact input/output descriptors: one BFLOAT16, FLOAT32, or FLOAT64 rank-four input;
  one same-type, same-gradient-eligibility rank-four output; fully static extents; resolved layouts;
  non-negative storage offsets and strides; and an injective output layout.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [rank glossary](../../../../glossary.md#term-rank); [gradient glossary](../../../../glossary.md#term-gradient); [layouts glossary](../../../../glossary.md#layout); [injective glossary](../../../../glossary.md#term-injective))</span>
- For each [spatial axis](../../../../glossary.md#spatial-axes-channels-and-window-sampling), with input [extent](../../../../glossary.md#term-extent) `D`, [kernel](../../../../glossary.md#kernel)-position count `k`, symmetric padding per
  side <span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>`p`, [dilation](../../../../glossary.md#term-dilation) `d`, and [stride](../../../../glossary.md#term-stride) `s`, compute with checked `long` arithmetic:

  ```text
  effectiveKernel = d * (k - 1) + 1
  numerator       = D + 2 * p - effectiveKernel
  floor output    = floor(numerator / s) + 1
  ceil output     = ceil(numerator / s) + 1
  ```

- Reject a negative numerator, arithmetic [overflow](../../../../glossary.md#term-overflow), or any disagreement between the recomputed
  `[N,C,Hout,Wout]` and the supplied result Shape. Ceil mode is the literal symmetric padded
  grid: do not remove a terminal window whose start lies entirely in trailing padding.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `(n,c,oh,ow)` has input-window start
  `(oh * strideHeight - paddingHeight, ow * strideWidth - paddingWidth)`. Kernel coordinates are
  visited in increasing height and then increasing width order, and sample positions add the
  corresponding dilation. Use checked arithmetic for every derived extent, product, flattened
  count, byte span, address, range endpoint, and geometry array component.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- The CPU may reject a Model-valid occurrence whose exact kernel-position product, flattened
  output-cell count, address span, or invocation geometry cannot be represented safely in its
  checked <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`long`/carrier boundary. Such rejection is fail-closed capability truth, not a change
  to Model semantics.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

### Maximum-[pooling](../../../../glossary.md#term-pooling) numerical contract

- Padding positions do not participate in maximum selection. An output window with no in-bounds
  sampled coordinate stores exact negative infinity in the input/result type.<span data-glossary-navigation="true"> ([Padding glossary](../../../../glossary.md#padding); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- For a window containing at least one in-bounds sample, any NaN wins over every non-NaN and the
  first logical NaN wins among multiple NaNs. Otherwise ordinary numerical maximum applies,
  positive zero wins over negative zero, infinities use ordinary order, and equal candidates keep
  the first increasing kernel-height/kernel-width sample.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [NaN glossary](../../../../glossary.md#term-nan); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [kernel glossary](../../../../glossary.md#kernel))</span>
- Preserve the selected non-NaN represented value exactly. BFLOAT16 does not promote to an
  accumulator for max; compare its decoded represented value and retain the selected BFLOAT16
  bits for non-NaN values. NaN payload/sign and signaling preservation remain unspecified by
  Model, so tests must assert the selected NaN class and first-winner behavior without inventing a
  stronger portable payload guarantee.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [accumulator glossary](../../../../glossary.md#accumulator-type); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [Model glossary](../../../../glossary.md#term-model); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- The generated and direct-oracle implementations use the same one-pass first-winner state over
  increasing logical kernel coordinates. No indices output, hidden winner buffer, unpooling
  state, or workspace is created.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [buffer glossary](../../../../glossary.md#term-buffer))</span>

### Average-[pooling](../../../../glossary.md#term-pooling) numerical contract

- The divisor is always the positive mathematical product `kernelHeight * kernelWidth`. Dilation
  changes coordinates, not the number of divisor positions. Every in-bounds coordinate
  contributes its input value; every out-of-bounds coordinate contributes conceptual exact
  positive zero and still counts once in the divisor.<span data-glossary-navigation="true"> ([Dilation glossary](../../../../glossary.md#term-dilation); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- BFLOAT16 and FLOAT32 accumulate and divide in FLOAT32. FLOAT64 accumulates and divides in
  FLOAT64. BFLOAT16 narrows once after the final division. Perform one final division by the exact
  positive divisor; there is no valid-sample count, divisor override, integer accumulation,
  mixed-type promotion, or zero-divisor case.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- The generated scalar form and the optimal clean Java oracle use the same increasing
  kernel-height/kernel-width accumulation order. This deterministic CPU implementation choice is
  within Model's permitted reassociation boundary and does not claim bitwise equality with other
  backends.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [oracle glossary](../../../../glossary.md#term-test-oracle); [kernel glossary](../../../../glossary.md#kernel); [CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [reassociation glossary](../../../../glossary.md#term-reassociation); [backends glossary](../../../../glossary.md#backend))</span>
- Preserve the Model special-value classes: any in-bounds NaN produces NaN; opposing infinities
  produce NaN; otherwise a present infinity keeps its sign. Conceptual padding introduces no NaN
  or infinity. An exact-zero finite result is negative zero only when every divisor contribution
  is an in-bounds negative zero; cancellation, any positive zero, or any conceptual padding
  produces positive zero. An all-padding window stores exact positive zero.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [padding glossary](../../../../glossary.md#padding); [finite glossary](../../../../glossary.md#term-finite); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- Implement the signed-zero rule explicitly rather than assuming an incidental Java addition
  order will distinguish cancellation, conceptual padding, and an all-negative-zero domain.
  Preserve only the special-value and rounding guarantees selected by Model; do not promise NaN
  payload/sign preservation or cross-backend finite bit identity.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [rounding glossary](../../../../glossary.md#term-rounding); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

### Static [layouts](../../../../glossary.md#layout), carriers, zero extents, and [aliasing](../../../../glossary.md#term-aliasing)

- Support arbitrary resolved non-negative input and output strides that pass checked address-span
  validation. Dense layout is not a semantic prerequisite. A read-only input may be non-injective;
  the output must be injective because every logical output cell is written exactly once.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [validation glossary](../../../../glossary.md#term-validation); [injective glossary](../../../../glossary.md#term-injective))</span>
- Support the exact same-type [carrier](../../../../glossary.md#term-storage-carrier) combinations for each kind: [BFLOAT16](../../../../glossary.md#term-bfloat16) `short[]` or
  [`MemorySegment`](../../../../glossary.md#term-memorysegment), [FLOAT32](../../../../glossary.md#term-float32) `float[]` or `MemorySegment`, and [FLOAT64](../../../../glossary.md#term-float64) `double[]` or
  `MemorySegment`, including all-array, all-segment, and mixed input/output forms. Carrier objects,
  base addresses, storage offsets, extents, strides, and run identities remain cold invocation
  facts outside generated class identity.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Reject every physical output/input overlap before output mutation or worker submission. This
  task does not select an in-place exception, even for a one-by-one kernel, because one uniform
  disjoint-output contract is the bounded safe fallback. There is only one input and one output,
  so no input/input or output/output pair policy is needed.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>
- Static zero batch or channel extents produce zero output cells and submit no worker work. Static
  zero height or width remains admissible only when the checked padding/effective-kernel formula
  yields valid positive spatial output extents; those windows use the all-padding result policy.
  Valid geometry never creates a zero spatial output extent under the current Model formulas.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../../../glossary.md#term-channel); [padding glossary](../../../../glossary.md#padding); [spatial glossary](../../../../glossary.md#term-spatial-axes); [extent glossary](../../../../glossary.md#term-extent); [Model glossary](../../../../glossary.md#term-model))</span>
- Empty output performs no generated invocation, allocation, write, or validity mutation inside
  the CPU child. The surrounding existing Runtime contract continues to own partition-level
  validity transitions.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Runtime glossary](../../../../glossary.md#term-runtime); [partition glossary](../../../../glossary.md#partition))</span>

### [CPU](../../../../glossary.md#term-cpu)-private [lowering](../../../../glossary.md#lowering), generation, and execution

- Add one focused [immutable](../../../../glossary.md#term-immutable) CPU-private `CpuPool2dIr` carrying only code-shaping family, data type,
  scalar numerical form, and exact input/output access regimes. Add one <span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype))</span>`CpuPool2dLowering` that
  owns checked semantic geometry and range facts. Do not add a public type, shared lowering
  abstraction, pooling registry, generic window interpreter, or common <span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [registry glossary](../../../../glossary.md#term-registry))</span>`PoolNd` framework.
- Use one generated realization, `DIRECT_SCALAR`, for both kinds and all admitted cases. The same
  generated class body is reused by scalar and parallel-scalar orchestration for compatible
  code-shaping facts. No Vector API, tiling, unrolling, fixed-Shape class, window materialization,<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api); [tiling glossary](../../../../glossary.md#tiling); [unrolling glossary](../../../../glossary.md#term-unrolling); [Shape glossary](../../../../glossary.md#shape); [materialization glossary](../../../../glossary.md#materialization))</span>
  `UNFOLD2D` [decomposition](../../../../glossary.md#term-decomposition), or interpreted [scalar](../../../../glossary.md#term-scalar) [Runtime](../../../../glossary.md#term-runtime) fallback is selected in this task.
- The generated portable body is the production semantic baseline and safe fallback. The separate
  `CpuPool2dReferenceKernel` is a conformance and performance oracle only; Runtime must never call
  it to interpret an operation or IR.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [oracle glossary](../../../../glossary.md#term-test-oracle); [operation glossary](../../../../glossary.md#operation); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Thread the family through the existing CPU-private unit lowering, portable specialization,
  preparation plan, finalizer, generated Class-File dispatcher, executable geometry binding, and
  artifact cache. Use the established primitive geometry-array plus typed-carriers plus<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [binding glossary](../../../../glossary.md#term-binding); [artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache))</span>
  `start`/`end` entry boundary; add no new Runtime or shared [Prepare](../../../../glossary.md#prepare) contract.
- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 54 to 55. Pool2d class identity uses a new
  schema-55 projection containing every code-shaping pooling fact. Unchanged pre-0008F families
  retain their schema-52 class identity and byte-identical class bytes; MATMUL retains its
  schema-54 class identity and byte-identical class bytes. All compatibility envelopes advance to
  current-only schema 55, so schema-54 envelopes become safe misses.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [schema glossary](../../../../glossary.md#term-schema); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Generated classes remain final and field-free with one exact static entry. Their hot body may
  contain primitive index arithmetic, direct array/segment access, type-specific comparisons or
  accumulation, one average division, and a direct store only. It must contain no allocation,
  boxing, reflection, `Map` lookup, graph/value traversal, operation switch, route selection,
  cache access, layout construction, helper dispatch, or worker management.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [graph glossary](../../../../glossary.md#graph); [layout glossary](../../../../glossary.md#layout); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>

### [Partition](../../../../glossary.md#partition) [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag), representation, [publication](../../../../glossary.md#term-publication), and ranges

- The shared [immutable](../../../../glossary.md#term-immutable) [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) remains the sole structural source. Pooling becomes one
  atomic executable seed in CPU-owned decomposition. CPU does not reconstruct complete partition
  adjacency or retain another graph.<span data-glossary-navigation="true"> ([Pooling glossary](../../../../glossary.md#term-pooling); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [seed glossary](../../../../glossary.md#term-seed); [CPU glossary](../../../../glossary.md#term-cpu); [decomposition glossary](../../../../glossary.md#term-decomposition); [graph glossary](../../../../glossary.md#graph))</span>
- CPU 0008B's baseline decomposition and strict unit order remain authoritative. A pooling unit is
  a numerical-family barrier. Surrounding pointwise or affine work remains a separate valid unit
  with an ordinary declared boundary unless an already implemented rule independently proves a
  different topology.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [topology glossary](../../../../glossary.md#term-topology))</span>
- Pooling epilogue fusion is explicitly out of scope. CPU 0008C's closed recognizer does not
  include pooling, and its task explicitly deferred pooling execution to 0008G. Do not infer an
  ADD, activation, CLAMP, affine view, decomposed reduction, or another suffix around Pool2d.
  CPU 0008D therefore receives no new pooling fused/split alternative or profitability fact.<span data-glossary-navigation="true"> ([epilogue glossary](../../../../glossary.md#term-epilogue); [fusion glossary](../../../../glossary.md#term-fusion); [scope glossary](../../../../glossary.md#term-scope); [activation glossary](../../../../glossary.md#term-activation); [view glossary](../../../../glossary.md#term-view-meanings); [fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
- Select direct input/output representation as the only ordinary Pool2d plan. Do not broaden CPU
  0008E's pointwise materialization grammar, promote a materialized input, or introduce a hidden
  contiguous copy. Existing materialized pointwise variants remain complete candidate-only facts
  and ordinary preparation remains direct. A future Pool2d materialization candidate requires a
  separately planned complete candidate, resource, and end-to-end evidence boundary.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Declare exactly the input read and output write buffer requirements before shared slot
  assignment. Declare no workspace, partial, index, count, winner, column, combine, padding, or
  scratch resource. Finalization may only generate/load the already selected Pool2d artifact and
  bind the already assigned slots.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [padding glossary](../../../../glossary.md#padding); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Flatten the logical output-cell domain in stable NCHW order. Each half-open generated range owns
  complete output cells and every cell's complete kernel window. Reuse existing caller-owned<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [half-open glossary](../../../../glossary.md#term-half-open-interval); [kernel glossary](../../../../glossary.md#kernel))</span>
  [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup), configured range-count, and minimum-work policy; do not add a Pool2d threshold
  constant. Parallel execution partitions only output cells, never kernel coordinates, and has no
  partial reduction, combine phase, atomics, locks, nested workers, or worker-count-dependent
  arithmetic.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Runtime receives only immutable prepared geometry, typed buffer/workspace selections, and the
  generated handle. It performs no graph interpretation, allocation policy, geometry derivation,
  alias decision, route search, materialization decision, or pooling dispatch.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [alias glossary](../../../../glossary.md#aliasing-and-arena); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>

### Performance and generated-code evidence

- Every generated pooling specialization must follow the semantic algorithm, hot-loop/dataflow
  shape, carrier/layout case, and avoidable-overhead profile of an optimal clean Java
  implementation at the same Shape-polymorphic specialization boundary. The oracle receives the
  same primitive geometry, typed carriers, and half-open output-cell range.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [specialization glossary](../../../../glossary.md#term-specialization); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [oracle glossary](../../../../glossary.md#term-test-oracle); [half-open glossary](../../../../glossary.md#term-half-open-interval))</span>
- Semantic validation precedes timing. Retain a bounded evidence ledger that covers both kinds,
  all three types, dense and general positive-stride layouts, padding, dilation, floor and literal
  ceil grids, heap arrays, segments, mixed carriers, all-padding windows, and scalar/parallel
  range subdivision. Special-value and zero-domain cases are semantic evidence; do not time tiny
  exceptional cases as performance representatives.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation); [heap glossary](../../../../glossary.md#term-heap); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Time at least these six hot representatives in five fresh isolated Java 26 forks: dense
  FLOAT64 max 3-by-3; padded/dilated FLOAT32 max with a ceil tail; mixed-carrier BFLOAT16 max;
  dense FLOAT64 average 3-by-3; padded/dilated FLOAT32 average with a ceil tail; and mixed-carrier
  BFLOAT16 average. If a general-layout code shape is distinct from those rows, add one
  representative for each affected kind without changing production selection policy.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Use the established CPU evidence protocol: fixed heap, deterministic inputs, randomized
  generated/direct order, at least five warmup batches, nine measured rounds, adaptive batches of
  at least 25 ms, exact pre/post semantic verification, no retry or discarded measured miss, and
  generated/direct ratio <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [warmup glossary](../../../../glossary.md#term-warmup); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>`<= 1.15x` for every accepted fork and each row's aggregate median. This
  existing repository gate is reused; no new performance threshold or production constant is
  introduced.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [median glossary](../../../../glossary.md#term-median); [constant glossary](../../../../glossary.md#term-constant))</span>
- Retain generated bytes and digest, specialization/lowering manifests, entry descriptor,
  Class-File parse, complete <span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest); [lowering glossary](../../../../glossary.md#lowering); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>`javap -c -v -p`, constant-pool/member report, forbidden-reference
  scan, semantic checksum, environment, direct-oracle identity/source, fork reports, aggregate,
  and a checksummed evidence manifest outside the repository.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [checksum glossary](../../../../glossary.md#term-checksum); [identity glossary](../../../../glossary.md#term-identity-meanings); [manifest glossary](../../../../glossary.md#term-manifest))</span>
- Retain byte-identical generated controls for representative schema-52 pointwise, affine-copy,
  reduction, Conv2d, and Conv3d forms and one schema-54 MATMUL form. Compatibility changes to 55;
  their class projections, binary names, emitted bodies, and class bytes do not.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [pointwise glossary](../../../../glossary.md#term-pointwise); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>

### Documentation and [Javadoc](../../../../glossary.md#term-javadoc)

- After executable Java and generated evidence stabilize, use a separate clean
  documentation-focused agent/thread. It receives this task, the actual diff, exact test and
  evidence results, affected APIs/behavior, architecture constraints, expected documentation,
  and validation commands.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Finalize meaningful Javadoc for every changed CPU type and member, including NCHW coordinates,
  both numerical policies, geometry, carrier/layout/alias rules, range ownership, resource and
  lifecycle boundaries, nullability, results, and expected failures.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [NCHW glossary](../../../../glossary.md#term-nchw); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [alias glossary](../../../../glossary.md#aliasing-and-arena); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Update `docs/backend-guide/cpu-backend.md` with the implemented direct portable Pool2d boundary,
  max/average semantic distinction, static/layout/carrier limits, range/resource model, safe
  fallback, and no-fusion/no-materialization boundary. Review <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization))</span>`docs/glossary.md`; change it only
  if implementation changes a reusable term, otherwise record why its existing pooling entry is
  accurate.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling))</span>
- Synchronize this task, the CPU master plan, and roadmap only after all executable,
  performance, documentation, status, and scope gates pass. The documentation context reuses
  successful Java tests unless it changes executable Java or records a concrete stale-evidence
  risk.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Native/OpenBLAS, Accelerate, oneMKL, oneDNN, AOCL, ZenDNN, JNI, or Foreign Function and Memory
  downcall pooling routes.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [oneMKL glossary](../../../../glossary.md#term-onemkl); [oneDNN glossary](../../../../glossary.md#term-onednn); [AOCL glossary](../../../../glossary.md#term-aocl); [ZenDNN glossary](../../../../glossary.md#term-zendnn); [JNI glossary](../../../../glossary.md#term-jni); [Foreign Function and Memory glossary](../../../../glossary.md#application-binary-interface-and-native-bridge); [downcall glossary](../../../../glossary.md#term-downcall))</span>
- Backward/autograd implementation, saved max indices, hidden outputs, unpooling, gradient
  formulas, training orchestration, or changes to Compiler 0005D.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [gradient glossary](../../../../glossary.md#term-gradient); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Dynamic or binding-dependent Shapes, unresolved layouts, negative storage strides, runtime
  geometry, sparse, quantized, complex, FLOAT16, integral, BOOL, or mixed-type pooling.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [runtime glossary](../../../../glossary.md#term-runtime); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Global, adaptive, one-dimensional, three-dimensional, channels-last, asymmetric, automatic, or
  transposed pooling; configurable count-padding, valid-sample average, divisor override, or
  padding value.<span data-glossary-navigation="true"> ([channels-last glossary](../../../../glossary.md#spatial-axes-channels-and-window-sampling); [padding glossary](../../../../glossary.md#padding); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- [Pooling](../../../../glossary.md#term-pooling) [epilogue](../../../../glossary.md#term-epilogue) recognition/[fusion](../../../../glossary.md#term-fusion), decomposed pooling recognition, [graph](../../../../glossary.md#graph) rewriting,
  `UNFOLD2D` materialization, column buffers, winner/index buffers, input materialization,
  vectorization, tiling, unrolling, or fixed-Shape specialization.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [buffers glossary](../../../../glossary.md#term-buffer); [tiling glossary](../../../../glossary.md#tiling); [unrolling glossary](../../../../glossary.md#term-unrolling); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- New public/shared APIs or abstractions; changes to Model, Compiler, Planning, Prepare, Runtime,
  Config, Trace, Backend Contract, OpenBLAS provider, Engine, architecture, ADRs, dependencies,
  Gradle, architecture tests, backend conformance, or integration tests.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [Backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Global autotuning policy, tuning-cache lookup/mutation, measurement-driven preparation,
  Runtime profiling selection, or any CPU 0010-plus native/tuning work.<span data-glossary-navigation="true"> ([autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup); [Runtime profiling glossary](../../../../glossary.md#runtime-profiling))</span>
- Attention, [loss](../../../../glossary.md#loss), or any [CPU](../../../../glossary.md#term-cpu) 0008H-or-later task specification or implementation.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [CPU backend master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
- [CPU 0008B partition-DAG decomposition](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- [CPU 0008C typed recognition](0008c-typed-specialized-subgraph-and-epilogue-recognition.md)
- [CPU 0008D fusion profitability](0008d-bounded-fusion-profitability-and-typed-decision-facts.md)<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion))</span>
- [CPU 0008E representation candidates](0008e-bounded-multi-input-materialization-and-representation-reuse.md)
- [CPU 0008E1 shared DAG adoption](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md)
- [CPU 0008F portable MATMUL](0008f-portable-matmul-execution-and-bounded-linear-epilogues.md)<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- [Model 0020A max Pool2d](../../../modules/model/tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model 0020A1 average Pool2d](../../../modules/model/tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)
- [Compiler 0005D pooling gradients](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [pooling glossary](../../../../glossary.md#term-pooling); [gradients glossary](../../../../glossary.md#term-gradient))</span>

## Architecture constraints

- `ARCHITECTURE.md` remains authoritative. Planning selects CPU ownership; CPU analysis owns
  exact lowering, route/representation selection, and complete resource declaration; shared
  Prepare assigns slots; CPU finalization constructs the executable; Runtime invokes prepared
  work only.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lowering glossary](../../../../glossary.md#lowering); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- The shared [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) is the sole complete partition-structure source. CPU-private Pool2d
  geometry and IR retain only exact unit-local semantic and code-shaping facts.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Generated JVM bytecode is the portable production route and truthful fallback. Reference Java
  is an oracle and never a Runtime interpreter.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [bytecode glossary](../../../../glossary.md#term-bytecode); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- All semantic validation, geometry, layout, carrier, address-span, injectivity, alias, range,
  resource, schema, and route decisions are cold. Generated hot code receives primitive geometry
  and direct typed carriers only.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [alias glossary](../../../../glossary.md#aliasing-and-arena); [schema glossary](../../../../glossary.md#term-schema))</span>
- Each generated specialization preserves the optimal clean Java algorithm and avoidable-overhead
  profile at the same specialization boundary. A general fallback does not excuse avoidable
  overhead in a proved case.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- CPU 0008B–0008E1 ownership remains unchanged: shared DAG structure; CPU-owned unit formation,
  recognition, profitability, and representation candidates; candidate-only materialization;
  strict inter-unit execution; and no Runtime topology or policy interpretation.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [materialization glossary](../../../../glossary.md#materialization); [topology glossary](../../../../glossary.md#term-topology))</span>
- Capability advertisement becomes true only after the complete admitted matrix is implemented
  and validated. Any unsupported, dynamic, unresolved, unsafe, overflowed, or mismatched
  occurrence remains fail-closed.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — sole public capability provider.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — new [immutable](../../../../glossary.md#term-immutable) Pool2d code-shaping facts.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — checked Pool2d semantic geometry and
  direct scalar selection.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct generated Pool2d body.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — optimal clean Java [oracle](../../../../glossary.md#term-test-oracle).
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema) and [specialization](../../../../glossary.md#term-specialization) [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — resource declaration, assignment
  handoff, and finalization facts.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold typed geometry binding and
  prepared range invocation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — generated portable plan.

Packages added or changed:

- No package is added. Existing [CPU](../../../../glossary.md#term-cpu)-private packages gain focused Pool2d leaves only.

Type placement:

- `...internal.ir.CpuPool2dIr` — immutable Pool2d family/type/access class identity.
- `...internal.lowering.CpuPool2dLowering` — descriptor validation, exact NCHW geometry, range,
  and direct scalar plan owner.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [NCHW glossary](../../../../glossary.md#term-nchw); [owner glossary](../../../../glossary.md#term-owner))</span>
- `...internal.codegen.emit.CpuPool2dEmitter` — generated direct [scalar](../../../../glossary.md#term-scalar) max/average body.
- `...internal.reference.CpuPool2dReferenceKernel` — same-boundary semantic and performance
  oracle, never production dispatch.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Existing preparation, cache, generator, executable, and capability owners carry only the
  minimal new family branch/facts required by those responsibilities.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [carry glossary](../../../../glossary.md#term-carry))</span>

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production [owners](../../../../glossary.md#term-owner):

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPool2dEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPool2dIr.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPool2dLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionDagDecomposer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuPool2dReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`

Expected focused test owners:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuBatchNormTrainingEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv2dEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv3dEvidenceTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPool2dGeneratedKernelTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPool2dPerformanceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPartitionDagGeneratedEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseLedgerEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPool2dIrTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPool2dLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuPool2dReferenceTest.java`

If implementation proves a named existing suite is not the current [owner](../../../../glossary.md#term-owner), substitute one matching
CPU test path and record the source-backed reason without increasing the maximum [scope](../../../../glossary.md#term-scope).

The five existing evidence owners added above are mechanical [schema](../../../../glossary.md#term-schema) invalidation controls only:
their stale assertions must advance from current schema 54 to current schema 55 after Pool2d
changes the compatibility envelope. They do not widen Pool2d behavioral scope, add evidence rows,
or authorize changes to their underlying completed families. The focused
`CpuClassFileKernelGeneratorTest`, `CpuShapePolymorphicArtifactTest`, and
`CpuPartitionDagDecomposerTest` suites remain required [validation](../../../../glossary.md#term-validation) owners below but are not
expected changed paths.

The targeted documentation pass may update affected CPU package/type Javadocs,
`docs/backend-guide/cpu-backend.md`, `docs/glossary.md` only if a reusable term changes, this task,
the CPU master plan, and the roadmap. No other documentation path is authorized.

## Maximum [scope](../../../../glossary.md#term-scope)

The implementation may create or modify at most 38 repository paths:

- exactly 16 [CPU](../../../../glossary.md#term-cpu) production paths, including the [sealed](../../../../glossary.md#term-sealed) `CpuPortableKernelIr` [owner](../../../../glossary.md#term-owner) required for
  `CpuPool2dIr` to join its hierarchy;
- exactly 17 CPU test paths, comprising the 12 Pool2d implementation/evidence paths and the five
  mechanical schema-55 invalidation controls named above; and
- exactly five possible documentation/planning paths: CPU backend guide, conditional glossary,
  this task, CPU master plan, and roadmap.

The ceiling is justified by one cohesive generated family crossing the existing capability, [IR](../../../../glossary.md#term-intermediate-representation-ir),
[lowering](../../../../glossary.md#lowering), preparation, [schema](../../../../glossary.md#term-schema), generation, execution, [oracle](../../../../glossary.md#term-test-oracle), and evidence seams. It is a ceiling,
not a [target](../../../../glossary.md#term-target). No [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), Config, [Trace](../../../../glossary.md#trace), [Backend](../../../../glossary.md#backend) Contract,
[OpenBLAS](../../../../glossary.md#term-openblas) provider, [Engine](../../../../glossary.md#term-engine), architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), architecture-test, backend-[conformance](../../../../glossary.md#term-conformance),
integration-test, resource, generated-source, or build path may change. If a production path
beyond the explicitly inventoried 16, an eighteenth test path, a sixth documentation path, or any
shared/public change is necessary, stop and revise the plan before editing.

## Acceptance criteria

- [CPU](../../../../glossary.md#term-cpu) capability admits all and only fully static, resolved-[layout](../../../../glossary.md#layout), supported-[carrier](../../../../glossary.md#term-storage-carrier) exact
  `MAX_POOL2D`/`MaxPool2dAttrs` and `AVERAGE_POOL2D`/`AveragePool2dAttrs` occurrences across
  BFLOAT16, FLOAT32, and FLOAT64 with exact one-input/one-output descriptor relationships.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Checked cold lowering reproduces the literal NCHW floor/ceil Shape, exact starts and dilated
  coordinates, and rejects kind/attrs, rank, type, Shape, gradient, layout, span, injectivity,
  overflow, and unsupported-carrier mismatches before artifact access or writes.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [NCHW glossary](../../../../glossary.md#term-nchw); [Shape glossary](../../../../glossary.md#shape); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [rank glossary](../../../../glossary.md#term-rank); [gradient glossary](../../../../glossary.md#term-gradient); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [overflow glossary](../../../../glossary.md#term-overflow); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Generated max pooling preserves excluded padding, all-padding negative infinity, first logical
  NaN/equal winner, positive-over-negative zero, infinity order, and original selected BFLOAT16
  bits across every admitted layout/carrier/range form.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [padding glossary](../../../../glossary.md#padding); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- Generated average pooling preserves fixed count-padding, FLOAT32 accumulation/division for
  BFLOAT16/FLOAT32, FLOAT64 accumulation/division for FLOAT64, one final division and BFLOAT16
  narrowing, NaN/opposing/single-infinity classes, exact signed-zero rule, and all-padding
  positive zero.<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Zero N/C domains perform zero work; zero spatial input with valid padded geometry produces exact
  all-padding outputs. Output is injective and completely disjoint from input before any write or
  worker submission. All-array, all-segment, and mixed carriers pass.<span data-glossary-navigation="true"> ([spatial glossary](../../../../glossary.md#term-spatial-axes); [injective glossary](../../../../glossary.md#term-injective))</span>
- Exactly one generated `DIRECT_SCALAR` form is the production fallback for every admitted case.
  Scalar and parallel-scalar execution share its class and are equal across range subdivisions and
  worker counts. Each range owns complete disjoint output cells and complete windows; there is no
  partial/combine state, workspace, allocation, atomics, locks, or nested parallelism.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar))</span>
- Pooling is one atomic shared-DAG-derived CPU unit. No pooling epilogue fact, fusion candidate,
  materialization candidate/promotion, <span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [epilogue glossary](../../../../glossary.md#term-epilogue); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization))</span>`UNFOLD2D` decomposition, graph reconstruction, or Runtime
  interpretation is introduced. Surrounding work retains the exact valid CPU 0008B split.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Preparation declares only the input and output buffers before assignment. Finalization adds no
  resources or policy and realizes only the selected artifact. Runtime receives cold-bound typed
  carriers, primitive geometry, and ranges.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer))</span>
- Schema advances exactly to 55. Stale schema-54 envelopes fail safely; Pool2d identity includes
  every code-shaping fact; unchanged schema-52 families and schema-54 MATMUL retain exact binary
  names and byte-identical class bytes.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Raw Class-File parsing and complete `javap -c -v -p` inspection prove one final field-free class
  and static entry per selected artifact, expected direct primitive operations, and no forbidden
  helper, allocation, boxing, reflection, map, cache, graph, route, layout-construction, or worker
  reference.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Generated and optimal clean Java implementations use the same Shape-polymorphic specialized
  algorithm, accumulation/selection order, carrier/layout case, geometry, and range boundary.
  Every required performance fork and aggregate passes the established <span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>`<= 1.15x` gate.
- Capability, preparation, schema, execution, semantic, generated, reference, performance,
  Javadoc, Markdown, exact-scope, status, and whitespace validation pass. No Java test suite is
  repeated by the documentation context absent executable change or a recorded concrete risk.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [validation glossary](../../../../glossary.md#term-validation))</span>
- No public API, shared module, dependency, architecture rule, Gradle configuration, native route,
  conformance/integration boundary, attention/loss behavior, or CPU 0008H-or-later task file
  changes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [loss glossary](../../../../glossary.md#loss))</span>
- A separate documentation-focused context finalizes affected Javadocs, CPU guide, glossary
  impact, task evidence/status, master plan, and roadmap before this task becomes `Complete`.

## Tests / [validation](../../../../glossary.md#term-validation)

### Tier 1: focused implementation [validation](../../../../glossary.md#term-validation)

After executable Java stabilizes, [run](../../../../glossary.md#term-run) this exact focused command once:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool2dIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionDagDecomposerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGeneratorTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPool2dGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuShapePolymorphicArtifactTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuPool2dReferenceTest
```

The focused suites must cover both kinds and attribute pairings; exact floor/ceil and all-[padding](../../../../glossary.md#padding)
geometry; all types; special values and [signed zero](../../../../glossary.md#term-signed-zero); zero extents; arbitrary positive [strides](../../../../glossary.md#term-stride);
arrays, segments, and mixed carriers; overlap/[injectivity](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) failures before writes; [scalar](../../../../glossary.md#term-scalar)/parallel
range [ownership](../../../../glossary.md#term-ownership-meanings); [schema](../../../../glossary.md#term-schema)/[specialization](../../../../glossary.md#term-specialization)/[cache](../../../../glossary.md#term-cache)/finalizer propagation; exact [descriptors](../../../../glossary.md#term-descriptor); and
unchanged split/recognition/[materialization](../../../../glossary.md#materialization) behavior.

### Tier 2: generated-code and performance evidence

[Run](../../../../glossary.md#term-run) semantic/Class-File checks through the Tier-1 generated suite. Then create one fresh absolute
evidence directory and run the opt-in five-[fork](../../../../glossary.md#term-fork-meanings) evidence [owner](../../../../glossary.md#term-owner) exactly once:

```bash
SYNAPTIK_CPU_POOL2D_PERFORMANCE=true \
SYNAPTIK_CPU_POOL2D_EVIDENCE_ROOT=<fresh-absolute-evidence-directory> \
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPool2dPerformanceTest \
  --rerun-tasks
```

`CpuPool2dPerformanceTest` must launch and retain exactly five fresh fixed-[heap](../../../../glossary.md#term-heap) Java 26 subprocess
[forks](../../../../glossary.md#term-jvm-fork), aggregate all measured rows, reject only a whole pre-measurement environmental/control
failure with a recorded reason, and fail the [Gradle](../../../../glossary.md#term-gradle) command on any semantic, structural,
per-fork, aggregate, inventory, or [manifest](../../../../glossary.md#term-manifest) failure. The full [CPU](../../../../glossary.md#term-cpu) suite runs this owner in a
non-measuring assumption/skip mode when the opt-in environment variable is absent.

Inspect every retained generated class with the [JDK](../../../../glossary.md#term-java-development-kit-jdk) [Class-File API](../../../../glossary.md#term-class-file-api) and:

```bash
javap -c -v -p <each-retained-generated-pool2d-class>
```

Record the exact evidence directory, manifest [digest](../../../../glossary.md#term-digest), row inventory, environment, all fork and
aggregate ratios, rejected pre-measurement forks if any, and zero forbidden-reference findings in
this task before completion. Do not rerun, replace, or relabel a measured failure.

### Tier 3: [CPU](../../../../glossary.md#term-cpu) capability [checkpoint](../../../../glossary.md#term-checkpoint)

After focused and performance evidence stabilizes, [run](../../../../glossary.md#term-run) the complete CPU [module](../../../../glossary.md#term-module) suite exactly once:

```bash
./gradlew :backends:cpu:test
```

This module checkpoint is required because capability, family [lowering](../../../../glossary.md#lowering), preparation, [schema](../../../../glossary.md#term-schema),
generated execution, [cache](../../../../glossary.md#term-cache) compatibility, [carrier](../../../../glossary.md#term-storage-carrier) [binding](../../../../glossary.md#term-binding), and parallel ranges change together.
The checkpoint must also exercise the five inventoried evidence controls whose only authorized
edit is advancing their stale current-schema assertion from 54 to 55; those mechanical edits do
not add Pool2d cases or reopen the completed [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization), [Conv2d](../../../../glossary.md#convolution--conv2d), [Conv3d](../../../../glossary.md#conv3d--ncdhw), [partition](../../../../glossary.md#partition)-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag),
or [pointwise](../../../../glossary.md#term-pointwise) evidence scopes.
Do not run repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration, [Model](../../../../glossary.md#term-model), or [Compiler](../../../../glossary.md#term-compiler)
test suites unless implementation discovers an actual shared/boundary change. The Model and
Compiler contracts are read-only prerequisites and their completed evidence is reused.

### Tier 4: documentation and repository hygiene

The separate documentation-focused context runs after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates rendered affected Javadoc pages; local Markdown links and heading anchors;
balanced fences; terminology; final newlines and trailing whitespace; exact package/type
placement; [schema](../../../../glossary.md#term-schema) 55 and preserved 52/54 projections; exact changed-path inventory within the
38-path ceiling with exactly 16 production and 17 test paths; no generated/evidence leakage; and
synchronized 0008F Complete, 0008G Complete, 0008H Draft status. The implementation retained
0008G [Ready](../../../../glossary.md#term-ready) and 0008H Draft until these gates passed; 0008H remains Draft.

Repository-wide [validation](../../../../glossary.md#term-validation) is deferred to [CPU](../../../../glossary.md#term-cpu) 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes one [concrete
backend](../../../../glossary.md#concrete-backend) [module](../../../../glossary.md#term-module) without changing dependencies, architecture, shared build configuration, or a
shared/public contract.

## Dependencies

- [CPU 0008F](0008f-portable-matmul-execution-and-bounded-linear-epilogues.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> — Complete.
- [CPU 0008E1](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md) — Complete.
- [CPU 0008E](0008e-bounded-multi-input-materialization-and-representation-reuse.md) — Complete.
- [CPU 0008D](0008d-bounded-fusion-profitability-and-typed-decision-facts.md) — Complete.
- [CPU 0008C](0008c-typed-specialized-subgraph-and-epilogue-recognition.md) — Complete.
- [CPU 0008B](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md) — Complete.
- [Model 0020A](../../../modules/model/tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span> — Complete.
- [Model 0020A1](../../../modules/model/tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md) — Complete.
- [Compiler 0005D](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span> — Complete.

## Follow-up tasks

- CPU 0008H remains Draft and owns portable scaled-dot-product attention execution. Do not create
  its detailed specification or implement it in this task.
- [CPU](../../../../glossary.md#term-cpu) 0008I remains Draft and owns portable [loss](../../../../glossary.md#loss)-family execution.
- CPU 0009 remains Draft and owns the portable generated-coverage closure checkpoint after the
  ordered 0008H and 0008I families complete.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Any vectorized/tiled Pool2d realization, Pool2d input materialization, native DNN route,
  adaptive/global pooling, or pooling epilogue fusion requires a separately planned task with its
  own eligibility, resource, numerical, and performance evidence.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [DNN glossary](../../../../glossary.md#term-dnn); [pooling glossary](../../../../glossary.md#term-pooling); [epilogue glossary](../../../../glossary.md#term-epilogue); [fusion glossary](../../../../glossary.md#term-fusion))</span>

## Architecture impact

Expected impact: None.

This task adds one [CPU](../../../../glossary.md#term-cpu)-private portable family within the existing concrete-[backend ownership](../../../../glossary.md#backend-ownership) of
[lowering](../../../../glossary.md#lowering), route selection, exact declarations, finalization, generation, and execution. It uses
the existing shared [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag), [Prepare](../../../../glossary.md#prepare) slot assignment, [Runtime](../../../../glossary.md#term-runtime) prepared-executable boundary,
and [dependency direction](../../../../glossary.md#java-platform-and-build-tools). It changes no architecture rule or [dependency](../../../../glossary.md#term-dependency), so no
`ARCHITECTURE.md`, focused architecture page, [architecture decision record](../../../../glossary.md#term-architecture-decision-record-adr) ([ADR](../../../../glossary.md#adr-ci-and-qualification)), or [architecture
test](../../../../glossary.md#term-architecture-test) update is required. If implementation proves any shared/public contract or architecture
change necessary, stop and report the conflict before editing outside this task.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are implementing Synaptik CPU task 0008G in a separate clean context.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md, the CPU master plan, completed CPU
tasks 0008B-0008F, Model tasks 0020A-0020A1, Compiler task 0005D, documentation rules/profiles,
this task file, and every directly affected source/test owner before editing.

Implement this task exactly as specified. Preserve the shared PartitionDag and CPU 0008B-0008E1
ownership boundaries. Add only direct generated scalar/parallel-scalar max and fixed-count average
Pool2d execution, schema 55 identity, checked static NCHW geometry, direct carriers, disjoint
output-cell ranges, and the same-boundary optimal clean Java oracle. Do not add fusion,
materialization, vector/tiled/native routes, backward work, dynamic geometry, shared/public APIs,
or CPU 0008H-or-later planning.

Run the exact focused, five-fork evidence, full CPU checkpoint, and scope checks. After executable
behavior stabilizes, hand the actual diff and exact Java/evidence results to a distinct clean
documentation-focused context. That context must finalize affected Javadocs, CPU guide, glossary
impact, planning evidence/status, and documentation validation in the same overall change without
repeating successful Java tests unless executable Java changes or a concrete risk is recorded.

Do not stage, commit, or push. Stop on architecture/shared-module need, scope overflow, or a
failed fixed evidence gate rather than widening scope or changing policy silently. Do not mark
0008G Complete until every acceptance, evidence, documentation, exact-path, and whitespace gate
passes; keep 0008H and later Draft.
```

## Local decisions

- Pooling is one cohesive CPU family because max and average share exact NCHW geometry, carriers,
  resources, output-cell ownership, and generated entry shape while retaining separate family-
  specific numerical bodies.<span data-glossary-navigation="true"> ([Pooling glossary](../../../../glossary.md#term-pooling); [CPU glossary](../../../../glossary.md#term-cpu); [NCHW glossary](../../../../glossary.md#term-nchw); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [shape glossary](../../../../glossary.md#shape))</span>
- The first portable implementation has exactly one direct scalar code shape. Parallelism is
  outer output-cell orchestration using existing workers; no vector or tile realization is
  justified by current source/evidence before implementation.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector))</span>
- Increasing kernel-height/kernel-width traversal is selected for generated/reference parity and
  deterministic CPU results. This is compatible with average pooling's permitted reassociation
  and max pooling's required first-winner order.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Direct representations are complete and selected. Pool2d does not broaden the candidate-only
  materialization program established by CPU 0008E.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- Pooling epilogue fusion is not in scope because CPU 0008C explicitly excludes pooling
  recognition and current code has no pooling fact or executable form to consume.<span data-glossary-navigation="true"> ([epilogue glossary](../../../../glossary.md#term-epilogue); [fusion glossary](../../../../glossary.md#term-fusion); [scope glossary](../../../../glossary.md#term-scope))</span>
- Schema 55 is required because Pool2d adds a new emitted family and code-shaping identity.
  Existing schema-52 and MATMUL schema-54 class projections remain stable.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>

## Known limitations

- Only fully static, resolved-layout NCHW BFLOAT16/FLOAT32/FLOAT64 Pool2d executes. Model symbolic
  Shape construction remains valid but is outside this CPU execution boundary.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [NCHW glossary](../../../../glossary.md#term-nchw); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [Model glossary](../../../../glossary.md#term-model); [Shape glossary](../../../../glossary.md#shape); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- The first route is scalar or parallel-scalar only. General positive-stride layouts and every
  array/segment mixture are correctness fallbacks, not broad performance claims.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [stride glossary](../../../../glossary.md#term-stride))</span>
- Average finite non-zero results are deterministic for this selected CPU traversal but are not a
  promise of bitwise equality across backends because Model permits reassociation.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [backends glossary](../../../../glossary.md#backend); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- No pooling fusion, input materialization, native route, saved max indices, or backward execution
  is added.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context passed the focused Pool2d and affected [CPU](../../../../glossary.md#term-cpu) tests, the [schema](../../../../glossary.md#term-schema)-54
[regression](../../../../glossary.md#term-regression) controls, generated structural scans, and the one required full CPU [checkpoint](../../../../glossary.md#term-checkpoint): 596
tests, zero failures, zero errors, and four expected skips. No successful Java test suite was
repeated by the documentation context.

The retained evidence root is
`/private/tmp/synaptik-cpu-0008g-evidence-20260830-001`. Its [checksum](../../../../glossary.md#term-checksum) [manifest](../../../../glossary.md#term-manifest) validates 32 listed
[artifacts](../../../../glossary.md#term-artifact), and `manifest.digest` validates the manifest content, for 33 checksum-valid evidence
entries. It contains five successful [fork](../../../../glossary.md#term-fork-meanings) CSVs, six aggregate rows, no `failed-forks` directory,
the byte-identical direct-[oracle](../../../../glossary.md#term-test-oracle) source, generated classes, `javap` output, member scans,
specializations, and environment facts. The six generated/direct aggregate medians are
`0.998144202`, `1.010875034`, `1.051055579`, `1.074063542`, `1.106718493`, and `0.931805101`; all
passed the fixed `<= 1.15x` gate.

Clean documentation context `01a05161-2d20-7b70-9168-83269bf208b1` independently reconciled the
final implementation and tests
with the architecture, [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation) contracts, CPU guide, glossary, and affected Javadocs. The
final `:backends:cpu:javadoc` task passed with only the two expected incubating-[module](../../../../glossary.md#term-module) warnings.
Local Markdown links and anchors and all four changed Markdown fence counts passed. Exact [scope](../../../../glossary.md#term-scope) is
37 paths: 16 production/[Javadoc](../../../../glossary.md#term-javadoc), 17 tests/evidence, and four documentation/[planning](../../../../glossary.md#term-planning) paths. The
`git diff --check` validation passed, and no retained evidence artifact is present in the
repository.

## Implementation notes

- `CpuPool2dLowering` revalidates exact static NCHW descriptors, literal floor/ceil geometry,
  checked layouts, output injectivity, fixed divisor, and flattened output-cell count. It threads
  immutable geometry through the existing unit preparation/finalization/executable boundary with
  exactly one input, one output, and zero workspace.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [descriptors glossary](../../../../glossary.md#term-descriptor); [layouts glossary](../../../../glossary.md#layout); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- `CpuPool2dIr` and `CpuPool2dEmitter` add one `DIRECT_SCALAR` code shape for MAX and AVERAGE over
  BFLOAT16, FLOAT32, and FLOAT64 typed arrays, segments, and mixed carriers. Scalar and
  parallel-scalar orchestration invoke the same generated body over disjoint half-open ranges of
  complete output cells.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [Scalar glossary](../../../../glossary.md#term-scalar); [half-open glossary](../../../../glossary.md#term-half-open-interval))</span>
- The generated max and average loops implement the Model-owned padding, first-winner, signed-zero,
  accumulator, division, and BFLOAT16 narrowing rules without window materialization, hot helper
  dispatch, partial results, or a combine step. Cold executable validation rejects overlap before
  any write or worker submission and avoids invocation for empty output.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [padding glossary](../../../../glossary.md#padding); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [accumulator glossary](../../../../glossary.md#accumulator-type); [narrowing glossary](../../../../glossary.md#term-narrowing); [materialization glossary](../../../../glossary.md#materialization); [dispatch glossary](../../../../glossary.md#term-dispatch); [validation glossary](../../../../glossary.md#term-validation))</span>
- Schema 55 owns the Pool2d projection. Existing schema-52 family bytes and MATMUL schema-54 bytes
  remain stable, while older compatibility envelopes are safe misses.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- `CpuPool2dReferenceKernel` is the optimal clean Java test/performance oracle only; Runtime never
  calls it as an interpreter or fallback. No planned deviation, fusion, materialization, vector,
  native, backward, public/shared, dependency, or architecture change was required.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [Runtime glossary](../../../../glossary.md#term-runtime); [fusion glossary](../../../../glossary.md#term-fusion); [vector glossary](../../../../glossary.md#term-vector); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

Completed portable generated MAX_POOL2D and fixed-divisor AVERAGE_POOL2D execution for the full
admitted static [NCHW](../../../../glossary.md#term-nchw) [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) boundary. The change includes checked [lowering](../../../../glossary.md#lowering) and
geometry, [schema](../../../../glossary.md#term-schema)-55 [identity](../../../../glossary.md#term-identity-meanings), [scalar](../../../../glossary.md#term-scalar)/parallel-scalar typed generated loops, cold [carrier](../../../../glossary.md#term-storage-carrier)/[alias](../../../../glossary.md#aliasing-and-arena)
[validation](../../../../glossary.md#term-validation), an optimal clean Java [oracle](../../../../glossary.md#term-test-oracle), semantic/structural/performance coverage, affected
Javadocs, the [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) guide, and synchronized task/master/roadmap evidence.

Validation passed at the task's focused, full CPU, structural, schema, performance, [Javadoc](../../../../glossary.md#term-javadoc),
Markdown, [checksum](../../../../glossary.md#term-checksum), exact-path, and whitespace tiers. The existing glossary [pooling](../../../../glossary.md#term-pooling) entry remains
accurate and required no edit. Shared [API](../../../../glossary.md#term-application-programming-interface-api), architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), architecture tests, backend
[conformance](../../../../glossary.md#term-conformance), integration, [Gradle](../../../../glossary.md#term-gradle), [Compiler](../../../../glossary.md#term-compiler)/Training guides, and unrelated [operation](../../../../glossary.md#operation) contracts also
required no change because the implementation is confined to the existing CPU-private portable
route and does not alter their contracts. No unresolved issue or required follow-up remains for
0008G. CPU 0008H remains Draft and is the next planned family task.

Status: Complete
