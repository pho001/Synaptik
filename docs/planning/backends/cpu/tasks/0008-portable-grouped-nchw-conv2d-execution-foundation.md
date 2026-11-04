# Task 0008: Portable Grouped [NCHW](../../../../glossary.md#term-nchw) [Conv2d](../../../../glossary.md#convolution--conv2d) Execution Foundation

## Reader context

This task records the grouped NCHW Conv2d foundation: `CpuConv2dLowering` selects static geometry,
`CpuConv2dIr` identifies code-shaping facts, and `CpuConv2dEmitter` produces direct typed loops.
`CpuConv2dReferenceKernel` is the independent semantic and performance oracle; Class-File inspection
is the structural oracle proving that generated entries contain computation rather than hidden
dispatch. Intrinsic bias and external ADD retain distinct numerical boundaries.

Here, `CpuPreparedExecutableSequence` represents only the closed two-unit materialized suffix.
Analysis declares its intermediate as an ordinary graph-value buffer before assignment; cold binding
validates both children before writes. Runtime, not the composite, owns validity and publication.
Direct convolution requires no scratch or persistent prepared resource, and workers own complete
output cells.

Read the opening flow as two alternatives: a retained epilogue stays in one generated unit, while a
materialized suffix uses two ordered units. It is not yet general graph decomposition.

Current-topic navigation (not a replacement for the historical record):

- [current grouped NCHW convolution meaning](../../../../reference/tensor/convolution.md#grouped-nchw-conv2d-expressions) — Explains channel groups, intrinsic bias, and checked spatial geometry without implying backend support.
- [current CPU preparation and invocation handoffs](../../../../explanation/packages/cpu.md#read-the-pipeline-before-the-packages) — Names the current analysis, shared assignment, finalization, cold binding, and Runtime participants.

## Status

Complete

## Goal

Add truthful portable generated execution for the current first-class grouped [NCHW](../../../../glossary.md#term-nchw) [`CONV2D`](../../../../glossary.md#convolution--conv2d)
[operation](../../../../glossary.md#operation), with or without its intrinsic [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) output-[channel](../../../../glossary.md#term-channel) [bias](../../../../glossary.md#term-bias), over fully static resolved
[layouts](../../../../glossary.md#layout). Establish the smallest family-owned Conv2d-led [epilogue](../../../../glossary.md#term-epilogue) and deterministic split boundary
needed before dimensional-[convolution](../../../../glossary.md#term-channels-first-convolution-layers) closure, without introducing general [partition](../../../../glossary.md#partition)-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [fusion](../../../../glossary.md#term-fusion),
recognition, profitability, or representation [planning](../../../../glossary.md#term-planning).

## Motivation and [mental model](../../../../glossary.md#term-mental-model)

The previous Draft [CPU](../../../../glossary.md#term-cpu) 0008 row combined [Conv2d](../../../../glossary.md#convolution--conv2d), [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), two [pooling](../../../../glossary.md#term-pooling) families, attention, three
[loss](../../../../glossary.md#loss) families, and bounded epilogues. Those families have different semantic algorithms,
multi-output and reduction behavior, [Shape](../../../../glossary.md#shape) rules, numerical policies, resources, generated-loop
shapes, and evidence [matrices](../../../../glossary.md#term-review-matrix). Implementing them together would violate the [planning](../../../../glossary.md#term-planning) guide's
cohesive-task rule and make failures difficult to isolate.

This task therefore establishes one complete capability:

```text
one CPU-owned partition containing one grouped NCHW Conv2d occurrence
  -> validate exact static descriptors, layouts, types, geometry, aliases, and publication
  -> lower one family-owned convolution unit
  -> optionally retain one narrowly legal Conv2d-led pointwise epilogue
  -> otherwise form a deterministic family-local materialized split
  -> generate or safely reuse exact compatible class bytes
  -> cold-bind direct typed carriers and optional worker state
  -> execute the prepared recipe without graph or route decisions
```

MATMUL, pooling, attention, and loss execution remain explicitly named Draft follow-ups. CPU
0008A remains the immediate next task because Conv1d visibly composes through Conv2d and [Conv3d](../../../../glossary.md#conv3d--ncdhw)
needs the [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) route as its implementation [oracle](../../../../glossary.md#term-test-oracle).

## [Scope](../../../../glossary.md#term-scope)

### Exact admitted [Conv2d](../../../../glossary.md#convolution--conv2d) [occurrence](../../../../glossary.md#term-occurrence)

- Add truthful `CpuCapabilityProvider` [admission](../../../../glossary.md#term-admission) for exactly `Conv2dKind.CONV2D` with
  `Conv2dAttrs` and exactly ordered `[input, weight]` or `[input, weight, bias]` inputs plus one
  output.
- Admit only fully static [Shapes](../../../../glossary.md#shape) and present resolved `LayoutDescriptor` values accepted by the
  existing [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) boundary.
- Require input Shape `[N, C_in, H, W]`, [weight](../../../../glossary.md#term-weight) Shape
  `[C_out, C_in/groups, K_h, K_w]`, and output Shape `[N, C_out, H_out, W_out]` with the exact
  Model/Compiler formula and constraints.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Require positive static kernel extents, positive stride/dilation/groups, non-negative symmetric
  padding, exact checked channel divisibility, exact<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [padding glossary](../../../../glossary.md#padding); [channel glossary](../../../../glossary.md#term-channel))</span>
  `weightChannelsPerGroup * groups == C_in`, and exact optional [bias](../../../../glossary.md#term-bias) Shape `[C_out]`.
- Preserve the exact ordered floating promotion across input, weight, then optional intrinsic
  bias. Admit FLOAT64, FLOAT32, and BFLOAT16 occurrences only where the result descriptor matches
  that promotion.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Preserve all six current Model data types as the closed repository type contract: Conv2d rejects
  BOOL, INT32, and INT64 because the Model operation is floating-only. <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [operation glossary](../../../../glossary.md#operation))</span>[`FLOAT16`](../../../../glossary.md#term-float16) is not a current
  [`DataType`](../../../../glossary.md#data-type--datatype); if that future type appears before its Model semantic task and an explicit CPU route,
  capability and lowering must fail closed rather than infer support from a two-byte carrier.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Support zero batch, input height/width, output-channel, input-channel, or output-spatial extents
  only where the current Model/Compiler geometry makes the descriptor legal. Empty output domains
  perform no read, write, generated invocation, or worker submission. An empty channel
  contraction begins from positive zero before optional intrinsic bias.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [spatial glossary](../../../../glossary.md#term-spatial-axes); [contraction glossary](../../../../glossary.md#contraction-dimension); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>

### Semantic and numerical contract

- Implement grouped [NCHW](../../../../glossary.md#term-nchw) cross-correlation, not [kernel](../../../../glossary.md#kernel)-reversing mathematical [convolution](../../../../glossary.md#term-channels-first-convolution-layers).
- For each output coordinate, traverse the selected contiguous input-channel group, kernel height,
  and kernel width in increasing logical order. Out-of-range padded coordinates are conceptual
  positive zero and participate in ordinary IEEE-754 multiplication, including multiplication by
  infinity.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [contiguous glossary](../../../../glossary.md#term-contiguous); [channel glossary](../../../../glossary.md#term-channel); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [IEEE glossary](../../../../glossary.md#term-ieee); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- FLOAT64 output uses FLOAT64 accumulation. FLOAT32 and BFLOAT16 output use FLOAT32 accumulation;
  BFLOAT16 narrows once at the final semantic output boundary.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Optional intrinsic bias participates exactly once in the selected accumulation domain before the
  final output conversion.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>
- Preserve the Model permission for reassociation and fused multiply-add without claiming
  cross-backend bitwise identity. Within each selected generated specialization, however, the
  optimal clean Java oracle and generated code must use the identical semantic algorithm,
  traversal, hot-loop/dataflow shape, rounding boundaries, and avoidable-overhead profile.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [reassociation and fused multiply-add glossary](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor); [backend glossary](../../../../glossary.md#backend); [identity glossary](../../../../glossary.md#term-identity-meanings); [specialization glossary](../../../../glossary.md#term-specialization); [oracle glossary](../../../../glossary.md#term-test-oracle); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Define and test NaN, infinity, signed-zero, conceptual-padding, empty-contraction, mixed-floating,
  group, stride, dilation, and symmetric-padding behavior directly from the current Model contract.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [padding glossary](../../../../glossary.md#padding); [contraction glossary](../../../../glossary.md#contraction-dimension); [stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation))</span>

### [Layout](../../../../glossary.md#layout), access, and [aliasing](../../../../glossary.md#term-aliasing)

- Consume existing resolved layouts and existing typed [heap](../../../../glossary.md#term-heap)-array, native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment),
  and mixed-carrier cold bindings. Do not add another shared representation or storage API.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [bindings glossary](../../../../glossary.md#term-binding); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Derive checked base offsets, element strides, referenced spans, group geometry, kernel geometry,
  and output-coordinate traversal during analysis/finalization or cold binding, never per output
  element through a graph object or generic layout lookup.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [strides glossary](../../../../glossary.md#term-stride); [kernel glossary](../../../../glossary.md#kernel); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [graph glossary](../../../../glossary.md#graph); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Support arbitrary legal non-negative resolved strides and offsets through a correct typed scalar
  fallback. Add dense or otherwise specialized forms only when cold proof and retained evidence
  justify them.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Validate complete carrier compatibility, liveness, byte order/alignment, referenced spans, and
  output/input overlap before the first output mutation or worker submission.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Reject every output/input overlap for this first capability. Input/input aliasing, including a
  repeated exact carrier, is allowed only when each logical read remains valid and the complete
  spans pass. Output workspace overlap is rejected if a family-local split declares workspace.
- Do not introduce in-place convolution, hidden copying, provider packing, layout reorder,
  representation reuse policy, or automatic alias analysis.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>

### [Publication](../../../../glossary.md#term-publication) and resource contract

- The final published Conv2d or epilogue result remains a normal graph output backed by its
  declared buffer. No backend-private value may replace a requested publication.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [epilogue glossary](../../../../glossary.md#term-epilogue); [graph glossary](../../../../glossary.md#graph); [buffer glossary](../../../../glossary.md#term-buffer); [backend glossary](../../../../glossary.md#backend))</span>
- A same-unit legal intermediate remains a graph/logical value without a Runtime slot only when it
  is single-use, unpublished, and all semantic and resource conditions below hold.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Every deterministic split intermediate is declared exactly once as a normal buffer resource with
  checked byte size and alignment before shared slot assignment. It is not an analysis-local
  workspace and must participate in ordinary Runtime validity transitions.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- The direct Conv2d computation requires zero scratch workspace. Any implementation proposal that
  needs im2col, packing, a partial-sum buffer, a combine buffer, or another scratch resource must
  stop and replan; this foundation selects direct grouped traversal.<span data-glossary-navigation="true"> ([im2col glossary](../../../../glossary.md#term-im2col))</span>
- No persistent prepared resource, native provider resource, transfer recipe, or new shared
  resource kind is added.<span data-glossary-navigation="true"> ([prepared resource glossary](../../../../glossary.md#prepared-resource--preparedresource))</span>

### Bounded [Conv2d](../../../../glossary.md#convolution--conv2d)-led [epilogue](../../../../glossary.md#term-epilogue) and safe split

- A [partition](../../../../glossary.md#partition) may contain one Conv2d [occurrence](../../../../glossary.md#term-occurrence) followed in direct single-use order by:
  - an optional compatible ordinary `BinaryArithmeticKind.ADD` [bias](../../../../glossary.md#term-bias) occurrence; and
  - at most one already [CPU](../../../../glossary.md#term-cpu)-supported exact [unary](../../../../glossary.md#term-unary) [activation](../../../../glossary.md#term-activation) or `ScalarElementwiseKind.CLAMP`.
- The optional external ADD is bias-like only when one operand is the Conv2d result, the other is
  an external input whose exact resolved right-aligned broadcast produces the Conv2d output Shape,
  and both operands and output have the same FLOAT64 or FLOAT32 type. Intrinsic Conv2d rank-one bias
  remains a separate first-class Conv2d input and may coexist only if the graph explicitly contains
  both operations.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [graph glossary](../../../../glossary.md#graph))</span>
- The optional activation is limited to an existing exact FLOAT64/FLOAT32 pointwise opcode whose
  current capability, numerical algorithm, and carrier topology are already implemented. It must
  consume the immediately preceding result exactly once and produce the same Shape and type.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [opcode glossary](../../../../glossary.md#term-opcode); [carrier glossary](../../../../glossary.md#term-storage-carrier); [topology glossary](../../../../glossary.md#term-topology))</span>
- Fusion is legal only for a connected straight-line chain with one Conv2d, single-use unpublished
  intermediates, one final publication, compatible layouts/accesses, no alias, state, random,
  multi-output, exceptional validation, or numerical-order conflict, and no new workspace.<span data-glossary-navigation="true"> ([Fusion glossary](../../../../glossary.md#term-fusion); [publication glossary](../../../../glossary.md#term-publication); [layouts glossary](../../../../glossary.md#layout); [alias glossary](../../../../glossary.md#aliasing-and-arena); [validation glossary](../../../../glossary.md#term-validation))</span>
- Fused code must preserve every operation boundary's represented result. It must not fold external
  ADD into the convolution accumulator, reassociate activation with convolution or ADD, erase an
  intermediate narrowing, or treat intrinsic and external bias as one semantic role.<span data-glossary-navigation="true"> ([Fused glossary](../../../../glossary.md#specialization-and-fusion); [operation glossary](../../../../glossary.md#operation); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [accumulator glossary](../../../../glossary.md#accumulator-type); [narrowing glossary](../../../../glossary.md#term-narrowing))</span>
- BFLOAT16 Conv2d is supported only as a direct first-class occurrence, including its intrinsic
  bias. External BFLOAT16 pointwise epilogues remain fail-closed because current CPU pointwise
  arithmetic does not support BFLOAT16.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- When a CPU-owned partition contains the admitted Conv2d-led bounded topology but fusion is
  illegal, lower a deterministic family-local split: direct Conv2d materializes its semantic
  output, followed by the existing exact pointwise unit where that unit is itself supported.
- This split is not general partition-DAG decomposition. Reject a partition with a predecessor of
  Conv2d, fan-out inside the proposed chain, more than the bounded nodes, another specialized
  family, horizontal fusion, multiple Conv2d occurrences, or an unsupported pointwise suffix.
  CPU 0008B owns the general solution; capability tests must keep unsupported complete partition
  shapes fail-closed.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [fan-out glossary](../../../../glossary.md#term-fan-out); [nodes glossary](../../../../glossary.md#node))</span>

### Narrow two-unit materialized-split contract

The family-local split is one explicit exception to the current [CPU](../../../../glossary.md#term-cpu) plan's one-unit [invariant](../../../../glossary.md#term-invariant). It
is permitted only when all of the following are true:

- the [partition](../../../../glossary.md#partition) has the exact bounded [Conv2d](../../../../glossary.md#convolution--conv2d)-led straight-line [topology](../../../../glossary.md#term-topology) admitted above;
- the direct Conv2d lead is independently supported;
- the complete remaining non-empty suffix, consisting of the optional ADD, the optional
  activation/clamp, or both in admitted order, is independently realizable as one existing exact
  pointwise unit;<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation))</span>
- [fusion](../../../../glossary.md#term-fusion) of the complete chain is illegal under the gates above; and
- exactly one materialized Conv2d-result buffer is sufficient to connect the two units while
  preserving every requested graph publication.

The selected plan is then tagged as the CPU-private Conv2d materialized-suffix form and contains
exactly two units in [producer](../../../../glossary.md#term-producer)-to-consumer order. Every existing family, direct Conv2d, and legal
[fused](../../../../glossary.md#specialization-and-fusion) Conv2d [epilogue](../../../../glossary.md#term-epilogue) retains exactly one unit. A two-unit plan with another tag or family, a third
unit, a second materialized edge, [fan-out](../../../../glossary.md#term-fan-out) to another computation, a predecessor unit, another
specialized family, or a suffix that is not independently supported fails preparation. This tag
is not a general [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag), fusion-candidate, recognition, profitability, or representation-[planning](../../../../glossary.md#term-planning)
[model](../../../../glossary.md#term-model) and must not be reused by CPU 0008B–0008E as their general contract.

Each of the two `ExecutionUnitPlan` entries owns its exact portable route plan, ordered boundary
[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values, access [bindings](../../../../glossary.md#term-binding), requested and generated [carrier](../../../../glossary.md#term-storage-carrier) patterns, logical range
extents and count, [scalar](../../../../glossary.md#term-scalar)/parallel-scalar strategy, selected range count, minimum work per worker,
zero [vector](../../../../glossary.md#term-vector)-species bits, unit-local geometry, and diagnostic [lowering](../../../../glossary.md#lowering) reason. Existing one-unit
plans may retain their current partition-level mirrors for compatibility, but the split finalizer
and executable must consume the per-unit facts and must not reuse Conv2d ranges, bindings,
carriers, or geometry for the [pointwise](../../../../glossary.md#term-pointwise) suffix. Neither split unit may select [materialization](../../../../glossary.md#materialization) or
workspace.

Analysis declares all distinct [graph](../../../../glossary.md#graph)-value [buffers](../../../../glossary.md#term-buffer) before returning
[`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis), including the Conv2d result exactly once in deterministic producer
order before the final suffix result. Its byte size is the checked [referenced element span](../../../../glossary.md#referenced-element-span) times
the [descriptor](../../../../glossary.md#term-descriptor) byte width and its [alignment](../../../../glossary.md#term-alignment) is that byte width, exactly like another CPU graph
buffer. The value is never a `Workspace` requirement. Repeated operand roles reuse the same
declaration and later the same assigned buffer position. Requested Conv2d or suffix-intermediate
publications remain ordinary publications; the two-unit ceiling applies only when the suffix can
still be represented by one exact pointwise unit without dropping any of them.

After shared assignment, `CpuPartitionFinalizer` validates the complete declaration/assignment
set and both units before [artifact](../../../../glossary.md#term-artifact) realization. It resolves each unit's selections by exact
`ValueId`, verifies the [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) worker group independently against that unit's range count, and
loads or generates exactly two artifacts in unit order. It then returns one CPU-private composite
[`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), because the unchanged shared [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition) and schedule [validation](../../../../glossary.md#term-validation)
require one executable [occurrence](../../../../glossary.md#term-occurrence) per planned partition. The dedicated
`CpuPreparedExecutableSequence` [owner](../../../../glossary.md#term-owner) is permitted solely for this two-unit form; it is not a
general unit list, scheduler, interpreter, or fusion planner.

The composite declares every external input as `READ_ONLY` and the materialized Conv2d result and
final suffix result as `WRITE_ONLY`, deduplicated by exact buffer selection. Cold binding resolves
and type-checks all selected representations, constructs both child bound invocations, and
validates complete per-unit and cross-unit span, [liveness](../../../../glossary.md#term-liveness), alignment, writability, and overlap
preconditions before either child is called or any worker work is submitted. The second unit's
read of the intermediate is an internal ordered [dependency](../../../../glossary.md#term-dependency) of the already-bound composite; it is
not exposed as a pre-existing [Runtime](../../../../glossary.md#term-runtime) read, because a newly created intermediate starts invalid.

Runtime therefore applies its ordinary [atomic](../../../../glossary.md#resource-transaction-and-rollback) executable transition: it validates all external
reads and invalidates every copy of both declared outputs before the composite call, then marks
the exact intermediate and final written copies valid only after both child invocations return
successfully. The composite performs no direct [`RunState`](../../../../glossary.md#run-state--runstate) validity mutation. If either unit
throws, Runtime marks neither output valid, publishes nothing, and performs its existing failure
cleanup; bytes written by the first unit remain logically invalid. This preserves failure-before-
mutation for every validation that can be completed cold and failure-before-[publication](../../../../glossary.md#term-publication) for an
execution failure without adding a [backend](../../../../glossary.md#backend)-owned coherence rule.

Execution is strictly `Conv2d artifact -> completion/join -> pointwise artifact`. Each unit owns
disjoint complete ranges under its own prepared strategy. A shared borrowed [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) may
serve both units, but their submissions never overlap; the second starts only after the first has
completed successfully. Empty ranges submit no work and do not call their generated entry.

The final publication suffix remains Runtime-owned and unchanged. It may publish any requested
declared result only after the single composite occurrence succeeds; the CPU composite neither
publishes nor transfers [ownership](../../../../glossary.md#term-ownership-meanings). Each artifact retains its own existing [specialization](../../../../glossary.md#term-specialization) and
generated-artifact [cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings). The Conv2d and pointwise keys exclude plan tags, `ValueId`
values, slots, representations, addresses, and [run state](../../../../glossary.md#run-state--runstate). There is no cached or persisted
composite artifact and no family-wide two-unit [cache key](../../../../glossary.md#term-cache-key); the lowering [manifest](../../../../glossary.md#term-manifest) may record the
ordered split and the two artifact keys for diagnostics only.

Generated-code evidence must inspect both realized classes and prove that each contains its own
direct typed loop. Split semantic/performance evidence compares the ordered generated pair with
two optimal clean Java loops over the same ordinary intermediate carrier and identical [operation](../../../../glossary.md#operation)
boundaries. Existing pointwise evidence may support the unchanged suffix emitter, but this task
must still retain the exact split suffix class inspection, ordered-invocation integration check,
intermediate validity/publication checks, and complete generated-pair performance result.

### Portable route, generated code, and execution

- Add one focused CPU-private Conv2d IR/geometry owner, one lowering owner, one direct Class-File
  emitter, and one independent direct Java reference/oracle owner. Reuse existing carrier,
  specialization, artifact, worker, prepared-executable, and staged-finalization contracts.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [owner glossary](../../../../glossary.md#term-owner); [lowering glossary](../../../../glossary.md#lowering); [oracle glossary](../../../../glossary.md#term-test-oracle); [specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- The family IR identity includes kind, intrinsic-bias presence, ordered boundary data types,
  accumulation/output type, grouping and kernel/stride/padding/dilation semantics, admitted
  epilogue topology, access/carrier code-shaping facts, and selected execution strategy. Runtime
  identities, <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [bias glossary](../../../../glossary.md#term-bias); [kernel glossary](../../../../glossary.md#kernel); [stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation); [epilogue glossary](../../../../glossary.md#term-epilogue); [topology glossary](../../../../glossary.md#term-topology); [Runtime glossary](../../../../glossary.md#term-runtime))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values, exact [carrier](../../../../glossary.md#term-storage-carrier) objects, addresses, slots, and [run state](../../../../glossary.md#run-state--runstate) stay out.
- Advance `CpuGeneratorSchema` exactly once. Older schema envelopes are incompatible safe misses;
  no migration reader is added.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Generate direct typed [scalar](../../../../glossary.md#term-scalar) bodies for `double[]`, `float[]`, raw-[BFLOAT16](../../../../glossary.md#term-bfloat16) `short[]`,
  [`MemorySegment`](../../../../glossary.md#term-memorysegment), and the admitted mixed patterns. Generated entries are field-free and
  constructor-free and expose one typed static entry matching the selected boundary pattern.
- Use scalar and bounded parallel-scalar execution over deterministic disjoint complete output-cell
  ranges. The generated method retains primitive range inputs. Each output cell is owned by one
  range; no partial/combine accumulation or synchronization is permitted.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Vector API convolution, im2col, Winograd, FFT, native BLAS/DNN providers, packing, autotuning,
  dynamic route choice, and shape-baked specialization are excluded.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [im2col glossary](../../../../glossary.md#term-im2col); [Winograd glossary](../../../../glossary.md#alternative-convolution-algorithms); [FFT glossary](../../../../glossary.md#alternative-convolution-algorithms); [BLAS glossary](../../../../glossary.md#term-blas); [DNN glossary](../../../../glossary.md#term-dnn); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [shape glossary](../../../../glossary.md#shape))</span>
- The optimal clean Java oracle must be independently readable and must not delegate its hot loop
  to generated code or to the same emitter helper. Generated hot loops must not be bridge-only and
  must not call a Synaptik-owned per-output or per-contribution helper.
- Runtime execution uses already prepared direct fields. No operation/graph inspection, route
  decision, cache lookup, reflection, string dispatch, boxing, collection/map lookup, allocation,
  type discovery, layout construction, or repeated compatibility cast is allowed in generated or
  bound hot work.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph); [cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [layout glossary](../../../../glossary.md#layout))</span>
- If specialization cannot be proved, choose the direct typed general scalar generated fallback.
  Never fall back to a Runtime interpreter or advertise a form that preparation cannot realize.

### Documentation and [Javadoc](../../../../glossary.md#term-javadoc)

- Add or update meaningful Javadoc for every affected production type and method, including
  semantic roles, geometry, numerical boundaries, ownership, resources, concurrency, nullability,
  return semantics, and failure conditions.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Update `docs/backend-guide/cpu-backend.md` with the current admitted Conv2d subset, direct
  traversal, epilogue/split boundary, carrier/layout/alias/resource behavior, generated-code
  evidence, and limitations.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [epilogue glossary](../../../../glossary.md#term-epilogue); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Update `docs/glossary.md` only if implementation introduces a reusable new term. Otherwise record
  a reasoned no-change conclusion.
- The separate clean documentation-focused pass must review the final diff, Model/Compiler Conv2d
  contracts, affected Javadocs/package summaries, CPU guide, glossary, task, master plan, and
  roadmap. It reuses successful Java/performance evidence unless it changes executable behavior or
  records a concrete stale-evidence risk.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- MATMUL and linear execution; max/average pooling; scaled dot-product attention; mean-squared,
  dense categorical, or index categorical loss execution. Draft CPU 0008F–0008I own them.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [linear glossary](../../../../glossary.md#term-linear-meanings); [pooling glossary](../../../../glossary.md#term-pooling); [scaled dot-product attention glossary](../../../../glossary.md#scaled-dot-product-attention); [mean glossary](../../../../glossary.md#term-mean); [loss glossary](../../../../glossary.md#loss))</span>
- Conv1d end-to-end composition [validation](../../../../glossary.md#term-validation) or direct [Conv3d](../../../../glossary.md#conv3d--ncdhw) execution; [CPU](../../../../glossary.md#term-cpu) 0008A owns them.
- General partition-DAG decomposition, horizontal/vertical fusion, typed specialized-subgraph
  recognition, general legality/profitability facts, or bounded representation/materialization
  variants; CPU 0008B–0008E own them. The exact tagged two-unit Conv2d materialized-suffix form is
  the sole exception and must not become reusable general DAG machinery.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- A public or [backend](../../../../glossary.md#backend)-generic `ConvNd`, another Model/Compiler operation, inference or gradient
  change, Compiler 0006C, or any public API expansion.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [operation glossary](../../../../glossary.md#operation); [inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Dynamic or unresolved Shapes/layouts, asymmetric implicit padding, transposed/deformable/causal
  specialized convolution, depthwise-special naming, sparse/quantized convolution, dynamic rank,
  or arbitrary-rank convolution.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [padding glossary](../../../../glossary.md#padding); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [rank glossary](../../../../glossary.md#term-rank))</span>
- FLOAT16, mixed-precision accumulation policy beyond current ordered promotion, BFLOAT16 external
  pointwise epilogues, relaxed math, or a new determinism/numerical mode.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [precision glossary](../../../../glossary.md#term-precision); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Vector/native convolution, OpenBLAS, Accelerate, oneMKL, oneDNN, AOCL, ZenDNN, im2col, Winograd,
  FFT, packing/reorder, tuning, tuning-cache use, measurement-driven preparation, tracing payloads,
  or benchmark-setting mutation.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [oneMKL glossary](../../../../glossary.md#term-onemkl); [oneDNN glossary](../../../../glossary.md#term-onednn); [AOCL glossary](../../../../glossary.md#term-aocl); [ZenDNN glossary](../../../../glossary.md#term-zendnn); [im2col glossary](../../../../glossary.md#term-im2col); [Winograd glossary](../../../../glossary.md#alternative-convolution-algorithms); [FFT glossary](../../../../glossary.md#alternative-convolution-algorithms); [cache glossary](../../../../glossary.md#term-cache))</span>
- New shared Prepare, Runtime, Planning, Backend Contract, Config, Trace, Engine, architecture,
  dependency, Gradle, toolchain, or module-boundary contracts.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Planning glossary](../../../../glossary.md#term-planning); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [module glossary](../../../../glossary.md#term-module))</span>
- General backend conformance or end-to-end Engine integration infrastructure. CPU 0009 owns the
  portable conformance checkpoint; Engine 0004 owns the typed end-to-end convolution lifecycle.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Planning,
  Prepare, Runtime, concrete backend, generated CPU kernel, performance-evidence, and testing rules<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [concrete backend glossary](../../../../glossary.md#concrete-backend); [CPU glossary](../../../../glossary.md#term-cpu); [kernel glossary](../../../../glossary.md#kernel))</span>
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Backend-owned lowering glossary](../../../../glossary.md#backend-owned-lowering))</span>
- [ADR 0008: Performance evidence and tuning boundaries](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership))</span>
- [CPU kernel strategy](../../../../design/notes/cpu-kernel-strategy.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Kernel routes](../../../../backend-guide/kernel-routes.md)
- [Planning guide](../../../planning-guide.md)

## Architecture constraints

- Planning continues to select only CPU ownership. CPU analysis owns Conv2d lowering, bounded
  family-local epilogue legality/split, route choice, and exact resource declarations.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [lowering glossary](../../../../glossary.md#lowering); [epilogue glossary](../../../../glossary.md#term-epilogue))</span>
- [CPU](../../../../glossary.md#term-cpu) analysis remains deterministic from [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) and explicit CPU inputs. Finalization
  generates or reuses the already selected artifact only after shared slot assignment and adds no
  undeclared resource.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Shared Prepare still receives one analysis and one finalized executable for the planned
  partition. The tagged split is internal to that CPU executable; it requires no change to<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [partition glossary](../../../../glossary.md#partition))</span>
  [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition), [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule), [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner), or any shared validity [API](../../../../glossary.md#term-application-programming-interface-api).
- Runtime owns the composite occurrence's normal access transition. The CPU sequence neither
  changes <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>[`RunState`](../../../../glossary.md#run-state--runstate) validity nor publishes an intermediate from inside [backend](../../../../glossary.md#backend) work.
- Runtime receives only immutable prepared recipes and direct cold-bound invocations. Its hot path
  sees no <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [hot path glossary](../../../../glossary.md#term-hot-path))</span>[`Operation`](../../../../glossary.md#operation), `CompiledNode`, route, [layout](../../../../glossary.md#layout), or [graph](../../../../glossary.md#graph) fact.
- Providers do not reinterpret graphs. This task adds no provider route.
- Generated code follows an optimal clean Java implementation of the identical specialized case as
  its design, review, and performance oracle.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Benchmark/performance evidence is observational and cannot mutate production settings. No search
  or cache mutation occurs in Runtime.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Any required public/shared/module/dependency/architecture change is a stop condition, not an
  implementation detail.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — truthful public capability [admission](../../../../glossary.md#term-admission) only.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — [immutable](../../../../glossary.md#term-immutable) family code-shaping [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — family geometry, bounded epilogue,
  and split construction.<span data-glossary-navigation="true"> ([epilogue glossary](../../../../glossary.md#term-epilogue))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct Class-File body emission.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent direct Java [oracle](../../../../glossary.md#term-test-oracle).
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — already selected route facts.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — exact declarations/finalization state.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold [binding](../../../../glossary.md#term-binding) and direct execution.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema) and [compatibility identity](../../../../glossary.md#identity-meanings-and-algebraic-identities).

Packages added or changed:

- No package is added. Only the existing [CPU](../../../../glossary.md#term-cpu)-private responsibility packages above may change.

Expected new type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr` — immutable Conv2d and admitted
  epilogue code-shaping identity.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLowering` — exact static
  geometry, boundary, fusion, and family-local split owner.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [owner glossary](../../../../glossary.md#term-owner))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConv2dEmitter` — direct generated
  scalar computation body.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference.CpuConv2dReferenceKernel` — independent
  clean Java semantic/performance oracle.
- `io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableSequence` — the
  sole two-unit Conv2d materialized-suffix composite, with complete cold validation and ordered
  direct child invocation; it is not a general scheduler.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

Tests mirror their production packages. A test-only performance harness remains under
`internal.codegen.emit`; it is not production tuning infrastructure.

## Affected files

Expected production and [Javadoc](../../../../glossary.md#term-javadoc) paths:

- `CpuCapabilityProvider.java` and [CPU](../../../../glossary.md#term-cpu) package Javadoc;
- `internal/ir/CpuPortableKernelIr.java`, new `CpuConv2dIr.java`, and [IR](../../../../glossary.md#term-intermediate-representation-ir) package Javadoc;
- `internal/lowering/CpuPartitionLowering.java`, new `CpuConv2dLowering.java`, and lowering package
  Javadoc;<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new `CpuConv2dEmitter.java`, and emitter
  package Javadoc;
- `internal/reference/CpuConv2dReferenceKernel.java` and reference package Javadoc;
- `internal/cache/CpuGeneratorSchema.java` and compatibility owners only where Conv2d identity
  requires them;<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- route, prepare, finalizer, and executable owners plus package Javadocs only where the new IR,
  family-local split, declarations, or direct bound fields require integration; and<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>
- new `internal/executable/CpuPreparedExecutableSequence.java` only for the exact tagged two-unit
  split and its atomic Runtime-facing access declaration.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

Expected tests:

- focused capability, IR, lowering, preparation/finalization, executable/binding, reference,
  generated-kernel, schema/cache-compatibility, package-inventory, evidence, and performance tests;<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [kernel glossary](../../../../glossary.md#kernel); [schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache))</span>
- explicit negative [matrices](../../../../glossary.md#term-matrix-meanings) for non-floating types, the exact six-value [`DataType`](../../../../glossary.md#data-type--datatype) inventory and
  absence of FLOAT16 admission, unresolved Shapes or layouts, descriptor mismatch, illegal
  geometry, unsupported partitions, illegal epilogues,
  overlap, span/alignment/carrier mismatch, and mutation-before-validation;<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [admission glossary](../../../../glossary.md#term-admission); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [alignment glossary](../../../../glossary.md#term-alignment); [carrier glossary](../../../../glossary.md#term-storage-carrier); [validation glossary](../../../../glossary.md#term-validation))</span>
- unchanged-family controls chosen from current [pointwise](../../../../glossary.md#term-pointwise) and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization) generated routes.

Expected documentation/[planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`;
- `docs/glossary.md` only if terminology changes;
- this task;
- `docs/planning/backends/cpu/master-plan.md`; and
- `docs/planning/roadmap.md`.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- 19 production/Javadoc paths, including exactly the four planned family production types and at
  most the one additional <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`CpuPreparedExecutableSequence` private [owner](../../../../glossary.md#term-owner) named above;
- 14 [CPU](../../../../glossary.md#term-cpu) test/resource paths;
- the five named documentation/[planning](../../../../glossary.md#term-planning) paths, with the glossary optional; and
- 38 total paths.

No other new production owner and no shared [module](../../../../glossary.md#term-module), architecture, [Gradle](../../../../glossary.md#term-gradle), [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance),
integration, benchmark-tool, later task-specification, or vendor-provider path is permitted. If
the bounded family-local split or [epilogue](../../../../glossary.md#term-epilogue) cannot fit this ceiling, or if another production
owner/new type is necessary, stop and replan rather than folding general 0008B–0008E work into
this task.

## Acceptance criteria

- CPU capability returns true exactly for the admitted static resolved-layout grouped NCHW Conv2d
  matrix and false for every excluded kind/type/signature/descriptor/geometry case.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [signature glossary](../../../../glossary.md#term-signature); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Direct unbiased and intrinsic-biased Conv2d execute correctly for groups 1, intermediate groups,
  and depthwise geometry; unit/non-unit stride and dilation; zero/non-zero symmetric padding;
  zero/empty legal domains; FLOAT64, FLOAT32, BFLOAT16, and ordered mixed-floating inputs; heap,
  segment, and selected mixed carriers; contiguous and arbitrary legal resolved layouts.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [padding glossary](../../../../glossary.md#padding); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [heap glossary](../../../../glossary.md#term-heap); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Results match the current Model/Compiler Shape, grouping, promotion, conceptual-padding,
  accumulation, bias, special-value, and final-conversion contracts.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Shape glossary](../../../../glossary.md#shape); [bias glossary](../../../../glossary.md#term-bias))</span>
- The bounded FLOAT64/FLOAT32 external ADD plus at-most-one activation/clamp topology is fused only
  when every stated legality gate holds and preserves each semantic operation boundary.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [topology glossary](../../../../glossary.md#term-topology); [fused glossary](../../../../glossary.md#specialization-and-fusion); [operation glossary](../../../../glossary.md#operation))</span>
- Every other admitted bounded topology uses a deterministic family-local materialized split with
  exact declarations, or fails capability/preparation closed if the suffix is not independently
  supported. No requested result or split intermediate is silently dropped.
- The only multi-unit plan is the exact tagged Conv2d materialized-suffix form with exactly two
  complete per-unit fact sets. Every old family, direct Conv2d, and legal fused Conv2d plan still
  rejects `units().size() != 1`; malformed tags, unit counts, boundaries, or roles fail before
  finalization.
- Analysis declares the Conv2d intermediate exactly once as an ordinary buffer before shared
  assignment, never as workspace. Finalization resolves exact assigned selections and realizes
  both ordered cache-compatible artifacts without adding a requirement.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [cache glossary](../../../../glossary.md#term-cache); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Cold binding validates both units and every cross-unit alias/access condition before the first
  write or worker submission. One composite execution runs Conv2d to completion before the
  pointwise suffix; Runtime invalidates and validates the intermediate and final output through
  the composite's ordinary declared accesses and publishes only after complete success.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [alias glossary](../../../../glossary.md#aliasing-and-arena); [pointwise glossary](../../../../glossary.md#term-pointwise); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Scalar and parallel-scalar strategies, ranges, and worker-capacity checks are unit-local. The
  two units never overlap worker submissions, and an empty unit submits no work.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar))</span>
- BFLOAT16 direct Conv2d works with intrinsic bias where otherwise legal; BFLOAT16 external
  pointwise epilogues, all non-floating types, and future FLOAT16 fail closed.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- All validation and overlap checks complete before output mutation or worker submission. Failure
  exposes no partially written published result.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Direct Conv2d uses zero workspace and scalar or parallel-scalar complete-output-cell ranges.
  Parallel and scalar results follow the same per-cell algorithm and do not share partial state.
- Generated classes are deterministic, final, field-free, constructor-free, and direct. Retained
  `javap -c -v`/Class-File inspection proves the expected typed entry signatures and absence of
  method handles, `invokedynamic`, dynamic constants, bootstrap methods, reflection, boxing,
  collection/map dispatch, allocation, route lookup, graph/operation references, and Synaptik-owned
  per-output/per-contribution helper calls.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [dispatch glossary](../../../../glossary.md#term-dispatch); [lookup glossary](../../../../glossary.md#term-lookup); [graph glossary](../../../../glossary.md#graph))</span>
- Split evidence inspects both exact generated classes, proves ordered direct invocation with no
  bridge/interpreter body, and compares the generated pair against two optimal clean Java loops
  using the same ordinary intermediate representation.<span data-glossary-navigation="true"> ([intermediate representation glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Every performance target compares generated execution with an optimal clean Java implementation
  of the identical specialized case. Each target and unchanged-family control passes five accepted
  isolated fixed-heap forks and the aggregate median at generated/direct <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [median glossary](../../../../glossary.md#term-median))</span>`<= 1.15x`.
- Evidence includes at minimum representative dense/grouped/depthwise, padded/dilated/strided,
  general-layout/mixed-carrier, BFLOAT16, fused-epilogue, and parallel forms. One whole fork may be
  rejected only before measurement for a recorded environment/classpath/control failure; no ratio
  sample is discarded.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [epilogue glossary](../../../../glossary.md#term-epilogue); [fork glossary](../../../../glossary.md#term-fork-meanings); [classpath glossary](../../../../glossary.md#term-classpath); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Schema advances exactly once and older artifacts are safe misses. Deterministic regeneration
  produces identical class bytes for the same specialization.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Current pointwise, affine/movement, indexing/scatter/fold/order/random, scan/reduction,
  softmax/normalization, batch-normalization, materialization, persistence, worker, preparation,
  and execution tests remain green.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [materialization glossary](../../../../glossary.md#materialization); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- No Model, Compiler, Planning, Prepare, Runtime, Engine, Backend Contract, Config, Trace, provider,
  architecture, dependency, build, conformance, integration, or public API contract changes.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Production and test types match the package map and maximum-[scope](../../../../glossary.md#term-scope) ceiling.
- A separate clean documentation-focused agent pass finalizes affected Javadocs, CPU guide,
  glossary impact, and planning evidence in the same overall change.
- [CPU](../../../../glossary.md#term-cpu) 0008 becomes `Complete` only after all gates pass. CPU 0008A is then the sole Ready CPU task;
  CPU 0008B–0008I and 0009 onward remain Draft and have no premature detailed task files.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused tests for the affected [owners](../../../../glossary.md#term-owner), for example:

```bash
./gradlew :backends:cpu:test \
  --tests '*CpuConv2d*' \
  --tests '*CpuCapabilityProviderTest' \
  --tests '*CpuPartitionPreparerTest' \
  --tests '*CpuPartitionFinalizerTest' \
  --tests '*CpuPreparedExecutableTest' \
  --tests '*CpuPreparedExecutableSequenceTest' \
  --tests '*CpuGeneratedKernelShapeTest' \
  --tests '*CpuInternalPackageInventoryTest'
```

After executable code stabilizes, run exactly one authoritative uncached [CPU](../../../../glossary.md#term-cpu) suite:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Retain machine-readable test counts and the deterministic Class-File/evidence bundle. Run the
task-owned evidence test or harness in five fresh isolated Java 26 fixed-[heap](../../../../glossary.md#term-heap) [forks](../../../../glossary.md#term-fork-meanings) using its exact
documented command. Each [fork](../../../../glossary.md#term-fork-meanings) must use deterministic inputs, randomized generated/direct order,
at least five [warmup](../../../../glossary.md#term-warmup) [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), nine measured rounds, adaptive batches of at least 25 ms, semantic
checks before timing, and explicit environment/control acceptance. Record per-case ratios,
aggregate medians, accepted/rejected forks, generated-class checksums, complete `javap -c -v`
member reports, and the evidence [manifest](../../../../glossary.md#term-manifest) [digest](../../../../glossary.md#term-digest).

The final documentation-focused pass receives the stabilized diff, final CPU XML, retained
performance/Class-File bundle, and exact implementation commands. It does not rerun successful
Java/performance work unless executable behavior changed or a concrete stale-evidence risk is
recorded. After final Javadocs it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also validates local Markdown links and anchors, headings, balanced fences, final newlines,
terminology, generated [Javadoc](../../../../glossary.md#term-javadoc) pages, exact path/type ceilings, [schema](../../../../glossary.md#term-schema)/status/order, absence of
later task specifications, and empty staging.

Repository-wide and architecture suites remain deferred to CPU 0009 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci)
because this task changes no [dependency](../../../../glossary.md#term-dependency) or shared boundary. [Backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) tests are not
added in this task because the current `testing/backend-conformance` [module](../../../../glossary.md#term-module) has only its foundation
marker and this task promises one CPU-private admitted subset, not a cross-backend contract; CPU
0009 is the named conformance [checkpoint](../../../../glossary.md#term-checkpoint). Integration tests are not required because [Engine](../../../../glossary.md#term-engine) does
not yet expose the public [lifecycle](../../../../glossary.md#lifecycle) and this task makes no end-to-end promise; Engine 0004 owns the
three-[rank](../../../../glossary.md#term-rank) [convolution](../../../../glossary.md#term-channels-first-convolution-layers) lifecycle. Any implementation change to those facts is a stop condition and
requires replanning plus proportionate validation.

## Dependencies

- Complete CPU 0005A–0007F2, including current whole-partition lowering, access plans, generated
  artifact/schema, materialization, worker/resource, direct generated-family, corrective parity,
  preparation/finalization, and bound-execution contracts.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [artifact glossary](../../../../glossary.md#term-artifact); [schema glossary](../../../../glossary.md#term-schema); [materialization glossary](../../../../glossary.md#materialization))</span>
- Complete Model 0020 owns grouped NCHW Conv2d semantics, attributes, public expressions, promotion,
  Shape, numerical meaning, and provenance.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- Complete Model 0025G owns visible Conv1d composition but is not implemented by this task; its
  existence fixes CPU 0008A's immediate validation dependency.<span data-glossary-navigation="true"> ([Conv1d composition glossary](../../../../glossary.md#conv1d-composition); [validation glossary](../../../../glossary.md#term-validation); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Complete Model 0025H and Compiler 0006B fix the later Conv3d forward contract and ordering but do
  not expand this task to rank three.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [rank glossary](../../../../glossary.md#term-rank))</span>
- Complete Compiler 0001–0006B own capture, inference/validation, gradients for Conv2d, publication,
  planning artifacts, and Conv3d forward adoption. Draft Compiler 0006C is not a prerequisite for
  forward CPU execution.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning))</span>
- Current shared Planning, Prepare, Runtime, Backend Contract, Config, and Trace contracts are
  sufficient and unchanged.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0008A: validate [NCW](../../../../glossary.md#term-ncw) Conv1d through visible composition and add direct grouped [NCDHW](../../../../glossary.md#conv3d--ncdhw) [Conv3d](../../../../glossary.md#conv3d--ncdhw).
- CPU 0008B: general [partition](../../../../glossary.md#partition)-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) computation-unit [decomposition](../../../../glossary.md#term-decomposition) and bounded [fusion](../../../../glossary.md#term-fusion).
- CPU 0008C: typed specialized-subgraph and [epilogue](../../../../glossary.md#term-epilogue) recognition.
- CPU 0008D: bounded fusion profitability and typed cold decision facts.
- CPU 0008E: bounded multi-input [materialization](../../../../glossary.md#materialization) and representation reuse.
- CPU 0008F: portable [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) execution and bounded [linear](../../../../glossary.md#term-linear-meanings) epilogues.
- CPU 0008G: portable max/average Pool2d execution.
- CPU 0008H: portable scaled-dot-product attention execution.
- CPU 0008I: portable [loss](../../../../glossary.md#loss)-family execution.
- CPU 0009: portable generated-coverage and [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) closure [checkpoint](../../../../glossary.md#term-checkpoint).
- [Engine](../../../../glossary.md#term-engine) 0004: typed end-to-end Conv1d/[Conv2d](../../../../glossary.md#convolution--conv2d)/Conv3d [lifecycle](../../../../glossary.md#lifecycle) [validation](../../../../glossary.md#term-validation).
- [Compiler](../../../../glossary.md#term-compiler) 0006C: separately prove and close Conv3d [gradients](../../../../glossary.md#term-gradient); it does not block forward execution.

## Architecture impact

Expected impact: None.

This task consumes the existing concrete-[backend](../../../../glossary.md#backend) analysis, [lowering](../../../../glossary.md#lowering), exact-resource declaration,
post-assignment finalization, generated-[kernel](../../../../glossary.md#kernel), cold-[binding](../../../../glossary.md#term-binding), worker, and prepared-execution
boundaries. It changes no [owner](../../../../glossary.md#term-owner), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), shared contract, or public [API](../../../../glossary.md#term-application-programming-interface-api). If the
implementation needs a shared contract, public surface, or architecture change, stop and report
the exact conflict before editing that boundary.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the isolated implementation agent for Synaptik CPU task 0008.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0008-portable-grouped-nchw-conv2d-execution-foundation.md.
Read the task's directly referenced Model, Compiler, CPU source/test, architecture, generated-code,
performance-evidence, and documentation contracts. Implement exactly the Ready specification and
do not implement later CPU tasks. Stop and report any architecture, shared-contract, or maximum-
scope conflict rather than inventing a boundary.

After implementation and recorded CPU/Class-File/performance validation, hand the stabilized diff
and evidence to a separate clean documentation-focused agent. That pass must follow
docs/developer-guide/documentation-rules.md, finalize Javadocs, CPU guide, glossary impact, and
planning evidence in the same overall change, and must not repeat successful Java/performance
work unless executable behavior changes or it records a concrete reason.

Do not commit or push unless the coordinating user explicitly authorizes it. Update this task's
local decisions, known limitations, validation evidence, implementation notes, completion summary,
and final status only from actual results.
```

## Local decisions

- The bounded illegal-but-independently-supported Conv2d suffix stays in CPU 0008 because the
  materialized split is part of this task's accepted Conv2d capability and its implementation is
  already in progress. Inserting or reordering another task would separate a required fallback
  from the capability it makes truthful.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Shared Prepare and Runtime remain unchanged. One CPU-private composite executable represents
  the two ordered artifacts to the existing one-executable-per-partition handoff, while its outer
  access declaration lets Runtime perform the ordinary atomic validity transition.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [artifacts glossary](../../../../glossary.md#term-artifact); [partition glossary](../../../../glossary.md#partition); [handoff glossary](../../../../glossary.md#term-handoff); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- `CpuPartitionPreparationPlan` preserves the one-unit invariant by default and admits exactly
  two units only under the closed Conv2d materialized-suffix tag. The exception is deliberately
  unsuitable as CPU 0008B's future general DAG model.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [model glossary](../../../../glossary.md#term-model))</span>
- One additional private executable-sequence [owner](../../../../glossary.md#term-owner) is preferable to adding general scheduling to
  `CpuPreparedExecutable` or changing shared schedule contracts. No other new production owner is
  permitted by this task.

## Known limitations

- The first route is direct scalar/parallel-scalar grouped NCHW traversal; vector/native,
  packing/im2col/Winograd/FFT, tuning, and shape-baked specialization remain deferred.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [NCHW glossary](../../../../glossary.md#term-nchw); [vector glossary](../../../../glossary.md#term-vector); [im2col glossary](../../../../glossary.md#term-im2col); [Winograd glossary](../../../../glossary.md#alternative-convolution-algorithms); [FFT glossary](../../../../glossary.md#alternative-convolution-algorithms); [shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- BFLOAT16 supports direct Conv2d and intrinsic bias only; external BFLOAT16 pointwise epilogues
  remain unsupported.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [bias glossary](../../../../glossary.md#term-bias); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Family-local epilogue/split handling does not solve general predecessor, fan-out, multi-Conv2d,
  horizontal, or mixed-specialized-family partitions; CPU 0008B owns that closure.<span data-glossary-navigation="true"> ([epilogue glossary](../../../../glossary.md#term-epilogue); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- [Backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) and public end-to-end [Engine](../../../../glossary.md#term-engine) [validation](../../../../glossary.md#term-validation) remain at [CPU](../../../../glossary.md#term-cpu) 0009 and Engine 0004.

## [Validation](../../../../glossary.md#term-validation) evidence

- The one authoritative uncached [CPU](../../../../glossary.md#term-cpu) validation completed successfully on 2026-08-26 with
  `./gradlew :backends:cpu:test --rerun-tasks`: 485 tests, 3 skipped, 0 failures, and 0 errors;
  all 22 requested Gradle tasks executed.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Focused implementation runs covered capability, lowering, generated bodies, deterministic
  evidence, preparation/finalization, composite binding/execution, package inventory, and retained
  existing-family preparation behavior before the authoritative run.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding); [run glossary](../../../../glossary.md#term-run))</span>
- Five fresh isolated OpenJDK 26.0.1 fixed-[heap](../../../../glossary.md#term-heap) [forks](../../../../glossary.md#term-fork-meanings) used `--enable-preview`,
  `--add-modules jdk.incubator.vector`, `-Xms1g`, and `-Xmx1g`. Each fork passed all eight
  generated/direct targets and controls with five randomized warmup rounds, nine measured rounds,
  and adaptive batches of at least 25 ms. Every per-fork ratio was at most 1.15. Aggregate medians
  were: dense FLOAT32 0.994777342, grouped FLOAT64 1.057987982, depthwise BFLOAT16 0.942743986,
  general-layout mixed carriers 0.966266916, fused ADD+RELU 0.994248242, parallel FLOAT32
  0.979604459, materialized split pair 0.993886270, and unchanged FLOAT32 ADD control 0.994390269.
  Accepted forks: 5; rejected forks: 0.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [targets glossary](../../../../glossary.md#term-target); [warmup glossary](../../../../glossary.md#term-warmup); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [layout glossary](../../../../glossary.md#layout); [fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
- Retained evidence is under
  `/private/tmp/synaptik-cpu-0008-retained-evidence-20260826`. It contains the five raw fork CSVs,
  aggregate summary, eight deterministic generated classes, compatibility bytes, specializations,
  descriptors, lowering manifests, member reports, SHA-256 checksums, and complete retained<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
  `javap -c -v` reports (2,629 lines). The [manifest](../../../../glossary.md#term-manifest)-file [SHA-256](../../../../glossary.md#term-sha-256) is
  `c501650e90412ce5f7664fbfa1f6fd4c42f695ed171f05ffcd838ee871d9ba8b`.
- Class-File inspection found one direct static `invoke` method per final field-free class and no
  constructor, allocation opcode, invokedynamic, dynamic constant, bootstrap method, reflection,
  boxing, collection/map dispatch, or Synaptik-owned hot helper call. The only calls in retained
  bodies are exact primitive <span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [constant glossary](../../../../glossary.md#term-constant); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>`Math.max`, [BFLOAT16](../../../../glossary.md#term-bfloat16) bit conversion, and typed [`MemorySegment`](../../../../glossary.md#term-memorysegment)
  access where selected.
- The mandatory separate documentation-focused review finalized every affected Conv2d and
  composite-executable Javadoc, the directly relevant CPU package summaries, the CPU backend
  guide, this task, the CPU master plan, the roadmap, and the glossary's existing Conv2d/current-
  schema entries. It found no architecture-contract, shared API, Gradle, conformance, integration,
  or additional terminology owner requiring change.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Javadoc glossary](../../../../glossary.md#term-javadoc); [backend glossary](../../../../glossary.md#backend); [schema glossary](../../../../glossary.md#term-schema); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [conformance glossary](../../../../glossary.md#term-conformance); [owner glossary](../../../../glossary.md#term-owner))</span>
- Final CPU Javadoc generation, local Markdown link/anchor/fence validation, terminology/status
  checks, retained-evidence digest verification, generated-document inspection, exact scope and
  staging checks, and <span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed. The successful CPU Java and performance suites
  were not repeated because the documentation pass changed no executable behavior.
- Documentation context `01a03e0b-d36d-7063-9adc-6f614b8ccc87` ran
  `./gradlew :backends:cpu:javadoc`: all 11 tasks completed successfully (2 executed, 9
  up-to-date), with only the two expected incubating Vector API warnings and no Javadoc content
  warning. The targeted local checker reported <span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api))</span>`Markdown validation passed for 5 files`; all
  manifest entries reported `OK`; final scope was exactly 37 paths; staged path count was zero;
  and tracked plus untracked whitespace checks produced no diagnostic.
- Glossary revision was necessary because its existing Conv2d entry still described concrete CPU
  execution as planned and its generated-artifact entries stopped at schema 50. The review
  corrected those existing definitions without introducing a new reusable term or glossary owner.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>

## Implementation notes

- Added the four planned family [owners](../../../../glossary.md#term-owner) (`CpuConv2dIr`, `CpuConv2dLowering`, `CpuConv2dEmitter`, and
  `CpuConv2dReferenceKernel`) plus the sole permitted `CpuPreparedExecutableSequence` [owner](../../../../glossary.md#term-owner).
- Direct generated grouped NCHW traversal supports the exact admitted FLOAT64, FLOAT32, BFLOAT16,
  ordered-promotion, intrinsic-bias, static-layout, grouping, padding, stride, dilation, empty,
  carrier, and scalar/parallel-scalar matrix. FLOAT32/BFLOAT16 hot work remains in primitive
  FLOAT32 locals, and selected segment accesses use frozen typed layouts.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [bias glossary](../../../../glossary.md#term-bias); [layout glossary](../../../../glossary.md#layout); [padding glossary](../../../../glossary.md#padding); [stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [carrier glossary](../../../../glossary.md#term-storage-carrier); [scalar glossary](../../../../glossary.md#term-scalar); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Legal external FLOAT64/FLOAT32 ADD and ADD+RELU chains retain represented operation boundaries
  in one direct Conv2d artifact. Other admitted activation/clamp forms, and requested intermediate
  publications that make fusion illegal, use the exact tagged two-unit materialized suffix when
  the complete pointwise suffix is independently supported; unsupported suffixes fail closed.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [artifact glossary](../../../../glossary.md#term-artifact); [activation glossary](../../../../glossary.md#term-activation); [fusion glossary](../../../../glossary.md#term-fusion); [pointwise glossary](../../../../glossary.md#term-pointwise); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Split analysis declares one ordinary intermediate before shared assignment, retains complete
  unit-local route/boundary/access/carrier/range/worker/geometry facts, and finalizes exactly two
  ordered cache-compatible artifacts. The composite validates child and cross-unit spans and
  aliases before either child can execute, declares external reads plus intermediate/final writes
  once, and performs no direct <span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>[`RunState`](../../../../glossary.md#run-state--runstate) validity or [publication](../../../../glossary.md#term-publication) mutation.
- Generator schema advanced exactly once to version 51. Existing artifacts are safe misses and
  deterministic regeneration produces identical bytes.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Final scope is 19 production/Javadoc paths, including exactly five new production types; 13 CPU
  test paths; 5 documentation/planning paths; and 37 total paths. No shared module, architecture, Gradle,
  conformance, integration, later-task specification, or public API path changed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [Javadoc glossary](../../../../glossary.md#term-javadoc); [CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Completion summary

[CPU](../../../../glossary.md#term-cpu) 0008 is complete. It delivers direct generated grouped [NCHW](../../../../glossary.md#term-nchw) [Conv2d](../../../../glossary.md#convolution--conv2d) for [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), and
[BFLOAT16](../../../../glossary.md#term-bfloat16) with intrinsic optional [bias](../../../../glossary.md#term-bias), groups/depthwise geometry, explicit [padding](../../../../glossary.md#padding)/[stride](../../../../glossary.md#term-stride)/[dilation](../../../../glossary.md#term-dilation),
typed arrays/segments/mixed carriers, [scalar](../../../../glossary.md#term-scalar) or parallel-scalar complete-output-cell ranges, legal
ADD and ADD-plus-RELU epilogues, and the sole tagged two-unit materialized suffix. The composite
uses one ordinary intermediate [buffer](../../../../glossary.md#term-buffer) and presents one [atomic](../../../../glossary.md#resource-transaction-and-rollback) [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) access boundary
to [Runtime](../../../../glossary.md#term-runtime) while retaining strict child order and no direct validity mutation. Generator [schema](../../../../glossary.md#term-schema) is
51. Implementation, retained structural/performance evidence, [Javadoc](../../../../glossary.md#term-javadoc), [backend](../../../../glossary.md#backend)-guide, glossary,
[planning](../../../../glossary.md#term-planning), and final documentation [validation](../../../../glossary.md#term-validation) gates passed without [widening](../../../../glossary.md#term-widening) shared contracts.

CPU 0008A is now the sole [Ready](../../../../glossary.md#term-ready) CPU task. CPU 0008B through CPU 0008E remain Draft in their
established order.

Status: Complete
