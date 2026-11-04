# Task 0008A: Portable Channels-First Dimensional [Convolution](../../../../glossary.md#term-channels-first-convolution-layers) Closure

## Reader context

This task records two different dimensional-convolution routes. `CpuConv1dCompositionLowering`
recognizes the literal two-expansion, Conv2d, squeeze graph and delegates equivalent
singleton-height access geometry to the existing Conv2d unit. Virtual intermediate graph values
receive no separate buffers; original rank-three carrier offsets and strides remain the storage
boundary. Conv3d instead uses `CpuConv3dIr`, `CpuConv3dLowering`, and `CpuConv3dEmitter` for a
first-class rank-five occurrence.

Here, `CpuConv3dReferenceKernel` supplies the independent semantic and performance oracle, while the
generated-form ledger and Class-File scans provide structural coverage evidence. Analysis declares
external buffers and zero workspace; cold binding validates carriers and workers. Runtime retains
publication ownership, and parallel work divides complete output cells rather than partial sums.

The two Conv1d expansion branches in the diagram are graph dependencies, not separate convolution
calls. Virtual rank edits preserve the checked storage mapping without allocating intermediate
values.

Current-topic navigation (not a replacement for the historical record):

- [current visible Conv1d composition](../../../../reference/tensor/convolution.md#grouped-ncw-conv1d-composition) — Shows why singleton-height rank edits surround Conv2d rather than define another operation kind.
- [current first-class NCDHW convolution meaning](../../../../reference/tensor/convolution.md#grouped-ncdhw-conv3d-expressions) — Keeps the rank-five geometry and first-class operation distinct from the Conv1d composition.

## Status

Complete

## Goal

Close the current portable channels-first forward-[convolution](../../../../glossary.md#term-channels-first-convolution-layers) [slice](../../../../glossary.md#slice) in the [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend). Validate
[NCW](../../../../glossary.md#term-ncw) Conv1d from its actual visible two-branch `EXPAND_DIMS(2) -> CONV2D -> SQUEEZE(2)` compiled
composition through CPU analysis, finalization, generated execution, and result [publication](../../../../glossary.md#term-publication), and
add direct grouped [NCDHW](../../../../glossary.md#conv3d--ncdhw) [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) execution with truthful resources and bounded [scalar](../../../../glossary.md#term-scalar) parallelism.

Keep both paths [rank](../../../../glossary.md#term-rank)-specific. Reuse the completed [Conv2d](../../../../glossary.md#convolution--conv2d) implementation only where source and
tests prove an identical CPU-private responsibility; do not create public or backend-generic
`ConvNd`, dynamic-rank machinery, or general [partition](../../../../glossary.md#partition)-directed-acyclic-[graph](../../../../glossary.md#graph) ([DAG](../../../../glossary.md#term-directed-acyclic-graph-dag)) handling.

## Motivation and [mental model](../../../../glossary.md#term-mental-model)

The repository exposes two different [rank](../../../../glossary.md#term-rank)-extension mechanisms:

```text
Conv1d convenience                       first-class Conv3d

NCW input ---- EXPAND_DIMS(2) --\        NCDHW input --\
                                  CONV2D                 CONV3D -> NCDHW result
NCW weight --- EXPAND_DIMS(2) --/        5-D weight ----/
                                  |
                             SQUEEZE(2) -> NCW result
```

Conv1d must remain those ordinary visible [graph](../../../../glossary.md#graph) [nodes](../../../../glossary.md#node). [CPU](../../../../glossary.md#term-cpu) analysis may recognize exactly that
family-owned [topology](../../../../glossary.md#term-topology) and keep its [singleton](../../../../glossary.md#term-singleton) [views](../../../../glossary.md#term-view-meanings) virtual, but it must not rewrite the [compiled
graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) or claim general [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [fusion](../../../../glossary.md#term-fusion). [Conv3d](../../../../glossary.md#conv3d--ncdhw) is already one ordinary flat [Compiler](../../../../glossary.md#term-compiler) node, so CPU owns
rank-five [validation](../../../../glossary.md#term-validation), [lowering](../../../../glossary.md#lowering), resources, generation, [binding](../../../../glossary.md#term-binding), and direct execution.

## [Scope](../../../../glossary.md#term-scope)

### Exact [NCW](../../../../glossary.md#term-ncw) [Conv1d composition](../../../../glossary.md#conv1d-composition)

- Admit one [CPU](../../../../glossary.md#term-cpu)-owned [partition](../../../../glossary.md#partition) containing exactly four semantic occurrences: two independent
  `EXPAND_DIMS` occurrences at literal [axis](../../../../glossary.md#term-axis) `2`, one [`CONV2D`](../../../../glossary.md#convolution--conv2d) consuming both expanded values and
  optional original rank-one bias, and one <span data-glossary-navigation="true"> ([rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`SQUEEZE` at literal axis `2` consuming the Conv2d
  result. Permit either deterministic topological order of the two independent expansions; admit
  no other node, edge, output, or fan-out.<span data-glossary-navigation="true"> ([topological order glossary](../../../../glossary.md#term-topological-order); [node glossary](../../../../glossary.md#node); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- Require the external input [Shape](../../../../glossary.md#shape) `[N, C_in, W]`, external [weight](../../../../glossary.md#term-weight) Shape
  `[C_out, C_in/groups, K_w]`, optional [bias](../../../../glossary.md#term-bias) `[C_out]`, and final output
  `[N, C_out, W_out]`. Verify the visible intermediate Shapes
  `[N, C_in, 1, W]`, `[C_out, C_in/groups, 1, K_w]`, and
  `[N, C_out, 1, W_out]` exactly.
- Require the Conv2d attributes to be the exact [Model](../../../../glossary.md#term-model) mapping
  `(strideHeight=1, strideWidth=stride, paddingHeight=0, paddingWidth=padding,
  dilationHeight=1, dilationWidth=dilation, groups=groups)` and require both rank edits to carry
  the exact literal axis. Similar-looking arbitrary rank-edit/Conv2d graphs fail this recognizer.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [carry glossary](../../../../glossary.md#term-carry); [graphs glossary](../../../../glossary.md#graph))</span>
- Require all three intermediate values to be single-use, unpublished, and virtual according to
  the complete memory requirements. Declare only original input, original weight, optional bias,
  and final NCW output buffers. Declare zero workspace and no intermediate materialization.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [materialization glossary](../../../../glossary.md#materialization))</span>
- Synthesize only the CPU-private singleton-height access geometry needed by the existing direct
  Conv2d unit. Preserve the original rank-three carrier bases, offsets, spans, and non-negative
  strides; a singleton-axis stride has no storage extent and must not trigger copying. Write the
  final result directly through the rank-three output carrier using the equivalent singleton-
  height rank-four address interpretation.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [carrier glossary](../../../../glossary.md#term-storage-carrier); [strides glossary](../../../../glossary.md#term-stride); [extent glossary](../../../../glossary.md#term-extent))</span>
- Keep the exact captured/compiled graph visible in fixtures and assertions. Recognition is a
  bounded lowering fact, not a Model operation, Compiler rewrite, Planning rule, or public
  capability called <span data-glossary-navigation="true"> ([compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [lowering glossary](../../../../glossary.md#lowering); [operation glossary](../../../../glossary.md#operation); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning))</span>`CONV1D`.
- Exercise the composition from complete [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) analysis through resource assignment,
  finalization, cold binding, scalar and eligible parallel-scalar invocation, and final NCW
  result. No Engine-facing claim is made before Engine 0004.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [scalar glossary](../../../../glossary.md#term-scalar); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Reuse the existing `CpuConv2dIr`, emitter, reference semantics, cache identity, and execution
  body when the synthesized rank-four geometry is indistinguishable from direct singleton-height
  Conv2d. Do not create a second Conv1d numerical kernel or silently materialize either view.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [kernel glossary](../../../../glossary.md#kernel); [view glossary](../../../../glossary.md#term-view-meanings))</span>

### Exact admitted grouped [NCDHW](../../../../glossary.md#conv3d--ncdhw) [Conv3d](../../../../glossary.md#conv3d--ncdhw) [occurrence](../../../../glossary.md#term-occurrence)

- Add truthful `CpuCapabilityProvider` [admission](../../../../glossary.md#term-admission) for exactly `Conv3dKind.CONV3D` with
  `Conv3dAttrs`, ordered `[input, weight]` or `[input, weight, bias]`, and one output.
- Admit only fully static [Shapes](../../../../glossary.md#shape) with present resolved [layouts](../../../../glossary.md#layout). Require input
  `[N, C_in, D, H, W]`, [weight](../../../../glossary.md#term-weight) `[C_out, C_in/groups, K_d, K_h, K_w]`, optional [bias](../../../../glossary.md#term-bias) `[C_out]`,
  and output `[N, C_out, D_out, H_out, W_out]` with the exact [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) formulas.
- Require positive static kernel extents, stride, dilation, and groups; non-negative symmetric
  padding; checked channel divisibility; exact <span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [padding glossary](../../../../glossary.md#padding); [channel glossary](../../../../glossary.md#term-channel))</span>`weightChannelsPerGroup * groups == C_in`; exact
  bias length; exact output geometry; matching gradient-request metadata; and an injective output.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [injective glossary](../../../../glossary.md#term-injective))</span>
- Admit FLOAT64, FLOAT32, and BFLOAT16 boundaries in any ordered combination allowed by Model
  promotion. Reject BOOL, INT32, INT64, every unresolved Shape/layout, and any future floating type
  until an explicit semantic and CPU route exists.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Support groups `1`, intermediate groups, and depthwise geometry; zero or empty legal output
  domains; typed primitive arrays, native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment) values, and mixed carriers; and
  arbitrary legal resolved non-negative offsets and strides through the general scalar form.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>

### [Conv3d](../../../../glossary.md#conv3d--ncdhw) semantic and numerical contract

- Implement grouped [NCDHW](../../../../glossary.md#conv3d--ncdhw) cross-correlation without reversing the stored [kernel](../../../../glossary.md#kernel).
- For each complete output cell, visit its contiguous group-local input channels, kernel depth,
  kernel height, then kernel width in increasing logical order. Padded coordinates are conceptual
  positive zero and still participate in ordinary multiplication, including multiplication by
  infinity.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- Initialize from optional intrinsic bias exactly once. FLOAT64 output accumulates in FLOAT64;
  FLOAT32 and BFLOAT16 output accumulate in FLOAT32; BFLOAT16 narrows once at the final store.
  An empty contraction starts from positive zero before optional bias.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- Preserve Model permission for reassociation and fused multiply-add without claiming cross-
  backend bitwise identity. For every generated specialization, the independent optimal clean
  Java implementation and emitted code must use the same semantic algorithm, loop/dataflow shape,
  rounding boundaries, address progression, validation envelope, and avoidable-overhead profile.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [reassociation and fused multiply-add glossary](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor); [backend glossary](../../../../glossary.md#backend); [identity glossary](../../../../glossary.md#term-identity-meanings); [specialization glossary](../../../../glossary.md#term-specialization); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape); [rounding glossary](../../../../glossary.md#term-rounding); [validation glossary](../../../../glossary.md#term-validation))</span>
- Cover NaN, infinity, signed zero, conceptual padding, mixed promotion, bias, empty contraction,
  groups/depthwise, stride, dilation, padding, and arbitrary legal layout behavior explicitly.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero); [padding glossary](../../../../glossary.md#padding); [stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [layout glossary](../../../../glossary.md#layout))</span>

### [Conv3d](../../../../glossary.md#conv3d--ncdhw) [epilogue](../../../../glossary.md#term-epilogue) boundary

- The only Conv3d epilogue in this task is the optional intrinsic [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [bias](../../../../glossary.md#term-bias) carried by the
  `CONV3D` [occurrence](../../../../glossary.md#term-occurrence).
- A Conv3d-led external ADD, activation, clamp, publication fan-out, or any other suffix is not
  directly fused and does not select CPU 0008's Conv2d-only materialized-suffix form. Such a
  multi-node Conv3d partition fails this task's specialized lowering closed.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [publication glossary](../../../../glossary.md#term-publication); [fan-out glossary](../../../../glossary.md#term-fan-out); [fused glossary](../../../../glossary.md#specialization-and-fusion); [CPU glossary](../../../../glossary.md#term-cpu); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [node glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering))</span>
- Do not generalize `CpuConv2dIr.Epilogue`, `CpuConv2dLowering`, or
  `CpuPreparedExecutableSequence` into an all-rank convolution policy. CPU 0008B owns general DAG
  unit decomposition/materialized fallback, and CPU 0008C owns later closed typed epilogue
  recognition.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>

### Resources, generation, and bounded parallelism

- Direct Conv3d declares exactly input, weight, optional intrinsic bias, and output buffers with
  checked spans, sizes, access modes, and alignments. It declares zero workspace, zero
  materializations, one computation unit, one portable route plan, one generated artifact, and
  one prepared executable.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [weight glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias); [buffers glossary](../../../../glossary.md#term-buffer); [artifact glossary](../../../../glossary.md#term-artifact); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>
- No im2col, packing, reorder, partial-sum, combine, provider, native, or persistent prepared
  resource is permitted. If correctness or performance requires one, stop and replan.<span data-glossary-navigation="true"> ([im2col glossary](../../../../glossary.md#term-im2col); [prepared resource glossary](../../../../glossary.md#prepared-resource--preparedresource))</span>
- Generated [scalar](../../../../glossary.md#term-scalar) work owns a [half-open](../../../../glossary.md#term-half-open-interval) range of complete output cells in logical
  `N, C_out, D_out, H_out, W_out` order. Decode each cell once, then run the complete group-local
  contraction. Empty output invokes no generated entry and submits no worker work.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- Parallel-scalar execution partitions only those complete output-cell ranges through the current
  CPU worker owner and existing bounded cold heuristic. A cell has exactly one writer; workers
  share no partial state, atomics, workspace, combine phase, or nested submission. Scalar and
  parallel-scalar use the same generated body and per-cell algorithm.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Cold binding validates exact carriers, native accessibility/order/alignment, spans, output
  writability, worker capacity, and every output/input overlap before the first write or worker
  submission. Input/input sharing is allowed where all reads remain valid; every output/input
  overlap is rejected.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Add one [rank](../../../../glossary.md#term-rank)-specific [immutable](../../../../glossary.md#term-immutable) `CpuConv3dIr`, one focused lowerer, one focused emitter, and one
  independent direct Java reference owner. Advance the generator schema exactly once and make all
  older envelopes safe misses.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Keep generated classes deterministic, final, field-free, and constructor-free, with one typed
  static entry and primitive cold-supplied geometry/range arguments. Generated hot code contains
  no graph/layout/operation lookup, generic dispatch, route selection, cache access, allocation,
  boxing, reflection, collections, synchronization, or Synaptik numerical helper calls.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [graph glossary](../../../../glossary.md#graph); [layout glossary](../../../../glossary.md#layout); [operation glossary](../../../../glossary.md#operation); [lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [cache glossary](../../../../glossary.md#term-cache); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>

### Proved reuse boundary

- Reuse existing `CpuAccessPlan`, carrier bindings, specialization/cache machinery, resource
  assignment, finalization, prepared-executable binding, worker orchestration, and complete-output-
  cell range infrastructure because those responsibilities are already rank-neutral in source.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [bindings glossary](../../../../glossary.md#term-binding); [specialization glossary](../../../../glossary.md#term-specialization); [cache glossary](../../../../glossary.md#term-cache); [rank glossary](../../../../glossary.md#term-rank))</span>
- Reuse the Conv2d numerical unit for exact Conv1d composition because the Model mapping proves a
  singleton-height Conv2d with identical semantic work.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv1d composition glossary](../../../../glossary.md#conv1d-composition); [Model glossary](../../../../glossary.md#term-model); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Keep Conv3d IR, lowering, emitter, and oracle rank-specific. Extract a new CPU-private shared
  component from rank two and rank three only if both completed implementations otherwise contain
  the same invariant and algorithmic responsibility, the extraction reduces duplication, and its
  name does not imply arbitrary rank. Record the proof and both consumers in <span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [oracle glossary](../../../../glossary.md#term-test-oracle); [CPU glossary](../../../../glossary.md#term-cpu); [invariant glossary](../../../../glossary.md#term-invariant))</span>`Local decisions`.
- Do not share merely because fields have similar names. No universal convolution geometry,
  epilogue, IR, rank loop, public facade, registry, or generic helper is planned.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [epilogue glossary](../../../../glossary.md#term-epilogue); [facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry))</span>

### Generated-code and performance evidence

- Freeze a machine-readable task-owned generated-form ledger before final measurement. A form is
  distinct whenever result/accumulator type, intrinsic-bias control flow, carrier access topology,
  address-progression body, or another emitted hot-loop/dataflow fact changes class bytes or
  executed instructions. Every distinct hot form admitted by this task must have a ledger row;
  any unledgered form is either added and evidenced within scope or rejected closed.<span data-glossary-navigation="true"> ([accumulator type glossary](../../../../glossary.md#accumulator-type); [bias glossary](../../../../glossary.md#term-bias); [carrier glossary](../../../../glossary.md#term-storage-carrier); [topology glossary](../../../../glossary.md#term-topology); [dataflow glossary](../../../../glossary.md#term-dataflow); [scope glossary](../../../../glossary.md#term-scope))</span>
- The ledger must cover every new Conv3d emitted form and every Conv1d composition form whose
  lowering changes the previously evidenced Conv2d entry or invocation shape. Cache keys that
  generate byte-identical bodies may share a row only when an automated equivalence check proves
  identical bytes and typed entry semantics; record every covered key in that row.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Conv1d composition glossary](../../../../glossary.md#conv1d-composition); [lowering glossary](../../../../glossary.md#lowering); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [shape glossary](../../../../glossary.md#shape); [Cache glossary](../../../../glossary.md#term-cache))</span>
- At minimum the ledger includes unbiased and biased Conv1d; grouped and depthwise Conv1d;
  FLOAT64, FLOAT32, and BFLOAT16 result/accumulator forms; dense heap, all-segment, selected mixed-
  carrier, and arbitrary-layout access forms; unbiased and biased Conv3d; groups <span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [accumulator glossary](../../../../glossary.md#accumulator-type); [heap glossary](../../../../glossary.md#term-heap); [layout glossary](../../../../glossary.md#layout))</span>`1`, intermediate
  groups, and depthwise Conv3d; padded/strided/dilated Conv3d; and scalar plus parallel-scalar
  orchestration. Add an unchanged direct Conv2d control and its materialized-suffix control.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- For every retained generated class in the ledger, save deterministic bytes and checksum, parse
  the complete Class-File, and inspect complete <span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum))</span>`javap -c -v` output. Automated member/reference
  scans must prove the typed entry, expected loop/address/multiply/add/conversion shape, exact
  allowed primitive/JDK calls, and absence of fields, constructors, method handles,<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
  `invokedynamic`, dynamic constants, bootstrap methods, allocation, boxing, reflection,
  collection/map dispatch, graph/layout/operation references, hidden bridge/interpreter calls,
  and Synaptik-owned per-cell or per-contribution helpers.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [dispatch glossary](../../../../glossary.md#term-dispatch); [graph glossary](../../../../glossary.md#graph); [operation glossary](../../../../glossary.md#operation))</span>
- Every ledger row representing a distinct hot specialized form must compare generated execution
  with its own optimal clean Java implementation in five fresh accepted fixed-heap Java 26 forks.
  Each fork uses deterministic inputs, randomized generated/direct order, at least five warmup
  batches, nine measured rounds, adaptive batches of at least 25 ms, and semantic equality before
  timing. Every per-fork ratio and aggregate median must be <span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [warmup glossary](../../../../glossary.md#term-warmup); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [median glossary](../../../../glossary.md#term-median))</span>`<= 1.15x`.
- Reject a whole fork only before measurement for a recorded environment, classpath, or control
  failure. Discard no measured ratio. Retain per-row ratios, medians, environment, source and class
  digests, accepted/rejected forks, inspection reports, and one evidence-manifest digest. Claims
  remain bounded to those exact forms and environment.<span data-glossary-navigation="true"> ([classpath glossary](../../../../glossary.md#term-classpath); [manifest glossary](../../../../glossary.md#term-manifest); [digest glossary](../../../../glossary.md#term-digest))</span>

### Documentation and [Javadoc](../../../../glossary.md#term-javadoc)

- Update meaningful Javadoc for every changed Java type/method contract, including inputs,
  returns, failures, ownership, resources, range semantics, and nullability where applicable.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Update `docs/backend-guide/cpu-backend.md` with current Conv1d-composition and direct Conv3d
  behavior, resources, epilogue boundary, schema, validation, and bounded evidence. Do not write
  planned Engine or cross-backend behavior as current.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [epilogue glossary](../../../../glossary.md#term-epilogue); [schema glossary](../../../../glossary.md#term-schema); [validation glossary](../../../../glossary.md#term-validation); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend))</span>
- Update `docs/glossary.md` only if the implementation introduces or changes a reusable term;
  otherwise record a reasoned no-change conclusion.
- Finalize this task, the CPU master plan, and roadmap from actual evidence. CPU 0008B remains
  Draft and becomes next only after this task is Complete.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Any public, Model, Compiler, Planning, Prepare, Runtime, Engine, Backend Contract, Config, Trace,
  or backend-generic <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace))</span>`ConvNd` [API](../../../../glossary.md#term-application-programming-interface-api); dynamic [rank](../../../../glossary.md#term-rank); or a new [operation](../../../../glossary.md#operation) kind.
- Symbolic/dynamic [Shape](../../../../glossary.md#shape) or unresolved-[layout](../../../../glossary.md#layout) [CPU](../../../../glossary.md#term-cpu) execution.
- [Conv3d](../../../../glossary.md#conv3d--ncdhw) backward, training, saved values, [gradients](../../../../glossary.md#term-gradient), or [Compiler](../../../../glossary.md#term-compiler) 0006C.
- General partition-DAG decomposition, fusion, recognition, profitability, materialization,
  representation reuse, or decision facts from CPU 0008B–0008E.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization))</span>
- External Conv3d epilogues and [generalization](../../../../glossary.md#term-generalization) of the [Conv2d](../../../../glossary.md#convolution--conv2d)-only two-unit sequence.
- [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[linear](../../../../glossary.md#term-linear-meanings), [pooling](../../../../glossary.md#term-pooling), attention, or [loss](../../../../glossary.md#loss) execution from CPU 0008F–0008I.
- Vector convolution, native/provider route, packing, reorder, im2col, autotuning, cache
  measurement, hardware-specific specialization, or shared architecture changes.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [im2col glossary](../../../../glossary.md#term-im2col); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [cache glossary](../../../../glossary.md#term-cache); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Runtime route selection, scheduling policy, graph inspection, validity-policy changes, or public
  end-to-end Engine integration.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Unrelated refactoring, dependency/build changes, backend conformance, integration-test
  infrastructure, or architecture-contract changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- `ARCHITECTURE.md`: Model semantic ownership; Compiler flat validated graph; Planning ownership;
  concrete-backend analysis, exact pre-assignment declarations, post-assignment finalization,
  generated-code ownership, Runtime prepared-recipe execution, and performance oracle discipline.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [graph glossary](../../../../glossary.md#graph); [Planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [Runtime glossary](../../../../glossary.md#term-runtime); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- `docs/architecture/current-architecture-plan.md`: explanatory compile/plan/prepare/run sequence
  and current module boundaries.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [module glossary](../../../../glossary.md#term-module))</span>
- `docs/planning/planning-guide.md`: Ready-task completeness, scope, validation tiers, evidence,
  and clean implementation/documentation context requirements.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [scope glossary](../../../../glossary.md#term-scope); [validation glossary](../../../../glossary.md#term-validation))</span>
- `docs/planning/backends/cpu/master-plan.md`: [CPU](../../../../glossary.md#term-cpu) package map and ordered 0008A–0008I frontier.
- Complete [Model](../../../../glossary.md#term-model) tasks 0025G and 0025H, [Compiler](../../../../glossary.md#term-compiler) task 0006B, and CPU task 0008.

## Architecture constraints

- Model remains the sole semantic owner. CPU consumes the exact Conv1d composition and Conv3d
  contract without redefining Shapes, promotion, numerical meaning, or operation identity.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu); [Conv1d composition glossary](../../../../glossary.md#conv1d-composition); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Shapes glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Compiler retains one visible flat graph. CPU-private Conv1d recognition changes neither captured
  nodes nor publication identity.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [nodes glossary](../../../../glossary.md#node); [publication glossary](../../../../glossary.md#term-publication))</span>
- CPU analysis/lowering owns family legality, exact resources, route choice, specialization, and
  generated artifacts before Runtime. Finalization only resolves already declared assignments.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [artifacts glossary](../../../../glossary.md#term-artifact); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Runtime invokes prepared executables and owns validity/publication transitions; it does not
  inspect operations, select a route, allocate undeclared resources, or decompose a graph.
- Generated code follows the optimal clean Java oracle's specialized loop/dataflow and
  avoidable-overhead profile. Performance claims require the complete task ledger and five-fork
  evidence above.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [dataflow glossary](../../../../glossary.md#term-dataflow); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- No dependency direction, shared contract, module boundary, public API, or architecture rule may
  change. If one is required, stop and report the conflict.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — exact [occurrence](../../../../glossary.md#term-occurrence) capability [admission](../../../../glossary.md#term-admission).
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — [immutable](../../../../glossary.md#term-immutable) [rank](../../../../glossary.md#term-rank)-specific code [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — exact [topology](../../../../glossary.md#term-topology) and geometry [owners](../../../../glossary.md#term-owner).
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct Class-File emission.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent clean Java [oracle](../../../../glossary.md#term-test-oracle).
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema)/[compatibility identity](../../../../glossary.md#identity-meanings-and-algebraic-identities).
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — existing selected-route facts.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — declaration and finalization facts.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — existing cold [binding](../../../../glossary.md#term-binding)/execution.

Packages added or changed:

- No package is added. Only the existing [CPU](../../../../glossary.md#term-cpu)-private packages above may change.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv3dIr` — immutable rank-five geometry,
  type, access, bias, algorithm, and code-shaping identity.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv1dCompositionLowering` — exact
  four-node/two-branch recognition and singleton-view folding; not a general graph lowerer.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [singleton glossary](../../../../glossary.md#term-singleton); [view glossary](../../../../glossary.md#term-view-meanings); [graph glossary](../../../../glossary.md#graph))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLowering` — exact static NCDHW
  geometry, boundaries, resources, and direct-family lowering.<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [lowering glossary](../../../../glossary.md#lowering))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConv3dEmitter` — direct typed
  complete-output-cell Class-File body.
- `io.github.pho001.synaptik.backend.cpu.internal.reference.CpuConv3dReferenceKernel` — independent
  optimal clean Java semantic and performance oracle.

Tests mirror production packages. The generated-form ledger and performance harness remain test-
only evidence, not production tuning infrastructure.

## Affected files

Expected production and [Javadoc](../../../../glossary.md#term-javadoc) paths:

- `CpuCapabilityProvider.java` and [CPU](../../../../glossary.md#term-cpu) package Javadoc;
- `internal/ir/CpuPortableKernelIr.java`, new `CpuConv3dIr.java`, and [IR](../../../../glossary.md#term-intermediate-representation-ir) package Javadoc;
- `internal/lowering/CpuPartitionLowering.java`, new `CpuConv1dCompositionLowering.java`, new
  `CpuConv3dLowering.java`, and [lowering](../../../../glossary.md#lowering) package Javadoc;
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new `CpuConv3dEmitter.java`, and emitter
  package Javadoc;
- `internal/reference/CpuConv3dReferenceKernel.java` and reference package Javadoc;
- `internal/cache/CpuGeneratorSchema.java` and existing specialization/artifact compatibility
  owners only where rank-five identity requires them;<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [rank glossary](../../../../glossary.md#term-rank); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- existing portable-route, preparation-plan, preparer, finalizer, prepared-executable, and package
  Javadoc owners only where the new rank-specific IR, geometry, declarations, or typed fields
  require integration; and
- an optional single new narrowly named CPU-private common component only after the reuse proof in
  this specification is recorded. It must have exactly the two rank-specific consumers.

Expected test/resource paths:

- focused [Conv1d composition](../../../../glossary.md#conv1d-composition) [topology](../../../../glossary.md#term-topology)/lowering/end-execution tests;
- focused Conv3d capability, IR, lowering, reference, generated-kernel, preparation/finalization,
  executable/binding, schema/cache, package-inventory, evidence, and performance tests/resources;<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [kernel glossary](../../../../glossary.md#kernel); [binding glossary](../../../../glossary.md#term-binding); [schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache))</span>
- negative topology, rank/axis/attribute, descriptor, geometry, type, layout, publication/fan-out,
  resource, carrier/span/alignment, alias, and validation-before-write matrices; and<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [publication glossary](../../../../glossary.md#term-publication); [fan-out glossary](../../../../glossary.md#term-fan-out); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alignment glossary](../../../../glossary.md#term-alignment); [alias glossary](../../../../glossary.md#aliasing-and-arena); [validation glossary](../../../../glossary.md#term-validation); [matrices glossary](../../../../glossary.md#term-matrix-meanings))</span>
- unchanged direct and split [Conv2d](../../../../glossary.md#convolution--conv2d) controls.

Expected documentation/[planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`;
- `docs/glossary.md` only if terminology changes;
- this task;
- `docs/planning/backends/cpu/master-plan.md`; and
- `docs/planning/roadmap.md`.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- 24 production/Javadoc paths, including exactly the five planned new rank-specific types and at
  most one optional proved-common private type;<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- 17 [CPU](../../../../glossary.md#term-cpu) test/resource paths;
- the five named documentation/[planning](../../../../glossary.md#term-planning) paths, with the glossary optional; and
- 46 total paths.

The ceiling exceeds the ordinary 12–18-file guardrail because one ordered cohesive capability
must validate an existing four-[node](../../../../glossary.md#node) composition and add the [rank](../../../../glossary.md#term-rank)-five family through the same
CPU [lifecycle](../../../../glossary.md#lifecycle), generated-code, and evidence boundaries. No [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) test source, shared
Java, [Gradle](../../../../glossary.md#term-gradle), architecture, [conformance](../../../../glossary.md#term-conformance), integration, later task specification, or vendor path is
permitted. If the work needs another production [owner](../../../../glossary.md#term-owner) or exceeds a category, stop and replan.

## Acceptance criteria

- The exact visible Conv1d [graph](../../../../glossary.md#graph), including two independent `EXPAND_DIMS(2)` producers, one mapped
  Conv2d occurrence, and one <span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`SQUEEZE(2)`, is asserted and executes correctly through analysis,
  assignment, finalization, binding, scalar/parallel invocation, and NCW output.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [scalar glossary](../../../../glossary.md#term-scalar); [NCW glossary](../../../../glossary.md#term-ncw))</span>
- Conv1d keeps all three intermediate graph values virtual and single-use, declares only external
  boundaries, uses zero workspace/materialization, and reuses the existing Conv2d generated unit.
  Wrong axes, mapping, ranks, edges, fan-out, publication, or added nodes fail closed.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [axes glossary](../../../../glossary.md#term-axis); [fan-out glossary](../../../../glossary.md#term-fan-out); [publication glossary](../../../../glossary.md#term-publication); [nodes glossary](../../../../glossary.md#node); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- CPU capability is true exactly for the admitted static resolved-layout grouped NCDHW Conv3d
  matrix and false for every excluded signature/type/descriptor/geometry case.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [layout glossary](../../../../glossary.md#layout); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [signature glossary](../../../../glossary.md#term-signature); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Unbiased and intrinsic-biased [Conv3d](../../../../glossary.md#conv3d--ncdhw) produce correct results for groups `1`, intermediate
  groups, depthwise, unit/non-unit stride/dilation, zero/non-zero padding, legal empty domains,
  all three result types, mixed promotion, arrays, segments, mixed carriers, and general layouts.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [padding glossary](../../../../glossary.md#padding))</span>
- Results preserve the exact current Model/Compiler Shape, grouping, traversal, promotion,
  conceptual-padding, bias, accumulation, special-value, and final-conversion contracts.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Shape glossary](../../../../glossary.md#shape); [bias glossary](../../../../glossary.md#term-bias))</span>
- Conv3d accepts intrinsic bias only. Every external suffix or multi-node Conv3d-led partition
  fails the specialized path and never selects the Conv2d-only fused/split epilogue machinery.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [fused glossary](../../../../glossary.md#specialization-and-fusion); [epilogue glossary](../../../../glossary.md#term-epilogue))</span>
- Analysis declares exact Conv3d buffers and zero workspace/materialization before assignment;
  finalization introduces no requirement. Empty output performs no call or worker submission.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer))</span>
- Scalar and bounded parallel-scalar execution own disjoint complete output cells, use identical
  per-cell generated work, and require no atomics, scratch, partials, or combine phase.
- All cold carrier/span/alignment/accessibility/overlap/worker validation completes before output
  mutation or submission. Failures expose no partially valid result.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [alignment glossary](../../../../glossary.md#term-alignment); [validation glossary](../../../../glossary.md#term-validation))</span>
- The generator schema advances once; old artifacts are safe misses; identical specializations
  regenerate byte-identical classes.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- The generated-form ledger is closed over every distinct new hot emitted form. Complete Class-
  File and decompilation inspection passes for every retained class with all required positive
  and forbidden-member/reference checks.<span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation))</span>
- Every distinct ledger hot form and both unchanged Conv2d controls pass all five per-fork and
  aggregate generated/direct <span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>`<= 1.15x` gates against optimal clean Java oracles. No measured
  sample is discarded.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Existing CPU families and the direct/fused/split Conv2d contract remain green and unchanged in
  meaning. No general DAG, external Conv3d epilogue, native/vector route, or later family appears.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [vector glossary](../../../../glossary.md#term-vector))</span>
- No public API, shared module, dependency, build, architecture, Runtime policy, conformance, or
  integration contract changes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Production/test types match the package map and all category/path ceilings.
- A separate clean documentation-focused agent finalizes Javadocs, CPU guide, glossary impact,
  planning status, links, and evidence in the same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- CPU 0008A is marked Complete only from actual evidence. CPU 0008B becomes the sole next Ready
  CPU task; no 0008B detailed specification is created here.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused tests for affected [owners](../../../../glossary.md#term-owner), for example:

```bash
./gradlew :modules:model:test --tests '*TensorConv1dExpressionTest' --tests '*Conv3d*'
./gradlew :modules:compiler:test --tests '*Conv3dCompilerTest'
./gradlew :backends:cpu:test \
  --tests '*Conv1d*' \
  --tests '*Conv2d*' \
  --tests '*Conv3d*' \
  --tests '*CpuCapabilityProviderTest' \
  --tests '*CpuPartitionPreparerTest' \
  --tests '*CpuPartitionFinalizerTest' \
  --tests '*CpuPreparedExecutableTest' \
  --tests '*CpuInternalPackageInventoryTest'
```

The [Model](../../../../glossary.md#term-model) and [Compiler](../../../../glossary.md#term-compiler) commands verify the unchanged visible composition and first-class [Conv3d](../../../../glossary.md#conv3d--ncdhw)
forward contracts; they do not authorize changes in those [modules](../../../../glossary.md#term-module). After [CPU](../../../../glossary.md#term-cpu) executable code
stabilizes, run exactly one authoritative uncached CPU suite:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Run the task-owned evidence harness in five fresh isolated Java 26 fixed-[heap](../../../../glossary.md#term-heap) [forks](../../../../glossary.md#term-fork-meanings) with the exact
protocol in [Scope](../../../../glossary.md#term-scope). Retain machine-readable XML counts, the form ledger, semantic results, ratios,
class bytes/checksums, complete Class-File and `javap -c -v` reports, forbidden-reference [scan](../../../../glossary.md#term-cumulative-scan),
accepted/rejected [fork](../../../../glossary.md#term-fork-meanings) [log](../../../../glossary.md#term-log), and [manifest](../../../../glossary.md#term-manifest) [digest](../../../../glossary.md#term-digest).

The final clean documentation-focused context receives the stable diff and exact Java/evidence
results. It does not repeat successful Java or performance runs unless it changes executable
behavior or records a concrete stale-evidence risk. After final [Javadoc](../../../../glossary.md#term-javadoc) edits it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also checks Markdown links/anchors, heading order, balanced fences, final newlines, terminology,
generated Javadoc pages, form-ledger/evidence digests, exact path/type ceilings, [schema](../../../../glossary.md#term-schema)/status/order,
absence of a 0008B task file, and empty staging.

Repository-wide validation remains deferred to CPU 0009 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci) because this
task changes no shared [dependency](../../../../glossary.md#term-dependency) or architecture boundary. [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) remains CPU 0009:
the current module has only its foundation and this task promises a CPU-private subset, not a
cross-backend result contract. [Engine](../../../../glossary.md#term-engine) 0004 owns public [lifecycle](../../../../glossary.md#lifecycle) integration. If implementation
changes those facts, stop and replan with proportionate validation.

## Dependencies

- Complete CPU 0008 and its direct grouped NCHW Conv2d, bounded Conv2d-only epilogue/split,
  generated-schema-51, resource, preparation/finalization, worker, binding, and evidence contracts.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [epilogue glossary](../../../../glossary.md#term-epilogue); [schema glossary](../../../../glossary.md#term-schema); [binding glossary](../../../../glossary.md#term-binding))</span>
- Complete [Model](../../../../glossary.md#term-model) 0025G, which defines Conv1d solely as the exact visible composition consumed here.
- Complete Model 0025H, which owns first-class Conv3d Shape, grouping, promotion, numerical, bias,
  and provenance semantics.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Shape glossary](../../../../glossary.md#shape); [bias glossary](../../../../glossary.md#term-bias); [provenance glossary](../../../../glossary.md#provenance))</span>
- Complete Compiler 0006B, which adopts one flat Conv3d forward node and independently validates
  its exact descriptors/obligations. Draft Compiler 0006C is not a forward prerequisite.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [node glossary](../../../../glossary.md#node); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Existing Planning, Prepare, Runtime, Backend Contract, Config, Trace, cache, and worker contracts
  are sufficient and remain unchanged.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [cache glossary](../../../../glossary.md#term-cache))</span>

## Follow-up tasks

- CPU 0008B: general partition-DAG computation-unit decomposition and bounded fusion; sole Ready
  CPU task after completion.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [Ready glossary](../../../../glossary.md#term-ready))</span>
- CPU 0008C: closed typed specialized-subgraph and epilogue recognition, including any selected
  external Conv3d epilogue.<span data-glossary-navigation="true"> ([epilogue glossary](../../../../glossary.md#term-epilogue))</span>
- [CPU](../../../../glossary.md#term-cpu) 0008D: bounded [fusion](../../../../glossary.md#term-fusion) profitability and typed cold decision facts.
- CPU 0008E: bounded multi-input [materialization](../../../../glossary.md#materialization) and representation reuse.
- CPU 0008F–0008I: [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[linear](../../../../glossary.md#term-linear-meanings), Pool2d, attention, and [loss](../../../../glossary.md#loss) execution.
- CPU 0009: generated-coverage/[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) [checkpoint](../../../../glossary.md#term-checkpoint).
- [Compiler](../../../../glossary.md#term-compiler) 0006C: separate [Conv3d](../../../../glossary.md#conv3d--ncdhw) [gradient](../../../../glossary.md#term-gradient) closure.
- [Engine](../../../../glossary.md#term-engine) 0004: public Conv1d/[Conv2d](../../../../glossary.md#convolution--conv2d)/Conv3d [lifecycle](../../../../glossary.md#lifecycle) integration.

## Architecture impact

Expected impact: None.

This task consumes existing concrete-[backend](../../../../glossary.md#backend) analysis, exact declaration, finalization, generated-
[artifact](../../../../glossary.md#term-artifact), worker, cold-[binding](../../../../glossary.md#term-binding), and prepared-execution boundaries. Exact Conv1d recognition and
[rank](../../../../glossary.md#term-rank)-specific [Conv3d](../../../../glossary.md#conv3d--ncdhw) are [CPU](../../../../glossary.md#term-cpu) implementation facts. If implementation requires a shared/public
contract, new [dependency](../../../../glossary.md#term-dependency), [Runtime](../../../../glossary.md#term-runtime) policy, or architecture rule, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the isolated implementation agent for Synaptik CPU task 0008A.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0008a-portable-channels-first-dimensional-convolution-closure.md.
Read the task's directly referenced Model, Compiler, CPU source/tests, generated-code/performance,
and documentation contracts. Implement exactly the Ready specification. Do not implement later
CPU tasks. Stop and report any architecture, shared-contract, or maximum-scope conflict.

After implementation and recorded Java/Class-File/performance validation, hand the stable diff
and evidence to a separate clean documentation-focused agent. That pass must follow
docs/developer-guide/documentation-rules.md, independently finalize Javadocs, the CPU guide,
glossary impact, planning evidence/status, and documentation checks in the same overall change,
and must not repeat successful Java/performance work unless executable behavior changes or a
concrete risk is recorded.

Do not commit or push unless the coordinating user explicitly authorizes it. Update this task's
local decisions, known limitations, validation evidence, implementation notes, completion summary,
and final status only from actual results.
```

## Local decisions

- Planning fixes the Conv1d lowering boundary to one exact four-node/two-branch topology. This is
  the smallest source-backed way to validate the visible composition without requiring CPU 0008B's
  general DAG decomposition.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [lowering glossary](../../../../glossary.md#lowering); [node glossary](../../../../glossary.md#node); [topology glossary](../../../../glossary.md#term-topology); [CPU glossary](../../../../glossary.md#term-cpu); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Planning fixes Conv3d epilogues to intrinsic bias only. CPU 0008's external ADD/RELU and two-unit
  materialized suffix are rank-two family contracts, while 0008B–0008C explicitly own their later
  generalization.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [bias glossary](../../../../glossary.md#term-bias); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [generalization glossary](../../../../glossary.md#term-generalization))</span>
- No new shared convolution component is pre-approved by resemblance alone. The implementation
  may add at most one narrowly named private component only after recording the concrete rank-two/
  rank-three responsibility and both consumers.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [rank glossary](../../../../glossary.md#term-rank))</span>
- Direct complete-output-cell traversal with zero workspace is the only admitted Conv3d algorithm.
  A packing, im2col, partial, combine, or native requirement is a replanning trigger.<span data-glossary-navigation="true"> ([im2col glossary](../../../../glossary.md#term-im2col))</span>

## Known limitations

- Conv1d CPU execution is limited to the exact current visible composition with virtual singleton
  views and no extra publication, fan-out, or suffix.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [singleton glossary](../../../../glossary.md#term-singleton); [views glossary](../../../../glossary.md#term-view-meanings); [publication glossary](../../../../glossary.md#term-publication); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- Conv3d CPU execution is forward-only, fully static, resolved-layout, scalar/parallel-scalar, and
  has intrinsic bias only.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [layout glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [bias glossary](../../../../glossary.md#term-bias))</span>
- Public Engine integration, cross-backend conformance, gradients, general DAG/epilogue handling,
  vector/native routes, and later heavy families remain the named follow-ups above.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [gradients glossary](../../../../glossary.md#term-gradient); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [epilogue glossary](../../../../glossary.md#term-epilogue); [vector glossary](../../../../glossary.md#term-vector))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean [planning](../../../../glossary.md#term-planning) context `01a03e57-b2a0-7001-ab6e-f566b85c4f1b` started from clean `main` at
  `e0494057`. It read `AGENTS.md`, the full architecture contract and current architecture plan,
  the planning guide and roadmap, the CPU master plan, complete CPU 0008, Model 0025G/0025H, and
  Compiler 0006B task specifications, plus directly relevant Model/Compiler/CPU source, tests,
  CPU/Tensor/Compile guides, documentation rules/profiles, and commit history.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [commit glossary](../../../../glossary.md#term-commit))</span>
- Source inspection confirmed Conv1d is exactly the visible two-expansion/Conv2d/squeeze graph;
  Conv3d is one flat forward node; current CPU lowering has one Conv2d-first special path and an
  otherwise straight-line boundary; Conv2d external epilogue and sole two-unit materialized split
  are rank-two-specific; direct Conv2d uses complete-output-cell ranges, zero workspace, schema 51,
  direct generated scalar code, and an independent clean Java oracle. These facts determine this
  task's exact Conv1d recognizer and intrinsic-bias-only Conv3d epilogue boundary.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [graph glossary](../../../../glossary.md#graph); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [node glossary](../../../../glossary.md#node); [lowering glossary](../../../../glossary.md#lowering); [epilogue glossary](../../../../glossary.md#term-epilogue); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [schema glossary](../../../../glossary.md#term-schema); [scalar glossary](../../../../glossary.md#term-scalar); [oracle glossary](../../../../glossary.md#term-test-oracle); [bias glossary](../../../../glossary.md#term-bias))</span>
- Planning-only checks passed: [canonical](../../../../glossary.md#term-canonical-form) 20 task headings in order; exactly one [CPU](../../../../glossary.md#term-cpu) task-table
  [`Ready`](../../../../glossary.md#term-ready) row. At that planning checkpoint, 0008A was linked and Ready while 0008B was unlinked,
  Draft, next, and without a task file; both
  introduced local Markdown links resolve; ten balanced fences; terminal line feed; no carriage
  returns or trailing whitespace; exactly the task, CPU master plan, and roadmap changed; empty
  staging; and tracked <span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`git diff --check` plus new-file whitespace checks. No Java, Gradle,
  Javadoc, Class-File, or performance command was run for this planning-only change.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [Javadoc glossary](../../../../glossary.md#term-javadoc); [run glossary](../../../../glossary.md#term-run))</span>
- Documentation context `01a03e87-9b14-7003-a6e9-e984bcf17144` applied the General,
  API-and-Javadoc, Developer-guide, Backend-guide, Planning, and Example profiles. It changed
  Javadoc in <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Backend glossary](../../../../glossary.md#backend))</span>`CpuGeneratorSchema`, `CpuConv3dIr`, `CpuPartitionLowering`,
  `CpuPartitionPreparationPlan`, and `CpuConv3dLowering`; updated the CPU backend guide; and
  finalized this task, the CPU master plan, and the roadmap. It independently reviewed the
  remaining changed/new CPU contracts, including capability, route, preparation/finalization,
  prepared execution, the Conv1d recognizer, Conv3d emitter/reference oracle, tests, and ledger;
  their existing or implementation-drafted Javadocs remained accurate and complete, so no
  further edits were needed.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- The same pass reviewed `ARCHITECTURE.md`, the current architecture explanation, package
  summaries, Tensor/Compiler convolution contracts, and the glossary without change. The task
  changes no architecture owner, dependency, public API, package boundary, build, backend-
  conformance, integration, or Engine contract. It introduces no reusable term and changes no
  existing term's semantic meaning, so the glossary correctly remains unchanged.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [owner glossary](../../../../glossary.md#term-owner); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Final `./gradlew :backends:cpu:javadoc` completed successfully: 11 tasks, 2 executed and 9
  up-to-date, with only the two expected incubating Vector API warnings and no Javadoc content
  warning. Generated pages for the Conv3d IR, lowerer and geometry, and reference oracle were
  inspected for the finalized contracts. The targeted local checker reported<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
  `Markdown validation passed for 4 files`, including local links, anchors, and balanced fences.
- The retained evidence [manifest](../../../../glossary.md#term-manifest) contains 230 checksummed entries; `sha256sum -c` reported every
  entry `OK`, and its own [digest](../../../../glossary.md#term-digest) is
  `dd185e49c691fb9761414c414c6bdf88507c463c109600deaeb62448c8c2292c`. Final scope is exactly
  35 paths: 15 production, 16 test/resource, and 4 documentation/planning paths. New-file final
  newline and tracked/untracked whitespace checks passed. <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` and
  `git diff --cached --check` produced no diagnostic. Status checks found 0008A Complete, exactly
  one CPU Ready row for 0008B, no 0008B detailed spec, and zero staged paths.
- Successful Model, Compiler, CPU, Class-File, and performance evidence was reused rather than
  rerun because this pass changed documentation and Javadoc only, not executable Java behavior.
  No validation limitation or unresolved documentation issue remains.

## Implementation notes

- Implementation context `01a03e65-a069-71b0-811b-ce55dcf875aa` worked from HEAD
  `e0494057` with the [planning](../../../../glossary.md#term-planning)-only task/master-plan/roadmap diff preserved and unstaged.
- Exact Conv1d recognition accepts only the two axis-2 expansions feeding the third-node Conv2d
  and the axis-2 squeeze of that Conv2d result. It proves private single-use intermediates and
  address-preserving singleton descriptors, then delegates the rank-four boundary descriptors to
  the existing Conv2d lowering, geometry, emitter, and complete-output-cell execution unit. The
  three intermediate values remain virtual; no new Conv1d generated body or entry shape exists.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [node glossary](../../../../glossary.md#node); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [singleton glossary](../../../../glossary.md#term-singleton); [descriptors glossary](../../../../glossary.md#term-descriptor); [lowering glossary](../../../../glossary.md#lowering); [shape glossary](../../../../glossary.md#shape))</span>
- Direct [Conv3d](../../../../glossary.md#conv3d--ncdhw) uses the new [rank](../../../../glossary.md#term-rank)-specific `CpuConv3dIr`, `CpuConv3dLowering`,
  `CpuConv3dEmitter`, and independent `CpuConv3dReferenceKernel`. The emitted NCDHW grouped loop
  owns complete output cells, applies optional intrinsic bias once, loads weights even for
  conceptual padding, multiplies positive zero by that weight, preserves FLOAT64/FLOAT32
  accumulation and final BFLOAT16 conversion, and uses zero workspace/materialization. Dense heap
  carriers use proved dense integer address progression; segment, mixed-carrier, and arbitrary
  resolved layouts use exact long element addressing.<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [bias glossary](../../../../glossary.md#term-bias); [weights glossary](../../../../glossary.md#term-weight); [padding glossary](../../../../glossary.md#padding); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [materialization glossary](../../../../glossary.md#materialization); [heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layouts glossary](../../../../glossary.md#layout); [element glossary](../../../../glossary.md#term-element))</span>
- Conv3d reuses only existing rank-neutral carrier emission, specialization/cache, immutable
  artifact, range orchestration, worker, binding, overlap, and buffer infrastructure. Its IR,
  layout geometry, lowerer, emitter, and oracle remain rank-specific; no shared ConvNd component
  was introduced. Conv1d reuses the completed rank-two numerical unit because its singleton-height
  mapping is exact.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [cache glossary](../../../../glossary.md#term-cache); [immutable glossary](../../../../glossary.md#term-immutable); [artifact glossary](../../../../glossary.md#term-artifact); [binding glossary](../../../../glossary.md#term-binding); [buffer glossary](../../../../glossary.md#term-buffer); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [oracle glossary](../../../../glossary.md#term-test-oracle); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Conv3d-led multi-node partitions and external epilogues fail in the one-node Conv3d lowerer.
  Vector/native selection, materialization, and Conv2d fused/split epilogues remain unavailable to
  Conv3d. Generator schema advanced once from 51 to 52.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [schema glossary](../../../../glossary.md#term-schema))</span>
- The implementation changed 15 CPU production paths and 16 CPU test/resource paths, including
  five new production types, six new test types, and one generated-form-ledger resource; the three
  planning paths are the pre-existing planning change plus this implementation evidence. These
  counts remain within the task ceilings and no other module source, dependency, build, public API,
  architecture, conformance, or integration path changed.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The frozen 12-row machine-readable ledger maps three unchanged Conv1d composition forms to their
  previously evidenced Conv2d bodies, closes seven distinct Conv3d generated/performance forms,
  and records direct plus materialized-suffix Conv2d controls. Seven deterministic Conv3d classes,
  compatibility bytes, SHA-256 values, parsed Class-File member scans, and complete <span data-glossary-navigation="true"> ([Conv1d composition glossary](../../../../glossary.md#conv1d-composition))</span>`javap -c -v`
  reports passed typed-entry, loop/arithmetic/conversion, allowed-call, and forbidden-reference
  inspection. No fields, constructors, method handles, `invokedynamic`, dynamic/bootstrap
  constants, allocation, boxing, reflection, collection dispatch, graph/layout operation
  references, or Synaptik-owned hot helpers were present.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [dispatch glossary](../../../../glossary.md#term-dispatch); [graph glossary](../../../../glossary.md#graph); [operation glossary](../../../../glossary.md#operation))</span>
- After the ledger was frozen, five fresh accepted Java 26 `-Xms1g -Xmx1g` forks passed every
  per-fork <span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>`<= 1.15x` gate with no rejected protocol fork. Aggregate generated/direct medians were:
  dense FLOAT32 `0.667123855x`; grouped biased [FLOAT32](../../../../glossary.md#term-float32) `0.817644722x`; grouped biased [FLOAT64](../../../../glossary.md#term-float64)
  `1.066896079x`; depthwise [BFLOAT16](../../../../glossary.md#term-bfloat16) `1.062009415x`; all-segment FLOAT32 `0.817853071x`;
  arbitrary-layout mixed carrier `0.860624087x`; and parallel-[scalar](../../../../glossary.md#term-scalar) FLOAT32 `0.683170212x`.
  Three pre-ledger measured tuning samples are retained separately as developmental calibration and
  are not part of the frozen protocol. The unchanged Conv2d control aggregates also passed:
  direct dense <span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>`0.992271187x`, grouped `1.035211861x`, depthwise `0.939897066x`, general mixed
  `0.971556498x`, [fused](../../../../glossary.md#specialization-and-fusion) `0.988055832x`, parallel `0.987585697x`, materialized split
  `0.991748527x`, and [pointwise](../../../../glossary.md#term-pointwise) add control `0.988852838x`.
- Focused validation passed for Model Conv1d/Conv3d (21 tests), Compiler Conv3d (4 tests), and the
  prescribed CPU families/boundaries (123 tests after correcting a test-only artifact-cache key
  collision). The single authoritative <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [cache key glossary](../../../../glossary.md#term-cache-key))</span>`./gradlew :backends:cpu:test --rerun-tasks` passed 494
  tests with zero failures/errors and three established skips. Retained XML, forks, ledger,
  classes, checksums, inspection reports, controls, developmental samples, and manifest live at
  `/private/tmp/synaptik-cpu-0008a-retained-evidence-20260826`; the [manifest](../../../../glossary.md#term-manifest)-file [SHA-256](../../../../glossary.md#term-sha-256) is
  `dd185e49c691fb9761414c414c6bdf88507c463c109600deaeb62448c8c2292c`.
- Documentation context `01a03e87-9b14-7003-a6e9-e984bcf17144` independently reviewed the actual
  production, test, resource, and planning diff against `AGENTS.md`, `ARCHITECTURE.md`, the current
  architecture plan, documentation rules, the General/API-and-Javadoc/Developer-guide/Backend-
  guide/Planning/Example profiles, the CPU guide and glossary, the planning guide/roadmap/CPU
  master plan, this task, complete CPU 0008, Model 0025G/0025H, and Compiler 0006B contracts.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Backend glossary](../../../../glossary.md#backend))</span>
- The documentation pass finalized schema-52 and new Conv3d contract Javadocs, including the
  immutable IR, exact topology/lowering boundary, cold rank-five layouts, packed geometry,
  compatibility constructors, ownership, nullability, ranges, results, and failure conditions.
  It updated the CPU guide with the exact visible Conv1d recognizer, direct Conv3d numerical and
  resource behavior, scalar/parallel-scalar range ownership, carrier and overlap validation,
  intrinsic-bias-only epilogue boundary, schema 52, a checked geometry example, and bounded
  generated-code/performance evidence. It synchronized this task, the CPU master plan, and the
  roadmap so 0008A is Complete and 0008B is the sole Ready CPU task without a detailed spec.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [nullability glossary](../../../../glossary.md#term-nullability); [epilogue glossary](../../../../glossary.md#term-epilogue); [Ready glossary](../../../../glossary.md#term-ready))</span>
- The glossary was reviewed without change. The implementation changes the current execution
  status of already defined Conv1d composition, Conv3d, generated-kernel, and schema concepts; it
  introduces no reusable term and changes none of those terms' semantic meanings. Current-status
  detail therefore remains in the CPU guide and planning evidence rather than duplicating it in
  the central terminology dictionary.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>

## Completion summary

[CPU](../../../../glossary.md#term-cpu) 0008A is complete. It recognizes only the exact visible [NCW](../../../../glossary.md#term-ncw) Conv1d two-expansion/[Conv2d](../../../../glossary.md#convolution--conv2d)/
squeeze composition while keeping its [singleton](../../../../glossary.md#term-singleton) [views](../../../../glossary.md#term-view-meanings) virtual, and it adds direct grouped [NCDHW](../../../../glossary.md#conv3d--ncdhw)
[Conv3d](../../../../glossary.md#conv3d--ncdhw) generated CPU execution with optional intrinsic [bias](../../../../glossary.md#term-bias), arbitrary admitted non-negative
resolved [layouts](../../../../glossary.md#layout), [heap](../../../../glossary.md#term-heap)/segment/mixed carriers, zero workspace/[materialization](../../../../glossary.md#materialization), and [scalar](../../../../glossary.md#term-scalar) or
bounded parallel-scalar complete-output-cell work. External Conv3d epilogues and general [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag)
handling [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). Generator [schema](../../../../glossary.md#term-schema) is 52. The recorded implementation tests, generated-code
inspection, five-[fork](../../../../glossary.md#term-fork-meanings) performance evidence, finalized Javadocs, CPU guide, [planning](../../../../glossary.md#term-planning) status, and
documentation [validation](../../../../glossary.md#term-validation) all pass without changing architecture or shared contracts. CPU 0008B
is the sole [Ready](../../../../glossary.md#term-ready) CPU task and has no detailed task specification.

Status: Complete
