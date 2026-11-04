# [CPU](../../../../glossary.md#term-cpu) Task 0008F: Portable [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) Execution and Bounded [Linear](../../../../glossary.md#term-linear-meanings) Epilogues

## Reader context

This task records portable generated MATMUL execution, including rank-one promotion, batch
broadcasting, numeric promotion, and bounded linear epilogues. `CpuMatmulLowering` produces
`CpuMatmulIr` and checked geometry; `CpuMatmulCandidateSelector` chooses among four family-local
scalar/vector and direct/tiled realizations. These choices differ from `CpuFusionDecision`'s
fused-versus-split epilogue selection and `CpuRepresentationPlanner`'s candidate-only input-copy
alternatives.

`CpuMatmulEmitter` generates the production body; `CpuMatmulReferenceKernel` is a semantic and
specialized-algorithm oracle, not a Runtime interpreter. Class-File inspection is the structural
oracle. Shared Prepare assigns declared resources, finalization binds artifacts, and CPU execution
borrows its supplied `CpuWorkerGroup`; worker ranges own disjoint outputs while each accumulator
traverses full K. Geometry describes resolved invocation storage; declared copy workspaces are run
scratch, not persistent model state.

Current-topic navigation (not a replacement for the historical record):

- [current MATMUL-backed linear composition](../../../../reference/tensor/matmul-linear.md#linear-projection-convenience) — Distinguishes Tensor composition and visible transpose/bias boundaries from a state-owning NN layer.
- [current ordinary-production route policy](../../../../design/notes/cpu-kernel-strategy.md#current-ordinary-production-decision) — Prevents historical vector or parallel evidence from being read as current production-default authorization.

## Status

Complete.

The implementation, sealed performance evidence, and independent clean-context documentation
finalization are complete. [CPU](../../../../glossary.md#term-cpu) 0008G is the next Draft frontier and still has no detailed task
specification.

## Goal

Implement the complete current static [Model](../../../../glossary.md#term-model) [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) semantic family on the [CPU](../../../../glossary.md#term-cpu) portable generated-
[bytecode](../../../../glossary.md#term-bytecode) route, including [vector](../../../../glossary.md#term-vector), [matrix](../../../../glossary.md#term-matrix-meanings), batched, and right-aligned [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-broadcast forms, and
realize the exact CPU 0008C [linear](../../../../glossary.md#term-linear-meanings)-[epilogue](../../../../glossary.md#term-epilogue) recognition as either one proved [fused](../../../../glossary.md#specialization-and-fusion) unit or its
existing valid split.

The implementation must provide a generated optimal [scalar](../../../../glossary.md#term-scalar) fallback for every admitted non-[BOOL](../../../../glossary.md#data-type--datatype)
numeric case and may select only one of four bounded family-local realization kinds: scalar direct,
N-vector direct, scalar M/N-tiled, or N-vector M/N-tiled. It must use existing caller-owned CPU
workers only across independent output tiles, declare every representation transition before
[Runtime](../../../../glossary.md#term-runtime), preserve a safe direct plan with tuning disabled, and prove generated semantics and hot-
loop shape against an equally specialized clean Java [oracle](../../../../glossary.md#term-test-oracle).

## [Scope](../../../../glossary.md#term-scope)

### Current source-backed baseline

- [Model](../../../../glossary.md#term-model) task 0019 and `MatmulKind` admit operands of rank at least one. A rank-one left operand is
  promoted to semantic <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`[1,K]` and loses the result M axis; a rank-one right operand is promoted to
  semantic `[K,1]` and loses the result N [axis](../../../../glossary.md#term-axis). Leading [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) prefixes broadcast from the right.
- The current result type is the floating promotion of any [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`FLOAT32`](../../../../glossary.md#term-float32), or [`FLOAT64`](../../../../glossary.md#term-float64)
  pair, or the signed-integral promotion of an [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype) pair. BOOL and cross-category pairs
  are not MATMUL semantics. BFLOAT16 results accumulate in FLOAT32; FLOAT32 and FLOAT64 results
  accumulate in their result type. Integral results use result-width modular arithmetic.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [modular arithmetic glossary](../../../../glossary.md#term-modular-arithmetic))</span>
- Zero extents are legal. An empty K domain writes positive floating zero or integral zero for each
  output element. Model permits floating reassociation and fused multiply-add behavior; it does not
  promise bitwise equality across implementations or backends.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [reassociation and fused multiply-add glossary](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor); [backends glossary](../../../../glossary.md#backend))</span>
- Compiler structured inference independently revalidates rank, K equality, right-aligned batch
  broadcasting, result type, and exact result descriptor. Compiler linear lowering remains the
  visible <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [broadcasting glossary](../../../../glossary.md#broadcasting); [descriptor glossary](../../../../glossary.md#term-descriptor); [lowering glossary](../../../../glossary.md#lowering))</span>`PERMUTE([1,0]) -> MATMUL -> optional ADD` composition; there is no hidden [`LINEAR`](../../../../glossary.md#term-linear-meanings) kind.
- [CPU](../../../../glossary.md#term-cpu) capability currently does not execute [MATMUL](../../../../glossary.md#matrix-multiplication--matmul). CPU 0008C recognizes exactly one `MATMUL` with
  two inputs and one output, plus an optional external rank-one ADD matching the result's final
  extent and at most one exact terminal activation or CLAMP. The recognized floating suffix is
  currently restricted to FLOAT32/FLOAT64. A canonical transposed weight is only the exact private
  rank-two <span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [activation glossary](../../../../glossary.md#term-activation); [canonical glossary](../../../../glossary.md#term-canonical-form); [weight glossary](../../../../glossary.md#term-weight); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`PERMUTE` with axes `[1,0]`; no broader pattern is inferred. Its MATMUL fact remains
  `UNSUPPORTED_ANCHOR` until this task supplies executable baseline units.
- CPU 0008B supplies bounded partition-DAG units and sequential inter-unit execution. CPU 0008D
  owns complete-topology profitability and preserves canonical split. CPU 0008E supplies explicit,
  candidate-only whole-value affine materialization with declared workspaces and copy units;
  ordinary preparation currently selects direct representation with<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [topology glossary](../../../../glossary.md#term-topology); [materialization glossary](../../../../glossary.md#materialization))</span>
  `DIRECT_MATERIALIZATION_UNPROVED`. CPU 0008E1 makes shared [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) occurrences the sole
  structural graph source while leaving CPU-owned unit, candidate, IR, and resource facts intact.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- The portable generated entry already receives typed carriers, one primitive geometry array, and
  `start`/`end` range [arguments](../../../../glossary.md#term-argument). [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) is caller-owned. Runtime performs no graph
  interpretation, route search, allocation, or policy decision. Generator schema 53 describes the
  current artifact forms and must not be reused for a new MATMUL class form.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [schema glossary](../../../../glossary.md#term-schema); [artifact glossary](../../../../glossary.md#term-artifact))</span>

### Semantic geometry and [lowering](../../../../glossary.md#lowering)

- Lower every admitted [occurrence](../../../../glossary.md#term-occurrence) once, during [CPU](../../../../glossary.md#term-cpu) analysis, to a checked [immutable](../../../../glossary.md#term-immutable) `CpuMatmulIr`
  and a cold geometry record. Normalize exact left/right/result extents, offsets, non-negative
  element strides, carriers, data types, batch rank, right-aligned batch extents and per-operand
  batch strides, checked <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [strides glossary](../../../../glossary.md#term-stride); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [rank glossary](../../../../glossary.md#term-rank))</span>`batchCount`, M, K, N, vector-promotion flags, removed result axes, and the
  exact logical-result-to-storage mapping.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [axes glossary](../../../../glossary.md#term-axis))</span>
- Represent an absent promoted M or N axis internally as extent one without reading or writing a
  nonexistent physical axis. A missing or broadcast batch coordinate has effective stride zero;
  every non-broadcast coordinate preserves the resolved layout stride. Batch flattening and
  unflattening use one stable right-to-left order.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [layout glossary](../../../../glossary.md#layout))</span>
- Revalidate rank, K equality, right-aligned batch compatibility, result Shape and type, fully
  static extents, resolved layouts, non-negative strides, injective output, address spans, and
  exact carrier/type compatibility. Use checked arithmetic for ranks, products, byte sizes,
  addresses, tile counts, flattened work counts, and range endpoints. Fail closed before artifact
  generation, writes, or work submission on disagreement, overflow, dynamic facts, negative
  strides, unsupported carriers, or unproved output injectivity.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [injective glossary](../../../../glossary.md#term-injective); [carrier glossary](../../../../glossary.md#term-storage-carrier); [Fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [artifact glossary](../../../../glossary.md#term-artifact); [overflow glossary](../../../../glossary.md#term-overflow); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Reject output overlap with either input, the optional bias, and every declared materialized
  source before any write or worker submission. Distinct output work units must have disjoint
  address sets. Input/input alias is harmless only when both are read-only and no declared copy or
  output transition can overwrite either span; otherwise fail closed.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Preserve arbitrary admitted positive-stride layouts and heap/segment carrier mixtures through
  the scalar fallback. Dense storage is not a semantic prerequisite.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [scalar glossary](../../../../glossary.md#term-scalar))</span>

### Numeric coverage

- Admit all and only current non-[BOOL](../../../../glossary.md#data-type--datatype) [Model](../../../../glossary.md#term-model) pairs: all nine ordered floating pairs over
  [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`FLOAT32`](../../../../glossary.md#term-float32), and [`FLOAT64`](../../../../glossary.md#term-float64), and all four ordered signed-integral pairs over [`INT32`](../../../../glossary.md#data-type--datatype) and
  [`INT64`](../../../../glossary.md#data-type--datatype). The result type and [accumulator](../../../../glossary.md#accumulator-type) policy are exactly Model promotion.
- Decode each BFLOAT16 operand to FLOAT32, perform FLOAT32 accumulation when the promoted result is
  BFLOAT16 or FLOAT32, and round once on the final BFLOAT16 store. FLOAT64 accumulation uses
  `double`. INT32 and INT64 accumulation use Java's wrapping arithmetic at the promoted width; no
  widening accumulator, saturation, overflow exception, or floating conversion is permitted.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [saturation glossary](../../../../glossary.md#term-saturation); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Each output accumulator starts at the type-specific zero and traverses K in increasing logical
  order exactly once. A selected floating direct oracle may use the same explicit fused-
  multiply-add decision as generated code, because Model permits it, but generated and direct
  implementations must use the same decision for evidence. Candidate identity records that
  numerical form.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [fused glossary](../../../../glossary.md#specialization-and-fusion); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

### Bounded realization set and cold selection

Use this closed realization vocabulary. No other tile family, unroll search, [runtime](../../../../glossary.md#term-runtime) policy, or
[shape](../../../../glossary.md#shape)-specific class is permitted in this task.

| Realization | Exact code shape | Eligibility |
|---|---|---|
| `DIRECT_SCALAR` | One output at a time; initialize one [accumulator](../../../../glossary.md#accumulator-type), traverse full K, apply the [epilogue](../../../../glossary.md#term-epilogue), store once. No coarse tile loop. | Mandatory fallback for every admitted semantic/[layout](../../../../glossary.md#layout)/[carrier](../../../../glossary.md#term-storage-carrier) case. Selected for empty work, K zero, M or N equal to one, or checked multiply-add count below 4,096. |
| `DIRECT_N_VECTOR` | One M row at a time; broadcast one left [scalar](../../../../glossary.md#term-scalar), load [contiguous](../../../../glossary.md#term-contiguous) right N [lanes](../../../../glossary.md#term-lane), update N-lane accumulators through full K, store contiguous result lanes, then execute a scalar N tail. | Same-type [FLOAT32](../../../../glossary.md#term-float32), [FLOAT64](../../../../glossary.md#term-float64), [INT32](../../../../glossary.md#data-type--datatype), or [INT64](../../../../glossary.md#data-type--datatype) only; preferred species for that exact type; K positive; right logical N [stride](../../../../glossary.md#term-stride) and result logical N stride both one; no [fused](../../../../glossary.md#specialization-and-fusion) terminal; and at least one full species. |
| `TILED_SCALAR_2X2` | Coarse `(batch,mTile,nTile)` loop with at most four scalar accumulators, full-K traversal inside the microtile, and explicit M/N tails. | Any admitted type/layout/carrier after direct-small gates; M and N at least two; checked multiply-add count at least 16,384. |
| `TILED_N_VECTOR_2X2` | Two M rows by two preferred-species N [vectors](../../../../glossary.md#term-vector), at most four vector accumulators, full-K traversal, then scalar M/N tails. | `DIRECT_N_VECTOR` type/layout eligibility, M at least two, N at least two species, and checked multiply-add count at least 65,536. |

- The realization-kind enum contains exactly these four values. For any occurrence, construct only
  the eligible subset and always retain <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>`DIRECT_SCALAR`. The fixed direct, 2-by-2 scalar, and
  2-by-2-species vector forms bound generated code size and live accumulator state; the
  thresholds below are a conservative first-route policy that the required seam and performance
  evidence must justify, not universal optimality or hardware-cache claims.
  Apply the mandatory direct-small gates first. Otherwise, from 4,096 through 65,535 operations an
  eligible direct-vector form wins; without vector eligibility, direct scalar wins through 16,383
  and scalar tiling wins from 16,384. At or above 65,536, the tiled-vector form wins when eligible
  and scalar tiling wins otherwise. These conservative constants are policy facts, not cache-size
  claims. The performance gate must validate them; a failed gate blocks completion and requires a
  planning correction rather than an undocumented threshold change.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [tiling glossary](../../../../glossary.md#tiling); [constants glossary](../../../../glossary.md#term-constant); [planning glossary](../../../../glossary.md#term-planning))</span>
- Generated-artifact identity contains only realization kind, exact operand/result and accumulator
  types, carrier pattern, access-form facts, preferred species bits when used, numerical form,
  epilogue kind, materialization identity, and the fixed tile parameters above. Extents and batch
  count stay in checked invocation geometry; tile counts, threshold outcomes, range count, and
  worker count are invocation facts, not class facts. Compatible Shapes therefore reuse one
  artifact, and the task adds a bounded number of class forms rather than one class per Shape.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [materialization glossary](../../../../glossary.md#materialization); [parameters glossary](../../../../glossary.md#term-parameter); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- BFLOAT16, mixed-width, non-unit-N, and terminal-epilogue cases are scalar-only in this task.
  This is an explicit eligibility boundary, not a statement that the Vector API can never support
  them. Vector tails are scalar; masked/gather/scatter vector access is not introduced.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [Vector API glossary](../../../../glossary.md#term-vector-api); [gather glossary](../../../../glossary.md#gather))</span>
- Coarse M/N tile loops appear only in a selected tiled realization. Small, vector/matrix
  degenerate, and direct-vector artifacts retain their simpler direct loop shape.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>

### K traversal and parallel [ownership](../../../../glossary.md#term-ownership-meanings)

- Every realization performs full-K accumulation inside each output or output microtile. This task
  has no cache K blocking, packed K panels, parallel K splitting, partial-result buffer, combine
  phase, atomic update, lock, or worker-count-dependent reduction order.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [buffer glossary](../../../../glossary.md#term-buffer); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Cache K blocking is distinct from parallel K splitting, but both are deferred. A later task may
  add cache blocking only with a proved per-output numerical-order contract and explicit resource
  identity. Parallel K splitting requires a separate numerical/determinism and combine design.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- A work unit is exactly one flattened [occurrence](../../../../glossary.md#term-occurrence) under the selected geometry. `DIRECT_SCALAR`
  uses unit M and N extents. `DIRECT_N_VECTOR` uses one complete `(batch,m)` row: its sole N tile
  coordinate is zero and the generated body owns the complete N vector loop plus scalar tail.
  Tiled forms use their actual coarse <span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [scalar tail glossary](../../../../glossary.md#term-scalar-tail))</span>`(batch,mTile,nTile)` [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). The generated `start` and
  `end` arguments index only this work-unit domain. Stable row-major flattening owns every output
  cell exactly once and permits only disjoint work-unit ranges.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Parallel execution reuses the existing caller-owned [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup), cold selected range count,
  and minimum-work policy. It submits no work for an empty output and performs no generated worker
  creation, synchronization, allocation, or nested parallelism. K is never in the parallel range.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

### [Bias](../../../../glossary.md#term-bias) and terminal epilogues

- Consume only exact facts already recognized by CPU 0008C. Do not search the graph again, infer a
  hidden linear operator, generalize transpose, reinterpret an arbitrary ADD as bias, or fuse
  through publication, fan-out, unsupported access, alias, numerical, resource, representation,
  or profitability barriers.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [graph glossary](../../../../glossary.md#graph); [linear glossary](../../../../glossary.md#term-linear-meanings); [transpose glossary](../../../../glossary.md#term-transpose); [publication glossary](../../../../glossary.md#term-publication); [fan-out glossary](../../../../glossary.md#term-fan-out); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- A fused bias is the existing external rank-one operand whose sole extent equals N, whose type is
  exactly the MATMUL result type, and whose resolved access is safe. The generated order is
  full-K accumulator, ADD, optional one terminal activation or CLAMP, then exactly one result
  store. Preserve the recognized ADD operand order and exact terminal attributes.<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [extent glossary](../../../../glossary.md#term-extent); [accumulator glossary](../../../../glossary.md#accumulator-type); [activation glossary](../../../../glossary.md#term-activation))</span>
- Fused epilogues are FLOAT32/FLOAT64 only. Scalar candidates may implement the exact CPU 0008C
  terminal set. Vector candidates may fuse NONE or exact bias ADD only; a terminal activation or
  CLAMP retains the complete split candidate in this task. A bias that widens the MATMUL product,
  BFLOAT16 or integral suffix, or any unsupported access also remains split.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [Scalar glossary](../../../../glossary.md#term-scalar); [Vector glossary](../../../../glossary.md#term-vector); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Decomposition first constructs the canonical executable split: one bare MATMUL baseline unit
  plus the ordinary pointwise suffix unit when a suffix exists. Recognition associates the exact
  CPU 0008C fact with those one or two baseline units and replaces <span data-glossary-navigation="true"> ([Decomposition glossary](../../../../glossary.md#term-decomposition); [canonical glossary](../../../../glossary.md#term-canonical-form); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>`UNSUPPORTED_ANCHOR` only for
  that fact with a new closed executable-alternatives disposition. That disposition alone relaxes
  CPU 0008D's recognition barrier and contributes exactly one complete fused topology in addition
  to the unchanged split; <span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>`EXISTING_SPECIALIZED`, `ORDINARY_SPLIT`, other
  `UNSUPPORTED_ANCHOR` facts, and every non-[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) fact retain their current meanings.
- The existing CPU 0008D profitability selector ranks the complete fused and split topologies; no
  family-local heuristic may bypass or contradict its <span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>`CpuFusionDecision`. Preparation verifies
  that both alternatives cover the same recognized members and external boundaries exactly once,
  and that the selected identity is one of those alternatives. Rejection or uncertainty selects
  the split without losing valid MATMUL execution.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>

### Representation and [materialization](../../../../glossary.md#materialization)

- Direct input representations are always a complete executable option, including with tuning
  disabled. Materialization may only use CPU 0008E's explicit contiguous affine-copy units,
  declared workspaces, stable source identities, publication rules, representation transitions,
  and candidate accounting.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [publication glossary](../../../../glossary.md#term-publication))</span>
- Admit at most one whole-logical-input copy per complete [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) candidate. [CPU](../../../../glossary.md#term-cpu) 0008E's existing
  `CO_CONSUMED_PAIR` rule rejects two copied sources consumed by the same represented instruction,
  so this task must not invent a two-copy MATMUL exception. A canonical transposed weight may
  therefore remain a direct strided <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [weight glossary](../../../../glossary.md#term-weight))</span>`[K,N]` view or become one declared contiguous whole-value
  copy. Do not add panel packing, invocation-local arrays/segments, hidden copying, or same-unit
  copy work.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings))</span>
- Every materialized candidate records checked bytes, copy cost, consumer reuse, publication
  barrier, direct-to-contiguous transition, and all copy/kernel units in complete-plan cost. The
  existing <span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>`DIRECT_MATERIALIZATION_UNPROVED` behavior remains the no-tuning production default
  unless fresh task evidence proves an exact promoted admitted set end to end. Kernel-only speed
  cannot justify a representation transition.

### Generated [artifact](../../../../glossary.md#term-artifact) and execution integration

- Add a family-specific IR/lowerer, bounded candidate selector, emitter, and direct reference
  oracle. Thread the selected MATMUL form through the existing portable specialization, preparation
  plan, finalizer, executable geometry binding, artifact identity, and class generator without a
  new public facade or generic graph interpreter.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [oracle glossary](../../../../glossary.md#term-test-oracle); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [specialization glossary](../../../../glossary.md#term-specialization); [binding glossary](../../../../glossary.md#term-binding); [identity glossary](../../../../glossary.md#term-identity-meanings); [facade glossary](../../../../glossary.md#term-facade); [graph glossary](../../../../glossary.md#graph))</span>
- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 53 to 54 because MATMUL introduces a new
  emitted family and artifact identity/form. Compatibility bytes use schema 54 for every
  specialization, so schema-53 envelopes fail safely. MATMUL class-identity bytes use a new
  schema-54 family projection containing every code-shaping MATMUL fact; unchanged pre-0008F
  specializations retain the existing schema-52 class-identity projection. Representative
  unchanged controls must therefore retain exact binary names and class bytes.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Generated classes remain final, field-free, with one exact static entry and no Synaptik helper
  call from the hot body. Cold geometry validation and direct carrier binding precede invocation.
  Hot loops contain primitive index arithmetic, direct array/segment access, primitive or Vector
  API accumulation, epilogue arithmetic, and stores only—no allocation, boxing, reflection,<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [carrier glossary](../../../../glossary.md#term-storage-carrier); [Vector API glossary](../../../../glossary.md#term-vector-api); [epilogue glossary](../../../../glossary.md#term-epilogue))</span>
  `Map` lookup, graph/value traversal, route switch, helper dispatch, cache lookup, or worker
  management.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [cache glossary](../../../../glossary.md#term-cache))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [OpenBLAS](../../../../glossary.md#term-openblas), Accelerate, [oneMKL](../../../../glossary.md#term-onemkl), [AOCL](../../../../glossary.md#term-aocl), [JNI](../../../../glossary.md#term-jni), [FFM](../../../../glossary.md#term-ffm) downcalls, or any other native [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) route.
- Dynamic Shapes, negative strides, sparse/quantized MATMUL, FLOAT16, complex types, BOOL,
  cross-category arithmetic, transposed-MATMUL attributes, or inferred graph patterns.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [strides glossary](../../../../glossary.md#term-stride); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [graph glossary](../../../../glossary.md#graph))</span>
- Cache K blocking, K-panel packing, parallel K splitting, partial sums, combine workspaces,
  worker-dependent reduction order, atomics, locks, nested workers, or invocation-local packing.<span data-glossary-navigation="true"> ([Cache glossary](../../../../glossary.md#term-cache))</span>
- Vector BFLOAT16, mixed-width vector arithmetic, vector gather/scatter, masked vector tails, or
  vector terminal activations/CLAMP.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [gather glossary](../../../../glossary.md#gather); [activations glossary](../../../../glossary.md#term-activation))</span>
- More than one bias ADD, more than one terminal operation, arbitrary bias broadcasting, or any
  epilogue outside CPU 0008C's closed recognition set.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [operation glossary](../../../../glossary.md#operation); [broadcasting glossary](../../../../glossary.md#broadcasting); [epilogue glossary](../../../../glossary.md#term-epilogue))</span>
- Public API, Model, Compiler, Planning, Prepare, Runtime, Backend Contract, OpenBLAS provider,
  architecture, dependency, Gradle, conformance, integration, Engine, tuning, or tracing changes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>
- A detailed [CPU](../../../../glossary.md#term-cpu) 0008G task specification or implementation of any later family.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`docs/architecture/current-architecture-plan.md`](../../../../architecture/current-architecture-plan.md)
- [`docs/planning/planning-guide.md`](../../../planning-guide.md)
- [CPU backend master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
- [CPU 0008B](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)
- [CPU 0008C](0008c-typed-specialized-subgraph-and-epilogue-recognition.md)
- [CPU 0008D](0008d-bounded-fusion-profitability-and-typed-decision-facts.md)
- [CPU 0008E](0008e-bounded-multi-input-materialization-and-representation-reuse.md)
- [CPU 0008E1](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md)
- [Model 0019](../../../modules/model/tasks/0019-matmul-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model 0019D](../../../modules/model/tasks/0019d-linear-convenience.md)
- [Compiler 0005D](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>

## Architecture constraints

- `ARCHITECTURE.md` remains authoritative. Planning chooses CPU ownership; CPU analysis owns exact
  lowering, candidates, representation/resource declarations, and portable/native route choice;
  shared Prepare assigns declared slots; Runtime invokes immutable prepared work only.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lowering glossary](../../../../glossary.md#lowering); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- The shared [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) is the sole partition-structure source. CPU-private IR may retain only
  exact selected structural and semantic facts, never the complete cross-backend graph.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [backend glossary](../../../../glossary.md#backend); [graph glossary](../../../../glossary.md#graph))</span>
- Generated JVM bytecode is the permanent portable production route and truthful semantic
  fallback. Reference code is an oracle and evidence owner, not a Runtime interpreter.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [bytecode glossary](../../../../glossary.md#term-bytecode); [oracle glossary](../../../../glossary.md#term-test-oracle); [owner glossary](../../../../glossary.md#term-owner))</span>
- All expensive geometry, legality, alias, candidate, tile, species, representation, and range
  decisions occur before Runtime. Generated hot code receives primitive facts and direct carriers.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Every generated-code realization must preserve the semantic algorithm, hot-loop/dataflow shape,
  and avoidable-overhead profile of an optimal clean Java implementation of the same specialized
  case. A safe generic fallback does not excuse avoidable overhead in a proved specialized path.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape))</span>
- Inter-unit order stays strict. Child parallelism owns disjoint output ranges only. Planning and
  Runtime gain no fusion, packing, MATMUL, or tuning policy.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- No new module dependency or architecture rule is authorized. If implementation discovers one is
  necessary, stop and request an architecture decision instead of widening this task.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [widening glossary](../../../../glossary.md#term-widening))</span>

## Package impact

All implementation remains under
`io.github.pho001.synaptik.backend.cpu.internal`:

- `ir`: one [sealed](../../../../glossary.md#term-sealed)-family `CpuMatmulIr` with exact structural [identity](../../../../glossary.md#term-identity-meanings) and one cold geometry [carrier](../../../../glossary.md#term-storage-carrier);
- [`lowering`](../../../../glossary.md#lowering): `CpuMatmulLowering` and `CpuMatmulCandidateSelector`, with existing decomposer,
  recognizer, representation planner, and partition lowering as integration owners;<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
- `codegen.emit`: `CpuMatmulEmitter` plus existing class-generator [dispatch](../../../../glossary.md#term-dispatch);
- `reference`: `CpuMatmulReferenceKernel`, used only by tests/evidence;
- [`prepare`](../../../../glossary.md#prepare), [`cache`](../../../../glossary.md#term-cache), and `executable`: only the narrow plan, schema, finalization, geometry-binding,
  and range-execution adaptations needed for the selected family.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [binding glossary](../../../../glossary.md#term-binding))</span>

`CpuCapabilityProvider` remains the sole supported public [CPU](../../../../glossary.md#term-cpu) [API](../../../../glossary.md#term-application-programming-interface-api); no new supported package or
public cross-[module](../../../../glossary.md#term-module) type is introduced.

## Affected files

Expected production [owners](../../../../glossary.md#term-owner):

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuSpecializedSubgraph.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuMatmulIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionDagDecomposer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuSpecializedSubgraphRecognizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlanner.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMatmulLowering.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMatmulCandidateSelector.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuMatmulEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuMatmulReferenceKernel.java`

Expected focused test owners are:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuMatmulIrTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMatmulLoweringTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMatmulCandidateSelectorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionDagDecomposerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuSpecializedSubgraphRecognizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuFusionProfitabilitySelectorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlannerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuShapePolymorphicArtifactTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuMatmulGeneratedKernelTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuMatmulPerformanceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuMatmulReferenceTest.java`

If implementation proves a named existing suite is not the current [owner](../../../../glossary.md#term-owner), substitute one matching
[CPU](../../../../glossary.md#term-cpu) test path and record the source-backed [ownership](../../../../glossary.md#term-ownership-meanings) reason without increasing the ceiling.

The targeted documentation pass may update affected CPU package/type Javadocs,
`docs/backend-guide/cpu-backend.md`, terminology in `docs/glossary.md` only if the implementation
changes a term, this task, the CPU master plan, and the roadmap. It must record reasoned no-change
conclusions for all reviewed documents.

## Maximum [scope](../../../../glossary.md#term-scope)

The implementation ceiling is 42 repository paths: at most 19 [CPU](../../../../glossary.md#term-cpu) production paths, 18 CPU test
paths, and five documentation/[planning](../../../../glossary.md#term-planning) paths: the CPU [backend](../../../../glossary.md#backend) guide, conditional glossary, this
task, CPU master plan, and roadmap. The unusually broad ceiling is justified because one
cohesive generated family must cross the existing [sealed](../../../../glossary.md#term-sealed) [IR](../../../../glossary.md#term-intermediate-representation-ir), recognition, representation,
preparation, [schema](../../../../glossary.md#term-schema), generation, execution, [oracle](../../../../glossary.md#term-test-oracle), and evidence boundaries without introducing a
parallel framework. It is a ceiling, not a [target](../../../../glossary.md#term-target).

No [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), Planning, [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), backend-contract, [OpenBLAS](../../../../glossary.md#term-openblas)-provider, [Engine](../../../../glossary.md#term-engine),
architecture, architecture-test, backend-[conformance](../../../../glossary.md#term-conformance), integration-test, resource, generated-
source, or build path may change. If source analysis proves one of those changes necessary, stop
and revise the plan before editing it.

## Acceptance criteria

- CPU capability truthfully admits every fully static, resolved-layout, supported-carrier current
  non-BOOL MATMUL pair and rejects all unsupported rank, shape, type, layout, alias, output-
  injectivity, carrier, dynamic, or overflow cases.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier); [BOOL glossary](../../../../glossary.md#data-type--datatype); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [rank glossary](../../../../glossary.md#term-rank); [shape glossary](../../../../glossary.md#shape); [alias glossary](../../../../glossary.md#aliasing-and-arena); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Cold lowering exactly normalizes vector promotion/removal, right-aligned batch broadcasting,
  batchCount, M/K/N, layouts, result mapping, and checked spans for vector-vector, vector-matrix,
  matrix-vector, matrix-matrix, and batched cases.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [vector glossary](../../../../glossary.md#term-vector); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [broadcasting glossary](../../../../glossary.md#broadcasting); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Generated execution matches the Model policies for all floating promotions, BFLOAT16
  accumulation/rounding, modular INT32/INT64 arithmetic, zero sizes, empty K identity, tails,
  broadcasting, transposed-weight views, arbitrary admitted layouts, and heap/segment mixtures.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [rounding glossary](../../../../glossary.md#term-rounding); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [identity glossary](../../../../glossary.md#term-identity-meanings); [weight glossary](../../../../glossary.md#term-weight); [views glossary](../../../../glossary.md#term-view-meanings); [heap glossary](../../../../glossary.md#term-heap))</span>
- `DIRECT_SCALAR` remains complete and selectable with tuning disabled. The other three realization
  kinds exist only under the exact eligibility and thresholds in this task; no per-Shape class
  explosion or hot-path policy appears.
- Each output accumulator traverses full K. There is no K blocking or splitting, partial sum,
  combine, atomic, lock, hidden workspace, panel pack, or worker-count-dependent arithmetic.<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Parallel ranges deterministically own disjoint `(batch,mTile,nTile)` work units and reject every
  output overlap before worker submission. Scalar and parallel execution are semantically equal.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar))</span>
- Exact 0008C bias/terminal facts produce a fused candidate only when legality, access, alias,
  numerical, representation, resource, and profitability facts all prove it. The exact split is
  always retained and selected on uncertainty. Epilogue order is accumulator, bias ADD, optional
  terminal, final store.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [fused glossary](../../../../glossary.md#specialization-and-fusion); [Epilogue glossary](../../../../glossary.md#term-epilogue))</span>
- Canonical executable split units exist before recognition. Only the new MATMUL executable-
  alternatives disposition relaxes the 0008D barrier; fused and split alternatives cover the
  same recognized members and external boundaries exactly once, and the existing<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form))</span>
  `CpuFusionDecision` selects between them.
- Direct and one-copy-left or one-copy-right candidates use only declared 0008E affine-copy
  resources and complete-plan accounting. Two-copy MATMUL remains `CO_CONSUMED_PAIR`; no hidden
  allocation or copy occurs during invocation.
- Schema advances exactly to 54. Compatibility identities reject stale schema-53 envelopes; new
  MATMUL class identities distinguish every code-shaping fact; unchanged generated controls keep
  the schema-52 class-identity projection, binary names, and byte-identical classes.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Raw Class-File parsing and complete `javap -c -v -p` inspection prove one final field-free class
  and static entry per selected artifact, expected primitive/Vector operations, and no Synaptik
  helper, allocation, boxing, reflection, map, cache, graph, or route references in hot loops.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache); [graph glossary](../../../../glossary.md#graph))</span>
- The generated implementation and optimal clean Java oracle use the same specialized algorithm,
  accumulation policy, loop/tile shape, carrier/layout case, and epilogue. Semantic checks precede
  timing.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- No public API, module dependency, architecture rule, Gradle configuration, native route, or later
  task specification changes. CPU 0008G remains Draft without a detailed specification.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- A separate documentation-focused context reviews and finalizes all affected Javadocs,
  explanatory documentation, examples, and glossary impact before implementation is marked
  Complete.

## Tests / [validation](../../../../glossary.md#term-validation)

### Tier 1: focused implementation checks

- Add unit tests for every rank family and right-aligned batch case, including unequal batch ranks,
  both-sided singleton broadcasting, vector-vector scalar result, rank-one axis removal, zero batch,
  zero M/N, empty K, M/N/vector tails, transposed-weight recognition, and checked overflow.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [singleton glossary](../../../../glossary.md#term-singleton); [broadcasting glossary](../../../../glossary.md#broadcasting); [vector glossary](../../../../glossary.md#term-vector); [scalar glossary](../../../../glossary.md#term-scalar); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [axis glossary](../../../../glossary.md#term-axis); [weight glossary](../../../../glossary.md#term-weight); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Cover all 13 ordered admitted type pairs, BFLOAT16/FLOAT32 accumulation distinctions, final
  BFLOAT16 rounding, INT32/INT64 modular overflow, mixed-width promotion, positive floating zero,
  NaN/infinity where applicable, arbitrary resolved positive-stride layouts, all heap, all segment,
  and representative mixed carriers.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [rounding glossary](../../../../glossary.md#term-rounding); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [stride glossary](../../../../glossary.md#term-stride); [layouts glossary](../../../../glossary.md#layout); [heap glossary](../../../../glossary.md#term-heap))</span>
- Test exact vector eligibility and fallback for every type, mixed width, unit/non-unit N stride,
  species/tail, fused terminal, and small/large threshold boundary at 4,096, 16,384, and 65,536
  checked multiply-adds. Assert at most four realization identities and stable repeatable selection.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
- Test output/input/bias/materialization overlap rejection before any output mutation or worker
  submission, output injectivity, zero-work no-submission, exact flattened range ownership,
  multiple worker counts, and scalar/parallel parity.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [materialization glossary](../../../../glossary.md#materialization); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Test canonical split construction before recognition, recognition-to-baseline association, and
  that only the MATMUL executable-alternatives disposition relaxes the 0008D barrier. Cover fused
  ADD, fused ADD plus each exact terminal, unsupported/widening bias,
  publication/fan-out/access/alias barriers, canonical transpose only, unchanged non-MATMUL fact
  sets, identical fused/split member and boundary coverage, profitable <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [widening glossary](../../../../glossary.md#term-widening); [publication glossary](../../../../glossary.md#term-publication); [fan-out glossary](../../../../glossary.md#term-fan-out); [alias glossary](../../../../glossary.md#aliasing-and-arena); [transpose glossary](../../../../glossary.md#term-transpose))</span>`CpuFusionDecision`, and
  deterministic split fallback.
- Test direct, one-copy-left, and one-copy-right complete candidates; explicit
  `CO_CONSUMED_PAIR` rejection for a two-copy MATMUL candidate; checked byte and copy cost;
  resource declaration/assignment/binding; reuse; transition publication; direct production
  behavior with tuning disabled; and absence of invocation-local allocation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Test schema-54 compatibility, stale-envelope rejection, specialization equality/hash/bytes,
  deterministic class bytes, cache reuse/collision checks, prepared-plan/finalizer propagation, and
  exact entry descriptor and geometry binding.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [specialization glossary](../../../../glossary.md#term-specialization); [hash glossary](../../../../glossary.md#term-hash); [cache glossary](../../../../glossary.md#term-cache); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>

### Tier 2: generated evidence and performance

- Freeze a bounded ledger covering these eight hot forms: [FLOAT64](../../../../glossary.md#term-float64) [vector](../../../../glossary.md#term-vector)-vector `K=257` scalar
  direct; FLOAT32 matrix-vector <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>`[64,127] x [127]` [scalar](../../../../glossary.md#term-scalar) direct; FLOAT64 direct-vector
  `[2,63] x [63,128]`; [INT32](../../../../glossary.md#data-type--datatype) direct-vector `[2,65] x [65,96]`; [BFLOAT16](../../../../glossary.md#term-bfloat16) scalar-tiled
  `[32,63] x [63,48]`; mixed BFLOAT16/[FLOAT32](../../../../glossary.md#term-float32) scalar-tiled `[32,63] x [63,48]`; FLOAT32
  vector-tiled `[32,127] x [127,256]`; and FLOAT64 batched/broadcast vector-tiled
  `[3,16,127] x [1,127,256]`. Add one FLOAT32 bias-only fused scalar control and one terminal
  split control without expanding the timed ledger beyond ten rows.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
- Attach two named materialization companion comparisons to the corresponding existing ledger
  rows. <span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>`MATERIALIZE-RIGHT-F32` uses left `[32,127]` and the canonical transpose view of a
  contiguous FLOAT32 weight <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [transpose glossary](../../../../glossary.md#term-transpose); [view glossary](../../../../glossary.md#term-view-meanings); [contiguous glossary](../../../../glossary.md#term-contiguous); [weight glossary](../../../../glossary.md#term-weight))</span>`[256,127]`, producing logical right `[127,256]` with [strides](../../../../glossary.md#term-stride)
  `[1,127]`. `MATERIALIZE-RIGHT-F64-BATCH` uses left `[3,16,127]` and the same canonical transpose
  construction from a contiguous FLOAT64 `[256,127]` weight, broadcast across the left batch.
  For each, retain the complete one-copy-right candidate plan and compare it with the optimal
  direct strided plan; these named companion comparisons do not replace the dense direct vector-
  tiled controls or create hidden ledger rows.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- For every ledger row retain specialization and lowering manifests, generated bytes and SHA-256,
  descriptor, constant-pool/member-reference report, Class-File parse report, complete<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [lowering glossary](../../../../glossary.md#lowering); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [descriptor glossary](../../../../glossary.md#term-descriptor); [constant glossary](../../../../glossary.md#term-constant))</span>
  `javap -c -v -p`, forbidden-reference scan, semantic checksum, environment, and direct-oracle
  source/identity facts under one checksummed evidence manifest outside the repository.<span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum); [oracle glossary](../../../../glossary.md#term-test-oracle); [identity glossary](../../../../glossary.md#term-identity-meanings); [manifest glossary](../../../../glossary.md#term-manifest))</span>
- [Run](../../../../glossary.md#term-run) five fresh accepted Java 26 [forks](../../../../glossary.md#term-fork-meanings) with `--enable-preview`,
  `--add-modules jdk.incubator.vector`, and fixed `-Xms1g -Xmx1g`. Each fork uses deterministic
  inputs, randomized generated/direct order, at least five warmup batches, nine measured rounds,
  adaptive batches of at least 25 ms, and semantic equality before timing. Reject a whole fork only
  before measurement for a recorded environmental/control failure; never discard, retry, average
  away, or relabel a measured miss.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [warmup glossary](../../../../glossary.md#term-warmup))</span>
- Every generated/direct row in every accepted fork and every row's aggregate median must satisfy
  generated/direct <span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>`<= 1.15x`. Also time the complete retained one-copy candidate plans for the two
  named materialization controls against their corresponding optimal direct strided plans;
  promotion remains disabled unless every admitted represented case independently satisfies the
  same per-fork and aggregate gate. A failed required gate blocks completion.
- Retain byte-identical controls with schema-54 compatibility envelopes and the existing schema-52
  class-identity projection for representative unchanged pointwise, affine-copy, reduction, and
  Conv2d forms, proving that only compatibility rejection—not their emitted body, binary name, or
  stable projected class identity—changed.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [pointwise glossary](../../../../glossary.md#term-pointwise); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>

### Tier 3: [module](../../../../glossary.md#term-module) [checkpoint](../../../../glossary.md#term-checkpoint)

- Run focused Model MATMUL and Compiler structured-inference/gradient tests only as unchanged
  semantic consumers when needed for source-backed confirmation; do not edit those modules.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Run the complete CPU module test suite exactly once after focused tests and generated evidence
  stabilize. This is proportionate because capability, sealed IR, preparation, artifact schema,
  generated execution, parallel ranges, and materialization all change.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [artifact glossary](../../../../glossary.md#term-artifact); [schema glossary](../../../../glossary.md#term-schema); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Run](../../../../glossary.md#term-run) [CPU](../../../../glossary.md#term-cpu) [Javadoc](../../../../glossary.md#term-javadoc) and render/inspect the affected generated pages after the documentation pass.
- Do not add or run architecture, backend-conformance, or integration suites unless implementation
  discovers an actual boundary, shared-contract, or end-to-end behavior change. Record the
  reasoned no-change conclusion otherwise.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

### Tier 4: final repository checks

- Inspect the complete implementation diff, `git diff --check`, staged/unstaged scope, generated
  or resource leakage, schema/status consistency, Java/package inventory, Markdown links, anchors,
  fences, and final newlines.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [schema glossary](../../../../glossary.md#term-schema))</span>
- Verify [CPU](../../../../glossary.md#term-cpu) 0008F is `Complete` only after implementation, evidence, and the clean targeted
  documentation pass finish; CPU 0008G remains `Draft` with no task file.

## Dependencies

- [CPU 0008E1](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> — Complete.
- [CPU 0008E](0008e-bounded-multi-input-materialization-and-representation-reuse.md) — Complete.
- [CPU 0008D](0008d-bounded-fusion-profitability-and-typed-decision-facts.md) — Complete.
- [CPU 0008C](0008c-typed-specialized-subgraph-and-epilogue-recognition.md) — Complete.
- [CPU 0008B](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md) — Complete.
- [Model 0019](../../../modules/model/tasks/0019-matmul-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span> — Complete.
- [Model 0019D](../../../modules/model/tasks/0019d-linear-convenience.md) — Complete.
- [Compiler 0005D](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span> — Complete.

## Follow-up tasks

- CPU 0008G remains the next Draft frontier and owns portable Max/Average Pool2d execution. Do not
  create its detailed specification in this task.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Later native-route work may add provider-backed MATMUL as a peer route after portable truth and
  evidence are complete.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Later tuning may promote exact materialized candidates from compatible complete-plan evidence.
- A separate future MATMUL optimization task may consider cache K blocking or panel packing. Any
  parallel K split requires its own partial-result, combine, numerical, determinism, and resource
  contract and is not implied by this task.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>

## Architecture impact

None expected. This task implements a new [CPU](../../../../glossary.md#term-cpu)-private portable family within the existing
[backend-owned lowering](../../../../glossary.md#backend-owned-lowering), declared-resource, generated-[artifact](../../../../glossary.md#term-artifact), caller-owned-worker, and [Runtime](../../../../glossary.md#term-runtime)
prepared-execution boundaries. It changes no allowed [dependency](../../../../glossary.md#term-dependency) and therefore should require no
architecture document, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), or architecture-test update. Stop for clarification if implementation
proves otherwise.

## Implementation prompt

Implement [CPU](../../../../glossary.md#term-cpu) task 0008F in a separate clean implementation context. Read `AGENTS.md`,
`ARCHITECTURE.md`, the current architecture plan, [planning](../../../../glossary.md#term-planning) guide and roadmap, CPU master plan,
tasks 0008B–0008E1, this specification, [Model](../../../../glossary.md#term-model) 0019/0019D, [Compiler](../../../../glossary.md#term-compiler) 0005D, documentation rules and
profiles, and all directly affected CPU source/tests in full before editing.

Add the exact cold [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) geometry, complete non-[BOOL](../../../../glossary.md#data-type--datatype) numeric [lowering](../../../../glossary.md#lowering), four bounded realization
kinds, generated [scalar](../../../../glossary.md#term-scalar) fallback, narrow N-[vector](../../../../glossary.md#term-vector) eligibility, optional M/N [tiling](../../../../glossary.md#tiling), deterministic
output-tile parallel ranges, exact 0008C [fused](../../../../glossary.md#specialization-and-fusion)/split epilogues, and explicit 0008E [materialization](../../../../glossary.md#materialization)
candidates described here. Keep full-K accumulation in every output microtile; do not add K
blocking, K splitting, native calls, hidden packing/allocation, hot-path policy, or per-[Shape](../../../../glossary.md#shape) class
[identity](../../../../glossary.md#term-identity-meanings). Advance [schema](../../../../glossary.md#term-schema) to 54 and preserve unchanged generated controls.

Implement and [run](../../../../glossary.md#term-run) the focused semantic, legality, representation, preparation, Class-File,
execution, and performance evidence; then run exactly one full CPU [checkpoint](../../../../glossary.md#term-checkpoint). Use an optimal clean
Java implementation of each specialized case as the [oracle](../../../../glossary.md#term-test-oracle). Retain the complete checksummed
evidence outside the repository. After executable behavior stabilizes, hand the diff to a distinct
clean documentation-focused context to finalize Javadocs, the CPU guide/glossary impact, this task,
the master plan, and roadmap without repeating successful Java tests unless documentation changes
executable Java.

Do not stage, [commit](../../../../glossary.md#term-commit), or push unless the user separately authorizes it. Do not mark implementation
Complete until every acceptance criterion, evidence gate, documentation gate, exact-[scope](../../../../glossary.md#term-scope) check,
and whitespace check passes. If a required architecture/shared-[module](../../../../glossary.md#term-module) change or a failed fixed
performance gate appears, stop and report it rather than [widening](../../../../glossary.md#term-widening) scope or changing policy
silently.

## Local decisions

- Full-K traversal is selected for this first implementation. M/N output tiling is cache-oriented
  coarse work decomposition; it does not imply K blocking.<span data-glossary-navigation="true"> ([tiling glossary](../../../../glossary.md#tiling); [cache glossary](../../../../glossary.md#term-cache); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- The task has exactly four realization identities and fixed 2-by-2 scalar or 2-by-2-species vector
  microtiles. Conservative work thresholds are explicit cold policy, not hardware cache claims.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector))</span>
- Vector eligibility is intentionally limited to same-type FLOAT32/FLOAT64/INT32/INT64 with unit
  right/result N stride. BFLOAT16, mixed-width, gather/scatter, and terminal vectorization use the
  scalar or split fallback.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [stride glossary](../../../../glossary.md#term-stride); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [gather glossary](../../../../glossary.md#gather))</span>
- Whole-logical-input affine copies are the only admitted packing-equivalent representation.
  Panel packing is deferred.
- Bias and terminal fusion consume CPU 0008C facts exactly. Linear remains visible Model/Compiler
  composition; CPU invents no new semantic operation.<span data-glossary-navigation="true"> ([Bias glossary](../../../../glossary.md#term-bias); [fusion glossary](../../../../glossary.md#term-fusion); [CPU glossary](../../../../glossary.md#term-cpu); [Linear glossary](../../../../glossary.md#term-linear-meanings); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [operation glossary](../../../../glossary.md#operation))</span>
- Schema 54 is required because the generated artifact family/form changes. Unchanged emitted
  controls remain byte-identical.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [artifact glossary](../../../../glossary.md#term-artifact))</span>

## Known limitations

- The portable kernels are deliberately not a vendor-BLAS replacement and make no hardware-
  specific cache-size promise.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel); [BLAS glossary](../../../../glossary.md#term-blas))</span>
- Scalar tails and scalar BFLOAT16/mixed-width/general-layout paths may leave later optimization
  opportunities. They are nevertheless required production semantics and safe fallback.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [layout glossary](../../../../glossary.md#layout))</span>
- No [cache](../../../../glossary.md#term-cache) K blocking or packed panels are available, even for very large K.
- Terminal [activations](../../../../glossary.md#term-activation)/CLAMP are [scalar](../../../../glossary.md#term-scalar)-[fused](../../../../glossary.md#specialization-and-fusion) or split; they are not [vector](../../../../glossary.md#term-vector)-fused in this task.
- Ordinary preparation does not promote 0008E materialization without fresh complete-plan proof;
  candidate existence is not a tuning claim.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The clean implementation context ran the exact 17 named focused baseline suites after executable
  code stabilized. The retained XML reports 17 suites, 182 tests, zero failures, zero errors, and
  zero skips.
- The clean implementation context then ran the one required full CPU checkpoint. The retained XML
  reports 109 suites, 576 tests, zero failures, zero errors, and three expected opt-in skips.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- The same implementation context ran `./gradlew :backends:cpu:javadoc`; it passed before this
  final documentation review. The documentation context changed Javadoc only, not executable Java,
  so it reused both successful Java test runs and regenerated final CPU Javadoc afterward.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Raw Class-File parsing, complete `javap -c -v -p`, specialization/member reports, and forbidden
  generated-member/allocation scans cover the ten frozen kernel rows and two named materialization
  companions. Both forbidden scans have zero findings. Unchanged pointwise, affine-copy,
  reduction, and Conv2d controls retain their schema-52 class projection, binary names, and exact
  class bytes while compatibility advances to schema 54.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [materialization glossary](../../../../glossary.md#materialization); [pointwise glossary](../../../../glossary.md#term-pointwise); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [schema glossary](../../../../glossary.md#term-schema); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Five immutable Java 26.0.1 fixed-heap attempt-16 forks passed semantic checks and every required
  per-fork and aggregate generated/direct <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [heap glossary](../../../../glossary.md#term-heap); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>`<= 1.15x` gate. [Kernel](../../../../glossary.md#kernel) aggregate medians are
  `0.970930`, `0.989785`, `0.991573`, `0.990020`, `0.885035`, `0.990787`, `0.988615`,
  `0.946980`, `0.996487`, and `0.991202`; the worst individual kernel ratio is `1.002573`.
  Complete one-copy-right materialization companion medians are `0.351500` and `0.582291`.
  These measurements retain the candidates but do not change ordinary direct-selection policy.
- Final retained evidence is
  `/private/tmp/synaptik-cpu-0008f-retained-evidence-attempt-16-20260829`: 212 checksummed files;
  `MANIFEST.sha256` [SHA-256](../../../../glossary.md#term-sha-256)
  `a88806a9118c1a967ecc77eaf6da9582d15afa959e8368348a0d2ba2a47d4b61`; retained
  `CpuMatmulPerformanceTest.java` SHA-256
  `7ef10929c059652ced68345af5e1d245e92a01574e999080748546262392cc2e`. Independent documentation
  review reran `sha256sum -c MANIFEST.sha256`; all 212 entries passed. Attempt 15 is superseded and
  is not completion evidence.
- Final [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) evidence at `/private/tmp/synaptik-cpu-0008f-final-jit-20260829` shows independent
  337–868-byte direct geometry/range bodies and generated bodies invoked through the same typed
  carriers, mutable <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>`long[]` geometry, and `long start`/`long end` boundary. Some bodies inline and
  others do not under the recorded JIT policy; parity does not require identical inlining or
  bytecode.<span data-glossary-navigation="true"> ([inlining glossary](../../../../glossary.md#term-inlining); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Clean documentation context `/root` independently reviewed the complete diff, every changed
  production contract and MATMUL test, the retained evidence, CPU guide, glossary, architecture,
  architecture tests, capability/schema contracts, task, master plan, and roadmap. It finalized
  affected Javadocs, the CPU MATMUL explanation, the existing MATMUL glossary entry, and planning
  evidence/status without changing executable behavior.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [planning glossary](../../../../glossary.md#term-planning))</span>
- Final documentation validation: `./gradlew :backends:cpu:javadoc`; repository-local Markdown
  link/anchor, fence, and final-newline checks; exact 0008F/0008G status checks; changed-path scope
  inventory; <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`; and trailing-whitespace checks all passed. Exact command outcomes
  are recorded in the documentation context's completion summary.
- Architecture impact remains none. `ARCHITECTURE.md`, the current architecture explanation,
  ADRs, architecture tests, shared modules, Gradle configuration, backend conformance, integration
  tests, public Tensor/API guides, and the low-level OpenBLAS provider remain unchanged because the
  capability stays within the existing CPU-owned lowering, generation, resource declaration,
  finalization, and Runtime invocation boundaries.<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Implementation notes

- CPU capability and lowering now cover vector, matrix, batched, and right-aligned batch-broadcast
  MATMUL across all thirteen ordered non-BOOL numeric promotions. Checked cold geometry preserves
  rank-one axis removal, arbitrary admitted positive strides, exact carriers, injective output,
  alias safety, and empty domains.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [vector glossary](../../../../glossary.md#term-vector); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [BOOL glossary](../../../../glossary.md#data-type--datatype); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [axis glossary](../../../../glossary.md#term-axis); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Generated [schema](../../../../glossary.md#term-schema)-54 execution selects exactly `DIRECT_SCALAR`, `DIRECT_N_VECTOR`,
  `TILED_SCALAR_2X2`, or `TILED_N_VECTOR_2X2`. Flattened ranges own exact disjoint output cells,
  complete rows, or M/N tiles. Every accumulator performs one increasing full-K traversal; there
  is no K split, partial result, combine, panel pack, hidden allocation, or hot-path policy.<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type))</span>
- Exact 0008C FLOAT32/FLOAT64 bias and optional one-terminal facts can form one fused scalar unit;
  the canonical split remains complete and is selected on any uncertainty. Vector forms allow
  bias-only fusion but retain terminal suffixes as split work.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [bias glossary](../../../../glossary.md#term-bias); [fused glossary](../../../../glossary.md#specialization-and-fusion); [scalar glossary](../../../../glossary.md#term-scalar); [canonical glossary](../../../../glossary.md#term-canonical-form); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Direct and at most one whole-value affine-copy input representation are complete candidates.
  Two co-consumed copies remain rejected, and ordinary preparation remains direct despite the two
  passing retained companion measurements because no general automatic promotion policy was
  proved.
- The optimal clean Java oracle now uses typed carriers, mutable primitive geometry, and long
  range bounds at the same shape-polymorphic specialization boundary as generated code. This is
  the task's algorithm and avoidable-overhead oracle, not a byte-identity or guaranteed-inlining
  requirement.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [mutable glossary](../../../../glossary.md#term-mutable); [shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization); [identity glossary](../../../../glossary.md#term-identity-meanings); [inlining glossary](../../../../glossary.md#term-inlining))</span>

## Completion summary

- Completed changes: implemented the complete portable static MATMUL family, four bounded
  realizations, exact full-K work ownership, bounded linear epilogues/split fallback, schema-54
  artifacts, representation candidates, prepared execution, and retained semantic/structural/
  performance evidence; finalized all required documentation and status records.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [linear glossary](../../../../glossary.md#term-linear-meanings); [schema glossary](../../../../glossary.md#term-schema); [artifacts glossary](../../../../glossary.md#term-artifact); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- Files changed or created: 19 CPU production/Javadoc paths, 18 CPU test paths, and five
  documentation/planning paths (this task, CPU master plan, roadmap, CPU backend guide, and the
  existing MATMUL glossary entry); no generated artifact or evidence file was added to the
  repository.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend))</span>
- Tests and validation: 17 focused suites/182 tests and the one 109-suite/576-test CPU checkpoint
  passed in the implementation context; final CPU Javadoc, attempt-16 manifest verification,
  Markdown/status/scope/format checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [manifest glossary](../../../../glossary.md#term-manifest); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed in the documentation context.
- Documentation-agent review: mandatory clean documentation context `/root` completed the targeted
  API-Javadoc, backend-guide, general, and planning-profile review without executable Java edits.
- Documentation impact: added the current portable MATMUL section to the CPU backend guide and
  synchronized task, master-plan, and roadmap status/evidence. No other explanatory or API guide
  needs a change because no other public API, workflow, or shared contract changed.
- Javadoc review: corrected MATMUL emitter work-unit semantics, prepared-executable binding/range
  ownership, and preparation-plan constructor contracts. All other changed Java Javadocs remain
  accurate for behavior, invariants, inputs, results, failures, ownership, mutation, range, and
  concurrency boundaries.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Glossary impact: no new reusable domain term was introduced. The existing MATMUL entry was
  updated because CPU execution support changed; all other glossary definitions remain accurate.
- Architecture impact: none. The authoritative contract and current architecture plan remain
  accurate; there is no ADR or architecture-test change because module ownership and dependency
  rules did not change.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Public [API](../../../../glossary.md#term-application-programming-interface-api) impact: no API shape changed. `CpuCapabilityProvider` truthfully changes reported
  support for the implemented static MATMUL subset; Model Tensor/API and Compiler contracts remain
  unchanged and need no documentation edit.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Unresolved issues: None.
- Follow-up required: None for CPU 0008F. CPU 0008G remains the next Draft frontier without a
  detailed specification.

Status: Complete
