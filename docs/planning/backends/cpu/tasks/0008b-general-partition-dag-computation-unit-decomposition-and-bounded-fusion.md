# Task 0008B: General [Partition](../../../../glossary.md#partition)-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) Computation-Unit [Decomposition](../../../../glossary.md#term-decomposition) and Bounded [Fusion](../../../../glossary.md#term-fusion)

## Reader context

This task records replacement of the Conv2d-only sequence with `CpuPartitionDagDecomposer` and
`CpuPreparedPartitionExecutable`. Established numerical families remain indivisible seeds; only
ordinary `CpuKernelIr` pointwise work gains bounded vertical and horizontal fusion. The maximally
split plan is the semantic fallback. Cross-unit `ValueId` edges become ordinary buffers, while
family workspace remains explicit run scratch, not persistent model state or optional external-read
copying.

Here, finalization resolves all assignments before artifact lookup, and composite binding validates
every child before writes. Hot execution follows fixed topological order with child completion
between calls; Runtime owns the atomic validity transition. The two-store Java oracle tests
semantic/performance equivalence, while Class-File inspection proves direct primitive structure.

The diagram describes cold decomposition and resource declarations. Its dependency edges do not
create a run-time scheduler: bound units execute in fixed order.

Current-topic navigation (not a replacement for the historical record):

- [current CPU preparation and invocation handoffs](../../../../explanation/packages/cpu.md#read-the-pipeline-before-the-packages) — Names the current analysis, shared assignment, finalization, cold binding, and Runtime participants.
- [current generated-class boundary](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit) — Explains emitters, typed entries, hidden classes, and the limits of Class-File inspection.

## Status

Complete

## Goal

Replace the [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend)'s straight-line/single-unit whole-[partition](../../../../glossary.md#partition) assumption and the closed
[Conv2d](../../../../glossary.md#convolution--conv2d) two-unit exception with one bounded CPU-private [model](../../../../glossary.md#term-model) for decomposing a complete CPU-owned
partition [directed acyclic graph](../../../../glossary.md#term-directed-acyclic-graph-dag) ([DAG](../../../../glossary.md#intermediate-representation-dag-and-ssa)) into ordered computation units. Fuse only the currently
proved [pointwise](../../../../glossary.md#term-pointwise) forms across legal vertical and horizontal edges, retain established affine-[view](../../../../glossary.md#term-view-meanings)
and specialized-family units as indivisible [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), and declare every remaining inter-unit edge as
one ordinary logical-value [buffer](../../../../glossary.md#term-buffer) before shared assignment.

Return one [atomic](../../../../glossary.md#resource-transaction-and-rollback) partition executable. Cold [binding](../../../../glossary.md#term-binding) must validate every child, resource, [carrier](../../../../glossary.md#term-storage-carrier),
span, worker requirement, and cross-unit [alias](../../../../glossary.md#aliasing-and-arena) constraint before any write. Hot execution must [run](../../../../glossary.md#term-run)
already-bound units in deterministic [topological order](../../../../glossary.md#term-topological-order), completing each unit and joining its
workers before the next unit starts. Illegal, unsupported, or hard-budget-exceeding [fusion](../../../../glossary.md#term-fusion) must
retain the complete deterministic materialized-split [topology](../../../../glossary.md#term-topology); it must not make an otherwise
individually supported partition fail.

## Motivation and [mental model](../../../../glossary.md#term-mental-model)

The current [CPU](../../../../glossary.md#term-cpu) implementation has two incompatible closed forms:

```text
ordinary partition                     CPU 0008 exception

partition -> one unit                  Conv2d unit -> buffer -> pointwise unit
          -> one artifact                         -> two artifacts
          -> one executable                       -> one atomic two-child executable
```

CPU 0008B replaces the [cardinality](../../../../glossary.md#term-cardinality) exception with a single [model](../../../../glossary.md#term-model):

```text
complete CPU-owned partition DAG
              |
              v
  deterministic atomic seed units
              |
              +-- legal bounded pointwise vertical fusion
              +-- legal bounded pointwise horizontal fusion
              |
              v
  stable topological unit list + dependency indices
              |
              +-- external read buffers
              +-- one ordinary Buffer(ValueId) per materialized DAG edge/publication
              +-- exact unit-local workspaces with partition-unique requirement IDs
              |
              v
  one CPU-private atomic partition executable
```

"Materialized split" in this task means an ordinary `PreparationResourceRequirement.Buffer` for
an existing [graph](../../../../glossary.md#graph) [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) crossing computation units. It is not CPU 0005D's optional [contiguous](../../../../glossary.md#term-contiguous)
copy of one external read boundary into an anonymous workspace, and it is not CPU 0008E's later
bounded choice among external read-boundary representation variants.

## [Scope](../../../../glossary.md#term-scope)

### Complete [partition](../../../../glossary.md#partition)-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [validation](../../../../glossary.md#term-validation) and deterministic [topology](../../../../glossary.md#term-topology)

- Analyze exactly one complete partition whose [owner](../../../../glossary.md#term-owner) is `CpuCapabilityProvider.CPU_BACKEND_ID`.
  Retain the existing hard maximum of eight compiled nodes. Empty partitions, duplicate node or
  output identities, missing projected values or memory facts, cycles, consumers preceding their
  producers in the supplied complete context, and a node not independently supported by a current
  CPU lowering family fail before declarations or artifact access.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [lowering glossary](../../../../glossary.md#lowering); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Build [CPU](../../../../glossary.md#term-cpu)-private [producer](../../../../glossary.md#term-producer), consumer, [graph](../../../../glossary.md#graph)-[publication](../../../../glossary.md#term-publication), and [constant](../../../../glossary.md#term-constant) facts from the complete
  [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext). Do not query Runtime state, perform a hidden graph rewrite, or mutate the
  compiled nodes.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Treat the supplied node order as the validated stable topological order. Stable node ordinal,
  output ordinal, input ordinal, and then <span data-glossary-navigation="true"> ([topological order glossary](../../../../glossary.md#term-topological-order))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) only as a final identity check determine all
  lists; do not depend on hash iteration or object identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>
- Create the maximally split baseline first. Each seed is the smallest complete occurrence already
  accepted by current CPU lowering: one ordinary pointwise node; one established affine-view
  chain when its current all-affine straight-line rules admit the chain; one current one-node
  movement, indexing, scatter, fold, ordering, random, scan, aggregate, normalization, batch-
  normalization, Conv2d, or Conv3d family; the exact CPU 0008A four-node Conv1d composition; or an
  already proved closed current Conv2d fused occurrence. Do not split the interior of an
  established numerical or multi-output semantic unit.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [occurrence glossary](../../../../glossary.md#term-occurrence); [pointwise glossary](../../../../glossary.md#term-pointwise); [view glossary](../../../../glossary.md#term-view-meanings); [scan glossary](../../../../glossary.md#term-cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Conv1d composition glossary](../../../../glossary.md#conv1d-composition); [fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
- If more than one seed could claim a node, select the longest already-established closed
  occurrence; ties use the smallest first node ordinal and then the existing family dispatch
  order in <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>`CpuPartitionLowering`. This rule preserves the current Conv1d and direct/fused Conv2d
  routes without introducing CPU 0008C recognition.
- Assign final unit indices with stable Kahn ordering: repeatedly select the ready unit whose
  smallest member-node ordinal is lowest. Each unit records the strictly increasing indices of
  its direct producer units. The stored order must already satisfy every dependency; execution is
  not permitted to discover or schedule dependencies dynamically.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

### Legal vertical and horizontal [fusion](../../../../glossary.md#term-fusion)

- [CPU](../../../../glossary.md#term-cpu) 0008B admits new fusion only when every [fused](../../../../glossary.md#specialization-and-fusion) [operation](../../../../glossary.md#operation) lowers to the existing ordinary
  `CpuKernelIr` pointwise instruction/store vocabulary. It may generalize the current pointwise
  straight line to a pointwise DAG with multiple stores. It does not fuse through affine-copy,
  movement, indexing, scatter, fold, ordering, random, scan, reduction, softmax, normalization,
  batch-normalization, Conv2d, Conv3d, or exact Conv1d-composition seeds.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- A vertical edge is eligible only when one pointwise producer value has exactly one in-partition
  consumer, is not a graph output, and the producer and consumer units have one identical checked
  iteration domain, compatible result/input data type, compatible normalized access mapping, and
  no other dependency cycle after contraction. The fused IR must preserve the current operation
  order, typed intermediate semantics, scalar-power realization, canonical BOOL behavior, and
  final stores.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [partition glossary](../../../../glossary.md#partition); [graph glossary](../../../../glossary.md#graph); [data type glossary](../../../../glossary.md#data-type--datatype); [dependency glossary](../../../../glossary.md#term-dependency); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [scalar-power realization glossary](../../../../glossary.md#scalar-power-realization); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- A horizontal edge is a CPU-private fusion relation between two dependency-independent
  pointwise units. It is eligible only when both are ready from the same predecessor set, have the
  same checked iteration domain and execution strategy, have disjoint outputs, neither consumes
  the other's output, and one combined pointwise IR can preserve both branches in stable member-
  node order. Shared external inputs are deduplicated by <span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot); final stores remain ordered by
  producer node and output ordinal.
- Consider vertical pairs first in producer-unit/member-node order and then horizontal pairs in
  left-unit/right-unit order. After a successful contraction, restart vertical consideration from
  the first unit. After vertical convergence, process horizontal pairs and restart at the first
  pair after each contraction. Test one pair at most once for one unchanged topology identity;
  contraction creates a new topology identity. Stop when no legal within-budget contraction
  remains or the candidate-attempt ceiling is reached.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Selection in this task is deterministic maximal legal fusion: accept the first legal pair that
  satisfies every hard budget. There is no cost comparison, profitability threshold, candidate
  ranking, typed accepted/rejected fact model, or tuning input. CPU 0008D owns those concerns.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

### [Fusion](../../../../glossary.md#term-fusion) barriers and [fail-closed](../../../../glossary.md#term-fail-closed) semantics

- A value with more than one in-partition consumer is a vertical-fusion barrier. Independent
  consumers may still be horizontally fused if their own horizontal rules hold, but the producer
  value remains a materialized split boundary.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [fused glossary](../../../../glossary.md#specialization-and-fusion); [producer glossary](../../../../glossary.md#term-producer))</span>
- A graph-output/publication obligation is a fusion barrier across that value. A fused unit may
  have multiple final graph-output stores, but CPU 0008B does not keep a published intermediate
  virtual and does not change publication identity or order.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication); [CPU glossary](../../../../glossary.md#term-cpu); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Any unresolved or potentially overlapping write/read alias relation, non-injective output,
  negative or out-of-span address, carrier incompatibility, or mismatch in normalized access
  geometry is a fusion barrier. Existing one-unit family-specific alias rules remain authoritative.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [injective glossary](../../../../glossary.md#term-injective); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Explicit state transition, random/counter advancement, dropout, or any other stateful occurrence
  is an indivisible seed and a barrier on every incident edge. It executes exactly once in original
  topological order.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [occurrence glossary](../../../../glossary.md#term-occurrence); [seed glossary](../../../../glossary.md#term-seed); [topological order glossary](../../../../glossary.md#term-topological-order))</span>
- Reduction, scan, ordering, softmax, normalization, batch-normalization, convolution, fold, and
  scatter units are numerical-order or family-algorithm barriers. Their current loop order,
  accumulation, rounding, tie, state, and multi-output contracts must remain unchanged. No
  pointwise epilogue recognition is inferred around them in this task.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [rounding glossary](../../../../glossary.md#term-rounding); [pointwise glossary](../../../../glossary.md#term-pointwise); [epilogue glossary](../../../../glossary.md#term-epilogue))</span>
- Affine-view chains remain governed by CPU 0006. A current legal unpublished straight-line affine
  chain may remain one seed; branching, publication, and mixed affine/pointwise edges materialize.
  CPU 0008B does not invent a mixed view/pointwise generated form.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings))</span>
- A fusion-lowering exception, unsupported carrier/access combination, failed hard-budget proof,
  or exhausted fusion-attempt budget rejects only that contraction and retains the already valid
  split units. Failure of the maximally split baseline itself fails the partition before resource
  declaration; there is no Runtime fallback and no partial plan.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

### Exact hard budgets

The following are semantic/resource safety ceilings, not profitability scores:

| Budget | Ceiling | Counting rule | On exceedance |
| --- | ---: | --- | --- |
| [Partition](../../../../glossary.md#partition) [nodes](../../../../glossary.md#node) | 8 | Complete compiled nodes, preserving the existing cap | Fail before declarations |
| Final units | 8 | Stable computation units after contraction | Retain split; fail only if the baseline exceeds the node-derived cap |
| [Fan-out](../../../../glossary.md#term-fan-out) considered for [fusion](../../../../glossary.md#term-fusion) | 7 | Distinct in-partition consumers of one value | Materialize; vertical fusion requires exactly one |
| Fusion attempts | 28 | Every tested vertical or horizontal unit pair, successful or rejected | Stop contracting and retain remaining split units |
| Nodes per newly [fused](../../../../glossary.md#specialization-and-fusion) [pointwise](../../../../glossary.md#term-pointwise) unit | 8 | Member compiled nodes | Retain split |
| Materialized boundaries per unit | 16 | Distinct non-virtual input and output [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot)s | Retain split |
| Simultaneously live [IR](../../../../glossary.md#term-intermediate-representation-ir) values | 16 | Maximum live set from stable instruction/store [liveness](../../../../glossary.md#term-liveness), including materialized inputs and pending stores | Retain split |
| Indexing-complexity units | 32 | Sum per materialized [binding](../../../../glossary.md#term-binding): `DENSE_LINEAR`/`SCALAR_ALL_ZERO` = 1, `LAST_AXIS_BIAS` = 2, `BLOCK_OUTER` = 3, `GENERAL_ODOMETER` = 4 | Retain split |
| Generated-code-size units | 64 | 8 base + 4 per pointwise instruction + 3 per store + indexing-complexity units + 1 per virtual value | Retain split |

- Apply the last five ceilings only to a newly proposed pointwise contraction. Established atomic
  family forms retain their own completed bounds and validation; CPU 0008B must not retrospectively
  reject them using a pointwise estimator.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Compute liveness after stable IR construction: an input becomes live at first use, a virtual
  value at definition, and a value dies after its last instruction or store use. Multiple uses by
  one instruction count once. Checked arithmetic overflow rejects the contraction and retains the
  split form.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- `generated-code-size units` is an analysis-time structural estimator, not a claim about exact
  Class-File byte length. The existing artifact-store byte limit remains independently enforced.
  Analysis must not generate a class to decide fusion.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Keep the existing per-unit `CpuSpecializationBudget(4, 1, 0, 0)`: at most four representation
  candidates, one realized artifact, and no fixed-shape or unroll variants for each selected unit.
  A partition may therefore realize at most eight artifacts and never more than one artifact per
  final unit.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>

### Materialized split resources and workspaces

- After final [topology](../../../../glossary.md#term-topology) selection, derive resources once. Declare each distinct materialized [graph](../../../../glossary.md#graph)
  [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) exactly once as `PreparationResourceRequirement.Buffer`, with checked referenced span
  times data-type width and alignment equal to the data-type width. Order declarations by first
  unit use in stable unit order, inputs before outputs within a unit; deduplicate by <span data-glossary-navigation="true"> ([referenced span glossary](../../../../glossary.md#referenced-element-span); [alignment glossary](../../../../glossary.md#term-alignment))</span>`ValueId` and
  reject conflicting geometry.
- Every value crossing two final units is materialized even if it is not a graph output. It is an
  ordinary logical-value buffer, visible in both units' boundary lists, written by exactly one
  producer unit, and read by every dependent unit. It is never a workspace, anonymous temporary,
  hidden generated allocation, or Runtime-created value.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [producer glossary](../../../../glossary.md#term-producer); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Preserve exact graph publications and external reads. [Partition](../../../../glossary.md#partition)-level access is `READ_ONLY` only
  for values no unit writes and `WRITE_ONLY` for every value written by any unit, including
  internal split buffers and graph outputs. `READ_WRITE` remains forbidden.
- Move current execution/resource geometry into complete per-unit facts. Each unit owns its route,
  IR, specialization, boundary bindings/carriers, strategy/ranges, optional CPU 0005D
  materialization, optional exact workspace and purpose, affine pairs, and applicable family
  geometry. The partition plan owns only the ordered unit/dependency topology and deduplicated
  partition resource/access view; it must not mirror first-unit geometry as partition truth.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [specialization glossary](../../../../glossary.md#term-specialization); [bindings glossary](../../../../glossary.md#term-binding); [CPU glossary](../../../../glossary.md#term-cpu); [materialization glossary](../../../../glossary.md#materialization); [dependency glossary](../../../../glossary.md#term-dependency); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Preserve CPU 0005D exactly for a selected one-unit plan. Disable its optional external-read
  contiguous-copy selection when the final topology has more than one unit; multi-unit external
  read representation choices belong to CPU 0008E. This does not disable workspaces intrinsically
  required by scatter, ordering, reductions, Layer normalization, or batch-normalization units.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [Layer normalization glossary](../../../../glossary.md#layer-normalization); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Rebase every unit-local workspace requirement to partition-unique ID equal to its final unit
  index. Units without a workspace leave an unused ID. Declare workspaces in unit order with the
  exact existing byte size/alignment and carry the rebased ID through unit-local materialization
  or family geometry. Workspace bytes remain run-owned; no generated code allocates them.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [run glossary](../../../../glossary.md#term-run))</span>
- Each child receives only its exact assigned workspace selection and validates it cold. The
  composite makes no workspace-alias assumption; if the shared plan legally reuses storage, strict
  sequential execution ensures two unit workspaces are never accessed concurrently.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>

### Preparation, finalization, and one [atomic](../../../../glossary.md#resource-transaction-and-rollback) executable

- Replace `PlanForm.ONE_UNIT`/`CONV2D_MATERIALIZED_SUFFIX` cardinality inference with a general
  validated partition form covering one through eight units. The current Conv2d materialized
  suffix must be expressed through the same unit/dependency/resource model, with no special
  preparer or finalizer branch.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality); [inference glossary](../../../../glossary.md#term-inference); [partition glossary](../../../../glossary.md#partition); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [dependency glossary](../../../../glossary.md#term-dependency); [model glossary](../../../../glossary.md#term-model))</span>
- Finalization resolves the exact complete buffer and workspace assignment sets before the first
  artifact-store lookup. It then realizes exactly one already-selected artifact per unit in unit
  order. It cannot add a buffer/workspace, change a unit boundary, fuse/split, change strategy, or
  select another route after assignment.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Replace the exact two-child `CpuPreparedExecutableSequence` with a narrowly CPU-private general
  partition composite. It retains an immutable ordered child list and immutable dependency facts;
  it is not a Runtime graph interpreter, scheduler, public executable DAG, or service registry.
  One-unit plans may continue to return the direct <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [immutable glossary](../../../../glossary.md#term-immutable); [Runtime glossary](../../../../glossary.md#term-runtime); [graph glossary](../../../../glossary.md#graph); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [registry glossary](../../../../glossary.md#term-registry))</span>`CpuPreparedExecutable` when that preserves the
  same externally observable contract.
- Composite cold binding first validates every outer CPU buffer/workspace representation and all
  write/read and write/write overlaps among distinct selected buffer spans. It then binds every
  child into a temporary immutable list in unit order. Any child carrier, span, workspace, worker,
  geometry, or alias failure aborts binding before execution and before any write.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Hot execution invokes the already-bound children in stable unit order. Each child finishes its
  complete selected range set and joins its borrowed worker group before the next begins. On the
  first thrown failure, no later child runs. The composite performs no validity mutation: unchanged
  Runtime surrounds it with one atomic invalid-before/valid-after transition for all partition
  writes, so a failed partition leaves every declared output/split write invalid.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Do not add unit-level parallel scheduling. Existing child-local bounded parallelism remains the
  only concurrency. Dependency indices are validated cold and retained for deterministic topology
  identity, diagnostics, and later planning, not interpreted dynamically at run time.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency); [topology glossary](../../../../glossary.md#term-topology); [identity glossary](../../../../glossary.md#term-identity-meanings); [planning glossary](../../../../glossary.md#term-planning); [run glossary](../../../../glossary.md#term-run))</span>

### Generated-code and performance evidence

- Preserve byte-identical generated artifacts for unchanged one-unit pointwise, affine, Conv1d,
  Conv2d, Conv3d, and other specialized seeds whenever their existing specialization and IR are
  identical. Orchestration-only reuse needs structural/cache-identity regression evidence, not a
  duplicate performance benchmark.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [pointwise glossary](../../../../glossary.md#term-pointwise); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [specialization glossary](../../../../glossary.md#term-specialization); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [regression glossary](../../../../glossary.md#term-regression))</span>
- For the genuinely new multi-store horizontal pointwise IR, implement and retain an optimal clean
  Java oracle with the same typed operations, evaluation order, hot-loop/dataflow shape, stores,
  carrier forms, and avoidable-overhead profile. Inspect generated Class-File/decompilation for
  direct primitive loops and absence of hidden helpers, allocation, boxing, reflection, string
  dispatch, or operation interpretation.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [evaluation glossary](../../../../glossary.md#term-evaluation); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape); [carrier glossary](../../../../glossary.md#term-storage-carrier); [decompilation glossary](../../../../glossary.md#term-decompilation); [dispatch glossary](../../../../glossary.md#term-dispatch); [operation glossary](../../../../glossary.md#operation))</span>
- Run fixed-shape generated-versus-direct evidence for each genuinely new emitted form. Every fork
  and aggregate median must satisfy generated/direct <span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [fork glossary](../../../../glossary.md#term-fork-meanings); [median glossary](../../../../glossary.md#term-median))</span>`<= 1.15x`; never weaken, average away, or
  relabel a failed gate. Use the repository's accepted fork/rejection and evidence-manifest rules.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest))</span>
- A new decomposition that emits only byte-identical existing unit artifacts needs an explicit
  optimal direct-Java whole-partition baseline only when the task makes a performance claim about
  the composite orchestration itself. Otherwise record exact reuse and semantic/resource evidence.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [partition glossary](../../../../glossary.md#partition))</span>

### Documentation and [Javadoc](../../../../glossary.md#term-javadoc)

- After stable implementation and Java/generated-code evidence, use a separate clean
  documentation-focused agent/thread as required by `AGENTS.md` and
  `docs/developer-guide/documentation-rules.md`.
- Finalize meaningful Javadoc for every changed CPU type, constructor, record component, method,
  exception, ownership/lifecycle rule, topology order, budget, resource, and failure boundary.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [topology glossary](../../../../glossary.md#term-topology))</span>
- Update `docs/backend-guide/cpu-backend.md` with the general unit-DAG mental model, deterministic
  pointwise fusion subset, hard barriers/budgets, split-buffer versus external-materialization
  distinction, exact workspace/resource lifecycle, atomic binding/execution behavior, safe
  fallback, and one compact fork/join example.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span><span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [pointwise glossary](../../../../glossary.md#term-pointwise); [fusion glossary](../../../../glossary.md#term-fusion); [buffer glossary](../../../../glossary.md#term-buffer); [materialization glossary](../../../../glossary.md#materialization); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [binding glossary](../../../../glossary.md#term-binding); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- Review the glossary. Add or change an entry only if implementation introduces a reusable term;
  otherwise record a reasoned no-change conclusion. Update this task, the CPU master plan, and the
  roadmap with actual evidence and status in the same overall change.

## Out of [scope](../../../../glossary.md#term-scope)

- CPU 0008C typed specialized-subgraph or epilogue recognition, including new MATMUL,
  convolution, reduction, softmax, or normalization recognition.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [epilogue glossary](../../../../glossary.md#term-epilogue); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- CPU 0008D profitability ranking, estimated complete-plan cost, legal-candidate comparison,
  typed accepted/rejected/selected facts, Trace payloads, or tuning inspection.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [tuning inspection glossary](../../../../glossary.md#tuning-inspection))</span>
- CPU 0008E multi-input external read-boundary materialization, representation variants, reuse
  identities, or re-ranking.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- A public pattern registry, pattern DSL, public fusion API, generic Runtime graph interpreter,
  dynamic unit scheduler, Runtime route selection, or Runtime graph inspection.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [DSL glossary](../../../../glossary.md#term-dsl); [fusion glossary](../../../../glossary.md#term-fusion); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Runtime glossary](../../../../glossary.md#term-runtime); [graph glossary](../../../../glossary.md#graph))</span>
- New Model operation kinds, Compiler rewrites, Planning abstractions, shared Prepare APIs,
  architecture rules, module dependencies, or service/provider boundaries.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [module glossary](../../../../glossary.md#term-module))</span>
- Mixed affine-view/pointwise fusion, fusion through state/random/numerical-order barriers,
  universal <span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>`ConvNd`, new heavy-family kernels, native/vendor routes, measurement, tuning caches,
  packing, reorder, or relaxed numerical modes.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel))</span>
- Hidden graph values, generated heap/native allocation, post-assignment resources, silent
  fallback to interpreted operations, or partial execution after cold validation failure.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [validation glossary](../../../../glossary.md#term-validation))</span>

## Architecture references

- `ARCHITECTURE.md`: complete-partition backend analysis, computation-unit and fusion ownership,
  exact pre-assignment resource declaration, backend finalization, generated-code discipline, one
  partition executable, cold binding, Runtime atomic validity, and hot-path prohibitions.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [backend glossary](../../../../glossary.md#backend); [fusion glossary](../../../../glossary.md#term-fusion); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [binding glossary](../../../../glossary.md#term-binding); [Runtime glossary](../../../../glossary.md#term-runtime); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- `docs/architecture/current-architecture-plan.md`: current module ownership and absence of an
  approved shared-contract change.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- `docs/planning/planning-guide.md`: detailed-task frontier, validation tiers, completion evidence,
  and separate documentation pass.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `docs/planning/backends/cpu/master-plan.md`: ordered CPU 0008B–0008E decomposition,
  recognition, profitability, and representation sequence.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Completed CPU tasks 0005A, 0005D, 0006, 0008, and 0008A: atomic unit reset, external-read
  materialization, affine views, Conv2d two-unit exception, and dimensional convolution closure.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [views glossary](../../../../glossary.md#term-view-meanings); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>

## Architecture constraints

- Planning selects the CPU owner; only CPU analysis lowers, decomposes, fuses, selects, and
  declares exact resources. Shared Planning, Prepare, Runtime, Model, and Compiler remain unaware
  of CPU unit topology.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner); [Prepare glossary](../../../../glossary.md#prepare); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [topology glossary](../../../../glossary.md#term-topology))</span>
- Analysis completes before shared slot assignment. Finalization validates assignments and
  realizes selected artifacts but cannot change topology, route, resources, or specialization.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- [Runtime](../../../../glossary.md#term-runtime) receives one [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) for the partition and owns the atomic access-validity
  transition. The CPU composite owns only cold validation and deterministic invocation of fixed
  already-bound children.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [validation glossary](../../../../glossary.md#term-validation))</span>
- Generated entry points contain direct specialized computation. Neither generated code nor the
  composite may inspect [`Operation`](../../../../glossary.md#operation), `CompiledNode`, graph topology, route policy, or candidate
  policy on the hot path.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- [Buffer](../../../../glossary.md#term-buffer) resources use existing logical [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot)s. Workspace resources are explicit run-owned
  requirements. No public/shared abstraction or architecture-document change is pre-approved.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>

## Package impact

All production [ownership](../../../../glossary.md#term-ownership-meanings) remains inside `backends/cpu`:

- `io.github.pho001.synaptik.backend.cpu.internal.lowering`: add the private deterministic
  partition-DAG decomposition/fusion owner and its hard-budget/topology value types; reuse current
  family lowerers and ordinary pointwise IR construction.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [owner glossary](../../../../glossary.md#term-owner); [topology glossary](../../../../glossary.md#term-topology); [pointwise glossary](../../../../glossary.md#term-pointwise); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`: generalize the partition plan,
  per-unit facts, analysis/resource assembly, and finalization. Do not add a shared Prepare type.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable`: replace the closed Conv2d sequence
  with one CPU-private fixed-unit composite.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.ir`, `.codegen`, `.cache`, and `.memory`: change
  only if the existing multi-store pointwise representation/emitter, structural identity, or cold
  binding requires a narrowly owned correction. Do not introduce another IR hierarchy or cache.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [binding glossary](../../../../glossary.md#term-binding); [cache glossary](../../../../glossary.md#term-cache))</span>

No package move, exported package, new [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), or shared source change is allowed.

## Affected files

Expected implementation paths, refined only within maximum [scope](../../../../glossary.md#term-scope):

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionDagDecomposer.java`
  (new sole topology/fusion owner, including private/nested hard-budget facts)<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [fusion glossary](../../../../glossary.md#term-fusion); [owner glossary](../../../../glossary.md#term-owner))</span>
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedPartitionExecutable.java`
  (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableSequence.java`
  (remove after migrating its contract and tests)
- conditionally, only if the existing multi-store representation/emission is insufficient:
  `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`,
  `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLoopEmitter.java`,
  `CpuScalarEmitter.java`, `CpuVectorInstructionEmitter.java`, and
  `CpuClassFileKernelGenerator.java` in that same emitter package
- focused CPU unit tests for decomposition, plan validation, preparation/resources, finalization,
  composite binding/execution, pointwise lowering/generation, existing Conv2d sequence migration,
  affine/specialized barriers, and inventory<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [decomposition glossary](../../../../glossary.md#term-decomposition); [validation glossary](../../../../glossary.md#term-validation); [binding glossary](../../../../glossary.md#term-binding); [pointwise glossary](../../../../glossary.md#term-pointwise); [lowering glossary](../../../../glossary.md#lowering); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- focused generated-Class-File/direct-Java evidence and manifest resources for genuinely new
  emitted forms<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest))</span>
- `docs/backend-guide/cpu-backend.md`
- this task, `docs/planning/backends/cpu/master-plan.md`, and `docs/planning/roadmap.md`
- glossary only if the documentation review proves a reusable terminology change

Implementation may rename the two proposed new private types only if it records why the final
names better match these exact owners. It may not add a public [`Graph`](../../../../glossary.md#graph), `FusionManager`, [registry](../../../../glossary.md#term-registry),
[DSL](../../../../glossary.md#term-dsl), generic scheduler, or catch-all utility.

## Maximum [scope](../../../../glossary.md#term-scope)

- At most 14 CPU production/Javadoc paths, including the two proposed new types and removal of the
  old sequence type.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- At most 14 [CPU](../../../../glossary.md#term-cpu) test/evidence paths.
- At most four explanatory/planning documentation paths beyond this already-created task: CPU
  guide, glossary if needed, CPU master plan, and roadmap.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- No shared Java, Model, Compiler, Planning, Prepare, Runtime, Engine, Gradle, architecture,
  architecture-test, backend-conformance, or integration-test change.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- One new decomposition owner and one new composite executable are the intended type budget.
  Additional top-level production types, a second IR, or a second planning pass require recorded
  necessity and must remain inside the 14-path production cap; otherwise stop and replan.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [owner glossary](../../../../glossary.md#term-owner); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- If the capability cannot be completed, documented, and validated within these limits, stop with
  `Status: Incomplete`; do not omit a required family [seed](../../../../glossary.md#term-seed), resource, failure path, or evidence gate.

## Acceptance criteria

- One complete CPU-owned DAG of one through eight current supported nodes deterministically
  becomes one through eight validated computation units with stable dependencies and exact
  resource declarations; unsupported or malformed baseline work fails before declarations.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [nodes glossary](../../../../glossary.md#node))</span>
- Existing one-unit pointwise/affine/specialized routes, exact Conv1d, direct/fused Conv2d, direct
  Conv3d, carrier patterns, numerical results, cache identities, and family resources remain
  unchanged when their selected unit IR/facts are unchanged.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [fused glossary](../../../../glossary.md#specialization-and-fusion); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [carrier glossary](../../../../glossary.md#term-storage-carrier); [cache glossary](../../../../glossary.md#term-cache); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- The CPU 0008 published/illegal Conv2d suffix uses the general decomposition model, declares its
  intermediate once, realizes two artifacts, and executes through one atomic composite without a
  Conv2d-specific analysis/finalization branch.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [model glossary](../../../../glossary.md#term-model); [artifacts glossary](../../../../glossary.md#term-artifact); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- A legal `ADD -> GELU -> MUL` chain remains one vertical pointwise unit. A legal pair of
  independent same-domain pointwise branches becomes one multi-store horizontal pointwise unit
  and exactly matches both publications.
- A diamond/fan-out fixture materializes the producer value, may fuse only independently legal
  consumer siblings, and records stable split buffers/unit dependencies. Publication, affine/
  pointwise, Conv3d/pointwise, reduction/pointwise, and dropout/state edges remain deterministic
  materialized barriers.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [fixture glossary](../../../../glossary.md#term-fixture); [producer glossary](../../../../glossary.md#term-producer); [buffers glossary](../../../../glossary.md#term-buffer); [Publication glossary](../../../../glossary.md#term-publication); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- Hard indexing, live-value, generated-size, boundary, node, unit, and attempt budgets are tested
  at and immediately beyond their ceilings. A rejected contraction leaves the complete split
  plan; it never loses a node, edge, output, workspace, or current supported route.
- Every cross-unit [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) is one ordinary buffer and never a workspace. CPU 0005D external
  materialization remains distinguishable and is disabled for multi-unit plans. Two workspace-
  requiring seed units receive exact unique requirement IDs and child-local selections.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [seed glossary](../../../../glossary.md#term-seed))</span>
- Finalization rejects missing, extra, mismatched, or post-analysis buffer/workspace assignments
  before artifact lookup and realizes exactly one artifact per selected unit in stable order.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- Composite binding validates all children and cross-unit aliases before writes. Tests prove an
  invalid later child, undersized later workspace/worker group, read-only output, or overlapping
  distinct output/read span leaves carriers unchanged. Execution proves strict unit completion/
  join order, no later unit after failure, and one outer atomic validity result.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Generated multi-store horizontal pointwise code matches an optimal clean Java oracle, contains
  no hidden operation interpretation/allocation/boxing/reflection/string dispatch, and passes all
  required <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>`<= 1.15x` fork and aggregate gates. Byte-identical reused forms have explicit reuse
  evidence and no redundant benchmark claim.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- Javadocs, CPU guide, glossary conclusion, task evidence/status, master plan, and roadmap are
  independently finalized by the documentation pass. CPU 0008B becomes `Complete` only after all
  evidence passes; CPU 0008C is then the sole [`Ready`](../../../../glossary.md#term-ready) [CPU](../../../../glossary.md#term-cpu) task and no 0008C detail file is created.

## Tests / [validation](../../../../glossary.md#term-validation)

### Tier 1: focused [CPU](../../../../glossary.md#term-cpu) semantics and contracts

- Add focused decomposition tests for stable topology, disconnected ready seeds, vertical and
  horizontal fusion, diamond fan-out, publications, state/random and numerical-order barriers,
  affine/specialized barriers, unsupported baseline failure, deterministic repeatability, and
  every hard-budget edge.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [topology glossary](../../../../glossary.md#term-topology); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [fusion glossary](../../../../glossary.md#term-fusion); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- Extend preparation-plan/preparer/finalizer tests for per-unit facts, dependency validation,
  exact deduplicated buffers, unique workspace IDs, explicit carrier mapping, malformed resource
  sets, artifact order/count, and migrated Conv2d split behavior.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [validation glossary](../../../../glossary.md#term-validation); [buffers glossary](../../../../glossary.md#term-buffer); [carrier glossary](../../../../glossary.md#term-storage-carrier); [artifact glossary](../../../../glossary.md#term-artifact); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- Replace `CpuPreparedExecutableSequenceTest` with general composite tests covering all-units cold
  validation, alias rejection before writes, strict execution/join/failure order, multiple outputs,
  multiple workspaces, and one atomic Runtime validity transition.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Retain focused regression coverage for pointwise, affine views, Conv1d composition, Conv2d,
  Conv3d, reductions, random/dropout, multi-output batch-normalization, and current materialization.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [pointwise glossary](../../../../glossary.md#term-pointwise); [views glossary](../../../../glossary.md#term-view-meanings); [Conv1d composition glossary](../../../../glossary.md#conv1d-composition); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [materialization glossary](../../../../glossary.md#materialization))</span>

[Run](../../../../glossary.md#term-run) once after implementation stabilizes:

```text
./gradlew :backends:cpu:test
```

### Tier 2: generated-code and performance evidence

- Run the CPU generated Class-File/decompilation/import/member-reference checks for the new
  multi-store pointwise form and current reused controls.<span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Run repository-standard fixed-shape performance forks and manifest verification for each new
  emitted form, recording every fork and aggregate <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [manifest glossary](../../../../glossary.md#term-manifest); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>`<= 1.15x` result and any accepted sample
  rejection. Do not substitute an orchestration benchmark for generated/direct evidence.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- [Run](../../../../glossary.md#term-run) [CPU](../../../../glossary.md#term-cpu) [Javadoc](../../../../glossary.md#term-javadoc) after the documentation pass finalizes changed contracts:

```text
./gradlew :backends:cpu:javadoc
```

### Tier 3: documentation and repository hygiene

- Validate Markdown links, anchors, fences, required headings, task statuses, trailing whitespace,
  final newlines, and glossary decision according to the documentation rules.
- Confirm [CPU](../../../../glossary.md#term-cpu) 0008B is the sole detailed [`Ready`](../../../../glossary.md#term-ready) task during implementation; on completion update
  it to `Complete`, make CPU 0008C the sole `Ready` CPU row, and confirm no 0008C task file exists.
- Confirm package inventory and new/moved type placement match this task.
- Confirm the final implementation change stays within maximum scope and contains no shared,
  architecture, Gradle, conformance, or integration paths.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>
- [Run](../../../../glossary.md#term-run):

```text
git diff --check
git diff --cached --check
git status --short
```

This task is not the CPU capability [checkpoint](../../../../glossary.md#term-checkpoint). Repository-wide tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance),
integration tests, and final capability closure remain CPU 0009/[CI](../../../../glossary.md#term-continuous-integration-ci) unless implementation changes a
shared contract or [dependency](../../../../glossary.md#term-dependency), which is outside [scope](../../../../glossary.md#term-scope) and requires stopping rather than expanding.

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0005A: [atomic](../../../../glossary.md#resource-transaction-and-rollback) [partition](../../../../glossary.md#partition)/unit/[IR](../../../../glossary.md#term-intermediate-representation-ir)/[artifact](../../../../glossary.md#term-artifact)/executable architecture reset.
- Complete CPU 0005D: bounded one-external-read workspace [materialization](../../../../glossary.md#materialization) contract.
- Complete CPU 0006: current straight-line affine-[view](../../../../glossary.md#term-view-meanings) folding and materialized boundary rules.
- Complete CPU 0008: direct/[fused](../../../../glossary.md#specialization-and-fusion) [Conv2d](../../../../glossary.md#convolution--conv2d) and the temporary two-unit materialized suffix.
- Complete CPU 0008A: exact [Conv1d composition](../../../../glossary.md#conv1d-composition) and direct [Conv3d](../../../../glossary.md#conv3d--ncdhw) closure.
- Existing [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/[Planning](../../../../glossary.md#term-planning)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) contracts referenced by those completed tasks.

## Follow-up tasks

- CPU 0008C: closed typed specialized-subgraph and epilogue recognition; becomes the next Ready
  task only after 0008B completes.<span data-glossary-navigation="true"> ([epilogue glossary](../../../../glossary.md#term-epilogue); [Ready glossary](../../../../glossary.md#term-ready))</span>
- CPU 0008D: bounded profitability ranking and typed legal/accepted/rejected/selected decision
  facts.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span>
- [CPU](../../../../glossary.md#term-cpu) 0008E: bounded multi-input external read-boundary [materialization](../../../../glossary.md#materialization) and representation reuse.
- CPU 0008F–0008I: [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), [pooling](../../../../glossary.md#term-pooling), attention, and [loss](../../../../glossary.md#loss) family execution.
- CPU 0009: generated-coverage, capability, [conformance](../../../../glossary.md#term-conformance), and integration [checkpoint](../../../../glossary.md#term-checkpoint).

No detailed 0008C or later task specification is created by this [planning](../../../../glossary.md#term-planning) task.

## Architecture impact

Expected impact: None.

This task exercises existing concrete-[backend ownership](../../../../glossary.md#backend-ownership) of complete-[partition](../../../../glossary.md#partition) [lowering](../../../../glossary.md#lowering), [fusion](../../../../glossary.md#term-fusion),
exact resources, finalization, generated [artifacts](../../../../glossary.md#term-artifact), and one partition executable. It generalizes a
[CPU](../../../../glossary.md#term-cpu)-private closed [cardinality](../../../../glossary.md#term-cardinality) exception without changing a shared contract. If implementation
requires a new shared/public [topology](../../../../glossary.md#term-topology), resource, scheduling, route, or [lifecycle](../../../../glossary.md#lifecycle) contract, stop
and report the conflicting architecture rule and required decision.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the isolated implementation agent for Synaptik CPU task 0008B.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md.
Read the task's directly referenced completed CPU tasks and relevant final CPU source/tests.
Implement exactly the Ready specification. Do not implement CPU 0008C or later work. Stop and
report any architecture, shared-contract, or maximum-scope conflict.

After stable implementation and recorded Java/Class-File/performance validation, hand the same
diff and evidence to a separate clean documentation-focused agent. That pass must follow
docs/developer-guide/documentation-rules.md, independently finalize affected Javadocs, the CPU
guide, glossary impact, planning evidence/status, and documentation checks without repeating
successful Java/performance validation unless executable behavior changes or a concrete risk is
recorded.

Do not commit or push unless the coordinating user explicitly authorizes it. Update this task's
local decisions, known limitations, validation evidence, implementation notes, completion summary,
and final status only from actual results.
```

## Local decisions

- The maximally split topology is the semantic fallback. 0008B performs deterministic maximal
  legal pointwise fusion only; it does not call a legal fusion "unprofitable" because 0008D owns
  profitability and decision facts.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [pointwise glossary](../../../../glossary.md#term-pointwise); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Current affine chains and specialized families are indivisible seeds. This keeps CPU 0008B
  cohesive and preserves established numerical/resource contracts while still removing the
  general single-unit restriction.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Published intermediates and fan-out values materialize. Although ordinary pointwise IR can emit
  multiple stores, keeping a publication or fan-out producer virtual would broaden liveness,
  alias, and publication policy beyond this task's deterministic safety boundary.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [publication glossary](../../../../glossary.md#term-publication); [producer glossary](../../../../glossary.md#term-producer); [liveness glossary](../../../../glossary.md#term-liveness); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Multi-unit CPU 0005D external-read materialization is disabled, while intrinsic family
  workspaces remain supported per unit. CPU 0008E owns representation variants; forbidding all
  unit workspaces would make the decomposition falsely nongeneral for already supported families.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Strict sequential unit execution is selected. It is deterministic, preserves current child
  numerical/worker behavior, and needs no new scheduling contract; unit-level parallel DAG
  scheduling is neither required nor approved.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- Hard ceilings use the existing eight-node bound and natural eight-node maximum edge/fan-out
  counts. Pointwise indexing/liveness/code-size ceilings are exact structural gates, not measured
  cost heuristics, so they do not pre-empt CPU 0008D.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>
- Open questions: None. Architecture ownership, policy separation, topology order, hard ceilings,
  resource identity, workspace rebasing, and execution order are fixed by this specification; an
  implementation need outside those decisions is a stop-and-replan condition.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Known limitations

- New fusion is limited to ordinary pointwise IR. Mixed affine/pointwise, specialized epilogues,
  numerical multi-pass fusion, and state/random fusion remain deliberately split.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [pointwise glossary](../../../../glossary.md#term-pointwise); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Unit execution is topologically sequential; only existing child-local worker parallelism runs
  concurrently.
- Optional CPU 0005D external-read materialization is available only to one-unit plans until CPU
  0008E supplies a bounded multi-unit representation model.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [materialization glossary](../../../../glossary.md#materialization); [model glossary](../../../../glossary.md#term-model))</span>
- Hard fusion budgets may retain more materialized units than a later profitable plan. CPU 0008D
  owns ranking without weakening this task's legality/resource fallback.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span>
- This is not public Engine integration or the CPU capability/conformance checkpoint; CPU 0009
  retains that closure.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean implementation context: `01a03eaf-f654-7ef2-928f-592e051c467e`, based on
  `7223c95e520fc3f1e343d1633b1db891457439c2`; no staging, [commit](../../../../glossary.md#term-commit), or push was performed.
- Development compilation and focused lowering/preparation/finalization/composite/inventory tests
  passed. The focused generated evidence test passed direct semantics, Class-File member and
  allocation scans, and five accepted Java 26 fixed-heap forks.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [lowering glossary](../../../../glossary.md#lowering); [heap glossary](../../../../glossary.md#term-heap))</span>
- Follow-up clean implementation task `/root/cpu_0008b_followup` (no UUID exposed) corrected the
  completion blockers without staging, committing, or pushing. Its one authoritative final
  `./gradlew :backends:cpu:test --rerun-tasks` passed 99 suites and 512 tests with three existing
  expected skips, zero failures, and zero errors. Executable Java did not change afterward.
- Retained evidence directory:
  `/private/tmp/synaptik-cpu-0008b-evidence`. The authoritative-suite fresh accepted ratios are
  `1.058323828x`, `1.028316543x`, `1.058771135x`, `1.111571279x`, and `1.079831620x`; aggregate
  median is <span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>`1.058771135x`. Six earlier rejected attempts are retained accurately at `1.202949878x`,
  `1.213340020x`, `1.169101552x`, `1.199627968x`, `1.198502292x`, and `1.161363984x`; none is
  included in the accepted aggregate. The SHA-256 digest of <span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [digest glossary](../../../../glossary.md#term-digest))</span>`manifest.sha256` is
  `cbb1c7e06550a49303916dfe82be798f99e8f14e616420924e03710aa7d8b2c5` and every manifest entry
  verifies.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest))</span>
- The retained generated evidence test compares the two-store FLOAT64 NEG loop against optimal
  clean Java with the same one-pass primitive dataflow and ordered stores. The retained 457-byte
  Class-File has SHA-256<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>
  `9c39fe6d782c90e8df951bc0553127fd36c1c3e7e9b712dcf3469a721a9ad08e` and decompiles to one
  61-instruction direct primitive loop with two ordered `dastore` instructions. Complete semantic,
  Class-File, and `javap` inspection found one method and zero fields, member references, field or
  invocation instructions, allocations, boxing paths, reflection paths, and string constants;
  raw-bit checks include NaN and signed zero. Existing one-unit artifacts retain their unchanged
  IR structural keys and specializations, so this task makes no redundant performance claim for
  those identity-reused forms.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero); [artifacts glossary](../../../../glossary.md#term-artifact); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Clean documentation context: `01a03ece-baf7-7d00-8cc9-2a7928129279`. It independently reviewed
  the final production/test diff and retained evidence, finalized the affected Javadocs, CPU guide,
  glossary, task, master plan, and roadmap, and changed no executable Java behavior. The General,
  API/Javadoc, Planning, and Backend-guide profiles were applied. The glossary required updates
  because the existing CPU portable-route, preparation-plan, prepared-executable, execution-unit,
  and kernel-IR entries still described 0008B as Draft and the former straight-line/single-output
  boundary.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [Backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel))</span>
- The documentation review's liveness blocker is resolved. External inputs now become live at
  first actual use, instruction results become live at definition, duplicate same-instruction
  operands count once, and values die after their final instruction/store event. Indexing
  complexity now also follows the specified materialized-binding-only rule with checked
  arithmetic. Exact 16/17 live-value and at/next boundary, indexing, and code-size decisions are
  pinned by focused tests.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [binding glossary](../../../../glossary.md#term-binding))</span>
- Fifteen new focused acceptance tests cover diamond and seven-way fan-out, publication, affine,
  numerical reduction, random/state barriers, deterministic repeatability, one/eight node and unit
  edges, structural hard-ceiling edges, malformed later topology, exact first-use buffers and
  carriers, two workspace-bearing units with final IDs zero and one, missing/extra/mismatched
  complete resource sets, missing/undersized/closed later workers, invalid later carrier and
  workspace, read-only output, cold failure-before-write, no later execution after child failure,
  and successful one-range then two-range completion. Existing composite coverage retains
  cross-unit alias rejection before the Conv2d write.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [publication glossary](../../../../glossary.md#term-publication); [node glossary](../../../../glossary.md#node); [topology glossary](../../../../glossary.md#term-topology); [buffers glossary](../../../../glossary.md#term-buffer); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alias glossary](../../../../glossary.md#aliasing-and-arena); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- A focused documentation-context regeneration of the exact retained two-store NEG IR produced a
  457-byte Java 26 Class-File with SHA-256
  `9c39fe6d782c90e8df951bc0553127fd36c1c3e7e9b712dcf3469a721a9ad08e`.
  `javap -v -c -p /private/tmp/synaptik-cpu-0008b-multistore.class` showed one static primitive
  method, one direct integer-indexed loop, two `daload`/`dneg` computations, two ordered
  `dastore` instructions, no method invocation, no field access, no allocation opcode, and no
  constant-pool member reference. This closes the decompilation/hidden-helper inspection for the
  genuinely new emitted form without widening the performance claim.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [decompilation glossary](../../../../glossary.md#term-decompilation); [widening glossary](../../../../glossary.md#term-widening))</span>
- Final clean documentation context `/root/cpu_0008b_doc_review` independently inspected the
  final production/test diff, first-use/definition/final-use liveness, materialized-only indexing,
  final-index workspace rebasing, dead special-case removal, the focused acceptance matrix, and
  every retained accepted and rejected performance sample. It changed no executable Java behavior,
  applied the General, API/Javadoc, Planning, and Backend-guide profiles, corrected the remaining
  stale Conv2d-sequence guide text, finalized the glossary and synchronized planning statuses.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- `./gradlew :backends:cpu:javadoc` passed after final documentation edits with only the two
  expected incubating-Vector-module warnings. Focused Markdown, status, exact-scope, manifest,
  staged-index, and whitespace validation also passed in the final documentation context.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [module glossary](../../../../glossary.md#term-module); [scope glossary](../../../../glossary.md#term-scope))</span>
- From `/private/tmp/synaptik-cpu-0008b-evidence`, `shasum -a 256 -c manifest.sha256` passed all
  16 retained entries and `shasum -a 256 manifest.sha256 generated/multi-store-neg.class`
  reproduced both recorded digests. Independent `javap -v -c -p` inspection reproduced the exact
  one-method/zero-field direct loop and absence of member references. Five-file Markdown
  link/anchor and fence/final-newline checks passed; all canonical task headings were present.
  Planning-status checks found exactly one Ready CPU row, 0008C, and no 0008C detail file.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Ready glossary](../../../../glossary.md#term-ready))</span>
- `git diff --check` and `git diff --cached --check` passed. Exact combined scope is seven CPU
  production/Javadoc paths, eight CPU test/evidence paths, and five documentation/planning paths,
  with no shared Java, architecture, Gradle, conformance, or integration path. The exact package
  inventory passed with 114 Java files across 11 declared packages; <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>`git diff --cached --name-only`
  was empty and final status retained only intended unstaged paths.

## Implementation notes

- Added `CpuPartitionDagDecomposer` as the sole CPU-private owner of complete-partition validation,
  longest established seed selection, stable topological ordering, dependency indices, vertical-
  first/horizontal-second contraction, and the task's hard pointwise fusion budgets.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner); [partition glossary](../../../../glossary.md#partition); [validation glossary](../../../../glossary.md#term-validation); [seed glossary](../../../../glossary.md#term-seed); [topological glossary](../../../../glossary.md#term-topological-order); [dependency glossary](../../../../glossary.md#term-dependency); [pointwise glossary](../../../../glossary.md#term-pointwise); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Generalized ordinary pointwise lowering to deterministic DAG instruction order and multiple
  final stores while preserving existing instruction semantics and one-unit structural identity.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Generalized preparation to one through eight per-unit plans. Every unit retains its route,
  strategy, boundary/carrier facts, optional exact workspace with requirement ID equal to its final
  unit index, family geometry, and dependencies. Multi-unit plans disable CPU 0005D external-read
  materialization and deduplicate ordinary logical-value buffers in first-use unit order.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [materialization glossary](../../../../glossary.md#materialization); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- Corrected workspace rebasing so each selected family unit first passes its established one-unit
  validation with local requirement ID zero, then the general partition assembly creates the exact
  final-index declaration and carries that identity into unit runtime facts and requirements.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- Replaced the [Conv2d](../../../../glossary.md#convolution--conv2d)-only sequence with `CpuPreparedPartitionExecutable`. Finalization resolves
  all exact buffer/workspace assignments and validates worker requirements before artifact lookup;
  the composite validates every selected representation and cross-buffer overlap before binding
  children, then executes already-bound children strictly in stable unit order.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup); [binding glossary](../../../../glossary.md#term-binding))</span>
- The published/illegal Conv2d suffix now reaches the same decomposer and general composite as all
  other split plans. Publication, fan-out, affine/specialized, numerical-order, and state/random
  edges retain materialized ordinary-buffer barriers.<span data-glossary-navigation="true"> ([Publication glossary](../../../../glossary.md#term-publication); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- The clean documentation pass finalized Javadocs for topology ordering, defensive snapshots,
  ownership, resource lifecycle, parameters, results, failures, and the cold/hot-path boundary. It
  added the implemented DAG/fusion/resource mental model to the CPU guide and reconciled the
  directly affected glossary entries without changing architecture authority.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span><span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [parameters glossary](../../../../glossary.md#term-parameter))</span>

## Completion summary

- Completed changes: deterministic bounded partition-DAG decomposition; vertical and horizontal
  ordinary-pointwise fusion; stable per-unit dependencies; exact split buffers and per-unit
  workspaces; general finalization; atomic sequential composite; multi-store generated evidence.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [pointwise glossary](../../../../glossary.md#term-pointwise); [fusion glossary](../../../../glossary.md#term-fusion); [buffers glossary](../../../../glossary.md#term-buffer); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- [CPU](../../../../glossary.md#term-cpu) production files changed/created/removed: `CpuPartitionDagDecomposer`,
  `CpuPartitionLowering`, `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`,
  `CpuPartitionFinalizer`, `CpuPreparedPartitionExecutable`, and removal of
  `CpuPreparedExecutableSequence` (seven paths, inside the fourteen-path ceiling).
- CPU test/evidence files changed/created/removed: package inventory, pointwise lowering,
  preparation, removal and replacement of the sequence/composite test path, decomposition and
  resource acceptance matrices, and generated multi-store evidence (eight paths, inside the
  fourteen-path ceiling).<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [matrices glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Tests and validation: authoritative CPU suite passed 512 tests across 99 suites with three
  expected skips; five accepted fixed-heap generated/direct forks and aggregate passed <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [heap glossary](../../../../glossary.md#term-heap))</span>`<= 1.15x`;
  rejected measurements were retained.
- Documentation-agent review: final clean context `/root/cpu_0008b_doc_review` independently
  finalized all affected Javadocs and the CPU guide, glossary, task evidence/status, CPU master
  plan, and roadmap after reviewing the follow-up implementation and retained evidence.
- Documentation impact: the CPU guide now explains general decomposition, legal pointwise fusion,
  hard budgets, split buffers versus external materialization, unit workspaces, atomic binding and
  sequential execution, safe split fallback, and a fork/join example.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [binding glossary](../../../../glossary.md#term-binding); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- Javadoc review: affected types now document topology, immutability, defensive copying, plan and
  resource ownership, nullability, failures, cold validation, and hot-path behavior; CPU Javadoc
  generation passed.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [topology glossary](../../../../glossary.md#term-topology); [defensive copying glossary](../../../../glossary.md#memory-management-atomic-counters-and-complexity); [ownership glossary](../../../../glossary.md#term-ownership); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Glossary impact: updated the existing CPU portable-route, preparation-plan, prepared-executable,
  execution-unit, and kernel-IR definitions; no new glossary heading was needed because 0008B
  changes the implemented boundary of existing reusable terms rather than introducing a new term.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Unresolved issues: None.
- Follow-up required: None. CPU 0008C is the sole Ready CPU task; no detailed 0008C task
  specification was created.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

Status: Complete
