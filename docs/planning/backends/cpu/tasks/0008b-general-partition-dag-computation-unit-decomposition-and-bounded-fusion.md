# Task 0008B: General Partition-DAG Computation-Unit Decomposition and Bounded Fusion

## Status

Complete

## Goal

Replace the CPU backend's straight-line/single-unit whole-partition assumption and the closed
Conv2d two-unit exception with one bounded CPU-private model for decomposing a complete CPU-owned
partition directed acyclic graph (DAG) into ordered computation units. Fuse only the currently
proved pointwise forms across legal vertical and horizontal edges, retain established affine-view
and specialized-family units as indivisible seeds, and declare every remaining inter-unit edge as
one ordinary logical-value buffer before shared assignment.

Return one atomic partition executable. Cold binding must validate every child, resource, carrier,
span, worker requirement, and cross-unit alias constraint before any write. Hot execution must run
already-bound units in deterministic topological order, completing each unit and joining its
workers before the next unit starts. Illegal, unsupported, or hard-budget-exceeding fusion must
retain the complete deterministic materialized-split topology; it must not make an otherwise
individually supported partition fail.

## Motivation and mental model

The current CPU implementation has two incompatible closed forms:

```text
ordinary partition                     CPU 0008 exception

partition -> one unit                  Conv2d unit -> buffer -> pointwise unit
          -> one artifact                         -> two artifacts
          -> one executable                       -> one atomic two-child executable
```

CPU 0008B replaces the cardinality exception with a single model:

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
an existing graph `ValueId` crossing computation units. It is not CPU 0005D's optional contiguous
copy of one external read boundary into an anonymous workspace, and it is not CPU 0008E's later
bounded choice among external read-boundary representation variants.

## Scope

### Complete partition-DAG validation and deterministic topology

- Analyze exactly one complete partition whose owner is `CpuCapabilityProvider.CPU_BACKEND_ID`.
  Retain the existing hard maximum of eight compiled nodes. Empty partitions, duplicate node or
  output identities, missing projected values or memory facts, cycles, consumers preceding their
  producers in the supplied complete context, and a node not independently supported by a current
  CPU lowering family fail before declarations or artifact access.
- Build CPU-private producer, consumer, graph-publication, and constant facts from the complete
  `PrepareContext`. Do not query Runtime state, perform a hidden graph rewrite, or mutate the
  compiled nodes.
- Treat the supplied node order as the validated stable topological order. Stable node ordinal,
  output ordinal, input ordinal, and then `ValueId` only as a final identity check determine all
  lists; do not depend on hash iteration or object identity.
- Create the maximally split baseline first. Each seed is the smallest complete occurrence already
  accepted by current CPU lowering: one ordinary pointwise node; one established affine-view
  chain when its current all-affine straight-line rules admit the chain; one current one-node
  movement, indexing, scatter, fold, ordering, random, scan, aggregate, normalization, batch-
  normalization, Conv2d, or Conv3d family; the exact CPU 0008A four-node Conv1d composition; or an
  already proved closed current Conv2d fused occurrence. Do not split the interior of an
  established numerical or multi-output semantic unit.
- If more than one seed could claim a node, select the longest already-established closed
  occurrence; ties use the smallest first node ordinal and then the existing family dispatch
  order in `CpuPartitionLowering`. This rule preserves the current Conv1d and direct/fused Conv2d
  routes without introducing CPU 0008C recognition.
- Assign final unit indices with stable Kahn ordering: repeatedly select the ready unit whose
  smallest member-node ordinal is lowest. Each unit records the strictly increasing indices of
  its direct producer units. The stored order must already satisfy every dependency; execution is
  not permitted to discover or schedule dependencies dynamically.

### Legal vertical and horizontal fusion

- CPU 0008B admits new fusion only when every fused operation lowers to the existing ordinary
  `CpuKernelIr` pointwise instruction/store vocabulary. It may generalize the current pointwise
  straight line to a pointwise DAG with multiple stores. It does not fuse through affine-copy,
  movement, indexing, scatter, fold, ordering, random, scan, reduction, softmax, normalization,
  batch-normalization, Conv2d, Conv3d, or exact Conv1d-composition seeds.
- A vertical edge is eligible only when one pointwise producer value has exactly one in-partition
  consumer, is not a graph output, and the producer and consumer units have one identical checked
  iteration domain, compatible result/input data type, compatible normalized access mapping, and
  no other dependency cycle after contraction. The fused IR must preserve the current operation
  order, typed intermediate semantics, scalar-power realization, canonical BOOL behavior, and
  final stores.
- A horizontal edge is a CPU-private fusion relation between two dependency-independent
  pointwise units. It is eligible only when both are ready from the same predecessor set, have the
  same checked iteration domain and execution strategy, have disjoint outputs, neither consumes
  the other's output, and one combined pointwise IR can preserve both branches in stable member-
  node order. Shared external inputs are deduplicated by `ValueId`; final stores remain ordered by
  producer node and output ordinal.
- Consider vertical pairs first in producer-unit/member-node order and then horizontal pairs in
  left-unit/right-unit order. After a successful contraction, restart vertical consideration from
  the first unit. After vertical convergence, process horizontal pairs and restart at the first
  pair after each contraction. Test one pair at most once for one unchanged topology identity;
  contraction creates a new topology identity. Stop when no legal within-budget contraction
  remains or the candidate-attempt ceiling is reached.
- Selection in this task is deterministic maximal legal fusion: accept the first legal pair that
  satisfies every hard budget. There is no cost comparison, profitability threshold, candidate
  ranking, typed accepted/rejected fact model, or tuning input. CPU 0008D owns those concerns.

### Fusion barriers and fail-closed semantics

- A value with more than one in-partition consumer is a vertical-fusion barrier. Independent
  consumers may still be horizontally fused if their own horizontal rules hold, but the producer
  value remains a materialized split boundary.
- A graph-output/publication obligation is a fusion barrier across that value. A fused unit may
  have multiple final graph-output stores, but CPU 0008B does not keep a published intermediate
  virtual and does not change publication identity or order.
- Any unresolved or potentially overlapping write/read alias relation, non-injective output,
  negative or out-of-span address, carrier incompatibility, or mismatch in normalized access
  geometry is a fusion barrier. Existing one-unit family-specific alias rules remain authoritative.
- Explicit state transition, random/counter advancement, dropout, or any other stateful occurrence
  is an indivisible seed and a barrier on every incident edge. It executes exactly once in original
  topological order.
- Reduction, scan, ordering, softmax, normalization, batch-normalization, convolution, fold, and
  scatter units are numerical-order or family-algorithm barriers. Their current loop order,
  accumulation, rounding, tie, state, and multi-output contracts must remain unchanged. No
  pointwise epilogue recognition is inferred around them in this task.
- Affine-view chains remain governed by CPU 0006. A current legal unpublished straight-line affine
  chain may remain one seed; branching, publication, and mixed affine/pointwise edges materialize.
  CPU 0008B does not invent a mixed view/pointwise generated form.
- A fusion-lowering exception, unsupported carrier/access combination, failed hard-budget proof,
  or exhausted fusion-attempt budget rejects only that contraction and retains the already valid
  split units. Failure of the maximally split baseline itself fails the partition before resource
  declaration; there is no Runtime fallback and no partial plan.

### Exact hard budgets

The following are semantic/resource safety ceilings, not profitability scores:

| Budget | Ceiling | Counting rule | On exceedance |
| --- | ---: | --- | --- |
| Partition nodes | 8 | Complete compiled nodes, preserving the existing cap | Fail before declarations |
| Final units | 8 | Stable computation units after contraction | Retain split; fail only if the baseline exceeds the node-derived cap |
| Fan-out considered for fusion | 7 | Distinct in-partition consumers of one value | Materialize; vertical fusion requires exactly one |
| Fusion attempts | 28 | Every tested vertical or horizontal unit pair, successful or rejected | Stop contracting and retain remaining split units |
| Nodes per newly fused pointwise unit | 8 | Member compiled nodes | Retain split |
| Materialized boundaries per unit | 16 | Distinct non-virtual input and output `ValueId`s | Retain split |
| Simultaneously live IR values | 16 | Maximum live set from stable instruction/store liveness, including materialized inputs and pending stores | Retain split |
| Indexing-complexity units | 32 | Sum per materialized binding: `DENSE_LINEAR`/`SCALAR_ALL_ZERO` = 1, `LAST_AXIS_BIAS` = 2, `BLOCK_OUTER` = 3, `GENERAL_ODOMETER` = 4 | Retain split |
| Generated-code-size units | 64 | 8 base + 4 per pointwise instruction + 3 per store + indexing-complexity units + 1 per virtual value | Retain split |

- Apply the last five ceilings only to a newly proposed pointwise contraction. Established atomic
  family forms retain their own completed bounds and validation; CPU 0008B must not retrospectively
  reject them using a pointwise estimator.
- Compute liveness after stable IR construction: an input becomes live at first use, a virtual
  value at definition, and a value dies after its last instruction or store use. Multiple uses by
  one instruction count once. Checked arithmetic overflow rejects the contraction and retains the
  split form.
- `generated-code-size units` is an analysis-time structural estimator, not a claim about exact
  Class-File byte length. The existing artifact-store byte limit remains independently enforced.
  Analysis must not generate a class to decide fusion.
- Keep the existing per-unit `CpuSpecializationBudget(4, 1, 0, 0)`: at most four representation
  candidates, one realized artifact, and no fixed-shape or unroll variants for each selected unit.
  A partition may therefore realize at most eight artifacts and never more than one artifact per
  final unit.

### Materialized split resources and workspaces

- After final topology selection, derive resources once. Declare each distinct materialized graph
  `ValueId` exactly once as `PreparationResourceRequirement.Buffer`, with checked referenced span
  times data-type width and alignment equal to the data-type width. Order declarations by first
  unit use in stable unit order, inputs before outputs within a unit; deduplicate by `ValueId` and
  reject conflicting geometry.
- Every value crossing two final units is materialized even if it is not a graph output. It is an
  ordinary logical-value buffer, visible in both units' boundary lists, written by exactly one
  producer unit, and read by every dependent unit. It is never a workspace, anonymous temporary,
  hidden generated allocation, or Runtime-created value.
- Preserve exact graph publications and external reads. Partition-level access is `READ_ONLY` only
  for values no unit writes and `WRITE_ONLY` for every value written by any unit, including
  internal split buffers and graph outputs. `READ_WRITE` remains forbidden.
- Move current execution/resource geometry into complete per-unit facts. Each unit owns its route,
  IR, specialization, boundary bindings/carriers, strategy/ranges, optional CPU 0005D
  materialization, optional exact workspace and purpose, affine pairs, and applicable family
  geometry. The partition plan owns only the ordered unit/dependency topology and deduplicated
  partition resource/access view; it must not mirror first-unit geometry as partition truth.
- Preserve CPU 0005D exactly for a selected one-unit plan. Disable its optional external-read
  contiguous-copy selection when the final topology has more than one unit; multi-unit external
  read representation choices belong to CPU 0008E. This does not disable workspaces intrinsically
  required by scatter, ordering, reductions, Layer normalization, or batch-normalization units.
- Rebase every unit-local workspace requirement to partition-unique ID equal to its final unit
  index. Units without a workspace leave an unused ID. Declare workspaces in unit order with the
  exact existing byte size/alignment and carry the rebased ID through unit-local materialization
  or family geometry. Workspace bytes remain run-owned; no generated code allocates them.
- Each child receives only its exact assigned workspace selection and validates it cold. The
  composite makes no workspace-alias assumption; if the shared plan legally reuses storage, strict
  sequential execution ensures two unit workspaces are never accessed concurrently.

### Preparation, finalization, and one atomic executable

- Replace `PlanForm.ONE_UNIT`/`CONV2D_MATERIALIZED_SUFFIX` cardinality inference with a general
  validated partition form covering one through eight units. The current Conv2d materialized
  suffix must be expressed through the same unit/dependency/resource model, with no special
  preparer or finalizer branch.
- Finalization resolves the exact complete buffer and workspace assignment sets before the first
  artifact-store lookup. It then realizes exactly one already-selected artifact per unit in unit
  order. It cannot add a buffer/workspace, change a unit boundary, fuse/split, change strategy, or
  select another route after assignment.
- Replace the exact two-child `CpuPreparedExecutableSequence` with a narrowly CPU-private general
  partition composite. It retains an immutable ordered child list and immutable dependency facts;
  it is not a Runtime graph interpreter, scheduler, public executable DAG, or service registry.
  One-unit plans may continue to return the direct `CpuPreparedExecutable` when that preserves the
  same externally observable contract.
- Composite cold binding first validates every outer CPU buffer/workspace representation and all
  write/read and write/write overlaps among distinct selected buffer spans. It then binds every
  child into a temporary immutable list in unit order. Any child carrier, span, workspace, worker,
  geometry, or alias failure aborts binding before execution and before any write.
- Hot execution invokes the already-bound children in stable unit order. Each child finishes its
  complete selected range set and joins its borrowed worker group before the next begins. On the
  first thrown failure, no later child runs. The composite performs no validity mutation: unchanged
  Runtime surrounds it with one atomic invalid-before/valid-after transition for all partition
  writes, so a failed partition leaves every declared output/split write invalid.
- Do not add unit-level parallel scheduling. Existing child-local bounded parallelism remains the
  only concurrency. Dependency indices are validated cold and retained for deterministic topology
  identity, diagnostics, and later planning, not interpreted dynamically at run time.

### Generated-code and performance evidence

- Preserve byte-identical generated artifacts for unchanged one-unit pointwise, affine, Conv1d,
  Conv2d, Conv3d, and other specialized seeds whenever their existing specialization and IR are
  identical. Orchestration-only reuse needs structural/cache-identity regression evidence, not a
  duplicate performance benchmark.
- For the genuinely new multi-store horizontal pointwise IR, implement and retain an optimal clean
  Java oracle with the same typed operations, evaluation order, hot-loop/dataflow shape, stores,
  carrier forms, and avoidable-overhead profile. Inspect generated Class-File/decompilation for
  direct primitive loops and absence of hidden helpers, allocation, boxing, reflection, string
  dispatch, or operation interpretation.
- Run fixed-shape generated-versus-direct evidence for each genuinely new emitted form. Every fork
  and aggregate median must satisfy generated/direct `<= 1.15x`; never weaken, average away, or
  relabel a failed gate. Use the repository's accepted fork/rejection and evidence-manifest rules.
- A new decomposition that emits only byte-identical existing unit artifacts needs an explicit
  optimal direct-Java whole-partition baseline only when the task makes a performance claim about
  the composite orchestration itself. Otherwise record exact reuse and semantic/resource evidence.

### Documentation and Javadoc

- After stable implementation and Java/generated-code evidence, use a separate clean
  documentation-focused agent/thread as required by `AGENTS.md` and
  `docs/developer-guide/documentation-rules.md`.
- Finalize meaningful Javadoc for every changed CPU type, constructor, record component, method,
  exception, ownership/lifecycle rule, topology order, budget, resource, and failure boundary.
- Update `docs/backend-guide/cpu-backend.md` with the general unit-DAG mental model, deterministic
  pointwise fusion subset, hard barriers/budgets, split-buffer versus external-materialization
  distinction, exact workspace/resource lifecycle, atomic binding/execution behavior, safe
  fallback, and one compact fork/join example.
- Review the glossary. Add or change an entry only if implementation introduces a reusable term;
  otherwise record a reasoned no-change conclusion. Update this task, the CPU master plan, and the
  roadmap with actual evidence and status in the same overall change.

## Out of scope

- CPU 0008C typed specialized-subgraph or epilogue recognition, including new MATMUL,
  convolution, reduction, softmax, or normalization recognition.
- CPU 0008D profitability ranking, estimated complete-plan cost, legal-candidate comparison,
  typed accepted/rejected/selected facts, Trace payloads, or tuning inspection.
- CPU 0008E multi-input external read-boundary materialization, representation variants, reuse
  identities, or re-ranking.
- A public pattern registry, pattern DSL, public fusion API, generic Runtime graph interpreter,
  dynamic unit scheduler, Runtime route selection, or Runtime graph inspection.
- New Model operation kinds, Compiler rewrites, Planning abstractions, shared Prepare APIs,
  architecture rules, module dependencies, or service/provider boundaries.
- Mixed affine-view/pointwise fusion, fusion through state/random/numerical-order barriers,
  universal `ConvNd`, new heavy-family kernels, native/vendor routes, measurement, tuning caches,
  packing, reorder, or relaxed numerical modes.
- Hidden graph values, generated heap/native allocation, post-assignment resources, silent
  fallback to interpreted operations, or partial execution after cold validation failure.

## Architecture references

- `ARCHITECTURE.md`: complete-partition backend analysis, computation-unit and fusion ownership,
  exact pre-assignment resource declaration, backend finalization, generated-code discipline, one
  partition executable, cold binding, Runtime atomic validity, and hot-path prohibitions.
- `docs/architecture/current-architecture-plan.md`: current module ownership and absence of an
  approved shared-contract change.
- `docs/planning/planning-guide.md`: detailed-task frontier, validation tiers, completion evidence,
  and separate documentation pass.
- `docs/planning/backends/cpu/master-plan.md`: ordered CPU 0008B–0008E decomposition,
  recognition, profitability, and representation sequence.
- Completed CPU tasks 0005A, 0005D, 0006, 0008, and 0008A: atomic unit reset, external-read
  materialization, affine views, Conv2d two-unit exception, and dimensional convolution closure.

## Architecture constraints

- Planning selects the CPU owner; only CPU analysis lowers, decomposes, fuses, selects, and
  declares exact resources. Shared Planning, Prepare, Runtime, Model, and Compiler remain unaware
  of CPU unit topology.
- Analysis completes before shared slot assignment. Finalization validates assignments and
  realizes selected artifacts but cannot change topology, route, resources, or specialization.
- Runtime receives one `PreparedExecutable` for the partition and owns the atomic access-validity
  transition. The CPU composite owns only cold validation and deterministic invocation of fixed
  already-bound children.
- Generated entry points contain direct specialized computation. Neither generated code nor the
  composite may inspect `Operation`, `CompiledNode`, graph topology, route policy, or candidate
  policy on the hot path.
- Buffer resources use existing logical `ValueId`s. Workspace resources are explicit run-owned
  requirements. No public/shared abstraction or architecture-document change is pre-approved.

## Package impact

All production ownership remains inside `backends/cpu`:

- `io.github.pho001.synaptik.backend.cpu.internal.lowering`: add the private deterministic
  partition-DAG decomposition/fusion owner and its hard-budget/topology value types; reuse current
  family lowerers and ordinary pointwise IR construction.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`: generalize the partition plan,
  per-unit facts, analysis/resource assembly, and finalization. Do not add a shared Prepare type.
- `io.github.pho001.synaptik.backend.cpu.internal.executable`: replace the closed Conv2d sequence
  with one CPU-private fixed-unit composite.
- `io.github.pho001.synaptik.backend.cpu.internal.ir`, `.codegen`, `.cache`, and `.memory`: change
  only if the existing multi-store pointwise representation/emitter, structural identity, or cold
  binding requires a narrowly owned correction. Do not introduce another IR hierarchy or cache.

No package move, exported package, new module dependency, or shared source change is allowed.

## Affected files

Expected implementation paths, refined only within maximum scope:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionDagDecomposer.java`
  (new sole topology/fusion owner, including private/nested hard-budget facts)
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
  affine/specialized barriers, and inventory
- focused generated-Class-File/direct-Java evidence and manifest resources for genuinely new
  emitted forms
- `docs/backend-guide/cpu-backend.md`
- this task, `docs/planning/backends/cpu/master-plan.md`, and `docs/planning/roadmap.md`
- glossary only if the documentation review proves a reusable terminology change

Implementation may rename the two proposed new private types only if it records why the final
names better match these exact owners. It may not add a public `Graph`, `FusionManager`, registry,
DSL, generic scheduler, or catch-all utility.

## Maximum scope

- At most 14 CPU production/Javadoc paths, including the two proposed new types and removal of the
  old sequence type.
- At most 14 CPU test/evidence paths.
- At most four explanatory/planning documentation paths beyond this already-created task: CPU
  guide, glossary if needed, CPU master plan, and roadmap.
- No shared Java, Model, Compiler, Planning, Prepare, Runtime, Engine, Gradle, architecture,
  architecture-test, backend-conformance, or integration-test change.
- One new decomposition owner and one new composite executable are the intended type budget.
  Additional top-level production types, a second IR, or a second planning pass require recorded
  necessity and must remain inside the 14-path production cap; otherwise stop and replan.
- If the capability cannot be completed, documented, and validated within these limits, stop with
  `Status: Incomplete`; do not omit a required family seed, resource, failure path, or evidence gate.

## Acceptance criteria

- One complete CPU-owned DAG of one through eight current supported nodes deterministically
  becomes one through eight validated computation units with stable dependencies and exact
  resource declarations; unsupported or malformed baseline work fails before declarations.
- Existing one-unit pointwise/affine/specialized routes, exact Conv1d, direct/fused Conv2d, direct
  Conv3d, carrier patterns, numerical results, cache identities, and family resources remain
  unchanged when their selected unit IR/facts are unchanged.
- The CPU 0008 published/illegal Conv2d suffix uses the general decomposition model, declares its
  intermediate once, realizes two artifacts, and executes through one atomic composite without a
  Conv2d-specific analysis/finalization branch.
- A legal `ADD -> GELU -> MUL` chain remains one vertical pointwise unit. A legal pair of
  independent same-domain pointwise branches becomes one multi-store horizontal pointwise unit
  and exactly matches both publications.
- A diamond/fan-out fixture materializes the producer value, may fuse only independently legal
  consumer siblings, and records stable split buffers/unit dependencies. Publication, affine/
  pointwise, Conv3d/pointwise, reduction/pointwise, and dropout/state edges remain deterministic
  materialized barriers.
- Hard indexing, live-value, generated-size, boundary, node, unit, and attempt budgets are tested
  at and immediately beyond their ceilings. A rejected contraction leaves the complete split
  plan; it never loses a node, edge, output, workspace, or current supported route.
- Every cross-unit `ValueId` is one ordinary buffer and never a workspace. CPU 0005D external
  materialization remains distinguishable and is disabled for multi-unit plans. Two workspace-
  requiring seed units receive exact unique requirement IDs and child-local selections.
- Finalization rejects missing, extra, mismatched, or post-analysis buffer/workspace assignments
  before artifact lookup and realizes exactly one artifact per selected unit in stable order.
- Composite binding validates all children and cross-unit aliases before writes. Tests prove an
  invalid later child, undersized later workspace/worker group, read-only output, or overlapping
  distinct output/read span leaves carriers unchanged. Execution proves strict unit completion/
  join order, no later unit after failure, and one outer atomic validity result.
- Generated multi-store horizontal pointwise code matches an optimal clean Java oracle, contains
  no hidden operation interpretation/allocation/boxing/reflection/string dispatch, and passes all
  required `<= 1.15x` fork and aggregate gates. Byte-identical reused forms have explicit reuse
  evidence and no redundant benchmark claim.
- Javadocs, CPU guide, glossary conclusion, task evidence/status, master plan, and roadmap are
  independently finalized by the documentation pass. CPU 0008B becomes `Complete` only after all
  evidence passes; CPU 0008C is then the sole `Ready` CPU task and no 0008C detail file is created.

## Tests / validation

### Tier 1: focused CPU semantics and contracts

- Add focused decomposition tests for stable topology, disconnected ready seeds, vertical and
  horizontal fusion, diamond fan-out, publications, state/random and numerical-order barriers,
  affine/specialized barriers, unsupported baseline failure, deterministic repeatability, and
  every hard-budget edge.
- Extend preparation-plan/preparer/finalizer tests for per-unit facts, dependency validation,
  exact deduplicated buffers, unique workspace IDs, explicit carrier mapping, malformed resource
  sets, artifact order/count, and migrated Conv2d split behavior.
- Replace `CpuPreparedExecutableSequenceTest` with general composite tests covering all-units cold
  validation, alias rejection before writes, strict execution/join/failure order, multiple outputs,
  multiple workspaces, and one atomic Runtime validity transition.
- Retain focused regression coverage for pointwise, affine views, Conv1d composition, Conv2d,
  Conv3d, reductions, random/dropout, multi-output batch-normalization, and current materialization.

Run once after implementation stabilizes:

```text
./gradlew :backends:cpu:test
```

### Tier 2: generated-code and performance evidence

- Run the CPU generated Class-File/decompilation/import/member-reference checks for the new
  multi-store pointwise form and current reused controls.
- Run repository-standard fixed-shape performance forks and manifest verification for each new
  emitted form, recording every fork and aggregate `<= 1.15x` result and any accepted sample
  rejection. Do not substitute an orchestration benchmark for generated/direct evidence.
- Run CPU Javadoc after the documentation pass finalizes changed contracts:

```text
./gradlew :backends:cpu:javadoc
```

### Tier 3: documentation and repository hygiene

- Validate Markdown links, anchors, fences, required headings, task statuses, trailing whitespace,
  final newlines, and glossary decision according to the documentation rules.
- Confirm CPU 0008B is the sole detailed `Ready` task during implementation; on completion update
  it to `Complete`, make CPU 0008C the sole `Ready` CPU row, and confirm no 0008C task file exists.
- Confirm package inventory and new/moved type placement match this task.
- Confirm the final implementation change stays within maximum scope and contains no shared,
  architecture, Gradle, conformance, or integration paths.
- Run:

```text
git diff --check
git diff --cached --check
git status --short
```

This task is not the CPU capability checkpoint. Repository-wide tests, backend conformance,
integration tests, and final capability closure remain CPU 0009/CI unless implementation changes a
shared contract or dependency, which is outside scope and requires stopping rather than expanding.

## Dependencies

- Complete CPU 0005A: atomic partition/unit/IR/artifact/executable architecture reset.
- Complete CPU 0005D: bounded one-external-read workspace materialization contract.
- Complete CPU 0006: current straight-line affine-view folding and materialized boundary rules.
- Complete CPU 0008: direct/fused Conv2d and the temporary two-unit materialized suffix.
- Complete CPU 0008A: exact Conv1d composition and direct Conv3d closure.
- Existing Model/Compiler/Planning/Prepare/Runtime contracts referenced by those completed tasks.

## Follow-up tasks

- CPU 0008C: closed typed specialized-subgraph and epilogue recognition; becomes the next Ready
  task only after 0008B completes.
- CPU 0008D: bounded profitability ranking and typed legal/accepted/rejected/selected decision
  facts.
- CPU 0008E: bounded multi-input external read-boundary materialization and representation reuse.
- CPU 0008F–0008I: MATMUL, pooling, attention, and loss family execution.
- CPU 0009: generated-coverage, capability, conformance, and integration checkpoint.

No detailed 0008C or later task specification is created by this planning task.

## Architecture impact

Expected impact: None.

This task exercises existing concrete-backend ownership of complete-partition lowering, fusion,
exact resources, finalization, generated artifacts, and one partition executable. It generalizes a
CPU-private closed cardinality exception without changing a shared contract. If implementation
requires a new shared/public topology, resource, scheduling, route, or lifecycle contract, stop
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
  profitability and decision facts.
- Current affine chains and specialized families are indivisible seeds. This keeps CPU 0008B
  cohesive and preserves established numerical/resource contracts while still removing the
  general single-unit restriction.
- Published intermediates and fan-out values materialize. Although ordinary pointwise IR can emit
  multiple stores, keeping a publication or fan-out producer virtual would broaden liveness,
  alias, and publication policy beyond this task's deterministic safety boundary.
- Multi-unit CPU 0005D external-read materialization is disabled, while intrinsic family
  workspaces remain supported per unit. CPU 0008E owns representation variants; forbidding all
  unit workspaces would make the decomposition falsely nongeneral for already supported families.
- Strict sequential unit execution is selected. It is deterministic, preserves current child
  numerical/worker behavior, and needs no new scheduling contract; unit-level parallel DAG
  scheduling is neither required nor approved.
- Hard ceilings use the existing eight-node bound and natural eight-node maximum edge/fan-out
  counts. Pointwise indexing/liveness/code-size ceilings are exact structural gates, not measured
  cost heuristics, so they do not pre-empt CPU 0008D.
- Open questions: None. Architecture ownership, policy separation, topology order, hard ceilings,
  resource identity, workspace rebasing, and execution order are fixed by this specification; an
  implementation need outside those decisions is a stop-and-replan condition.

## Known limitations

- New fusion is limited to ordinary pointwise IR. Mixed affine/pointwise, specialized epilogues,
  numerical multi-pass fusion, and state/random fusion remain deliberately split.
- Unit execution is topologically sequential; only existing child-local worker parallelism runs
  concurrently.
- Optional CPU 0005D external-read materialization is available only to one-unit plans until CPU
  0008E supplies a bounded multi-unit representation model.
- Hard fusion budgets may retain more materialized units than a later profitable plan. CPU 0008D
  owns ranking without weakening this task's legality/resource fallback.
- This is not public Engine integration or the CPU capability/conformance checkpoint; CPU 0009
  retains that closure.

## Validation evidence

- Clean implementation context: `01a03eaf-f654-7ef2-928f-592e051c467e`, based on
  `7223c95e520fc3f1e343d1633b1db891457439c2`; no staging, commit, or push was performed.
- Development compilation and focused lowering/preparation/finalization/composite/inventory tests
  passed. The focused generated evidence test passed direct semantics, Class-File member and
  allocation scans, and five accepted Java 26 fixed-heap forks.
- Follow-up clean implementation task `/root/cpu_0008b_followup` (no UUID exposed) corrected the
  completion blockers without staging, committing, or pushing. Its one authoritative final
  `./gradlew :backends:cpu:test --rerun-tasks` passed 99 suites and 512 tests with three existing
  expected skips, zero failures, and zero errors. Executable Java did not change afterward.
- Retained evidence directory:
  `/private/tmp/synaptik-cpu-0008b-evidence`. The authoritative-suite fresh accepted ratios are
  `1.058323828x`, `1.028316543x`, `1.058771135x`, `1.111571279x`, and `1.079831620x`; aggregate
  median is `1.058771135x`. Six earlier rejected attempts are retained accurately at `1.202949878x`,
  `1.213340020x`, `1.169101552x`, `1.199627968x`, `1.198502292x`, and `1.161363984x`; none is
  included in the accepted aggregate. The SHA-256 digest of `manifest.sha256` is
  `cbb1c7e06550a49303916dfe82be798f99e8f14e616420924e03710aa7d8b2c5` and every manifest entry
  verifies.
- The retained generated evidence test compares the two-store FLOAT64 NEG loop against optimal
  clean Java with the same one-pass primitive dataflow and ordered stores. The retained 457-byte
  Class-File has SHA-256
  `9c39fe6d782c90e8df951bc0553127fd36c1c3e7e9b712dcf3469a721a9ad08e` and decompiles to one
  61-instruction direct primitive loop with two ordered `dastore` instructions. Complete semantic,
  Class-File, and `javap` inspection found one method and zero fields, member references, field or
  invocation instructions, allocations, boxing paths, reflection paths, and string constants;
  raw-bit checks include NaN and signed zero. Existing one-unit artifacts retain their unchanged
  IR structural keys and specializations, so this task makes no redundant performance claim for
  those identity-reused forms.
- Clean documentation context: `01a03ece-baf7-7d00-8cc9-2a7928129279`. It independently reviewed
  the final production/test diff and retained evidence, finalized the affected Javadocs, CPU guide,
  glossary, task, master plan, and roadmap, and changed no executable Java behavior. The General,
  API/Javadoc, Planning, and Backend-guide profiles were applied. The glossary required updates
  because the existing CPU portable-route, preparation-plan, prepared-executable, execution-unit,
  and kernel-IR entries still described 0008B as Draft and the former straight-line/single-output
  boundary.
- The documentation review's liveness blocker is resolved. External inputs now become live at
  first actual use, instruction results become live at definition, duplicate same-instruction
  operands count once, and values die after their final instruction/store event. Indexing
  complexity now also follows the specified materialized-binding-only rule with checked
  arithmetic. Exact 16/17 live-value and at/next boundary, indexing, and code-size decisions are
  pinned by focused tests.
- Fifteen new focused acceptance tests cover diamond and seven-way fan-out, publication, affine,
  numerical reduction, random/state barriers, deterministic repeatability, one/eight node and unit
  edges, structural hard-ceiling edges, malformed later topology, exact first-use buffers and
  carriers, two workspace-bearing units with final IDs zero and one, missing/extra/mismatched
  complete resource sets, missing/undersized/closed later workers, invalid later carrier and
  workspace, read-only output, cold failure-before-write, no later execution after child failure,
  and successful one-range then two-range completion. Existing composite coverage retains
  cross-unit alias rejection before the Conv2d write.
- A focused documentation-context regeneration of the exact retained two-store NEG IR produced a
  457-byte Java 26 Class-File with SHA-256
  `9c39fe6d782c90e8df951bc0553127fd36c1c3e7e9b712dcf3469a721a9ad08e`.
  `javap -v -c -p /private/tmp/synaptik-cpu-0008b-multistore.class` showed one static primitive
  method, one direct integer-indexed loop, two `daload`/`dneg` computations, two ordered
  `dastore` instructions, no method invocation, no field access, no allocation opcode, and no
  constant-pool member reference. This closes the decompilation/hidden-helper inspection for the
  genuinely new emitted form without widening the performance claim.
- Final clean documentation context `/root/cpu_0008b_doc_review` independently inspected the
  final production/test diff, first-use/definition/final-use liveness, materialized-only indexing,
  final-index workspace rebasing, dead special-case removal, the focused acceptance matrix, and
  every retained accepted and rejected performance sample. It changed no executable Java behavior,
  applied the General, API/Javadoc, Planning, and Backend-guide profiles, corrected the remaining
  stale Conv2d-sequence guide text, finalized the glossary and synchronized planning statuses.
- `./gradlew :backends:cpu:javadoc` passed after final documentation edits with only the two
  expected incubating-Vector-module warnings. Focused Markdown, status, exact-scope, manifest,
  staged-index, and whitespace validation also passed in the final documentation context.
- From `/private/tmp/synaptik-cpu-0008b-evidence`, `shasum -a 256 -c manifest.sha256` passed all
  16 retained entries and `shasum -a 256 manifest.sha256 generated/multi-store-neg.class`
  reproduced both recorded digests. Independent `javap -v -c -p` inspection reproduced the exact
  one-method/zero-field direct loop and absence of member references. Five-file Markdown
  link/anchor and fence/final-newline checks passed; all canonical task headings were present.
  Planning-status checks found exactly one Ready CPU row, 0008C, and no 0008C detail file.
- `git diff --check` and `git diff --cached --check` passed. Exact combined scope is seven CPU
  production/Javadoc paths, eight CPU test/evidence paths, and five documentation/planning paths,
  with no shared Java, architecture, Gradle, conformance, or integration path. The exact package
  inventory passed with 114 Java files across 11 declared packages; `git diff --cached --name-only`
  was empty and final status retained only intended unstaged paths.

## Implementation notes

- Added `CpuPartitionDagDecomposer` as the sole CPU-private owner of complete-partition validation,
  longest established seed selection, stable topological ordering, dependency indices, vertical-
  first/horizontal-second contraction, and the task's hard pointwise fusion budgets.
- Generalized ordinary pointwise lowering to deterministic DAG instruction order and multiple
  final stores while preserving existing instruction semantics and one-unit structural identity.
- Generalized preparation to one through eight per-unit plans. Every unit retains its route,
  strategy, boundary/carrier facts, optional exact workspace with requirement ID equal to its final
  unit index, family geometry, and dependencies. Multi-unit plans disable CPU 0005D external-read
  materialization and deduplicate ordinary logical-value buffers in first-use unit order.
- Corrected workspace rebasing so each selected family unit first passes its established one-unit
  validation with local requirement ID zero, then the general partition assembly creates the exact
  final-index declaration and carries that identity into unit runtime facts and requirements.
- Replaced the Conv2d-only sequence with `CpuPreparedPartitionExecutable`. Finalization resolves
  all exact buffer/workspace assignments and validates worker requirements before artifact lookup;
  the composite validates every selected representation and cross-buffer overlap before binding
  children, then executes already-bound children strictly in stable unit order.
- The published/illegal Conv2d suffix now reaches the same decomposer and general composite as all
  other split plans. Publication, fan-out, affine/specialized, numerical-order, and state/random
  edges retain materialized ordinary-buffer barriers.
- The clean documentation pass finalized Javadocs for topology ordering, defensive snapshots,
  ownership, resource lifecycle, parameters, results, failures, and the cold/hot-path boundary. It
  added the implemented DAG/fusion/resource mental model to the CPU guide and reconciled the
  directly affected glossary entries without changing architecture authority.

## Completion summary

- Completed changes: deterministic bounded partition-DAG decomposition; vertical and horizontal
  ordinary-pointwise fusion; stable per-unit dependencies; exact split buffers and per-unit
  workspaces; general finalization; atomic sequential composite; multi-store generated evidence.
- CPU production files changed/created/removed: `CpuPartitionDagDecomposer`,
  `CpuPartitionLowering`, `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`,
  `CpuPartitionFinalizer`, `CpuPreparedPartitionExecutable`, and removal of
  `CpuPreparedExecutableSequence` (seven paths, inside the fourteen-path ceiling).
- CPU test/evidence files changed/created/removed: package inventory, pointwise lowering,
  preparation, removal and replacement of the sequence/composite test path, decomposition and
  resource acceptance matrices, and generated multi-store evidence (eight paths, inside the
  fourteen-path ceiling).
- Tests and validation: authoritative CPU suite passed 512 tests across 99 suites with three
  expected skips; five accepted fixed-heap generated/direct forks and aggregate passed `<= 1.15x`;
  rejected measurements were retained.
- Documentation-agent review: final clean context `/root/cpu_0008b_doc_review` independently
  finalized all affected Javadocs and the CPU guide, glossary, task evidence/status, CPU master
  plan, and roadmap after reviewing the follow-up implementation and retained evidence.
- Documentation impact: the CPU guide now explains general decomposition, legal pointwise fusion,
  hard budgets, split buffers versus external materialization, unit workspaces, atomic binding and
  sequential execution, safe split fallback, and a fork/join example.
- Javadoc review: affected types now document topology, immutability, defensive copying, plan and
  resource ownership, nullability, failures, cold validation, and hot-path behavior; CPU Javadoc
  generation passed.
- Glossary impact: updated the existing CPU portable-route, preparation-plan, prepared-executable,
  execution-unit, and kernel-IR definitions; no new glossary heading was needed because 0008B
  changes the implemented boundary of existing reusable terms rather than introducing a new term.
- Unresolved issues: None.
- Follow-up required: None. CPU 0008C is the sole Ready CPU task; no detailed 0008C task
  specification was created.

Status: Complete
