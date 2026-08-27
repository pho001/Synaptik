# Task 0003A: Immutable Partition-Local DAG Analysis Projection

## Status

Complete

## Goal

Introduce one public immutable Prepare-owned directed acyclic graph (DAG) projection for exactly
one `PlannedPartition`. The projection gives concrete backend analysis stable partition-local
nodes and precomputed structural producer, consumer, edge, external-input-occurrence, and local-
sink facts without exposing the complete cross-backend `CompiledGraphModel` or making each
backend reconstruct those facts independently.

`PrepareContext` must retain this projection as its sole node/topology source of truth.
`PrepareContext.nodes()` remains a compatibility view that delegates to the projection rather
than retaining a second independent node list. `GraphPreparation` constructs the projection once
per planned partition through the authoritative complete-graph projection path and passes that
exact instance into the backend context.

## Scope

### Public immutable partition projection

- Add `io.github.pho001.synaptik.prepare.analysis.PartitionDag`, or a more precise domain name if
  implementation evidence demonstrates that `PartitionDag` is misleading. The final name and
  API must remain Prepare-owned, backend-neutral, public, immutable, and specific to one exact
  planned partition.
- Retain the partition's exact `CompiledNode` references in validated stable topological order.
  The projection contains no node from another partition and no `CompiledGraphModel` reference.
- Precompute immutable structural facts from each `CompiledNode`'s ordered input and output
  `ValueId` lists. Facts must retain exact producer and consumer node positions plus output/input
  port positions.
- Preserve repeated input occurrences independently. If one node consumes the same `ValueId` in
  two input ports, both consumer occurrences are present in port order.
- Preserve every output port of a multi-output node independently and associate each produced
  `ValueId` with its exact node and output-port position.
- Identify every partition-external input occurrence: an input port whose `ValueId` has no
  producer inside this partition. This is a structural occurrence, not an ownership or transfer
  decision.
- Identify local sink nodes deterministically. A local sink has no output occurrence consumed by
  another node in this partition; graph publication and cross-partition use do not change that
  topology-only definition.
- Expose only narrow indexed or value-keyed queries required by current and near-term analysis
  consumers. Returned lists and occurrence facts are immutable deterministic snapshots. Do not
  introduce a mutable generic graph library, visitor/callback framework, or rewrite surface.

### Fail-closed construction and validation

- Require the projection's node IDs to equal `PlannedPartition.nodeIds()` exactly in size,
  identity, and order.
- Reject duplicate node IDs and duplicate produced `ValueId` identities, including duplicates
  across output ports of the same node or different nodes.
- Reject a node input produced by that same node or by a later partition node. A missing local
  producer is an external-input occurrence and remains valid.
- Reject every disagreement between the supplied `PlannedPartition` and stored nodes before the
  projection can reach backend analysis.
- Derive adjacency and edge facts once from the validated ordered ports. Do not accept caller-
  supplied adjacency that could disagree with the nodes.
- Preserve deterministic order based on partition node order and port order, never hash-map
  iteration, object identity, or numeric `ValueId` magnitude.

### PrepareContext and orchestration integration

- Replace the independent stored node-list component of `PrepareContext` with the exact
  partition-local DAG projection. `nodes()` delegates directly to `partitionDag().nodes()` (or
  the chosen equivalent accessor) and remains immutable.
- Preserve the current values, logical memory requirements, constants, backend inputs, fully
  static descriptor validation, exact-reference retention, deterministic collection order, and
  staged backend analysis/finalization behavior.
- Preserve a source-compatible public constructor accepting the current
  `(PlannedPartition, List<CompiledNode>, List<GraphValue>,
  List<LogicalMemoryRequirement>, Map<ValueId, ScalarValue>, I)` arguments if Java record/class
  rules and the final truthful API shape permit it. That constructor must construct exactly one
  validated projection and delegate to the authoritative representation; it must not store the
  list separately.
- Make `GraphPreparation` construct each partition projection once while it holds the complete
  compile graph, then pass only that projection and the existing partition-local value/memory/
  constant facts into `PrepareContext`. A concrete backend must never receive the complete
  cross-backend `CompiledGraphModel`.
- Keep all contexts constructed and validated before the first backend preparer invocation, as
  required by completed Prepare 0003.

### Compatibility evidence

- Record the exact final API shape from implementation evidence. In particular, test and document
  the canonical constructor, any list-taking compatibility constructor, record components or
  class properties, `nodes()` delegation, reflection surface, equality, `hashCode`, and
  `toString` behavior.
- Source compatibility for the current constructor call shape is the planned goal where
  technically valid. Do not claim binary compatibility: changing a public record component or
  canonical constructor can change descriptors, record reflection, equality, and textual form.
- Prefer the narrowest truthful API. Do not preserve obsolete record-component or textual shape
  by retaining duplicate topology state or inventing a compatibility wrapper with ambiguous
  ownership.

### Documentation and validation

- Add detailed Javadoc for the projection, every public occurrence/edge fact and query, all
  inputs/results/failures, immutability, ordering, repeated occurrences, multi-output semantics,
  and topology-only limitations.
- Update `PrepareContext`, `GraphPreparation`, and `prepare.analysis` package Javadocs where the
  source of node/topology facts changes.
- Add focused Prepare tests for structural construction, validation, stable order, immutable
  queries, delegation, compatibility, reflection, equality, and orchestration identity.
- Run downstream CPU source-compatibility and behavior checks without migrating CPU DAG
  reconstruction. CPU 0008E1 owns consumption and deletion of duplicated CPU scans later.
- Inspect architecture tests for the new public Prepare package surface and dependency rules.
  Update them only if their existing inventories or public-surface rules require the new type;
  no dependency edge is expected to change.
- Use a separate clean documentation-focused agent after executable Java stabilizes. That pass
  finalizes affected Javadocs, public/Prepare API explanations, backend-contributor guidance,
  glossary impact, and planning evidence without repeating successful Java tests unless it
  changes executable behavior or records a concrete stale-evidence risk.
- Run repository-wide validation after implementation and documentation stabilize because this
  task changes a shared public backend-analysis contract consumed by concrete backends.

## Out of scope

- CPU adoption, changes to `CpuPartitionDagDecomposer`, or deletion of CPU producer/consumer scans
- CPU MATMUL, linear epilogues, fusion policy, profitability, materialization selection, generated
  code, artifact schema, performance claims, capability reporting, or Runtime behavior
- cross-partition producer/consumer ownership inference, publication policy, logical/physical
  memory policy, materialization policy, fusion legality, route choice, cost, or scheduling
- a complete-model DAG, global graph access, graph regions, callbacks, visitor APIs, rewrites,
  mutable adjacency, execution units, backend-specific facts, or a generic graph framework
- changes to `CompiledGraphModel`, `CompiledNode`, `PlannedPartition`, logical-memory semantics,
  Compiler artifacts, Runtime recipes, Engine, Config, Trace, or another backend
- Prepare 0004 opaque candidate/tuning handoff or any tuning/cache behavior
- an architecture-contract, ADR, dependency, Gradle, backend-conformance, or integration change
  unless implementation exposes a concrete contradiction; in that case stop and report it

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): Prepare lifecycle, Prepare ownership,
  backend-facing Compiler isolation, staged backend analysis/finalization, and Runtime boundary.
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md).
- [`lifecycle`](../../../../architecture/lifecycle.md).
- [`module boundaries`](../../../../architecture/module-boundaries.md).
- [`dependency rules`](../../../../architecture/dependency-rules.md).
- [`Runtime, Prepare, and Backend Boundary`](../../../../architecture/runtime-prepare-backend-boundary.md).
- [`ADR 0010`](../../../../design/decisions/0010-staged-backend-preparation.md).
- [`planning guide`](../../../planning-guide.md).
- [`Prepare master plan`](../master-plan.md).
- Complete Prepare tasks [0001](0001-backend-partition-analysis-and-resource-declaration.md),
  [0002](0002-backend-partition-finalization-handoff.md), and
  [0003](0003-prepare-orchestration-and-validation.md).
- [`CPU master plan`](../../../backends/cpu/master-plan.md) and completed CPU 0008B–0008E as
  downstream behavior authorities.

## Architecture constraints

- `modules/prepare` owns the backend-neutral projection and may depend only on its existing
  authorized inward modules. It must not depend on a concrete backend.
- Concrete backends receive only exact partition-scoped semantic/planning facts and backend-owned
  inputs, never `CompileArtifacts`, `CompiledGraphModel`, or another Compiler-owned aggregate.
- Planning still selects ownership only. The projection must not derive cross-partition owner,
  route, kernel, fusion, memory, publication, transfer, or performance policy from topology.
- Backend analysis remains the owner of lowering, specialization, fusion, route choice, and exact
  shared-resource declarations. Finalization and Runtime contracts remain unchanged.
- The Runtime hot path remains free of `Operation`, `CompiledNode`, graph inspection, selection,
  and scheduling decisions.
- The graph remains flat. This task introduces no region, callback, nested graph, control-flow
  representation, or execution DAG.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.prepare.analysis` — public partition-scoped backend-analysis facts.
- `io.github.pho001.synaptik.prepare` — shared complete-graph projection orchestration.

Packages added or changed:

- No new package. The existing public `prepare.analysis` package gains one cohesive topology
  projection and only the narrowly required immutable nested or sibling fact types.

Type placement:

- `io.github.pho001.synaptik.prepare.analysis.PartitionDag` (preferred name) — Prepare-owned
  immutable structural projection for exactly one planned partition.
- Any public producer/consumer/edge occurrence type must be nested in or colocated with
  `PartitionDag` and justified as part of the same narrow query contract. Do not add a generic
  `graph`, `util`, or backend-specific package.

Tests mirror the production packages. Black-box reflection tests remain in the existing Prepare
test packages.

## Affected files

Expected production/Javadoc paths:

- new `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PartitionDag.java`;
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PrepareContext.java`;
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/package-info.java`;
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/GraphPreparation.java`.

Expected focused Prepare test paths:

- new `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/analysis/PartitionDagTest.java`;
- existing `PrepareContextTest`, `AnalysisPublicShapeTest`, `GraphPreparationTest`, and
  `GraphPreparationPublicShapeTest` only as required by the final truthful API shape.

Expected downstream compatibility-test path:

- existing
  `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuNonAffineMovementLoweringTest.java`
  solely to move its malformed duplicate-producer rejection expectation from CPU lowering to
  `PartitionDag`/`PrepareContext` construction. No other assertion, fixture, CPU behavior, or CPU
  test owner is reopened.

Expected documentation/planning completion paths:

- `docs/api/public-api.md` and the focused Prepare/backend API or guide page that currently
  explains `PrepareContext`;
- `docs/architecture/runtime-prepare-backend-boundary.md` only for implemented-status
  clarification, not authority changes;
- `docs/glossary.md` only if the final public term requires a reusable definition;
- this task, the Prepare master plan, CPU master plan, and roadmap.

Review only unless a concrete accepted rule requires a change: `ARCHITECTURE.md`, ADR 0010,
module/lifecycle/dependency explanations, `CompiledGraphModel`, `PlannedPartition`, Runtime and
Compiler APIs, CPU 0008B–0008E source/tests, build files, architecture tests, backend conformance,
and integration tests.

## Maximum scope

This task may create or modify at most 20 paths:

- 4 Prepare production/Javadoc paths, including exactly 1 new top-level production type;
- 5 Prepare test paths, including exactly 1 new top-level test type;
- at most 3 explanatory API/architecture/guide paths;
- at most 1 glossary path;
- 4 planning paths: this task, Prepare master plan, CPU master plan, and roadmap; and
- exactly 1 existing CPU test path, solely
  `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuNonAffineMovementLoweringTest.java`,
  with no CPU production change and no broader CPU behavior or test refactor; and
- at most 2 architecture-test paths only if an existing exhaustive package/public-surface
  inventory requires them.

No CPU production or other CPU test, Model, Planning, Compiler, Runtime, Engine, Config, Trace,
other-backend, Gradle, ADR, backend-conformance, or integration path may change. The one authorized
CPU test edit may only make the existing malformed duplicate-producer case assert rejection while
constructing `PartitionDag` or `PrepareContext`, before CPU lowering. If another production type,
package, module edge, behavior owner, CPU assertion, or CPU test path is required, stop and replan.

## Acceptance criteria

- One public immutable Prepare-owned partition DAG contains exactly the planned partition's
  stable topological nodes and no complete-model reference or out-of-partition node.
- Producer/output-port, consumer/input-port, edge, repeated-input, multi-output, external-input-
  occurrence, and local-sink facts are exact, immutable, deterministic, and queryable without
  backend reconstruction.
- Duplicate producers, duplicate node IDs, self/later-produced dependencies, and partition/order
  disagreement fail closed before backend analysis.
- The existing malformed duplicate-producer case in `CpuNonAffineMovementLoweringTest` asserts
  this rejection at `PartitionDag`/`PrepareContext` construction rather than expecting CPU
  lowering to own it; no other CPU expectation or behavior changes.
- `PrepareContext` retains the DAG as the sole node/topology state; `nodes()` delegates to it and
  no second list is stored or copied independently.
- The current list-taking construction form remains source-compatible if technically valid and
  delegates through exactly one DAG construction. The task records any unavoidable source-shape
  limitation truthfully.
- Reflection, record-component/class-property shape, constructors, equality, `hashCode`,
  `toString`, and compatibility behavior are pinned by automated tests and documented without a
  binary-compatibility promise.
- `GraphPreparation` constructs each DAG once through the authoritative full-graph projection
  phase, retains all-contexts-before-analysis ordering, and passes no `CompiledGraphModel` or
  complete cross-backend node list to a backend.
- Values, logical memory requirements, constants, backend inputs, fully-static validation,
  staged analysis/finalization, slot assignment, schedule assembly, and prepared result behavior
  remain unchanged.
- The public query surface contains no generic mutation/rewrite/callback/region/scheduling API and
  no CPU, route, memory, publication, fusion, or performance policy.
- Focused Prepare tests, downstream CPU source/behavior checks, Javadoc, documentation validation,
  repository tests, applicable architecture tests, exact scope, and whitespace gates pass.
- A separate clean documentation-focused pass finalizes all affected Javadocs and documentation,
  records glossary impact, and reuses stable implementation-test evidence appropriately.
- CPU production source and all CPU 0008B–0008E topology identities, stable order, legality,
  candidate counts, generated code, schema 53, performance evidence, and Runtime behavior remain
  unchanged. CPU test source changes only at the one explicitly authorized malformed-construction
  expectation.
- CPU 0008E1 remains Draft without a detailed specification; CPU 0008F remains Draft without a
  detailed specification; Prepare 0004 remains Draft and follows 0003A without renumbering.

## Tests / validation

Focused implementation tests must cover:

- empty, one-node, linear, branching/diamond, disconnected, repeated-input, and multi-output
  partitions;
- exact producer/output-port and consumer/input-port positions, external-input occurrence order,
  edge order, local sinks, value lookup, node lookup, and immutable snapshots;
- duplicate node IDs, duplicate producers within/across nodes, partition size/order/identity
  disagreement, self-dependency, later-producer dependency, nulls, and malformed ports;
- `PrepareContext.nodes()` exact DAG delegation and exact reference retention;
- current values/memory/constants/backend-input validation and dynamic-Shape failure;
- canonical and compatibility constructor behavior, reflection/record components, equality,
  `hashCode`, and `toString`;
- `GraphPreparation` constructs all partition DAGs before analysis, constructs each once, passes
  only partition-local nodes, and preserves analysis/finalization/schedule ordering.

Run focused tests while implementation stabilizes. After the last executable Prepare change run
one final affected-module command:

```bash
./gradlew :modules:prepare:test
```

Compile and test the downstream CPU consumer, proving that the compatibility surface remains
sufficient before CPU 0008E1. The sole permitted CPU edit is the expectation-only relocation in
`CpuNonAffineMovementLoweringTest` described above:

```bash
./gradlew :backends:cpu:test
```

The clean documentation-focused pass receives those exact results, changes no executable
behavior, and runs:

```bash
./gradlew :modules:prepare:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also inspects rendered Javadocs and validates all changed Markdown links/anchors, canonical
task headings, fences, final newlines, terminology, exact path scope, package placement, public
surface, no full-graph exposure, status/dependency order, absence of CPU 0008E1/0008F detailed
specifications, and empty staging.

After implementation and documentation stabilize, run the shared-contract checkpoint once:

```bash
./gradlew test :testing:architecture-tests:test
```

Architecture-test source changes are conditional on an existing exhaustive inventory. Backend
conformance and integration source remain unchanged because this task adds no concrete capability
or end-to-end behavior; their existing suites participate through the repository checkpoint.

## Dependencies

- Prepare 0001–0003 — Complete.
- Planning 0004–0006 and Compiler 0005–0006 — Complete producers of the current immutable graph,
  partitions, memory, constants, and publication facts.
- Runtime 0002–0014 — Complete consumers of unchanged prepared memory/schedule contracts.
- ADR 0010 — Accepted/current.
- CPU 0008B–0008E — Complete downstream reconstruction and behavior evidence that motivates this
  user-authorized interleave.

No unfinished architecture or shared-contract dependency blocks implementation.

## Follow-up tasks

- CPU 0008E1, `Shared partition-DAG adoption and reconstruction removal`, is a separate Draft CPU
  task. It consumes the shared projection in `CpuPartitionDagDecomposer` and directly related
  cold analysis, removes duplicated producer/consumer scans where the shared facts are exact, and
  preserves every CPU 0008B–0008E behavior/evidence invariant. No detailed CPU 0008E1 task
  specification exists yet.
- CPU 0008F, portable MATMUL execution and bounded linear epilogues, remains Draft, depends on CPU
  0008E1, and has no detailed task specification.
- Prepare 0004 remains the existing Draft opaque backend-candidate/tuning-artifact handoff. It
  follows 0003A without renumbering and is not implemented or specified here.

## Architecture impact

Expected impact: None.

This task makes the existing Prepare projection more explicit and prevents complete-model graph
leakage to concrete backends. It stays within Prepare's current ownership of exact partition-
scoped semantic/planning facts and the existing dependency direction. It changes no authoritative
architecture rule, ADR decision, module edge, Runtime hot path, or concrete backend policy.

If implementation requires changing `ARCHITECTURE.md`, exposing `CompiledGraphModel` to a backend,
moving backend-specific analysis into Prepare, adding a shared execution/scheduling graph, or
changing dependency direction, stop and report the exact uncertainty instead of implementing it.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the isolated implementation agent for Synaptik Prepare task 0003A. Do not stage, commit,
or push.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md, the focused
lifecycle/module-boundary/dependency/runtime-prepare-backend documents and ADR 0010,
docs/planning/planning-guide.md, docs/planning/roadmap.md, the Prepare master plan and tasks
0001–0003, this task, the CPU master plan and completed CPU 0008B–0008E tasks, documentation
rules/profiles, and final affected source/tests/Javadocs.

Implement exactly this Ready task inside its authorized scope. Add one immutable public
partition-local DAG projection in Prepare, make it PrepareContext's sole topology source, retain a
truthful source-compatible list-taking constructor where technically valid, and make
GraphPreparation construct/pass the projection once. Preserve existing values, memory,
constants, backend inputs, static validation, staged lifecycle, and downstream CPU behavior. Do
not migrate CPU reconstruction, expose the complete model DAG, add policy or graph-rewrite APIs,
or implement Prepare 0004, CPU 0008E1, CPU 0008F, or later work. Stop on architecture,
compatibility, package, or maximum-scope conflict.

After stable implementation and recorded Prepare/CPU validation, hand the same diff and evidence
to a separate clean documentation-focused agent. That pass must independently finalize affected
Javadocs, API/guide/boundary documentation, glossary impact, and planning evidence, then run the
recorded documentation checks without repeating successful Java tests unless executable behavior
changes or a concrete stale-evidence risk is recorded. Run the repository/architecture checkpoint
once only after both passes stabilize. Update this task's decisions, limitations, evidence,
implementation notes, completion summary, and status only from actual results.
```

## Local decisions

- `PartitionDag` is the preferred domain name because the value represents one exact planned
  partition, not the complete compiled graph or an execution schedule. Implementation may select
  a more precise name only with recorded source/API evidence and synchronized package Javadoc.
- Structural occurrence facts preserve positions rather than deduplicating by `ValueId`.
  Value-keyed queries may group immutable occurrence lists, but repeated consumer ports remain
  distinct.
- A local sink is topology-only: none of the node's output ports has a local consumer. It does not
  infer publication, partition boundary, memory, or backend ownership.
- The authoritative DAG derives adjacency from ordered `CompiledNode` ports. No constructor takes
  caller-authored edge or adjacency collections.
- `PrepareContext.nodes()` remains for source compatibility but is a derived compatibility view.
  The final stored component/property must be the DAG, not both DAG and nodes.
- Record reflection/equality/textual compatibility changes are accepted only when they are the
  truthful consequence of one-source-of-truth state and are pinned explicitly. Binary
  compatibility is not promised.
- CPU adoption is deliberately separate so this Prepare capability can be validated independently
  while the unchanged list compatibility view keeps the existing CPU source compiling.
- The malformed duplicate-producer fixture in `CpuNonAffineMovementLoweringTest` already crosses
  the new Prepare-owned fail-closed boundary. Its rejection expectation therefore belongs at
  `PartitionDag`/`PrepareContext` construction, not at `CpuPartitionLowering.lower`; moving that
  one expectation is compatibility maintenance and does not authorize CPU production or policy
  work.
- No glossary change is assumed. The documentation pass must update it only if the final public
  term establishes a reusable project-wide distinction not already covered.
- Open questions: None. A need for global ownership facts, policy, execution scheduling, another
  public framework, or architecture changes is a stop-and-replan condition.

## Known limitations

- The projection is partition-local and cannot answer cross-partition ownership, publication,
  transfer, memory, fusion, route, or performance questions from topology alone.
- `PrepareContext` remains fully static; this task adds no dynamic binding or run-dynamic geometry.
- Existing CPU code may continue reconstructing equivalent facts through `nodes()` until CPU
  0008E1. This task intentionally does not remove that duplication.
- The one authorized CPU test correction changes only where one malformed duplicate-producer
  fixture expects failure. It does not broaden CPU adoption, lowering behavior, or test cleanup.
- The compatibility constructor, if retained, provides source-level convenience only. Record
  component, reflection, equality, `hashCode`, `toString`, serialized form, and binary linkage are
  not promised to remain identical.
- No public generic graph framework, rewrite facility, execution scheduler, or full-model view is
  introduced.

## Validation evidence

Planning/documentation correction context: `/root/prepare_0003a_doc_correction`.

- `./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest.rejectsExcludedWindowSignaturesTypesLayoutsAndOverflow`
  failed with one test failure at `CpuNonAffineMovementLoweringTest.java:177`. The failure occurs
  while constructing `PrepareContext` from two nodes that produce the same `ValueId`, before the
  line-179 assertion can invoke CPU lowering. This is the demonstrated maximum-scope contradiction
  corrected by authorizing only that existing CPU test's expectation relocation.
- The correction context inspected the complete current Prepare Java/test diff, including the new
  `PartitionDag` and `PartitionDagTest`, and found no need to authorize CPU production behavior or
  any second CPU test path.
- The first local Ruby heading validator stopped before document evaluation because this Ruby
  lacks `Array#filter_map`; the compatible `map`/`compact` retry passed all 20 canonical headings,
  relative Markdown links, backtick fences, and the final newline.
- The current worktree scope validator passed with 12 authorized paths and no CPU production or
  second CPU test path. `git diff --check` and `git diff --cached --check` passed, and
  `git diff --cached --name-only` was empty.
- `git diff --no-index --check /dev/null <this-task>` produced no whitespace diagnostic; its exit
  status was the expected `1` because the task is an untracked non-empty file.

Planning/documentation context: `01a04334-7f08-78f3-9bf0-6efb1684b68e`.

- The context read the required architecture, lifecycle, module-boundary, dependency,
  Runtime/Prepare/backend, ADR 0010, planning, Prepare, CPU 0008B–0008E, documentation-profile,
  source, and focused-test material. No architecture-contract change or uncertainty was found.
- A local Ruby validator passed all four changed Markdown files. It confirmed that relative link
  targets exist, backtick fences are balanced, and every file has a final newline.
- Canonical heading inspection found all required task headings in order from `Status` through
  `Completion summary`. The first Ruby heading helper failed before evaluating the file because
  this environment lacks `Array#filter_map`; the compatible `map`/`compact` retry passed all 20
  headings. This was a validator compatibility failure, not a document failure.
- At that pre-implementation planning checkpoint, status/dependency scans found exactly one
  `Ready` master-plan row, Prepare 0003A. CPU 0008E1 and CPU 0008F were Draft; CPU 0008F depended
  on 0008E1; Prepare 0004 remained Draft after 0003A without renumbering.
- `find docs/planning/backends/cpu/tasks -maxdepth 1 -type f` found no CPU 0008E1 or CPU 0008F
  task specification.
- The combined tracked/untracked path scan found exactly the four authorized planning paths.
  `git status --short -uall` confirmed all four are unstaged and the staged index is empty.
- `git diff --check` and `git diff --cached --check` passed.
- No Java, test, Javadoc, Gradle, repository, performance, generated-code, or schema command ran
  because this pass changes planning Markdown only. No implementation evidence is claimed.

Implementation context evidence reused by the final documentation pass:

- The final implementation-owned `./gradlew :modules:prepare:test` passed 41 tests with no
  failures, errors, or skips. No executable Java or test changed afterward in the documentation
  context.
- The final implementation-owned `./gradlew :backends:cpu:test` passed 104 suites and 544 tests
  with 3 expected skips and no failures or errors. The sole CPU edit is the authorized malformed
  duplicate-producer assertion relocation; CPU production is unchanged.
- Focused implementation tests passed before those final module commands. The implementation
  reported exactly 13 authorized paths and an empty staged index at handoff.

Final clean documentation context: `01a043d7-113c-7ee2-8257-42678c1a7be4`.

- Reviewed the authoritative architecture contract, focused lifecycle/module/Prepare/backend
  explanations, public API status, backend preparer guide, glossary, General/API-Javadoc/
  Planning/Example profiles, final Java/Javadoc/test diff, task, Prepare and CPU master plans, and
  roadmap. No architecture, dependency, Gradle, Runtime, Compiler, Model, region, policy, or
  complete-model exposure conflict was found.
- Finalized `PartitionDag`, `PrepareContext`, `GraphPreparation`, and package Javadocs plus the
  public API status, focused Runtime/Prepare/backend boundary, backend preparer guide, and
  glossary. The glossary change is warranted because `PartitionDag` is a reusable public project
  term whose topology-only boundary must be distinguished from policy.
- Documented the exact public shape: public final immutable `PartitionDag`; exact occurrence/port
  order including repeated inputs and multi-output nodes; external-input occurrences; local
  sinks; topology-only limitations; and the five-component canonical `PrepareContext` record plus
  its six-argument source-compatible constructor. No binary compatibility is promised for the
  changed record descriptor, reflection, equality, hash code, or text form.
- `./gradlew :modules:prepare:javadoc` first passed with two no-main-description warnings on
  `externalInputs()` and `localSinks()`. After correcting those Javadocs, the final identical
  command passed cleanly. Rendered HTML and `javap -public` inspection confirmed the intended
  type, constructor, method, nested-record, and documentation shape.
- The first local Ruby Markdown validator stopped on an interpolated heading-regex syntax error;
  the second stopped because this Ruby lacks `Array#filter_map`. The compatible
  `map`/`compact` validator then passed all eight changed Markdown files for relative link targets,
  anchors, backtick/tilde fences, and final newlines.
- The exact-scope validator passed with 17 authorized paths, within the maximum 20. It found
  exactly the one authorized CPU test path, no CPU production, no second CPU test, and no
  forbidden path. CPU 0008E1 and CPU 0008F remain Draft and have no detailed task specification;
  Prepare 0004 remains Draft after 0003A.
- `git diff --check` and `git diff --cached --check` passed. `git diff --cached --name-only` was
  empty, and final status inspection confirmed all 17 paths remain unstaged.
- The task-required checkpoint ran exactly once as
  `./gradlew test :testing:architecture-tests:test` and passed. Current XML reports total 364
  suites and 2,491 tests, with 3 expected skips and no failures or errors; architecture tests
  contributed 4 suites and 6 tests. The checkpoint's Prepare report contains 11 suites and 42
  tests with no skips/failures/errors, and its CPU report contains 104 suites and 544 tests with
  the same 3 expected skips and no failures/errors.

## Implementation notes

- Added one Prepare-owned immutable `PartitionDag` that validates exact planned membership and
  topological order, derives deterministic occurrence and adjacency facts once, and exposes no
  complete graph, region, policy, or execution model.
- `PrepareContext` now stores the DAG as its only topology component. Its canonical record has five
  components; the six-argument partition-and-node-list constructor constructs exactly one DAG and
  preserves source calls without claiming binary compatibility.
- `GraphPreparation` constructs every partition DAG once while projecting the complete compile
  graph and completes all contexts before invoking a backend preparer.
- The existing CPU malformed duplicate-producer fixture now expects rejection at the shared
  Prepare construction boundary. No CPU production behavior, reconstruction, generated code,
  schema, fusion policy, or Runtime behavior changed.
- CPU adoption and reconstruction deletion remain Draft CPU 0008E1. CPU 0008F remains Draft after
  it, and neither task has a detailed specification.

## Completion summary

- Completed changes: Added the immutable partition-local DAG projection, made it
  `PrepareContext`'s sole topology source, integrated one-time graph preparation, and relocated
  the one stale CPU malformed-input expectation to the shared validation boundary.
- Files changed or created: 17 authorized paths: 4 Prepare production/Javadoc, 4 Prepare tests,
  4 explanatory documentation/glossary, 4 planning, and exactly 1 CPU test.
- Tests and validation: Reused the stable 41-test Prepare and 544-test CPU implementation runs;
  final Prepare Javadoc, rendered/public-shape inspection, Markdown, exact-scope, staging, and
  whitespace gates passed; the one required repository/architecture checkpoint passed 2,491
  tests with 3 expected skips and no failures/errors.
- Documentation-agent review: Clean documentation context
  `01a043d7-113c-7ee2-8257-42678c1a7be4` independently finalized the affected Javadocs,
  explanatory documentation, glossary, planning status, and evidence without changing executable
  Java or tests.
- Documentation impact: Updated public API status, the focused Runtime/Prepare/backend boundary,
  backend contributor guidance, glossary, and synchronized task/master/roadmap state.
- Javadoc review: Final Prepare Javadoc passed cleanly; rendered HTML and `javap -public` confirmed
  the intended immutable DAG and five-component/six-argument context surface.
- Glossary impact: Added `PartitionDag` and updated `PrepareContext` because the new public term and
  topology-versus-policy distinction are reusable across backend contributors.
- Unresolved issues: None.
- Follow-up required: None for Prepare 0003A. CPU 0008E1 and CPU 0008F remain separate Draft work
  without detailed specifications.

Status: Complete
