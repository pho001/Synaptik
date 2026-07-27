# Planning Master Plan

## Goal

Make backend-neutral compile-time ownership, partitioning, capability, and logical memory decisions.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- backend intent propagation
- capability query contracts and matrices
- backend-neutral ownership scoring
- maximal same-owner partitioning
- logical materialization and memory requirements

## Out of scope

- fusion implementation
- concrete kernel or route selection
- physical allocation
- prepared execution and runtime residency

## Module invariants

- Planning answers where work runs, not which implementation runs it.
- Scoring uses compile-time information only.
- Scoring output is backend ownership.

## Allowed dependencies

- modules/model, with public visibility when planning signatures expose model types
- modules/config
- modules/backend-contract, with public visibility when planning signatures expose backend
  identity types
- modules/trace

## Forbidden dependencies

- runtime, prepare, engine, and concrete backend modules

## Package structure

```text
io.github.pho001.synaptik.planning/
  capability/  public operation capability query/provider contracts and one per-occurrence
               owner-selection collaboration over internal hard eligibility/baseline comparison
  ownership/   later public/cross-package ownership, candidate, and scoring contracts when a
               concrete compiler or partition consumer justifies them
  partition/   public immutable partition recipe and public deterministic maximal consecutive
               same-owner generation for the concrete compiler consumer
  memory/      public logical value requirements and plan plus public stateless derivation from
               the closed graph and ordered complete partitions for Compiler
```

The module root is not a catch-all facade. Task 0001 opens only `capability` with the immutable
operation-occurrence question and inward-facing provider collaboration. Complete task 0002 keeps its
per-query eligibility result and evaluation package-private because no external planner consumer
yet justifies a public matrix/evaluator or a public config signature. Complete task 0003 keeps its
smallest consumer colocated with that internal value, uses the eligible identity list directly as
the candidate set, and adds no public ownership package or facade. Complete task 0004 opens
`partition` with the architecture-named immutable recipe plus an internal generator over complete
per-node ownership; it did not initially widen the current capability selector or add
orchestration. Compiler 0005 now supplies the concrete cross-package consumer: it adds one
colocated public owner-selection collaboration and widens only the existing partition and memory
operations in their owning packages. Compiler retains graph-wide sequencing and owner-map
assembly.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Operation capability-query foundation](tasks/0001-operation-capability-query-foundation.md) | Complete | Completed model milestone, backend-contract 0001–0004, config 0001–0002, and trace foundation | Added one immutable operation-occurrence query and one backend capability-provider contract without a matrix, eligibility evaluation, diagnostics result, device query, or provider implementation. |
| 0002 | [Per-query backend hard eligibility](tasks/0002-per-query-backend-hard-eligibility.md) | Complete | 0001, config hard intent, and backend-contract supplied availability | Combined validated provider/snapshot associations, backend-level support, current availability, and exact hard intent into an internal ordered `BackendId` list without a public matrix, scoring, device selection, or ownership choice. |
| 0003 | [Ownership candidates and baseline scoring](tasks/0003-ownership-candidates-and-baseline-scoring.md) | Complete | 0002, config partition-scoring configuration | Consumes the internal hard-eligible list directly as the candidate set and selects one exact `BackendId` through preferred-class-first, provider-order-stable baseline comparison, without a candidate record, cost classification, cost profile, or implementation-route interpretation. |
| 0004 | [Maximal same-owner partitioning](tasks/0004-maximal-same-owner-partitioning.md) | Complete | 0003 | Groups maximal consecutive runs in validated topological node order by equal selected `BackendId`, producing owner-plus-node-ID recipes without compiler orchestration, graph-boundary duplication, lowering, or executable construction. |
| 0005 | [Logical materialization and memory requirements](tasks/0005-logical-materialization-and-memory-requirements.md) | Complete | 0004 | Added immutable per-value producer/consumer/output requirements and an architecture-named logical memory plan derived internally from the closed graph and ordered partitions, without physical sizing, allocation, transfers, publication binding, or runtime residency. |
| 0006 | [Planning contract closure audit](tasks/0006-planning-contract-closure-audit.md) | Complete | 0001–0005 | Audited capability, ownership, partition, logical-memory, public/internal surfaces, documentation, dependencies, and the compiler/prepare handoff; the durable audit records a `CLOSED` verdict without executable changes. |

## Milestones

- Intent and capability
- Ownership scoring
- Partitioning and logical memory

## Current status

Complete after the `CLOSED` verdict in the
[planning contract closure audit](planning-contract-closure-audit.md). Tasks 0001–0002 remain
Complete with the public query/provider boundary and package-private provider-ordered hard-
eligibility result.
Task 0003 adds the smallest internal owner-selection consumer: the eligible `BackendId` list is
the candidate set, preferred matches precede nonmatches, provider order resolves ties and
fallback, and an empty list fails terminally before selection.

Compiler 0005 is now the concrete consumer anticipated by the closure audit. Public
`BackendOwnerPlanning.selectOwner(...)` composes the unchanged package-private eligibility and
selector once per final graph node. Public `MaximalSameOwnerPartitioning.partition(...)` and
`LogicalMemoryPlanning.plan(...)` retain their completed semantics and are invoked directly after
Compiler assembles the complete owner map. Planning still owns no graph-wide workflow, compiler
artifact, provider registry, numeric cost model, device/route choice, physical memory, prepare, or
runtime behavior.

Task 0004 now provides one public `PlannedPartition(owner, nodeIds)` recipe and public stateless
maximal consecutive-run generation over a complete `Map<NodeId, BackendId>`. It preserves stored
topological order, equality-based ownership, exact graph-node and first-owner references, and
immutable results, including the empty result for a zero-node graph. It adds no ownership row,
public orchestration facade, graph traversal or reordering, graph-boundary DTO, phase split, cost
profile, lowering, or executable construction.

[Task 0005](tasks/0005-logical-materialization-and-memory-requirements.md) is Complete. It adds the
public `LogicalMemoryRequirement` and `LogicalMemoryPlan` recipes plus public stateless derivation
from `CompiledGraphModel` and ordered complete partitions. The generator validates complete graph-
order coverage and maximal owner runs before retaining exact `ValueId`, `TensorDescriptor`,
producer partition, distinct consumer partitions, and graph-output facts in graph-value order.
It adds no eager byte count, lifetime, transfer/copy, physical slot/allocation,
`PublicationBinding`, public orchestration, prepare/runtime/backend behavior, or cost
classification.

[Task 0006](tasks/0006-planning-contract-closure-audit.md) is Complete. Its historical clean
documentation-focused audit confirmed that the five then-public declarations were sufficient and
that the four evaluator/generator operations could remain internal until compiler orchestration
had a concrete consumer. Compiler 0005 supplies that consumer and opens only the three
package-cohesive operations it needs; eligibility and its selector remain internal. The audit's
semantic and dependency verdict remains unchanged.

Config task 0004 remains Draft pending a concrete cost-bearing consumer. No task is Ready, no
later detailed specification exists, and a separate reassessment must select the next frontier.

## Open questions

- The exact classification and units for a future planning cost profile remain deferred until a
  concrete cost-bearing consumer exists; they did not block the completed cost-free baseline.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- The current capability contract asks whether one named backend can semantically own an
  immutable operation occurrence. It is distinct from availability, hard-eligibility evaluation,
  scoring, routing, preparation, and execution.
- Concrete backends may later implement the planning-owned inward-facing provider contract;
  planning does not depend on them and task 0001 added no implementation.
- Planning task 0001 intentionally interleaves before config task 0003 so the typed question is
  stable before a scoring preference is exposed. The frontier then returns to config 0003.
- Complete config task 0003 keeps hard eligibility separate from soft ranking through one
  `Optional<DeviceClass>` preference. Planning will later own interpretation after its candidate
  and baseline scoring contracts are stable.
- Task 0002 uses provider encounter order as the deterministic backend order and associates
  provider/snapshot inputs by equal `BackendId`, never by parallel list position. Duplicate or
  missing identities are invalid supplied composition and fail before any capability call.
- Task 0002 keeps its exact evaluator/result package-private and retains only eligible
  provider-owned `BackendId` references. Exact-device/class requirements use the matching snapshot
  only to prove current matching availability; they neither claim device-level capability nor
  choose or retain a device.
- A valid no-match task-0002 result is an empty immutable list. Later planning orchestration must
  stop before scoring it and may not weaken a hard requirement; the public compile failure
  contract remains with that later consumer.
- Planning cost is separate from model autotuning. Planning may consume only backend-
  neutral estimates; it must not interpret routes, vector species/lanes, unroll, threads, chunks,
  tiles, kernels, workload-cache values, or other backend parameter vocabulary.
- Planning owns the semantics and generation of complete valid ownership/partition candidates.
  Future tuning orchestration may measure a bounded set opaquely but does not construct or
  reinterpret those candidates.
- Completed Planning 0003 establishes the cost-free baseline consumer without defining a cost
  classification. Config 0004 may later store only the immutable backend-neutral inputs that a
  later concrete cost-bearing consumer actually requires.
- No stable shared production `OperationFamily` or workload-bucket contract exists. Planning 0003
  must not invent one merely to keep the implementation queue moving.
- Task 0003 uses the existing package-private `BackendEligibility` directly through one
  package-private stateless selector in `planning.capability`. Compiler 0005 leaves both internal
  and adds one public colocated collaboration that composes them for a single query.
- The task-0002 eligible `BackendId` list is the complete task-0003 candidate set. No production
  candidate record or score map is required.
- Baseline comparison is lexicographic rather than numeric: the first eligible backend with the
  configured preferred class wins; when no preference or match exists, the first eligible backend
  wins; provider order resolves every tie.
- Empty hard eligibility fails internally before scoring with the exact task-0003 failure. Current
  package-private compiler orchestration adds node context and may not weaken the hard
  requirement.
- The current baseline consumes no cost quantity and therefore needs no production operation-
  family, workload-bucket, or cost-profile classification. Config 0004 remains Draft.
- Planning task 0004 uses `CompiledGraphModel.nodes()` as its exact deterministic adjacency and
  order contract. “Adjacent” means consecutive positions in that validated topological list, not
  graph-edge connectivity, numeric node-ID order, map order, graph phase, or operation family.
- The complete per-node ownership handoff is `Map<NodeId, BackendId>`. It associates by typed
  equality without introducing a production ownership-row type or relying on parallel list
  position. Task 0004 does not invoke or widen the package-private task-0003 selector.
- `PlannedPartition` contains only one exact `BackendId` owner reference and an immutable ordered
  non-empty `NodeId` list. The generator retains graph node references and the first node's owner
  reference while comparing identities by equality.
- Graph inputs and outputs remain values, a multi-output producer remains one indivisible node,
  and fan-out, merge, publication, or phase changes do not split an equal-owner consecutive run.
  Completed task 0005 derives boundary and logical-materialization facts from the graph plus
  partition node IDs; transfers and physical materialization remain with later lifecycle owners.
- The partition recipe and stateless generation operation are public for cross-package compiler
  consumption. Compiler, not Planning, assembles the complete owner map and controls graph-wide
  orchestration.
- Planning task 0005 consumes only `CompiledGraphModel` plus an ordered `List<PlannedPartition>`.
  It validates complete graph-order coverage and adjacent-owner maximality before deriving any
  value requirement because the public partition DTO can also be constructed directly.
- Task 0005 represents logical materialization through each value's exact producing partition,
  distinct consuming partitions, and graph-output flag. These primitive facts express partition
  inputs/outputs, cross-owner boundaries, publication preservation, and partition-internal values
  without a closed role enum or duplicated boundary lists on `PlannedPartition`.
- `TensorDescriptor` remains the logical size carrier. Task 0005 does not calculate element or
  byte counts because dynamic/expression dimensions can remain unresolved and physical backend
  representation, alignment, and padding belong after compile-time planning.
- `PublicationBinding` remains outside task 0005. It is standalone model data for a future
  compiler-owned `PublicationPlan`; `CompiledGraphModel.outputs()` supplies only the logical
  preservation obligation at this frontier.

## Risks

- Leaking backend route selection or runtime residency into compile-time scoring.
- Reintroducing a backend-wide average that hides materially different operation families, data
  types, shapes, and sizes.
- Growing eligibility into a public matrix, graph identity, scoring, device selection, or
  rejection diagnostics before those consumers are concrete.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).

Task 0001 adds only the query/provider contracts and package documentation, the two required
public dependency-visibility changes, focused planning and architecture tests, and synchronized
explanatory/planning documentation. Its implementation context passed 13 tests across four suites
with no failures, errors, or skips. The independent documentation context reused that evidence,
changed no executable behavior, and finalized the affected Javadocs, examples, terminology,
current-versus-planned boundaries, glossary entries, and planning records. The single final root
suite passed 1,079 tests across 141 suites with no skips, failures, or errors. A separate planning
step then made config task 0003 Ready; it is now Complete.

Config task 0003 is now Complete after its focused config validation and independent
documentation pass. It changed no planning Java, provider contract, module dependency, or scoring
behavior. A separate reassessment selected only Planning task 0002 as Ready because hard
eligibility needs no performance profile or scoring formula. Planning task 0003 and Config task
0004 remained Draft; after 0002, Config 0004 profile contracts were the likely next area before
Planning 0003 scoring, subject to another reassessment. Planning task 0002 is now Complete after
its focused implementation, final planning suite, independent documentation pass, and focused
documentation validation. A following reassessment drafted Config task 0004, but the terminology/
ownership reset retired that unimplemented design. Planning 0003 is now the prerequisite for a
future Config 0004 cost profile. Planning 0003 is now Complete with an exact cost-free consumer,
deterministic comparison, tie, and failure contract; Config 0004 remains Draft until a later
concrete cost-bearing consumer establishes classification and units. No next task becomes Ready
without a separate frontier reassessment.

Planning task 0003 completed in implementation context `/root/implement_planning_0003` and clean
documentation context `/root/implement_planning_0003/docs_planning_0003`. Its focused selector
suite passed 13 tests and the final planning suite passed 38 tests across three suites, all with
zero failures, errors, or skips. Planning Javadoc passed, generated public pages preserve the
public/internal boundary and omit the package-private selector from public indexes, Markdown
validation passed for 227 files, 4,014 links, 246 anchors, and 2,844 fence markers, and final
scope, status, source-surface, newline, trailing-whitespace, and `git diff --check` checks passed.
The completed change is exactly its authorized fourteen paths.

Planning task 0004 completed in implementation context `/root/implement_planning_0004` and clean
documentation context `/root/implement_planning_0004/docs_planning_0004`. Its two focused suites
passed 11 tests, and the final planning suite passed 49 tests across five suites with no failures,
errors, or skips. The documentation pass changed no executable behavior and reused that evidence.
It finalized the partition Javadocs, public/internal status explanations, glossary, task, master
plans, and roadmap; planning Javadoc, repository Markdown, generated-page, exact fifteen-path,
status, later-spec, dependency, forbidden-surface, newline, whitespace, and `git diff --check`
validation passed. A separate frontier reassessment then made only Planning 0005 Ready with its
detailed logical materialization/memory specification. Planning 0006, Config 0004+, Trace 0003+,
and Compiler work remain Draft, and no other detailed future specification was created.

Planning task 0005 completed in implementation context `/root/implement_planning_0005` and clean
documentation context `/root/implement_planning_0005/planning_0005_docs`. Its three focused suites
passed 14 tests and the final planning suite passed 63 tests across eight suites, all with zero
failures, errors, or skips. The documentation pass changed no executable behavior and reused that
evidence. It finalized the logical-memory Javadocs, architecture/API/guide/glossary explanations,
current-versus-planned boundaries, task evidence, master plans, and roadmap. Planning Javadoc,
repository Markdown, generated public/internal page, exact eighteen-path, status, later-spec,
dependency, forbidden-surface, newline, whitespace, and `git diff --check` validation passed.
Planning 0006, Config 0004+, Trace 0003+, and Compiler work remain Draft without another detailed
specification, and no global task is Ready pending a separate frontier reassessment.

A separate frontier reassessment in planning context `/root/plan_planning_0006` found no missing
planning semantic or architecture prerequisite after task 0005. It created only the detailed
documentation-only Planning 0006 closure-audit specification and made that row Ready. The audit
must verify the public/internal boundary and future compiler/prepare handoff before the selected
planning milestone can become Complete; every later task remains Draft without a detailed spec.
