# [Planning](../../../glossary.md#term-planning) Master Plan

> **Editorial historical reading key.**
>
> Treat the task statuses and authorized frontiers here as historical checkpoints, not a current
> support inventory. Compiler supplies exact operation occurrences, providers, availability snapshots,
> and intent to Planning; Planning determines backend ownership, consecutive same-owner partitions,
> and logical per-value requirements. Compiler owns graph-wide sequencing and owner-map assembly. A
> provider’s semantic-support answer is neither availability nor a device, route, kernel, or
> executable selection.
>
> The [current capability and partition
> contracts](../../../reference/compiler/planning.md#current-operation-capability-contracts) explain
> the implemented per-occurrence collaboration. The [current artifact
> pipeline](../../../reference/compiler/artifacts.md#current-complete-artifact-compilation-and-integration-port)
> shows how Compiler integrates its results. Planning’s immutable identities and descriptors cross
> that boundary without physical buffers or cleanup ownership; Prepare, backends, Runtime, and Engine
> retain downstream responsibilities. In particular, this plan’s pending activity/query frontier and
> older body-query constructor narrative must not override the current five-component query contract.

## Goal

Make [backend](../../../glossary.md#backend)-neutral [compile](../../../glossary.md#compile)-time [ownership](../../../glossary.md#term-ownership-meanings), partitioning, capability, and logical memory decisions.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../glossary.md#architecture-contract))</span>
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>

## [Scope](../../../glossary.md#term-scope)

- [backend intent](../../../glossary.md#backend-intent--backendintent) propagation
- capability query contracts and [matrices](../../../glossary.md#term-matrix-meanings)
- [backend](../../../glossary.md#backend)-neutral [ownership](../../../glossary.md#term-ownership-meanings) scoring
- maximal same-[owner](../../../glossary.md#term-owner) partitioning
- logical [materialization](../../../glossary.md#materialization) and memory requirements

## Out of [scope](../../../glossary.md#term-scope)

- [fusion](../../../glossary.md#term-fusion) implementation
- concrete [kernel](../../../glossary.md#kernel) or route selection
- physical allocation
- [prepared execution](../../../glossary.md#prepared-execution--preparedexecution) and [runtime](../../../glossary.md#term-runtime) [residency](../../../glossary.md#residency)

## [Module](../../../glossary.md#term-module) invariants

- [Planning](../../../glossary.md#term-planning) answers where work runs, not which implementation runs it.
- Scoring uses [compile](../../../glossary.md#compile)-time information only.
- Scoring output is [backend ownership](../../../glossary.md#backend-ownership).

## Allowed dependencies

- [modules](../../../glossary.md#term-module)/[model](../../../glossary.md#term-model), with public visibility when [planning](../../../glossary.md#term-planning) [signatures](../../../glossary.md#term-signature) expose model types
- modules/config
- modules/backend-contract, with public visibility when planning signatures expose backend
  identity types<span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend); [backend identity glossary](../../../glossary.md#backend-identity--backendid))</span>
- modules/[trace](../../../glossary.md#trace)

## Forbidden dependencies

- [runtime](../../../glossary.md#term-runtime), [prepare](../../../glossary.md#prepare), [engine](../../../glossary.md#term-engine), and [concrete backend](../../../glossary.md#concrete-backend) [modules](../../../glossary.md#term-module)

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

The [module](../../../glossary.md#term-module) root is not a catch-all [facade](../../../glossary.md#term-facade). Task 0001 opens only `capability` with the [immutable](../../../glossary.md#term-immutable)
[operation](../../../glossary.md#operation)-[occurrence](../../../glossary.md#term-occurrence) question and inward-facing provider collaboration. Complete task 0002 keeps its
per-query eligibility result and [evaluation](../../../glossary.md#term-evaluation) package-private because no external planner consumer
yet justifies a public [matrix](../../../glossary.md#term-matrix-meanings)/evaluator or a public config [signature](../../../glossary.md#term-signature). Complete task 0003 keeps its
smallest consumer colocated with that internal value, uses the eligible [identity](../../../glossary.md#term-identity-meanings) list directly as
the candidate set, and adds no public [ownership](../../../glossary.md#term-ownership-meanings) package or facade. Complete task 0004 opens
[`partition`](../../../glossary.md#partition) with the architecture-named immutable recipe plus an internal generator over complete
per-[node](../../../glossary.md#node) ownership; it did not initially widen the current capability selector or add
orchestration. [Compiler](../../../glossary.md#term-compiler) 0005 now supplies the concrete cross-package consumer: it adds one
colocated public [owner](../../../glossary.md#term-owner)-selection collaboration and widens only the existing partition and memory
operations in their owning packages. Compiler retains [graph](../../../glossary.md#graph)-wide sequencing and owner-map
assembly.

## [Dependency](../../../glossary.md#term-dependency) [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) and [authorized frontier](../../../glossary.md#term-authorized-frontier)

Planning0001–0007 Complete + accepted Model0034/0035 Complete → Planning0008 Complete → Compiler0011
[AD](../../../glossary.md#autograd)/[capture](../../../glossary.md#term-capture-meanings) Complete → Prepare0010 projection/resource [handoff](../../../glossary.md#term-handoff) Complete → CPU0030 bounded [F32](../../../glossary.md#floating-point-precision-and-rounding) [pointwise](../../../glossary.md#term-pointwise) execution Complete.
Authorized [Planning](../../../glossary.md#term-planning) frontier: the neutral carrier/query source [slice](../../../glossary.md#slice) of
[Compiler0017 Ready](../compiler/tasks/0017-per-value-typed-length-scan-inference-and-ad.md),
under the accepted coordinated root/scoped interface and exact common source. This is one new
ScanActivityPlan in existing capability package and fifth query component, not AD [ownership](../../../glossary.md#term-ownership-meanings) or
a new dependency. Complete0008 retains its accepted historical body-query result.
[Compiler0011 AD/capture](../compiler/tasks/0011-general-scan-capture-and-first-order-ad.md) is Complete,
accepted by main 2026-10-22 after independent code/docs APPROVE and compatibility [checkpoint](../../../glossary.md#term-checkpoint).
[Prepare0010](../prepare/tasks/0010-general-scan-projection-and-resource-handoff.md) is Complete:
scoped projection/demand [validation](../../../glossary.md#term-validation) and resource handoff, main accepted 2026-10-22 after Huygens
final Class C code/docs APPROVE and the exact compatibility checkpoint.
At the old0008/CPU0030 checkpoint all listed frontiers were None; that historical bounded F32
pointwise result remains Complete. The current coordinated all-[PURE](../../../glossary.md#term-pure) continuation is solely
Compiler0017/CPU0081 [Ready](../../../glossary.md#term-ready), with independent source scopes and MAIN integration; no [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[persistence](../../../glossary.md#term-persistence)
or cost-[model](../../../glossary.md#term-model) successor follows.

Historical0008 common base: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted uncommitted
authority/Model0034/0035, Planning0008, Compiler0011, Prepare0010 and CPU0030 work. Parallel group None;
no other writer/conflict active. Main confirms empty index,
unchanged protected/Model snapshots at acceptance. Main owns shared documents and serial integration:
clean implementation → independent clean code/docs APPROVE → main's named downstream [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names)/
architecture/provider-[regression](../../../glossary.md#term-regression) checkpoint, all complete. The brief records all seven integration
fields and evidence; no root/native/performance campaign at this checkpoint.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Operation capability-query foundation](tasks/0001-operation-capability-query-foundation.md) | Complete | Completed [model](../../../glossary.md#term-model) milestone, [backend](../../../glossary.md#backend)-contract 0001–0004, config 0001–0002, and [trace](../../../glossary.md#trace) foundation | Added one [immutable](../../../glossary.md#term-immutable) [operation](../../../glossary.md#operation)-[occurrence](../../../glossary.md#term-occurrence) query and one [backend capability](../../../glossary.md#backend-capability)-provider contract without a [matrix](../../../glossary.md#term-matrix-meanings), eligibility [evaluation](../../../glossary.md#term-evaluation), diagnostics result, [device](../../../glossary.md#term-device) query, or provider implementation. |
| 0002 | [Per-query backend hard eligibility](tasks/0002-per-query-backend-hard-eligibility.md)<span data-glossary-navigation="true"> ([backend hard eligibility glossary](../../../glossary.md#backend-hard-eligibility))</span> | Complete | 0001, config hard intent, and backend-contract supplied availability | Combined validated provider/[snapshot](../../../glossary.md#term-snapshot) associations, backend-level support, current availability, and exact hard intent into an internal ordered [`BackendId`](../../../glossary.md#backend-identity--backendid) list without a public matrix, scoring, device selection, or [ownership](../../../glossary.md#term-ownership-meanings) choice. |
| 0003 | [Ownership candidates and baseline scoring](tasks/0003-ownership-candidates-and-baseline-scoring.md) | Complete | 0002, config [partition](../../../glossary.md#partition)-scoring configuration | Consumes the internal hard-eligible list directly as the candidate set and selects one exact `BackendId` through preferred-class-first, provider-order-stable baseline comparison, without a candidate record, cost classification, cost profile, or implementation-route interpretation. |
| 0004 | [Maximal same-owner partitioning](tasks/0004-maximal-same-owner-partitioning.md) | Complete | 0003 | Groups maximal consecutive runs in validated [topological](../../../glossary.md#term-topological-order) [node](../../../glossary.md#node) order by equal selected `BackendId`, producing [owner](../../../glossary.md#term-owner)-plus-node-ID recipes without [compiler](../../../glossary.md#term-compiler) orchestration, [graph](../../../glossary.md#graph)-boundary duplication, [lowering](../../../glossary.md#lowering), or executable construction. |
| 0005 | [Logical materialization and memory requirements](tasks/0005-logical-materialization-and-memory-requirements.md)<span data-glossary-navigation="true"> ([materialization glossary](../../../glossary.md#materialization))</span> | Complete | 0004 | Added immutable per-value [producer](../../../glossary.md#term-producer)/consumer/output requirements and an architecture-named [logical memory plan](../../../glossary.md#logical-memory-plan) derived internally from the closed graph and ordered partitions, without physical sizing, allocation, transfers, [publication](../../../glossary.md#term-publication) [binding](../../../glossary.md#term-binding), or [runtime](../../../glossary.md#term-runtime) [residency](../../../glossary.md#residency). |
| 0006 | [Planning contract closure audit](tasks/0006-planning-contract-closure-audit.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../glossary.md#term-planning))</span> | Complete | 0001–0005 | Audited capability, ownership, partition, logical-memory, public/internal surfaces, documentation, dependencies, and the compiler/[prepare](../../../glossary.md#prepare) [handoff](../../../glossary.md#term-handoff); the durable audit records a `CLOSED` verdict without executable changes. |
| 0007 | [Capability provider status reconciliation](tasks/0007-capability-provider-status-reconciliation.md) | Complete | 0006; current [CPU](../../../glossary.md#term-cpu) provider, Compiler [SPI](../../../glossary.md#term-service-provider-interface-spi), and [Engine](../../../glossary.md#term-engine) composition | Corrected and independently reviewed the public guide/[API](../../../glossary.md#term-application-programming-interface-api) status for the shipped CPU provider and current Engine [compile](../../../glossary.md#compile) consumer without changing capability or architecture. |
| 0008 | [General-scan body-bearing capability query](tasks/0008-general-scan-body-capability-query.md)<span data-glossary-navigation="true"> ([scan body glossary](../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span> | Complete | 0001–0007; accepted Model0034/0035 | Optional compiled-body transport and old constructor retained; final independent APPROVE and main compatibility gates accepted 2026-10-22; no scan execution. |

## Milestones

- Intent and capability
- [Ownership](../../../glossary.md#term-ownership-meanings) scoring
- Partitioning and logical memory

## Current status

Complete through0008. Current neutral activity/query frontier is the coupled Compiler0017 [Ready](../../../glossary.md#term-ready)
[slice](../../../glossary.md#slice) above; source/actual capability remains pending.
The baseline has the `CLOSED` verdict in the
[planning contract closure audit](planning-contract-closure-audit.md). Tasks 0001–0002 remain
Complete with the public query/provider boundary and package-private provider-ordered hard-
eligibility result.
Task 0003 adds the smallest internal [owner](../../../glossary.md#term-owner)-selection consumer: the eligible [`BackendId`](../../../glossary.md#backend-identity--backendid) list is
the candidate set, preferred matches precede nonmatches, provider order resolves ties and
fallback, and an empty list fails terminally before selection.

[Compiler](../../../glossary.md#term-compiler) 0005 is now the concrete consumer anticipated by the closure audit. Public
`BackendOwnerPlanning.selectOwner(...)` composes the unchanged package-private eligibility and
selector once per final [graph](../../../glossary.md#graph) [node](../../../glossary.md#node). Public `MaximalSameOwnerPartitioning.partition(...)` and
`LogicalMemoryPlanning.plan(...)` retain their completed semantics and are invoked directly after
Compiler assembles the complete owner map. [Planning](../../../glossary.md#term-planning) still owns no graph-wide workflow, compiler
[artifact](../../../glossary.md#term-artifact), provider [registry](../../../glossary.md#term-registry), numeric [cost model](../../../glossary.md#compiler-transformations-and-dominance), [device](../../../glossary.md#term-device)/route choice, physical memory, [prepare](../../../glossary.md#prepare), or
[runtime](../../../glossary.md#term-runtime) behavior.

Task 0004 now provides one public `PlannedPartition(owner, nodeIds)` recipe and public stateless
maximal consecutive-[run](../../../glossary.md#term-run) generation over a complete `Map<NodeId, BackendId>`. It preserves stored
[topological order](../../../glossary.md#term-topological-order), equality-based [ownership](../../../glossary.md#term-ownership-meanings), exact graph-node and first-owner references, and
[immutable](../../../glossary.md#term-immutable) results, including the empty result for a zero-node graph. It adds no ownership row,
public orchestration [facade](../../../glossary.md#term-facade), graph traversal or reordering, graph-boundary [DTO](../../../glossary.md#data-transfer-object--dto), phase split, cost
profile, [lowering](../../../glossary.md#lowering), or executable construction.

[Task 0005](tasks/0005-logical-materialization-and-memory-requirements.md) is Complete. It adds the
public `LogicalMemoryRequirement` and `LogicalMemoryPlan` recipes plus public stateless derivation
from [`CompiledGraphModel`](../../../glossary.md#compiled-graph--compiledgraphmodel) and ordered complete partitions. The generator validates complete graph-
order coverage and maximal owner runs before retaining exact [`ValueId`](../../../glossary.md#logical-valueid-versus-physical-memory-slot), `TensorDescriptor`,
[producer](../../../glossary.md#term-producer) [partition](../../../glossary.md#partition), distinct consumer partitions, and graph-output facts in graph-value order.
It adds no [eager](../../../glossary.md#term-eager-evaluation) byte count, [lifetime](../../../glossary.md#term-lifetime), transfer/copy, physical slot/allocation,
`PublicationBinding`, public orchestration, prepare/runtime/[backend](../../../glossary.md#backend) behavior, or cost
classification.

[Task 0006](tasks/0006-planning-contract-closure-audit.md) is Complete. Its historical clean
documentation-focused audit confirmed that the five then-public declarations were sufficient and
that the four evaluator/generator operations could remain internal until compiler orchestration
had a concrete consumer. Compiler 0005 supplies that consumer and opens only the three
package-cohesive operations it needs; eligibility and its selector remain internal. The audit's
semantic and [dependency](../../../glossary.md#term-dependency) verdict remains unchanged.

Config task 0004 remains Draft pending a concrete cost-bearing consumer. Documentation-only
Planning 0007 is Complete after independent review of the provider/[compile](../../../glossary.md#compile) status correction; it
creates no new Planning or [backend capability](../../../glossary.md#backend-capability). The glossary `Compile` entry and
`GraphCompilationPort` follow-up is already closed by Compiler 0006B10; do not schedule it again.

## Open questions

- The exact classification and units for a future planning cost profile remain deferred until a
  concrete cost-bearing consumer exists; they did not block the completed cost-free baseline.<span data-glossary-navigation="true"> ([planning cost profile glossary](../../../glossary.md#term-planning-cost-profile))</span>

## Decisions made

- The implementation must follow the current [architecture contract](../../../glossary.md#architecture-contract).
- Legacy code is capability evidence only; new implementation is written from scratch.
- The current capability contract asks whether one named backend can semantically own an
  immutable operation occurrence. It is distinct from availability, hard-eligibility evaluation,
  scoring, routing, preparation, and execution.<span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend); [immutable glossary](../../../glossary.md#term-immutable); [operation glossary](../../../glossary.md#operation); [occurrence glossary](../../../glossary.md#term-occurrence); [evaluation glossary](../../../glossary.md#term-evaluation))</span>
- Concrete backends may later implement the planning-owned inward-facing provider contract;
  planning does not depend on them and task 0001 added no implementation.
- Planning task 0001 intentionally interleaves before config task 0003 so the typed question is
  stable before a scoring preference is exposed. The frontier then returns to config 0003.
- Complete config task 0003 keeps hard eligibility separate from soft [ranking](../../../glossary.md#term-ranking) through one
  `Optional<DeviceClass>` preference. Planning will later own interpretation after its candidate
  and baseline scoring contracts are stable.
- Task 0002 uses provider encounter order as the deterministic backend order and associates
  provider/snapshot inputs by equal <span data-glossary-navigation="true"> ([snapshot glossary](../../../glossary.md#term-snapshot))</span>[`BackendId`](../../../glossary.md#backend-identity--backendid), never by parallel list position. Duplicate or
  missing identities are invalid supplied composition and fail before any capability call.<span data-glossary-navigation="true"> ([identities glossary](../../../glossary.md#term-identity-meanings))</span>
- Task 0002 keeps its exact evaluator/result package-private and retains only eligible
  provider-owned `BackendId` references. Exact-device/class requirements use the matching snapshot
  only to prove current matching availability; they neither claim device-level capability nor
  choose or retain a device.<span data-glossary-navigation="true"> ([device glossary](../../../glossary.md#term-device))</span>
- A valid no-match task-0002 result is an empty immutable list. Later planning orchestration must
  stop before scoring it and may not weaken a hard requirement; the public compile failure
  contract remains with that later consumer.<span data-glossary-navigation="true"> ([compile glossary](../../../glossary.md#compile))</span>
- Planning cost is separate from model autotuning. Planning may consume only backend-
  neutral estimates; it must not interpret routes, vector species/lanes, unroll, threads, chunks,
  tiles, kernels, workload-cache values, or other backend parameter vocabulary.<span data-glossary-navigation="true"> ([model autotuning glossary](../../../glossary.md#autotuning--model-autotuning); [vector species glossary](../../../glossary.md#term-vector-species); [lanes glossary](../../../glossary.md#term-lane); [kernels glossary](../../../glossary.md#kernel); [cache glossary](../../../glossary.md#term-cache); [parameter glossary](../../../glossary.md#term-parameter))</span>
- Planning owns the semantics and generation of complete valid ownership/partition candidates.
  Future tuning orchestration may measure a bounded set opaquely but does not construct or
  reinterpret those candidates.<span data-glossary-navigation="true"> ([partition glossary](../../../glossary.md#partition))</span>
- Completed Planning 0003 establishes the cost-free baseline consumer without defining a cost
  classification. Config 0004 may later store only the immutable backend-neutral inputs that a
  later concrete cost-bearing consumer actually requires.
- No stable shared production `OperationFamily` or workload-bucket contract exists. Planning 0003
  must not invent one merely to keep the implementation queue moving.
- Task 0003 uses the existing package-private `BackendEligibility` directly through one
  package-private stateless selector in `planning.capability`. Compiler 0005 leaves both internal
  and adds one public colocated collaboration that composes them for a single query.<span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler))</span>
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
- [Planning](../../../glossary.md#term-planning) task 0004 uses `CompiledGraphModel.nodes()` as its exact deterministic adjacency and
  order contract. “Adjacent” means consecutive positions in that validated topological list, not
  graph-edge connectivity, numeric node-ID order, map order, graph phase, or operation family.<span data-glossary-navigation="true"> ([topological glossary](../../../glossary.md#term-topological-order); [graph glossary](../../../glossary.md#graph); [graph phase glossary](../../../glossary.md#graph-phase))</span>
- The complete per-[node](../../../glossary.md#node) [ownership](../../../glossary.md#term-ownership-meanings) [handoff](../../../glossary.md#term-handoff) is `Map<NodeId, BackendId>`. It associates by typed
  equality without introducing a production ownership-row type or relying on parallel list
  position. Task 0004 does not invoke or widen the package-private task-0003 selector.
- `PlannedPartition` contains only one exact `BackendId` owner reference and an immutable ordered
  non-empty <span data-glossary-navigation="true"> ([owner glossary](../../../glossary.md#term-owner))</span>[`NodeId`](../../../glossary.md#nodeid) list. The generator retains graph node references and the first node's owner
  reference while comparing identities by equality.
- Graph inputs and outputs remain values, a multi-output producer remains one indivisible node,
  and fan-out, merge, publication, or phase changes do not split an equal-owner consecutive run.
  Completed task 0005 derives boundary and logical-materialization facts from the graph plus
  partition node IDs; transfers and physical materialization remain with later lifecycle owners.<span data-glossary-navigation="true"> ([producer glossary](../../../glossary.md#term-producer); [fan-out glossary](../../../glossary.md#term-fan-out); [publication glossary](../../../glossary.md#term-publication); [run glossary](../../../glossary.md#term-run); [materialization glossary](../../../glossary.md#materialization); [lifecycle glossary](../../../glossary.md#lifecycle))</span>
- The partition recipe and stateless generation operation are public for cross-package compiler
  consumption. Compiler, not Planning, assembles the complete owner map and controls graph-wide
  orchestration.
- Planning task 0005 consumes only [`CompiledGraphModel`](../../../glossary.md#compiled-graph--compiledgraphmodel) plus an ordered `List<PlannedPartition>`.
  It validates complete graph-order coverage and adjacent-owner maximality before deriving any
  value requirement because the public partition DTO can also be constructed directly.<span data-glossary-navigation="true"> ([DTO glossary](../../../glossary.md#data-transfer-object--dto))</span>
- Task 0005 represents logical materialization through each value's exact producing partition,
  distinct consuming partitions, and graph-output flag. These primitive facts express partition
  inputs/outputs, cross-owner boundaries, publication preservation, and partition-internal values
  without a closed role enum or duplicated boundary lists on `PlannedPartition`.
- `TensorDescriptor` remains the logical size carrier. Task 0005 does not calculate element or
  byte counts because dynamic/expression dimensions can remain unresolved and physical backend
  representation, alignment, and padding belong after compile-time planning.<span data-glossary-navigation="true"> ([element glossary](../../../glossary.md#term-element); [dimensions glossary](../../../glossary.md#dimension); [alignment glossary](../../../glossary.md#term-alignment); [padding glossary](../../../glossary.md#padding))</span>
- `PublicationBinding` remains outside task 0005. It is standalone model data for a future
  compiler-owned <span data-glossary-navigation="true"> ([model glossary](../../../glossary.md#term-model))</span>`PublicationPlan`; `CompiledGraphModel.outputs()` supplies only the logical
  preservation obligation at this frontier.

## Risks

- Leaking [backend route](../../../glossary.md#backend-route) selection or [runtime](../../../glossary.md#term-runtime) [residency](../../../glossary.md#residency) into [compile](../../../glossary.md#compile)-time scoring.
- Reintroducing a backend-wide average that hides materially different operation families, data
  types, shapes, and sizes.<span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend); [operation glossary](../../../glossary.md#operation); [shapes glossary](../../../glossary.md#shape))</span>
- Growing eligibility into a public matrix, graph identity, scoring, device selection, or
  rejection diagnostics before those consumers are concrete.<span data-glossary-navigation="true"> ([matrix glossary](../../../glossary.md#term-matrix-meanings); [graph glossary](../../../glossary.md#graph); [identity glossary](../../../glossary.md#term-identity-meanings); [device glossary](../../../glossary.md#term-device))</span>

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).

Task 0001 adds only the query/provider contracts and package documentation, the two required
public [dependency](../../../glossary.md#term-dependency)-visibility changes, focused [planning](../../../glossary.md#term-planning) and architecture tests, and synchronized
explanatory/planning documentation. Its implementation context passed 13 tests across four suites
with no failures, errors, or skips. The independent documentation context reused that evidence,
changed no executable behavior, and finalized the affected Javadocs, examples, terminology,
current-versus-planned boundaries, glossary entries, and planning records. The single final root
suite passed 1,079 tests across 141 suites with no skips, failures, or errors. A separate planning
step then made config task 0003 [Ready](../../../glossary.md#term-ready); it is now Complete.

Config task 0003 is now Complete after its focused config [validation](../../../glossary.md#term-validation) and independent
documentation pass. It changed no planning Java, provider contract, [module](../../../glossary.md#term-module) dependency, or scoring
behavior. A separate reassessment selected only Planning task 0002 as Ready because hard
eligibility needs no performance profile or scoring formula. Planning task 0003 and Config task
0004 remained Draft; after 0002, Config 0004 profile contracts were the likely next area before
Planning 0003 scoring, subject to another reassessment. Planning task 0002 is now Complete after
its focused implementation, final planning suite, independent documentation pass, and focused
documentation validation. A following reassessment drafted Config task 0004, but the terminology/
[ownership](../../../glossary.md#term-ownership-meanings) reset retired that unimplemented design. Planning 0003 is now the prerequisite for a
future Config 0004 cost profile. Planning 0003 is now Complete with an exact cost-free consumer,
deterministic comparison, tie, and failure contract; Config 0004 remains Draft until a later
concrete cost-bearing consumer establishes classification and units. No next task becomes Ready
without a separate frontier reassessment.

Planning task 0003 completed in implementation context `/root/implement_planning_0003` and clean
documentation context `/root/implement_planning_0003/docs_planning_0003`. Its focused selector
suite passed 13 tests and the final planning suite passed 38 tests across three suites, all with
zero failures, errors, or skips. Planning [Javadoc](../../../glossary.md#term-javadoc) passed, generated public pages preserve the
public/internal boundary and omit the package-private selector from public indexes, Markdown
validation passed for 227 files, 4,014 links, 246 anchors, and 2,844 fence markers, and final
[scope](../../../glossary.md#term-scope), status, source-surface, newline, trailing-whitespace, and `git diff --check` checks passed.
The completed change is exactly its authorized fourteen paths.

Planning task 0004 completed in implementation context `/root/implement_planning_0004` and clean
documentation context `/root/implement_planning_0004/docs_planning_0004`. Its two focused suites
passed 11 tests, and the final planning suite passed 49 tests across five suites with no failures,
errors, or skips. The documentation pass changed no executable behavior and reused that evidence.
It finalized the [partition](../../../glossary.md#partition) Javadocs, public/internal status explanations, glossary, task, master
plans, and roadmap; planning Javadoc, repository Markdown, generated-page, exact fifteen-path,
status, later-spec, dependency, forbidden-surface, newline, whitespace, and `git diff --check`
validation passed. A separate frontier reassessment then made only Planning 0005 Ready with its
detailed logical [materialization](../../../glossary.md#materialization)/memory specification. Planning 0006, Config 0004+, [Trace](../../../glossary.md#trace) 0003+,
and [Compiler](../../../glossary.md#term-compiler) work remain Draft, and no other detailed future specification was created.

Planning task 0005 completed in implementation context `/root/implement_planning_0005` and clean
documentation context `/root/implement_planning_0005/planning_0005_docs`. Its three focused suites
passed 14 tests and the final planning suite passed 63 tests across eight suites, all with zero
failures, errors, or skips. The documentation pass changed no executable behavior and reused that
evidence. It finalized the logical-memory Javadocs, architecture/[API](../../../glossary.md#term-application-programming-interface-api)/guide/glossary explanations,
current-versus-planned boundaries, task evidence, master plans, and roadmap. Planning Javadoc,
repository Markdown, generated public/internal page, exact eighteen-path, status, later-spec,
dependency, forbidden-surface, newline, whitespace, and `git diff --check` validation passed.
At that completion point, Planning 0006, Config 0004+, Trace 0003+, and Compiler work remained
Draft without another detailed specification; no global task was Ready pending reassessment.

A separate frontier reassessment in planning context `/root/plan_planning_0006` found no missing
planning semantic or architecture prerequisite after task 0005. It created only the detailed
documentation-only Planning 0006 closure-audit specification and made that row Ready. The audit
must verify the public/internal boundary and future compiler/[prepare](../../../glossary.md#prepare) [handoff](../../../glossary.md#term-handoff) before the selected
planning milestone can become Complete; every later task remains Draft without a detailed spec.
