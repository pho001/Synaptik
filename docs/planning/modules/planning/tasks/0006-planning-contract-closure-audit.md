# Task 0006: [Planning](../../../../glossary.md#term-planning) Contract Closure Audit

> **Editorial historical reading key.**
>
> This is the historical exit gate for the selected Planning milestone, not a claim that downstream
> execution layers remain absent today. Its five-public-declaration and four-internal-operation
> conclusions belong to that checkpoint. The participants are backend capability providers answering
> exact occurrence questions, Planning applying ownership and logical-recipe rules, and a then-future
> Compiler assembling graph-wide results. Descriptors, backend identities, node IDs and logical value
> relationships cross these boundaries—not allocated buffers or live backend services.
>
> For the present consumer boundary, see [Compiler artifact
> integration](../../../../reference/compiler/artifacts.md#current-complete-artifact-compilation-and-integration-port);
> for current selection and partition contracts, see [Planning
> contracts](../../../../reference/compiler/planning.md#current-operation-capability-contracts).
> Planning supplies semantic rules, Compiler orchestrates their graph-wide application, and later
> resource owners—not these immutable recipes—carry cleanup responsibility. The recorded audit checks
> remain historical evidence.

## Status

Complete

## Goal

[Run](../../../../glossary.md#term-run) the final [planning](../../../../glossary.md#term-planning)-only exit gate for the selected `modules/planning` milestone. Audit the
implemented capability question/provider collaboration, hard eligibility, cost-free baseline
[owner](../../../../glossary.md#term-owner) selection, maximal same-owner partitioning, logical [materialization](../../../../glossary.md#materialization) and memory recipes,
package/[API](../../../../glossary.md#term-application-programming-interface-api) boundaries, dependencies, documentation, and downstream [lifecycle](../../../../glossary.md#lifecycle) [handoff](../../../../glossary.md#term-handoff) against
current source and tests.

The task produces one durable audit [artifact](../../../../glossary.md#term-artifact) with a falsifiable closure verdict. It changes no
Java or executable behavior. A successful `CLOSED` verdict means the selected planning contracts
are coherent and sufficiently stable for later [compiler](../../../../glossary.md#term-compiler)-owned orchestration and
`CompileArtifacts`; it does not claim that a compiler, public planning workflow, [prepare](../../../../glossary.md#prepare) layer,
[runtime](../../../../glossary.md#term-runtime), [concrete backend](../../../../glossary.md#concrete-backend), cost-bearing scoring policy, or [model](../../../../glossary.md#term-model)-[autotuning](../../../../glossary.md#autotuning--model-autotuning) workflow exists.

[Mental model](../../../../glossary.md#term-mental-model):

```text
authoritative architecture
  + completed Planning 0001-0005 source, tests, and Javadocs
  + current config/backend-contract/model inputs
  + public/internal package and dependency boundaries
  + API, guide, glossary, task, master-plan, and roadmap consistency
    -> planning contract closure verdict
       CLOSED
       BLOCKING_GAP
       ARCHITECTURE_DECISION_REQUIRED
```

## [Scope](../../../../glossary.md#term-scope)

- Create `docs/planning/modules/planning/planning-contract-closure-audit.md` as the sole durable
  detailed result artifact.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Inventory every current production and test file under `modules/planning`, every generated
  public Javadoc page, and every package-private planning type that must remain absent from the
  generated public indexes.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Audit the exact task-0001 capability boundary:
  - [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) retains one [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence) and ordered [descriptor](../../../../glossary.md#term-descriptor) snapshots;
  - [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider) represents one stable backend identity and a deterministic
    backend-level semantic support answer; and<span data-glossary-navigation="true"> ([backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>
  - neither contract performs discovery, availability evaluation, hard matching, scoring,
    routing, preparation, or execution.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Audit task-0002 hard eligibility against the complete provider/snapshot association, exact
  backend-identity matching, availability and hard-requirement filtering, provider call order,
  immutable provider-order result, no-match behavior, and device non-selection rules.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [identity glossary](../../../../glossary.md#term-identity-meanings); [device glossary](../../../../glossary.md#term-device))</span>
- Audit task-0003 baseline owner selection against its complete hard-eligible candidate set,
  preferred-class-first/provider-order comparison, exact reference retention, terminal empty
  eligibility, and absence of numeric scoring, cost/workload classifications, routes, or devices.
- Audit task-0004 partitioning against complete owner-map validation, stored topological node
  order, equality-defined ownership transitions, maximal consecutive runs, exact graph-node and
  first-owner references, multi-output indivisibility, and zero-node behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [topological glossary](../../../../glossary.md#term-topological-order); [node glossary](../../../../glossary.md#node); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Audit task-0005 logical-memory derivation against complete ordered partition coverage,
  adjacent-owner maximality, graph-value order, exact value/descriptor/partition references,
  producer and distinct consumer relationships, graph-output preservation, dynamic/expression
  Shape support, and zero-node pass-through behavior.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [Shape glossary](../../../../glossary.md#shape))</span>
- Record the exact public planning surface. Decide whether each public declaration is necessary,
  sufficient, and minimal for current cross-package recipe or provider use. Explicitly check that
  no public capability matrix, eligibility result, owner-selection result, owner map, planning
  facade, device choice, cost profile, transfer plan, or physical-memory DTO is currently needed.<span data-glossary-navigation="true"> ([capability matrix glossary](../../../../glossary.md#review-tables-and-support-matrices))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix); [facade glossary](../../../../glossary.md#term-facade); [DTO glossary](../../../../glossary.md#data-transfer-object--dto))</span>
- Record every package-private planning declaration and operation. Make an explicit evidence-
  backed decision whether `BackendEligibility.evaluate`, `BackendOwnerSelection.select`,
  `MaximalSameOwnerPartitioning.partition`, and `LogicalMemoryPlanning.plan` may remain internal
  until a concrete compiler-owned orchestrator establishes the narrow invocation boundary.
- Audit every exact validation, ordering, equality, identity, and reference-retention rule relied
  on across tasks 0001-0005. Distinguish value equality used for association from exact reference
  retention promised by a result.
- Audit the package map and exact planning Gradle dependencies/visibility. Confirm that model and
  backend-contract are public dependencies only where public signatures expose their types, config
  and trace remain internal dependencies, and no runtime, prepare, engine, or concrete-backend
  dependency exists.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [model glossary](../../../../glossary.md#term-model); [signatures glossary](../../../../glossary.md#term-signature); [trace glossary](../../../../glossary.md#trace); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Review generated Javadocs, explanatory architecture/status documentation, public and compile API
  pages, provider and backend-selection examples, memory-planning design note, glossary, completed
  task history, planning/config/backend-contract/trace/compiler/prepare/runtime/engine master
  plans, and the roadmap for accurate current-versus-planned wording.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Audit the downstream [handoff](../../../../glossary.md#term-handoff) without designing it:
  - [compiler](../../../../glossary.md#term-compiler) owns [graph](../../../../glossary.md#graph)-wide orchestration, [owner](../../../../glossary.md#term-owner)-map assembly, [publication](../../../../glossary.md#term-publication) [planning](../../../../glossary.md#term-planning),
    `CompileArtifacts`, public [compile](../../../../glossary.md#compile) failures, and diagnostics;
  - planning owns the current [backend](../../../../glossary.md#backend)-neutral rules and [immutable](../../../../glossary.md#term-immutable) [partition](../../../../glossary.md#partition)/logical-memory output;
  - a later concrete compiler consumer may justify one narrow public planning collaboration while
    the current operations remain internal now;
  - prepare consumes immutable compile artifacts and owns the shared transition to prepared state;
    and<span data-glossary-navigation="true"> ([compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
  - physical memory, transfers, lowering, routes, kernels, residency, and execution remain with
    prepare, concrete backends, and runtime according to the architecture contract.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [residency glossary](../../../../glossary.md#residency); [architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- Classify every discovered gap as blocking, non-blocking deferred integration, documentation
  drift correctable within this task, confirmed no-change, or an architecture decision.
- Run one final combined planning-milestone checkpoint after the artifact and documentation are
  stable, then synchronize task, planning master-plan, and roadmap status with the verdict.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- If the verdict is `CLOSED`, mark task 0006, the selected planning milestone, and the roadmap
  planning area `Complete`. Record the future compiler/prepare handoff, but create no next detailed
  task specification and do not make a later task Ready.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- If the verdict is `BLOCKING_GAP`, the audit task may become `Complete`, but keep the planning
  milestone and roadmap area open. Add at most one concise Draft master-plan row for the first
  cohesive blocking frontier; do not create its detailed specification.
- If the verdict is `ARCHITECTURE_DECISION_REQUIRED`, stop without changing architecture. Mark
  task 0006 `Blocked`, keep the planning milestone open, and use the canonical incomplete
  completion status.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- creating, deleting, renaming, moving, or modifying Java production or test files
- modifying Gradle, project dependencies, module structure, generated Javadocs, architecture
  tests, backend-conformance tests, integration tests, or any executable artifact<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- changing `ARCHITECTURE.md`, an [ADR](../../../../glossary.md#term-architecture-decision-record-adr), or an architecture rule or [dependency direction](../../../../glossary.md#java-platform-and-build-tools)
- implementing, repairing, widening, or making public any capability, eligibility, owner-
  selection, partitioning, or logical-memory operation found during the audit<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [owner glossary](../../../../glossary.md#term-owner); [operation glossary](../../../../glossary.md#operation))</span>
- adding a public planning facade, pipeline, matrix, evaluator, owner map, result row, compiler
  adapter, provider registry, discovery mechanism, or service locator<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [facade glossary](../../../../glossary.md#term-facade); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- implementing [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings) or orchestration, `CompileArtifacts`, `PublicationPlan`, public
  compile failures, compile diagnostics, graph transformations, autograd, or publication behavior<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [compile diagnostics glossary](../../../../glossary.md#compile-diagnostics); [graph glossary](../../../../glossary.md#graph); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication))</span>
- defining cost/workload families, buckets, units, estimates, numeric scoring, planning-cost
  profiles, Config 0004, tuning candidates, tuning caches, or model-autotuning inputs<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- adding [trace](../../../../glossary.md#trace) attributes, [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) [schemas](../../../../glossary.md#term-schema), emission, correlation domains, or [serialization](../../../../glossary.md#term-serialization)
- implementing prepare, runtime, engine, concrete backend, lowering, fusion, specialization,
  transfer, physical memory, route, kernel, schedule, residency, or execution behavior<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [specialization glossary](../../../../glossary.md#term-specialization); [kernel glossary](../../../../glossary.md#kernel); [residency glossary](../../../../glossary.md#residency))</span>
- selecting a device, retaining a device identity in an ownership result, or interpreting backend-
  specific implementation parameters<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- editing completed task specifications except this task's own execution evidence and final status
- creating a detailed compiler, config, trace, planning follow-up, prepare, runtime, engine,
  backend, or other future task specification
- unrelated documentation cleanup or turning the result artifact into a tutorial or speculative
  future API design<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Partition scoring](../../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition))</span>
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Tracing](../../../../architecture/tracing.md)
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [ADR 0001](../../../../design/decisions/0001-layered-architecture.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0002](../../../../design/decisions/0002-backend-owned-lowering.md)
- [ADR 0004](../../../../design/decisions/0004-partition-scoring.md)
- [ADR 0008](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [Memory planning strategy](../../../../design/notes/memory-planning-strategy.md)<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Architecture style](../../../../developer-guide/documentation/architecture-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Planning master plan](../master-plan.md)
- [Completed Planning 0001](0001-operation-capability-query-foundation.md)
- [Completed Planning 0002](0002-per-query-backend-hard-eligibility.md)
- [Completed Planning 0003](0003-ownership-candidates-and-baseline-scoring.md)
- [Completed Planning 0004](0004-maximal-same-owner-partitioning.md)
- [Completed Planning 0005](0005-logical-materialization-and-memory-requirements.md)
- [Public API status](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative. This audit may confirm or report the existing boundary but
  cannot create, reinterpret, or change an architecture rule.
- Planning owns backend-neutral capability analysis, backend ownership, maximal same-owner
  partitioning, and logical materialization/memory requirements. It answers where work runs, not
  which backend implementation runs it.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [backend ownership glossary](../../../../glossary.md#backend-ownership); [owner glossary](../../../../glossary.md#term-owner); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Compiler](../../../../glossary.md#term-compiler) owns [graph](../../../../glossary.md#graph)-wide [planning](../../../../glossary.md#term-planning) orchestration and [immutable](../../../../glossary.md#term-immutable) `CompileArtifacts`. The absence of
  a current compiler-owned orchestrator does not by itself require planning to expose isolated
  public evaluator methods before their consumer exists.
- Prepare and concrete backends own lowering, implementation routes, executable construction, and
  physical realization. Runtime owns prepared execution and per-run residency/state.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [run glossary](../../../../glossary.md#term-run); [residency glossary](../../../../glossary.md#residency))</span>
- Current source and tests are primary implementation evidence. Completed tasks and explanatory
  documents are supporting evidence, not authority over source or architecture.
- Public planning DTOs may expose model and backend-contract types through existing public
  dependencies. Planning must not expose or retain live backend services in compile-time plans.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [compile glossary](../../../../glossary.md#compile))</span>
- A contradiction requiring Java, test, Gradle, architecture, or cross-layer behavior changes
  changes the finding/verdict; it does not expand this task's scope.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [scope glossary](../../../../glossary.md#term-scope))</span>

## Package impact

No Java package is added, changed, or moved.

The audit reviews these existing [planning](../../../../glossary.md#term-planning) packages:

- `io.github.pho001.synaptik.planning.capability` — public occurrence/provider contracts and
  internal hard-eligibility/baseline-owner operations;<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [owner glossary](../../../../glossary.md#term-owner))</span>
- `io.github.pho001.synaptik.planning.partition` — public immutable partition recipe and internal
  maximal consecutive same-owner generation; and<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [partition glossary](../../../../glossary.md#partition))</span>
- `io.github.pho001.synaptik.planning.memory` — public logical value requirements/plan and internal
  closed-graph derivation.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>

The audit must not create a root [facade](../../../../glossary.md#term-facade), [`ownership`](../../../../glossary.md#term-ownership-meanings) implementation package, `util`, `common`,
[compiler](../../../../glossary.md#term-compiler) adapter, public orchestration package, or another Java surface.

## Affected files

Always created or updated — exactly six paths:

- add `docs/planning/modules/planning/planning-contract-closure-audit.md`
- add and then finalize this task
- update `docs/planning/modules/planning/master-plan.md`
- update `docs/planning/modules/config/master-plan.md` only to synchronize the planning-closure
  effect on its current-status narrative; Config 0004 and later rows remain Draft
- update `docs/planning/modules/trace/master-plan.md` only to synchronize the planning-closure
  effect on its current-status narrative; Trace 0003 and later rows remain Draft
- update `docs/planning/roadmap.md`

Conditionally permitted only for documentation drift proved against current behavior — at most
seven paths:

- `docs/architecture/partition-scoring.md`
- `docs/design/notes/memory-planning-strategy.md`
- `docs/api/public-api.md`
- `docs/api/compile-api.md`
- `docs/backend-guide/capability-provider.md`
- `docs/user-guide/backend-selection.md`
- `docs/glossary.md`

Review without modification: `AGENTS.md`; `ARCHITECTURE.md`; all ADRs and other focused
architecture documents; every [planning](../../../../glossary.md#term-planning) source/test file and generated [Javadoc](../../../../glossary.md#term-javadoc) page; planning
[Gradle](../../../../glossary.md#term-gradle); [model](../../../../glossary.md#term-model) [graph](../../../../glossary.md#graph)/[publication](../../../../glossary.md#term-publication) contracts; config/[backend](../../../../glossary.md#backend)-contract/[trace](../../../../glossary.md#trace) Java and tests; completed
Planning 0001-0005 history; config/backend-contract/trace/[compiler](../../../../glossary.md#term-compiler)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[engine](../../../../glossary.md#term-engine) master
plans; architecture, [conformance](../../../../glossary.md#term-conformance), and integration tests; concrete backends; all other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

This documentation-only audit may create or modify at most the thirteen paths listed above. The six
always-affected paths are required. A conditional path may change only to correct status,
terminology, link, example, or boundary wording already established by current source/tests and
the [architecture contract](../../../../glossary.md#architecture-contract).

If a finding needs Java/[Javadoc](../../../../glossary.md#term-javadoc) source, tests, [Gradle](../../../../glossary.md#term-gradle), an architecture contract/[ADR](../../../../glossary.md#term-architecture-decision-record-adr), another
[planning](../../../../glossary.md#term-planning) file, a fourteenth path, or new behavior, do not expand scope. Record `BLOCKING_GAP` or
`ARCHITECTURE_DECISION_REQUIRED` with the exact [owner](../../../../glossary.md#term-owner) and required follow-up.

## Required audit [artifact](../../../../glossary.md#term-artifact)

The durable artifact must contain exactly these top-level sections, with focused subsections and
tables as needed:

1. `Executive conclusion and closure verdict`
2. `Authority, scope, and method`
3. `Architecture and lifecycle boundary assessment`
4. `Production, test, and generated-Javadoc inventory`
5. `Capability query and provider assessment`
6. `Hard-eligibility assessment`
7. `Owner-selection and scoring assessment`
8. `Partition recipe and generation assessment`
9. `Logical-memory recipe and derivation assessment`
10. `Public DTO and package-private operation assessment`
11. `Validation, equality, ordering, identity, and reference matrix`
12. `Package and dependency assessment`
13. `Documentation, examples, glossary, and planning-history consistency`
14. `Compiler, CompileArtifacts, publication, and prepare handoff`
15. `Deferred and downstream-owned work`
16. `Findings, severity, and disposition`
17. `Checkpoint evidence and planning-milestone decision`

### Closure verdict vocabulary

The first section records exactly one verdict:

- `CLOSED` — no blocking planning contract, documentation, dependency, or architecture conflict
  remains. Explicit downstream integration may remain deferred when its owner and non-blocking
  rationale are recorded.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [dependency glossary](../../../../glossary.md#term-dependency); [owner glossary](../../../../glossary.md#term-owner))</span>
- `BLOCKING_GAP` — the audit is complete, but at least one planning-owned gap prevents milestone
  closure. Every gap names evidence, impact, owner, dependency, and the first bounded follow-up.
- `ARCHITECTURE_DECISION_REQUIRED` — evidence conflicts with the authoritative architecture or
  requires a new ownership/dependency decision. Stop without editing architecture.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

Findings use exactly these labels:

- `BLOCKING`
- `NON_BLOCKING_DEFERRED`
- `DOCUMENTATION_DRIFT`
- `NO_CHANGE_CONFIRMED`

An architecture conflict is recorded under the global `ARCHITECTURE_DECISION_REQUIRED` verdict,
not disguised as ordinary documentation drift.

### Required evidence [matrix](../../../../glossary.md#term-review-matrix)

The [artifact](../../../../glossary.md#term-artifact) contains one [matrix](../../../../glossary.md#term-matrix-meanings) with exactly these columns:

| Area | Current contract | Primary evidence | Required [invariant](../../../../glossary.md#term-invariant) or question | Finding label | Disposition |
|---|---|---|---|---|---|

It contains at least one row for each of these areas:

- capability query/provider;
- hard eligibility;
- baseline [owner](../../../../glossary.md#term-owner) selection;
- maximal same-owner partitioning;
- logical-memory derivation;
- public [DTO](../../../../glossary.md#data-transfer-object--dto) minimality;
- package-private [operation](../../../../glossary.md#operation) visibility;
- [validation](../../../../glossary.md#term-validation) and failure ordering;
- equality-based association and exact-reference retention;
- package/[dependency](../../../../glossary.md#term-dependency) surface;
- generated Javadocs and public/internal index boundary;
- [API](../../../../glossary.md#term-application-programming-interface-api)/guide examples and glossary terminology;
- completed task/master-plan/roadmap history;
- [compiler](../../../../glossary.md#term-compiler) orchestration and `CompileArtifacts` [handoff](../../../../glossary.md#term-handoff);
- [publication](../../../../glossary.md#term-publication)-plan boundary; and
- [prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) physical-realization boundary.

Every row names source/test/API evidence or an exact inspection command. Completed task prose
alone is not sufficient evidence. The matrix must explicitly answer both questions:

1. May the four current operations remain package-private until the compiler-owned orchestration
   consumer establishes a narrow collaboration boundary?
2. Are the five current public planning declarations sufficient and minimal for the selected
   milestone and downstream immutable recipe handoff?<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [immutable glossary](../../../../glossary.md#term-immutable))</span>

## Acceptance criteria

- The durable artifact contains every required top-level section, exactly one closure verdict,
  only the permitted finding labels, and the complete required evidence matrix.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span>
- Every current planning production/test file and generated public/internal Javadoc boundary is
  included in a recorded inventory method; no declaration or test suite is omitted.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The capability provider/query, hard eligibility, owner selection, partitioning, and logical-
  memory contracts are compared with their current source/tests rather than inferred only from
  completed task status.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Exact validation and failure ordering, immutable snapshots, equality rules, encounter/graph/
  partition ordering, and exact reference-retention promises are recorded for every current stage.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [partition glossary](../../../../glossary.md#partition))</span>
- The audit explicitly concludes whether the current four package-private operations may remain
  internal until a compiler-owned orchestrator exists. A `CLOSED` verdict must explain why that
  deferral does not leave a missing planning semantic contract.
- The audit explicitly concludes whether the five public declarations are sufficient and minimal.
  A `CLOSED` verdict must reject premature public matrices, owner rows/maps, orchestration,
  physical memory, cost/workload profiles, devices, routes, and executable state.
- Package placement and the exact Gradle dependency/visibility surface match the current master
  plan and architecture; no forbidden dependency or concrete-backend coupling exists.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend))</span>
- Generated Javadocs accurately render the five public declarations and package summaries while
  omitting `BackendEligibility`, `BackendOwnerSelection`, `MaximalSameOwnerPartitioning`, and
  `LogicalMemoryPlanning` from public class indexes/pages.
- Explanatory/API/guide/glossary examples and current-versus-planned wording match current source.
  Any permitted documentation drift is corrected and recorded; Java-Javadoc drift is blocking
  because Java edits are forbidden.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The [artifact](../../../../glossary.md#term-artifact) distinguishes [planning](../../../../glossary.md#term-planning)-owned semantics from future [compiler](../../../../glossary.md#term-compiler) orchestration,
  `CompileArtifacts`, compiler-owned `PublicationPlan`, public failure translation, prepare,
  physical memory, backend lowering, runtime residency, and execution.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency))</span>
- Cost-bearing scoring, cost/workload classifications, Config 0004, trace payloads, and tuning
  remain non-blocking Draft/downstream work unless current evidence proves a planning-owned gap.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>
- Completed Planning 0001-0005 history remains unchanged. No later detailed task specification is
  created, and no later task is made Ready.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- The final combined repository/planning checkpoint, Markdown validation, exact path/status/spec-
  directory checks, final newlines, trailing whitespace, and <span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`git diff --check` pass.
- No Java, test, [Gradle](../../../../glossary.md#term-gradle), `ARCHITECTURE.md`, ADR, architecture-test, conformance, integration,
  compiler, prepare, runtime, backend, or other executable path changes.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Verdict/status synchronization follows the exact rules in Scope and is reflected consistently
  in this task, the planning master plan, and the roadmap.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [Scope glossary](../../../../glossary.md#term-scope))</span>
- The task records local decisions, limitations, exact commands/results, audit context identity,
  inventory method, findings, milestone disposition, and canonical completion summary.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

This task is a clean-context documentation-focused audit and changes no executable Java. Inspect
source, tests, generated Javadocs, and documentation before the final [checkpoint](../../../../glossary.md#term-checkpoint). Then [run](../../../../glossary.md#term-run) exactly
one combined [Gradle](../../../../glossary.md#term-gradle) checkpoint:

```bash
./gradlew test :testing:architecture-tests:test :modules:planning:javadoc
```

Gradle may deduplicate tasks already selected by root `test`. Record executed/up-to-date/skipped
task counts and [planning](../../../../glossary.md#term-planning)/architecture/repository test counts from XML reports where available. Do
not rerun a successful Java suite merely to reformat evidence.

Run final documentation and working-tree validation:

```bash
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
rg --files docs/planning/modules/planning/tasks | sort
rg -n '^Ready$|\| Ready \|' docs/planning
```

Also inspect the generated planning package summaries, five public type pages,
`allclasses-index.html`, and `overview-tree.html`. Confirm the four package-private [operation](../../../../glossary.md#operation)
types have no generated public page or index entry.

For exact [scope](../../../../glossary.md#term-scope), compare the tracked/untracked path list with the six required and seven
conditional paths. Confirm all changed paths are Markdown and reject any path under Java source,
Java tests, Gradle, `ARCHITECTURE.md`, `docs/design/decisions`, or `testing`.

At task start there must be exactly one [Ready](../../../../glossary.md#term-ready) task heading and one Ready master-plan row:
Planning 0006. At a completed audit outcome there must be no Ready task or row. A blocking finding
may add at most one concise Draft row and no detailed specification. Preserve all completed task
files and verify that `tasks/` contains no new file except this task.

## Documentation-focused execution

No second documentation-agent pass is required or permitted merely to repeat this task. The task
itself must execute in a separate clean documentation-focused context, distinct from [Planning](../../../../glossary.md#term-planning)
0001-0005 implementation and documentation contexts. It performs no Java or behavior change, so
the same clean context owns the audit [artifact](../../../../glossary.md#term-artifact), any permitted Markdown corrections, generated-
[Javadoc](../../../../glossary.md#term-javadoc) inspection, Markdown [validation](../../../../glossary.md#term-validation), and completion evidence.

If execution discovers a need to edit Java/Javadoc source or executable behavior, it must record a
blocking finding rather than hand the edit to another agent inside this task.

## Dependencies

- Complete Planning tasks 0001-0005, including their implementation tests, final Javadocs,
  explanatory documentation, and independent documentation-pass evidence.
- Complete [backend](../../../../glossary.md#backend)-contract tasks 0001-0004 and current [`BackendId`](../../../../glossary.md#backend-identity--backendid), availability, device-class,
  and hard-requirement contracts.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- Complete config tasks 0001-0003 and current backend intent, compile mode, graph optimization,
  and soft partition-scoring preference inputs.<span data-glossary-navigation="true"> ([backend intent glossary](../../../../glossary.md#backend-intent--backendintent); [compile mode glossary](../../../../glossary.md#compile-mode--compilemode); [graph optimization glossary](../../../../glossary.md#forward-graph-optimization); [partition glossary](../../../../glossary.md#partition))</span>
- Complete [model](../../../../glossary.md#term-model) [graph](../../../../glossary.md#graph)/[operation](../../../../glossary.md#operation)/[descriptor](../../../../glossary.md#term-descriptor)/[publication](../../../../glossary.md#term-publication) foundations, especially
  [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), `CompiledNode`, `GraphValue`, `Operation`, `TensorDescriptor`, typed graph
  identifiers, graph phases, and standalone `PublicationBinding`.
- Accepted architecture [ownership](../../../../glossary.md#term-ownership-meanings) and [lifecycle](../../../../glossary.md#lifecycle) boundaries for [planning](../../../../glossary.md#term-planning), [compiler](../../../../glossary.md#term-compiler),
  `CompileArtifacts`, [prepare](../../../../glossary.md#prepare), concrete backends, [runtime](../../../../glossary.md#term-runtime), and [engine](../../../../glossary.md#term-engine).

## Follow-up tasks

- A future [compiler](../../../../glossary.md#term-compiler) frontier owns [graph](../../../../glossary.md#graph)-wide invocation of [planning](../../../../glossary.md#term-planning), complete [owner](../../../../glossary.md#term-owner)-map assembly,
  `CompileArtifacts`, `PublicationPlan`, diagnostics, and public compile-failure translation. This
  audit creates no detailed compiler task and no public planning orchestration surface.
- A later concrete compiler consumer may justify a narrow public planning collaboration or adapter.
  That consumer-driven integration must preserve the audited internal semantics and may not move
  compiler orchestration into planning.
- Config 0004 and cost-bearing ownership scoring remain Draft until a concrete planning cost
  consumer establishes backend-neutral classifications and units. They are not prerequisites for
  the selected cost-free planning milestone unless this audit proves otherwise.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend))</span>
- [Trace](../../../../glossary.md#trace) [compile](../../../../glossary.md#compile) payloads remain Draft until stable compiler [producer](../../../../glossary.md#term-producer)/emission contracts exist.
- Prepare, runtime, engine, concrete backends, and model-autotuning remain with their future
  owners. This task does not choose or specify the next global frontier; a separate reassessment
  must do so after closure.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [model glossary](../../../../glossary.md#term-model); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>

## Architecture impact

Expected impact: None.

The task audits implementation against the existing contract and may correct documentation-only
drift. If evidence requires a new public [ownership](../../../../glossary.md#term-ownership-meanings) rule, [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), orchestration [owner](../../../../glossary.md#term-owner),
cost/profile contract, architecture exception, or [lifecycle](../../../../glossary.md#lifecycle) change, stop and return
`ARCHITECTURE_DECISION_REQUIRED` without editing architecture.

## Implementation prompt

Use this prompt in one separate clean documentation-focused task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md and the General/Planning/Architecture/API/Example
profiles, docs/planning/roadmap.md, planning/config/backend-contract/trace/compiler/prepare/runtime/
engine master plans, completed Planning tasks 0001-0005, current planning/model/config/backend-
contract source and tests, generated planning Javadocs, directly relevant API/guides/glossary,
memory-planning note, and
docs/planning/modules/planning/tasks/0006-planning-contract-closure-audit.md in full.

Execute task 0006 exactly as a documentation-only closure audit. Create its one durable result
artifact, use only its verdict and finding vocabulary, fill the required evidence matrix, run the
single final checkpoint and documentation/scope/status checks, and synchronize status exactly.
Do not change Java, tests, Gradle, architecture contracts/ADRs/tests, compiler/prepare/runtime/
backend behavior, or create a later detailed task. Stop and report any architecture or scope
conflict. This clean documentation-focused context is the required documentation pass; do not
spawn a redundant second documentation pass.
```

## Local decisions

- Frontier reassessment selects Planning 0006 as the next task because Planning 0001-0005 now
  cover the complete selected planning pipeline from one occurrence-level capability question
  through immutable partition and logical-memory recipes. The next responsible action is to prove
  closure, not add another semantic or public Java layer.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [immutable glossary](../../../../glossary.md#term-immutable); [partition glossary](../../../../glossary.md#partition))</span>
- The closure task is documentation/design audit only. Implementing a finding would destroy the
  independence of the exit gate and exceed the current authorization.
- A durable audit artifact is justified by the model-closure precedent and by the amount of
  evidence that should remain available after the concise master-plan/roadmap status changes.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [model glossary](../../../../glossary.md#term-model))</span>
- The current planning-stage decision is that the four evaluator/generator operations may remain
  package-private. No current external consumer needs to call an isolated stage, and exposing one
  now would preempt the compiler-owned orchestration boundary. Task execution must independently
  verify this and may overturn it only by recording a blocking or architecture finding.
- The current planning-stage decision is that the five public declarations are sufficient and
  minimal: the provider/query form the inward capability collaboration, while partition and
  logical-memory records are immutable cross-package compile recipes. Eligibility, owner
  selection, owner-map assembly, and generation remain implementation operations, not retained
  public artifacts.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [owner glossary](../../../../glossary.md#term-owner))</span>
- A future narrow public planning collaboration is non-blocking consumer-driven integration. It
  may be added with the concrete compiler consumer while preserving current semantics; its absent
  shape should not be guessed in this closure task.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- `PublicationBinding` remains outside [planning](../../../../glossary.md#term-planning). A future [compiler](../../../../glossary.md#term-compiler)-owned `PublicationPlan` owns
  graph membership, tensor-to-value publication context, policy, and diagnostics.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [tensor glossary](../../../../glossary.md#tensor); [publication glossary](../../../../glossary.md#term-publication))</span>
- Numeric/cost-bearing scoring and Config 0004 remain non-blocking. The selected deterministic
  preferred-class/provider-order baseline is a complete current ownership policy and requires no
  speculative operation-family, workload bucket, cost unit, or backend profile.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [operation glossary](../../../../glossary.md#operation); [backend glossary](../../../../glossary.md#backend))</span>
- Use a single combined repository checkpoint because this task closes a module milestone. The
  audit still changes no executable behavior and must not repeat the successful command in a
  second context.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [module glossary](../../../../glossary.md#term-module))</span>

## Known limitations

- No current public workflow invokes the internal [planning](../../../../glossary.md#term-planning) stages end to end.
- No current [compiler](../../../../glossary.md#term-compiler) consumes planning or constructs `CompileArtifacts`, `PublicationPlan`,
  compile diagnostics, or a public no-match failure.<span data-glossary-navigation="true"> ([compile diagnostics glossary](../../../../glossary.md#compile-diagnostics))</span>
- No production [concrete backend](../../../../glossary.md#concrete-backend) implements [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider); provider conformance is
  future backend work.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [backend glossary](../../../../glossary.md#backend))</span>
- The current owner selector is the deterministic cost-free baseline, not a general numeric or
  profile-guided cost model.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [cost model glossary](../../../../glossary.md#compiler-transformations-and-dominance))</span>
- The logical memory plan contains relationships and descriptors, not byte sizes, lifetimes,
  transfers, slots, allocation, physical representation, or residency.<span data-glossary-navigation="true"> ([logical memory plan glossary](../../../../glossary.md#logical-memory-plan); [descriptors glossary](../../../../glossary.md#term-descriptor); [residency glossary](../../../../glossary.md#residency))</span>
- Generated Javadoc success proves rendering and visibility only; the audit must still inspect
  semantic wording against source/tests.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Closure of the selected milestone does not freeze compatibility or prevent later consumer-
  driven integration tasks. It confirms only that no current planning-owned semantic gap blocks
  the next lifecycle layers.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- [Planning](../../../../glossary.md#term-planning) context `/root/plan_planning_0006` read the repository instructions, architecture
  contract and focused architecture/ADR documents, documentation rules/profiles, planning guide
  and roadmap, planning/config/backend-contract/trace/compiler/prepare/runtime/engine master
  plans, completed Planning 0001-0005 evidence, model closure-audit precedent, current planning
  source/test/Javadocs, model graph/publication contracts, public/compile APIs, provider/backend-
  selection guides, glossary, and memory-planning note.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend); [trace glossary](../../../../glossary.md#trace); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication); [compile glossary](../../../../glossary.md#compile))</span>
- Frontier reassessment found no architecture conflict or concrete missing planning semantic.
  Current public DTOs carry the cross-package immutable recipe facts, while the internal
  operations can remain hidden until a compiler consumer establishes one narrow orchestration
  boundary.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Planning-stage `python3 /tmp/validate_synaptik_markdown.py` passed for 230 Markdown files, 4,132
  local links, 251 local anchors, 2,900 fence markers, final newlines, and trailing whitespace.
- Planning-stage `git diff --check` passed with no output.
- Final status checks found exactly one Ready task heading and exactly one Ready master-plan row:
  Planning 0006. Planning 0001-0005 remain Complete; Config 0004+, Trace 0003+, and compiler,
  prepare, runtime, and later planning work remain Draft. The planning task directory contains
  exactly the completed 0001-0005 specs plus this one Ready 0006 spec; no later detailed config,
  trace, compiler, prepare, runtime, or planning specification exists.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- The final tracked/untracked audit contains 19 paths: the exact preserved 18-path Planning 0005
  implementation/documentation change plus this one new task file. Planning 0006 changed content
  only in its new task, planning/config/trace master-plan status text, and the roadmap; the latter
  four paths were already part of the uncommitted Planning 0005 documentation change.
- Comparison with the initial status snapshot confirms this planning context added no Java, test,
  Gradle, <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [Gradle glossary](../../../../glossary.md#term-gradle))</span>`ARCHITECTURE.md`, ADR, architecture-test, conformance, integration, compiler, prepare,
  runtime, backend, or other executable path. Existing untracked planning-memory Java/tests belong
  to completed Planning 0005 and were preserved without modification.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- Manual review confirmed the canonical task sections, exact audit artifact outline, verdict and
  finding vocabulary, required evidence matrix, six required plus seven conditional execution
  paths, explicit public/internal decisions, compiler/<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [artifact glossary](../../../../glossary.md#term-artifact))</span>`CompileArtifacts`/publication/prepare
  handoff, standalone clean-context prompt, no-redundant-documentation-pass rule, and status
  synchronization outcomes.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Clean documentation context `/root/audit_planning_0006` independently inventoried all 12
  production files and eight test suites, read their source and tests, inspected the exact Gradle
  surface, and created the durable
  [planning contract closure audit](../planning-contract-closure-audit.md) with exactly the 17
  required sections, complete evidence matrix, and `CLOSED` verdict.
- The audit confirms that the exact five public declarations are sufficient and minimal and that
  `BackendEligibility.evaluate`, `BackendOwnerSelection.select`,
  `MaximalSameOwnerPartitioning.partition`, and `LogicalMemoryPlanning.plan` may remain package-
  private until a concrete compiler consumer establishes a narrow collaboration boundary.
- The single final `./gradlew test :testing:architecture-tests:test
  :modules:planning:javadoc` checkpoint passed. Gradle reported 47 actionable tasks: 2 executed,
  45 up-to-date, and 0 skipped. XML reports contain 1,137 repository tests across 149 suites, 63
  planning tests across eight suites, and three architecture tests across three suites, all with
  zero failures, errors, or skips.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Generated Javadocs contain all three planning package summaries and the five public type pages.
  `allclasses-index.html` and `overview-tree.html` expose exactly those public types; the four
  package-private operation types have no page or index entry.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Audit review found one permitted documentation drift: the capability-provider lifecycle diagram
  still called partitioning planned. It now records current internal maximal partitioning and
  immutable partition/logical-memory recipes while keeping public orchestration and cost-bearing
  scoring planned. No other conditional documentation correction was required.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [partition glossary](../../../../glossary.md#partition))</span>
- The final 21-path tree is the preserved 18-path Planning 0005 change, the Planning 0006 task
  specification/status path, the required audit artifact, and the one permitted capability-
  provider guide correction. The audit context changed no Java, test, Gradle, architecture
  contract/ADR/test, conformance, integration, compiler, prepare, runtime, backend, or other
  executable path.
- Final `python3 /tmp/validate_synaptik_markdown.py` passed for 231 Markdown files, 4,134 local
  links, 251 local anchors, 2,916 fence markers, final newlines, and trailing whitespace. Final
  `git diff --check` passed with no output, and the exact path scan returned 21 paths. The combined
  documentation shell reported exit code 1 only because its final exact
  `rg -n '^Ready$|\| Ready \|' docs/planning` command correctly found zero Ready headings or rows;
  no preceding validation command failed. The task-directory scan contains only Planning
  0001-0006, with no later detailed task specification or next frontier selection. Planning
  0001-0006 and the selected planning roadmap area are Complete; Config 0004+, Trace 0003+,
  compiler, prepare, runtime, and later work remain Draft.

## Implementation notes

The audit found no [planning](../../../../glossary.md#term-planning)-owned gap or architecture conflict. It corrected only the stale
capability-provider [lifecycle](../../../../glossary.md#lifecycle) phrase, recorded downstream [compiler](../../../../glossary.md#term-compiler)/[publication](../../../../glossary.md#term-publication)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/
[backend](../../../../glossary.md#backend)/cost/[trace](../../../../glossary.md#trace) work as explicitly owned deferral, and synchronized the `CLOSED` verdict. It
created no follow-up row or detailed task because a separate frontier reassessment owns that
choice.

## Completion summary

Completed the [planning](../../../../glossary.md#term-planning) contract closure audit and recorded the durable `CLOSED` verdict. Created
`docs/planning/modules/planning/planning-contract-closure-audit.md`; finalized this task; updated
the planning/config/[trace](../../../../glossary.md#trace) master-plan and roadmap status narratives; and corrected the one stale
capability-provider [lifecycle](../../../../glossary.md#lifecycle) phrase. The single combined [Gradle](../../../../glossary.md#term-gradle) [checkpoint](../../../../glossary.md#term-checkpoint), repository/planning/
architecture XML test counts, generated-[Javadoc](../../../../glossary.md#term-javadoc) visibility inspection, Markdown [validation](../../../../glossary.md#term-validation),
[scope](../../../../glossary.md#term-scope)/status/task-directory checks, and `git diff --check` passed. No unresolved issue remains.
Future [compiler](../../../../glossary.md#term-compiler) integration, [publication](../../../../glossary.md#term-publication) planning, [prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) physical realization,
cost-bearing scoring, tracing payloads, and [model autotuning](../../../../glossary.md#autotuning--model-autotuning) remain intentionally downstream; a
separate reassessment must select any next task.

Status: Complete
