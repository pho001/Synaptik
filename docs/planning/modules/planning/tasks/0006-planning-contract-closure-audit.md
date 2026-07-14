# Task 0006: Planning Contract Closure Audit

## Status

Complete

## Goal

Run the final planning-only exit gate for the selected `modules/planning` milestone. Audit the
implemented capability question/provider collaboration, hard eligibility, cost-free baseline
owner selection, maximal same-owner partitioning, logical materialization and memory recipes,
package/API boundaries, dependencies, documentation, and downstream lifecycle handoff against
current source and tests.

The task produces one durable audit artifact with a falsifiable closure verdict. It changes no
Java or executable behavior. A successful `CLOSED` verdict means the selected planning contracts
are coherent and sufficiently stable for later compiler-owned orchestration and
`CompileArtifacts`; it does not claim that a compiler, public planning workflow, prepare layer,
runtime, concrete backend, cost-bearing scoring policy, or model-autotuning workflow exists.

Mental model:

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

## Scope

- Create `docs/planning/modules/planning/planning-contract-closure-audit.md` as the sole durable
  detailed result artifact.
- Inventory every current production and test file under `modules/planning`, every generated
  public Javadoc page, and every package-private planning type that must remain absent from the
  generated public indexes.
- Audit the exact task-0001 capability boundary:
  - `OperationCapabilityQuery` retains one operation occurrence and ordered descriptor snapshots;
  - `BackendCapabilityProvider` represents one stable backend identity and a deterministic
    backend-level semantic support answer; and
  - neither contract performs discovery, availability evaluation, hard matching, scoring,
    routing, preparation, or execution.
- Audit task-0002 hard eligibility against the complete provider/snapshot association, exact
  backend-identity matching, availability and hard-requirement filtering, provider call order,
  immutable provider-order result, no-match behavior, and device non-selection rules.
- Audit task-0003 baseline owner selection against its complete hard-eligible candidate set,
  preferred-class-first/provider-order comparison, exact reference retention, terminal empty
  eligibility, and absence of numeric scoring, cost/workload classifications, routes, or devices.
- Audit task-0004 partitioning against complete owner-map validation, stored topological node
  order, equality-defined ownership transitions, maximal consecutive runs, exact graph-node and
  first-owner references, multi-output indivisibility, and zero-node behavior.
- Audit task-0005 logical-memory derivation against complete ordered partition coverage,
  adjacent-owner maximality, graph-value order, exact value/descriptor/partition references,
  producer and distinct consumer relationships, graph-output preservation, dynamic/expression
  Shape support, and zero-node pass-through behavior.
- Record the exact public planning surface. Decide whether each public declaration is necessary,
  sufficient, and minimal for current cross-package recipe or provider use. Explicitly check that
  no public capability matrix, eligibility result, owner-selection result, owner map, planning
  facade, device choice, cost profile, transfer plan, or physical-memory DTO is currently needed.
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
  dependency exists.
- Review generated Javadocs, explanatory architecture/status documentation, public and compile API
  pages, provider and backend-selection examples, memory-planning design note, glossary, completed
  task history, planning/config/backend-contract/trace/compiler/prepare/runtime/engine master
  plans, and the roadmap for accurate current-versus-planned wording.
- Audit the downstream handoff without designing it:
  - compiler owns graph-wide orchestration, owner-map assembly, publication planning,
    `CompileArtifacts`, public compile failures, and diagnostics;
  - planning owns the current backend-neutral rules and immutable partition/logical-memory output;
  - a later concrete compiler consumer may justify one narrow public planning collaboration while
    the current operations remain internal now;
  - prepare consumes immutable compile artifacts and owns the shared transition to prepared state;
    and
  - physical memory, transfers, lowering, routes, kernels, residency, and execution remain with
    prepare, concrete backends, and runtime according to the architecture contract.
- Classify every discovered gap as blocking, non-blocking deferred integration, documentation
  drift correctable within this task, confirmed no-change, or an architecture decision.
- Run one final combined planning-milestone checkpoint after the artifact and documentation are
  stable, then synchronize task, planning master-plan, and roadmap status with the verdict.
- If the verdict is `CLOSED`, mark task 0006, the selected planning milestone, and the roadmap
  planning area `Complete`. Record the future compiler/prepare handoff, but create no next detailed
  task specification and do not make a later task Ready.
- If the verdict is `BLOCKING_GAP`, the audit task may become `Complete`, but keep the planning
  milestone and roadmap area open. Add at most one concise Draft master-plan row for the first
  cohesive blocking frontier; do not create its detailed specification.
- If the verdict is `ARCHITECTURE_DECISION_REQUIRED`, stop without changing architecture. Mark
  task 0006 `Blocked`, keep the planning milestone open, and use the canonical incomplete
  completion status.

## Out of scope

- creating, deleting, renaming, moving, or modifying Java production or test files
- modifying Gradle, project dependencies, module structure, generated Javadocs, architecture
  tests, backend-conformance tests, integration tests, or any executable artifact
- changing `ARCHITECTURE.md`, an ADR, or an architecture rule or dependency direction
- implementing, repairing, widening, or making public any capability, eligibility, owner-
  selection, partitioning, or logical-memory operation found during the audit
- adding a public planning facade, pipeline, matrix, evaluator, owner map, result row, compiler
  adapter, provider registry, discovery mechanism, or service locator
- implementing compiler capture or orchestration, `CompileArtifacts`, `PublicationPlan`, public
  compile failures, compile diagnostics, graph transformations, autograd, or publication behavior
- defining cost/workload families, buckets, units, estimates, numeric scoring, planning-cost
  profiles, Config 0004, tuning candidates, tuning caches, or model-autotuning inputs
- adding trace attributes, payload schemas, emission, correlation domains, or serialization
- implementing prepare, runtime, engine, concrete backend, lowering, fusion, specialization,
  transfer, physical memory, route, kernel, schedule, residency, or execution behavior
- selecting a device, retaining a device identity in an ownership result, or interpreting backend-
  specific implementation parameters
- editing completed task specifications except this task's own execution evidence and final status
- creating a detailed compiler, config, trace, planning follow-up, prepare, runtime, engine,
  backend, or other future task specification
- unrelated documentation cleanup or turning the result artifact into a tutorial or speculative
  future API design

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Tracing](../../../../architecture/tracing.md)
- [Training graph](../../../../architecture/training-graph.md)
- [ADR 0001](../../../../design/decisions/0001-layered-architecture.md)
- [ADR 0002](../../../../design/decisions/0002-backend-owned-lowering.md)
- [ADR 0004](../../../../design/decisions/0004-partition-scoring.md)
- [ADR 0008](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [Memory planning strategy](../../../../design/notes/memory-planning-strategy.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Architecture style](../../../../developer-guide/documentation/architecture-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
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
- [Compile API](../../../../api/compile-api.md)
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative. This audit may confirm or report the existing boundary but
  cannot create, reinterpret, or change an architecture rule.
- Planning owns backend-neutral capability analysis, backend ownership, maximal same-owner
  partitioning, and logical materialization/memory requirements. It answers where work runs, not
  which backend implementation runs it.
- Compiler owns graph-wide planning orchestration and immutable `CompileArtifacts`. The absence of
  a current compiler-owned orchestrator does not by itself require planning to expose isolated
  public evaluator methods before their consumer exists.
- Prepare and concrete backends own lowering, implementation routes, executable construction, and
  physical realization. Runtime owns prepared execution and per-run residency/state.
- Current source and tests are primary implementation evidence. Completed tasks and explanatory
  documents are supporting evidence, not authority over source or architecture.
- Public planning DTOs may expose model and backend-contract types through existing public
  dependencies. Planning must not expose or retain live backend services in compile-time plans.
- A contradiction requiring Java, test, Gradle, architecture, or cross-layer behavior changes
  changes the finding/verdict; it does not expand this task's scope.

## Package impact

No Java package is added, changed, or moved.

The audit reviews these existing planning packages:

- `io.github.pho001.synaptik.planning.capability` — public occurrence/provider contracts and
  internal hard-eligibility/baseline-owner operations;
- `io.github.pho001.synaptik.planning.partition` — public immutable partition recipe and internal
  maximal consecutive same-owner generation; and
- `io.github.pho001.synaptik.planning.memory` — public logical value requirements/plan and internal
  closed-graph derivation.

The audit must not create a root facade, `ownership` implementation package, `util`, `common`,
compiler adapter, public orchestration package, or another Java surface.

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
architecture documents; every planning source/test file and generated Javadoc page; planning
Gradle; model graph/publication contracts; config/backend-contract/trace Java and tests; completed
Planning 0001-0005 history; config/backend-contract/trace/compiler/prepare/runtime/engine master
plans; architecture, conformance, and integration tests; concrete backends; all other modules.

## Maximum scope

This documentation-only audit may create or modify at most the thirteen paths listed above. The six
always-affected paths are required. A conditional path may change only to correct status,
terminology, link, example, or boundary wording already established by current source/tests and
the architecture contract.

If a finding needs Java/Javadoc source, tests, Gradle, an architecture contract/ADR, another
planning file, a fourteenth path, or new behavior, do not expand scope. Record `BLOCKING_GAP` or
`ARCHITECTURE_DECISION_REQUIRED` with the exact owner and required follow-up.

## Required audit artifact

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
  rationale are recorded.
- `BLOCKING_GAP` — the audit is complete, but at least one planning-owned gap prevents milestone
  closure. Every gap names evidence, impact, owner, dependency, and the first bounded follow-up.
- `ARCHITECTURE_DECISION_REQUIRED` — evidence conflicts with the authoritative architecture or
  requires a new ownership/dependency decision. Stop without editing architecture.

Findings use exactly these labels:

- `BLOCKING`
- `NON_BLOCKING_DEFERRED`
- `DOCUMENTATION_DRIFT`
- `NO_CHANGE_CONFIRMED`

An architecture conflict is recorded under the global `ARCHITECTURE_DECISION_REQUIRED` verdict,
not disguised as ordinary documentation drift.

### Required evidence matrix

The artifact contains one matrix with exactly these columns:

| Area | Current contract | Primary evidence | Required invariant or question | Finding label | Disposition |
|---|---|---|---|---|---|

It contains at least one row for each of these areas:

- capability query/provider;
- hard eligibility;
- baseline owner selection;
- maximal same-owner partitioning;
- logical-memory derivation;
- public DTO minimality;
- package-private operation visibility;
- validation and failure ordering;
- equality-based association and exact-reference retention;
- package/dependency surface;
- generated Javadocs and public/internal index boundary;
- API/guide examples and glossary terminology;
- completed task/master-plan/roadmap history;
- compiler orchestration and `CompileArtifacts` handoff;
- publication-plan boundary; and
- prepare/runtime/backend physical-realization boundary.

Every row names source/test/API evidence or an exact inspection command. Completed task prose
alone is not sufficient evidence. The matrix must explicitly answer both questions:

1. May the four current operations remain package-private until the compiler-owned orchestration
   consumer establishes a narrow collaboration boundary?
2. Are the five current public planning declarations sufficient and minimal for the selected
   milestone and downstream immutable recipe handoff?

## Acceptance criteria

- The durable artifact contains every required top-level section, exactly one closure verdict,
  only the permitted finding labels, and the complete required evidence matrix.
- Every current planning production/test file and generated public/internal Javadoc boundary is
  included in a recorded inventory method; no declaration or test suite is omitted.
- The capability provider/query, hard eligibility, owner selection, partitioning, and logical-
  memory contracts are compared with their current source/tests rather than inferred only from
  completed task status.
- Exact validation and failure ordering, immutable snapshots, equality rules, encounter/graph/
  partition ordering, and exact reference-retention promises are recorded for every current stage.
- The audit explicitly concludes whether the current four package-private operations may remain
  internal until a compiler-owned orchestrator exists. A `CLOSED` verdict must explain why that
  deferral does not leave a missing planning semantic contract.
- The audit explicitly concludes whether the five public declarations are sufficient and minimal.
  A `CLOSED` verdict must reject premature public matrices, owner rows/maps, orchestration,
  physical memory, cost/workload profiles, devices, routes, and executable state.
- Package placement and the exact Gradle dependency/visibility surface match the current master
  plan and architecture; no forbidden dependency or concrete-backend coupling exists.
- Generated Javadocs accurately render the five public declarations and package summaries while
  omitting `BackendEligibility`, `BackendOwnerSelection`, `MaximalSameOwnerPartitioning`, and
  `LogicalMemoryPlanning` from public class indexes/pages.
- Explanatory/API/guide/glossary examples and current-versus-planned wording match current source.
  Any permitted documentation drift is corrected and recorded; Java-Javadoc drift is blocking
  because Java edits are forbidden.
- The artifact distinguishes planning-owned semantics from future compiler orchestration,
  `CompileArtifacts`, compiler-owned `PublicationPlan`, public failure translation, prepare,
  physical memory, backend lowering, runtime residency, and execution.
- Cost-bearing scoring, cost/workload classifications, Config 0004, trace payloads, and tuning
  remain non-blocking Draft/downstream work unless current evidence proves a planning-owned gap.
- Completed Planning 0001-0005 history remains unchanged. No later detailed task specification is
  created, and no later task is made Ready.
- The final combined repository/planning checkpoint, Markdown validation, exact path/status/spec-
  directory checks, final newlines, trailing whitespace, and `git diff --check` pass.
- No Java, test, Gradle, `ARCHITECTURE.md`, ADR, architecture-test, conformance, integration,
  compiler, prepare, runtime, backend, or other executable path changes.
- Verdict/status synchronization follows the exact rules in Scope and is reflected consistently
  in this task, the planning master plan, and the roadmap.
- The task records local decisions, limitations, exact commands/results, audit context identity,
  inventory method, findings, milestone disposition, and canonical completion summary.

## Tests / validation

This task is a clean-context documentation-focused audit and changes no executable Java. Inspect
source, tests, generated Javadocs, and documentation before the final checkpoint. Then run exactly
one combined Gradle checkpoint:

```bash
./gradlew test :testing:architecture-tests:test :modules:planning:javadoc
```

Gradle may deduplicate tasks already selected by root `test`. Record executed/up-to-date/skipped
task counts and planning/architecture/repository test counts from XML reports where available. Do
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
`allclasses-index.html`, and `overview-tree.html`. Confirm the four package-private operation
types have no generated public page or index entry.

For exact scope, compare the tracked/untracked path list with the six required and seven
conditional paths. Confirm all changed paths are Markdown and reject any path under Java source,
Java tests, Gradle, `ARCHITECTURE.md`, `docs/design/decisions`, or `testing`.

At task start there must be exactly one Ready task heading and one Ready master-plan row:
Planning 0006. At a completed audit outcome there must be no Ready task or row. A blocking finding
may add at most one concise Draft row and no detailed specification. Preserve all completed task
files and verify that `tasks/` contains no new file except this task.

## Documentation-focused execution

No second documentation-agent pass is required or permitted merely to repeat this task. The task
itself must execute in a separate clean documentation-focused context, distinct from Planning
0001-0005 implementation and documentation contexts. It performs no Java or behavior change, so
the same clean context owns the audit artifact, any permitted Markdown corrections, generated-
Javadoc inspection, Markdown validation, and completion evidence.

If execution discovers a need to edit Java/Javadoc source or executable behavior, it must record a
blocking finding rather than hand the edit to another agent inside this task.

## Dependencies

- Complete Planning tasks 0001-0005, including their implementation tests, final Javadocs,
  explanatory documentation, and independent documentation-pass evidence.
- Complete backend-contract tasks 0001-0004 and current `BackendId`, availability, device-class,
  and hard-requirement contracts.
- Complete config tasks 0001-0003 and current backend intent, compile mode, graph optimization,
  and soft partition-scoring preference inputs.
- Complete model graph/operation/descriptor/publication foundations, especially
  `CompiledGraphModel`, `CompiledNode`, `GraphValue`, `Operation`, `TensorDescriptor`, typed graph
  identifiers, graph phases, and standalone `PublicationBinding`.
- Accepted architecture ownership and lifecycle boundaries for planning, compiler,
  `CompileArtifacts`, prepare, concrete backends, runtime, and engine.

## Follow-up tasks

- A future compiler frontier owns graph-wide invocation of planning, complete owner-map assembly,
  `CompileArtifacts`, `PublicationPlan`, diagnostics, and public compile-failure translation. This
  audit creates no detailed compiler task and no public planning orchestration surface.
- A later concrete compiler consumer may justify a narrow public planning collaboration or adapter.
  That consumer-driven integration must preserve the audited internal semantics and may not move
  compiler orchestration into planning.
- Config 0004 and cost-bearing ownership scoring remain Draft until a concrete planning cost
  consumer establishes backend-neutral classifications and units. They are not prerequisites for
  the selected cost-free planning milestone unless this audit proves otherwise.
- Trace compile payloads remain Draft until stable compiler producer/emission contracts exist.
- Prepare, runtime, engine, concrete backends, and model-autotuning remain with their future
  owners. This task does not choose or specify the next global frontier; a separate reassessment
  must do so after closure.

## Architecture impact

Expected impact: None.

The task audits implementation against the existing contract and may correct documentation-only
drift. If evidence requires a new public ownership rule, module dependency, orchestration owner,
cost/profile contract, architecture exception, or lifecycle change, stop and return
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
  closure, not add another semantic or public Java layer.
- The closure task is documentation/design audit only. Implementing a finding would destroy the
  independence of the exit gate and exceed the current authorization.
- A durable audit artifact is justified by the model-closure precedent and by the amount of
  evidence that should remain available after the concise master-plan/roadmap status changes.
- The current planning-stage decision is that the four evaluator/generator operations may remain
  package-private. No current external consumer needs to call an isolated stage, and exposing one
  now would preempt the compiler-owned orchestration boundary. Task execution must independently
  verify this and may overturn it only by recording a blocking or architecture finding.
- The current planning-stage decision is that the five public declarations are sufficient and
  minimal: the provider/query form the inward capability collaboration, while partition and
  logical-memory records are immutable cross-package compile recipes. Eligibility, owner
  selection, owner-map assembly, and generation remain implementation operations, not retained
  public artifacts.
- A future narrow public planning collaboration is non-blocking consumer-driven integration. It
  may be added with the concrete compiler consumer while preserving current semantics; its absent
  shape should not be guessed in this closure task.
- `PublicationBinding` remains outside planning. A future compiler-owned `PublicationPlan` owns
  graph membership, tensor-to-value publication context, policy, and diagnostics.
- Numeric/cost-bearing scoring and Config 0004 remain non-blocking. The selected deterministic
  preferred-class/provider-order baseline is a complete current ownership policy and requires no
  speculative operation-family, workload bucket, cost unit, or backend profile.
- Use a single combined repository checkpoint because this task closes a module milestone. The
  audit still changes no executable behavior and must not repeat the successful command in a
  second context.

## Known limitations

- No current public workflow invokes the internal planning stages end to end.
- No current compiler consumes planning or constructs `CompileArtifacts`, `PublicationPlan`,
  compile diagnostics, or a public no-match failure.
- No production concrete backend implements `BackendCapabilityProvider`; provider conformance is
  future backend work.
- The current owner selector is the deterministic cost-free baseline, not a general numeric or
  profile-guided cost model.
- The logical memory plan contains relationships and descriptors, not byte sizes, lifetimes,
  transfers, slots, allocation, physical representation, or residency.
- Generated Javadoc success proves rendering and visibility only; the audit must still inspect
  semantic wording against source/tests.
- Closure of the selected milestone does not freeze compatibility or prevent later consumer-
  driven integration tasks. It confirms only that no current planning-owned semantic gap blocks
  the next lifecycle layers.

## Validation evidence

- Planning context `/root/plan_planning_0006` read the repository instructions, architecture
  contract and focused architecture/ADR documents, documentation rules/profiles, planning guide
  and roadmap, planning/config/backend-contract/trace/compiler/prepare/runtime/engine master
  plans, completed Planning 0001-0005 evidence, model closure-audit precedent, current planning
  source/test/Javadocs, model graph/publication contracts, public/compile APIs, provider/backend-
  selection guides, glossary, and memory-planning note.
- Frontier reassessment found no architecture conflict or concrete missing planning semantic.
  Current public DTOs carry the cross-package immutable recipe facts, while the internal
  operations can remain hidden until a compiler consumer establishes one narrow orchestration
  boundary.
- Planning-stage `python3 /tmp/validate_synaptik_markdown.py` passed for 230 Markdown files, 4,132
  local links, 251 local anchors, 2,900 fence markers, final newlines, and trailing whitespace.
- Planning-stage `git diff --check` passed with no output.
- Final status checks found exactly one Ready task heading and exactly one Ready master-plan row:
  Planning 0006. Planning 0001-0005 remain Complete; Config 0004+, Trace 0003+, and compiler,
  prepare, runtime, and later planning work remain Draft. The planning task directory contains
  exactly the completed 0001-0005 specs plus this one Ready 0006 spec; no later detailed config,
  trace, compiler, prepare, runtime, or planning specification exists.
- The final tracked/untracked audit contains 19 paths: the exact preserved 18-path Planning 0005
  implementation/documentation change plus this one new task file. Planning 0006 changed content
  only in its new task, planning/config/trace master-plan status text, and the roadmap; the latter
  four paths were already part of the uncommitted Planning 0005 documentation change.
- Comparison with the initial status snapshot confirms this planning context added no Java, test,
  Gradle, `ARCHITECTURE.md`, ADR, architecture-test, conformance, integration, compiler, prepare,
  runtime, backend, or other executable path. Existing untracked planning-memory Java/tests belong
  to completed Planning 0005 and were preserved without modification.
- Manual review confirmed the canonical task sections, exact audit artifact outline, verdict and
  finding vocabulary, required evidence matrix, six required plus seven conditional execution
  paths, explicit public/internal decisions, compiler/`CompileArtifacts`/publication/prepare
  handoff, standalone clean-context prompt, no-redundant-documentation-pass rule, and status
  synchronization outcomes.
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
  zero failures, errors, or skips.
- Generated Javadocs contain all three planning package summaries and the five public type pages.
  `allclasses-index.html` and `overview-tree.html` expose exactly those public types; the four
  package-private operation types have no page or index entry.
- Audit review found one permitted documentation drift: the capability-provider lifecycle diagram
  still called partitioning planned. It now records current internal maximal partitioning and
  immutable partition/logical-memory recipes while keeping public orchestration and cost-bearing
  scoring planned. No other conditional documentation correction was required.
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

The audit found no planning-owned gap or architecture conflict. It corrected only the stale
capability-provider lifecycle phrase, recorded downstream compiler/publication/prepare/runtime/
backend/cost/trace work as explicitly owned deferral, and synchronized the `CLOSED` verdict. It
created no follow-up row or detailed task because a separate frontier reassessment owns that
choice.

## Completion summary

Completed the planning contract closure audit and recorded the durable `CLOSED` verdict. Created
`docs/planning/modules/planning/planning-contract-closure-audit.md`; finalized this task; updated
the planning/config/trace master-plan and roadmap status narratives; and corrected the one stale
capability-provider lifecycle phrase. The single combined Gradle checkpoint, repository/planning/
architecture XML test counts, generated-Javadoc visibility inspection, Markdown validation,
scope/status/task-directory checks, and `git diff --check` passed. No unresolved issue remains.
Future compiler integration, publication planning, prepare/runtime/backend physical realization,
cost-bearing scoring, tracing payloads, and model autotuning remain intentionally downstream; a
separate reassessment must select any next task.

Status: Complete
