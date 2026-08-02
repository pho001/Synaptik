# Task 0011: Runtime Contract Closure Audit

## Status

Complete

## Goal

Run the final documentation-only capability checkpoint for the selected `modules/runtime`
milestone. Audit the complete Runtime 0001-0010 contract against current source, tests, generated
Javadocs, dependencies, explanatory documentation, and preserved task history, then produce one
durable evidence-backed closure verdict.

This task changes no executable Java. It must not repair a discovered implementation, test,
Javadoc-source, build, dependency, or architecture defect. Such a defect is a finding for separate
future planning. A successful `CLOSED` verdict means the current prepared-execution and per-run
state contracts are coherent at their documented boundary; it does not claim that public result-
value access, complete Prepare orchestration, Engine composition, a concrete backend, run tracing,
run policy, or tuning exists.

Mental model:

```text
authoritative architecture
  + completed Runtime 0001-0010 source, tests, and generated Javadocs
  + exact prepared-versus-per-run lifecycle and resource ownership
  + cold binding, direct bound traversal, validity, publication, and cleanup
  + package/dependency/API/example/glossary/history consistency
    -> Runtime contract closure verdict
       CLOSED
       BLOCKING_GAP
       ARCHITECTURE_DECISION_REQUIRED
```

## Scope

- Create `docs/planning/modules/runtime/runtime-contract-closure-audit.md` as the sole durable
  detailed result artifact.
- Inventory every current Runtime production file, test file and suite, generated public Javadoc
  page, public declaration, package-private declaration, and package.
- Preserve every completed Runtime 0001-0010 task and its decisions, limitations, validation
  evidence, no-change conclusions, and follow-ups. Reconcile task claims with primary evidence;
  do not treat completed prose as sufficient proof.
- Audit public and package-private API cohesion, package ownership, immutability, nominal type
  separation, equality-based association, exact-reference retention, and reference-identity
  invariants.
- Audit the complete lifecycle in exact order:
  - immutable `PreparedMemoryPlan` and `PreparedExecution` construction and exact-plan identity;
  - one isolated `RunState` creation from dense caller inputs;
  - validation before ownership transfer and all-or-cleaned cold creation;
  - cold binding of executable, transfer, and publication occurrences before traversal;
  - first-only representation creation, executable invocation, transfer/materialization,
    validity transitions, and publication in schedule order;
  - construction of the whole-state `RunResult` lease on success;
  - reverse deterministic cleanup, idempotence, closed-state behavior, and primary/suppressed
    failure preservation; and
  - empty plans/schedules/results plus repeated and concurrent runs.
- Audit the separation between reusable prepared recipes and mutable per-run state. Confirm the
  documented lifetime of borrowed caller inputs, run-owned buffers/workspaces, backend-owned
  physical representations, bound direct-reference actions, and the result lease.
- Audit aliasing and validity semantics: zero, one, or multiple valid copies; destination-valid
  transfer no-op; success-only transfer validity; conservative executable write invalidation;
  publication of an already-valid named copy; permitted result-position aliasing; and no hidden
  fallback, coherence, copy, conversion, or value extraction.
- Audit repeated/concurrent-run isolation. Prepared recipes may be shared, but active runs must
  not share `RunState`, validity bits, run-owned resources, bound invocations, publications, or
  cleanup ownership.
- Audit the cold/hot boundary in both source and bytecode-oriented tests. Bound schedule traversal
  must use direct resolved references and must not inspect graphs, resolve services, rediscover
  backends, search routes or kernels, use reflection, perform map or boxing lookup, synchronize,
  or allocate avoidably per bound occurrence.
- Audit package cohesion across `runtime.memory`, `runtime.resource`, `runtime.execution`,
  `runtime.schedule`, and `runtime.run`, including whether every public declaration has a current
  cross-package/backend consumer and every package-private declaration remains an implementation
  detail.
- Audit the exact Runtime Gradle dependency/visibility surface and imports against Runtime's
  boundary with Model, Config, Backend Contract, Trace, Compiler, Planning, Prepare, Engine, and
  concrete backends. Confirm that Runtime does not absorb graph, planning, prepare, composition,
  discovery, lowering, route, kernel, or concrete-backend ownership.
- Review generated Runtime Javadocs and public indexes, Runtime and Public API pages, all current
  Runtime examples, backend guide, glossary, focused architecture explanations, master plans,
  completed task history, and roadmap for semantic accuracy and current-versus-planned wording.
- Record whether stale planned-current wording exists. Correct only permitted explanatory
  Markdown drift; Java/Javadoc-source drift is a finding because Java edits are forbidden.
- Audit validation coverage and decide explicitly whether the capability checkpoint triggers
  repository-wide, architecture, backend-conformance, or integration validation. Run the one
  combined checkpoint defined below; record conformance/integration as not triggered unless a
  current concrete backend or end-to-end Engine execution path makes them applicable.
- Classify public result-value access, Prepare 0003 translation/orchestration, Engine composition,
  concrete backend realization/execution, Trace run payloads, Config run/publication policy, and
  performance tuning under their current owners. Treat them as non-blocking only when current
  Runtime is independently coherent without them.
- Classify every finding with the exact verdict and finding vocabularies below.
- Synchronize this task, the Runtime master plan, and roadmap only after the durable artifact and
  final checkpoint are stable.
- If the verdict is `CLOSED`, mark task 0011, the selected Runtime milestone, and the roadmap
  Runtime area `Complete`. Do not select, create, or specify Runtime 0012 or make Prepare 0003 or
  any later task Ready.
- If the verdict is `BLOCKING_GAP`, the audit task may become `Complete`, but keep the Runtime
  milestone and roadmap area open. Record the precise owner and bounded future finding in the
  artifact; do not create a follow-up task row or detailed specification inside 0011.
- If the verdict is `ARCHITECTURE_DECISION_REQUIRED`, stop without changing architecture, keep
  task 0011 `Blocked` and the Runtime milestone open, and use the canonical incomplete status.

## Out of scope

- creating, deleting, renaming, moving, or modifying Java production or test files
- modifying Java/Javadoc source, generated Javadocs, Gradle, dependency configuration, module
  structure, architecture tests, backend-conformance tests, integration tests, or executable files
- changing `ARCHITECTURE.md`, an ADR, an architecture rule, ownership, lifecycle, or dependency
  direction
- implementing or repairing prepared plans, schedules, state creation, binding, representation
  creation, executable invocation, transfer, validity, publication, result leases, or cleanup
- adding public result-value access, tensor reconstruction, mapping, copying, conversion, or a
  public run/Engine facade
- implementing Prepare 0003, translation/orchestration, coverage validation, final schedule
  construction, or prepared-execution assembly
- implementing Engine composition, backend registration/discovery, a concrete backend,
  allocation, lowering, route/kernel selection, device mechanics, or execution
- adding Trace payloads, emission, correlation, serialization, Config run/publication policy, or
  profiling/tuning behavior
- widening Runtime's public API, publishing package-private helpers, introducing a facade,
  registry, service locator, reflection path, generic resource map, or implicit coherence manager
- modifying any completed Runtime 0001-0010 task history
- creating Runtime 0012, another Runtime task row, or any detailed Prepare, Engine, backend,
  Trace, Config, tuning, or other future task specification
- making Prepare 0003 or any later task Ready
- unrelated documentation cleanup or speculative future API design

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Tracing](../../../../architecture/tracing.md)
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [ADR 0001](../../../../design/decisions/0001-layered-architecture.md)
- [ADR 0002](../../../../design/decisions/0002-backend-owned-lowering.md)
- [ADR 0003](../../../../design/decisions/0003-typed-trace-dtos.md)
- [ADR 0006](../../../../design/decisions/0006-no-runtime-service-locator.md)
- [ADR 0008](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Architecture style](../../../../developer-guide/documentation/architecture-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Backend guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Runtime master plan](../master-plan.md)
- [Completed Runtime 0001](0001-prepared-buffer-slot-identity.md)
- [Completed Runtime 0002](0002-prepared-memory-and-workspace-contracts.md)
- [Completed Runtime 0003](0003-run-state-and-runtime-resource-foundation.md)
- [Completed Runtime 0004](0004-prepared-executable-and-bound-invocation.md)
- [Completed Runtime 0005](0005-prepared-schedule-contract.md)
- [Completed Runtime 0006](0006-prepared-execution-aggregate.md)
- [Completed Runtime 0007](0007-representation-creation-and-residency-foundation.md)
- [Completed Runtime 0008](0008-prepared-buffer-transfer-and-materialization-schedule.md)
- [Completed Runtime 0009](0009-publication-and-result-schedule-steps.md)
- [Completed Runtime 0010](0010-prepared-runner-and-dynamic-execution.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Public API status](../../../../api/public-api.md)
- [Backend guide](../../../../backend-guide/writing-a-backend.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative. The audit may confirm or report the current boundary but
  cannot reinterpret or change it.
- Prepared artifacts are immutable reusable recipes. Every active complete logical run owns one
  distinct mutable `RunState` spanning all participating backends.
- Runtime owns logical per-run state, validity, lifecycle, scheduling, binding orchestration,
  publication coordinates, leases, and cleanup. Concrete backends own physical representation
  implementations and mechanics.
- Caller input resources are borrowed for the documented run/result lifetime. Internal buffer and
  workspace representations are run-owned and close with the state unless ownership has moved to
  the result lease.
- Compatibility, selection resolution, and physical-reference binding happen before hot
  traversal. Bound actions execute through direct references without graph, discovery, route,
  kernel, reflection, map, boxing, synchronization, or service-lookup work.
- Runtime does not depend on Compiler, Planning, Prepare, Engine, or concrete backends. It must not
  absorb their graph, orchestration, composition, discovery, lowering, or physical implementation
  responsibilities.
- Trace and Config are allowed dependencies only for stable typed contracts actually consumed.
  Their absence from the current run path must not be filled speculatively.
- Source and tests are primary evidence. Generated Javadocs prove rendered visibility and wording;
  completed tasks and explanatory documents are supporting evidence.
- A contradiction requiring Java, tests, Gradle, architecture, or cross-layer behavior changes
  changes the finding or verdict; it never expands this task's scope.

## Package impact

No Java package is added, changed, moved, or removed.

The audit reviews these existing Runtime packages:

- `io.github.pho001.synaptik.runtime.memory` — prepared-plan-local buffer/workspace identities and
  immutable final geometry;
- `io.github.pho001.synaptik.runtime.resource` — nominal backend-owned physical roles and immutable
  prepared representation creation descriptions;
- `io.github.pho001.synaptik.runtime.execution` — immutable prepared execution/executable/transfer
  recipes and per-run bound direct-reference actions;
- `io.github.pho001.synaptik.runtime.schedule` — immutable exact-plan ordered semantic occurrences;
  and
- `io.github.pho001.synaptik.runtime.run` — per-run binding, creation, validity, publication,
  result leasing, cleanup, and orchestration.

The audit must reject a root facade, catch-all lifecycle manager, generic resource registry,
service locator, backend adapter, or utility package unless current evidence instead requires a
blocking or architecture finding.

## Affected files

Always created or updated during execution — exactly four paths:

- add `docs/planning/modules/runtime/runtime-contract-closure-audit.md`
- finalize this task
- update `docs/planning/modules/runtime/master-plan.md`
- update `docs/planning/roadmap.md`

Conditionally permitted only for explanatory Markdown drift proved against current source/tests —
at most five paths:

- `docs/architecture/runtime-prepare-backend-boundary.md`
- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/backend-guide/writing-a-backend.md`
- `docs/glossary.md`

Review without modification: `AGENTS.md`; `ARCHITECTURE.md`; all ADRs; all other architecture
documents; every Runtime production/test file and generated Javadoc page; Runtime Gradle;
completed Runtime 0001-0010 tasks; Prepare, Backend Contract, Trace, Config, Engine, Compiler,
Planning, Model, and concrete-backend source, tests, master plans, and current public boundaries;
architecture, conformance, and integration tests; all other modules.

## Maximum scope

This documentation-only audit may create or modify at most the nine paths listed above. The four
always-affected paths are required. A conditional path may change only to correct a link, term,
example, status explanation, or current-versus-planned statement already established by current
source/tests and architecture.

If a finding needs Java/Javadoc source, tests, Gradle, `ARCHITECTURE.md`, an ADR, an architecture
test, another planning file, a tenth path, or new behavior, do not expand scope. Record
`BLOCKING_GAP` or `ARCHITECTURE_DECISION_REQUIRED` with evidence, owner, and required follow-up.

## Required audit artifact

The durable artifact must contain exactly these top-level sections, with focused subsections and
tables as needed:

1. `Executive conclusion and closure verdict`
2. `Authority, scope, and method`
3. `Architecture and module-boundary assessment`
4. `Production, test, public-surface, package, and generated-Javadoc inventory`
5. `Prepared artifact and exact-plan contract assessment`
6. `Run-state creation, ownership, and cold-binding assessment`
7. `Representation, validity, transfer, and aliasing assessment`
8. `Executable traversal, publication, result lease, and cleanup assessment`
9. `Empty, repeated, concurrent, and failure-path assessment`
10. `Hot-path performance-boundary assessment`
11. `Public and package-private API cohesion assessment`
12. `Package and dependency assessment`
13. `Documentation, examples, glossary, and completed-history consistency`
14. `Validation coverage and checkpoint applicability`
15. `Deferred and downstream-owned work`
16. `Findings, severity, and disposition`
17. `Checkpoint evidence and Runtime-milestone decision`

### Closure verdict vocabulary

The first section records exactly one verdict:

- `CLOSED` — no blocking Runtime contract, documentation, dependency, validation, or architecture
  conflict remains. Explicit downstream work may remain deferred when its owner and non-blocking
  rationale are recorded.
- `BLOCKING_GAP` — the audit is complete, but at least one Runtime-owned gap prevents milestone
  closure. Every gap names evidence, impact, owner, dependency, and the bounded future work; 0011
  does not implement or specify that work.
- `ARCHITECTURE_DECISION_REQUIRED` — evidence conflicts with authoritative architecture or needs a
  new ownership/dependency decision. Stop without editing architecture.

Findings use exactly these labels:

- `BLOCKING`
- `NON_BLOCKING_DEFERRED`
- `DOCUMENTATION_DRIFT`
- `NO_CHANGE_CONFIRMED`

An architecture conflict is recorded through the global `ARCHITECTURE_DECISION_REQUIRED` verdict,
not disguised as documentation drift.

### Required evidence matrix

The artifact contains one matrix with exactly these columns:

| Area | Current contract | Primary evidence | Required invariant or question | Finding label | Disposition |
|---|---|---|---|---|---|

It contains at least one row for each area below:

- production, test, package, public-surface, generated-Javadoc, and task-history inventories;
- prepared memory/execution immutability and exact-plan reference identity;
- nominal buffer/workspace roles and prepared-versus-run identity separation;
- public declaration necessity/minimality and package-private implementation visibility;
- validation/failure ordering, immutable snapshots, equality, exact-reference retention, and
  ownership transfer;
- dense caller-input binding and all-or-cleaned `RunState` creation;
- borrowed input and run-owned buffer/workspace lifetime;
- per-buffer multi-copy residency/validity and workspace exclusion from validity;
- cold executable, transfer, and publication binding before traversal;
- first-only representation creation and rollback;
- transfer/materialization no-op, success-only validity, and source/destination behavior;
- executable read/write overlap and conservative output-validity transitions;
- dense publication suffix, valid-copy requirement, result-position aliasing, and no fallback;
- whole-state `RunResult` lease, result-value non-exposure, and cleanup ownership transfer;
- close idempotence, reverse cleanup, primary/suppressed failure preservation, and closed-state
  failures;
- empty plans/schedules/results and repeated/concurrent-run isolation;
- direct-reference bound hot path and every prohibited lookup/search/reflection/allocation class;
- package cohesion and exact Gradle/import dependency surface;
- Model, Config, Backend Contract, Trace, Compiler, Planning, Prepare, Engine, and concrete-backend
  boundary compliance;
- generated Javadocs, Runtime/Public API examples, backend guide, and glossary wording;
- repository-wide, architecture, conformance, and integration checkpoint applicability;
- public result-value access and Engine ownership;
- Prepare 0003 translation/orchestration;
- concrete backend realization/execution;
- Trace run payloads, Config run/publication policy, and performance tuning; and
- completed Runtime 0001-0010 history plus task/master-plan/roadmap status.

Every row names source, tests, generated documentation, or an exact inspection command. Completed
task prose alone is insufficient. The matrix must explicitly answer:

1. Is every current public declaration necessary, sufficient, and cohesive, and may every current
   package-private declaration remain internal?
2. Do exact reference-identity and nominal separation promises survive every construction and
   binding boundary?
3. Does every successful, failed, empty, repeated, and concurrent run preserve the documented
   lifetime, validity, publication, lease, and cleanup invariants?
4. Does bound traversal remain free of every prohibited hot-path mechanism?
5. Are all missing higher-layer capabilities assigned to an explicit non-Runtime owner without
   leaving a Runtime-owned semantic gap?

## Acceptance criteria

- The durable artifact contains all 17 required top-level sections, exactly one closure verdict,
  only the permitted finding labels, and the complete evidence matrix.
- Every Runtime production/test file, test suite, package, public and package-private declaration,
  and generated public Javadoc page/index boundary is included in a reproducible inventory.
- Every completed Runtime 0001-0010 task remains present and unchanged. Its decisions,
  limitations, evidence, no-change conclusions, and follow-ups are represented in the audit
  without treating task prose as primary implementation proof.
- The audit records the exact prepared/run identity model: model/planning identities, prepared
  slots/recipes, physical representations, and per-run bindings remain nominally separate; exact
  plan references and documented element references are preserved.
- The complete `PreparedMemoryPlan`/`PreparedExecution` through `RunResult.close()` lifecycle is
  traced against source and tests, including validation order, ownership transfer, rollback,
  cleanup order, idempotence, and suppressed failures.
- Borrowed caller inputs and run-owned representations have explicit lifetimes. Reusable recipes
  retain no mutable run state, and repeated/concurrent runs have isolated state, validity, bound
  actions, resources, leases, and cleanup.
- Multi-copy validity, creation, transfer, executable write invalidation, publication, aliasing,
  and result leasing agree across source, tests, Javadocs, and examples without hidden coherence
  or value-access claims.
- Empty and repeated runs plus all tested failure paths are audited explicitly.
- The cold/hot split is checked with source and existing hot-path tests. A `CLOSED` verdict
  confirms no graph inspection, service/backend discovery, route/kernel search, reflection, map
  or boxing lookup, synchronization, or avoidable per-occurrence allocation in bound traversal.
- Public/package-private API and five-package cohesion are assessed declaration by declaration;
  no facade, generic manager, registry, or premature public helper is inferred.
- Exact Gradle visibility and imports comply with architecture. Runtime has no Compiler, Planning,
  Prepare, Engine, or concrete-backend dependency and no concrete backend implementation leaks.
- Generated Javadocs and Runtime/Public API, examples, backend guide, glossary, and planning status
  distinguish current behavior from downstream plans. Permitted Markdown drift is corrected and
  recorded; Java/Javadoc-source drift cannot be repaired within 0011.
- The audit explicitly records why repository-wide and architecture validation are triggered by
  this capability checkpoint. It records backend-conformance and integration as not applicable
  unless current executable consumers provide a concrete reason to trigger them.
- Public result-value access, Prepare 0003, Engine composition, concrete backends, Trace run
  payloads, Config run/publication policy, and tuning each have an evidence-backed owner and
  closure disposition.
- Prepare 0003 and every later task remain Draft without a detailed specification. No Runtime
  0012 row/specification or other future detailed task is created.
- Status synchronization follows the exact verdict rules in Scope. The unexecuted audit remains
  Ready; only its future execution may record a final verdict and completion status.
- The one combined checkpoint, generated-Javadoc inspection, Markdown links/anchors/fences/unique
  anchors/final-newline/trailing-whitespace checks, exact scope/status/order/history checks, and
  `git diff --check` pass.
- No Java, test, Gradle, `ARCHITECTURE.md`, ADR, architecture-test, conformance, integration,
  Prepare, Engine, backend, tracing, configuration, or other executable path changes.
- The task records local decisions, known limitations, exact future commands/results, audit
  context identity, findings, milestone disposition, and canonical completion summary.

## Tests / validation

Validation tier: capability checkpoint. This is a clean-context documentation-only audit. Runtime
0010's successful final focused evidence (17 suites and 143 tests) may be reused during source/test
review. After the artifact and permitted Markdown corrections are stable, run exactly one combined
capability checkpoint:

```bash
./gradlew test :testing:architecture-tests:test :modules:runtime:javadoc
```

This single command intentionally triggers repository-wide and architecture validation because
0011 is a module capability checkpoint, and renders Runtime Javadocs for final inspection. Gradle
may deduplicate tasks. Record executed/up-to-date/skipped task counts and repository, Runtime, and
architecture XML test totals where available. Do not run `:modules:runtime:test` separately or
repeat a successful suite merely to restate evidence.

Do not run backend-conformance or integration suites unless the audit identifies a current
concrete backend or end-to-end Engine execution consumer that makes one applicable. Otherwise
record both as `NON_BLOCKING_DEFERRED` or not triggered with the exact ownership rationale.

Run final documentation and working-tree validation:

```bash
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
rg --files docs/planning/modules/runtime/tasks | sort
rg -n '^Ready$|\| Ready \|' docs/planning
```

Also inspect Runtime package summaries, every public type page, `allclasses-index.html`, and
`overview-tree.html`; compare generated visibility with source declarations and package-private
types. Validate links, anchors, duplicate/unique anchors, fences, final newlines, and trailing
whitespace for all changed Markdown.

For exact scope, compare the tracked/untracked list with the four required and five conditional
paths. Reject any path under Java source/tests, Gradle, `ARCHITECTURE.md`,
`docs/design/decisions`, or `testing`. At task start there must be exactly one Ready task heading
and one Ready Runtime master-plan row: Runtime 0011. Preserve Runtime 0001-0010 and confirm no
Runtime 0012, Prepare 0003 specification, or other later Ready task was created.

## Documentation-focused execution

This task is itself the required clean documentation-focused context. No second documentation
agent/pass is required or permitted merely to repeat it. The same context owns the audit artifact,
permitted Markdown corrections, Javadoc inspection, checkpoint, validation, and completion
evidence because no executable behavior changes.

If execution discovers Java/Javadoc-source, test, Gradle, architecture, backend, Prepare, Engine,
Trace, Config, or other executable work, record a finding and stop or close according to the
verdict rules. Do not spawn an implementation task inside 0011.

## Documentation impact

- Add one durable Runtime closure-audit artifact with source-backed inventories, lifecycle and
  ownership analysis, findings, checkpoint evidence, and final milestone disposition.
- Finalize this task's evidence, completion summary, and status without deleting or rewriting any
  completed Runtime task history.
- Synchronize only the Runtime master plan and roadmap as required by the verdict.
- Conditionally correct only the five explicitly permitted explanatory Markdown paths when the
  audit proves current-behavior drift. Every correction must be named in the findings matrix.
- Review all generated Runtime Javadocs for semantics and visibility but do not edit Java/Javadoc
  source. Any Javadoc-source defect becomes a separate finding.
- Review Runtime/Public APIs, examples, backend guidance, glossary terms, and focused boundary
  wording for accurate current-versus-planned status. Do not turn the audit into a tutorial or
  future API proposal.
- Record reasoned no-change conclusions for every reviewed documentation family that remains
  accurate, including architecture authority and adjacent module boundaries.

## Dependencies

- Complete Runtime 0001-0010, including source, tests, generated Javadocs, explanatory
  documentation, and recorded independent documentation evidence.
- Accepted ADR 0011 prepared-versus-per-run resource ownership and cold-binding decision.
- Complete Prepare 0001-0002 analysis, assignment, and finalization handoff; Prepare 0003 remains
  a downstream Draft orchestration task.
- Complete Backend Contract identity/requirement foundation and current backend-owned physical
  representation boundary.
- Current Model, Compiler, Planning, Trace, Config, and Engine ownership boundaries without
  importing their planned surfaces into Runtime.

## Follow-up tasks

- Public result-value access and public lifecycle composition remain with a future Engine/result
  owner. This audit neither designs nor specifies that API.
- Prepare 0003 remains Draft and owns complete translation/orchestration, coverage validation,
  schedule construction, and final prepared-execution assembly once it becomes the selected
  frontier. This task does not make it Ready or create its specification.
- Concrete backends own physical representation implementations, transfers, bound invocations,
  lowering, routes/kernels, and actual execution; applicable conformance work waits for them.
- Trace run payloads wait for a stable Runtime/Engine producer contract. Config run/publication
  policy waits for a concrete consumer. Performance measurement and tuning remain outside the
  Runtime hot path.
- A blocking finding requires separate future planning after 0011. This task creates no Runtime
  0012 row or specification and does not select the next global frontier.

## Architecture impact

Expected impact: None.

The task audits current implementation against the architecture contract and may correct only
explanatory Markdown drift. If evidence requires a new public ownership rule, dependency,
lifecycle allocation, backend collaboration, configuration/trace contract, or exception, stop
and return `ARCHITECTURE_DECISION_REQUIRED` without editing architecture.

## Implementation prompt

Use this prompt in one separate clean documentation-focused task/thread:

```text
You are a clean-context documentation and planning agent working in the Synaptik repository. Do
not spawn a redundant documentation agent. Do not commit or push. Do not implement or modify Java,
tests, Gradle, ARCHITECTURE.md, ADRs, architecture tests, backend-conformance/integration tests,
Prepare, Engine, backend, Trace, Config, or tuning behavior.

Read AGENTS.md, ARCHITECTURE.md, the focused lifecycle/module/dependency/runtime-prepare-backend/
tracing/performance architecture documents and relevant accepted ADRs, documentation rules and
General/Planning/Architecture/API/Backend Guide/Example profiles, planning guide and roadmap,
Runtime master plan and completed tasks 0001-0010, current Runtime source/tests/generated
Javadocs, Runtime/Public APIs, backend guide, glossary, architecture/dependency tests read-only,
and directly relevant Prepare/Backend Contract/Trace/Config/Engine/Compiler/Planning/backend
master plans and boundaries. Read task 0011 in full. Inspect initial git status and preserve
unrelated changes.

Execute Runtime 0011 exactly as a documentation-only closure audit. Create its durable artifact,
use only its verdict/finding vocabulary, fill every required evidence row, and audit source,
tests, public/package-private surface, lifecycle, ownership, validity, publication, cleanup,
hot-path, dependencies, documentation, history, deferred ownership, and validation applicability.
Run the one final combined checkpoint and all documentation/scope/status checks. Synchronize task,
Runtime master plan, and roadmap only as the verdict permits. Leave Prepare 0003 and all later
tasks Draft without specifications; do not create Runtime 0012. If a finding needs executable or
architecture work, record it for separate future planning rather than implementing it. Return the
canonical completion summary and exact Status line.
```

## Local decisions

- Runtime 0011 is the sole next unfinished Runtime task because 0001-0010 now span the selected
  prepared-execution and dynamic per-run lifecycle. The next responsible action is an independent
  closure audit, not another Runtime feature.
- The repository's existing Planning and Model closure-audit convention is reused exactly. A new
  broader severity framework or `CLOSED_WITH_DEFERRED_WORK` verdict is unnecessary because
  `CLOSED` already permits explicitly owned non-blocking deferrals.
- The durable artifact lives beside the Runtime master plan, matching repository precedent and
  keeping detailed evidence out of the concise status narrative.
- The task is documentation-only. Repairing a finding would mix implementation with the exit gate
  and violate the authorized scope.
- Runtime 0010's 17-suite/143-test result is valid prior focused evidence. The final command is one
  combined repository/architecture/Runtime-Javadoc checkpoint; there is no separate repeated
  Runtime test command.
- Repository and architecture validation are concretely triggered because this is the Runtime
  capability checkpoint and audits accumulated dependency/hot-path rules. Backend conformance and
  integration are not currently triggered because no concrete backend execution or Engine end-to-
  end path exists; execution must re-evaluate this against current evidence.
- No provisional closure verdict is asserted by this Ready specification. The execution context
  must derive exactly one verdict from primary evidence.
- A blocking finding is durable audit output, not authorization to create Runtime 0012 or modify
  another module's frontier.

## Known limitations

- `RunResult` currently leases Runtime state and exposes result count, but not public result values.
- Prepare 0003 public translation/orchestration and complete prepared-execution assembly remain
  Draft.
- No Engine composition root currently exposes a public compile/prepare/run lifecycle.
- No concrete backend currently supplies end-to-end physical representation creation, transfer,
  invocation, and publication execution through this Runtime path.
- Trace has no current run-payload family consumed by Runtime; Runtime 0010 emits none.
- Config run/publication policy remains Draft and is not an input to the current runner.
- Runtime contains contracts and hot-path structural checks, not measured backend performance or
  tuning policy.
- Generated Javadoc success establishes rendering and visibility only; semantic accuracy still
  requires inspection against source/tests.
- Closure does not freeze future consumer-driven extensions. It establishes only that no current
  Runtime-owned gap blocks downstream lifecycle work.

## Rejected alternatives

- Implementing a discovered defect inside 0011 was rejected because it would make the closure gate
  self-repairing and violate the documentation-only authorization.
- A `CLOSED_WITH_DEFERRED_WORK` verdict was rejected because the repository's established `CLOSED`
  definition already permits explicitly owned non-blocking deferrals.
- Creating Runtime 0012 preemptively was rejected because no finding exists before audit execution
  and future task selection belongs to a separate planning decision.
- Running Runtime tests separately before or after the combined checkpoint was rejected because
  valid focused 0010 evidence may be reused and the combined command already selects Runtime tests.
- Triggering backend-conformance or integration suites unconditionally was rejected because there
  is no current concrete backend execution or Engine end-to-end consumer; applicability remains an
  explicit audit question.
- Expanding the durable artifact into a repository-wide closure framework was rejected in favor of
  the existing small Runtime-specific closure-audit convention.

## Validation evidence

- Planning context inspected a clean initial working tree, the governing architecture and
  documentation/planning rules, focused ADRs and architecture explanations, completed Runtime
  0001-0010 history, current Runtime source/test/generated-Javadoc inventories, API/guide/glossary
  wording, dependency/architecture-test configuration, adjacent module boundaries, and existing
  Planning/Model closure-audit precedents.
- Clean execution/audit context `019fbee0-00ac-7530-8308-982306a7a9f8` started with exactly the
  Ready 0011 task, Runtime master
  plan, and roadmap as the three planning-stage changes. It preserved unrelated/completed history
  and created the sole durable artifact
  [`runtime-contract-closure-audit.md`](../runtime-contract-closure-audit.md).
- The artifact records verdict `BLOCKING_GAP`: Runtime cleanup failure identity, stale review-only
  architecture status, and missing Runtime architecture-test enforcement are `BLOCKING` findings.
  No architecture decision is required and no forbidden repair was attempted.
- The one combined capability checkpoint passed:

  ```bash
  ./gradlew test :testing:architecture-tests:test :modules:runtime:javadoc
  ```

  Gradle reported `BUILD SUCCESSFUL`, 53 actionable tasks, 8 executed, and 45 up-to-date. Current
  JUnit XML contains 205 suites and 1,530 tests with zero failures, errors, or skips: Runtime 17
  suites/143 tests, architecture 3 suites/3 tests, Model 127/1,031, Compiler 31/208, Planning
  9/68, Prepare 7/22, Backend Contract 4/22, Config 4/17, and Trace 3/16.
- Generated Runtime Javadocs contain all five package summaries, 34 public type pages,
  `allclasses-index.html`, and `overview-tree.html`. Inspection confirmed source/rendered public
  visibility agreement and package-private implementation exclusion. The rendered `RunState`
  cleanup contract also makes the source contradiction in `RUNTIME-CLEANUP-001` explicit.
- Backend-conformance and integration suites were not independently triggered: no concrete
  backend implements the Runtime path and Engine exposes no end-to-end execution consumer. Their
  root `test` tasks are `NO-SOURCE`; no separate suite was run.
- The specified no-argument `python3 /tmp/validate_synaptik_markdown.py` command exited
  successfully but reported `validated 0 Markdown files`; the explicit four-path invocation then
  reported `validated 4 Markdown files` and checked local links/anchors, unique effective anchors,
  balanced fences, final newlines, and trailing whitespace for every changed Markdown file.
- `git diff --check` passed with no output. The tracked/untracked union contains exactly the four
  authorized paths. `git status --short` contains only two modified planning files and the two
  expected untracked audit/task files.
- Runtime task inventory ends at 0011. `rg -n '^Ready$|\| Ready \|' docs/planning` returned no
  matches after synchronization. File/status checks confirm task/master-plan 0011 are Complete,
  the roadmap Runtime area remains in progress under `BLOCKING_GAP`, Prepare 0003 is Draft without
  a specification, no Runtime 0012 exists, and Runtime 0001-0010 have no diff.

## Implementation notes

- Added only the durable documentation-only audit artifact and synchronized this task, Runtime
  master plan, and roadmap according to `BLOCKING_GAP`.
- Inventoried 25 production files, 17 test files/suites, five packages, 34 rendered public type
  pages, the sole package-private top-level declaration, public/package-private members, exact
  imports/dependencies, and completed Runtime 0001-0010 history.
- Recorded three blocking findings for separate future planning without creating Runtime 0012:
  shared-throwable cleanup failure, stale general architecture status prose, and absent Runtime
  architecture-test enforcement.
- Left Prepare 0003 and every later task Draft without a detailed specification. No Java,
  Javadoc source, test, Gradle, architecture, ADR, conformance, integration, backend, Engine,
  Prepare, Trace, Config, or tuning behavior changed.

## Completion summary

Completed changes: created the durable Runtime contract closure audit with verdict
`BLOCKING_GAP`, complete inventories/evidence matrix, deferred-owner classifications, and
verdict-permitted task/master-plan/roadmap status synchronization.

Files changed or created: exactly four paths—this task,
`docs/planning/modules/runtime/runtime-contract-closure-audit.md`,
`docs/planning/modules/runtime/master-plan.md`, and `docs/planning/roadmap.md`.

Tests or validation performed: the one combined repository/architecture/Runtime-Javadoc command
passed 205 suites and 1,530 tests with zero failures, errors, or skips; generated Runtime pages,
four-file Markdown links/anchors/fences/newlines/whitespace, exact scope/status/order/history,
Ready-task absence, later-specification absence, and `git diff --check` passed.

Unresolved issues: `RUNTIME-CLEANUP-001`, `DOCUMENTATION-STATUS-001`, and
`ARCHITECTURE-ENFORCEMENT-001` remain blocking findings; no architecture decision is required.

Required follow-up: separately plan the bounded Runtime cleanup repair, stale general
architecture-status correction, and Runtime architecture-test enforcement. Keep Prepare 0003 and
all later tasks Draft until a separate frontier decision; do not infer Runtime 0012 from this
audit.

Status: Complete
