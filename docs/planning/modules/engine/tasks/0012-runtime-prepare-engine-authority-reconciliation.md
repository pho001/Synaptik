# Task 0012: [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), and [Engine](../../../../glossary.md#term-engine) Authority Reconciliation

> **Editorial historical reading key.**
>
> Read this as a completed documentation-authority repair, not a specification of today's entire
> Engine surface. Its CPU-only composition and planned-registration wording records the reconciliation
> boundary then; explicit CPU/Metal construction is now documented in the [current composition
> contract](../../../../reference/engine/composition.md#current-explicit-cpu-and-metal-composition).
>
> Engine is the caller-facing orchestrator of compilation, preparation, and execution; it does not
> replace Compiler's logical semantics or backend implementation responsibility. Shared Prepare hands
> validated recipes and acquired persistent resources to Runtime. Runtime's prepared owner retains the
> exact plan and schedule, while its runner binds direct actions to one isolated run state. Repeated
> schedule occurrences are not additional resource owners. The [current aggregate and run
> orchestration](../../../../reference/runtime/orchestration.md#current-aggregate-and-run-orchestration)
> distinguishes persistent preparation ownership from result-owned completed state. The task's tests
> and review outcomes remain historical evidence.

## Status

Complete

Frontier verification: [Engine](../../../../glossary.md#term-engine) 0011 and [Compiler](../../../../glossary.md#term-compiler) 0006B8 are Complete, no other task is [`Ready`](../../../../glossary.md#term-ready) or
`In progress`, and the user explicitly authorized sequential remediation of the first confirmed
remaining-drift cluster after [commit](../../../../glossary.md#term-commit) `0276682e`. This task is the sole active frontier; its
dependencies are satisfied.

## Change class

Class C — this documentation-only repair changes the authoritative root [lifecycle](../../../../glossary.md#lifecycle) and the
incorporated [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine) contract across the public [facade](../../../../glossary.md#term-facade) and Runtime/Prepare
boundary. It changes no Java behavior or [API](../../../../glossary.md#term-application-programming-interface-api), but requires a clean documentation implementation
context and an independent clean review context.

## Goal

Make the architecture authority and its focused explanations describe the implemented
[owner](../../../../glossary.md#term-owner)-bound [Engine](../../../../glossary.md#term-engine) [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare)/[run](../../../../glossary.md#term-run) [facade](../../../../glossary.md#term-facade) and [Runtime](../../../../glossary.md#term-runtime) schedule-step/executable [model](../../../../glossary.md#term-model) exactly,
without presenting nonexistent APIs or the removed `PreparedUnit` concept as current.

## [Scope](../../../../glossary.md#term-scope)

- Replace the root's nonexistent `CompiledGraph.compile(...)`, `CompileConfig`, `PrepareConfig`,
  `PreparedExecution.run(...)`, and `RunOptions` example with the current owner-bound ordinary
  lifecycle: <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>`Engine.standard()`, `engine.compile(...)`, `engine.prepare(...)`, and
  `engine.run(...)`, including explicit closeable [ownership](../../../../glossary.md#term-ownership-meanings) where the example needs it.
- Reconcile the incorporated contract's Runtime, Prepare lifecycle, run lifecycle, and Engine
  sections with current source: <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run))</span>[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) retains one exact memory plan and schedule
  and owns persistent resources; [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule) orders optional creation, executable,
  transfer, and publication step occurrences; <span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>[`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner) owns synchronous run
  orchestration; there is no distinct <span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous))</span>`PreparedUnit`.
- State that current ordinary composition is one fresh fixed [CPU](../../../../glossary.md#term-cpu) integration through
  `Engine.standard()` and current advanced composition takes ownership of one supported CPU
  integration. Preserve generic explicit backend registration only as planned composition-time
  work, never as a current builder, discovery, service-locator, or hot-path mechanism.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Correct only the corresponding stale statements in `lifecycle.md` and
  `module-boundaries.md`.
- Synchronize this task, the [Engine](../../../../glossary.md#term-engine) master plan, and the roadmap after implementation and review.

## Non-goals

- No Java, test, Javadoc, Gradle, dependency, binary/source API, behavior, numerical, lifecycle,
  ownership, resource, concurrency, backend, tuning, or execution change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency); [backend glossary](../../../../glossary.md#backend))</span>
- No `SynaptikEngine.builder()`, generic registration API, mixed-owner schedule, Metal/CUDA
  Engine adapter, service locator, reflective discovery, or new public configuration surface.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [Engine glossary](../../../../glossary.md#term-engine); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- No change to [Compiler](../../../../glossary.md#term-compiler)/[autograd](../../../../glossary.md#autograd) authority, [compile](../../../../glossary.md#compile) semantics, `CompileArtifacts` shape or
  projection, concrete-backend behavior, tuning contracts, or any existing tuning ambiguity.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- No architecture decision record (ADR), architecture/conformance/integration test, glossary,
  Runtime API, Public API, other-module documentation, or broad status cleanup. If comparison
  evidence shows one of those needs correction, stop and report the required scope expansion.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#adr-ci-and-qualification))</span><span data-glossary-navigation="true"> ([architecture decision record glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test); [Runtime glossary](../../../../glossary.md#term-runtime); [module glossary](../../../../glossary.md#term-module); [scope glossary](../../../../glossary.md#term-scope))</span>

## Contracts

- [`ARCHITECTURE.md` heading `Authority, incorporation, and precedence`](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  and [`Scope-indexed normative contracts`](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
  — the root and its incorporated Runtime/Prepare/Engine contract are the only authority changed.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>
- [`ARCHITECTURE.md` heading `Core lifecycle`](../../../../../ARCHITECTURE.md#core-lifecycle) —
  reconcile its public example with current owner-bound Engine calls while preserving the inward
  compile/prepare/run flow.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run))</span>
- [`ARCHITECTURE.md` headings `Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing)
  and [`Dependency rules`](../../../../../ARCHITECTURE.md#dependency-rules) — Runtime owns prepared
  schedules/run state, Prepare owns the shared transition, and Engine owns explicit composition
  and public lifecycle.<span data-glossary-navigation="true"> ([run state glossary](../../../../glossary.md#run-state--runstate); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [`runtime-prepare-engine.md` headings `modules/runtime`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
  [`Prepare lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [`Run lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle),
  and [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — replace only stale API/type/ownership claims with the implemented model.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [model glossary](../../../../glossary.md#term-model))</span>

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Execution may change exactly these seven paths:

- `ARCHITECTURE.md` — `Core lifecycle` public example only.
- `docs/architecture/contracts/runtime-prepare-engine.md` — `modules/runtime`, `Prepare lifecycle`,
  `Run lifecycle`, `modules/engine`, and only directly necessary registration wording under the
  existing discovery prohibition.
- `docs/architecture/lifecycle.md` — [prepare](../../../../glossary.md#prepare)/[run](../../../../glossary.md#term-run) flows and current [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle) explanation.
- `docs/architecture/module-boundaries.md` — [Runtime](../../../../glossary.md#term-runtime) and Engine current-boundary summaries.
- this task brief — status and compact result evidence only.
- `docs/planning/modules/engine/master-plan.md` — task row, frontier, and remediation sequence only.
- `docs/planning/roadmap.md` — Engine row and repository frontier only.

Read-only source and focused evidence:

- `Engine`, `CompiledGraph`, Engine [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), `AdvancedEngine`, and
  `AdvancedPreparedExecution` in `modules/engine`.
- Runtime `PreparedExecution`, [`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule), [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), and
  [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner), plus Prepare [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation).
- `EngineTypedPublicShapeTest`, `EngineTypedLifecycleTest`, `EngineStandardCompositionTest`,
  `PreparedExecutionTest`, `PreparedScheduleTest`, `PreparedExecutionRunnerTest`, and
  `GraphPreparationTest`.
- `docs/api/runtime-api.md` headings `Current ordinary Engine boundary`, `Current aggregate and
  run orchestration`, and `Current prepared runner`; and `docs/api/public-api.md` heading
  `Current ordinary and advanced CPU lifecycle`. These and targeted `PreparedUnit`, Engine,
  prepared-schedule, and registration glossary entries are comparison references, not edit targets.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>

## Acceptance criteria

- The root example uses only current ordinary public types and calls, makes Engine the owner of
  compile, prepare, and run, and does not name the five nonexistent APIs listed in Scope.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run); [Scope glossary](../../../../glossary.md#term-scope))</span>
- The incorporated contract and explanations consistently state that [Runtime](../../../../glossary.md#term-runtime)
  [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) retains its exact plan and schedule and uniquely owns its persistent
  resources, while the stateless [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner) acquires the lease and performs a
  synchronous run with one isolated <span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [synchronous glossary](../../../../glossary.md#term-synchronous))</span>[`RunState`](../../../../glossary.md#run-state--runstate).
- The current schedule [model](../../../../glossary.md#term-model) is expressed as ordered `PreparedSchedule.Step` occurrences using
  the current representation-creation, executable, buffer-transfer, and publication recipes. An
  optional creation occurrence is first, publications form the dense final suffix, and repeated
  executable or transfer occurrences do not create resource ownership. No edited document claims
  that <span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [publication glossary](../../../../glossary.md#term-publication); [occurrence glossary](../../../../glossary.md#term-occurrence); [ownership glossary](../../../../glossary.md#term-ownership))</span>`PreparedUnit` exists, is owned by Runtime, or is created by [Prepare](../../../../glossary.md#prepare).
- The [Engine](../../../../glossary.md#term-engine) contract distinguishes current fixed [CPU](../../../../glossary.md#term-cpu) `Engine.standard()` and explicit
  `AdvancedEngine.takeOwnership(...)` composition from planned generic explicit registration.
  Planned registration is labeled as planned and does not invent a callable builder API.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Existing service-locator and reflective-discovery prohibitions remain intact. No edit changes
  dependency direction, Compiler/autograd, <span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd))</span>`CompileArtifacts`, tuning, backend support, or
  execution behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- The read-only Runtime and Public API references remain accurate after source/test comparison.
  The result records reasoned no-change conclusions for them, glossary, Java/Javadoc, Gradle,
  ADRs, architecture/conformance/integration tests, other modules, and repository-wide tests.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>
- The callable root example follows the General, Architecture, and Example profiles: it is
  current, bounded to what it proves, and states the important ownership/cleanup constraint.
- Only the seven allowed paths change; task/master/roadmap statuses remain synchronized; local
  links and anchors resolve, headings are unique, fences are balanced, files have final newlines
  and no trailing whitespace, and `git diff --check` passes.

## [Validation](../../../../glossary.md#term-validation)

The clean documentation implementation context runs the focused executable evidence once:

```bash
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.api.EngineTypedPublicShapeTest \
  --tests io.github.pho001.synaptik.engine.EngineTypedLifecycleTest \
  --tests io.github.pho001.synaptik.engine.EngineStandardCompositionTest
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.execution.PreparedExecutionTest \
  --tests io.github.pho001.synaptik.runtime.schedule.PreparedScheduleTest \
  --tests io.github.pho001.synaptik.runtime.run.PreparedExecutionRunnerTest
./gradlew :modules:prepare:test \
  --tests io.github.pho001.synaptik.prepare.GraphPreparationTest
```

After final documentation and [planning](../../../../glossary.md#term-planning) edits:

```bash
python3 /tmp/validate_synaptik_markdown.py \
  ARCHITECTURE.md docs/architecture/contracts/runtime-prepare-engine.md \
  docs/architecture/lifecycle.md docs/architecture/module-boundaries.md \
  docs/planning/modules/engine/tasks/0012-runtime-prepare-engine-authority-reconciliation.md \
  docs/planning/modules/engine/master-plan.md docs/planning/roadmap.md
rg -n 'CompiledGraph\.compile|CompileConfig|PrepareConfig|RunOptions|SynaptikEngine|PreparedUnit|PreparedExecution\.run' \
  ARCHITECTURE.md docs/architecture/contracts/runtime-prepare-engine.md \
  docs/architecture/lifecycle.md docs/architecture/module-boundaries.md
rg -n 'Backends are registered explicitly|Backends must be registered explicitly' \
  docs/architecture/contracts/runtime-prepare-engine.md
git diff --name-only -- '*.java' '*.gradle' '*.gradle.kts'
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --check
```

The two stale-text searches and executable/build-file diff command must produce no output. The
path inventory must contain exactly the seven allowed paths. If the temporary Markdown validator
is absent, create an equivalent validator outside the repository for changed-file local links and
anchors, unique headings, balanced fences, final newlines, and trailing whitespace.

Repository-wide validation is deferred to [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes documentation to match
already-tested behavior and changes no executable code, [dependency](../../../../glossary.md#term-dependency), build configuration, or
[module](../../../../glossary.md#term-module) boundary. The independent review reuses successful focused evidence unless executable
behavior changes or the evidence is missing or stale.

## Dependencies and follow-up

- Engine 0011 and Compiler 0006B8 are Complete. This is the user-authorized third sequential
  drift-remediation step and executes alone against the clean <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Compiler glossary](../../../../glossary.md#term-compiler))</span>`0276682e` baseline.
- Completing this task authorizes no later Engine, Compiler, backend, or broad drift work. Reassess
  the remaining drift before selecting another frontier.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

## Documentation and review impact

- The implementation context follows the documentation rules and the General, Architecture,
  Planning, and Example profiles. It compares every changed claim with current source, focused
  tests, and the unchanged Runtime/Public API references, and changes no Java or Javadoc.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- A separate clean independent review context is mandatory because this Class C task changes the
  architecture root and an incorporated scoped contract. It inspects the final diff and evidence,
  checks terminology/examples/links and targeted glossary impact, and records its completion
  summary before the task may become Complete.

## Result

The implementation and mandatory independent Class C review reconciled the root [lifecycle](../../../../glossary.md#lifecycle)
example, incorporated [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine) contract, and two focused explanations with current
Engine, Runtime, and Prepare source and named tests. The result removes current `PreparedUnit`,
nonexistent configuration/[run](../../../../glossary.md#term-run) methods, and generic-builder claims while preserving exact
plan/schedule/resource [ownership](../../../../glossary.md#term-ownership), step ordering, repeated-[occurrence](../../../../glossary.md#term-occurrence) semantics, and the existing
discovery/service-locator prohibitions.

The review tightened the [admission](../../../../glossary.md#term-admission) term to “prepared-execution run [lease](../../../../glossary.md#term-lease)” and distinguished
[backend](../../../../glossary.md#backend)-owned executable/transfer bound actions from Runtime-owned [publication](../../../../glossary.md#term-publication) [binding](../../../../glossary.md#term-binding). The
implementation context's three Engine tests, three Runtime tests, and `GraphPreparationTest`
passed once with `BUILD SUCCESSFUL`; review reused that evidence because no executable file
changed. Markdown [validation](../../../../glossary.md#term-validation), stale-name and registration searches, Java/[Gradle](../../../../glossary.md#term-gradle) diff inventory,
the exact seven-path inventory, and `git diff --check` passed after the review repair.

Profiles: General, Architecture, [Planning](../../../../glossary.md#term-planning), and Example. Runtime/Public [API](../../../../glossary.md#term-application-programming-interface-api) guides and targeted
glossary entries remain accurate. Java/[Javadoc](../../../../glossary.md#term-javadoc), Gradle, ADRs, architecture/[conformance](../../../../glossary.md#term-conformance)/integration
tests, other [modules](../../../../glossary.md#term-module), and repository-wide tests need no change because this task reconciles
authority with already-tested behavior and changes no executable contract or [dependency](../../../../glossary.md#term-dependency); [CI](../../../../glossary.md#term-continuous-integration-ci)
retains repository-wide validation.

Status: Complete
