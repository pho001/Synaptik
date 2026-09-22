# Task 0012: Runtime, Prepare, and Engine Authority Reconciliation

## Status

Ready

Frontier verification: Engine 0011 and Compiler 0006B8 are Complete, no other task is `Ready` or
`In progress`, and the user explicitly authorized sequential remediation of the first confirmed
remaining-drift cluster after commit `0276682e`. This task is the sole active frontier; its
dependencies are satisfied.

## Change class

Class C — this documentation-only repair changes the authoritative root lifecycle and the
incorporated Runtime/Prepare/Engine contract across the public facade and Runtime/Prepare
boundary. It changes no Java behavior or API, but requires a clean documentation implementation
context and an independent clean review context.

## Goal

Make the architecture authority and its focused explanations describe the implemented
owner-bound Engine compile/prepare/run facade and Runtime schedule-step/executable model exactly,
without presenting nonexistent APIs or the removed `PreparedUnit` concept as current.

## Scope

- Replace the root's nonexistent `CompiledGraph.compile(...)`, `CompileConfig`, `PrepareConfig`,
  `PreparedExecution.run(...)`, and `RunOptions` example with the current owner-bound ordinary
  lifecycle: `Engine.standard()`, `engine.compile(...)`, `engine.prepare(...)`, and
  `engine.run(...)`, including explicit closeable ownership where the example needs it.
- Reconcile the incorporated contract's Runtime, Prepare lifecycle, run lifecycle, and Engine
  sections with current source: `PreparedExecution` retains one exact memory plan and schedule
  and owns persistent resources; `PreparedSchedule` orders optional creation, executable,
  transfer, and publication step occurrences; `PreparedExecutionRunner` owns synchronous run
  orchestration; there is no distinct `PreparedUnit`.
- State that current ordinary composition is one fresh fixed CPU integration through
  `Engine.standard()` and current advanced composition takes ownership of one supported CPU
  integration. Preserve generic explicit backend registration only as planned composition-time
  work, never as a current builder, discovery, service-locator, or hot-path mechanism.
- Correct only the corresponding stale statements in `lifecycle.md` and
  `module-boundaries.md`.
- Synchronize this task, the Engine master plan, and the roadmap after implementation and review.

## Non-goals

- No Java, test, Javadoc, Gradle, dependency, binary/source API, behavior, numerical, lifecycle,
  ownership, resource, concurrency, backend, tuning, or execution change.
- No `SynaptikEngine.builder()`, generic registration API, mixed-owner schedule, Metal/CUDA
  Engine adapter, service locator, reflective discovery, or new public configuration surface.
- No change to Compiler/autograd authority, compile semantics, `CompileArtifacts` shape or
  projection, concrete-backend behavior, tuning contracts, or any existing tuning ambiguity.
- No architecture decision record (ADR), architecture/conformance/integration test, glossary,
  Runtime API, Public API, other-module documentation, or broad status cleanup. If comparison
  evidence shows one of those needs correction, stop and report the required scope expansion.

## Contracts

- [`ARCHITECTURE.md` heading `Authority, incorporation, and precedence`](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  and [`Scope-indexed normative contracts`](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
  — the root and its incorporated Runtime/Prepare/Engine contract are the only authority changed.
- [`ARCHITECTURE.md` heading `Core lifecycle`](../../../../../ARCHITECTURE.md#core-lifecycle) —
  reconcile its public example with current owner-bound Engine calls while preserving the inward
  compile/prepare/run flow.
- [`ARCHITECTURE.md` headings `Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing)
  and [`Dependency rules`](../../../../../ARCHITECTURE.md#dependency-rules) — Runtime owns prepared
  schedules/run state, Prepare owns the shared transition, and Engine owns explicit composition
  and public lifecycle.
- [`runtime-prepare-engine.md` headings `modules/runtime`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
  [`Prepare lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [`Run lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle),
  and [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — replace only stale API/type/ownership claims with the implemented model.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Execution may change exactly these seven paths:

- `ARCHITECTURE.md` — `Core lifecycle` public example only.
- `docs/architecture/contracts/runtime-prepare-engine.md` — `modules/runtime`, `Prepare lifecycle`,
  `Run lifecycle`, `modules/engine`, and only directly necessary registration wording under the
  existing discovery prohibition.
- `docs/architecture/lifecycle.md` — prepare/run flows and current Engine lifecycle explanation.
- `docs/architecture/module-boundaries.md` — Runtime and Engine current-boundary summaries.
- this task brief — status and compact result evidence only.
- `docs/planning/modules/engine/master-plan.md` — task row, frontier, and remediation sequence only.
- `docs/planning/roadmap.md` — Engine row and repository frontier only.

Read-only source and focused evidence:

- `Engine`, `CompiledGraph`, Engine `PreparedExecution`, `AdvancedEngine`, and
  `AdvancedPreparedExecution` in `modules/engine`.
- Runtime `PreparedExecution`, `PreparedSchedule`, `PreparedExecutable`, and
  `PreparedExecutionRunner`, plus Prepare `GraphPreparation`.
- `EngineTypedPublicShapeTest`, `EngineTypedLifecycleTest`, `EngineStandardCompositionTest`,
  `PreparedExecutionTest`, `PreparedScheduleTest`, `PreparedExecutionRunnerTest`, and
  `GraphPreparationTest`.
- `docs/api/runtime-api.md` headings `Current ordinary Engine boundary`, `Current aggregate and
  run orchestration`, and `Current prepared runner`; and `docs/api/public-api.md` heading
  `Current ordinary and advanced CPU lifecycle`. These and targeted `PreparedUnit`, Engine,
  prepared-schedule, and registration glossary entries are comparison references, not edit targets.

## Acceptance criteria

- The root example uses only current ordinary public types and calls, makes Engine the owner of
  compile, prepare, and run, and does not name the five nonexistent APIs listed in Scope.
- The incorporated contract and explanations consistently state that Runtime
  `PreparedExecution` retains its exact plan and schedule and uniquely owns its persistent
  resources, while the stateless `PreparedExecutionRunner` acquires the lease and performs a
  synchronous run with one isolated `RunState`.
- The current schedule model is expressed as ordered `PreparedSchedule.Step` occurrences using
  the current representation-creation, executable, buffer-transfer, and publication recipes. An
  optional creation occurrence is first, publications form the dense final suffix, and repeated
  executable or transfer occurrences do not create resource ownership. No edited document claims
  that `PreparedUnit` exists, is owned by Runtime, or is created by Prepare.
- The Engine contract distinguishes current fixed CPU `Engine.standard()` and explicit
  `AdvancedEngine.takeOwnership(...)` composition from planned generic explicit registration.
  Planned registration is labeled as planned and does not invent a callable builder API.
- Existing service-locator and reflective-discovery prohibitions remain intact. No edit changes
  dependency direction, Compiler/autograd, `CompileArtifacts`, tuning, backend support, or
  execution behavior.
- The read-only Runtime and Public API references remain accurate after source/test comparison.
  The result records reasoned no-change conclusions for them, glossary, Java/Javadoc, Gradle,
  ADRs, architecture/conformance/integration tests, other modules, and repository-wide tests.
- The callable root example follows the General, Architecture, and Example profiles: it is
  current, bounded to what it proves, and states the important ownership/cleanup constraint.
- Only the seven allowed paths change; task/master/roadmap statuses remain synchronized; local
  links and anchors resolve, headings are unique, fences are balanced, files have final newlines
  and no trailing whitespace, and `git diff --check` passes.

## Validation

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

After final documentation and planning edits:

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

Repository-wide validation is deferred to CI because this task changes documentation to match
already-tested behavior and changes no executable code, dependency, build configuration, or
module boundary. The independent review reuses successful focused evidence unless executable
behavior changes or the evidence is missing or stale.

## Dependencies and follow-up

- Engine 0011 and Compiler 0006B8 are Complete. This is the user-authorized third sequential
  drift-remediation step and executes alone against the clean `0276682e` baseline.
- Completing this task authorizes no later Engine, Compiler, backend, or broad drift work. Reassess
  the remaining drift before selecting another frontier.

## Documentation and review impact

- The implementation context follows the documentation rules and the General, Architecture,
  Planning, and Example profiles. It compares every changed claim with current source, focused
  tests, and the unchanged Runtime/Public API references, and changes no Java or Javadoc.
- A separate clean independent review context is mandatory because this Class C task changes the
  architecture root and an incorporated scoped contract. It inspects the final diff and evidence,
  checks terminology/examples/links and targeted glossary impact, and records its completion
  summary before the task may become Complete.

## Result

Empty until execution. Record the changed documents, exact validation outcomes, independent
review, reasoned no-change conclusions, limitations or follow-up, and final status here.
