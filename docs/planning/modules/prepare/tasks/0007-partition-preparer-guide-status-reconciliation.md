# Task 0007: Partition Preparer Guide Status Reconciliation

## Status

Ready

Frontier verification: Prepare 0001–0006 and Planning 0007 are Complete, no task is `Ready` or
`In progress`, and the user authorized the next confirmed drift repair. This task is the sole
active frontier and its dependencies are satisfied.

## Change class

Class C — this documentation-only repair corrects the explained Prepare/backend/Engine boundary,
physical-resource ownership, and persistent prepared-resource lifecycle. It changes no authority
or executable behavior but requires clean implementation and independent targeted review contexts.

## Goal

Make the partition-preparer backend guide describe the current CPU preparation, Engine
composition, per-run physical representations, tuning composition, and persistent prepared-
resource transaction while preserving the exact owner boundaries.

## Scope

- Replace obsolete statements that CPU adaptation, GraphPreparation integration, Config/Engine
  composition, production schedule assembly, physical allocation, and end-to-end CPU execution
  are all future.
- Distinguish current fixed CPU `Engine.standard()` and explicit advanced CPU ownership from still-
  planned generic backend registration and mixed-owner composition.
- Correct finalization/resource text: backend finalization may acquire immutable persistent
  prepared resources; shared Prepare owns returned resources transactionally until successful
  `PreparedExecution` construction; CPU currently returns none, while concrete backends own
  physical representations and cleanup.
- Correct validation and limitations/status paragraphs for current CPU conformance/integration,
  bounded two-phase Engine tuning, and remaining unsupported generic/dynamic functionality.
- Keep the existing contract example illustrative; do not turn it into a duplicate CPU backend
  implementation guide.
- Synchronize this task, Prepare master plan, and roadmap after implementation and review.

## Non-goals

- No Java, Javadoc, tests, Gradle, dependency, API, behavior, lifecycle, ownership, concurrency,
  native ABI, backend capability, tuning algorithm, cache, or architecture-authority change.
- No generic backend registry, mixed-owner Engine execution, dynamic dimensions, workspace reuse,
  persistent CPU complete-plan cache, executable serialization, or new trace schema.
- No broad rewrite of the example or separate CPU-backend-guide drift repair.
- No change to the separately tracked glossary/`GraphCompilationPort` follow-up from Planning 0007.

## Contracts

- [`ARCHITECTURE.md` headings `Core lifecycle`](../../../../../ARCHITECTURE.md#core-lifecycle),
  [`Core invariants`](../../../../../ARCHITECTURE.md#core-invariants), and
  [`Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing) — preserve
  compile/prepare/run separation, resource ownership, and Engine composition.
- [`runtime-prepare-engine.md` headings `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [`Prepare lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [`Run lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle), and
  [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine) —
  preserve analysis/finalization, transactional resource transfer, run state, and current CPU-only
  Engine composition.
- [`backend-execution.md` heading `Concrete backend modules`](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  — preserve backend ownership of lowering, storage, physical mechanics, and native resources.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Execution may change exactly these four paths:

- `docs/backend-guide/partition-preparer.md` — current-status, lifecycle/resource, registration,
  validation, and limitations paragraphs only.
- this task brief — status and compact result evidence only.
- `docs/planning/modules/prepare/master-plan.md` — task row and frontier only.
- `docs/planning/roadmap.md` — Prepare row and repository frontier only.

Read-only evidence:

- `GraphPreparation`, `BackendPartitionFinalizationResult`, Runtime `PreparedExecution`,
  `PreparedExecutionRunner`, `CpuBackendIntegration`, CPU schedule assembly, `Engine.standard()`,
  `AdvancedEngine.takeOwnership(...)`, and `Engine.prepareTuned(...)`.
- Focused Prepare, Runtime persistent-resource/runner, CPU integration, and Engine composition tests.
- Current public/runtime API explanations, ADR 0013, and targeted glossary entries for analysis,
  finalization, prepared resources, prepared execution, and Engine composition.

## Acceptance criteria

- The guide accurately presents current fixed CPU preparation and Engine execution, bounded
  two-phase tuning, per-run physical representations, and transactional prepared-resource support.
- It distinguishes shared Prepare ownership from concrete backend physical mechanics and Runtime
  run-state ownership; CPU's currently empty persistent-resource list is explicit where useful.
- Generic registration, mixed-owner composition, dynamic dimensions, workspace reuse, persistent
  CPU plan-cache reuse, and executable persistence remain clearly unsupported or planned.
- The illustrative example remains scoped and does not claim to be production CPU code.
- Only the four allowed paths change; no Java/Gradle files change; local links/anchors, headings,
  fences, final newlines, whitespace, exact path inventory, and `git diff --check` pass.

## Validation

```bash
./gradlew :modules:prepare:test \
  --tests io.github.pho001.synaptik.prepare.GraphPreparationTest
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.execution.PreparedExecutionTest \
  --tests io.github.pho001.synaptik.runtime.run.PreparedExecutionRunnerTest
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.spi.CpuBackendIntegrationAndCpuPreparedScheduleAssemblerPublicTest
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.EngineStandardCompositionTest
rg -n 'no supported CPU adapter|Engine composition and explicit backend registration are not implemented|future engine will supply|integration tests once Engine composition|no .*production Engine composition|cleanup require a later finalized prepared-resource lifecycle' \
  docs/backend-guide/partition-preparer.md
git diff --name-only -- '*.java' '*.gradle' '*.gradle.kts'
git diff --check
```

The stale-text and executable/build-file inventories must be empty. Validate changed Markdown and
the exact four-path inventory with the repository validator or an equivalent temporary script.

Repository-wide validation is deferred to CI because this task changes only explanatory status
and no code, dependency, or authority. Independent review reuses successful focused-test evidence.

## Dependencies and follow-up

- Prepare 0001–0006, Runtime 0016, CPU integration, Engine 0010/0013, tuning 0004, and Planning
  0007 are Complete.
- After completion, separately plan the CPU backend guide status cluster; this task does not
  combine that backend-specific guide work.
- The Planning 0007 glossary/`GraphCompilationPort` follow-up remains separately tracked.

## Documentation and review impact

- The implementation applies the General, Backend Guide, Architecture, and Planning profiles and
  performs a targeted glossary review.
- A separate clean independent review is mandatory because this Class C task explains the
  Runtime/Prepare/backend boundary, resource lifecycle, and Engine composition.

## Result

Empty until execution.
