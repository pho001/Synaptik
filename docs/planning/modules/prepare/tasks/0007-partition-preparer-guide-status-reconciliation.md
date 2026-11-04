# Task 0007: [Partition](../../../../glossary.md#partition) Preparer Guide Status Reconciliation

## Status

Complete

Frontier verification: [Prepare](../../../../glossary.md#prepare) 0001–0006 and [Planning](../../../../glossary.md#term-planning) 0007 are Complete, no task is [`Ready`](../../../../glossary.md#term-ready) or
`In progress`, and the user authorized the next confirmed drift repair. This task is the sole
active frontier and its dependencies are satisfied.

## Change class

Class C — this documentation-only repair corrects the explained [Prepare](../../../../glossary.md#prepare)/[backend](../../../../glossary.md#backend)/[Engine](../../../../glossary.md#term-engine) boundary,
physical-resource [ownership](../../../../glossary.md#term-ownership), and persistent prepared-resource [lifecycle](../../../../glossary.md#lifecycle). It changes no authority
or executable behavior but requires clean implementation and independent targeted review contexts.

## Goal

Make the [partition](../../../../glossary.md#partition)-preparer [backend](../../../../glossary.md#backend) guide describe the current [CPU](../../../../glossary.md#term-cpu) preparation, [Engine](../../../../glossary.md#term-engine)
composition, per-[run](../../../../glossary.md#term-run) physical representations, tuning composition, and persistent prepared-
[resource transaction](../../../../glossary.md#resource-transaction-and-rollback) while preserving the exact [owner](../../../../glossary.md#term-owner) boundaries.

## [Scope](../../../../glossary.md#term-scope)

- Replace obsolete statements that CPU adaptation, GraphPreparation integration, Config/Engine
  composition, production schedule assembly, physical allocation, and end-to-end CPU execution
  are all future.<span data-glossary-navigation="true"> ([GraphPreparation glossary](../../../../glossary.md#graph-preparation--graphpreparation); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Distinguish current fixed [CPU](../../../../glossary.md#term-cpu) `Engine.standard()` and explicit advanced CPU ownership from still-
  planned generic backend registration and mixed-owner composition.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend); [owner glossary](../../../../glossary.md#term-owner))</span>
- Correct finalization/resource text: backend finalization may acquire immutable persistent
  prepared resources; shared Prepare owns returned resources transactionally until successful<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
  [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) construction; CPU currently returns none, while concrete backends own
  physical representations and cleanup.
- Correct validation and limitations/status paragraphs for current CPU conformance/integration,
  bounded two-phase Engine tuning, and remaining unsupported generic/dynamic functionality.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Keep the existing contract example illustrative; do not turn it into a duplicate CPU backend
  implementation guide.
- Synchronize this task, [Prepare](../../../../glossary.md#prepare) master plan, and roadmap after implementation and review.

## Non-goals

- No Java, Javadoc, tests, Gradle, dependency, API, behavior, lifecycle, ownership, concurrency,
  native ABI, backend capability, tuning algorithm, cache, or architecture-authority change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [backend capability glossary](../../../../glossary.md#backend-capability); [cache glossary](../../../../glossary.md#term-cache))</span>
- No generic backend registry, mixed-owner Engine execution, dynamic dimensions, workspace reuse,
  persistent CPU complete-plan cache, executable serialization, or new trace schema.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [owner glossary](../../../../glossary.md#term-owner); [Engine glossary](../../../../glossary.md#term-engine); [dimensions glossary](../../../../glossary.md#dimension); [serialization glossary](../../../../glossary.md#term-serialization); [trace glossary](../../../../glossary.md#trace); [schema glossary](../../../../glossary.md#term-schema))</span>
- No broad rewrite of the example or separate [CPU](../../../../glossary.md#term-cpu)-[backend](../../../../glossary.md#backend)-guide drift repair.
- No change to the separately tracked glossary/`GraphCompilationPort` follow-up from [Planning](../../../../glossary.md#term-planning) 0007.

## Contracts

- [`ARCHITECTURE.md` headings `Core lifecycle`](../../../../../ARCHITECTURE.md#core-lifecycle),
  [`Core invariants`](../../../../../ARCHITECTURE.md#core-invariants), and
  [`Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing) — preserve
  compile/prepare/run separation, resource ownership, and Engine composition.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership); [Engine glossary](../../../../glossary.md#term-engine))</span>
- [`runtime-prepare-engine.md` headings `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [`Prepare lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [`Run lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle), and
  [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine) —
  preserve analysis/finalization, transactional resource transfer, run state, and current CPU-only
  Engine composition.<span data-glossary-navigation="true"> ([run state glossary](../../../../glossary.md#run-state--runstate); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- [`backend-execution.md` heading `Concrete backend modules`](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  — preserve backend ownership of lowering, storage, physical mechanics, and native resources.<span data-glossary-navigation="true"> ([backend ownership glossary](../../../../glossary.md#backend-ownership); [lowering glossary](../../../../glossary.md#lowering))</span>

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Execution may change exactly these four paths:

- `docs/backend-guide/partition-preparer.md` — current-status, lifecycle/resource, registration,
  validation, and limitations paragraphs only.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [validation glossary](../../../../glossary.md#term-validation))</span>
- this task brief — status and compact result evidence only.
- `docs/planning/modules/prepare/master-plan.md` — task row and frontier only.
- `docs/planning/roadmap.md` — [Prepare](../../../../glossary.md#prepare) row and repository frontier only.

Read-only evidence:

- [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), `BackendPartitionFinalizationResult`, [Runtime](../../../../glossary.md#term-runtime) [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution),
  [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner), `CpuBackendIntegration`, [CPU](../../../../glossary.md#term-cpu) schedule assembly, `Engine.standard()`,
  `AdvancedEngine.takeOwnership(...)`, and `Engine.prepareTuned(...)`.
- Focused Prepare, Runtime persistent-resource/runner, CPU integration, and [Engine](../../../../glossary.md#term-engine) composition tests.
- Current public/runtime API explanations, ADR 0013, and targeted glossary entries for analysis,
  finalization, prepared resources, prepared execution, and Engine composition.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>

## Acceptance criteria

- The guide accurately presents current fixed CPU preparation and Engine execution, bounded
  two-phase tuning, per-run physical representations, and transactional prepared-resource support.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [run glossary](../../../../glossary.md#term-run))</span>
- It distinguishes shared Prepare ownership from concrete backend physical mechanics and Runtime
  run-state ownership; CPU's currently empty persistent-resource list is explicit where useful.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [concrete backend glossary](../../../../glossary.md#concrete-backend); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Generic registration, mixed-owner composition, dynamic dimensions, workspace reuse, persistent
  CPU plan-cache reuse, and executable persistence remain clearly unsupported or planned.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [dimensions glossary](../../../../glossary.md#dimension); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- The illustrative example remains scoped and does not claim to be production [CPU](../../../../glossary.md#term-cpu) code.
- Only the four allowed paths change; no Java/Gradle files change; local links/anchors, headings,
  fences, final newlines, whitespace, exact path inventory, and <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`git diff --check` pass.

## [Validation](../../../../glossary.md#term-validation)

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

Repository-wide validation is deferred to [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes only explanatory status
and no code, [dependency](../../../../glossary.md#term-dependency), or authority. Independent review reuses successful focused-test evidence.

## Dependencies and follow-up

- Prepare 0001–0006, Runtime 0016, CPU integration, Engine 0010/0013, tuning 0004, and Planning
  0007 are Complete.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
- After completion, separately plan the CPU backend guide status cluster; this task does not
  combine that backend-specific guide work.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- The [Planning](../../../../glossary.md#term-planning) 0007 glossary/`GraphCompilationPort` follow-up remains separately tracked.

## Documentation and review impact

- The implementation applies the General, Backend Guide, Architecture, and Planning profiles and
  performs a targeted glossary review.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning))</span>
- A separate clean independent review is mandatory because this Class C task explains the
  Runtime/Prepare/backend boundary, resource lifecycle, and Engine composition.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Result

Implementation and mandatory independent Class C review are complete.

- Reconciled the focused guide with current CPU preparation, fixed ordinary and explicit advanced
  Engine composition, bounded two-phase tuning, per-run physical representations, and persistent
  prepared-resource ownership.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership))</span>
- Preserved generic registration, mixed-owner composition, dynamic dimensions, workspace reuse,
  persistent CPU complete-plan reuse, and executable persistence as unsupported or planned.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [dimensions glossary](../../../../glossary.md#dimension); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Updated the illustrative finalizer to return the current resource-bearing
  `BackendPartitionFinalizationResult`; it remains explicitly non-production and resource-free.
- Independent review checked the finalization result example and every CPU/Engine/tuning,
  per-run representation, ownership-transfer, rollback, lease, and fixed-versus-generic
  composition claim against current contracts, source, tests, ADR 0013, and current API guides.
  It tightened the request type and <span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [lease glossary](../../../../glossary.md#term-lease); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>[`RunState`](../../../../glossary.md#run-state--runstate)/[`RunResult`](../../../../glossary.md#run-result--runresult) cleanup wording without changing [scope](../../../../glossary.md#term-scope).
- Review reused the successful focused Prepare, Runtime, CPU, and Engine Gradle evidence because
  no executable Java changed. Stale-text, Java/Gradle diff, exact four-path inventory, local
  Markdown target/anchor, heading, fence, final-newline, whitespace, example/source-consistency,
  and <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Gradle glossary](../../../../glossary.md#term-gradle); [target glossary](../../../../glossary.md#term-target))</span>`git diff --check` [validation](../../../../glossary.md#term-validation) passed.
- Documentation impact: guide, task, master-plan frontier, and roadmap frontier updated; no new
  glossary term was introduced. Targeted glossary review found a pre-existing stale CPU-finalizer
  status paragraph under [`Backend partition finalization`](../../../../glossary.md#backend-partition-finalization); it is outside this four-path task and
  is recorded for the next focused glossary reconciliation rather than widened into this change.

Status: Complete
