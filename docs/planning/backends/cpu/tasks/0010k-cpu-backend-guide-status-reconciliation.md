# Task 0010K: CPU Backend Guide Status Reconciliation

## Status

Complete

Readiness verification recorded before execution: CPU 0010J and Prepare 0007 were Complete, no
task was `Ready` or `In progress`, and the user had authorized the final confirmed CPU-guide drift
cluster. Blocked, review-needed, and Draft CPU rows remain unchanged. This task is the sole active
frontier and its dependencies are satisfied.

## Change class

Class C — this documentation-only repair corrects the explained CPU/Prepare/Engine/tuning
composition and lifecycle across a concrete backend boundary. It changes no authority or
executable behavior but requires clean implementation and independent targeted review contexts.

## Goal

Make the CPU backend guide describe the current fixed CPU Engine facade, Config-driven bounded
two-phase autotuning, Prepare handoffs, and current public composition while preserving CPU's
strict non-ownership of measurement, caching, lifecycle orchestration, and Runtime selection.

## Scope

- Replace the obsolete opening claim that `CpuBackendIntegration` exists only for future Engine
  composition; identify current `Engine.standard()` and explicit advanced CPU composition while
  retaining the SPI/end-user-facade distinction.
- Correct current tuning status in the capability overview and local/complete-plan collaboration
  sections: Config request/policy and public `Engine.prepareTuned(...)` now compose Phase 1 and
  Phase 2 for the bounded CPU-only domain.
- Correct the older OpenBLAS tuning/composition paragraphs that say Prepare cannot transport the
  handoff or applications cannot request tuning through Config/Engine.
- Replace completed future-task prose around Config 0006A, Prepare 0004, CPU 0010I/0010J, and
  tuning 0001–0002 with the current bounded selection boundary.
- Correct the typical-mistakes row that calls Engine a future facade.
- Preserve still-current limitations: CPU owns no measurement/cache/fallback policy; Phase 2 is
  session-scoped; graph/owner/partition alternatives, mixed backends, generic registration,
  persistent CPU complete-plan reuse, and executable persistence remain unsupported.
- Synchronize this task, CPU master plan, and roadmap after implementation and review.

## Non-goals

- No Java, Javadoc, tests, Gradle, dependencies, API, behavior, numerics, capability, route,
  tuning algorithm, cache, native ABI, lifecycle, ownership, or architecture-authority change.
- No rewrite of operation-family coverage, performance ledgers, historical task evidence, native
  checkpoint results, or planned vendor routes.
- No claim that `CpuBackendIntegration` is an ordinary application API or that CPU owns Engine,
  tools/tuning, Runtime, or Config policy.
- No glossary/`GraphCompilationPort` follow-up; that separately tracked task will also repair the
  stale CPU-finalizer glossary entry.

## Contracts

- [`ARCHITECTURE.md` headings `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants)
  and [`Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing) —
  preserve Engine composition, CPU route ownership, and pre-Runtime tuning.
- [`backend-execution.md` headings `Concrete backend modules`](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
  [`Performance evidence and optimization tooling`](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling),
  and [`CPU backend routes`](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  — preserve CPU producer versus tools-owned tuning and the single CPU backend identity.
- [`runtime-prepare-engine.md` headings `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)
  and [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — preserve opaque Prepare transport and current fixed CPU-only public composition.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Execution may change exactly these four paths:

- `docs/backend-guide/cpu-backend.md` — only the current/future status paragraphs named in Scope.
- this task brief — status and compact result evidence only.
- `docs/planning/backends/cpu/master-plan.md` — task row and frontier only.
- `docs/planning/roadmap.md` — CPU row and repository frontier only.

Read-only evidence:

- `CpuBackendIntegration`, `CpuLocalWorkloadTuning`, `CpuCompletePlanTuning`,
  `CpuEngineBackendComposition`, `Engine.standard()`, `Engine.prepareTuned(...)`,
  `AdvancedEngine`, `ModelAutotuningRequest`, and `ModelAutotuningConfig`.
- Focused CPU integration/local/complete-plan tuning tests and Engine complete-plan composition
  tests.
- Current public API, tuning package Javadoc, partition-preparer guide, and targeted glossary
  entries for CPU integration, tuning phases, Engine, and complete plans.

## Acceptance criteria

- Every scoped stale current/future claim is corrected and no remaining guide statement says the
  current Engine facade, Prepare tuning transport, Config request, or public bounded Phase-2
  composition is unimplemented.
- The guide accurately distinguishes the CPU SPI/producer roles from Engine lifecycle,
  tools-owned measurement/cache/selection, Config policy, and Runtime execution.
- Current bounded CPU-only Phase 1/Phase 2 is explicit; generic/mixed/persistent-plan/executable
  limitations remain explicit.
- No unrelated operation coverage, evidence, benchmark, historical, or vendor-route prose changes.
- Only the four allowed paths change; no Java/Gradle files change; links, anchors, headings,
  fences, newlines, whitespace, exact path inventory, and `git diff --check` pass.

## Validation

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.spi.CpuBackendIntegrationAndCpuPreparedScheduleAssemblerPublicTest \
  --tests io.github.pho001.synaptik.backend.cpu.spi.CpuLocalWorkloadTuningPublicTest \
  --tests io.github.pho001.synaptik.backend.cpu.spi.CpuCompletePlanTuningPublicTest
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.CompletePlanAutotuningCompositionTest
rg -n 'future Engine composition|future Engine facade|user-facing Phase-2 Engine API remain unimplemented|applications cannot yet request tuning through Config or Engine|Public Config intent, Engine lifecycle integration, automatic composition, and autotuning remain future work|Shared Prepare does not yet transport these CPU-owned values|Future Config 0006A' \
  docs/backend-guide/cpu-backend.md
git diff --name-only -- '*.java' '*.gradle' '*.gradle.kts'
git diff --check
```

The stale-text and executable/build-file inventories must be empty. Validate changed Markdown and
the exact four-path inventory with the repository validator or an equivalent temporary script.

Repository-wide validation is deferred to CI because this task changes explanatory status only.
Independent review reuses successful focused-test evidence.

## Dependencies and follow-up

- CPU 0010I–0010J, Engine 0009/0013, tuning 0001–0004, Planning 0007, and Prepare 0007 are
  Complete.
- CPU 0007A1D remains Review needed, 0010D1/0011 remain Blocked, and 0012–0017 remain Draft.
- After completion, execute the separately tracked focused glossary/`GraphCompilationPort`
  reconciliation discovered by Planning/Prepare review; do not reopen this guide task.

## Documentation and review impact

- The implementation applies the General, Backend Guide, Architecture, and Planning profiles and
  performs a targeted glossary review.
- A separate clean independent review is mandatory because this Class C task explains CPU,
  Prepare, Engine, tuning, and Runtime ownership boundaries.

## Result

Implementation and mandatory independent Class C review complete.

- Corrected the named Engine, Config, Prepare, tuning, OpenBLAS-composition, and public-facade
  status claims without changing CPU operation coverage or architecture authority.
- Preserved CPU producer ownership, Engine lifecycle composition, Config policy, tools-owned
  measurement/cache/selection, Runtime execution, and current generic/mixed/persistence limits.
- Focused CPU and Engine tests passed; stale-text, Java/Gradle diff, Markdown structure/link,
  exact-path, whitespace, and `git diff --check` validation passed.
- The independent review reused that executable evidence, corrected the remaining capability-
  overview claim and direct synonymous Conv3d/provider/final-summary variants, and verified the
  final guide against the named contracts, current public APIs, focused tests, adjacent guides,
  and targeted glossary entries.

Status: Complete
