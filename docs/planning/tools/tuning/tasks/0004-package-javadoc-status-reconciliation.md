# Task 0004: Tuning Package Javadoc Status Reconciliation

## Status

Ready

Frontier verification: tuning 0001–0003 and Engine 0013 are Complete, no task is `Ready` or
`In progress`, and the user authorized the next confirmed drift repair. This task is the sole
active frontier and its dependencies are satisfied.

## Change class

Class B — the edit corrects public package Javadoc for a durable two-phase tuning capability and
its Engine composition. It changes no executable behavior or authority, but requires a clean
implementation context and independent targeted documentation review.

## Goal

Make `tools/tuning` package Javadoc describe the current public CPU Engine composition around the
generic Phase-1 and Phase-2 transactions without expanding or reallocating ownership.

## Scope

- Remove the obsolete claim that no public Engine invokes Phase 2.
- State narrowly that current `Engine.prepareTuned(...)` composes the generic two-phase tool for
  one eligible CPU workload occurrence and a session-scoped CPU complete-plan batch.
- Replace completed Config/Engine/CPU future-work claims with the actual remaining boundaries:
  broader occurrence extraction, graph/ownership/partition alternatives, mixed backends,
  persistent CPU complete-plan reuse, and executable persistence.
- Synchronize this task, tuning master plan, and roadmap after implementation and review.

## Non-goals

- No change to Java signatures, executable Java, tests, Gradle, module dependencies, tuning
  algorithms, caches, evidence, Engine behavior, backend capability, or architecture authority.
- No claim that `tools/tuning` owns Engine lifecycle orchestration, representative inputs,
  correctness bytes, Runtime execution, selected preparation, or CPU candidate meaning.
- No resolution of broader authority wording about workload extraction; report any actual
  contract conflict instead of editing an incorporated contract in this task.
- No capability-provider, partition-preparer, CPU-backend, or unrelated Javadoc cleanup.

## Contracts

- [`ARCHITECTURE.md` heading `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants) and
  [`Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing) — preserve
  pre-Runtime tuning, Engine composition, and tools-owned measurement/cache/selection.
- [`backend-execution.md` heading `Performance evidence and optimization tooling`](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  — preserve the coordinated two-phase workflow and opaque backend candidate boundary.
- [`runtime-prepare-engine.md` heading `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — preserve current public Engine composition and lifecycle ownership.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Execution may change exactly these four paths:

- `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/package-info.java` — package
  status and remaining-boundary paragraph only.
- this task brief — status and compact result evidence only.
- `docs/planning/tools/tuning/master-plan.md` — task row and frontier only.
- `docs/planning/roadmap.md` — tuning row and repository frontier only.

Read-only evidence:

- `Engine.prepareTuned(...)`, `AdvancedEngine.prepareTunedOrdinaryWithSession(...)`, and
  `CompletePlanAutotuningCompositionTest`.
- `CompletePlanTuning`, `WorkloadTuning`, their focused tests, and current public API/tuning
  explanations.
- Targeted glossary entries for model autotuning, workload tuning, and complete-plan tuning.

## Acceptance criteria

- Package Javadoc identifies current public Engine Phase-2 composition accurately and narrowly.
- Completed Config, Engine, and CPU composition are not described as future work.
- Remaining limitations stay explicit and do not imply generic multi-occurrence, graph,
  ownership/partition, mixed-backend, persistent CPU plan-cache, or executable-persistence support.
- Ownership of inputs, execution, correctness, cleanup, candidates, measurement, caching,
  selection, and selected preparation remains consistent with current source and contracts.
- Javadoc generation succeeds; only the four allowed paths change; no executable Java or Gradle
  file changes; local links/anchors and Markdown structure pass; `git diff --check` passes.

## Validation

```bash
./gradlew :tools:tuning:test \
  --tests io.github.pho001.synaptik.tools.tuning.CompletePlanTuningTest \
  --tests io.github.pho001.synaptik.tools.tuning.CompletePlanTuningValidationTest
./gradlew :tools:tuning:javadoc
rg -n 'no public Engine composition invokes|Config extension, Engine/CPU adaptation' \
  tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/package-info.java
git diff --name-only -- '*.java' ':!tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/package-info.java' '*.gradle' '*.gradle.kts'
git diff --check
```

The stale-text and unexpected executable/build-file searches must produce no output. Validate the
task/master/roadmap Markdown and exact four-path inventory with the repository's existing validator
or an equivalent temporary script outside the repository.

Repository-wide validation is deferred to CI because the task changes only package Javadoc and
planning status. The independent review reuses successful test/Javadoc evidence unless it changes
executable behavior or final Javadoc afterward.

## Dependencies and follow-up

- Tuning 0001–0003, Engine 0009, and Engine 0013 are Complete.
- After completion, separately plan the capability-provider guide drift; this task does not
  authorize or combine it.

## Documentation and review impact

- The implementation applies the General, API/Javadoc, and Planning profiles and performs a
  targeted glossary review.
- A separate independent documentation review is required because public package Javadoc describes
  a durable cross-module tuning workflow.

## Result

Empty until execution.
