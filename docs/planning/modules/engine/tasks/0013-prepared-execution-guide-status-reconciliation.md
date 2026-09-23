# Task 0013: Prepared Execution Guide Status Reconciliation

## Status

Complete

Frontier verification: Engine 0012 and Model 0025M are Complete, no other task is `Ready` or
`In progress`, and the user explicitly authorized continuing the confirmed documentation-drift
sequence. This task is the sole active frontier and its dependencies are satisfied.

## Change class

Class C — this documentation-only repair touches the explanatory Runtime/Prepare/backend boundary
and public prepared-execution lifecycle. It changes no authority, API, behavior, or code, but uses
a clean documentation implementation context and an independent targeted review context.

## Goal

Make the prepared-execution user guide and focused architecture explanation describe the current
complete-plan tuning, explicit prepared-handle closure, and public materialization capabilities.

## Scope

- Correct `docs/user-guide/preparing-execution.md` so current `prepareTuned(...)` is described as
  bounded CPU-only complete-plan tuning built from the current single eligible representative
  workload, rather than as lacking complete-plan tuning.
- Correct the guide's prepared-handle lifecycle: the handle is explicitly and idempotently
  closeable, rejects later use, and does not independently close an already-returned result.
- Replace the single stale `public output access remains later work` sentence in
  `docs/architecture/runtime-prepare-backend-boundary.md` with the current explicit detached host
  materialization boundary.
- Synchronize this task, the Engine master plan, and the roadmap after implementation and review.

## Non-goals

- No Java, Javadoc, tests, Gradle, dependencies, public API, behavior, ownership, concurrency,
  backend, tuning algorithm, cache, numerical, or authoritative architecture-contract change.
- No generic multi-partition, mixed-backend, CUDA, or Metal Engine composition claim.
- No broad rewrite of either document and no repair of the separately owned tuning, capability,
  partition-preparer, or CPU-backend documentation clusters.

## Contracts

- [`ARCHITECTURE.md` heading `Core lifecycle`](../../../../../ARCHITECTURE.md#core-lifecycle) and
  [`Core invariants`](../../../../../ARCHITECTURE.md#core-invariants) — preserve current Engine
  ownership, explicit closeable prepared/result handles, and detached materialization.
- [`runtime-prepare-engine.md` heading `Run lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)
  and [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — preserve the current prepared-execution lease, result, cleanup, and public composition model.
- [`backend-execution.md` heading `Performance evidence and optimization tooling`](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  — preserve the separation between Engine composition and tools-owned tuning transactions.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Execution may change exactly these five paths:

- `docs/user-guide/preparing-execution.md` — current tuning and prepared-handle lifecycle text.
- `docs/architecture/runtime-prepare-backend-boundary.md` — the single stale public-output-access
  sentence only.
- this task brief — status and compact result evidence only.
- `docs/planning/modules/engine/master-plan.md` — task row and frontier only.
- `docs/planning/roadmap.md` — Engine row and repository frontier only.

Read-only evidence:

- `Engine.prepareTuned(...)`, `PreparedExecution`, `RunResult`, `ModelAutotuningPreparation`, and
  their focused Engine tests.
- `docs/api/public-api.md` current ordinary lifecycle and autotuning sections.
- Targeted glossary entries for prepared execution, model autotuning, run result, and
  materialization.

## Acceptance criteria

- The guide truthfully identifies current tuning as bounded CPU-only complete-plan selection from
  one eligible representative workload, while keeping generic graph/partition and mixed-backend
  search explicitly unsupported.
- The guide states that `PreparedExecution` is explicitly and idempotently closeable, rejects later
  use after close, and does not close a previously returned `RunResult`.
- The architecture explanation states that current Engine output access uses explicit detached host
  materialization and does not expose Runtime representations or backend storage.
- No edit changes or expands authoritative architecture, current capability, or ownership.
- Only the five allowed paths change; links and anchors resolve, headings are unique, fences are
  balanced, files end with newlines, no trailing whitespace exists, and `git diff --check` passes.

## Validation

```bash
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.api.EngineTypedPublicShapeTest \
  --tests io.github.pho001.synaptik.engine.EngineTypedLifecycleTest \
  --tests io.github.pho001.synaptik.engine.CompletePlanAutotuningCompositionTest
rg -n 'not generic.*complete-plan|handle itself has no independent close operation|public output access remains later work' \
  docs/user-guide/preparing-execution.md \
  docs/architecture/runtime-prepare-backend-boundary.md
git diff --name-only -- '*.java' '*.gradle' '*.gradle.kts'
git diff --check
```

The stale-text search and executable/build-file inventory must produce no output. Validate changed
Markdown links, anchors, unique headings, balanced fences, final newlines, trailing whitespace,
and the exact five-path inventory with the repository's existing validator or an equivalent
temporary script outside the repository.

Repository-wide validation is deferred to CI because this task only reconciles explanations with
already-tested behavior and changes no executable code, dependency, build configuration, or
authority. The independent review reuses successful focused test evidence.

## Dependencies and follow-up

- Engine 0012 and Model 0025M are Complete. This is the next user-authorized sequential
  documentation-drift repair.
- After completion, separately plan the tuning package Javadoc drift; this task does not authorize
  or combine that work.

## Documentation and review impact

- The implementation context applies the General, User Guide, Architecture, and Planning profiles
  and performs a targeted glossary review.
- A separate clean independent review context is mandatory because this Class C task changes a
  public workflow explanation and the Runtime/Prepare/backend boundary explanation.

## Result

Complete:

- Reconciled the user guide with current bounded two-phase CPU tuning and explicit prepared-handle
  closure, and corrected the focused boundary explanation to current detached host materialization.
- Focused Engine tests passed; stale-text and executable/build-file searches were empty;
  Markdown structure, local links and anchors, exact path scope, and `git diff --check` passed.
- Targeted glossary review found no new or changed term; the existing autotuning, prepared
  execution, materialization, and run-result entries already describe the current meanings.
- Independent Class C review verified the claims against the current contracts, Engine source,
  focused tests, and public API documentation, and made the opening preparation example close its
  handle explicitly. No authority, API, executable behavior, terminology, or glossary change was
  needed.
