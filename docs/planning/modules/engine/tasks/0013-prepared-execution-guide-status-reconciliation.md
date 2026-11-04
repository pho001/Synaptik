# Task 0013: [Prepared Execution](../../../../glossary.md#prepared-execution--preparedexecution) Guide Status Reconciliation

> **Editorial historical reading key.**
>
> This task records a bounded explanatory repair: CPU complete-plan tuning, explicit prepared-handle
> closure, and detached output access were already implemented at its completion. Its CPU-only tuning
> statement is historical; the [current lifecycle
> reference](../../../../reference/engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle)
> also describes eligible singleton-NEG Metal tuning. Engine arranges representative trials and
> production preparation through backend-owned tuning collaborations; a tuning metadata carrier is not
> another preparation owner.
>
> The application closes its prepared handle and each returned result separately. Closing preparation
> rejects new runs without closing an existing result. Publication exposes an ordered occurrence, not
> detached bytes: Engine authenticates the selected occurrence and invokes its captured backend
> materialization adapter. The [current publication
> boundary](../../../../reference/runtime/publication.md#current-prepared-publication-and-result-lease)
> separates Runtime's borrowed representation access from explicit host copying. Caller input storage
> remains caller-owned throughout the result lease.

## Status

Complete

Frontier verification: [Engine](../../../../glossary.md#term-engine) 0012 and [Model](../../../../glossary.md#term-model) 0025M are Complete, no other task is [`Ready`](../../../../glossary.md#term-ready) or
`In progress`, and the user explicitly authorized continuing the confirmed documentation-drift
sequence. This task is the sole active frontier and its dependencies are satisfied.

## Change class

Class C — this documentation-only repair touches the explanatory [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[backend](../../../../glossary.md#backend) boundary
and public prepared-execution [lifecycle](../../../../glossary.md#lifecycle). It changes no authority, [API](../../../../glossary.md#term-application-programming-interface-api), behavior, or code, but uses
a clean documentation implementation context and an independent targeted review context.

## Goal

Make the prepared-execution user guide and focused architecture explanation describe the current
complete-plan tuning, explicit prepared-handle closure, and public [materialization](../../../../glossary.md#materialization) capabilities.

## [Scope](../../../../glossary.md#term-scope)

- Correct `docs/user-guide/preparing-execution.md` so current `prepareTuned(...)` is described as
  bounded CPU-only complete-plan tuning built from the current single eligible representative
  workload, rather than as lacking complete-plan tuning.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Correct the guide's prepared-handle lifecycle: the handle is explicitly and idempotently
  closeable, rejects later use, and does not independently close an already-returned result.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Replace the single stale `public output access remains later work` sentence in
  `docs/architecture/runtime-prepare-backend-boundary.md` with the current explicit detached host
  materialization boundary.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization))</span>
- Synchronize this task, the [Engine](../../../../glossary.md#term-engine) master plan, and the roadmap after implementation and review.

## Non-goals

- No Java, Javadoc, tests, Gradle, dependencies, public API, behavior, ownership, concurrency,
  backend, tuning algorithm, cache, numerical, or authoritative architecture-contract change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Gradle glossary](../../../../glossary.md#term-gradle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency); [cache glossary](../../../../glossary.md#term-cache))</span>
- No generic multi-[partition](../../../../glossary.md#partition), mixed-[backend](../../../../glossary.md#backend), [CUDA](../../../../glossary.md#term-cuda), or [Metal](../../../../glossary.md#term-metal) [Engine](../../../../glossary.md#term-engine) composition claim.
- No broad rewrite of either document and no repair of the separately owned tuning, capability,
  partition-preparer, or CPU-backend documentation clusters.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Contracts

- [`ARCHITECTURE.md` heading `Core lifecycle`](../../../../../ARCHITECTURE.md#core-lifecycle) and
  [`Core invariants`](../../../../../ARCHITECTURE.md#core-invariants) — preserve current Engine
  ownership, explicit closeable prepared/result handles, and detached materialization.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [detached glossary](../../../../glossary.md#detached-model-results); [materialization glossary](../../../../glossary.md#materialization))</span>
- [`runtime-prepare-engine.md` heading `Run lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)
  and [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — preserve the current prepared-execution lease, result, cleanup, and public composition model.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [model glossary](../../../../glossary.md#term-model))</span>
- [`backend-execution.md` heading `Performance evidence and optimization tooling`](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  — preserve the separation between Engine composition and tools-owned tuning transactions.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Execution may change exactly these five paths:

- `docs/user-guide/preparing-execution.md` — current tuning and prepared-handle [lifecycle](../../../../glossary.md#lifecycle) text.
- `docs/architecture/runtime-prepare-backend-boundary.md` — the single stale public-output-access
  sentence only.
- this task brief — status and compact result evidence only.
- `docs/planning/modules/engine/master-plan.md` — task row and frontier only.
- `docs/planning/roadmap.md` — [Engine](../../../../glossary.md#term-engine) row and repository frontier only.

Read-only evidence:

- `Engine.prepareTuned(...)`, [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [`RunResult`](../../../../glossary.md#run-result--runresult), `ModelAutotuningPreparation`, and
  their focused Engine tests.
- `docs/api/public-api.md` current ordinary lifecycle and [autotuning](../../../../glossary.md#autotuning--model-autotuning) sections.
- Targeted glossary entries for prepared execution, model autotuning, run result, and
  materialization.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [run result glossary](../../../../glossary.md#run-result--runresult); [materialization glossary](../../../../glossary.md#materialization))</span>

## Acceptance criteria

- The guide truthfully identifies current tuning as bounded CPU-only complete-plan selection from
  one eligible representative workload, while keeping generic graph/partition and mixed-backend
  search explicitly unsupported.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [graph glossary](../../../../glossary.md#graph); [partition glossary](../../../../glossary.md#partition); [backend glossary](../../../../glossary.md#backend))</span>
- The guide states that [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) is explicitly and idempotently closeable, rejects later
  use after close, and does not close a previously returned [`RunResult`](../../../../glossary.md#run-result--runresult).
- The architecture explanation states that current Engine output access uses explicit detached host
  materialization and does not expose Runtime representations or backend storage.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [detached glossary](../../../../glossary.md#detached-model-results); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- No edit changes or expands authoritative architecture, current capability, or [ownership](../../../../glossary.md#term-ownership-meanings).
- Only the five allowed paths change; links and anchors resolve, headings are unique, fences are
  balanced, files end with newlines, no trailing whitespace exists, and `git diff --check` passes.

## [Validation](../../../../glossary.md#term-validation)

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

Repository-wide validation is deferred to [CI](../../../../glossary.md#term-continuous-integration-ci) because this task only reconciles explanations with
already-tested behavior and changes no executable code, [dependency](../../../../glossary.md#term-dependency), build configuration, or
authority. The independent review reuses successful focused test evidence.

## Dependencies and follow-up

- Engine 0012 and Model 0025M are Complete. This is the next user-authorized sequential
  documentation-drift repair.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Model glossary](../../../../glossary.md#term-model))</span>
- After completion, separately plan the tuning package Javadoc drift; this task does not authorize
  or combine that work.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Documentation and review impact

- The implementation context applies the General, User Guide, Architecture, and Planning profiles
  and performs a targeted glossary review.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- A separate clean independent review context is mandatory because this Class C task changes a
  public workflow explanation and the Runtime/Prepare/backend boundary explanation.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>

## Result

Complete:

- Reconciled the user guide with current bounded two-phase CPU tuning and explicit prepared-handle
  closure, and corrected the focused boundary explanation to current detached host materialization.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [detached glossary](../../../../glossary.md#detached-model-results); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization))</span>
- Focused Engine tests passed; stale-text and executable/build-file searches were empty;
  Markdown structure, local links and anchors, exact path scope, and <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Targeted glossary review found no new or changed term; the existing autotuning, prepared
  execution, materialization, and run-result entries already describe the current meanings.<span data-glossary-navigation="true"> ([autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [run glossary](../../../../glossary.md#term-run))</span>
- Independent Class C review verified the claims against the current contracts, Engine source,
  focused tests, and public API documentation, and made the opening preparation example close its
  handle explicitly. No authority, API, executable behavior, terminology, or glossary change was
  needed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
