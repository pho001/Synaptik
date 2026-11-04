# Task 0004: Tuning Package [Javadoc](../../../../glossary.md#term-javadoc) Status Reconciliation

## Status

Complete

Frontier verification: tuning 0001–0003 and [Engine](../../../../glossary.md#term-engine) 0013 are Complete, no task is [`Ready`](../../../../glossary.md#term-ready) or
`In progress`, and the user authorized the next confirmed drift repair. This task is the sole
active frontier and its dependencies are satisfied.

## Change class

Class B — the edit corrects public package [Javadoc](../../../../glossary.md#term-javadoc) for a durable two-phase tuning capability and
its [Engine](../../../../glossary.md#term-engine) composition. It changes no executable behavior or authority, but requires a clean
implementation context and independent targeted documentation review.

## Goal

Make `tools/tuning` package [Javadoc](../../../../glossary.md#term-javadoc) describe the current public [CPU](../../../../glossary.md#term-cpu) [Engine](../../../../glossary.md#term-engine) composition around the
generic Phase-1 and Phase-2 transactions without expanding or reallocating [ownership](../../../../glossary.md#term-ownership-meanings).

## [Scope](../../../../glossary.md#term-scope)

- Remove the obsolete claim that no public [Engine](../../../../glossary.md#term-engine) invokes Phase 2.
- State narrowly that current `Engine.prepareTuned(...)` composes the generic two-phase tool for
  one eligible CPU workload occurrence and a session-scoped CPU complete-plan batch.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [occurrence glossary](../../../../glossary.md#term-occurrence); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Replace completed Config/Engine/CPU future-work claims with the actual remaining boundaries:
  broader occurrence extraction, graph/ownership/partition alternatives, mixed backends,
  persistent CPU complete-plan reuse, and executable persistence.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [partition glossary](../../../../glossary.md#partition); [backends glossary](../../../../glossary.md#backend); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Synchronize this task, tuning master plan, and roadmap after implementation and review.

## Non-goals

- No change to Java signatures, executable Java, tests, Gradle, module dependencies, tuning
  algorithms, caches, evidence, Engine behavior, backend capability, or architecture authority.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [Engine glossary](../../../../glossary.md#term-engine); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- No claim that `tools/tuning` owns Engine lifecycle orchestration, representative inputs,
  correctness bytes, Runtime execution, selected preparation, or CPU candidate meaning.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- No resolution of broader authority wording about workload extraction; report any actual
  contract conflict instead of editing an incorporated contract in this task.
- No capability-provider, [partition](../../../../glossary.md#partition)-preparer, [CPU](../../../../glossary.md#term-cpu)-[backend](../../../../glossary.md#backend), or unrelated [Javadoc](../../../../glossary.md#term-javadoc) cleanup.

## Contracts

- [`ARCHITECTURE.md` heading `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants) and
  [`Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing) — preserve
  pre-Runtime tuning, Engine composition, and tools-owned measurement/cache/selection.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [cache glossary](../../../../glossary.md#term-cache))</span>
- [`backend-execution.md` heading `Performance evidence and optimization tooling`](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  — preserve the coordinated two-phase workflow and opaque backend candidate boundary.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [backend glossary](../../../../glossary.md#backend))</span>
- [`runtime-prepare-engine.md` heading `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — preserve current public Engine composition and lifecycle ownership.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

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
  explanations.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Targeted glossary entries for [model autotuning](../../../../glossary.md#autotuning--model-autotuning), workload tuning, and complete-plan tuning.

## Acceptance criteria

- Package [Javadoc](../../../../glossary.md#term-javadoc) identifies current public [Engine](../../../../glossary.md#term-engine) Phase-2 composition accurately and narrowly.
- Completed Config, Engine, and [CPU](../../../../glossary.md#term-cpu) composition are not described as future work.
- Remaining limitations stay explicit and do not imply generic multi-occurrence, graph,
  ownership/partition, mixed-backend, persistent CPU plan-cache, or executable-persistence support.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [graph glossary](../../../../glossary.md#graph); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [partition glossary](../../../../glossary.md#partition); [backend glossary](../../../../glossary.md#backend); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Ownership of inputs, execution, correctness, cleanup, candidates, measurement, caching,
  selection, and selected preparation remains consistent with current source and contracts.
- Javadoc generation succeeds; only the four allowed paths change; no executable Java or Gradle
  file changes; local links/anchors and Markdown structure pass; <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`git diff --check` passes.

## [Validation](../../../../glossary.md#term-validation)

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

Repository-wide validation is deferred to [CI](../../../../glossary.md#term-continuous-integration-ci) because the task changes only package [Javadoc](../../../../glossary.md#term-javadoc) and
[planning](../../../../glossary.md#term-planning) status. The independent review reuses successful test/Javadoc evidence unless it changes
executable behavior or final Javadoc afterward.

## Dependencies and follow-up

- Tuning 0001–0003, [Engine](../../../../glossary.md#term-engine) 0009, and Engine 0013 are Complete.
- After completion, separately plan the capability-provider guide drift; this task does not
  authorize or combine it.

## Documentation and review impact

- The implementation applies the General, API/Javadoc, and Planning profiles and performs a
  targeted glossary review.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning))</span>
- A separate independent documentation review is required because public package Javadoc describes
  a durable cross-module tuning workflow.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

## Result

The package [Javadoc](../../../../glossary.md#term-javadoc) now records the current public two-phase [CPU](../../../../glossary.md#term-cpu) [Engine](../../../../glossary.md#term-engine) composition and its bounded
limitations without reallocating tuning, Engine, [Runtime](../../../../glossary.md#term-runtime), or [backend ownership](../../../../glossary.md#backend-ownership). Independent
documentation review clarified that `SESSION` is the complete-plan reuse [scope](../../../../glossary.md#term-scope) rather than a
property of the [candidate batch](../../../../glossary.md#candidate-batch) and found no broader contract change.

[Validation](../../../../glossary.md#term-validation):

- `./gradlew :tools:tuning:test --tests
  io.github.pho001.synaptik.tools.tuning.CompletePlanTuningTest --tests
  io.github.pho001.synaptik.tools.tuning.CompletePlanTuningValidationTest` passed;
- `./gradlew :tools:tuning:javadoc` passed after the final Javadoc edit, and rendered package
  inspection found the corrected Engine composition and remaining-boundary text;
- the brief's stale-text and unexpected executable/build-file searches produced no output;
- `python3 /tmp/validate_synaptik_markdown.py <three changed Markdown paths>` passed with
  `validated 3 Markdown files`;
- exact four-path/status [synchronization](../../../../glossary.md#term-synchronization) checks and `git diff --check` passed.

Independent review reused the successful focused Java-test evidence, regenerated final Javadoc
after its wording correction, inspected the rendered package summary, and repeated the stale-text,
Markdown/path/inventory, and whitespace checks successfully.

Documentation impact: applied the General, [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc, and [Planning](../../../../glossary.md#term-planning) profiles to the public
package Javadoc and compact planning evidence. Targeted glossary review found its current
two-phase Engine composition and limitations already accurate, so no glossary edit was needed.

Status: Complete
