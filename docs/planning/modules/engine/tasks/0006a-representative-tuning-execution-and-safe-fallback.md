# Task 0006A: Representative Tuning Execution and Safe Fallback Foundation

> **Editorial historical reading key.**
>
> This foundation is package-private, not a public tuning API. AdvancedEngine admits the operation and
> the representative session orders explicit Tensor inputs by Compiler metadata, validates all
> selected storage, and borrows wrappers once. For each supplied trial recipe, Runtime creates
> isolated mutable state, completes publications and returns a result that Engine closes without
> reading or copying payloads. Completion therefore establishes lifecycle consumption, not numerical
> equivalence. CPU owns recipe preparation; the later tuning consumer owns invocation counts, timing,
> ranking and cache work.
>
> Caller storage remains live and unmodified through session cleanup. Execution failure poisons the
> session; fallback is permitted only after proven cleanup and eligible recoverable failure, never for
> Error or lifecycle rejection. See the [current synchronous
> runner](../../../../reference/runtime/orchestration.md#current-prepared-runner) and [current tuned
> preparation
> lifecycle](../../../../reference/engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle).
> The archive's blocked consumer and CPU-only inventory are historical.

## Status

Complete

## Goal

Add the smallest [Engine](../../../../glossary.md#term-engine)-owned internal [lifecycle](../../../../glossary.md#lifecycle) capability needed before optional [model
autotuning](../../../../glossary.md#autotuning--model-autotuning) can be composed safely: bind one explicit representative [Tensor](../../../../glossary.md#tensor) input set in the
authoritative compiled-input order, execute any complete [CPU](../../../../glossary.md#term-cpu) trial recipe synchronously in a fresh
isolated [Runtime](../../../../glossary.md#term-runtime) state, consume completion without materializing results, clean every trial and
[borrowed](../../../../glossary.md#ownership-borrowing-and-lease) input wrapper deterministically, and permit ordinary CPU [heuristic](../../../../glossary.md#term-heuristic) preparation only when
failure isolation and cleanup are known to have completed.

This task establishes package-private execution and fallback machinery only. It does not expose an
[autotuning](../../../../glossary.md#autotuning--model-autotuning) method, translate [`ModelAutotuningConfig`](../../../../glossary.md#model-autotuning-request-facade--modelautotuningconfig), invoke `WorkloadTuning`, [rank](../../../../glossary.md#term-ranking) candidates,
access a [cache](../../../../glossary.md#term-cache), or select a CPU candidate. Engine 0007 remains the consumer that maps the public
request, obtains the CPU [handoff](../../../../glossary.md#term-handoff), delegates measurement/ranking/cache work to `tools/tuning`, and
uses this task's lifecycle boundary for every requested execution and final preparation.

```text
exact compiled graph + explicit representative Tensor values
  -> validate and order by final compiled inputs
  -> snapshot caller-owned HostTensorStorage and borrow wrappers once
  -> for each later-requested warmup or timed execution:
       prepare one fresh complete trial recipe
       execute in one fresh isolated RunState through every publication
       validate publication count and close the result lease
  -> close borrowed wrappers once in reverse order
  -> after successful cleanup, freshly prepare the selected production recipe
     or, only when policy permits, one ordinary safe-heuristic recipe
```

## [Scope](../../../../glossary.md#term-scope)

- Add one package-private [Engine](../../../../glossary.md#term-engine)-owned representative-execution session under
  `io.github.pho001.synaptik.engine`.
- Reuse Engine 0003 [binding](../../../../glossary.md#term-binding) semantics: join supplied [`Tensor`](../../../../glossary.md#tensor) values by [`TensorId`](../../../../glossary.md#tensorid), validate them
  against final `CompiledGraph.Input` [descriptors](../../../../glossary.md#term-descriptor), and order them only by
  `CompiledGraph.inputs()`.
- [Snapshot](../../../../glossary.md#term-snapshot) each selected Tensor's current `HostTensorStorage` once before the first trial. Validate
  the complete input set and every storage before borrowing any wrapper.<span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing))</span>
- Borrow CPU-compatible input representations once for the synchronous session. Engine owns and
  closes only the non-owning wrappers; caller storage and memory scopes remain caller-owned and
  must remain live, accessible, and unmodified through the session.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [synchronous glossary](../../../../glossary.md#term-synchronous))</span>
- Execute one supplied [immutable](../../../../glossary.md#term-immutable) [Runtime](../../../../glossary.md#term-runtime) [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) at a time with the existing
  [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner). Every call reuses only the ordered borrowed-input list and creates a
  fresh isolated <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>[`RunState`](../../../../glossary.md#run-state--runstate); it reuses no trial result or [mutable](../../../../glossary.md#term-mutable) execution state.
- Treat synchronous execution through all scheduled publication actions as sufficient output
  consumption for timing. Validate the expected publication count and immediately close the
  Runtime result. Do not inspect, compare, checksum, copy, or materialize publication payloads.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [checksum glossary](../../../../glossary.md#term-checksum))</span>
- Make result cleanup part of every complete candidate-execution action. A successful action
  returns only after result cleanup succeeds.
- On trial failure, preserve the exact primary `RuntimeException` or `Error`, attempt all owned
  cleanup once, suppress distinct cleanup failures in deterministic order, and poison the session.
  Do not retry or continue with another trial.
- Make session cleanup idempotent, close wrappers in reverse acquisition order, reject execution
  after cleanup starts, and retain the first cleanup failure with later distinct failures
  suppressed.<span data-glossary-navigation="true"> ([idempotent glossary](../../../../glossary.md#term-idempotent))</span>
- Define the package-private final-preparation/fallback boundary for Engine 0007:
  - prepare a selected production recipe freshly only after tuning and representative-session
    cleanup succeed;<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>
  - never promote a timed trial recipe or `RunState` to production;
  - required tuning propagates unavailability/failure without [heuristic](../../../../glossary.md#term-heuristic) preparation;
  - allowed fallback invokes ordinary `CpuBackendIntegration.prepare(artifacts)` exactly once,
    only after representative cleanup and failure isolation are proved complete;
  - forbid fallback for `Error`, Engine closure, ownership/lifecycle rejection, or any failed or
    uncertain cleanup; and<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  - if ordinary heuristic preparation fails, make its exact failure primary and suppress the
    earlier recoverable tuning failure when distinct.
- Add focused native-free tests for binding order and validation, storage snapshot and ownership,
  repeated isolated execution, publication completion, cleanup ordering, failure suppression,
  poisoned-session rejection, fresh selected preparation, and strict/allowed fallback decisions.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Finalize affected Javadocs/package documentation in the mandatory separate clean documentation
  context during implementation, and review explanatory documentation and glossary impact.

## Out of [scope](../../../../glossary.md#term-scope)

- a public [`Engine`](../../../../glossary.md#term-engine), `AdvancedEngine`, handle, result, request, or configuration [API](../../../../glossary.md#term-application-programming-interface-api)
- representative-value storage in [`ModelAutotuningConfig`](../../../../glossary.md#model-autotuning-request-facade--modelautotuningconfig), which intentionally carries identity
  only<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- model fingerprint construction, profile lookup, occurrence extraction/weight/context, or
  multi-partition aggregation<span data-glossary-navigation="true"> ([fingerprint glossary](../../../../glossary.md#term-fingerprint); [lookup glossary](../../../../glossary.md#term-lookup); [occurrence glossary](../../../../glossary.md#term-occurrence); [weight glossary](../../../../glossary.md#term-weight); [partition glossary](../../../../glossary.md#partition))</span>
- implementing `BackendWorkloadTuning` or `ColdCandidateMeasurement`, invoking `WorkloadTuning`,
  timing, ranking, tie breaking, evidence, cache access, compatibility, codecs, or persistence<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- enumerating or interpreting CPU candidate, route, representation, thread, provider, or hardware
  fields
- changing [CPU](../../../../glossary.md#term-cpu), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), Config, [Compiler](../../../../glossary.md#term-compiler), [Model](../../../../glossary.md#term-model), or tuning public contracts
- output correctness comparison or [host](../../../../glossary.md#term-host) [materialization](../../../../glossary.md#materialization)
- asynchronous execution, concurrent trials, retry, timeout, cancellation, or failed-session
  recovery<span data-glossary-navigation="true"> ([asynchronous glossary](../../../../glossary.md#term-asynchronous); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- tools/tuning 0002 [graph](../../../../glossary.md#graph)/plan search or CPU 0016 peer-route [generalization](../../../../glossary.md#term-generalization)
- hidden defaults, global state, service lookup, reflection, a registry, or generic callback
  framework<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [callback glossary](../../../../glossary.md#term-callback))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants, Engine,
  Runtime, Prepare, Concrete backend modules, Performance evidence and optimization tooling, and
  Dependency rules<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module); [Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Engine master plan](../master-plan.md)
- [Engine](../../../../glossary.md#term-engine) [0003](0003-typed-logical-input-binding-and-published-result-access.md) and
  [0006](0006-one-shot-scalar-objective-backward-convenience.md)
- [CPU 0010I](../../../backends/cpu/tasks/0010i-supported-cpu-local-workload-tuning-composition-adapter.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [Config 0006A](../../config/tasks/0006a-model-autotuning-request-configuration.md)
- [tools/tuning 0001](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)
- [Prepare 0004](../../prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)

## Architecture constraints

- [Engine](../../../../glossary.md#term-engine) owns logical input [binding](../../../../glossary.md#term-binding), orchestration, result consumption, and [lifecycle](../../../../glossary.md#lifecycle) policy.
- Runtime remains unaware of tuning. It executes one prepared recipe with ordered borrowed inputs
  in isolated per-run state through existing contracts.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- CPU owns valid candidate generation and exact trial, selected, and ordinary heuristic
  preparation. Engine must not derive CPU-private facts.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- `tools/tuning` solely owns invocation counts, timing, ranking, cache coordination, persistence,
  and evidence. This task creates only the lifecycle action that 0007 will supply to it.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Representative values/storage are caller-owned and retained only for the synchronous admitted
  operation. Engine owns only its wrappers and trial run state.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [operation glossary](../../../../glossary.md#operation))</span>
- All work is cold and precedes production Runtime execution. No tuning branch, cache access, or
  selection may enter a prepared hot loop.
- Fallback cannot grant eligibility, treat incompatible/corrupt data as a hit, accept partial
  selection, or conceal preparation/cleanup failure.
- Stop if implementation needs Runtime tuning state, CPU ownership of representative inputs, or
  Engine interpretation of backend candidate fields.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend))</span>

## Package impact

Only `io.github.pho001.synaptik.engine` changes. No package, public type, or public/protected member
is added. Prefer one package-private `RepresentativeExecutionSession` plus private orchestration
methods in `AdvancedEngine`; permit a second package-private final-preparation helper only if it
makes the fallback state machine independently testable without a generic [callback](../../../../glossary.md#term-callback) abstraction.

`EngineBackendComposition` continues to own [borrowing](../../../../glossary.md#term-borrowing) and ordinary [heuristic](../../../../glossary.md#term-heuristic) preparation.
`CpuEngineBackendComposition` may expose its retained `CpuLocalWorkloadTuning` through a concrete
package-private method in 0007; this task does not widen the shared composition seam.

## Affected files

Expected production/test paths:

- add `modules/engine/src/main/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSession.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/EngineBackendComposition.java` only
  if a named fallback operation is necessary instead of existing <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>[`prepare`](../../../../glossary.md#prepare)
- add `modules/engine/src/test/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSessionTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineTypedLifecycleTest.java` only
  when its binding/cleanup fixture is required<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/AdvancedEngineLifecycleTest.java`
  only for admission-versus-close coverage<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>

Expected documentation/[planning](../../../../glossary.md#term-planning) paths during implementation:

- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/package-info.java`
- `docs/architecture/performance-evidence-and-tuning.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` only if its lifecycle explanation is
  incomplete<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `docs/glossary.md` only for a genuinely new reusable term
- this task, [Engine master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>, and [roadmap](../../../roadmap.md)

Review without modification unless contradicted: `ARCHITECTURE.md`, ADRs, Config, tuning, [CPU](../../../../glossary.md#term-cpu),
Prepare and [Runtime](../../../../glossary.md#term-runtime) contracts, [Gradle](../../../../glossary.md#term-gradle) files, architecture tests, [conformance](../../../../glossary.md#term-conformance) tests, and integration
tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 13 implementation paths: 3 [Engine](../../../../glossary.md#term-engine) production, 3 Engine test, 1 Engine package-documentation,
3 explanatory documentation, and 3 [planning](../../../../glossary.md#term-planning) paths. The glossary may replace but not increase the
documentation allowance. No build or [dependency](../../../../glossary.md#term-dependency) change is expected because this task does not
invoke `tools/tuning`. Stop if another [module](../../../../glossary.md#term-module)'s source/public [API](../../../../glossary.md#term-application-programming-interface-api), a new dependency, or more than
one new production type is required.

## Failure semantics

1. [Engine](../../../../glossary.md#term-engine) [admission](../../../../glossary.md#term-admission) and [ownership](../../../../glossary.md#term-ownership-meanings) checks precede representative [argument](../../../../glossary.md#term-argument) inspection.
2. Input identity, descriptor, storage presence/capacity/liveness/accessibility are validated
   completely before the first borrow or trial preparation.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [liveness glossary](../../../../glossary.md#term-liveness))</span>
3. Partial borrow failure closes acquired wrappers in reverse order and preserves the borrow
   failure as primary.
4. Each action uses a fresh trial recipe and runner-created [`RunState`](../../../../glossary.md#run-state--runstate).
5. [Publication](../../../../glossary.md#term-publication)-count mismatch is trial failure, never a partial result.
6. Result closure follows every successful [run](../../../../glossary.md#term-run) and every later [validation](../../../../glossary.md#term-validation) failure.
7. Any trial or cleanup failure poisons the session; it permits no later trial, selected
   preparation, or fallback through that session.
8. Repeated close is [idempotent](../../../../glossary.md#term-idempotent) and consistently rethrows the retained exact cleanup failure.
9. Engine close waits for the synchronous admitted operation; normal rollback closes any result
   that cannot be published after closure begins.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [operation glossary](../../../../glossary.md#operation); [rollback glossary](../../../../glossary.md#term-rollback))</span>
10. Only a recoverable `RuntimeException` from tuning work is fallback-eligible. `Error`, Engine
    lifecycle/ownership rejection, and cleanup failure are not.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
11. Required tuning reports missing handoff or cache/candidate/trial/selected-preparation failure
    unchanged after cleanup and performs no ordinary preparation.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [cache glossary](../../../../glossary.md#term-cache))</span>
12. Allowed fallback performs exactly one ordinary preparation after proven cleanup. If it fails,
    that preparation failure is primary and the earlier tuning failure is suppressed when distinct.

## Hot-path and performance constraints

- Snapshot list/storage once and borrow each required input once per session. Do not repeat Tensor
  maps, storage lookup, descriptor validation, or wrapper allocation per sample.<span data-glossary-navigation="true"> ([Snapshot glossary](../../../../glossary.md#term-snapshot); [Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Still use fresh [CPU](../../../../glossary.md#term-cpu) preparation and [Runtime](../../../../glossary.md#term-runtime) state for each complete execution.
- Do not materialize outputs or allocate [canonical](../../../../glossary.md#term-canonical-form) [host](../../../../glossary.md#term-host) byte arrays in the measured action.
- Add no hot-path tuning branch, map [lookup](../../../../glossary.md#term-lookup), [dispatch](../../../../glossary.md#term-dispatch), reflection, measurement, or [cache](../../../../glossary.md#term-cache) access.
- Prefer direct private/package-private calls over a general event/[callback](../../../../glossary.md#term-callback) framework.

## Acceptance criteria

1. [Engine](../../../../glossary.md#term-engine) 0006A is `Complete`, ordered after 0006/[CPU](../../../../glossary.md#term-cpu) 0010I and before Blocked 0007.
2. No public [API](../../../../glossary.md#term-application-programming-interface-api) or non-Engine [module](../../../../glossary.md#term-module) changes.
3. Representative inputs follow exact Engine 0003 identity/descriptor/storage/final-order rules;
   validation precedes borrowing and execution.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [borrowing glossary](../../../../glossary.md#term-borrowing))</span>
4. Storage is snapshotted/[borrowed](../../../../glossary.md#ownership-borrowing-and-lease) once, remains caller-owned, and wrappers close once in reverse.
5. Every call uses a fresh recipe/state, completes publications, validates count, closes the result,
   and returns no output value.
6. [Warmup](../../../../glossary.md#term-warmup) and timed calls have identical [shape](../../../../glossary.md#shape); Engine performs no timing.
7. Failure [identity](../../../../glossary.md#term-identity-meanings)/suppression is deterministic and a failed session cannot be reused.
8. Selected production preparation is fresh and occurs after successful session cleanup.
9. Required tuning never falls back; allowed fallback calls ordinary preparation at most once and
   only after recoverable failure with proven isolation.
10. `Error`, lifecycle/ownership, and cleanup failure never fall back; fallback preparation failure
    remains visible as primary.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
11. Measurement/[ranking](../../../../glossary.md#term-ranking)/[cache](../../../../glossary.md#term-cache)/evidence/candidate interpretation stays outside Engine.
12. Native-free tests cover success and all material state/failure transitions.
13. A clean documentation context finalizes Javadocs/docs and records glossary/no-change findings.
14. Final Engine tests/[Javadoc](../../../../glossary.md#term-javadoc), applicable [architecture test](../../../../glossary.md#term-architecture-test), Markdown/[scope](../../../../glossary.md#term-scope)/status checks, and
    `git diff --check` pass.

## Tests / [validation](../../../../glossary.md#term-validation)

```bash
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.RepresentativeExecutionSessionTest \
  --tests io.github.pho001.synaptik.engine.EngineTypedLifecycleTest \
  --tests io.github.pho001.synaptik.engine.AdvancedEngineLifecycleTest
./gradlew :modules:engine:test
```

The documentation context reuses successful Java evidence unless it changes executable behavior:

```bash
./gradlew :modules:engine:javadoc
git diff --check
git status --short -uall
```

[Run](../../../../glossary.md#term-run) focused [Engine](../../../../glossary.md#term-engine) architecture tests only if `EngineBackendComposition` changes. No repository-wide,
[conformance](../../../../glossary.md#term-conformance), integration, real [OpenBLAS](../../../../glossary.md#term-openblas), or benchmark run is required. Validate changed Markdown
[targets](../../../../glossary.md#term-target)/anchors, unique headings, balanced fences, final newlines, trailing whitespace, statuses,
ordering, wording, and exact path [scope](../../../../glossary.md#term-scope).

## Dependencies

- [Engine](../../../../glossary.md#term-engine) 0001–0006 — Complete; 0003 supplies [binding](../../../../glossary.md#term-binding) and 0006 cleanup conventions.
- [Runtime](../../../../glossary.md#term-runtime) 0010/0015 — Complete; supply [synchronous](../../../../glossary.md#term-synchronous) isolated execution, publications, and cleanup.
- [CPU](../../../../glossary.md#term-cpu) 0010I — Complete; supplies exact trial/selected preparation and ordinary [heuristic](../../../../glossary.md#term-heuristic) [prepare](../../../../glossary.md#prepare).
- Prepare 0004 — Complete; supplies the [opaque](../../../../glossary.md#term-opaque) [partition](../../../../glossary.md#partition) [handoff](../../../../glossary.md#term-handoff).
- Config 0006A and tools/tuning 0001 — Complete contracts reviewed for future consumption but not
  invoked here.

[Metal](../../../../glossary.md#term-metal), [CUDA](../../../../glossary.md#term-cuda), provider 0004, CPU 0010D1/0016, and tuning 0002 are not dependencies.

## Follow-up tasks

- [Engine](../../../../glossary.md#term-engine) 0007 is the next [planning](../../../../glossary.md#term-planning) frontier but remains `Blocked` pending a separate clean
  reassessment. That planning pass may make it [`Ready`](../../../../glossary.md#term-ready) only after it can specify the public
  representative-value request,
  stable model/occurrence identities, Config-to-tuning mapping, CPU adapter, <span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`WorkloadTuning.tune`
  call, and selected production handle/evidence surface.
- tools/tuning 0002 remains later [graph](../../../../glossary.md#graph)/plan tuning; [CPU](../../../../glossary.md#term-cpu) 0016 remains later peer-route work.

## Architecture impact

Expected impact: None. This realizes existing [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings) at the authorized cold
boundary. If [asynchronous](../../../../glossary.md#term-asynchronous) [backend](../../../../glossary.md#backend) completion requires a [synchronization](../../../../glossary.md#term-synchronization) primitive absent from the
current [synchronous](../../../../glossary.md#term-synchronous) [CPU](../../../../glossary.md#term-cpu) schedule, stop and return to architecture [planning](../../../../glossary.md#term-planning).

## Implementation prompt

```text
You are the clean implementation agent for Synaptik Engine task 0006A. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, the planning guide, Engine master plan, and task 0006A in full,
plus referenced Engine 0003/0006, Runtime 0010/0015, Prepare 0004, CPU 0010I, Config 0006A, and
tools/tuning 0001 contracts and current affected source/tests.

Implement only the package-private Engine representative-execution/fallback foundation within its
path ceiling. Add no public API, tuning algorithm, cache work, candidate interpretation, output
materialization, user Config surface, or tools/tuning dependency. Preserve fresh trial recipe/state,
synchronous publication completion, cleanup/failure identity, and fallback only after proven
isolation. Stop for architecture uncertainty, missing synchronization semantics, another-module
change, or scope expansion.

After Engine tests stabilize, hand the exact diff/evidence to a distinct clean documentation
context in the same change. It must follow documentation-rules.md, finalize affected Javadocs and
focused docs/glossary review, update task/master/roadmap evidence, and not repeat successful Java
suites absent behavior change or concrete stale-evidence risk. Mark Complete only after all gates.
```

## Local decisions

- Use one explicit representative [Tensor](../../../../glossary.md#tensor) [binding](../../../../glossary.md#term-binding) set; public profile [lookup](../../../../glossary.md#term-lookup) belongs to 0007.
- Reuse [Engine](../../../../glossary.md#term-engine) 0003 binding rather than add tuning-specific [identity](../../../../glossary.md#term-identity-meanings)/raw-representation APIs.
- Borrow once per session to avoid per-[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) wrapper overhead; [prepare](../../../../glossary.md#prepare)/state remain fresh per [run](../../../../glossary.md#term-run).
- [Publication](../../../../glossary.md#term-publication) completion, not [host](../../../../glossary.md#term-host) [materialization](../../../../glossary.md#materialization), completes current [synchronous](../../../../glossary.md#term-synchronous) [CPU](../../../../glossary.md#term-cpu) execution.
- Cleanup success is prerequisite to fallback; uncertain resource state cannot continue.
- Fallback-preparation failure is primary because it is the final production-preparation failure;
  the earlier recoverable tuning failure remains suppressed context.
- Add no Engine [dependency](../../../../glossary.md#term-dependency) on `tools/tuning` before 0007 invokes it.

## Known limitations

- [CPU](../../../../glossary.md#term-cpu)-only because CPU is the sole complete [Engine](../../../../glossary.md#term-engine) composition.
- One explicit representative binding set; no profile lookup, sample set, weights, or occurrence
  fingerprints.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [lookup glossary](../../../../glossary.md#term-lookup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [weights glossary](../../../../glossary.md#term-weight); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- [Lifecycle](../../../../glossary.md#lifecycle) completion is validated, not numerical equivalence between candidates.
- No [asynchronous](../../../../glossary.md#term-asynchronous) [device](../../../../glossary.md#term-device) completion contract; future accelerators need an explicit prerequisite.
- No public evidence/diagnostic result; 0007 must select that boundary without exposing internals.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context `01a0b187-a358-7fe1-8c5d-e06f66b143ff` recorded these successful executable
validation results before the clean documentation pass:

- focused [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle) command: 48 tests;
- `./gradlew :modules:engine:test`: 57 tests, 0 failures, 0 errors, and 0 skips; and
- `git diff --check`: passed.

The documentation context inspected the final XML for all six Engine suites. It totals 57 tests,
including 16 `RepresentativeExecutionSessionTest` tests, with zero failures, errors, or skips. It
did not repeat Java tests because documentation changed no executable Java [token](../../../../glossary.md#term-token) and no stale-
evidence risk was found. It ran `./gradlew :modules:engine:javadoc`, relative Markdown [target](../../../../glossary.md#term-target),
changed/new link-anchor, heading, fence, line-ending, whitespace, status, ordering, and exact-[scope](../../../../glossary.md#term-scope)
checks, an executable-token source audit, and final `git diff --check`; all passed.

## Implementation notes

Implementation added one package-private `RepresentativeExecutionSession` and package-private
orchestration helpers in `AdvancedEngine`. [Engine](../../../../glossary.md#term-engine) [admission](../../../../glossary.md#term-admission) now precedes [ownership](../../../../glossary.md#term-ownership-meanings) and input
inspection and remains held across representative trials, reverse cleanup, and final selected or
fallback preparation. Exact final compiled inputs determine [binding](../../../../glossary.md#term-binding) membership and order; all
current caller-owned storage associations are validated and snapshotted before the first borrow.

Each supplied complete trial recipe runs through the stateless [Runtime](../../../../glossary.md#term-runtime) runner, which creates fresh
[run state](../../../../glossary.md#run-state--runstate) and result resources. Engine validates complete [publication](../../../../glossary.md#term-publication) count, closes every result,
and never inspects or materializes a [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication). Trial and cleanup failures prevent every later trial,
selected preparation, and fallback through the session. Cleanup is single-attempt and reverse-
ordered, retains the first cleanup failure, and preserves deterministic suppression.

Selected production preparation occurs freshly after successful cleanup. Strict tuning rethrows
its recoverable failure without ordinary preparation. Allowed fallback invokes ordinary [heuristic](../../../../glossary.md#term-heuristic)
preparation once after proven cleanup; fallback-preparation or final-publication failure is primary
with the tuning failure suppressed when distinct, while cleanup failure leaves the tuning failure
primary with distinct cleanup suppressed. A concurrent Engine close waits for the admission and
rejects a final recipe if closure wins the publication [race](../../../../glossary.md#term-race).

The clean documentation pass strengthened affected package-private Javadocs and updated the two
focused architecture explanations plus existing glossary entries whose planned/current wording
had become stale. `package-info.java` required no change: it documents the public package surface
and already correctly says tuning is not a current [API](../../../../glossary.md#term-application-programming-interface-api).

## Completion summary

```text
Completed changes:
- Added exact representative binding, isolated complete trial execution, deterministic cleanup,
  fresh selected preparation, and strict/allowed safe fallback under one Engine admission.
- Finalized affected package-private lifecycle/failure/ownership Javadocs and current-versus-
  planned architecture and glossary wording.
Files changed or created:
- modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java
- modules/engine/src/main/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSession.java
- modules/engine/src/test/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSessionTest.java
- docs/architecture/performance-evidence-and-tuning.md
- docs/architecture/runtime-prepare-backend-boundary.md
- docs/glossary.md
- docs/planning/modules/engine/tasks/0006a-representative-tuning-execution-and-safe-fallback.md
- docs/planning/modules/engine/master-plan.md
- docs/planning/roadmap.md
Tests and validation performed:
- Reused focused Engine lifecycle evidence: 48 tests passed.
- Reused ./gradlew :modules:engine:test evidence and inspected final XML: 57 tests, 0 failures,
  0 errors, 0 skips, including all 16 representative-session tests.
- ./gradlew :modules:engine:javadoc passed.
- Relative Markdown file targets, changed/new link anchors, unique headings, balanced backtick and
  tilde fences, LF line endings, final newlines, trailing whitespace, path scope, status, and
  ordering checks passed. No changed or new Markdown link contains an anchor fragment.
- Executable-token source comparison passed: Java edits changed comments/Javadocs only.
- git diff --check passed.
Documentation review:
- Public API and package-info.java remain unchanged; the package overview remains accurate.
- Updated two focused explanatory architecture documents because their planned/current wording was
  materially stale; no authoritative architecture rule, ADR, or architecture test changed.
- Updated existing glossary entries only; no private implementation term was added.
- No dependency/build, conformance/integration, or other-module change was required.
Unresolved issues: None for 0006A.
Required follow-up: Reassess Engine 0007 in a separate clean planning context; do not mark it Ready
or create its detailed specification until the public representative-value and identity/evidence
boundaries are resolved.
Implementation context: 01a0b187-a358-7fe1-8c5d-e06f66b143ff
Documentation context: 01a0b2e0-df4b-7ae3-b16e-5afb44220b1e

Status: Complete
```
