# Task 0009: Public Complete-Plan [Autotuning](../../../../glossary.md#autotuning--model-autotuning) Composition

## Status

Complete

## Goal

Extend the existing public [CPU](../../../../glossary.md#term-cpu)-only `Engine.prepareTuned(...)` [operation](../../../../glossary.md#operation) from its completed
Phase-1 local-workload [transaction](../../../../glossary.md#term-transaction) through the completed generic Phase-2 complete-plan
transaction. [Engine](../../../../glossary.md#term-engine) must preserve the authenticated Phase-1 decision, obtain CPU 0010J's bounded
complete-plan [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) around that decision, adapt Engine 0008A's exact representative correctness
[oracle](../../../../glossary.md#term-test-oracle) and fresh representative execution to tools/tuning 0002, and return one freshly prepared
production recipe for the authenticated Phase-2 winner.

```text
existing public request and one admitted representative-input session
  -> Phase 1: cache-first local-workload selection
  -> exact authenticated Phase-1 decision
  -> CPU complete-plan batch around that exact decision
  -> Phase 2: all correctness actions, then warmups and timed samples
  -> exact authenticated complete-plan decision
  -> representative cleanup
  -> one fresh selected production preparation
  -> existing public result with Phase-1 and complete-plan evidence
```

This is the first public composition of already implemented contracts. It adds no new tuning
algorithm, [backend](../../../../glossary.md#backend) candidate meaning, [Runtime](../../../../glossary.md#term-runtime) behavior, persistent CPU reuse, [Compiler](../../../../glossary.md#term-compiler) [graph](../../../../glossary.md#graph)
alternative, [Planning](../../../../glossary.md#term-planning) [ownership](../../../../glossary.md#term-ownership-meanings) alternative, or public Engine operation.

## Source-backed [dataflow](../../../../glossary.md#term-dataflow) and current-contract conclusion

1. `AdvancedEngine.prepareTunedOrdinary(...)` already owns the sole Engine admission, owner and
   request validation, representative binding, Phase-1 request translation, failure isolation,
   strict-versus-safe fallback, fresh production wrapping, and final close-race publication.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [admission glossary](../../../../glossary.md#term-admission); [owner glossary](../../../../glossary.md#term-owner); [validation glossary](../../../../glossary.md#term-validation); [binding glossary](../../../../glossary.md#term-binding); [race glossary](../../../../glossary.md#term-race); [publication glossary](../../../../glossary.md#term-publication))</span>
2. Phase 1 must [run](../../../../glossary.md#term-run) first through `WorkloadTuning.tune(...)`. Its decision-present selected
   handoff is authenticated against the exact original partition and batch. Do not call<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
   `CpuLocalWorkloadTuning.prepareSelected(...)` before Phase 2 and do not repeat Phase-1 candidate
   enumeration, warmups, samples, ranking, or cache work inside Phase 2.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [cache glossary](../../../../glossary.md#term-cache))</span>
3. Pass that exact `CpuLocalWorkloadTuning.SelectedDecision` to
   `CpuCompletePlanTuning.candidateHandoff(...)`. CPU freshly authenticates the Phase-1
   association and generates every retained legal one-partition fusion/split and representation
   alternative around it. No CPU value enters a tools or ordinary public signature.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [signature glossary](../../../../glossary.md#term-signature))</span>
4. Adapt the decision-empty [CPU](../../../../glossary.md#term-cpu) [handoff](../../../../glossary.md#term-handoff) to `CompletePlanTuningRequest`. A package-private typed
   Engine adapter delegates candidate enumeration, compatibility, identity, decision codec, and
   fresh trial/selected preparation to CPU without inspecting private candidate meaning.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [codec glossary](../../../../glossary.md#term-codec))</span>
5. Adapt `RepresentativeExecutionSession.captureCorrectnessReference(...)` and
   `compareCorrectness(...)` to `CompletePlanCorrectness`. Each action first obtains a fresh CPU
   trial recipe. The first candidate supplies the sole exact reference; every later candidate
   compares all ordered canonical publication bytes after result cleanup.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
6. Adapt the session's existing completion-only `execute(...)` to
   `CompleteCandidateMeasurement`. Every warmup and timed action freshly prepares the requested
   complete candidate and runs it in a fresh Runtime <span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [Runtime glossary](../../../../glossary.md#term-runtime))</span>[`RunState`](../../../../glossary.md#run-state--runstate); the tool alone counts and times
   invocations.
7. After tools/tuning authenticates or measures the winner, validate the exact original
   partition/batch association, selected record, rich evidence, and request identities. Only then
   clean the representative session and call `CpuCompletePlanTuning.prepareSelected(...)` once to
   construct the one fresh returned production recipe.
8. `ModelAutotuningRequest` and `Engine.prepareTuned(...)` are sufficient unchanged. The current
   `ModelAutotuningPreparation.Evidence` is not: it represents only Phase-1 workloads and cannot
   truthfully report the correctness actions and measurements that selected the Phase-2
   production recipe. Extend that existing result type narrowly as specified below; add no new
   top-level public type or Engine method.

## [Scope](../../../../glossary.md#term-scope)

- Preserve the exact current public `ModelAutotuningRequest` constructors and accessors and the
  exact `Engine.prepareTuned(CompiledGraph, ModelAutotuningRequest)` [signature](../../../../glossary.md#term-signature).
- Preserve Phase-1 request mapping and evidence semantics exactly. Run Phase 1 once, authenticate
  its selected handoff, and retain its rich evidence for the final result.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Add a package-private [CPU](../../../../glossary.md#term-cpu) complete-plan adapter in [Engine](../../../../glossary.md#term-engine), preferably beside the existing
  `CpuTuningAdapter`, without [widening](../../../../glossary.md#term-widening) `EngineBackendComposition` or creating a generic public
  backend facade.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [facade glossary](../../../../glossary.md#term-facade))</span>
- Extend `CpuEngineBackendComposition` with one package-private accessor for its retained
  `CpuCompletePlanTuning`; shared composition remains free of CPU tuning types.
- Map Config 0006B exhaustively to `CompletePlanTuningRequest`:
  - `completePlanBudget.maximumPlanCandidates` -> `maximumPlanCandidates`;
  - `completePlanBudget.warmupCount` -> Phase-2 `warmupCount`;
  - `completePlanBudget.timedSampleCount` -> Phase-2 `timedSampleCount`;
  - `completePlanBudget.maximumTotalPlanExecutions` ->
    `maximumTotalPlanExecutions`;
  - `completePlanBudget.maximumAggregateCorrectnessBytes` ->
    `maximumAggregateCorrectnessBytes`; and
  - the exact `modelPlanCache` reference -> `modelPlanCache`.
- Map the existing [objective](../../../../glossary.md#term-objective) exhaustively by enum switch and fix correctness to
  `EXACT_CANONICAL_BYTES`. Preserve the existing caller model identity and Config representative
  profile as tool-local snapshots.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Derive a versioned session target fingerprint from the current CPU complete-plan compatibility
  projection. This first producer is <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [producer glossary](../../../../glossary.md#term-producer))</span>`SESSION` scoped, so the fingerprint is transaction evidence
  rather than persistent compatibility authority. Derive one versioned Engine-owned policy
  identity for the fixed one-partition CPU producer, exact correctness policy, and unchanged
  candidate constraints. Use explicit binary tags and versions, never enum names, object text,
  reflection, or graph-local ID magnitudes.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction); [partition glossary](../../../../glossary.md#partition); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [graph glossary](../../../../glossary.md#graph))</span>
- Map CPU compatibility to tool compatibility with explicit Engine-owned producer and decision-
  codec identities. Map reuse scope exhaustively. Current CPU must map to <span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>`SESSION`.
- Pass the explicit [model](../../../../glossary.md#term-model)-plan path even for `SESSION`. Assert through tests that the generic
  transaction performs no query, create, read, write, temporary-file, move, or directory-force
  operation in that mode. Keep the same mapping valid for a future <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>`PERSISTENT` producer; do not
  branch in Engine on the path or emulate cache behavior.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Treat a missing eligible Phase-1 handoff or a missing complete-plan handoff as a recoverable
  tuning-transaction failure after representative-session construction. Required tuning
  propagates it after cleanup; allowed fallback performs exactly one fresh ordinary safe-
  heuristic preparation after cleanup.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Translate and authenticate both phases' evidence before public result construction. A returned
  `TUNED` result must contain the unchanged Phase-1 workload evidence plus complete-plan evidence
  for the recipe that was freshly prepared for production.
- Extend `ModelAutotuningPreparation.Evidence` with one required final component,
  `CompletePlanEvidence completePlan`, and add only these nested public declarations:
  - `CorrectnessAction { REFERENCE_CAPTURED, MATCH }`;
  - `CompletePlanCandidateEvidence(CandidateIdentity identity, CorrectnessAction
    correctnessAction, List<Long> elapsedSamplesNanos, SampleSummary summary)`; and
  - `CompletePlanEvidence(CompatibilityIdentity compatibility,
    ModelAutotuningConfig.CompletePlanBudget budget, Source source,
    List<CompletePlanCandidateEvidence> candidates, CandidateIdentity winnerIdentity,
    SampleSummary winnerSummary)`.
- The new values use the same defensive immutable, source-dependent candidate-list, exact sample
  summary, nanosecond-unit, and opaque-identity rules as the existing Phase-1 evidence. A measured
  complete-plan result has one row per candidate and preserves correctness action and encounter
  order; a cache hit has no candidate rows. Current CPU is session-scoped and therefore measured,
  but the public translation remains valid for future persistent hits.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [opaque glossary](../../../../glossary.md#term-opaque); [cache hit glossary](../../../../glossary.md#cache-and-memoization))</span>
- Add no compatibility constructor for the old five-component `Evidence`: after this task a
  `TUNED` result is incomplete without authenticated complete-plan evidence.
- Preserve all existing session poisoning, once-only cleanup, Engine admission, concurrent close,
  primary/suppressed throwable, and final-publication rules across the combined transaction.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [publication glossary](../../../../glossary.md#term-publication))</span>
- Complete the mandatory separate clean documentation-focused pass in the same future
  implementation change.

## Out of [scope](../../../../glossary.md#term-scope)

- a new public request, preparation result, top-level evidence type, Engine overload, advanced
  Engine method, public backend facade, or public correctness callback<span data-glossary-navigation="true"> ([preparation result glossary](../../../../glossary.md#model-autotuning-request--preparation-result); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [facade glossary](../../../../glossary.md#term-facade); [callback glossary](../../../../glossary.md#term-callback))</span>
- changing Config, tools/tuning, CPU, Prepare, Runtime, Compiler, Planning, Model, Trace, NN, or
  Training production source<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Model glossary](../../../../glossary.md#term-model); [Trace glossary](../../../../glossary.md#trace); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Phase-1 field reinterpretation: `maximumDistinctCacheMisses`,
  `maximumCandidatesPerMiss`, `Budget.warmupCount`, and `Budget.timedSampleCount` remain Phase 1
  only
- repeating Phase-1 local route search, timing a correctness run, reusing a correctness run as a
  warmup/sample, or promoting any trial recipe into production<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- interpreting CPU topology, fusion, representation, materialization, route, thread, provider,
  slot, executable, or codec bytes in Engine or tools<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization); [codec glossary](../../../../glossary.md#term-codec))</span>
- persistent CPU model-plan reuse, invented stable CPU target identity, filesystem access for a
  session-scoped batch, executable serialization, cache inspection, migration, repair, or eviction<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [identity glossary](../../../../glossary.md#term-identity-meanings); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [serialization glossary](../../../../glossary.md#term-serialization); [cache glossary](../../../../glossary.md#term-cache))</span>
- Compiler graph-transformation alternatives, Planning ownership/partition alternatives,
  multiple partitions, mixed backends, Metal/CUDA, tolerance policies, relaxed numerics,
  adaptive sampling, retry, timeout, cancellation, or asynchronous trials<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [partition glossary](../../../../glossary.md#partition); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [tolerance glossary](../../../../glossary.md#term-tolerance); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [cancellation glossary](../../../../glossary.md#term-cancellation); [asynchronous glossary](../../../../glossary.md#term-asynchronous))</span>
- Runtime tuning state, hot-path branches, cache access, graph inspection, service location,
  reflection, string dispatch, generic parameter maps, or registries<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- architecture-contract, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [dependency](../../../../glossary.md#term-dependency), or [Gradle](../../../../glossary.md#term-gradle) changes

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants, Engine,
  Runtime, Prepare, concrete backends, performance evidence and optimization tooling, and
  dependency rules<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [backends glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span> and [roadmap](../../../roadmap.md)
- [Engine master plan](../master-plan.md) and [Engine](../../../../glossary.md#term-engine)
  [0006A](0006a-representative-tuning-execution-and-safe-fallback.md),
  [0007](0007-optional-model-autotuning-composition.md),
  [0008](0008-engine-lifecycle-capability-checkpoint.md), and
  [0008A](0008a-representative-complete-plan-correctness-oracle.md)
- Config [0006A](../../config/tasks/0006a-model-autotuning-request-configuration.md) and
  [0006B](../../config/tasks/0006b-complete-plan-autotuning-request-configuration.md)
- tuning [0001](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md),
  [0002](../../../tools/tuning/tasks/0002-bounded-complete-plan-tuning-and-model-plan-cache.md),
  and [0003](../../../tools/tuning/tasks/0003-read-only-cache-plan-and-evidence-inspection.md)
- [CPU](../../../../glossary.md#term-cpu) [0010I](../../../backends/cpu/tasks/0010i-supported-cpu-local-workload-tuning-composition-adapter.md)
  and [0010J](../../../backends/cpu/tasks/0010j-supported-complete-plan-candidate-and-decision-producer.md)
- [Prepare 0004](../../prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)

## Architecture constraints

- Engine remains the outer composition root and owns representative values, public request/result
  translation, correctness bytes, cleanup, fallback, admission, and production publication.<span data-glossary-navigation="true"> ([composition root glossary](../../../../glossary.md#term-composition-root); [admission glossary](../../../../glossary.md#term-admission); [publication glossary](../../../../glossary.md#term-publication))</span>
- Config remains declarative and independent of tuning, [Engine](../../../../glossary.md#term-engine), and [CPU](../../../../glossary.md#term-cpu).
- tools/tuning owns Phase-2 validation, cache coordination, correctness-before-timing sequencing,
  invocation counts, timing, median selection, typed mismatch, compact record, and rich evidence.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [cache glossary](../../../../glossary.md#term-cache); [median glossary](../../../../glossary.md#term-median))</span>
- CPU owns Phase-1 authentication, the complete candidate set, compatibility, identities,
  decision codec, and fresh exact trial/selected preparation.<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication); [codec glossary](../../../../glossary.md#term-codec))</span>
- Prepare transports the exact decision-empty or decision-present typed handoff without
  interpreting it. Runtime executes only fresh prepared recipes and never tunes.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [handoff glossary](../../../../glossary.md#term-handoff); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Every correctness, warmup, timed, and returned selected-production action uses a fresh CPU
  preparation. Every execution receives fresh Runtime state and finishes result cleanup before
  its callback returns.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [callback glossary](../../../../glossary.md#term-callback))</span>
- Phase 2 preserves the exact Phase-1 winner. No selected complete-plan preparation may fall back
  to local heuristic selection or rerun local search.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- If implementation needs a public Engine callback, an inward-module change, a reverse
  dependency, persistent CPU identity, a widened Prepare role, or an architecture rule change,
  stop and return the task to planning.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [identity glossary](../../../../glossary.md#term-identity-meanings); [planning glossary](../../../../glossary.md#term-planning))</span>

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.engine` — owns the public result evidence and package-private
  composition adapters.

No package is added. `ModelAutotuningPreparation` remains the sole public result type;
`CompletePlanEvidence`, `CompletePlanCandidateEvidence`, and `CorrectnessAction` are nested
because they are inseparable from that result. Package-private adapters remain colocated with
`AdvancedEngine` and the [CPU](../../../../glossary.md#term-cpu) composition.

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths:

- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/CpuEngineBackendComposition.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/ModelAutotuningPreparation.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/package-info.java`

Expected test paths:

- add `modules/engine/src/test/java/io/github/pho001/synaptik/engine/CompletePlanAutotuningCompositionTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/ModelAutotuningCompositionTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSessionTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/EngineTypedPublicShapeTest.java`
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineModelAutotuningIntegrationTest.java`

Expected explanatory documentation paths:

- `docs/api/public-api.md`
- `docs/developer-guide/benchmarking.md`
- `docs/architecture/performance-evidence-and-tuning.md`
- `docs/architecture/runtime-prepare-backend-boundary.md`
- `docs/glossary.md`

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task specification
- `docs/planning/modules/engine/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless contradicted: `ARCHITECTURE.md`, ADRs, other architecture
pages, Config/tuning/[CPU](../../../../glossary.md#term-cpu)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) source and tests, [Gradle](../../../../glossary.md#term-gradle) files, architecture-test source,
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) source, other integration tests, and unrelated documentation/plans.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 18 paths: 5 [Engine](../../../../glossary.md#term-engine) production/[Javadoc](../../../../glossary.md#term-javadoc) paths, 5 Engine/[integration test](../../../../glossary.md#term-integration-test) paths, 5 focused
explanatory documentation paths, and exactly 3 [planning](../../../../glossary.md#term-planning) paths. This is one cohesive Engine
capability: package-private adaptation, public evidence completion, [lifecycle](../../../../glossary.md#lifecycle)/failure tests, and
the public [CPU](../../../../glossary.md#term-cpu)-only integration proof. The fifth production/Javadoc path updates the existing
public `Engine.prepareTuned(...)` contract from its Phase-1-only wording to the implemented
two-phase behavior; it changes no [signature](../../../../glossary.md#term-signature) or executable [token](../../../../glossary.md#term-token). If another production type,
top-level public type, package, [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), build path, explanatory path, or nineteenth path is required, stop and propose
the smallest prerequisite or follow-up instead of expanding the task.

## Failure, cleanup, fallback, and [concurrency](../../../../glossary.md#term-concurrency) semantics

1. Engine admission, compiled-owner validation, request validation, representative storage
   snapshotting, and borrowing retain the exact current order. These failures are never fallback
   inputs.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [admission glossary](../../../../glossary.md#term-admission); [owner glossary](../../../../glossary.md#term-owner); [validation glossary](../../../../glossary.md#term-validation); [borrowing glossary](../../../../glossary.md#term-borrowing))</span>
2. Phase-1 execution/copy-independent trial failure is classified exactly as today. A Runtime
   execution or cleanup failure records exact identity, poisons the session, cleans once, releases
   admission, and never enters Phase 2 or fallback.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [identity glossary](../../../../glossary.md#term-identity-meanings); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
3. Before Phase-2 correctness begins, missing handoff, request/identity translation, candidate
   enumeration, compatibility, cache, codec, budget, or result-authentication failure is
   recoverable only while the session remains open, unpoisoned, admitted, and no representative
   execution failure was recorded.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [cache glossary](../../../../glossary.md#term-cache); [codec glossary](../../../../glossary.md#term-codec); [authentication glossary](../../../../glossary.md#term-authentication))</span>
4. A clean `CorrectnessMismatchException` is normalized to one exact recoverable
   `IllegalStateException` with the mismatch as cause. It runs no warmup/sample and may reach the
   existing public fallback policy only after representative cleanup succeeds.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
5. Phase-2 correctness-copy, Runtime execution, result-count, copied-length, allocation, or
   result-cleanup failure is recognized by exact session failure identity and propagated
   unchanged. It never falls back.
6. Warmup or timed trial preparation failure before session execution is a recoverable tuning
   failure. Once `session.execute(...)` begins, its exact execution/cleanup failure is non-
   recoverable and poisons the session.
7. `IOException` becomes one exact `UncheckedIOException`; any impossible other checked failure
   becomes one exact `IllegalStateException` with cause. Existing unchecked failures and `Error`
   are not rewrapped. `Error` never falls back.
8. After the Phase-2 result is authenticated, representative cleanup, fresh selected preparation,
   evidence translation/construction, owner-bound wrapper construction, or final Engine
   publication failure is not fallback-eligible.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
9. Allowed fallback performs one ordinary safe-heuristic preparation, not a Phase-1-only selected
   preparation. Required tuning performs none. Cleanup failure forbids fallback; fallback failure
   is primary with the recoverable tuning failure suppressed when distinct.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
10. One Engine admission spans both transactions, cleanup, fresh selected/fallback preparation,
    complete public-result construction, and final publication. Engine close waits for admitted
    work; if close wins final publication, no handle or evidence escapes.

## Acceptance criteria

1. Task 0009 completes the sole Engine frontier selected after 0008A; no later Engine task is
   Ready or detailed.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Ready glossary](../../../../glossary.md#term-ready))</span>
2. `ModelAutotuningRequest` and `Engine.prepareTuned(...)` retain their exact public [shape](../../../../glossary.md#shape).
3. `ModelAutotuningPreparation.Evidence` gains exactly the required complete-plan component and
   the three nested declarations listed in Scope; no top-level public type or unrelated method is
   added.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>
4. Phase 1 runs exactly once and its authenticated selected decision is passed unchanged to CPU
   complete-plan candidate production. Phase 2 performs no local route search or Phase-1 timing.<span data-glossary-navigation="true"> ([complete-plan candidate glossary](../../../../glossary.md#complete-plan-candidate))</span>
5. The complete-plan handoff preserves the exact partition/batch and is decision-empty before the
   tool and decision-present afterward. Engine and tools never inspect CPU candidate fields.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [partition glossary](../../../../glossary.md#partition); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
6. Every Config 0006B complete-plan field maps one-for-one to the tool request; every existing
   Phase-1 field remains Phase-1-only. Objective and reuse-scope mapping use exhaustive switches.<span data-glossary-navigation="true"> ([Objective glossary](../../../../glossary.md#term-objective))</span>
7. The fixed exact correctness, target, policy, producer, and codec identities are versioned,
   deterministic within their documented scope, defensively copied, and contain no object text,
   enum name, graph-local ID magnitude, reflection result, or private CPU field decoding.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [producer glossary](../../../../glossary.md#term-producer); [codec glossary](../../../../glossary.md#term-codec); [graph glossary](../../../../glossary.md#graph))</span>
8. Current [CPU](../../../../glossary.md#term-cpu) `SESSION` behavior passes the exact explicit cache path but performs zero model-
   plan-cache filesystem operations. Synthetic typed tests keep the adapter correct for a future
   persistent hit without claiming current CPU persistence.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [model glossary](../../../../glossary.md#term-model); [persistence glossary](../../../../glossary.md#term-persistence))</span>
9. All candidates finish correctness and cleanup before any warmup or timed action. Correctness,
   warmup, timed, and selected production preparation are fresh; Runtime state is fresh per
   execution; no trial recipe is returned.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
10. Engine 0008A's first reference and later exact comparisons preserve publication order,
    aliases, empty values, signed zero, NaN payloads, aggregate-byte preflight, and once-only
    cleanup without exporting bytes.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [preflight glossary](../../../../glossary.md#term-preflight))</span>
11. The winner is authenticated against result handoff, selected record, compatibility, rich
    evidence, model/profile/target/objective/correctness/policy identities, and Config budget
    before fresh production preparation.
12. Public tuned evidence preserves unchanged Phase-1 evidence and immutable complete-plan
    compatibility, budget, source, correctness actions, raw timed samples, summaries, and winner.
    Fallback remains explicit and carries no evidence.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
13. Required and allowed fallback, missing Phase-1/Phase-2 handoffs, typed mismatch, preflight,
    cache/codec, trial preparation, execution/copy/cleanup, selected preparation, `Error`, final
    publication, and concurrent close follow the exact failure rules above with deterministic
    primary/suppressed identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
14. Runtime, Config, tuning, CPU, Prepare, Compiler, Planning, Model, build files, and module
    dependencies remain unchanged; architecture tests continue to prove the one-way<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [module glossary](../../../../glossary.md#term-module))</span>
    `engine -> tools/tuning` edge and inward independence.
15. Native-free focused tests cover mapping, order/counts, no-I/O SESSION behavior, exact
    handoffs, evidence, missing batches, mismatch, poisoning, suppression, fallback, and close
    races. One public standard-CPU integration test proves a real eligible complete-plan path when
    the fixture supplies at least two CPU alternatives; deterministic no-handoff fallback remains
    covered separately.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test); [fixture glossary](../../../../glossary.md#term-fixture))</span>
16. Javadocs and focused documentation distinguish both phases, current CPU session scope,
    compact cache versus rich evidence, exact correctness, fresh production preparation, and
    current-versus-future boundaries. Glossary impact is finalized.
17. Focused Engine, architecture, integration, Javadoc, repository, Markdown, public-shape,
    import/dependency, exact-scope, empty-staging, and diff validation all pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [dependency glossary](../../../../glossary.md#term-dependency); [validation glossary](../../../../glossary.md#term-validation))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation [run](../../../../glossary.md#term-run) focused [Engine](../../../../glossary.md#term-engine) tests, then the affected [module](../../../../glossary.md#term-module):

```bash
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.CompletePlanAutotuningCompositionTest \
  --tests io.github.pho001.synaptik.engine.ModelAutotuningCompositionTest \
  --tests io.github.pho001.synaptik.engine.RepresentativeExecutionSessionTest \
  --tests io.github.pho001.synaptik.engine.api.EngineTypedPublicShapeTest
./gradlew :modules:engine:test
```

Run the focused cross-module checks:

```bash
./gradlew :testing:architecture-tests:test \
  --tests io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest
./gradlew :testing:integration-tests:test \
  --tests io.github.pho001.synaptik.testing.integration.EngineModelAutotuningIntegrationTest
```

After executable Java stabilizes, run one repository [checkpoint](../../../../glossary.md#term-checkpoint):

```bash
./gradlew test
```

The repository run is justified because this task changes the public Engine evidence [shape](../../../../glossary.md#shape) and
composes Config, tuning, [CPU](../../../../glossary.md#term-cpu), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), and Engine in one end-to-end [lifecycle](../../../../glossary.md#lifecycle). Do not run
separate CPU, tuning, Runtime, Prepare, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), provider, or performance suites unless
a concrete failure points to them; their production contracts do not change.

The mandatory clean documentation context reuses successful Java evidence unless it changes
executable behavior, then runs:

```bash
./gradlew :modules:engine:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
git diff --cached --check
git status --short -uall
```

Render and inspect the Engine package, `ModelAutotuningPreparation`, and all new nested evidence
pages. Validate local Markdown [targets](../../../../glossary.md#term-target)/anchors, unique headings, fences, LF/final newlines,
trailing whitespace, terminology, exact 18-path ceiling, public shape, forbidden imports,
[dependency](../../../../glossary.md#term-dependency) inventory, synchronized status, and empty staging.

## Dependencies

- [Engine](../../../../glossary.md#term-engine) 0006A–0008A — Complete.
- Config 0006A–0006B — Complete.
- tools/tuning 0001–0003 — Complete; 0002 supplies the Phase-2 [transaction](../../../../glossary.md#term-transaction).
- [CPU](../../../../glossary.md#term-cpu) 0010I–0010J — Complete.
- [Prepare](../../../../glossary.md#prepare) 0004 and [Runtime](../../../../glossary.md#term-runtime) 0010/0015 — Complete.

No new upstream task, architecture decision, persistent CPU [fingerprint](../../../../glossary.md#term-fingerprint), [Compiler](../../../../glossary.md#term-compiler)/[Planning](../../../../glossary.md#term-planning)
candidate [producer](../../../../glossary.md#term-producer), or Config extension is required for this session-scoped first composition.

## Follow-up tasks

- None is made Ready by this task. Persistent CPU complete-plan compatibility, Compiler graph
  alternatives, Planning ownership alternatives, multiple partitions, mixed backends, tolerance
  policies, and executable serialization remain concise future master-plan concerns until their
  owners expose stable contracts.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [CPU glossary](../../../../glossary.md#term-cpu); [Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [backends glossary](../../../../glossary.md#backend); [tolerance glossary](../../../../glossary.md#term-tolerance); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- A later capability checkpoint may consolidate broader evidence only after another cohesive
  frontier exists; do not create it in this change.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Architecture impact

Expected impact: None.

This task realizes the existing [Engine](../../../../glossary.md#term-engine) composition-root, tools/tuning [transaction](../../../../glossary.md#term-transaction), [CPU](../../../../glossary.md#term-cpu) [producer](../../../../glossary.md#term-producer),
[Prepare](../../../../glossary.md#prepare) transport, [Runtime](../../../../glossary.md#term-runtime) execution, and Config declaration boundaries. It changes one existing
public Engine evidence value because the current Phase-1-only [shape](../../../../glossary.md#shape) cannot truthfully describe the
Phase-2 selection that now determines the returned recipe; it changes no [owner](../../../../glossary.md#term-owner), [dependency](../../../../glossary.md#term-dependency) rule,
or [lifecycle](../../../../glossary.md#lifecycle) contract. If implementation requires an authoritative change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik Engine task 0009. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/modules/engine/master-plan.md, and
docs/planning/modules/engine/tasks/0009-public-complete-plan-autotuning-composition.md in full,
plus every directly referenced completed Engine, Config, tuning, CPU, Prepare, Runtime,
documentation, and architecture contract named by the task. Inspect current source and tests;
do not rely on planning claims alone.

Implement exactly task 0009 within its 18-path ceiling. Preserve Phase-1 semantics, pass its exact
authenticated decision into CPU complete-plan generation, adapt Engine correctness and fresh
execution to the generic Phase-2 transaction, freshly prepare only the authenticated winner, and
preserve lifecycle/fallback/failure identity. Add only the specified nested complete-plan evidence
surface. Stop for architecture, dependency, inward-contract, persistent-identity, public-shape,
or scope conflict.

After executable validation stabilizes, hand the exact diff and recorded evidence to a distinct
clean documentation-focused context in the same overall change. That context must follow
docs/developer-guide/documentation-rules.md, independently inspect final source/tests, finalize
Javadocs and the five focused explanatory paths, review glossary impact, and avoid repeating
successful Java suites absent executable change or a concrete stale-evidence risk. Do not mark
the task Complete until every specified gate and that pass succeed.
```

## Local decisions

- Reuse the current public request and operation. Config 0006B already placed every Phase-2
  caller input behind the retained request's Config reference.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Extend the existing result evidence instead of inventing another public result or method. The
  three nested declarations are the minimum truthful outward representation of Phase-2
  correctness and timing.
- Use CPU's session compatibility projection as the current transaction's target evidence. Do
  not claim a stable persistent CPU target fingerprint.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [transaction glossary](../../../../glossary.md#term-transaction); [target glossary](../../../../glossary.md#term-target); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- Keep target/policy/producer/codec identity derivation package-private. Public evidence reports
  the authenticated compatibility, caller budget, correctness actions, timing, and winner rather
  than exporting cache-key plumbing.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [codec glossary](../../../../glossary.md#term-codec); [identity glossary](../../../../glossary.md#term-identity-meanings); [cache glossary](../../../../glossary.md#term-cache))</span>
- Missing complete alternatives is tuning unavailability, not successful Phase-1-only tuning.
  The existing public fallback policy governs it.
- Ordinary fallback remains the single safe cross-phase recovery recipe. A partially completed
  tuning phase never becomes production state.

## Known limitations

- CPU-only, exactly one non-empty maximal CPU partition, one representative input set, and at most
  one eligible Phase-1 local workload.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition))</span>
- Phase-2 candidates vary only CPU 0010J's retained fusion/split and materialization alternatives
  around the fixed compile graph, owner, partition, publications, and Phase-1 route decision.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization); [compile glossary](../../../../glossary.md#compile); [graph glossary](../../../../glossary.md#graph); [owner glossary](../../../../glossary.md#term-owner))</span>
- CPU compatibility is session-scoped, so every public call measures Phase 2 and performs no
  model-plan-cache I/O. The explicit path is retained for the generic future persistent case.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [cache glossary](../../../../glossary.md#term-cache))</span>
- Exact canonical represented-byte correctness and minimum median elapsed nanoseconds are the only
  supported Phase-2 policies.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [median glossary](../../../../glossary.md#term-median))</span>
- No [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable), [Runtime](../../../../glossary.md#term-runtime) state, representative bytes, or rich evidence is persisted.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `01a0bdf6-5615-7491-8f78-39184c10a773` recorded passing focused
  `CompletePlanAutotuningCompositionTest`, `ModelAutotuningCompositionTest`,
  `RepresentativeExecutionSessionTest`, and `EngineTypedPublicShapeTest`, followed by the full
  `:modules:engine:test` suite. This documentation context changed no executable Java behavior or
  tests and therefore reused that evidence.
- The [handoff](../../../../glossary.md#term-handoff) also recorded passing focused `EngineCompositionContractTest` and focused
  `EngineModelAutotuningIntegrationTest`. The integration class ran two tests: the eligible native
  path skipped on this host because no qualified OpenBLAS alternatives were available, while the
  deterministic no-handoff fallback path executed successfully.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- The initial implementation repository [checkpoint](../../../../glossary.md#term-checkpoint) `./gradlew test` was `BUILD SUCCESSFUL` in
  2m42s with 70 actionable tasks: 10 executed, 1 from cache, and 59 up-to-date. After the bounded
  compatibility-snapshot and public-method-Javadoc correction, implementation context<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [snapshot glossary](../../../../glossary.md#term-snapshot); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
  `01a0bdf6-5615-7491-8f78-39184c10a773` reran and passed the focused four [Engine](../../../../glossary.md#term-engine) tests, full
  `:modules:engine:test`, `:modules:engine:javadoc`, and final `./gradlew test`; the final root run
  reported 70 actionable tasks, 2 executed and 68 up-to-date. Diff checks passed, exact scope was
  18 paths, and staging was empty.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [scope glossary](../../../../glossary.md#term-scope))</span>
- Mandatory clean documentation-focused review context
  `01a0be0e-f34a-7b41-b5a6-70792f85b878` read the architecture and planning contracts,
  documentation rules and selected profiles, directly relevant predecessor tasks, final source
  and tests, and the complete worktree diff. It found no executable defect and made only Javadoc,
  Markdown, and planning changes.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation context `01a0be0e-f34a-7b41-b5a6-70792f85b878` ran the final Engine Javadoc,
  repository Markdown validation, rendered-page inspection, public-shape/no-extra-type review,
  exact-scope and forbidden-import/dependency inventories, whitespace/diff checks, and staging
  checks recorded below. No Java test suite was rerun because executable behavior did not change.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Final `./gradlew :modules:engine:javadoc` was `BUILD SUCCESSFUL` in 2s with 15 actionable
  tasks: 2 executed and 13 up-to-date. Rendered text from the Engine package,
  `ModelAutotuningPreparation`, `CorrectnessAction`, `CompletePlanCandidateEvidence`, and
  `CompletePlanEvidence` pages showed both phases, exact correctness-before-timing, current
  session scope/no model-plan-cache I/O, fresh preparation, component semantics, and constructor
  failures.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- After the bounded correction, the implementation context reran Engine Javadoc successfully.
  Documentation context `01a0be0e-f34a-7b41-b5a6-70792f85b878` inspected the resulting rendered
  `Engine.prepareTuned(...)` method plus the previously affected package and evidence pages. The
  public method now renders both phases, exact correctness before timing, fresh authenticated
  winner preparation, and explicit allowed safe fallback with no evidence. No further Java edit
  or Javadoc rerun was needed in this resumed documentation pass.
- The temporary `/tmp/validate_synaptik_markdown.py` validator passed all eight changed Markdown
  files for local targets and anchors, unique canonical headings, balanced fences, LF endings,
  final newlines, and trailing whitespace. <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`javap` confirmed the required six-component
  `Evidence`, exactly the three new nested public declarations and their record/enum shape, and no
  extra public top-level type.
- Final scope inspection found exactly 18 paths: 17 tracked modifications plus the one intended
  new Engine test. Production import inventory contained only the expected CPU complete-plan and
  tools/tuning contracts; no production reflection, filesystem API, new module dependency, or
  build file appeared. <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>`git diff --check` and `git diff --cached --check` passed, staging was
  empty, and final `git status --short -uall` listed only the exact 18 authorized paths.

## Implementation notes

- Phase 1 runs once and its authenticated selected decision passes unchanged into CPU complete-
  plan candidate production. Engine does not prepare a Phase-1 winner for production or repeat
  local search in Phase 2.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [prepare glossary](../../../../glossary.md#prepare))</span>
- Phase 2 maps every Config 0006B field and the exact model-plan path, completes exact canonical-
  byte correctness for all candidates before timing, and freshly prepares each correctness,
  warmup, and timed action. After winner authentication and representative cleanup, exactly one
  further fresh selected recipe becomes production state.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [canonical glossary](../../../../glossary.md#term-canonical-form); [warmup glossary](../../../../glossary.md#term-warmup); [authentication glossary](../../../../glossary.md#term-authentication))</span>
- Engine obtains complete-plan compatibility exactly once per transaction. The same non-null
  snapshot supplies target/request construction, the generic Phase-2 backend adapter, and result
  authentication, preventing one transaction from observing divergent producer projections.
  Focused synthetic coverage counts and proves exactly one compatibility projection for each
  successful transaction.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction); [snapshot glossary](../../../../glossary.md#term-snapshot); [target glossary](../../../../glossary.md#term-target); [backend glossary](../../../../glossary.md#backend); [producer glossary](../../../../glossary.md#term-producer))</span>
- Current [CPU](../../../../glossary.md#term-cpu) complete-plan compatibility is `SESSION`; the generic request receives the exact
  path but performs no model-plan-cache filesystem I/O. The translation and public evidence also
  represent a future authenticated <span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>`PERSISTENT` producer without claiming current persistent CPU
  reuse.
- `ModelAutotuningPreparation.Evidence` now requires `completePlan` and adds exactly the nested
  `CorrectnessAction`, `CompletePlanCandidateEvidence`, and `CompletePlanEvidence` declarations.
  Measured evidence retains immutable correctness actions and raw samples per candidate, while a
  cache hit has no candidate rows; compact reusable records remain separate from rich evidence.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [cache hit glossary](../../../../glossary.md#cache-and-memoization))</span>
- Existing strict/allowed fallback, poisoned representative execution, cleanup suppression,
  selected-preparation, and close-race semantics remain unchanged. Fallback is one fresh ordinary
  safe preparation and carries no evidence.<span data-glossary-navigation="true"> ([race glossary](../../../../glossary.md#term-race))</span>
- The integration fixture's reflection is test-only construction of resolved MATMUL compile
  artifacts without adding an integration-module dependency. It does not alter production
  Runtime behavior, dependency direction, or documentation claims and does not contradict this
  task or the architecture contract.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [compile artifacts glossary](../../../../glossary.md#compile-artifacts); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Runtime glossary](../../../../glossary.md#term-runtime); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- Documentation review selected General plus API/Javadoc for public API and Java contracts,
  General plus Developer guide and Example for benchmarking, General plus Architecture for the
  two architecture explanations, General for the glossary, and General plus Planning for the
  task, master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [benchmarking glossary](../../../../glossary.md#term-benchmarking); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Architecture/ADR no-change conclusion: task 0009 realizes the existing Engine composition-root
  boundary and changes no authority, dependency rule, lifecycle order, or architectural decision.
  Dependency/build/Gradle, Runtime, Config, tuning, CPU, Prepare, Compiler, Planning, Model,
  Tensor, Training, architecture-test, backend-conformance, provider, and other-module contracts
  required no change.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [lifecycle glossary](../../../../glossary.md#lifecycle); [Gradle glossary](../../../../glossary.md#term-gradle); [Compiler glossary](../../../../glossary.md#term-compiler); [Tensor glossary](../../../../glossary.md#tensor); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The five explanatory documents remain accurate after the one-snapshot correction because they
  describe the authenticated compatibility boundary, not the number of internal projection
  calls. The synchronized Engine master plan and roadmap already state Complete through 0009 and
  require no additional correction beyond their existing changed paths.

## Completion summary

- Completed changes: composed and documented the bounded two-phase public CPU workflow, exact
  correctness-before-timing order, fresh per-action and final preparation, authenticated rich
  evidence, current session-only/no-model-plan-I/O behavior, future persistent translation, and
  unchanged fallback and poisoning semantics.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [model glossary](../../../../glossary.md#term-model))</span>
- Production/[Javadoc](../../../../glossary.md#term-javadoc) files reviewed: `AdvancedEngine.java`,
  `CpuEngineBackendComposition.java`, `Engine.java`, `ModelAutotuningPreparation.java`, and [Engine](../../../../glossary.md#term-engine)
  `package-info.java`. The documentation work finalized Javadocs in `AdvancedEngine.java`,
  `Engine.java`, and `ModelAutotuningPreparation.java`; the other two already-changed Javadoc
  owners remained accurate after review.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>
- Explanatory documentation finalized: `docs/api/public-api.md`,
  `docs/developer-guide/benchmarking.md`,
  `docs/architecture/performance-evidence-and-tuning.md`,
  `docs/architecture/runtime-prepare-backend-boundary.md`, and `docs/glossary.md`.
- [Planning](../../../../glossary.md#term-planning) finalized: this task, `docs/planning/modules/engine/master-plan.md`, and
  `docs/planning/roadmap.md`. Engine is Complete through 0009; no later Engine task became Ready
  and no detailed task was created.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Final combined scope is exactly 18 paths: 5 production/Javadoc paths, 5 test paths, 5
  explanatory documentation paths, and 3 planning paths. No file was staged, committed, or
  pushed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Limitations remain deliberate: one representative input set, at most one local workload, one
  non-empty maximal CPU partition, CPU-owned retained topology/representation alternatives,
  exact canonical-byte correctness, session-scoped CPU reuse, and no broader graph/partition,
  mixed-backend, tolerance, or executable-persistence capability.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [topology glossary](../../../../glossary.md#term-topology); [canonical glossary](../../../../glossary.md#term-canonical-form); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [tolerance glossary](../../../../glossary.md#term-tolerance); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Implementation context: `01a0bdf6-5615-7491-8f78-39184c10a773`.
- Documentation review context: `01a0be0e-f34a-7b41-b5a6-70792f85b878`.

Status: Complete
