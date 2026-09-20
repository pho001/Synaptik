# Task 0009: Public Complete-Plan Autotuning Composition

## Status

Ready

## Goal

Extend the existing public CPU-only `Engine.prepareTuned(...)` operation from its completed
Phase-1 local-workload transaction through the completed generic Phase-2 complete-plan
transaction. Engine must preserve the authenticated Phase-1 decision, obtain CPU 0010J's bounded
complete-plan batch around that decision, adapt Engine 0008A's exact representative correctness
oracle and fresh representative execution to tools/tuning 0002, and return one freshly prepared
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
algorithm, backend candidate meaning, Runtime behavior, persistent CPU reuse, Compiler graph
alternative, Planning ownership alternative, or public Engine operation.

## Source-backed dataflow and current-contract conclusion

1. `AdvancedEngine.prepareTunedOrdinary(...)` already owns the sole Engine admission, owner and
   request validation, representative binding, Phase-1 request translation, failure isolation,
   strict-versus-safe fallback, fresh production wrapping, and final close-race publication.
2. Phase 1 must run first through `WorkloadTuning.tune(...)`. Its decision-present selected
   handoff is authenticated against the exact original partition and batch. Do not call
   `CpuLocalWorkloadTuning.prepareSelected(...)` before Phase 2 and do not repeat Phase-1 candidate
   enumeration, warmups, samples, ranking, or cache work inside Phase 2.
3. Pass that exact `CpuLocalWorkloadTuning.SelectedDecision` to
   `CpuCompletePlanTuning.candidateHandoff(...)`. CPU freshly authenticates the Phase-1
   association and generates every retained legal one-partition fusion/split and representation
   alternative around it. No CPU value enters a tools or ordinary public signature.
4. Adapt the decision-empty CPU handoff to `CompletePlanTuningRequest`. A package-private typed
   Engine adapter delegates candidate enumeration, compatibility, identity, decision codec, and
   fresh trial/selected preparation to CPU without inspecting private candidate meaning.
5. Adapt `RepresentativeExecutionSession.captureCorrectnessReference(...)` and
   `compareCorrectness(...)` to `CompletePlanCorrectness`. Each action first obtains a fresh CPU
   trial recipe. The first candidate supplies the sole exact reference; every later candidate
   compares all ordered canonical publication bytes after result cleanup.
6. Adapt the session's existing completion-only `execute(...)` to
   `CompleteCandidateMeasurement`. Every warmup and timed action freshly prepares the requested
   complete candidate and runs it in a fresh Runtime `RunState`; the tool alone counts and times
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

## Scope

- Preserve the exact current public `ModelAutotuningRequest` constructors and accessors and the
  exact `Engine.prepareTuned(CompiledGraph, ModelAutotuningRequest)` signature.
- Preserve Phase-1 request mapping and evidence semantics exactly. Run Phase 1 once, authenticate
  its selected handoff, and retain its rich evidence for the final result.
- Add a package-private CPU complete-plan adapter in Engine, preferably beside the existing
  `CpuTuningAdapter`, without widening `EngineBackendComposition` or creating a generic public
  backend facade.
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
- Map the existing objective exhaustively by enum switch and fix correctness to
  `EXACT_CANONICAL_BYTES`. Preserve the existing caller model identity and Config representative
  profile as tool-local snapshots.
- Derive a versioned session target fingerprint from the current CPU complete-plan compatibility
  projection. This first producer is `SESSION` scoped, so the fingerprint is transaction evidence
  rather than persistent compatibility authority. Derive one versioned Engine-owned policy
  identity for the fixed one-partition CPU producer, exact correctness policy, and unchanged
  candidate constraints. Use explicit binary tags and versions, never enum names, object text,
  reflection, or graph-local ID magnitudes.
- Map CPU compatibility to tool compatibility with explicit Engine-owned producer and decision-
  codec identities. Map reuse scope exhaustively. Current CPU must map to `SESSION`.
- Pass the explicit model-plan path even for `SESSION`. Assert through tests that the generic
  transaction performs no query, create, read, write, temporary-file, move, or directory-force
  operation in that mode. Keep the same mapping valid for a future `PERSISTENT` producer; do not
  branch in Engine on the path or emulate cache behavior.
- Treat a missing eligible Phase-1 handoff or a missing complete-plan handoff as a recoverable
  tuning-transaction failure after representative-session construction. Required tuning
  propagates it after cleanup; allowed fallback performs exactly one fresh ordinary safe-
  heuristic preparation after cleanup.
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
  but the public translation remains valid for future persistent hits.
- Add no compatibility constructor for the old five-component `Evidence`: after this task a
  `TUNED` result is incomplete without authenticated complete-plan evidence.
- Preserve all existing session poisoning, once-only cleanup, Engine admission, concurrent close,
  primary/suppressed throwable, and final-publication rules across the combined transaction.
- Complete the mandatory separate clean documentation-focused pass in the same future
  implementation change.

## Out of scope

- a new public request, preparation result, top-level evidence type, Engine overload, advanced
  Engine method, public backend facade, or public correctness callback
- changing Config, tools/tuning, CPU, Prepare, Runtime, Compiler, Planning, Model, Trace, NN, or
  Training production source
- Phase-1 field reinterpretation: `maximumDistinctCacheMisses`,
  `maximumCandidatesPerMiss`, `Budget.warmupCount`, and `Budget.timedSampleCount` remain Phase 1
  only
- repeating Phase-1 local route search, timing a correctness run, reusing a correctness run as a
  warmup/sample, or promoting any trial recipe into production
- interpreting CPU topology, fusion, representation, materialization, route, thread, provider,
  slot, executable, or codec bytes in Engine or tools
- persistent CPU model-plan reuse, invented stable CPU target identity, filesystem access for a
  session-scoped batch, executable serialization, cache inspection, migration, repair, or eviction
- Compiler graph-transformation alternatives, Planning ownership/partition alternatives,
  multiple partitions, mixed backends, Metal/CUDA, tolerance policies, relaxed numerics,
  adaptive sampling, retry, timeout, cancellation, or asynchronous trials
- Runtime tuning state, hot-path branches, cache access, graph inspection, service location,
  reflection, string dispatch, generic parameter maps, or registries
- architecture-contract, ADR, dependency, or Gradle changes

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants, Engine,
  Runtime, Prepare, concrete backends, performance evidence and optimization tooling, and
  dependency rules
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Planning guide](../../../planning-guide.md) and [roadmap](../../../roadmap.md)
- [Engine master plan](../master-plan.md) and Engine
  [0006A](0006a-representative-tuning-execution-and-safe-fallback.md),
  [0007](0007-optional-model-autotuning-composition.md),
  [0008](0008-engine-lifecycle-capability-checkpoint.md), and
  [0008A](0008a-representative-complete-plan-correctness-oracle.md)
- Config [0006A](../../config/tasks/0006a-model-autotuning-request-configuration.md) and
  [0006B](../../config/tasks/0006b-complete-plan-autotuning-request-configuration.md)
- tuning [0001](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md),
  [0002](../../../tools/tuning/tasks/0002-bounded-complete-plan-tuning-and-model-plan-cache.md),
  and [0003](../../../tools/tuning/tasks/0003-read-only-cache-plan-and-evidence-inspection.md)
- CPU [0010I](../../../backends/cpu/tasks/0010i-supported-cpu-local-workload-tuning-composition-adapter.md)
  and [0010J](../../../backends/cpu/tasks/0010j-supported-complete-plan-candidate-and-decision-producer.md)
- [Prepare 0004](../../prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)

## Architecture constraints

- Engine remains the outer composition root and owns representative values, public request/result
  translation, correctness bytes, cleanup, fallback, admission, and production publication.
- Config remains declarative and independent of tuning, Engine, and CPU.
- tools/tuning owns Phase-2 validation, cache coordination, correctness-before-timing sequencing,
  invocation counts, timing, median selection, typed mismatch, compact record, and rich evidence.
- CPU owns Phase-1 authentication, the complete candidate set, compatibility, identities,
  decision codec, and fresh exact trial/selected preparation.
- Prepare transports the exact decision-empty or decision-present typed handoff without
  interpreting it. Runtime executes only fresh prepared recipes and never tunes.
- Every correctness, warmup, timed, and returned selected-production action uses a fresh CPU
  preparation. Every execution receives fresh Runtime state and finishes result cleanup before
  its callback returns.
- Phase 2 preserves the exact Phase-1 winner. No selected complete-plan preparation may fall back
  to local heuristic selection or rerun local search.
- If implementation needs a public Engine callback, an inward-module change, a reverse
  dependency, persistent CPU identity, a widened Prepare role, or an architecture rule change,
  stop and return the task to planning.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.engine` — owns the public result evidence and package-private
  composition adapters.

No package is added. `ModelAutotuningPreparation` remains the sole public result type;
`CompletePlanEvidence`, `CompletePlanCandidateEvidence`, and `CorrectnessAction` are nested
because they are inseparable from that result. Package-private adapters remain colocated with
`AdvancedEngine` and the CPU composition.

## Affected files

Expected production/Javadoc paths:

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

Expected planning paths:

- this task specification
- `docs/planning/modules/engine/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless contradicted: `ARCHITECTURE.md`, ADRs, other architecture
pages, Config/tuning/CPU/Prepare/Runtime source and tests, Gradle files, architecture-test source,
backend-conformance source, other integration tests, and unrelated documentation/plans.

## Maximum scope

At most 17 paths: 4 Engine production/Javadoc paths, 5 Engine/integration test paths, 5 focused
explanatory documentation paths, and exactly 3 planning paths. This is one cohesive Engine
capability: package-private adaptation, public evidence completion, lifecycle/failure tests, and
the public CPU-only integration proof. If another production type, top-level public type, package,
module dependency, build path, explanatory path, or eighteenth path is required, stop and propose
the smallest prerequisite or follow-up instead of expanding the task.

## Failure, cleanup, fallback, and concurrency semantics

1. Engine admission, compiled-owner validation, request validation, representative storage
   snapshotting, and borrowing retain the exact current order. These failures are never fallback
   inputs.
2. Phase-1 execution/copy-independent trial failure is classified exactly as today. A Runtime
   execution or cleanup failure records exact identity, poisons the session, cleans once, releases
   admission, and never enters Phase 2 or fallback.
3. Before Phase-2 correctness begins, missing handoff, request/identity translation, candidate
   enumeration, compatibility, cache, codec, budget, or result-authentication failure is
   recoverable only while the session remains open, unpoisoned, admitted, and no representative
   execution failure was recorded.
4. A clean `CorrectnessMismatchException` is normalized to one exact recoverable
   `IllegalStateException` with the mismatch as cause. It runs no warmup/sample and may reach the
   existing public fallback policy only after representative cleanup succeeds.
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
   publication failure is not fallback-eligible.
9. Allowed fallback performs one ordinary safe-heuristic preparation, not a Phase-1-only selected
   preparation. Required tuning performs none. Cleanup failure forbids fallback; fallback failure
   is primary with the recoverable tuning failure suppressed when distinct.
10. One Engine admission spans both transactions, cleanup, fresh selected/fallback preparation,
    complete public-result construction, and final publication. Engine close waits for admitted
    work; if close wins final publication, no handle or evidence escapes.

## Acceptance criteria

1. Task 0009 is the sole Ready Engine frontier after Complete 0008A; no later Engine task is Ready
   or detailed.
2. `ModelAutotuningRequest` and `Engine.prepareTuned(...)` retain their exact public shape.
3. `ModelAutotuningPreparation.Evidence` gains exactly the required complete-plan component and
   the three nested declarations listed in Scope; no top-level public type or unrelated method is
   added.
4. Phase 1 runs exactly once and its authenticated selected decision is passed unchanged to CPU
   complete-plan candidate production. Phase 2 performs no local route search or Phase-1 timing.
5. The complete-plan handoff preserves the exact partition/batch and is decision-empty before the
   tool and decision-present afterward. Engine and tools never inspect CPU candidate fields.
6. Every Config 0006B complete-plan field maps one-for-one to the tool request; every existing
   Phase-1 field remains Phase-1-only. Objective and reuse-scope mapping use exhaustive switches.
7. The fixed exact correctness, target, policy, producer, and codec identities are versioned,
   deterministic within their documented scope, defensively copied, and contain no object text,
   enum name, graph-local ID magnitude, reflection result, or private CPU field decoding.
8. Current CPU `SESSION` behavior passes the exact explicit cache path but performs zero model-
   plan-cache filesystem operations. Synthetic typed tests keep the adapter correct for a future
   persistent hit without claiming current CPU persistence.
9. All candidates finish correctness and cleanup before any warmup or timed action. Correctness,
   warmup, timed, and selected production preparation are fresh; Runtime state is fresh per
   execution; no trial recipe is returned.
10. Engine 0008A's first reference and later exact comparisons preserve publication order,
    aliases, empty values, signed zero, NaN payloads, aggregate-byte preflight, and once-only
    cleanup without exporting bytes.
11. The winner is authenticated against result handoff, selected record, compatibility, rich
    evidence, model/profile/target/objective/correctness/policy identities, and Config budget
    before fresh production preparation.
12. Public tuned evidence preserves unchanged Phase-1 evidence and immutable complete-plan
    compatibility, budget, source, correctness actions, raw timed samples, summaries, and winner.
    Fallback remains explicit and carries no evidence.
13. Required and allowed fallback, missing Phase-1/Phase-2 handoffs, typed mismatch, preflight,
    cache/codec, trial preparation, execution/copy/cleanup, selected preparation, `Error`, final
    publication, and concurrent close follow the exact failure rules above with deterministic
    primary/suppressed identity.
14. Runtime, Config, tuning, CPU, Prepare, Compiler, Planning, Model, build files, and module
    dependencies remain unchanged; architecture tests continue to prove the one-way
    `engine -> tools/tuning` edge and inward independence.
15. Native-free focused tests cover mapping, order/counts, no-I/O SESSION behavior, exact
    handoffs, evidence, missing batches, mismatch, poisoning, suppression, fallback, and close
    races. One public standard-CPU integration test proves a real eligible complete-plan path when
    the fixture supplies at least two CPU alternatives; deterministic no-handoff fallback remains
    covered separately.
16. Javadocs and focused documentation distinguish both phases, current CPU session scope,
    compact cache versus rich evidence, exact correctness, fresh production preparation, and
    current-versus-future boundaries. Glossary impact is finalized.
17. Focused Engine, architecture, integration, Javadoc, repository, Markdown, public-shape,
    import/dependency, exact-scope, empty-staging, and diff validation all pass.

## Tests / validation

During implementation run focused Engine tests, then the affected module:

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

After executable Java stabilizes, run one repository checkpoint:

```bash
./gradlew test
```

The repository run is justified because this task changes the public Engine evidence shape and
composes Config, tuning, CPU, Prepare, Runtime, and Engine in one end-to-end lifecycle. Do not run
separate CPU, tuning, Runtime, Prepare, backend-conformance, provider, or performance suites unless
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
pages. Validate local Markdown targets/anchors, unique headings, fences, LF/final newlines,
trailing whitespace, terminology, exact 17-path ceiling, public shape, forbidden imports,
dependency inventory, synchronized status, and empty staging.

## Dependencies

- Engine 0006A–0008A — Complete.
- Config 0006A–0006B — Complete.
- tools/tuning 0001–0003 — Complete; 0002 supplies the Phase-2 transaction.
- CPU 0010I–0010J — Complete.
- Prepare 0004 and Runtime 0010/0015 — Complete.

No new upstream task, architecture decision, persistent CPU fingerprint, Compiler/Planning
candidate producer, or Config extension is required for this session-scoped first composition.

## Follow-up tasks

- None is made Ready by this task. Persistent CPU complete-plan compatibility, Compiler graph
  alternatives, Planning ownership alternatives, multiple partitions, mixed backends, tolerance
  policies, and executable serialization remain concise future master-plan concerns until their
  owners expose stable contracts.
- A later capability checkpoint may consolidate broader evidence only after another cohesive
  frontier exists; do not create it in this change.

## Architecture impact

Expected impact: None.

This task realizes the existing Engine composition-root, tools/tuning transaction, CPU producer,
Prepare transport, Runtime execution, and Config declaration boundaries. It changes one existing
public Engine evidence value because the current Phase-1-only shape cannot truthfully describe the
Phase-2 selection that now determines the returned recipe; it changes no owner, dependency rule,
or lifecycle contract. If implementation requires an authoritative change, stop and report it.

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

Implement exactly task 0009 within its 17-path ceiling. Preserve Phase-1 semantics, pass its exact
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
  caller input behind the retained request's Config reference.
- Extend the existing result evidence instead of inventing another public result or method. The
  three nested declarations are the minimum truthful outward representation of Phase-2
  correctness and timing.
- Use CPU's session compatibility projection as the current transaction's target evidence. Do
  not claim a stable persistent CPU target fingerprint.
- Keep target/policy/producer/codec identity derivation package-private. Public evidence reports
  the authenticated compatibility, caller budget, correctness actions, timing, and winner rather
  than exporting cache-key plumbing.
- Missing complete alternatives is tuning unavailability, not successful Phase-1-only tuning.
  The existing public fallback policy governs it.
- Ordinary fallback remains the single safe cross-phase recovery recipe. A partially completed
  tuning phase never becomes production state.

## Known limitations

- CPU-only, exactly one non-empty maximal CPU partition, one representative input set, and at most
  one eligible Phase-1 local workload.
- Phase-2 candidates vary only CPU 0010J's retained fusion/split and materialization alternatives
  around the fixed compile graph, owner, partition, publications, and Phase-1 route decision.
- CPU compatibility is session-scoped, so every public call measures Phase 2 and performs no
  model-plan-cache I/O. The explicit path is retained for the generic future persistent case.
- Exact canonical represented-byte correctness and minimum median elapsed nanoseconds are the only
  supported Phase-2 policies.
- No prepared executable, Runtime state, representative bytes, or rich evidence is persisted.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
