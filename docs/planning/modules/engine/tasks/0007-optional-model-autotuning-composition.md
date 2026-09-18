# Task 0007: Optional Model-Autotuning Composition

## Status

Ready

## Goal

Compose completed Config 0006A, tools/tuning 0001, CPU 0010I, and Engine 0006A into one optional
ordinary Engine preparation operation. A caller supplies an opaque model identity and one
representative Tensor input set. Engine maps the sole current CPU handoff into tuning, executes
requested trials through the representative session, and returns a freshly prepared selected
recipe with Engine-owned evidence, or an explicitly identified safe-heuristic fallback.

This is CPU-only local-workload tuning, not tools/tuning 0002 graph/plan tuning.

## Scope

- Add public final `ModelAutotuningRequest` with a retained non-null `ModelAutotuningConfig`, a
  nested immutable caller-defined `ModelIdentity(int schemaVersion, byte[] bytes)`, and an
  immutable list snapshot of representative `Tensor` references.
- The request owns only container structure and identity bytes. Tensor/storage remains
  caller-owned, is snapshotted only when the synchronous operation is admitted, and must stay
  live, accessible, and unmodified until return or failure.
- Add `Engine.prepareTuned(CompiledGraph, ModelAutotuningRequest)` returning public final
  `ModelAutotuningPreparation`: an owner-bound `PreparedExecution`, outcome `TUNED` or
  `SAFE_HEURISTIC_FALLBACK`, and optional immutable Engine-owned evidence.
- `Engine.prepareTuned` forwards raw `this`, `compiledGraph`, and `request` references directly to
  one new package-private `AdvancedEngine.prepareTunedOrdinary(...)` entry. The facade performs no
  null check, accessor call, list copy, owner check, or other inspection.
- That AdvancedEngine entry performs the sole `beginOperation()` first, validates non-null
  compiled graph and exact ordinary owner, then validates/reads the request, and directly creates
  `RepresentativeExecutionSession` under the already-held admission. It must not call the existing
  self-admitting `openRepresentativeExecutionSession` after `beginOperation()`.
- Reuse `RepresentativeExecutionSession` for binding, borrowing, execution, cleanup, failure
  identity, final preparation, suppression rules, and the close race. Add only the bounded
  already-admitted construction/final-publication observations below; never admit twice.
- Obtain `CpuLocalWorkloadTuning` only from `CpuEngineBackendComposition`; do not widen the shared
  composition seam with CPU candidate types.
- Invoke `candidateHandoff` once. A present handoff is the sole occurrence: occurrence index 0,
  partition index 0, weight 1. Absence is tuning unavailability, not zero-work success.
- Derive the bounded versioned occurrence-context fingerprint solely from those fixed indexes:
  schema version 1 and exactly eight big-endian bytes containing partition index 0 followed by
  occurrence index 0 as signed Java `int` values. The model fingerprint remains the separate
  request/evidence field. Together they identify this model occurrence for evidence, but neither
  value is a workload-cache key or selection authority.
- Translate Config exhaustively: objective by switch, four budget fields one-for-one, profile
  identity by defensive copy, exact cache `Path`, and fallback policy in Engine. Translate the
  caller model identity to the tool model fingerprint.
- Implement a private typed `BackendWorkloadTuning` adapter delegating its six operations to CPU
  and translating compatibility, candidate identity, and reuse scope without inspecting CPU
  fields. The measurement action freshly prepares one trial and executes it once through the
  representative session. Tuning alone owns counts, clock, ranking, cache, and evidence.
- Require exactly one selected handoff with the original partition/batch and a decision. Validate
  model/profile/policy and occurrence context/weight before fresh selected preparation.
- Translate successful rich evidence into nested Engine-owned values, preserving cache-hit versus
  measured source, occurrence index/weight, opaque compatibility/candidate identities, raw
  nanosecond samples, summaries, and winner. No CPU, Prepare, or tools/tuning type enters the
  ordinary public API. Evidence is present exactly for `TUNED`.
- Map absent handoff to one recoverable `IllegalStateException`. Normalize cache `IOException` to
  one exact `UncheckedIOException`, and any impossible other checked exception from the
  Engine-owned measurement collaboration to one exact `IllegalStateException` with cause. Feed
  that exact runtime failure into 0006A's existing strict/fallback state machine.
- Required tuning propagates the exact normalized failure after cleanup and performs no ordinary
  preparation. Allowed fallback returns a freshly ordinary-prepared recipe, explicit fallback
  outcome, and empty evidence only after cleanup succeeds.
- Add focused lifecycle/mapping/evidence/cache tests, update Engine's direct dependency inventory
  for `tools/tuning`, and complete the mandatory clean documentation pass.

## Exact public API

The implementation must declare exactly this new public surface; ordinary inherited `Object`
methods are omitted except where value semantics are explicitly required:

```java
public final class ModelAutotuningRequest {
    public ModelAutotuningRequest(
            ModelAutotuningConfig config,
            ModelIdentity modelIdentity,
            List<Tensor> representativeInputs);
    public ModelAutotuningConfig config();
    public ModelIdentity modelIdentity();
    public List<Tensor> representativeInputs();

    public static final class ModelIdentity {
        public ModelIdentity(int schemaVersion, byte[] bytes);
        public int schemaVersion();
        public byte[] bytes();
        public boolean equals(Object other);
        public int hashCode();
        public String toString();
    }
}

public final class ModelAutotuningPreparation {
    public PreparedExecution preparedExecution();
    public Outcome outcome();
    public Optional<Evidence> evidence();

    public enum Outcome { TUNED, SAFE_HEURISTIC_FALLBACK }
    public enum Source { CACHE_HIT, MEASURED }
    public enum ReuseScope { SESSION, PERSISTENT }

    public record Evidence(
            ModelAutotuningRequest.ModelIdentity modelIdentity,
            ModelAutotuningConfig.RepresentativeProfileIdentity representativeProfile,
            ModelAutotuningConfig.Objective objective,
            ModelAutotuningConfig.Budget budget,
            List<WorkloadEvidence> workloads) { }

    public record WorkloadEvidence(
            CompatibilityIdentity compatibility,
            long totalWeight,
            List<OccurrenceEvidence> occurrences,
            Source source,
            List<CandidateEvidence> candidates,
            CandidateIdentity winnerIdentity,
            SampleSummary winnerSummary) { }

    public record OccurrenceEvidence(
            int occurrenceIndex,
            int partitionIndex,
            long weight,
            ContextIdentity contextIdentity) { }

    public record CandidateEvidence(
            CandidateIdentity identity,
            List<Long> elapsedSamplesNanos,
            SampleSummary summary) { }

    public record SampleSummary(
            long minimumNanos,
            long medianNanos,
            long maximumNanos,
            int sampleCount) { }

    public static final class CompatibilityIdentity {
        public CompatibilityIdentity(int schemaVersion, byte[] bytes, ReuseScope reuseScope);
        public int schemaVersion();
        public byte[] bytes();
        public ReuseScope reuseScope();
        public boolean equals(Object other);
        public int hashCode();
        public String toString();
    }

    public static final class ContextIdentity {
        public ContextIdentity(int schemaVersion, byte[] bytes);
        public int schemaVersion();
        public byte[] bytes();
        public boolean equals(Object other);
        public int hashCode();
        public String toString();
    }

    public static final class CandidateIdentity {
        public CandidateIdentity(byte[] bytes);
        public byte[] bytes();
        public boolean equals(Object other);
        public int hashCode();
        public String toString();
    }
}

public final class Engine implements AutoCloseable {
    public ModelAutotuningPreparation prepareTuned(
            CompiledGraph compiledGraph,
            ModelAutotuningRequest request);
}
```

`ModelAutotuningPreparation` has exactly one package-private constructor
`ModelAutotuningPreparation(PreparedExecution preparedExecution, Outcome outcome,
Optional<Evidence> evidence)` in that order; it has no public/protected constructor and no
equality override.
`ModelAutotuningRequest` also retains object identity rather than overriding equality because its
representative Tensor references are live caller resources, not a durable value. Its constructor
checks arguments in component order, copies the input list with `List.copyOf`, rejects null
elements, retains the exact Config, identity, and Tensor references, and returns the same immutable
list snapshot on every accessor call.

The three identity classes snapshot constructor bytes and return a fresh byte array from every
`bytes()` call. They reject non-positive schema versions where present and null/empty bytes;
equality/hash use schema plus byte content (and reuse scope for compatibility), while diagnostic
text exposes only schema, byte length, and reuse scope. `CandidateIdentity` uses byte content only.

Every public record has the listed component order, public canonical constructor, component
accessors, and ordinary record value `equals`, `hashCode`, and `toString`. Constructors validate
the same invariants as the corresponding tuning evidence and use `List.copyOf`; candidate sample
elements are non-null/non-negative and summaries must match exact count/minimum/integer-middle
median/maximum. `Evidence.workloads()` and `WorkloadEvidence.occurrences()` are non-empty;
`candidates()` is non-empty exactly for `MEASURED` and empty for `CACHE_HIT`. Enum APIs consist of
the listed constants plus Java's generated `values()` and `valueOf(String)`.

The preparation always retains the exact `PreparedExecution` created for the exact input
`CompiledGraph` and therefore the exact owning `Engine`. `TUNED` requires present evidence;
`SAFE_HEURISTIC_FALLBACK` requires empty evidence. The preparation/result and all metadata have no
close operation and remain readable after Engine close. The contained prepared handle's
`compiledGraph()` metadata also remains readable, but the closed owner rejects every later run.

## Exact evidence mapping

- `Evidence` retains the exact request `ModelIdentity` and exact Config profile/objective/budget
  objects after validating equality with the tool result. Its ordered workload list is a fresh
  immutable translation and contains exactly one element in this task.
- `CompatibilityIdentity` preserves the tool compatibility schema version, bytes, and maps
  `SESSION`/`PERSISTENT` exhaustively by switch. It grants no compatibility authority itself.
- `OccurrenceEvidence` maps the sole authenticated tool occurrence to indexes `0`/`0`, weight `1`,
  and `ContextIdentity` schema version `1` plus exactly eight big-endian bytes containing partition
  index then occurrence index. Authentication requires exact byte equality with that request value.
- `Source` exhaustively maps `CACHE_HIT` or `MEASURED`. Cache hits have no candidate evidence but
  retain winner identity and compact summary. Measured workloads retain candidate encounter order,
  each candidate's identity, raw timed samples in encounter order, exact summary, and winner.
- Candidate identities preserve their exact opaque bytes. Summaries preserve minimum, integer-
  middle median, maximum, and sample count in nanoseconds. No backend fields are decoded.
- Evidence and public wrapper construction occurs only after tool-result authentication,
  representative cleanup, and fresh selected/fallback preparation succeed. Those complete objects
  remain local until final Engine publication succeeds; no object is returned if that check fails.
- Cache ordering remains exactly tuning 0001's: load/validate precedes enumeration or execution;
  a compatible hit performs no trial; a miss is atomically published only after complete
  measurement and decision codec round-trip. Engine never reads, repairs, deletes, or rolls back
  the cache. If tuning returns successfully and a later Engine authentication, selected-
  preparation, or final-publication step fails, an already published compatible workload entry
  remains valid independently even though no Engine result is returned. A tuning failure before
  publication preserves the prior cache as guaranteed by tuning 0001.

## Validation and lifecycle order

1. The facade forwards raw references. `AdvancedEngine.prepareTunedOrdinary` calls the sole
   `beginOperation()` before inspecting them, so closure wins over all argument validation. If
   `beginOperation()` itself rejects closure, no admission was acquired and no finish call occurs.
2. After admission, validate non-null `compiledGraph`, then exact ordinary Engine owner identity.
   A foreign handle fails before `request` inspection.
3. Validate non-null request reference. Request construction has already validated Config,
   identity, list structure, and elements. Only now read its three accessors.
4. Directly construct `RepresentativeExecutionSession`, or use a clearly named
   `createRepresentativeExecutionSessionAlreadyAdmitted`, with the immutable input snapshot. This
   path performs no admission. The session validates complete logical identity/descriptors,
   snapshots all current storage associations, validates all storage, then borrows wrappers. Any
   failure here is an input/lifecycle failure, not fallback-eligible.
5. Until session construction succeeds, AdvancedEngine retains admission ownership. Any
   null/owner/request/accessor/construction/binding/borrow failure calls `finishFailure()` exactly
   once. Successful construction transfers that one admission to the session; no unconditional
   outer `finally` may finish it again.
6. Only after the session opens, acquire the CPU tuning collaboration, obtain/map the handoff,
   translate Config/identity, and invoke tuning. One Engine admission remains held throughout.
7. Each measurement freshly prepares a trial and then calls `session.execute`. Fresh trial
   preparation failure is a recoverable tuning-transaction failure because no Runtime trial began;
   execution failure is marked by exact throwable identity, poisons/releases the session, and is
   never fallback-eligible.
8. Authenticate the result, then use a bounded overload/refactoring of the 0006A selected helper
   to clean wrappers and freshly prepare the inward selected recipe while retaining admission.
9. Still admitted, construct the ordinary `PreparedExecution` wrapper with exact ordinary owner,
   compiled handle, and inward recipe; then construct the complete `ModelAutotuningPreparation`
   and evidence (or fallback outcome/empty evidence). Pass that complete public result to a
   specifically typed session final-publication method backed by `finishHandle`.
10. Only successful `finishHandle` return publishes the preconstructed result. If close wins that
    synchronized check, the local objects do not escape. On success every returned metadata field
    is already complete; later close leaves metadata readable but prevents runs.

## Out of scope

- multiple partitions, occurrences, representative sets, weights, shape-only profiles, or corpus
  tuning; tools/tuning 0002; model-plan persistence; prepared-executable serialization
- deriving canonical model identity from Tensor, Compiler graph, or graph-local IDs
- numerical comparison/materialization, retry, timeout, cancellation, asynchronous tuning, or
  relaxed-numerics permission
- public CPU/tool/inward SPI types, a generic backend tuning facade, Metal/CUDA tuning, Runtime
  changes, hidden defaults/global state, discovery, service location, or reflection
- production Java changes outside Engine; test changes are limited to Engine, architecture, and
  the one public integration fixture named below

## Failure taxonomy

- Admission/owner/request/representative-binding failures propagate unchanged and release or roll
  back exactly as the existing Engine/0006A lifecycle requires. They are caller/lifecycle failures,
  never inputs to Config fallback, and return no result or evidence.
- Fallback-eligible preselection failures are: absent eligible handoff; Config-to-tool translation
  or tuning-request rejection; cache load/validation/publication failure; candidate/compatibility/
  identity/codec/budget/selection/result-authentication failure; and fresh `prepareTrial` failure
  before `session.execute` begins. They may enter
  `prepareAfterRecoverableTuningFailure` only while the session remains unpoisoned.
- Add a package-private synchronous identity observation to `RepresentativeExecutionSession`.
  `execute` records the exact caught `RuntimeException` or `Error` reference before poisoning and
  rethrowing it. A synchronized `isRepresentativeExecutionFailure(Throwable failure)` compares by
  `==`; it never wraps, replaces, or clears the reference. A second synchronized
  `canResolveRecoverableTuningFailure()` is true only while admission is retained, the session is
  open, no execution failure was recorded, and it is unpoisoned.
- The measurement callback calls `prepareTrial` before entering `session.execute`. A failure from
  `prepareTrial` therefore has no execution marker. Around `session.execute`, it catches
  `RuntimeException | Error` only to ensure the exact reference is recorded by the session and
  immediately rethrows that same object. The outer `WorkloadTuning.tune` catch first tests
  `isRepresentativeExecutionFailure(failure)` and rethrows the same object unchanged. Every other
  tuning `RuntimeException` is passed to the recoverable helper only when
  `canResolveRecoverableTuningFailure()` is true; otherwise it is rethrown unchanged.
- `IOException` from `WorkloadTuning.tune` becomes one new exact `UncheckedIOException` whose
  cause is the exact `IOException`. Any other checked `Exception` is impossible from the
  Engine-owned callback contract and becomes one new exact `IllegalStateException` whose cause is
  that exact checked failure. Strict mode rethrows that exact wrapper; allowed fallback uses that
  same wrapper for cleanup/suppression. Existing runtime failures are not rewrapped.
- A failure thrown by `session.execute` is a representative trial-execution failure. The session
  keeps it primary, attempts result/input cleanup, suppresses distinct cleanup failures, poisons
  and releases itself, and the composition rethrows that exact failure without invoking fallback.
- Any cleanup failure while resolving an otherwise recoverable tuning failure forbids fallback:
  the exact tuning failure remains primary and receives the cleanup failure as a distinct
  suppression, exactly as `prepareAfterRecoverableTuningFailure` specifies.
- `Error` is never normalized or fallback-eligible. It propagates unchanged after applicable
  owned cleanup.
- After a valid selected result exists, failure from representative cleanup, fresh
  `prepareSelected`, wrapper/result construction, or final Engine publication is not fallback-
  eligible. The bounded selected-helper overload/factoring propagates the exact failure, releases
  admission once, and never tries ordinary preparation.
- If allowed ordinary fallback preparation, wrapper/result construction, or final publication
  fails, that exact later failure is primary and the earlier recoverable tuning failure is
  suppressed when distinct. If fallback succeeds, the recoverable failure is intentionally not
  exposed; the returned outcome and empty evidence record only that safe heuristic preparation
  was used.
- No path returns partial evidence, a partial selection, or a result after any failure.

The 0006A helper adjustment is bounded and package-private. Add selected and recoverable-fallback
overloads, or factor their existing bodies into narrow private methods, so they can: retain the
single session admission through cleanup and fresh preparation; construct the ordinary wrapper and
complete public tuning result before final publication; then publish that specifically typed
result through `finishHandle`. Do not add a generic callback/factory framework. Preserve the old
helper entry points used by 0006A tests or update those tests to the factored equivalent without
weakening their exact primary/suppression assertions.

## Model identity reassessment

Caller-supplied opaque identity is the minimal truthful current choice. `TensorId`, `NodeId`, and
`ValueId` are explicitly local identity domains; object identity is not stable across model
reconstruction; and neither Model nor Compiler currently owns a canonical cross-compilation graph
serialization or fingerprint. Engine therefore must not hash an ad hoc subset of graph state and
claim canonical model identity.

The identity labels returned evidence only. Tuning 0001 deliberately excludes model/profile
fingerprints from workload-cache matching, so a collision or dishonest reuse cannot authorize a
candidate, make an incompatible cache entry a hit, or alter CPU compatibility validation. It can
mislabel or aggregate evidence under the caller's own identity domain. The caller consequently
owns schema meaning, collision resistance, uniqueness, and revision discipline. Engine owns
defensive copying and faithful propagation only. A registry, automatic graph fingerprint, or
identity verification service has no current owner or consumer and remains out of scope.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Planning guide](../../../planning-guide.md), [roadmap](../../../roadmap.md), and
  [Engine master plan](../master-plan.md)
- Engine [0006A](0006a-representative-tuning-execution-and-safe-fallback.md),
  [Config 0006A](../../config/tasks/0006a-model-autotuning-request-configuration.md),
  [tuning 0001](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md),
  [CPU 0010I](../../../backends/cpu/tasks/0010i-supported-cpu-local-workload-tuning-composition-adapter.md),
  and [Prepare 0004](../../prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)

## Required reading

Read `AGENTS.md`, `ARCHITECTURE.md`, the current architecture plan, planning guide, roadmap,
Engine master plan, Engine 0002/0003/0005A/0006/0006A, Config master/task 0006A, tuning
master/task 0001, CPU master/task 0010I, Prepare master/task 0004, and the focused performance,
lifecycle, module-boundary, dependency, public-API, Runtime/Prepare/backend, glossary, and
documentation-profile pages named by the implementation prompt. Inspect the actual current Engine,
Config, tuning, CPU, Compiler, Prepare, Runtime, and Model APIs/tests before editing.

## Architecture constraints

- Engine owns public composition, lifecycle, occurrence mapping, fallback, and evidence
  translation. Tuning owns measurement/cache/ranking/evidence; CPU owns candidates, compatibility,
  decisions, and preparation; Runtime executes prepared work only.
- Representative storage stays caller-owned. Engine owns wrappers and trial run/result cleanup;
  neither trial recipes nor mutable run state become production state.
- Caller model/profile identities are evidence assertions and never grant cache compatibility.
- No CPU-private or inward/tool handoff type, generic registry, broad facade, or parameter map may
  enter the public API.
- `modules/engine -> tools/tuning` is a direct outer-composition dependency; update the existing
  inventory test and preserve the absence of `tools/tuning -> modules/engine`.

`ARCHITECTURE.md` already makes Engine the composition root, assigns the explicit autotuning
workflow/cache/evidence to `tools/tuning`, and requires requested tuning to finish before Runtime.
It does not enumerate an exhaustive Engine dependency list or forbid Engine from composing a tool.
The new edge is therefore a concrete dependency realization of existing ownership, not a new
architecture decision. Implementation must update `docs/architecture/dependency-rules.md` and
`EngineCompositionContractTest` to explain/lock the direct edge and reverse-edge prohibition. No
root architecture or ADR edit is required unless implementation discovers an authoritative rule
that contradicts this assessment; in that case stop and replan with the coordinated architecture
updates rather than silently widening this task.

## Package impact

Use `io.github.pho001.synaptik.engine`. Add only `ModelAutotuningRequest` and
`ModelAutotuningPreparation` as public top-level types; nested request identity and evidence DTOs
remain cohesive parts of those types. Package-private adapters stay in the same package.

## Affected files

Expected implementation paths:

- `modules/engine/build.gradle.kts`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/CpuEngineBackendComposition.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSession.java`
- add `modules/engine/src/main/java/io/github/pho001/synaptik/engine/ModelAutotuningRequest.java`
- add `modules/engine/src/main/java/io/github/pho001/synaptik/engine/ModelAutotuningPreparation.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/package-info.java`
- add `modules/engine/src/test/java/io/github/pho001/synaptik/engine/ModelAutotuningCompositionTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSessionTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/EngineTypedPublicShapeTest.java`
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/EngineCompositionContractTest.java`
- add `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineModelAutotuningIntegrationTest.java`

Expected documentation/planning paths: `docs/api/public-api.md`, the performance/tuning,
runtime-boundary, and dependency-rule architecture explanations, benchmarking guide,
`docs/glossary.md`, this task, Engine master plan, and roadmap.

## Maximum scope

At most 22 paths: 8 Engine production/build, 5 Engine/architecture/integration test, 6 focused
explanatory documentation, and 3 planning paths. The cohesive ceiling is justified by the public
lifecycle, new direct dependency, and required documentation/architecture-test synchronization.
Stop for another module's production Java change, an architecture
contract change, more public top-level types, or a generic backend abstraction.

## Acceptance criteria

1. The exact request/result/method shape is public without inward, CPU, or tool types.
2. Identity/list immutability and caller Tensor/storage ownership are tested and documented.
3. Model identity remains caller-supplied; occurrence 0/partition 0/weight 1 is deterministic.
4. Config translation, typed CPU delegation, and tuning invocation are exact and field-free.
5. Cache hits execute no trials; misses execute exact requested counts in fresh Runtime state; all
   successful paths freshly prepare production state.
6. Evidence is authenticated, immutable, and completely mapped; fallback is explicit and has no
   evidence.
7. Required mode never falls back; allowed mode falls back once only after proven cleanup.
8. Trial/cleanup/`Error`/lifecycle/ownership/selected-preparation/close-race failures preserve
   0006A's primary and suppression semantics.
9. Focused tests prove facade non-inspection, one and only one admission, validation order,
   constructor/binding rollback exactly once, admission transfer, exact execution-failure identity,
   prepare-before-execute classification, unpoisoned recoverable gating, and public-result
   construction before the final close-race check.
10. Runtime and inward modules are unchanged; dependency tests lock the new one-way edge.
11. A native-free public integration test exercises Config/request construction through standard
    CPU composition, deterministic absent-handoff allowed fallback, the returned prepared handle,
    and a subsequent run. Focused Engine tests use a package-private typed synthetic orchestration
    seam to prove cache miss/measurement, selected preparation, compatible cache hit, evidence,
    injected failure, and concurrency without making OpenBLAS availability a test prerequisite.
12. A clean documentation context finalizes Javadocs, API/example prose, current/planned wording,
    dependency explanation, glossary impact, and planning evidence.
13. All validation below passes.

## Tests / validation

```bash
./gradlew :modules:engine:test \
  :testing:architecture-tests:test \
  :testing:integration-tests:test
```

After the clean documentation pass:

```bash
./gradlew :modules:engine:javadoc
./gradlew test
git diff --check
git status --short -uall
```

The focused Engine suite proves selected-tuning and state/failure details through a narrow
package-private synthetic seam; the architecture suite proves dependency direction; and the
integration suite proves the deterministic public standard-composition fallback path. Together
with the root checkpoint this is sufficient because no inward behavior or backend contract
changes. No real OpenBLAS, benchmark, or backend-conformance-specific run is required. Validate Markdown
relative links/anchors, unique headings, balanced fences, final newlines, trailing whitespace,
statuses, terminology, and exact scope.

## Dependencies

Engine 0001–0006A, Config 0006A, tools/tuning 0001, Prepare 0004, and CPU 0010I are Complete.
Tools/tuning 0002, CPU 0016, Metal, CUDA, and provider 0004 are not dependencies.

## Follow-up tasks

Engine 0008 remains the next Draft checkpoint. Multiple profiles/occurrences, corpus weighting,
tools/tuning 0002 graph/plan search, and CPU 0016 peer routes remain later work.

## Documentation impact

Implementation changes the ordinary Engine API, lifecycle workflow, direct dependency inventory,
and current tuning status. The documentation pass must update Engine package Javadoc, the public
API with one complete CPU-only example, both focused architecture explanations, dependency rules,
benchmarking/tuning guidance, and glossary entries for the public request/preparation outcome and
caller-defined model identity. It must keep tools/tuning 0002, multi-occurrence tuning, and
model-plan persistence explicitly planned. No documentation change outside the listed paths is
expected unless the final API reveals a concrete contradiction.

## Architecture impact

Expected architecture decision impact: None. Dependency impact: one new direct
`modules/engine -> tools/tuning` realization of the existing Engine-composition and tuning-
ownership contract, requiring the focused explanatory dependency update and architecture-test
inventory change named above. Stop if implementation needs a reverse edge, Runtime tuning state,
public CPU/tool handoff, derived canonical graph identity, or any authoritative contract change.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik Engine task 0007. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, the planning guide, Engine master plan, task 0007, and its directly
referenced completed contracts in full. Implement exactly the Ready task. Stop for architecture,
another-module Java, public inward/private type, Runtime tuning, identity, or scope conflict.

After executable validation, hand the exact diff/evidence to a distinct clean documentation
context. It must follow documentation-rules.md, finalize affected Javadocs/docs/glossary/planning,
and not repeat successful Java tests absent executable change or recorded stale-evidence risk.
```

## Local decisions

- Caller supplies model identity because no current Model/Compiler contract owns a stable
  cross-compilation fingerprint. It identifies returned evidence only and never participates in
  workload-cache matching, candidate eligibility, decision compatibility, or ownership checks.
  The caller owns its schema, uniqueness domain, collision resistance, and truthful reuse across
  model revisions; Engine validates shape/immutability but cannot detect a dishonest or colliding
  identity. Adding a registry, graph hashing, or verification service would be speculative.
- The sole CPU handoff is occurrence 0 with weight 1; no frequency is inferred.
- Engine translates evidence once so ordinary API leaks neither CPU nor generic tool handoffs.
- Checked cache failure is normalized before entering 0006A's runtime-only fallback boundary.
- Absent handoff is unavailability governed by Config fallback policy.

## Known limitations

CPU-only; one representative set; at most one local workload; caller-asserted identity; lifecycle
completion rather than numerical equivalence; in-memory rich evidence; no graph/plan tuning.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
