# Task 0007: Optional [Model](../../../../glossary.md#term-model)-[Autotuning](../../../../glossary.md#autotuning--model-autotuning) Composition

> **Editorial historical reading key.**
>
> This completed task records the original CPU-only, single-local-workload `prepareTuned` composition.
> The caller provides Config, opaque model identity and live representative Tensors; Engine translates
> requests and evidence, coordinates one admission and cleanup, and publishes a fresh prepared owner.
> Tools/tuning owns measurements, ranking and cache transactions; CPU owns candidate compatibility,
> decisions and preparation; Runtime executes recipes without tuning policy. Model/profile labels do
> not authorize cache hits or selection. Request list and identity-byte snapshots do not copy
> representative payloads.
>
> Engine closes trial results and wrappers while caller storage remains usable; the metadata carrier
> is not closeable, but its contained prepared handle owns preparation. See the [current two-phase
> tuning
> lifecycle](../../../../reference/engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle)
> and [current CPU/Metal
> composition](../../../../reference/engine/composition.md#current-explicit-cpu-and-metal-composition).
> Later complete-plan evidence and Metal tuning must not be read back into this archive's exact API or
> exclusions.

## Status

Complete

## Goal

Compose completed Config 0006A, tools/tuning 0001, [CPU](../../../../glossary.md#term-cpu) 0010I, and [Engine](../../../../glossary.md#term-engine) 0006A into one optional
ordinary Engine preparation [operation](../../../../glossary.md#operation). A caller supplies an [opaque](../../../../glossary.md#term-opaque) [model](../../../../glossary.md#term-model) [identity](../../../../glossary.md#term-identity-meanings) and one
representative [Tensor](../../../../glossary.md#tensor) input set. Engine maps the sole current CPU [handoff](../../../../glossary.md#term-handoff) into tuning, executes
requested trials through the representative session, and returns a freshly prepared selected
recipe with Engine-owned evidence, or an explicitly identified safe-[heuristic](../../../../glossary.md#term-heuristic) fallback.

This is CPU-only local-workload tuning, not tools/tuning 0002 [graph](../../../../glossary.md#graph)/plan tuning.

## [Scope](../../../../glossary.md#term-scope)

- Add public final `ModelAutotuningRequest` with a retained non-null [`ModelAutotuningConfig`](../../../../glossary.md#model-autotuning-request-facade--modelautotuningconfig), a
  nested immutable caller-defined `ModelIdentity(int schemaVersion, byte[] bytes)`, and an
  immutable list snapshot of representative <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>[`Tensor`](../../../../glossary.md#tensor) references.
- The request owns only container structure and identity bytes. Tensor/storage remains
  caller-owned, is snapshotted only when the synchronous operation is admitted, and must stay
  live, accessible, and unmodified until return or failure.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [synchronous glossary](../../../../glossary.md#term-synchronous); [operation glossary](../../../../glossary.md#operation))</span>
- Add `Engine.prepareTuned(CompiledGraph, ModelAutotuningRequest)` returning public final
  `ModelAutotuningPreparation`: an [owner](../../../../glossary.md#term-owner)-bound [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), outcome `TUNED` or
  `SAFE_HEURISTIC_FALLBACK`, and optional [immutable](../../../../glossary.md#term-immutable) [Engine](../../../../glossary.md#term-engine)-owned evidence.
- `Engine.prepareTuned` forwards raw `this`, `compiledGraph`, and `request` references directly to
  one new package-private `AdvancedEngine.prepareTunedOrdinary(...)` entry. The facade performs no
  null check, accessor call, list copy, owner check, or other inspection.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- That AdvancedEngine entry performs the sole `beginOperation()` first, validates non-null
  compiled graph and exact ordinary owner, then validates/reads the request, and directly creates<span data-glossary-navigation="true"> ([compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel))</span>
  `RepresentativeExecutionSession` under the already-held admission. It must not call the existing
  self-admitting <span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>`openRepresentativeExecutionSession` after `beginOperation()`.
- Reuse `RepresentativeExecutionSession` for binding, borrowing, execution, cleanup, failure
  identity, final preparation, suppression rules, and the close race. Add only the bounded
  already-admitted construction/final-publication observations below; never admit twice.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [borrowing glossary](../../../../glossary.md#term-borrowing); [race glossary](../../../../glossary.md#term-race); [publication glossary](../../../../glossary.md#term-publication))</span>
- Obtain `CpuLocalWorkloadTuning` only from `CpuEngineBackendComposition`; do not widen the shared
  composition seam with CPU candidate types.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Invoke `candidateHandoff` once. A present handoff is the sole occurrence: occurrence index 0,
  partition index 0, weight 1. Absence is tuning unavailability, not zero-work success.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [partition glossary](../../../../glossary.md#partition); [weight glossary](../../../../glossary.md#term-weight))</span>
- Derive the bounded versioned occurrence-context fingerprint solely from those fixed indexes:
  schema version 1 and exactly eight big-endian bytes containing partition index 0 followed by
  occurrence index 0 as signed Java <span data-glossary-navigation="true"> ([fingerprint glossary](../../../../glossary.md#term-fingerprint); [schema glossary](../../../../glossary.md#term-schema); [big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries))</span>`int` values. The model fingerprint remains the separate
  request/evidence field. Together they identify this model occurrence for evidence, but neither
  value is a workload-cache key or selection authority.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [cache key glossary](../../../../glossary.md#term-cache-key))</span>
- Translate Config exhaustively: objective by switch, four budget fields one-for-one, profile
  identity by defensive copy, exact cache <span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [defensive copy glossary](../../../../glossary.md#term-defensive-copy))</span>`Path`, and fallback policy in Engine. Translate the
  caller model identity to the tool model fingerprint.
- Implement a private typed `BackendWorkloadTuning` adapter delegating its six operations to CPU
  and translating compatibility, candidate identity, and reuse scope without inspecting CPU
  fields. The measurement action freshly prepares one trial and executes it once through the
  representative session. Tuning alone owns counts, clock, ranking, cache, and evidence.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span>
- Require exactly one selected handoff with the original partition/batch and a decision. Validate
  model/profile/policy and occurrence context/weight before fresh selected preparation.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Translate successful rich evidence into nested Engine-owned values, preserving cache-hit versus
  measured source, occurrence index/weight, opaque compatibility/candidate identities, raw
  nanosecond samples, summaries, and winner. No CPU, Prepare, or tools/tuning type enters the
  ordinary public API. Evidence is present exactly for <span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [Prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`TUNED`.
- Map absent [handoff](../../../../glossary.md#term-handoff) to one recoverable `IllegalStateException`. Normalize [cache](../../../../glossary.md#term-cache) `IOException` to
  one exact `UncheckedIOException`, and any impossible other checked exception from the
  Engine-owned measurement collaboration to one exact `IllegalStateException` with cause. Feed
  that exact runtime failure into 0006A's existing strict/fallback state machine.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- Required tuning propagates the exact normalized failure after cleanup and performs no ordinary
  preparation. Allowed fallback returns a freshly ordinary-prepared recipe, explicit fallback
  outcome, and empty evidence only after cleanup succeeds.
- Add focused lifecycle/mapping/evidence/cache tests, update Engine's direct dependency inventory
  for <span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>`tools/tuning`, and complete the mandatory clean documentation pass.

## Exact public [API](../../../../glossary.md#term-application-programming-interface-api)

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
`ModelAutotuningRequest` also retains object [identity](../../../../glossary.md#term-reference-identity) rather than overriding equality because its
representative [Tensor](../../../../glossary.md#tensor) references are live caller resources, not a durable value. Its constructor
checks [arguments](../../../../glossary.md#term-argument) in component order, copies the input list with `List.copyOf`, rejects null
[elements](../../../../glossary.md#term-element), retains the exact Config, [identity](../../../../glossary.md#term-identity-meanings), and Tensor references, and returns the same [immutable](../../../../glossary.md#term-immutable)
list [snapshot](../../../../glossary.md#term-snapshot) on every accessor call.

The three identity classes snapshot constructor bytes and return a fresh byte array from every
`bytes()` call. They reject non-positive [schema](../../../../glossary.md#term-schema) versions where present and null/empty bytes;
equality/[hash](../../../../glossary.md#term-hash) use schema plus byte content (and reuse [scope](../../../../glossary.md#term-scope) for compatibility), while diagnostic
text exposes only schema, byte length, and reuse scope. `CandidateIdentity` uses byte content only.

Every public record has the listed component order, public [canonical](../../../../glossary.md#term-canonical-form) constructor, component
accessors, and ordinary record value `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`. Constructors validate
the same invariants as the corresponding tuning evidence and use `List.copyOf`; candidate [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)
elements are non-null/non-negative and summaries must match exact count/minimum/integer-middle
[median](../../../../glossary.md#term-median)/maximum. `Evidence.workloads()` and `WorkloadEvidence.occurrences()` are non-empty;
`candidates()` is non-empty exactly for `MEASURED` and empty for `CACHE_HIT`. Enum APIs consist of
the listed [constants](../../../../glossary.md#term-constant) plus Java's generated `values()` and `valueOf(String)`.

The preparation always retains the exact [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) created for the exact input
`CompiledGraph` and therefore the exact owning [`Engine`](../../../../glossary.md#term-engine). `TUNED` requires present evidence;
`SAFE_HEURISTIC_FALLBACK` requires empty evidence. The preparation/result and all [metadata](../../../../glossary.md#term-metadata) have no
close [operation](../../../../glossary.md#operation) and remain readable after Engine close. The contained prepared handle's
`compiledGraph()` metadata also remains readable, but the closed [owner](../../../../glossary.md#term-owner) rejects every later [run](../../../../glossary.md#term-run).

## Exact evidence mapping

- `Evidence` retains the exact request `ModelIdentity` and exact Config profile/objective/budget
  objects after validating equality with the tool result. Its ordered workload list is a fresh
  immutable translation and contains exactly one element in this task.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [immutable glossary](../../../../glossary.md#term-immutable); [element glossary](../../../../glossary.md#term-element))</span>
- `CompatibilityIdentity` preserves the tool compatibility [schema](../../../../glossary.md#term-schema) version, bytes, and maps
  `SESSION`/`PERSISTENT` exhaustively by switch. It grants no compatibility authority itself.
- `OccurrenceEvidence` maps the sole authenticated tool [occurrence](../../../../glossary.md#term-occurrence) to indexes `0`/`0`, [weight](../../../../glossary.md#term-weight) `1`,
  and `ContextIdentity` schema version `1` plus exactly eight big-endian bytes containing partition
  index then occurrence index. Authentication requires exact byte equality with that request value.<span data-glossary-navigation="true"> ([big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [partition glossary](../../../../glossary.md#partition); [Authentication glossary](../../../../glossary.md#term-authentication))</span>
- `Source` exhaustively maps `CACHE_HIT` or `MEASURED`. Cache hits have no candidate evidence but
  retain winner identity and compact summary. Measured workloads retain candidate encounter order,
  each candidate's identity, raw timed samples in encounter order, exact summary, and winner.<span data-glossary-navigation="true"> ([Cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Candidate identities preserve their exact opaque bytes. Summaries preserve minimum, integer-
  middle median, maximum, and sample count in nanoseconds. No backend fields are decoded.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [median glossary](../../../../glossary.md#term-median); [backend glossary](../../../../glossary.md#backend))</span>
- Evidence and public wrapper construction occurs only after tool-result authentication,
  representative cleanup, and fresh selected/fallback preparation succeed. Those complete objects
  remain local until final Engine publication succeeds; no object is returned if that check fails.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [publication glossary](../../../../glossary.md#term-publication))</span>
- Cache ordering remains exactly tuning 0001's: load/validate precedes enumeration or execution;
  a compatible hit performs no trial; a miss is atomically published only after complete
  measurement and decision codec round-trip. Engine never reads, repairs, deletes, or rolls back
  the cache. If tuning returns successfully and a later Engine authentication, selected-
  preparation, or final-publication step fails, an already published compatible workload entry
  remains valid independently even though no Engine result is returned. A tuning failure before
  publication preserves the prior cache as guaranteed by tuning 0001.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [round-trip glossary](../../../../glossary.md#schema-serialization-and-checkpoint))</span>

## [Validation](../../../../glossary.md#term-validation) and [lifecycle](../../../../glossary.md#lifecycle) order

1. The [facade](../../../../glossary.md#term-facade) forwards raw references. `AdvancedEngine.prepareTunedOrdinary` calls the sole
   `beginOperation()` before inspecting them, so closure wins over all [argument](../../../../glossary.md#term-argument) validation. If
   `beginOperation()` itself rejects closure, no [admission](../../../../glossary.md#term-admission) was acquired and no finish call occurs.
2. After admission, validate non-null `compiledGraph`, then exact ordinary Engine owner identity.
   A foreign handle fails before <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [owner glossary](../../../../glossary.md#term-owner); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`request` inspection.
3. Validate non-null request reference. Request construction has already validated Config,
   identity, list structure, and elements. Only now read its three accessors.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>
4. Directly construct `RepresentativeExecutionSession`, or use a clearly named
   `createRepresentativeExecutionSessionAlreadyAdmitted`, with the immutable input snapshot. This
   path performs no admission. The session validates complete logical identity/descriptors,
   snapshots all current storage associations, validates all storage, then borrows wrappers. Any
   failure here is an input/lifecycle failure, not fallback-eligible.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
5. Until session construction succeeds, AdvancedEngine retains admission ownership. Any
   null/owner/request/accessor/construction/binding/borrow failure calls <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [binding glossary](../../../../glossary.md#term-binding))</span>`finishFailure()` exactly
   once. Successful construction transfers that one admission to the session; no unconditional
   outer `finally` may finish it again.
6. Only after the session opens, acquire the CPU tuning collaboration, obtain/map the handoff,
   translate Config/identity, and invoke tuning. One Engine admission remains held throughout.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [handoff glossary](../../../../glossary.md#term-handoff))</span>
7. Each measurement freshly prepares a trial and then calls `session.execute`. Fresh trial
   preparation failure is a recoverable tuning-transaction failure because no Runtime trial began;
   execution failure is marked by exact throwable identity, poisons/releases the session, and is
   never fallback-eligible.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction); [Runtime glossary](../../../../glossary.md#term-runtime); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
8. Authenticate the result, then use a bounded overload/refactoring of the 0006A selected helper
   to clean wrappers and freshly prepare the inward selected recipe while retaining admission.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>
9. Still admitted, construct the ordinary [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) wrapper with exact ordinary owner,
   compiled handle, and inward recipe; then construct the complete `ModelAutotuningPreparation`
   and evidence (or fallback outcome/empty evidence). Pass that complete public result to a
   specifically typed session final-publication method backed by <span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>`finishHandle`.
10. Only successful `finishHandle` return publishes the preconstructed result. If close wins that
    synchronized check, the local objects do not escape. On success every returned metadata field
    is already complete; later close leaves metadata readable but prevents runs.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- multiple partitions, occurrences, representative sets, weights, shape-only profiles, or corpus
  tuning; tools/tuning 0002; model-plan persistence; prepared-executable serialization<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight); [shape glossary](../../../../glossary.md#shape); [persistence glossary](../../../../glossary.md#term-persistence); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- deriving [canonical](../../../../glossary.md#term-canonical-form) [model](../../../../glossary.md#term-model) [identity](../../../../glossary.md#term-identity-meanings) from [Tensor](../../../../glossary.md#tensor), [Compiler](../../../../glossary.md#term-compiler) [graph](../../../../glossary.md#graph), or graph-local IDs
- numerical comparison/materialization, retry, timeout, cancellation, asynchronous tuning, or
  relaxed-numerics permission<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [cancellation glossary](../../../../glossary.md#term-cancellation); [asynchronous glossary](../../../../glossary.md#term-asynchronous))</span>
- public CPU/tool/inward SPI types, a generic backend tuning facade, Metal/CUDA tuning, Runtime
  changes, hidden defaults/global state, discovery, service location, or reflection<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [backend glossary](../../../../glossary.md#backend); [facade glossary](../../../../glossary.md#term-facade); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- production Java changes outside Engine; test changes are limited to Engine, architecture, and
  the one public integration fixture named below<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [fixture glossary](../../../../glossary.md#term-fixture))</span>

## Failure taxonomy

- Admission/owner/request/representative-binding failures propagate unchanged and release or roll
  back exactly as the existing Engine/0006A lifecycle requires. They are caller/lifecycle failures,
  never inputs to Config fallback, and return no result or evidence.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [binding glossary](../../../../glossary.md#term-binding); [release glossary](../../../../glossary.md#term-release); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Fallback-eligible preselection failures are: absent eligible handoff; Config-to-tool translation
  or tuning-request rejection; cache load/validation/publication failure; candidate/compatibility/
  identity/codec/budget/selection/result-authentication failure; and fresh <span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [cache glossary](../../../../glossary.md#term-cache); [validation glossary](../../../../glossary.md#term-validation); [codec glossary](../../../../glossary.md#term-codec); [authentication glossary](../../../../glossary.md#term-authentication))</span>`prepareTrial` failure
  before `session.execute` begins. They may enter
  `prepareAfterRecoverableTuningFailure` only while the session remains unpoisoned.
- Add a package-private [synchronous](../../../../glossary.md#term-synchronous) [identity](../../../../glossary.md#term-identity-meanings) observation to `RepresentativeExecutionSession`.
  `execute` records the exact caught `RuntimeException` or `Error` reference before poisoning and
  rethrowing it. A synchronized `isRepresentativeExecutionFailure(Throwable failure)` compares by
  `==`; it never wraps, replaces, or clears the reference. A second synchronized
  `canResolveRecoverableTuningFailure()` is true only while admission is retained, the session is
  open, no execution failure was recorded, and it is unpoisoned.
- The measurement [callback](../../../../glossary.md#term-callback) calls `prepareTrial` before entering `session.execute`. A failure from
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
  same wrapper for cleanup/suppression. Existing runtime failures are not rewrapped.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- A failure thrown by `session.execute` is a representative trial-execution failure. The session
  keeps it primary, attempts result/input cleanup, suppresses distinct cleanup failures, poisons
  and releases itself, and the composition rethrows that exact failure without invoking fallback.<span data-glossary-navigation="true"> ([releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
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
  was used.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- No path returns partial evidence, a partial selection, or a result after any failure.

The 0006A helper adjustment is bounded and package-private. Add selected and recoverable-fallback
overloads, or factor their existing bodies into narrow private methods, so they can: retain the
single session [admission](../../../../glossary.md#term-admission) through cleanup and fresh preparation; construct the ordinary wrapper and
complete public tuning result before final [publication](../../../../glossary.md#term-publication); then publish that specifically typed
result through `finishHandle`. Do not add a generic callback/factory framework. Preserve the old
helper entry points used by 0006A tests or update those tests to the factored equivalent without
weakening their exact primary/suppression assertions.

## [Model](../../../../glossary.md#term-model) [identity](../../../../glossary.md#term-identity-meanings) reassessment

Caller-supplied [opaque](../../../../glossary.md#term-opaque) identity is the minimal truthful current choice. [`TensorId`](../../../../glossary.md#tensorid), [`NodeId`](../../../../glossary.md#nodeid), and
[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) are explicitly local identity domains; object [identity](../../../../glossary.md#term-reference-identity) is not stable across model
reconstruction; and neither Model nor [Compiler](../../../../glossary.md#term-compiler) currently owns a [canonical](../../../../glossary.md#term-canonical-form) cross-[compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) [graph](../../../../glossary.md#graph)
[serialization](../../../../glossary.md#term-serialization) or [fingerprint](../../../../glossary.md#term-fingerprint). [Engine](../../../../glossary.md#term-engine) therefore must not [hash](../../../../glossary.md#term-hash) an ad hoc subset of graph state and
claim canonical model identity.

The identity labels returned evidence only. Tuning 0001 deliberately excludes model/profile
fingerprints from workload-[cache](../../../../glossary.md#term-cache) matching, so a collision or dishonest reuse cannot authorize a
candidate, make an incompatible cache entry a hit, or alter [CPU](../../../../glossary.md#term-cpu) compatibility [validation](../../../../glossary.md#term-validation). It can
mislabel or aggregate evidence under the caller's own [identity domain](../../../../glossary.md#term-identity-domain). The caller consequently
owns [schema](../../../../glossary.md#term-schema) meaning, collision resistance, uniqueness, and revision discipline. Engine owns
[defensive copying](../../../../glossary.md#memory-management-atomic-counters-and-complexity) and faithful propagation only. A [registry](../../../../glossary.md#term-registry), automatic graph fingerprint, or
identity verification service has no current [owner](../../../../glossary.md#term-owner) or consumer and remains out of [scope](../../../../glossary.md#term-scope).

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [roadmap](../../../roadmap.md), and
  [Engine master plan](../master-plan.md)
- [Engine](../../../../glossary.md#term-engine) [0006A](0006a-representative-tuning-execution-and-safe-fallback.md),
  [Config 0006A](../../config/tasks/0006a-model-autotuning-request-configuration.md),
  [tuning 0001](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md),
  [CPU 0010I](../../../backends/cpu/tasks/0010i-supported-cpu-local-workload-tuning-composition-adapter.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>,
  and [Prepare 0004](../../prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)

## Required reading

Read `AGENTS.md`, `ARCHITECTURE.md`, the current architecture plan, [planning](../../../../glossary.md#term-planning) guide, roadmap,
[Engine](../../../../glossary.md#term-engine) master plan, Engine 0002/0003/0005A/0006/0006A, Config master/task 0006A, tuning
master/task 0001, [CPU](../../../../glossary.md#term-cpu) master/task 0010I, [Prepare](../../../../glossary.md#prepare) master/task 0004, and the focused performance,
[lifecycle](../../../../glossary.md#lifecycle), [module](../../../../glossary.md#term-module)-boundary, [dependency](../../../../glossary.md#term-dependency), public-[API](../../../../glossary.md#term-application-programming-interface-api), [Runtime](../../../../glossary.md#term-runtime)/Prepare/[backend](../../../../glossary.md#backend), glossary, and
documentation-profile pages named by the implementation prompt. Inspect the actual current Engine,
Config, tuning, CPU, [Compiler](../../../../glossary.md#term-compiler), Prepare, Runtime, and [Model](../../../../glossary.md#term-model) APIs/tests before editing.

## Architecture constraints

- Engine owns public composition, lifecycle, occurrence mapping, fallback, and evidence
  translation. Tuning owns measurement/cache/ranking/evidence; CPU owns candidates, compatibility,
  decisions, and preparation; Runtime executes prepared work only.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [occurrence glossary](../../../../glossary.md#term-occurrence); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Representative storage stays caller-owned. Engine owns wrappers and trial run/result cleanup;
  neither trial recipes nor mutable run state become production state.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [mutable glossary](../../../../glossary.md#term-mutable); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- Caller [model](../../../../glossary.md#term-model)/profile [identities](../../../../glossary.md#term-identity-meanings) are evidence assertions and never grant [cache](../../../../glossary.md#term-cache) compatibility.
- No CPU-private or inward/tool handoff type, generic registry, broad facade, or parameter map may
  enter the public API.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade); [parameter glossary](../../../../glossary.md#term-parameter); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `modules/engine -> tools/tuning` is a direct outer-composition dependency; update the existing
  inventory test and preserve the absence of `tools/tuning -> modules/engine`.

`ARCHITECTURE.md` already makes [Engine](../../../../glossary.md#term-engine) the [composition root](../../../../glossary.md#term-composition-root), assigns the explicit [autotuning](../../../../glossary.md#autotuning--model-autotuning)
workflow/cache/evidence to `tools/tuning`, and requires requested tuning to finish before [Runtime](../../../../glossary.md#term-runtime).
It does not enumerate an exhaustive Engine [dependency](../../../../glossary.md#term-dependency) list or forbid Engine from composing a tool.
The new edge is therefore a concrete dependency realization of existing [ownership](../../../../glossary.md#term-ownership-meanings), not a new
architecture decision. Implementation must update `docs/architecture/dependency-rules.md` and
`EngineCompositionContractTest` to explain/lock the direct edge and reverse-edge prohibition. No
root architecture or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) edit is required unless implementation discovers an authoritative rule
that contradicts this assessment; in that case stop and replan with the coordinated architecture
updates rather than silently [widening](../../../../glossary.md#term-widening) this task.

## Package impact

Use `io.github.pho001.synaptik.engine`. Add only `ModelAutotuningRequest` and
`ModelAutotuningPreparation` as public top-level types; nested request [identity](../../../../glossary.md#term-identity-meanings) and evidence DTOs
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

Expected documentation/[planning](../../../../glossary.md#term-planning) paths: `docs/api/public-api.md`, the performance/tuning,
[runtime](../../../../glossary.md#term-runtime)-boundary, and [dependency](../../../../glossary.md#term-dependency)-rule architecture explanations, [benchmarking](../../../../glossary.md#term-benchmarking) guide,
`docs/glossary.md`, this task, [Engine](../../../../glossary.md#term-engine) master plan, and roadmap.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 23 paths: 8 [Engine](../../../../glossary.md#term-engine) production/build, 6 Engine/architecture/[integration test](../../../../glossary.md#term-integration-test), 6 focused
explanatory documentation, and 3 [planning](../../../../glossary.md#term-planning) paths. The implemented inventory is one path above the
planned 22 because the pre-existing `AdvancedEnginePublicShapeTest` also locks the ordinary
`Engine` method inventory and necessarily changed for `prepareTuned`. Retaining that coverage is
safer than dropping a required [API](../../../../glossary.md#term-application-programming-interface-api)-[shape](../../../../glossary.md#shape) assertion. The cohesive ceiling is justified by the public
[lifecycle](../../../../glossary.md#lifecycle), new direct [dependency](../../../../glossary.md#term-dependency), and required documentation/architecture-test [synchronization](../../../../glossary.md#term-synchronization).
Stop for another [module](../../../../glossary.md#term-module)'s production Java change, an [architecture
contract](../../../../glossary.md#architecture-contract) change, more public top-level types, or a generic [backend](../../../../glossary.md#backend) abstraction.

## Acceptance criteria

1. The exact request/result/method [shape](../../../../glossary.md#shape) is public without inward, [CPU](../../../../glossary.md#term-cpu), or tool types.
2. [Identity](../../../../glossary.md#term-identity-meanings)/list immutability and caller [Tensor](../../../../glossary.md#tensor)/storage [ownership](../../../../glossary.md#term-ownership-meanings) are tested and documented.
3. [Model](../../../../glossary.md#term-model) identity remains caller-supplied; [occurrence](../../../../glossary.md#term-occurrence) 0/[partition](../../../../glossary.md#partition) 0/[weight](../../../../glossary.md#term-weight) 1 is deterministic.
4. Config translation, typed CPU delegation, and tuning invocation are exact and field-free.
5. Cache hits execute no trials; misses execute exact requested counts in fresh Runtime state; all
   successful paths freshly prepare production state.<span data-glossary-navigation="true"> ([Cache glossary](../../../../glossary.md#term-cache); [prepare glossary](../../../../glossary.md#prepare))</span>
6. Evidence is authenticated, immutable, and completely mapped; fallback is explicit and has no
   evidence.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
7. Required mode never falls back; allowed mode falls back once only after proven cleanup.
8. Trial/cleanup/`Error`/lifecycle/ownership/selected-preparation/close-race failures preserve
   0006A's primary and suppression semantics.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [race glossary](../../../../glossary.md#term-race))</span>
9. Focused tests prove facade non-inspection, one and only one admission, validation order,
   constructor/binding rollback exactly once, admission transfer, exact execution-failure identity,
   prepare-before-execute classification, unpoisoned recoverable gating, and public-result
   construction before the final close-race check.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [admission glossary](../../../../glossary.md#term-admission); [binding glossary](../../../../glossary.md#term-binding); [rollback glossary](../../../../glossary.md#term-rollback))</span>
10. [Runtime](../../../../glossary.md#term-runtime) and inward [modules](../../../../glossary.md#term-module) are unchanged; [dependency](../../../../glossary.md#term-dependency) tests lock the new one-way edge.
11. A native-free public integration test exercises Config/request construction through standard
    CPU composition, deterministic absent-handoff allowed fallback, the returned prepared handle,
    and a subsequent run. Focused Engine tests use a package-private typed synthetic orchestration
    seam to prove cache miss/measurement, selected preparation, compatible cache hit, evidence,
    injected failure, and concurrency without making OpenBLAS availability a test prerequisite.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test); [handoff glossary](../../../../glossary.md#term-handoff); [run glossary](../../../../glossary.md#term-run); [Engine glossary](../../../../glossary.md#term-engine); [cache miss glossary](../../../../glossary.md#cache-and-memoization); [cache hit glossary](../../../../glossary.md#cache-and-memoization); [concurrency glossary](../../../../glossary.md#term-concurrency); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
12. A clean documentation context finalizes Javadocs, API/example prose, current/planned wording,
    dependency explanation, glossary impact, and planning evidence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
13. All [validation](../../../../glossary.md#term-validation) below passes.

## Tests / [validation](../../../../glossary.md#term-validation)

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

The focused [Engine](../../../../glossary.md#term-engine) suite proves selected-tuning and state/failure details through a narrow
package-private synthetic seam; the architecture suite proves [dependency direction](../../../../glossary.md#java-platform-and-build-tools); and the
integration suite proves the deterministic public standard-composition fallback path. Together
with the root [checkpoint](../../../../glossary.md#term-checkpoint) this is sufficient because no inward behavior or [backend](../../../../glossary.md#backend) contract
changes. No real [OpenBLAS](../../../../glossary.md#term-openblas), benchmark, or backend-[conformance](../../../../glossary.md#term-conformance)-specific [run](../../../../glossary.md#term-run) is required. Validate Markdown
relative links/anchors, unique headings, balanced fences, final newlines, trailing whitespace,
statuses, terminology, and exact [scope](../../../../glossary.md#term-scope).

## Dependencies

[Engine](../../../../glossary.md#term-engine) 0001–0006A, Config 0006A, tools/tuning 0001, [Prepare](../../../../glossary.md#prepare) 0004, and [CPU](../../../../glossary.md#term-cpu) 0010I are Complete.
Tools/tuning 0002, CPU 0016, [Metal](../../../../glossary.md#term-metal), [CUDA](../../../../glossary.md#term-cuda), and provider 0004 are not dependencies.

## Follow-up tasks

[Engine](../../../../glossary.md#term-engine) 0008 remains the next Draft [checkpoint](../../../../glossary.md#term-checkpoint). Multiple profiles/occurrences, corpus weighting,
tools/tuning 0002 [graph](../../../../glossary.md#graph)/plan search, and [CPU](../../../../glossary.md#term-cpu) 0016 peer routes remain later work.

## Documentation impact

Implementation changes the ordinary [Engine](../../../../glossary.md#term-engine) [API](../../../../glossary.md#term-application-programming-interface-api), [lifecycle](../../../../glossary.md#lifecycle) workflow, direct [dependency](../../../../glossary.md#term-dependency) inventory,
and current tuning status. The documentation pass must update Engine package [Javadoc](../../../../glossary.md#term-javadoc), the public
API with one complete [CPU](../../../../glossary.md#term-cpu)-only example, both focused architecture explanations, dependency rules,
[benchmarking](../../../../glossary.md#term-benchmarking)/tuning guidance, and glossary entries for the public request/preparation outcome and
caller-defined [model](../../../../glossary.md#term-model) [identity](../../../../glossary.md#term-identity-meanings). It must keep tools/tuning 0002, multi-[occurrence](../../../../glossary.md#term-occurrence) tuning, and
model-plan [persistence](../../../../glossary.md#term-persistence) explicitly planned. No documentation change outside the listed paths is
expected unless the final API reveals a concrete contradiction.

## Architecture impact

Expected architecture decision impact: None. [Dependency](../../../../glossary.md#term-dependency) impact: one new direct
`modules/engine -> tools/tuning` realization of the existing [Engine](../../../../glossary.md#term-engine)-composition and tuning-
[ownership](../../../../glossary.md#term-ownership-meanings) contract, requiring the focused explanatory dependency update and architecture-test
inventory change named above. Stop if implementation needs a reverse edge, [Runtime](../../../../glossary.md#term-runtime) tuning state,
public [CPU](../../../../glossary.md#term-cpu)/tool [handoff](../../../../glossary.md#term-handoff), derived [canonical](../../../../glossary.md#term-canonical-form) [graph](../../../../glossary.md#graph) [identity](../../../../glossary.md#term-identity-meanings), or any authoritative contract change.

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
  identity. Adding a registry, graph hashing, or verification service would be speculative.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings); [Compiler glossary](../../../../glossary.md#term-compiler); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [schema glossary](../../../../glossary.md#term-schema); [shape glossary](../../../../glossary.md#shape); [registry glossary](../../../../glossary.md#term-registry); [graph glossary](../../../../glossary.md#graph))</span>
- The sole [CPU](../../../../glossary.md#term-cpu) [handoff](../../../../glossary.md#term-handoff) is [occurrence](../../../../glossary.md#term-occurrence) 0 with [weight](../../../../glossary.md#term-weight) 1; no frequency is inferred.
- [Engine](../../../../glossary.md#term-engine) translates evidence once so ordinary [API](../../../../glossary.md#term-application-programming-interface-api) leaks neither CPU nor generic tool handoffs.
- Checked [cache](../../../../glossary.md#term-cache) failure is normalized before entering 0006A's [runtime](../../../../glossary.md#term-runtime)-only fallback boundary.
- Absent handoff is unavailability governed by Config fallback policy.

## Known limitations

[CPU](../../../../glossary.md#term-cpu)-only; one representative set; at most one local workload; caller-asserted [identity](../../../../glossary.md#term-identity-meanings); [lifecycle](../../../../glossary.md#lifecycle)
completion rather than numerical equivalence; in-memory rich evidence; no [graph](../../../../glossary.md#graph)/plan tuning.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context reused by this documentation pass ran
  `./gradlew :modules:engine:test :testing:architecture-tests:test
  :testing:integration-tests:test`: passed with Engine 67, architecture 9, and integration 10
  tests (86 total), with zero failures, errors, or skips. Executable Java did not change during
  documentation context <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>`01a0b324-31c9-70d1-95c0-e0f9710b6571`, so the suites were not repeated.
- Independent review context `01a0b316-61a5-7930-9374-86a3384149fd` reported no correctness
  findings; this pass reused that result.
- `./gradlew :modules:engine:javadoc`: passed after the final [Javadoc](../../../../glossary.md#term-javadoc) edits with no warnings.
- `./gradlew test`: passed the required repository checkpoint in 2m 46s; 70 actionable tasks,
  10 executed and 60 up-to-date. The expected native-access and incubator-module warnings did not
  fail the build.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [module glossary](../../../../glossary.md#term-module))</span>
- `/tmp/validate_synaptik_markdown.py` over the six explanatory and three planning paths passed
  relative targets and anchors, unique headings, balanced fences, final newlines, and trailing
  whitespace.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [targets glossary](../../../../glossary.md#term-target))</span>
- A source-edit audit of the documentation pass confirmed its Java changes are confined to
  comments/Javadoc; it changed no executable Java token. <span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token))</span>`git diff --check` passed.
- Exact final inventory validation passed with 23 paths: 8 production/build, 6 test, 6 focused
  explanatory documentation, and 3 planning paths. The extra test path is the recorded
  `AdvancedEnginePublicShapeTest` method-inventory coverage adjustment.
- Documentation types and profiles: `public-api.md` and Java contracts used General plus API and
  Javadoc style and Example format; architecture explanations used General plus Architecture
  style; <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`benchmarking.md` used General, Developer-guide, and Example profiles; the glossary used
  General style; this task, master plan, and roadmap used General plus Planning style.
- `ARCHITECTURE.md` and ADRs need no change: the direct `modules/engine -> tools/tuning` edge
  realizes the existing Engine composition-root and tuning ownership contract. Runtime and other
  modules need no change because tuning completes before Runtime and no inward contract changed.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Implementation notes

- Added exactly the requested public request/result surface and ordinary `prepareTuned` entry,
  retaining CPU/tool/Prepare types behind Engine-owned translation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Cache-first tuning maps the sole handoff to occurrence index 0, partition index 0, and weight 1.
  Compatible hits run no trial; misses use configured warmups and timed samples with median winner
  selection. Trials use fresh Runtime state and every successful tuned/fallback path freshly
  prepares production state.<span data-glossary-navigation="true"> ([Cache glossary](../../../../glossary.md#term-cache); [handoff glossary](../../../../glossary.md#term-handoff); [occurrence glossary](../../../../glossary.md#term-occurrence); [partition glossary](../../../../glossary.md#partition); [weight glossary](../../../../glossary.md#term-weight); [run glossary](../../../../glossary.md#term-run); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [median glossary](../../../../glossary.md#term-median); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `TUNED` has authenticated [immutable](../../../../glossary.md#term-immutable) evidence; `SAFE_HEURISTIC_FALLBACK` has none. Caller model
  and profile identities label evidence and do not authorize cache compatibility or selection.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Current limitations remain CPU-only, one representative set, at most one local workload, and no
  graph/plan or multi-occurrence tuning.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>

## Completion summary

- Completed changes: composed Config, tuning, CPU, Prepare, representative execution, fallback,
  evidence translation, and the public ordinary Engine API.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Files changed or created: exact 23-path inventory recorded above.
- Tests and validation: reused 86 focused tests and independent review; passed Engine Javadoc,
  repository-wide tests, Markdown checks, source-token audit, scope audit, and whitespace checks.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [token glossary](../../../../glossary.md#term-token); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `01a0b324-31c9-70d1-95c0-e0f9710b6571` finalized all
  affected Javadocs, six explanatory paths, glossary terminology, and three planning paths.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: public lifecycle/example, tuning and runtime boundaries, dependency edge,
  benchmarking guidance, glossary, status, and next frontier are synchronized.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [runtime glossary](../../../../glossary.md#term-runtime); [dependency edge glossary](../../../../glossary.md#term-dependency-edge); [benchmarking glossary](../../../../glossary.md#term-benchmarking))</span>
- Javadoc review: every affected public/package-private tuning contract was reviewed; new public
  request/evidence ownership, immutability, lifecycle, result, and failure contracts are detailed.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: added the [Engine](../../../../glossary.md#term-engine) request/result distinction and current bounded composition.
- Unresolved issues: None.
- Follow-up required: None. Engine 0008 remains the next Draft frontier.

Status: Complete
