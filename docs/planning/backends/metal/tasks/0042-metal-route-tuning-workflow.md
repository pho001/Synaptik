# Task 0042: [Metal](../../../../glossary.md#term-metal) Route Tuning Workflow

## Status

Complete

Implemented at `9feb2505705263b6efb417d606678c606c2b9598` from exact clean [planning](../../../../glossary.md#term-planning)
base `31ab01a65ff1e4842a43471a54e74544fb603760`. The bounded implementation and
[validation](../../../../glossary.md#term-validation) evidence below are complete; final independent Class C review returned `APPROVE` with
zero findings.

## Change class

Class C — this adds two supported [Metal](../../../../glossary.md#term-metal) collaborations, admits Metal to the public two-phase
`Engine.prepareTuned(...)` [lifecycle](../../../../glossary.md#lifecycle), changes cold route selection under authenticated tuning, and
generalizes [Engine](../../../../glossary.md#term-engine) fallback/[identity](../../../../glossary.md#term-identity-meanings) composition. Use one clean implementation context, exactly one
repository-wide build after focused [validation](../../../../glossary.md#term-validation), and an independent targeted Class C review.

## Goal

Connect the existing version-twelve [Metal](../../../../glossary.md#term-metal) [singleton](../../../../glossary.md#term-singleton)-`NEG` route candidate and decision foundation to
the current public [model](../../../../glossary.md#term-model)-[autotuning](../../../../glossary.md#autotuning--model-autotuning) workflow without exposing Metal-private route values, moving
selection into shared layers, changing the public request/result [shape](../../../../glossary.md#shape), or claiming persistent Metal
reuse.

The complete supported flow is:

```text
one exact non-empty Metal-owned singleton-NEG partition
  -> fresh Metal analysis proves both CUSTOM_SINGLE_NEG and MPSGRAPH valid
  -> public retained MetalLocalWorkloadTuning exposes an opaque decision-empty batch
  -> Engine adapts that batch to tools/tuning Phase 1
  -> representative executions measure both complete local routes
  -> Metal authenticates the exact selected Phase-1 decision
  -> public retained MetalCompletePlanTuning exposes one fixed complete-plan candidate
  -> tools/tuning Phase 2 captures correctness and measures that sole complete plan
  -> representative inputs are cleaned
  -> Metal freshly prepares the authenticated winner
  -> public PreparedExecution runs and materializes the exact NEG result
```

A valid Metal [partition](../../../../glossary.md#partition) with only [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph) is not a tuning workload. Ordinary Metal preparation
continues to use its correct deterministic [heuristic](../../../../glossary.md#term-heuristic), and an allowed tuning fallback invokes that
same [owner](../../../../glossary.md#term-owner)-local ordinary path.

## Existing foundation and bounded design decision

No separate shared prerequisite is required. Current contracts already provide:

- method-free `BackendTuningCandidateBatch` and `BackendTuningDecision` roles plus exact-[partition](../../../../glossary.md#partition)
  `BackendPartitionTuningHandoff` transport in [Prepare](../../../../glossary.md#prepare);
- `GraphPreparation.project(...)` and `GraphPreparation.prepare(...)` over [backend](../../../../glossary.md#backend)-owned
  `PartitionPreparation` values;
- generic `BackendWorkloadTuning` and `CompletePlanTuning`, including `SESSION` compatibility and a
  valid non-empty one-candidate complete-plan batch;
- a backend-generic private `AdvancedEngine.ModelAutotuningTuning` type seam; and
- the package-private Metal version-twelve generator, batch, decision, compatibility, and codec,
  plus fresh preparer authentication before declarations.<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication))</span>

The current production connection is nevertheless [CPU](../../../../glossary.md#term-cpu)-only: `AdvancedEngine` rejects a selected
non-CPU adapter, constructs `CpuTuningAdapter`, uses CPU-specific complete-plan policy/[producer](../../../../glossary.md#term-producer)/[codec](../../../../glossary.md#term-codec)
[identities](../../../../glossary.md#term-identity-meanings), reports CPU-only errors, and `RepresentativeExecutionSession` casts ordinary fallback
to CPU. Task 0042 owns the minimum private [Engine](../../../../glossary.md#term-engine) [generalization](../../../../glossary.md#term-generalization) needed to remove those assumptions.
It does not create a new public Engine method or a new shared Prepare/tools contract.

[Metal](../../../../glossary.md#term-metal) has no current complete [graph](../../../../glossary.md#graph), [topology](../../../../glossary.md#term-topology), [layout](../../../../glossary.md#layout), [ownership](../../../../glossary.md#term-ownership-meanings), or representation alternatives.
The exact selected Phase-1 route is therefore the sole current complete Metal plan. A [singleton](../../../../glossary.md#term-singleton)
Phase-2 [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) is truthful and accepted by `CompletePlanTuning`: it freshly prepares the whole plan,
[captures](../../../../glossary.md#term-capture-meanings) the exact [canonical](../../../../glossary.md#term-canonical-form)-[publication](../../../../glossary.md#term-publication) correctness reference, performs the caller-bounded timed
execution, authenticates the fixed winner, and freshly prepares production. It does not repeat
local route search or invent a second plan. Making Phase 2 optional would require a larger public
Config/evidence redesign and is excluded.

## [Scope](../../../../glossary.md#term-scope)

### Public retained [Metal](../../../../glossary.md#term-metal) local-workload collaboration

Add public final `io.github.pho001.synaptik.backend.metal.MetalLocalWorkloadTuning`, acquired only
through `MetalBackendIntegration.localWorkloadTuning()`. One open integration returns the same
retained instance on every call. It has no independent close [operation](../../../../glossary.md#operation) and borrows the integration's
exact context, [trace](../../../../glossary.md#trace) [producer](../../../../glossary.md#term-producer), native [ownership](../../../../glossary.md#term-ownership-meanings), and close [admission](../../../../glossary.md#term-admission).

Its supported surface mirrors the current [CPU](../../../../glossary.md#term-cpu) collaboration while wrapping, not exposing, the
existing private Metal values:

```java
public final class MetalLocalWorkloadTuning {
    public Optional<BackendPartitionTuningHandoff<CandidateBatch, SelectedDecision>>
            candidateHandoff(PrepareContext<?> context);
    public List<Candidate> candidates(CandidateBatch batch);
    public Compatibility compatibility(CandidateBatch batch);
    public CandidateIdentity candidateIdentity(Candidate candidate);
    public SelectedDecision selectedDecision(CandidateBatch batch, Candidate candidate);
    public byte[] encodeDecision(SelectedDecision decision);
    public Optional<SelectedDecision> decodeCompatibleDecision(
            CandidateBatch batch, byte[] encodedDecision);
    public PartitionPreparation<?, ?> trialPreparation(
            CandidateBatch batch, Candidate candidate);
    public PartitionPreparation<?, ?> selectedPreparation(
            CandidateBatch batch, SelectedDecision decision);

    public static final class CandidateBatch implements BackendTuningCandidateBatch { ... }
    public static final class SelectedDecision implements BackendTuningDecision { ... }
    public static final class Candidate { ... }
    public static final class Compatibility { ... }
    public static final class CandidateIdentity { ... }
    public enum ReuseScope { SESSION }
}
```

Nested constructors are not public. [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), candidate, and decision values are [immutable](../../../../glossary.md#term-immutable) and retain
an exact [owner](../../../../glossary.md#term-owner)/projection/batch association. They expose no route enum, workload [signature](../../../../glossary.md#term-signature), [device
context](../../../../glossary.md#device-submission-synchronization-and-execution-context), session nonce, native value, preparation plan, resource geometry, or trace producer.
Compatibility and candidate [identity](../../../../glossary.md#term-identity-meanings) expose only [schema](../../../../glossary.md#term-schema)/reuse [metadata](../../../../glossary.md#term-metadata) and defensive [canonical](../../../../glossary.md#term-canonical-form) byte
copies. Foreign owner, batch, [partition](../../../../glossary.md#partition), projection, or candidate values fail before preparation.

`candidateHandoff(...)` performs fresh ordinary Metal analysis and generates the full current batch
with a fixed internal maximum of two, independent of the caller's later tuning budget. It returns a
[handoff](../../../../glossary.md#term-handoff) only when the exact partition is the current [singleton](../../../../glossary.md#term-singleton)-`NEG` domain and fresh generation
contains exactly the complete ordered pair `CUSTOM_SINGLE_NEG`, then [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph). A valid one-candidate
`MPSGRAPH` batch returns empty; it is not measured or reported as tuned. Invalid or non-Metal
projections retain their current failures. [Engine](../../../../glossary.md#term-engine) separately enforces exactly one non-empty
Metal-owned partition before projection.

Enumeration, compatibility, identity, selection, encoding, and decoding delegate to the existing
version-twelve private generator/batch/decision/[codec](../../../../glossary.md#term-codec) semantics. The public values add association
safety only. Decision decode snapshots and bounds caller bytes and returns empty for malformed,
trailing, corrupt, unsupported, stale, changed-workload, wrong-session, unknown-candidate, or
pruned-candidate input.

`trialPreparation(...)` and `selectedPreparation(...)` construct a fresh
`MetalNegAnalysisInputs` carrying the exact decision-present handoff and the integration's retained
`MetalTraceProducer`, then return a Metal-owned `PartitionPreparation`. Fresh preparer analysis
regenerates and authenticates all version-twelve facts before fixing the route and declaring
resources. There is no silent [heuristic](../../../../glossary.md#term-heuristic) fallback after a supplied decision fails [authentication](../../../../glossary.md#term-authentication).
The [backend](../../../../glossary.md#backend) collaboration never binds representative inputs, executes, measures, selects a winner,
or accesses a [cache](../../../../glossary.md#term-cache) path.

### Public retained [Metal](../../../../glossary.md#term-metal) complete-plan collaboration

Add public final `io.github.pho001.synaptik.backend.metal.MetalCompletePlanTuning`, acquired only
through `MetalBackendIntegration.completePlanTuning()`. It retains the exact
`MetalLocalWorkloadTuning` collaboration and has the same integration [lifetime](../../../../glossary.md#term-lifetime) and [concurrency](../../../../glossary.md#term-concurrency)
rules.

It exposes the same [opaque](../../../../glossary.md#term-opaque) [operation](../../../../glossary.md#operation) family as `CpuCompletePlanTuning`: exact [handoff](../../../../glossary.md#term-handoff) construction,
ordered candidates, compatibility, [identity](../../../../glossary.md#term-identity-meanings), selection, bounded decision encode/decode, fresh trial
preparation, and fresh selected preparation. Its handoff requires the exact current projected
context and exact authenticated `MetalLocalWorkloadTuning.SelectedDecision`. It freshly proves that
the Phase-1 [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) remains the complete eligible two-route batch and that the selected decision
belongs to that projection.

```java
public final class MetalCompletePlanTuning {
    public BackendPartitionTuningHandoff<CandidateBatch, SelectedDecision>
            candidateHandoff(
                    PrepareContext<?> context,
                    MetalLocalWorkloadTuning.SelectedDecision phaseOneDecision);
    public List<Candidate> candidates(CandidateBatch batch);
    public Compatibility compatibility(CandidateBatch batch);
    public CandidateIdentity candidateIdentity(Candidate candidate);
    public SelectedDecision selectedDecision(CandidateBatch batch, Candidate candidate);
    public byte[] encodeDecision(SelectedDecision decision);
    public Optional<SelectedDecision> decodeCompatibleDecision(
            CandidateBatch batch, byte[] encodedDecision);
    public PartitionPreparation<?, ?> trialPreparation(
            CandidateBatch batch, Candidate candidate);
    public PartitionPreparation<?, ?> selectedPreparation(
            CandidateBatch batch, SelectedDecision decision);

    public static final class CandidateBatch implements BackendTuningCandidateBatch { ... }
    public static final class SelectedDecision implements BackendTuningDecision { ... }
    public static final class Candidate { ... }
    public static final class Compatibility { ... }
    public static final class CandidateIdentity { ... }
    public enum ReuseScope { SESSION }
}
```

`candidateHandoff(...)` either returns the decision-empty exact [singleton](../../../../glossary.md#term-singleton) batch or fails when the
supplied Phase-1 value is stale, foreign, or no longer eligible; it does not turn an invalid
decision into an empty optional. The private [Engine](../../../../glossary.md#term-engine) adapter wraps this successful handoff in its
existing optional return [shape](../../../../glossary.md#shape).

The resulting complete-plan batch contains exactly one opaque candidate: the fixed whole-[model](../../../../glossary.md#term-model) plan
whose sole [partition](../../../../glossary.md#partition) uses the authenticated Phase-1 route. Its compatibility and decision encoding
include the complete versioned Phase-1 compatibility and selected decision, exact live integration
session, fixed-plan meaning, and projection facts needed to reject foreign or stale values. Trial
and selected preparation delegate through fresh local selected preparation; they do not enumerate,
[rank](../../../../glossary.md#term-rank), or substitute a local route.

Define distinct outer Metal fixed-plan identities rather than reusing [CPU](../../../../glossary.md#term-cpu) identity bytes:

- one Metal fixed-plan policy identity for exact canonical-byte correctness, one fixed owner/partition,
  and the already-selected local route;<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [owner glossary](../../../../glossary.md#term-owner))</span>
- one Metal fixed-plan [producer](../../../../glossary.md#term-producer) identity; and
- one Metal fixed-plan decision-[codec](../../../../glossary.md#term-codec) identity.

Use explicit fixed numeric tags and an initial outer [schema](../../../../glossary.md#term-schema) owned by this new concept. Do not use
class names, enum names, Java [serialization](../../../../glossary.md#term-serialization), strings, reflection, or native addresses. Preserve the
existing CPU policy/producer/codec schema numbers and bytes exactly.

### Session-only [persistence](../../../../glossary.md#term-persistence) contract

Both [Metal](../../../../glossary.md#term-metal) collaborations declare only `SESSION` reuse. Local compatibility continues to contain
the live `MetalDeviceContext` session nonce through the existing version-twelve [codec](../../../../glossary.md#term-codec). The fixed
complete-plan compatibility is also bound to the exact live integration and Phase-1 decision.

The current generic `WorkloadTuning` [transaction](../../../../glossary.md#term-transaction) still loads and validates the request's explicit
workload-[cache](../../../../glossary.md#term-cache) path before applying per-workload reuse [scope](../../../../glossary.md#term-scope). Task 0042 does not change that tools
contract. A Metal `SESSION` workload can neither produce a persistent hit nor publish/update a Metal
entry. `CompletePlanTuning` performs no [model](../../../../glossary.md#term-model)-plan-cache filesystem [operation](../../../../glossary.md#operation) for its `SESSION`
[producer](../../../../glossary.md#term-producer). Tests use absent temporary paths and prove neither path is created or changed.

Do not add a stable [device](../../../../glossary.md#term-device)/library [fingerprint](../../../../glossary.md#term-fingerprint), promote either phase to `PERSISTENT`, change either
cache [artifact](../../../../glossary.md#term-artifact), add a Metal cache file, or make Metal interpret model/profile [identities](../../../../glossary.md#term-identity-meanings). The
caller-defined model and representative-profile identities remain [Engine](../../../../glossary.md#term-engine) evidence values. The
[compile](../../../../glossary.md#compile) numerical profile remains inside existing Metal version-twelve compatibility.

Metal [trace](../../../../glossary.md#trace) continues to report `NOT_QUERIED`: the [backend](../../../../glossary.md#backend) authenticates an optional decision but
does not observe whether outer tools loaded a path or measured a miss. Every trial and final
preparation retains the exact integration trace producer; tuning must not silently select an
untraced construction path.

### Minimal private [Engine](../../../../glossary.md#term-engine) [generalization](../../../../glossary.md#term-generalization)

Add a private `MetalTuningAdapter` beside `CpuTuningAdapter` and extend
`MetalEngineBackendComposition` with the exact local/complete collaboration accessors,
one-[partition](../../../../glossary.md#partition) `projectedContext(...)`, and [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) composition needed for fresh trial and
selected preparations.

[Dispatch](../../../../glossary.md#term-dispatch) the private tuning adapter from the already-selected [owner](../../../../glossary.md#term-owner) adapter:

- `CpuEngineBackendComposition` retains the exact current [CPU](../../../../glossary.md#term-cpu) flow;
- `MetalEngineBackendComposition` uses the new bounded [Metal](../../../../glossary.md#term-metal) flow; and
- empty, missing, mixed-owner, multiple-partition, or any other unsupported composition still fails
  before candidate work under its existing ownership/projection boundary.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

Remove CPU-specific assumptions only from the shared private path: [backend](../../../../glossary.md#backend)-neutral eligibility/error
wording, adapter dispatch, complete-plan policy [identity](../../../../glossary.md#term-identity-meanings) supply, and ordinary fallback composition.
`ALLOW_SAFE_HEURISTIC` prepares through the exact already-selected adapter and can never switch
owner. Preserve cleanup, exception identity, suppression ordering, representative-session poisoning,
[admission](../../../../glossary.md#term-admission), fresh production preparation, and final [publication](../../../../glossary.md#term-publication) behavior.

Make the private tuning seam supply its complete-plan policy identity. CPU continues to return the
exact existing policy bytes, and `CpuTuningAdapter` continues to construct the exact existing CPU
[producer](../../../../glossary.md#term-producer)/[codec](../../../../glossary.md#term-codec) identities. Metal supplies its distinct fixed-plan identities. The generic [target](../../../../glossary.md#term-target)
identity wrapper, correctness policy, budgets, evidence translation, and tools algorithms remain
unchanged.

[`ModelAutotuningConfig`](../../../../glossary.md#model-autotuning-request-facade--modelautotuningconfig), `ModelAutotuningRequest`, `ModelAutotuningPreparation`, `Engine`, and their
public methods/records/enums remain source-contract compatible. A Metal `TUNED` result carries the
existing public evidence [shape](../../../../glossary.md#shape): one Phase-1 workload with two measured candidates and one measured
[complete-plan candidate](../../../../glossary.md#complete-plan-candidate).

## Non-goals

- Mixed-owner tuning, multiple Metal partitions, multiple workload occurrences, deduplication across
  Engine-extracted occurrences, or owner/partition alternatives.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine); [partition glossary](../../../../glossary.md#partition))</span>
- Reporting an [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph)-only partition as tuned, measuring the heuristic against itself in Phase
  1, adding a route, or widening singleton-<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic); [widening glossary](../../../../glossary.md#term-widening); [singleton glossary](../../../../glossary.md#term-singleton))</span>`NEG` eligibility.
- Optional Phase 2, public Config/result changes, a new top-level Engine operation, generic
  graph/plan tuning, Compiler graph alternatives, or Planning ownership search.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Persistent/cross-session Metal decisions, stable device/library fingerprints, workload/model-plan
  cache format or lifecycle changes, cache inspection, executable persistence, or hidden global
  state.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [model glossary](../../../../glossary.md#term-model); [cache glossary](../../../../glossary.md#term-cache); [lifecycle glossary](../../../../glossary.md#lifecycle); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Benchmark tasks, baseline/report generation, performance matrices, device preprobes, broad Shape
  sweeps, or any performance claim beyond selecting the measured request-local winner.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings); [Shape glossary](../../../../glossary.md#shape))</span>
- Native [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary)/C source, [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, thirteen exports, [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) 11, [operation](../../../../glossary.md#operation) wires `1..19`,
  capability, availability, discovery, storage/transfer domains, route numerics, or Runtime hot-path
  changes.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Changes to existing version-twelve workload, exact-policy, candidate, compatibility, route-policy,
  or codec meaning merely because an outer adapter now consumes them.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>
- CPU candidate semantics, CPU persistence scope, CPU complete-plan alternatives, CPU identity bytes,
  or CPU public behavior.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [scope glossary](../../../../glossary.md#term-scope); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) keeps route
  selection cold and Runtime free of tuning.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>
  assigns candidate meaning, compatibility, decision authentication, lowering, and resources to
  Metal.<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication); [lowering glossary](../../../../glossary.md#lowering))</span>
- [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  assigns timing, winner selection, cache orchestration, and evidence to <span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>`tools/tuning` and [Engine](../../../../glossary.md#term-engine).
- [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)
  transports opaque backend values and composes fresh partition preparations.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [backend glossary](../../../../glossary.md#backend); [partition glossary](../../../../glossary.md#partition))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  requires the authenticated route and all resources before assignment/finalization.
- [Compile ownership and mixed-owner Prepare routing](../../../../architecture/contracts/runtime-prepare-engine.md#compile-ownership-and-mixed-owner-prepare-routing)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
  retains fail-closed selected-owner composition and excludes mixed-owner tuning.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- [ADR 0002](../../../../design/decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>,
  [ADR 0008](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md),
  [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md),
  [ADR 0013](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md),
  and [ADR 0015](../../../../design/decisions/0015-explicit-engine-backend-composition.md) remain
  authoritative.

Stop and return to [planning](../../../../glossary.md#term-planning) if implementation needs a new shared [Prepare](../../../../glossary.md#prepare)/tools [API](../../../../glossary.md#term-application-programming-interface-api), a public
Config/result break, persistent [Metal](../../../../glossary.md#term-metal) reuse, a native/capability/[schema](../../../../glossary.md#term-schema) change, mixed-[owner](../../../../glossary.md#term-owner) tuning,
or an alternate correctness/timing algorithm.

## Dependencies and integration

- Depends on: Metal 0004 and 0041 Complete; Prepare 0004/0008; tools/tuning 0001–0002/0004; Config
  0006A–0006B; CPU 0010I–0010J as public collaboration precedent; Engine 0006A–0009/0011/0015–0018;
  Runtime 0016.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Conflicts with: every concurrent Metal integration/preparation/candidate/codec/trace/API scope;
  Engine tuning/composition/lifecycle/fallback scope; CPU tuning identity changes; tools/tuning,
  public API, backend guide, architecture, master-plan, or roadmap edits.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [trace glossary](../../../../glossary.md#trace); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scope glossary](../../../../glossary.md#term-scope); [lifecycle glossary](../../../../glossary.md#lifecycle); [identity glossary](../../../../glossary.md#term-identity-meanings); [backend glossary](../../../../glossary.md#backend))</span>
- Parallel group: None.
- Common base revision: `31ab01a65ff1e4842a43471a54e74544fb603760`.
- Integration order: one serial implementation from the exact base, focused validation and docs,
  exactly one full repository build, then independent Class C review.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): [Metal](../../../../glossary.md#term-metal) 0042 executor.

## Files and symbols

Expected production surface:

- add `MetalLocalWorkloadTuning` and `MetalCompletePlanTuning` in the supported [Metal](../../../../glossary.md#term-metal) root package;
- `MetalBackendIntegration`, `MetalBackendRuntime`, and Metal package Javadoc retain/expose them and
  propagate the exact context/trace producer;<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [trace glossary](../../../../glossary.md#trace); [producer glossary](../../../../glossary.md#term-producer))</span>
- existing `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningDecision`,
  `MetalNegTuningCodec`, `MetalNegAnalysisInputs`, and `MetalNegPartitionPreparer` remain private and
  change only where the wrappers need association/fresh-authentication support;<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication))</span>
- `MetalEngineBackendComposition`, `AdvancedEngine`, and `RepresentativeExecutionSession` implement
  the private Metal adapter, identity supply, dispatch, and backend-neutral fallback;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [dispatch glossary](../../../../glossary.md#term-dispatch); [backend glossary](../../../../glossary.md#backend))</span>
- no production [dependency](../../../../glossary.md#term-dependency) is added from Metal to [Engine](../../../../glossary.md#term-engine) or `tools/tuning`.

Expected focused tests:

- add [lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) fake-native `MetalLocalWorkloadTuningTest` and `MetalCompletePlanTuningTest` coverage;
- update `ModelAutotuningCompositionTest` and `CompletePlanAutotuningCompositionTest` only for the
  backend-neutral private seam and CPU identity/behavior regression;<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- add one focused public scenario to `EngineExplicitCompositionMetalIntegrationTest`.

Update implementation Javadocs plus the exact current [CPU](../../../../glossary.md#term-cpu)-only/Metal-foundation statements found in
`docs/backend-guide/metal-backend.md`, `docs/api/public-api.md`, the [runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/Engine and
performance-tuning architecture explanations, `tools/tuning/package-info.java`, and [planning](../../../../glossary.md#term-planning) status.
Do not rewrite unrelated historical evidence.

## Acceptance criteria

1. One open `MetalBackendIntegration` returns the same retained local and complete-plan
   collaborations. They are concurrency-safe, own no independent native resource, reject new work
   after integration close, and expose only opaque association-checked values and defensive bytes.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency); [opaque glossary](../../../../glossary.md#term-opaque))</span>
2. Fresh exact [singleton](../../../../glossary.md#term-singleton)-`NEG` analysis exposes precisely two local candidates in safe-heuristic
   order. Every valid <span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>[`MPSGRAPH`](../../../../glossary.md#term-mpsgraph)-only partition returns no local tuning handoff. Caller budget does
   not prune the backend batch; the generic tool rejects an insufficient maximum of its own.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [handoff glossary](../../../../glossary.md#term-handoff); [backend glossary](../../../../glossary.md#backend); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
3. Local compatibility and decision handling exactly reuse version-twelve semantics and remain
   `SESSION`. Malformed, corrupt, stale, foreign-owner/session/projection, unknown, or cross-batch
   values fail closed. Trial and selected preparation freshly authenticate before exact custom
   versus MPSGraph declarations, with no heuristic substitution.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
4. The complete-plan collaboration accepts only the exact authenticated local winner and exposes
   one fixed complete candidate. Its distinct outer Metal policy/producer/codec identities,
   compatibility, candidate identity, and decision round trip are deterministic, bounded,
   defensive, session-bound, and distinguishable from CPU. Fresh preparation keeps the Phase-1
   route fixed.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [codec glossary](../../../../glossary.md#term-codec); [identity glossary](../../../../glossary.md#term-identity-meanings); [round trip glossary](../../../../glossary.md#term-round-trip); [CPU glossary](../../../../glossary.md#term-cpu))</span>
5. `WorkloadTuning` cannot hit or publish a persistent [Metal](../../../../glossary.md#term-metal) entry; `CompletePlanTuning` performs no
   model-plan cache I/O. Existing generic cache loading/validation behavior and both artifact
   formats remain unchanged.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation); [artifact glossary](../../../../glossary.md#term-artifact))</span>
6. Engine dispatches the selected single-owner CPU or Metal adapter. Metal multiple-partition and
   mixed-owner plans remain unsupported for tuning. Allowed fallback uses exactly the selected
   owner's ordinary preparation and never switches owner; required mode preserves the tuning
   failure. Cleanup, poisoning, suppression, close races, and final publication remain correct.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [publication glossary](../../../../glossary.md#term-publication))</span>
7. CPU tuning produces byte-for-byte identical complete-plan policy, producer, and codec identities
   and retains its existing eligibility, candidates, persistence scope, fallback, evidence, and
   public behavior.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [scope glossary](../../../../glossary.md#term-scope))</span>
8. Existing public [`ModelAutotuningConfig`](../../../../glossary.md#model-autotuning-request-facade--modelautotuningconfig) and `ModelAutotuningPreparation` signatures and invariants
   are unchanged. Metal evidence uses the existing two-phase shape and truthfully reports two local
   measured rows plus one measured complete-plan row.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [shape glossary](../../../../glossary.md#shape))</span>
9. Every tuned trial and final preparation retains the integration's trace producer. Preparation
   and invocation events remain route-truthful and <span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>`NOT_QUERIED`; tuning adds no [cache](../../../../glossary.md#term-cache)-status claim.
10. No native source/export/ABI/node schema/wire, capability, numerical profile, route semantics,
    transfer, Runtime hot path, benchmark, matrix, device preprobe, or existing Metal version-twelve
    identity changes.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [node glossary](../../../../glossary.md#node); [schema glossary](../../../../glossary.md#term-schema); [Runtime glossary](../../../../glossary.md#term-runtime); [hot path glossary](../../../../glossary.md#term-hot-path); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [device glossary](../../../../glossary.md#term-device))</span>
11. Focused Metal, Engine, CPU-regression, architecture, Javadoc, documentation, and diff checks
    pass; the one real public Metal workflow passes; exactly one full repository build passes; and
    independent Class C review approves.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation)

Implementation validation, in this order:

```bash
./gradlew :backends:metal:test \
  --tests '*MetalLocalWorkloadTuningTest' \
  --tests '*MetalCompletePlanTuningTest' \
  --tests '*MetalNegRouteCandidateGeneratorTest' \
  --tests '*MetalNegPreparedExecutionTest'
./gradlew :modules:engine:test \
  --tests '*ModelAutotuningCompositionTest' \
  --tests '*CompletePlanAutotuningCompositionTest'
./gradlew :testing:integration-tests:test \
  --tests 'io.github.pho001.synaptik.testing.integration.EngineExplicitCompositionMetalIntegrationTest.publicMetalRouteTuningPreparesAndRunsSingletonNeg'
./gradlew :testing:architecture-tests:test \
  --tests 'io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest'
./gradlew :backends:metal:javadoc :modules:engine:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/backend-guide/metal-backend.md docs/api/public-api.md \
  docs/architecture/contracts/runtime-prepare-engine.md \
  docs/architecture/lifecycle.md docs/architecture/module-boundaries.md \
  docs/architecture/performance-evidence-and-tuning.md \
  docs/architecture/runtime-prepare-backend-boundary.md \
  docs/planning/backends/metal/tasks/0042-metal-route-tuning-workflow.md \
  docs/planning/backends/metal/master-plan.md docs/planning/roadmap.md
./gradlew build
git diff --check
git status --short
```

The implementation [owner](../../../../glossary.md#term-owner) runs the full repository build exactly once after all focused checks and
documentation are complete. Independent review inspects the exact diff and generated focused/full
reports; it does not [run](../../../../glossary.md#term-run) a performance [matrix](../../../../glossary.md#term-matrix-meanings) or preprobe a [device](../../../../glossary.md#term-device). Review remediation reruns only
the checks affected by the finding unless production changes invalidate the single full-build
evidence.

### Implementation evidence

- The focused fake-native [Metal](../../../../glossary.md#term-metal) command passed
  `MetalLocalWorkloadTuningTest`, `MetalCompletePlanTuningTest`,
  `MetalNegRouteCandidateGeneratorTest`, and `MetalNegPreparedExecutionTest`.
- The focused [Engine](../../../../glossary.md#term-engine) command passed `ModelAutotuningCompositionTest`,
  `CompletePlanAutotuningCompositionTest`, and `RepresentativeExecutionSessionTest`. The exact
  regression assertions preserve CPU policy/producer/codec bytes and prove all three Metal
  identities differ.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- `./native/metal-macos-arm64/build.sh` passed. With its resulting [dylib](../../../../glossary.md#term-dylib) supplied through
  `SYNAPTIK_METAL_TEST_LIBRARY`, the one bounded
  `publicMetalRouteTuningPreparesAndRunsSingletonNeg` scenario passed on the actual Metal device.
  It observed `TUNED`, two measured session local rows, one measured session complete-plan row,
  absent cache files, and exact public NEG output.
- The focused `EngineCompositionContractTest` architecture check passed. Final
  `:backends:metal:javadoc :modules:engine:javadoc` completed without warnings.
- The ten-file Markdown validator passed, and `git diff --check` reported no errors.
- Exactly one full `./gradlew build` passed with 87 actionable tasks: 8 executed, 1 from cache,
  and 78 up to date. No performance matrix or device preprobe ran.
- Final independent Class C review inspected the exact implementation and evidence and returned
  `APPROVE` with zero findings. Task status is `Complete`.

The required public integration scenario is one bounded workflow, not a benchmark:

- use `Engine.builder()` with the ordinary configured Metal integration and no capability/device
  preprobe;
- [compile](../../../../glossary.md#compile) one positive static [canonical](../../../../glossary.md#term-canonical-form) [`FLOAT32`](../../../../glossary.md#term-float32) [singleton](../../../../glossary.md#term-singleton) `NEG` over one Metal [partition](../../../../glossary.md#partition);
- supply representative [host](../../../../glossary.md#term-host) input plus absent temporary workload/[model](../../../../glossary.md#term-model)-plan [cache](../../../../glossary.md#term-cache) paths;
- use Phase-1 budget `maximumDistinctCacheMisses=1`, `maximumCandidatesPerMiss=2`, zero warmups, and
  one timed sample; use Phase-2 maximum one candidate, zero warmups, one timed sample, total
  execution ceiling two, and the exact small correctness-byte ceiling;<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- assert `TUNED`, `SESSION`/`MEASURED` evidence, two local candidate rows, one fixed complete-plan
  row, and unchanged absent cache files;
- [run](../../../../glossary.md#term-run) the returned public [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) once and compare exact canonical NEG output bytes;
- close result, preparation, Engine, and integration-owned resources through the public [lifecycle](../../../../glossary.md#lifecycle).

Do not add [Shape](../../../../glossary.md#shape) sweeps, route timing assertions, winner assumptions, repeated-[device](../../../../glossary.md#term-device) discovery,
baselines, reports, or a standalone probe. Fake-native tests own deterministic route, association,
[codec](../../../../glossary.md#term-codec), [trace](../../../../glossary.md#trace), failure, and closure assertions. Existing [CPU](../../../../glossary.md#term-cpu) tests own [regression](../../../../glossary.md#term-regression) proof rather than a
second integration [matrix](../../../../glossary.md#term-matrix-meanings).

## Documentation and review impact

Document the two retained [Metal](../../../../glossary.md#term-metal) collaborations, [opaque](../../../../glossary.md#term-opaque) values, exact eligibility, fixed Phase-2
meaning, session-only reuse, generic workload-[cache](../../../../glossary.md#term-cache) read behavior, no Metal hit/[publication](../../../../glossary.md#term-publication),
[model](../../../../glossary.md#term-model)-plan no-I/O behavior, [trace](../../../../glossary.md#trace) `NOT_QUERIED`, public [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle), unchanged public request/
result [shape](../../../../glossary.md#shape), and all exclusions. Replace current statements that Metal has no Engine/tools adapter
or that `prepareTuned(...)` is universally [CPU](../../../../glossary.md#term-cpu)-only; retain CPU-only statements about
`Engine.standard()` and `AdvancedEngine.takeOwnership(CpuBackendIntegration)`.

Independent Class C review must cover public [API](../../../../glossary.md#term-application-programming-interface-api) minimality, [owner](../../../../glossary.md#term-owner)/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/projection association,
[codec](../../../../glossary.md#term-codec) bounds and stale-session rejection, exact candidate completeness, fresh resource declaration,
[singleton](../../../../glossary.md#term-singleton) complete-plan truthfulness, cache [scope](../../../../glossary.md#term-scope), trace propagation, CPU [identity](../../../../glossary.md#term-identity-meanings) byte stability,
Engine lifecycle/fallback failure precedence, actual public output, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), and docs.
