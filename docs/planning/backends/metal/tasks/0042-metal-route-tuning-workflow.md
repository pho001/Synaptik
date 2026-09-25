# Task 0042: Metal Route Tuning Workflow

## Status

Ready

This is the sole repository `Ready` task. Execute from exact clean planning base
`a961c0086c72b8457db1ea2b645810d60fed0a9d`; do not combine it with another frontier.

## Change class

Class C — this adds two supported Metal collaborations, admits Metal to the public two-phase
`Engine.prepareTuned(...)` lifecycle, changes cold route selection under authenticated tuning, and
generalizes Engine fallback/identity composition. Use one clean implementation context, exactly one
repository-wide build after focused validation, and an independent targeted Class C review.

## Goal

Connect the existing version-twelve Metal singleton-`NEG` route candidate and decision foundation to
the current public model-autotuning workflow without exposing Metal-private route values, moving
selection into shared layers, changing the public request/result shape, or claiming persistent Metal
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

A valid Metal partition with only `MPSGRAPH` is not a tuning workload. Ordinary Metal preparation
continues to use its correct deterministic heuristic, and an allowed tuning fallback invokes that
same owner-local ordinary path.

## Existing foundation and bounded design decision

No separate shared prerequisite is required. Current contracts already provide:

- method-free `BackendTuningCandidateBatch` and `BackendTuningDecision` roles plus exact-partition
  `BackendPartitionTuningHandoff` transport in Prepare;
- `GraphPreparation.project(...)` and `GraphPreparation.prepare(...)` over backend-owned
  `PartitionPreparation` values;
- generic `BackendWorkloadTuning` and `CompletePlanTuning`, including `SESSION` compatibility and a
  valid non-empty one-candidate complete-plan batch;
- a backend-generic private `AdvancedEngine.ModelAutotuningTuning` type seam; and
- the package-private Metal version-twelve generator, batch, decision, compatibility, and codec,
  plus fresh preparer authentication before declarations.

The current production connection is nevertheless CPU-only: `AdvancedEngine` rejects a selected
non-CPU adapter, constructs `CpuTuningAdapter`, uses CPU-specific complete-plan policy/producer/codec
identities, reports CPU-only errors, and `RepresentativeExecutionSession` casts ordinary fallback
to CPU. Task 0042 owns the minimum private Engine generalization needed to remove those assumptions.
It does not create a new public Engine method or a new shared Prepare/tools contract.

Metal has no current complete graph, topology, layout, ownership, or representation alternatives.
The exact selected Phase-1 route is therefore the sole current complete Metal plan. A singleton
Phase-2 batch is truthful and accepted by `CompletePlanTuning`: it freshly prepares the whole plan,
captures the exact canonical-publication correctness reference, performs the caller-bounded timed
execution, authenticates the fixed winner, and freshly prepares production. It does not repeat
local route search or invent a second plan. Making Phase 2 optional would require a larger public
Config/evidence redesign and is excluded.

## Scope

### Public retained Metal local-workload collaboration

Add public final `io.github.pho001.synaptik.backend.metal.MetalLocalWorkloadTuning`, acquired only
through `MetalBackendIntegration.localWorkloadTuning()`. One open integration returns the same
retained instance on every call. It has no independent close operation and borrows the integration's
exact context, trace producer, native ownership, and close admission.

Its supported surface mirrors the current CPU collaboration while wrapping, not exposing, the
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

Nested constructors are not public. Batch, candidate, and decision values are immutable and retain
an exact owner/projection/batch association. They expose no route enum, workload signature, device
context, session nonce, native value, preparation plan, resource geometry, or trace producer.
Compatibility and candidate identity expose only schema/reuse metadata and defensive canonical byte
copies. Foreign owner, batch, partition, projection, or candidate values fail before preparation.

`candidateHandoff(...)` performs fresh ordinary Metal analysis and generates the full current batch
with a fixed internal maximum of two, independent of the caller's later tuning budget. It returns a
handoff only when the exact partition is the current singleton-`NEG` domain and fresh generation
contains exactly the complete ordered pair `CUSTOM_SINGLE_NEG`, then `MPSGRAPH`. A valid one-candidate
`MPSGRAPH` batch returns empty; it is not measured or reported as tuned. Invalid or non-Metal
projections retain their current failures. Engine separately enforces exactly one non-empty
Metal-owned partition before projection.

Enumeration, compatibility, identity, selection, encoding, and decoding delegate to the existing
version-twelve private generator/batch/decision/codec semantics. The public values add association
safety only. Decision decode snapshots and bounds caller bytes and returns empty for malformed,
trailing, corrupt, unsupported, stale, changed-workload, wrong-session, unknown-candidate, or
pruned-candidate input.

`trialPreparation(...)` and `selectedPreparation(...)` construct a fresh
`MetalNegAnalysisInputs` carrying the exact decision-present handoff and the integration's retained
`MetalTraceProducer`, then return a Metal-owned `PartitionPreparation`. Fresh preparer analysis
regenerates and authenticates all version-twelve facts before fixing the route and declaring
resources. There is no silent heuristic fallback after a supplied decision fails authentication.
The backend collaboration never binds representative inputs, executes, measures, selects a winner,
or accesses a cache path.

### Public retained Metal complete-plan collaboration

Add public final `io.github.pho001.synaptik.backend.metal.MetalCompletePlanTuning`, acquired only
through `MetalBackendIntegration.completePlanTuning()`. It retains the exact
`MetalLocalWorkloadTuning` collaboration and has the same integration lifetime and concurrency
rules.

It exposes the same opaque operation family as `CpuCompletePlanTuning`: exact handoff construction,
ordered candidates, compatibility, identity, selection, bounded decision encode/decode, fresh trial
preparation, and fresh selected preparation. Its handoff requires the exact current projected
context and exact authenticated `MetalLocalWorkloadTuning.SelectedDecision`. It freshly proves that
the Phase-1 batch remains the complete eligible two-route batch and that the selected decision
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

`candidateHandoff(...)` either returns the decision-empty exact singleton batch or fails when the
supplied Phase-1 value is stale, foreign, or no longer eligible; it does not turn an invalid
decision into an empty optional. The private Engine adapter wraps this successful handoff in its
existing optional return shape.

The resulting complete-plan batch contains exactly one opaque candidate: the fixed whole-model plan
whose sole partition uses the authenticated Phase-1 route. Its compatibility and decision encoding
include the complete versioned Phase-1 compatibility and selected decision, exact live integration
session, fixed-plan meaning, and projection facts needed to reject foreign or stale values. Trial
and selected preparation delegate through fresh local selected preparation; they do not enumerate,
rank, or substitute a local route.

Define distinct outer Metal fixed-plan identities rather than reusing CPU identity bytes:

- one Metal fixed-plan policy identity for exact canonical-byte correctness, one fixed owner/partition,
  and the already-selected local route;
- one Metal fixed-plan producer identity; and
- one Metal fixed-plan decision-codec identity.

Use explicit fixed numeric tags and an initial outer schema owned by this new concept. Do not use
class names, enum names, Java serialization, strings, reflection, or native addresses. Preserve the
existing CPU policy/producer/codec schema numbers and bytes exactly.

### Session-only persistence contract

Both Metal collaborations declare only `SESSION` reuse. Local compatibility continues to contain
the live `MetalDeviceContext` session nonce through the existing version-twelve codec. The fixed
complete-plan compatibility is also bound to the exact live integration and Phase-1 decision.

The current generic `WorkloadTuning` transaction still loads and validates the request's explicit
workload-cache path before applying per-workload reuse scope. Task 0042 does not change that tools
contract. A Metal `SESSION` workload can neither produce a persistent hit nor publish/update a Metal
entry. `CompletePlanTuning` performs no model-plan-cache filesystem operation for its `SESSION`
producer. Tests use absent temporary paths and prove neither path is created or changed.

Do not add a stable device/library fingerprint, promote either phase to `PERSISTENT`, change either
cache artifact, add a Metal cache file, or make Metal interpret model/profile identities. The
caller-defined model and representative-profile identities remain Engine evidence values. The
compile numerical profile remains inside existing Metal version-twelve compatibility.

Metal trace continues to report `NOT_QUERIED`: the backend authenticates an optional decision but
does not observe whether outer tools loaded a path or measured a miss. Every trial and final
preparation retains the exact integration trace producer; tuning must not silently select an
untraced construction path.

### Minimal private Engine generalization

Add a private `MetalTuningAdapter` beside `CpuTuningAdapter` and extend
`MetalEngineBackendComposition` with the exact local/complete collaboration accessors,
one-partition `projectedContext(...)`, and `GraphPreparation` composition needed for fresh trial and
selected preparations.

Dispatch the private tuning adapter from the already-selected owner adapter:

- `CpuEngineBackendComposition` retains the exact current CPU flow;
- `MetalEngineBackendComposition` uses the new bounded Metal flow; and
- empty, missing, mixed-owner, multiple-partition, or any other unsupported composition still fails
  before candidate work under its existing ownership/projection boundary.

Remove CPU-specific assumptions only from the shared private path: backend-neutral eligibility/error
wording, adapter dispatch, complete-plan policy identity supply, and ordinary fallback composition.
`ALLOW_SAFE_HEURISTIC` prepares through the exact already-selected adapter and can never switch
owner. Preserve cleanup, exception identity, suppression ordering, representative-session poisoning,
admission, fresh production preparation, and final publication behavior.

Make the private tuning seam supply its complete-plan policy identity. CPU continues to return the
exact existing policy bytes, and `CpuTuningAdapter` continues to construct the exact existing CPU
producer/codec identities. Metal supplies its distinct fixed-plan identities. The generic target
identity wrapper, correctness policy, budgets, evidence translation, and tools algorithms remain
unchanged.

`ModelAutotuningConfig`, `ModelAutotuningRequest`, `ModelAutotuningPreparation`, `Engine`, and their
public methods/records/enums remain source-contract compatible. A Metal `TUNED` result carries the
existing public evidence shape: one Phase-1 workload with two measured candidates and one measured
complete-plan candidate.

## Non-goals

- Mixed-owner tuning, multiple Metal partitions, multiple workload occurrences, deduplication across
  Engine-extracted occurrences, or owner/partition alternatives.
- Reporting an `MPSGRAPH`-only partition as tuned, measuring the heuristic against itself in Phase
  1, adding a route, or widening singleton-`NEG` eligibility.
- Optional Phase 2, public Config/result changes, a new top-level Engine operation, generic
  graph/plan tuning, Compiler graph alternatives, or Planning ownership search.
- Persistent/cross-session Metal decisions, stable device/library fingerprints, workload/model-plan
  cache format or lifecycle changes, cache inspection, executable persistence, or hidden global
  state.
- Benchmark tasks, baseline/report generation, performance matrices, device preprobes, broad Shape
  sweeps, or any performance claim beyond selecting the measured request-local winner.
- Native Objective-C/C source, ABI v4, thirteen exports, node schema 11, operation wires `1..19`,
  capability, availability, discovery, storage/transfer domains, route numerics, or Runtime hot-path
  changes.
- Changes to existing version-twelve workload, exact-policy, candidate, compatibility, route-policy,
  or codec meaning merely because an outer adapter now consumes them.
- CPU candidate semantics, CPU persistence scope, CPU complete-plan alternatives, CPU identity bytes,
  or CPU public behavior.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) keeps route
  selection cold and Runtime free of tuning.
- [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  assigns candidate meaning, compatibility, decision authentication, lowering, and resources to
  Metal.
- [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  assigns timing, winner selection, cache orchestration, and evidence to `tools/tuning` and Engine.
- [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)
  transports opaque backend values and composes fresh partition preparations.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  requires the authenticated route and all resources before assignment/finalization.
- [Compile ownership and mixed-owner Prepare routing](../../../../architecture/contracts/runtime-prepare-engine.md#compile-ownership-and-mixed-owner-prepare-routing)
  retains fail-closed selected-owner composition and excludes mixed-owner tuning.
- [ADR 0002](../../../../design/decisions/0002-backend-owned-lowering.md),
  [ADR 0008](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md),
  [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md),
  [ADR 0013](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md),
  and [ADR 0015](../../../../design/decisions/0015-explicit-engine-backend-composition.md) remain
  authoritative.

Stop and return to planning if implementation needs a new shared Prepare/tools API, a public
Config/result break, persistent Metal reuse, a native/capability/schema change, mixed-owner tuning,
or an alternate correctness/timing algorithm.

## Dependencies and integration

- Depends on: Metal 0004 and 0041 Complete; Prepare 0004/0008; tools/tuning 0001–0002/0004; Config
  0006A–0006B; CPU 0010I–0010J as public collaboration precedent; Engine 0006A–0009/0011/0015–0018;
  Runtime 0016.
- Conflicts with: every concurrent Metal integration/preparation/candidate/codec/trace/API scope;
  Engine tuning/composition/lifecycle/fallback scope; CPU tuning identity changes; tools/tuning,
  public API, backend guide, architecture, master-plan, or roadmap edits.
- Parallel group: None.
- Common base revision: `a961c0086c72b8457db1ea2b645810d60fed0a9d`.
- Integration order: one serial implementation from the exact base, focused validation and docs,
  exactly one full repository build, then independent Class C review.
- Shared-document integration owner: Metal 0042 executor.

## Files and symbols

Expected production surface:

- add `MetalLocalWorkloadTuning` and `MetalCompletePlanTuning` in the supported Metal root package;
- `MetalBackendIntegration`, `MetalBackendRuntime`, and Metal package Javadoc retain/expose them and
  propagate the exact context/trace producer;
- existing `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningDecision`,
  `MetalNegTuningCodec`, `MetalNegAnalysisInputs`, and `MetalNegPartitionPreparer` remain private and
  change only where the wrappers need association/fresh-authentication support;
- `MetalEngineBackendComposition`, `AdvancedEngine`, and `RepresentativeExecutionSession` implement
  the private Metal adapter, identity supply, dispatch, and backend-neutral fallback;
- no production dependency is added from Metal to Engine or `tools/tuning`.

Expected focused tests:

- add lean fake-native `MetalLocalWorkloadTuningTest` and `MetalCompletePlanTuningTest` coverage;
- update `ModelAutotuningCompositionTest` and `CompletePlanAutotuningCompositionTest` only for the
  backend-neutral private seam and CPU identity/behavior regression;
- add one focused public scenario to `EngineExplicitCompositionMetalIntegrationTest`.

Update implementation Javadocs plus the exact current CPU-only/Metal-foundation statements found in
`docs/backend-guide/metal-backend.md`, `docs/api/public-api.md`, the runtime/Prepare/Engine and
performance-tuning architecture explanations, `tools/tuning/package-info.java`, and planning status.
Do not rewrite unrelated historical evidence.

## Acceptance criteria

1. One open `MetalBackendIntegration` returns the same retained local and complete-plan
   collaborations. They are concurrency-safe, own no independent native resource, reject new work
   after integration close, and expose only opaque association-checked values and defensive bytes.
2. Fresh exact singleton-`NEG` analysis exposes precisely two local candidates in safe-heuristic
   order. Every valid `MPSGRAPH`-only partition returns no local tuning handoff. Caller budget does
   not prune the backend batch; the generic tool rejects an insufficient maximum of its own.
3. Local compatibility and decision handling exactly reuse version-twelve semantics and remain
   `SESSION`. Malformed, corrupt, stale, foreign-owner/session/projection, unknown, or cross-batch
   values fail closed. Trial and selected preparation freshly authenticate before exact custom
   versus MPSGraph declarations, with no heuristic substitution.
4. The complete-plan collaboration accepts only the exact authenticated local winner and exposes
   one fixed complete candidate. Its distinct outer Metal policy/producer/codec identities,
   compatibility, candidate identity, and decision round trip are deterministic, bounded,
   defensive, session-bound, and distinguishable from CPU. Fresh preparation keeps the Phase-1
   route fixed.
5. `WorkloadTuning` cannot hit or publish a persistent Metal entry; `CompletePlanTuning` performs no
   model-plan cache I/O. Existing generic cache loading/validation behavior and both artifact
   formats remain unchanged.
6. Engine dispatches the selected single-owner CPU or Metal adapter. Metal multiple-partition and
   mixed-owner plans remain unsupported for tuning. Allowed fallback uses exactly the selected
   owner's ordinary preparation and never switches owner; required mode preserves the tuning
   failure. Cleanup, poisoning, suppression, close races, and final publication remain correct.
7. CPU tuning produces byte-for-byte identical complete-plan policy, producer, and codec identities
   and retains its existing eligibility, candidates, persistence scope, fallback, evidence, and
   public behavior.
8. Existing public `ModelAutotuningConfig` and `ModelAutotuningPreparation` signatures and invariants
   are unchanged. Metal evidence uses the existing two-phase shape and truthfully reports two local
   measured rows plus one measured complete-plan row.
9. Every tuned trial and final preparation retains the integration's trace producer. Preparation
   and invocation events remain route-truthful and `NOT_QUERIED`; tuning adds no cache-status claim.
10. No native source/export/ABI/node schema/wire, capability, numerical profile, route semantics,
    transfer, Runtime hot path, benchmark, matrix, device preprobe, or existing Metal version-twelve
    identity changes.
11. Focused Metal, Engine, CPU-regression, architecture, Javadoc, documentation, and diff checks
    pass; the one real public Metal workflow passes; exactly one full repository build passes; and
    independent Class C review approves.

## Validation

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

The implementation owner runs the full repository build exactly once after all focused checks and
documentation are complete. Independent review inspects the exact diff and generated focused/full
reports; it does not run a performance matrix or preprobe a device. Review remediation reruns only
the checks affected by the finding unless production changes invalidate the single full-build
evidence.

The required public integration scenario is one bounded workflow, not a benchmark:

- use `Engine.builder()` with the ordinary configured Metal integration and no capability/device
  preprobe;
- compile one positive static canonical `FLOAT32` singleton `NEG` over one Metal partition;
- supply representative host input plus absent temporary workload/model-plan cache paths;
- use Phase-1 budget `maximumDistinctCacheMisses=1`, `maximumCandidatesPerMiss=2`, zero warmups, and
  one timed sample; use Phase-2 maximum one candidate, zero warmups, one timed sample, total
  execution ceiling two, and the exact small correctness-byte ceiling;
- assert `TUNED`, `SESSION`/`MEASURED` evidence, two local candidate rows, one fixed complete-plan
  row, and unchanged absent cache files;
- run the returned public `PreparedExecution` once and compare exact canonical NEG output bytes;
- close result, preparation, Engine, and integration-owned resources through the public lifecycle.

Do not add Shape sweeps, route timing assertions, winner assumptions, repeated-device discovery,
baselines, reports, or a standalone probe. Fake-native tests own deterministic route, association,
codec, trace, failure, and closure assertions. Existing CPU tests own regression proof rather than a
second integration matrix.

## Documentation and review impact

Document the two retained Metal collaborations, opaque values, exact eligibility, fixed Phase-2
meaning, session-only reuse, generic workload-cache read behavior, no Metal hit/publication,
model-plan no-I/O behavior, trace `NOT_QUERIED`, public Engine lifecycle, unchanged public request/
result shape, and all exclusions. Replace current statements that Metal has no Engine/tools adapter
or that `prepareTuned(...)` is universally CPU-only; retain CPU-only statements about
`Engine.standard()` and `AdvancedEngine.takeOwnership(CpuBackendIntegration)`.

Independent Class C review must cover public API minimality, owner/batch/projection association,
codec bounds and stale-session rejection, exact candidate completeness, fresh resource declaration,
singleton complete-plan truthfulness, cache scope, trace propagation, CPU identity byte stability,
Engine lifecycle/fallback failure precedence, actual public output, dependency direction, and docs.
