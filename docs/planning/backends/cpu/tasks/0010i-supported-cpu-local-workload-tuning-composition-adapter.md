# Task 0010I: Supported [CPU](../../../../glossary.md#term-cpu) Local-Workload Tuning Composition Adapter

## Reader context — historical integration boundary

The supported collaboration is a boundary for an outer tuning owner, subsequently Engine, to ask CPU
for valid alternatives and fresh preparations. `candidateHandoff(...)` enumerates; trial/selected
preparation constructs a recipe; neither call runs representative inputs or chooses a measured
winner. Engine binds those inputs, invokes Runtime and closes trial results, while tools/tuning owns
measurements and cache policy. A returned candidate or decision is tied to its owning
integration/batch and owns no storage. The historical artifact-consuming method names below precede
the current artifact-free `PrepareContext` collaboration; follow the current reference for the
maintained API rather than copying old signatures.

Current reading: [Local-workload tuning collaboration](../../../../backend-guide/cpu-backend.md#cold-local-workload-tuning-collaboration).

## Status

Complete

## Goal

Expose the smallest supported [CPU](../../../../glossary.md#term-cpu)-owned integration collaboration that lets an outer composition
[owner](../../../../glossary.md#term-owner) use the completed CPU 0010E candidate and selected-decision semantics with completed
tools/tuning 0001 without importing any `.internal` type:

```text
one supported CompileArtifacts value
  -> fresh CPU analysis
  -> optional exact-partition typed candidate handoff
  -> CPU-owned enumeration, compatibility, identity, decision, and codec operations
  -> one freshly prepared complete candidate recipe per cold trial
  -> one freshly prepared final selected recipe
```

The collaboration remains inside the [lifetime](../../../../glossary.md#term-lifetime) of one `CpuBackendIntegration`. It exposes [opaque](../../../../glossary.md#term-opaque),
typed CPU values in the existing supported root package, while the internal [OpenBLAS](../../../../glossary.md#term-openblas) workload,
candidate, route, representation, provider, workspace, and coordination types remain hidden.
Ordinary `CpuBackendIntegration.prepare(CompileArtifacts)` remains the deterministic safe-
[heuristic](../../../../glossary.md#term-heuristic) path and is unchanged when tuning is absent or abandoned.

## Motivation and current seam

Before this task, [CPU](../../../../glossary.md#term-cpu) 0010E already owned the complete exact/default [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) bare-[MATMUL](../../../../glossary.md#matrix-multiplication--matmul)
[candidate batch](../../../../glossary.md#candidate-batch),
[canonical](../../../../glossary.md#term-canonical-form) workload compatibility, portable-first candidate order, stable candidate [identity](../../../../glossary.md#term-identity-meanings),
selected-decision matching, safe [heuristic](../../../../glossary.md#term-heuristic), representation/resource consequences, and fresh-
analysis decision consumption. CPU 0010F already owns the supported one-[partition](../../../../glossary.md#partition) [lifecycle](../../../../glossary.md#lifecycle),
automatic provider discovery and [qualification](../../../../glossary.md#term-qualification), finalization, schedule assembly, [concurrency](../../../../glossary.md#term-concurrency)
budget, thread coordination, and close. CPU 0010H added source-only published [constants](../../../../glossary.md#term-constant) to the same
complete preparation path. None of those earlier supported APIs transported a tuning [handoff](../../../../glossary.md#term-handoff),
enumerated a candidate, encoded a decision, or prepared a recipe under one explicit candidate.

[Prepare](../../../../glossary.md#prepare) 0004 supplies only method-free [opaque](../../../../glossary.md#term-opaque) marker roles and an exact-partition typed handoff.
Tools/tuning 0001 deliberately requires a caller-supplied typed collaboration and complete cold
candidate execution action. It must not import CPU internals, and CPU must not depend on
tools/tuning. This task publishes that route-specific supported seam; [Engine](../../../../glossary.md#term-engine) 0007 remains blocked
on its separate representative-execution, input-[binding](../../../../glossary.md#term-binding), cleanup, and fallback contract.

The existing contracts are sufficient. `CpuPartitionAnalysisInputs` already accepts an optional
CPU decision, the selector validates it against a fresh complete [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), the preparation plan
retains the selected identity and exact resources, and finalization validates the live provider
qualification and installs the selected bounded thread count. No shared-contract or architecture
decision is required.

## [Scope](../../../../glossary.md#term-scope)

- Add one cohesive supported root-package collaboration,
  `io.github.pho001.synaptik.backend.cpu.CpuLocalWorkloadTuning`, with the exact public shape in
  Package and API design.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Add `CpuBackendIntegration.localWorkloadTuning()`, returning the same retained collaboration for
  the integration lifetime without transferring ownership.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Produce an optional
  `BackendPartitionTuningHandoff<CandidateBatch, SelectedDecision>` from a fresh analysis of one
  supported `CompileArtifacts` value. The [handoff](../../../../glossary.md#term-handoff) retains the exact sole `PlannedPartition`, an
  empty selected decision, and a public opaque batch associated with that exact integration,
  artifacts value, partition, and fresh internal batch.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [artifacts glossary](../../../../glossary.md#term-artifact); [partition glossary](../../../../glossary.md#partition))</span>
- Return an empty optional, not a synthetic portable-only batch, when the artifacts are otherwise
  supported but current analysis has no tunable 0010E batch. Continue to reject zero, empty,
  non-CPU, mixed-owner, or multi-partition artifacts exactly as the supported integration does.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner))</span>
- Expose typed public operations for complete ordered candidate enumeration, compatibility,
  candidate identity, selected-decision construction, decision encoding, and compatible decoding.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Expose one preparation operation for a selected trial candidate and one for a selected decision.
  Each operation repeats authoritative CPU analysis and complete Graph preparation; neither reuses
  the enumeration analysis as authority.<span data-glossary-navigation="true"> ([Graph preparation glossary](../../../../glossary.md#graph-preparation--graphpreparation))</span>
- Keep candidate, batch, and decision values immutable and opaque. They may retain package-private
  delegates and association tokens but expose no CPU-internal field or route vocabulary.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [tokens glossary](../../../../glossary.md#term-token))</span>
- Use bounded, deterministic, versioned CPU-owned binary encodings for compatibility, candidate
  identity, and decisions. Use fixed numeric tags and length prefixes, never Java serialization,
  enum names, class names, reflection, or string dispatch.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [serialization glossary](../../../../glossary.md#term-serialization); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Preserve session-only versus persistent reuse. A persistent compatibility/decision encoding is
  derived only from the 0010E persistent workload projection. A session-only compatibility value
  includes one immutable per-integration nonce and is usable only with that live collaboration;
  it is never promoted to persistent reuse. Cross-session decoding of a session-only decision
  returns empty.
- Preserve CPU provider, coordinator, concurrency-budget, and close ownership. The collaboration
  owns no independent provider and has no independent <span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency))</span>`close()` [operation](../../../../glossary.md#operation).
- Add focused native-free tests and the minimum existing real/native-coordination coverage needed
  to prove the supported surface, association checks, fresh analysis, selected preparation,
  fallback preservation, codec behavior, cleanup, and closure.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>
- Finalize affected Javadocs, supported CPU package documentation, focused CPU backend guide text,
  glossary impact, and planning evidence in the mandatory separate clean documentation context.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- measurement, warmup counts, timed samples, clocks, objectives, median calculation, comparison,
  winner selection, occurrence weighting, deduplication, budgets, or tuning evidence<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [median glossary](../../../../glossary.md#term-median); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- cache paths, file parsing, cache lookup, mutation, persistence, atomic publication, corruption
  recovery, model/profile fingerprints, or tools/tuning orchestration<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup); [persistence glossary](../../../../glossary.md#term-persistence); [atomic publication glossary](../../../../glossary.md#term-atomic-publication); [model glossary](../../../../glossary.md#term-model))</span>
- representative Tensor values, binding, trial run invocation, publication reads, result cleanup,
  model execution, strict-versus-fallback policy, or any Engine 0007 behavior<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [binding glossary](../../../../glossary.md#term-binding); [run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication))</span>
- a [CPU](../../../../glossary.md#term-cpu) [dependency](../../../../glossary.md#term-dependency) on tools/tuning or [Engine](../../../../glossary.md#term-engine), or an implementation of
  `BackendWorkloadTuning` inside the CPU [module](../../../../glossary.md#term-module)
- Engine representative-execution or fallback contracts, an Engine 0007 specification or
  implementation, Engine 0008, or tools/tuning 0002
- cross-route generalization from CPU 0016; the supported seam wraps only the current CPU 0010E
  exact/default FLOAT32/FLOAT64 bare-MATMUL candidate contract<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- new route eligibility, candidate pruning or synthesis, reordered candidates, relaxed numerics,
  BFLOAT16 OpenBLAS, vendor-peer routes 0011–0015, or Config 0004–0006<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- changing the method-free [Prepare](../../../../glossary.md#prepare) marker roles or [handoff](../../../../glossary.md#term-handoff), [Runtime](../../../../glossary.md#term-runtime) APIs, [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution),
  Compiler artifacts, Planning ownership, Backend Contract, or provider APIs<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [artifacts glossary](../../../../glossary.md#term-artifact); [Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Backend glossary](../../../../glossary.md#backend))</span>
- exposing internal workload signatures, candidates, decisions, plans, representations, provider
  handles, qualifications, workspaces, thread coordinators, native addresses, or storage objects<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- generated-kernel changes, new generated forms, schema changes unrelated to the CPU adapter
  codec, structural performance gates, or generated-versus-clean-Java performance claims<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec))</span>
- a [registry](../../../../glossary.md#term-registry), [service locator](../../../../glossary.md#term-service-locator), generic [parameter](../../../../glossary.md#term-parameter) bag, `Object` payload, raw type, unchecked cast,
  callback discovery, plugin mechanism, or broad cross-backend tuning facade<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [callback glossary](../../../../glossary.md#term-callback); [plugin glossary](../../../../glossary.md#term-plugin); [facade glossary](../../../../glossary.md#term-facade))</span>
- architecture-contract, ADR, module-dependency, Gradle/build, backend-conformance, or integration-
  test changes<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/prepare`, Concrete backend modules, Performance evidence and optimization tooling,
  Prepare lifecycle, Runtime service locator, and Dependency rules<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module); [Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle); [Runtime glossary](../../../../glossary.md#term-runtime); [service locator glossary](../../../../glossary.md#term-service-locator); [Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0010E candidate and decision contract](0010e-float32-float64-openblas-tuning-candidates-and-compatible-decisions.md)
- [CPU 0010F supported lifecycle adapter](0010f-supported-cpu-lifecycle-integration-adapter.md)
- [CPU 0010H source-only constant materialization](0010h-source-only-published-constant-cpu-materialization.md)<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Prepare 0004 opaque tuning handoff](../../../modules/prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- [Tools/tuning 0001 typed collaboration](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)
- [Config 0006A request facade](../../../modules/config/tasks/0006a-model-autotuning-request-configuration.md)<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- [Engine master plan](../../../modules/engine/master-plan.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>

## Architecture constraints

- CPU owns route-specific candidates, eligibility, compatibility meaning, canonical codec
  semantics, selected-decision construction and validation, lowering, resource declarations,
  finalization, provider/thread lifetime, and the safe heuristic.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [canonical glossary](../../../../glossary.md#term-canonical-form); [codec glossary](../../../../glossary.md#term-codec); [validation glossary](../../../../glossary.md#term-validation); [lowering glossary](../../../../glossary.md#lowering); [lifetime glossary](../../../../glossary.md#term-lifetime); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Prepare transports the typed opaque batch and decision only. Its marker roles remain method-free
  and its handoff retains the exact partition without interpreting CPU content.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [handoff glossary](../../../../glossary.md#term-handoff); [partition glossary](../../../../glossary.md#partition))</span>
- Tools/tuning owns invocation counts, timing, objective comparison, winner selection, cache load,
  mutation, persistence, and evidence. CPU exposes operations an outer owner can adapt to that
  tool; CPU neither implements the tool interface nor duplicates tool behavior.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence); [owner glossary](../../../../glossary.md#term-owner))</span>
- Engine later owns representative input binding, complete trial run and cleanup, Config mapping,
  fallback policy, model/profile facts, occurrence context and weights, and final lifecycle
  composition. This task neither designs nor implements those contracts.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [binding glossary](../../../../glossary.md#term-binding); [run glossary](../../../../glossary.md#term-run); [model glossary](../../../../glossary.md#term-model); [occurrence glossary](../../../../glossary.md#term-occurrence); [weights glossary](../../../../glossary.md#term-weight); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Runtime receives only a fully prepared recipe. It never receives a candidate, batch, decision,
  cache policy, or tuning request and never selects or measures one.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- A candidate or selected decision cannot create eligibility, relax semantics, or change numerical
  and determinism requirements. Trial and final preparation repeat fresh authoritative analysis
  and require an exact compatible selection.
- Ordinary `prepare(CompileArtifacts)` remains correct, deterministic, and safe-heuristic-driven.
  Adapter absence, an empty handoff, stale bytes, or rejected selected preparation cannot change
  or poison that path.
- The collaboration and all values are scoped to one `CpuBackendIntegration`. Public operations
  are safe for concurrent independent calls while it remains open. Close prevents new adapter
  operations, coordinates with the existing provider owner, and does not acquire ownership of
  caller artifacts or returned recipes. Recipes and runs must not outlive the integration.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Current production composition has capacity one and supplies only the qualified single-thread
  OpenBLAS configuration. Fresh finalization therefore preserves the existing coordinator's
  verified count-one contract. This task must not add multiple provider counts or cross-route
  coordination; either would return the task to planning or belong to CPU 0016.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [planning glossary](../../../../glossary.md#term-planning))</span>
- Prepared trial recipes are immutable and reusable under the same lifetime rules as ordinary
  recipes. The outer owner performs one complete cold run for each requested warmup or sample and
  owns every Runtime run/result cleanup. CPU does not count or time runs.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Generated CPU code continues to obey the clean-Java equivalence rule. This adapter changes no
  generated algorithm or kernel, so it adds no new performance proof.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>

## Package and [API](../../../../glossary.md#term-application-programming-interface-api) design

Existing supported package:

```text
io.github.pho001.synaptik.backend.cpu/
  CpuCapabilityProvider
  CpuBackendIntegration
  CpuLocalWorkloadTuning
  package-info.java
```

`CpuLocalWorkloadTuning` is one public final collaboration with nested public [immutable](../../../../glossary.md#term-immutable) value
types so the supported package gains one cohesive concept rather than a family of unrelated
top-level types. Its exact intended surface is:

```java
public final class CpuLocalWorkloadTuning {
    public Optional<BackendPartitionTuningHandoff<CandidateBatch, SelectedDecision>>
            candidateHandoff(CompileArtifacts artifacts);
    public List<Candidate> candidates(CandidateBatch batch);
    public Compatibility compatibility(CandidateBatch batch);
    public CandidateIdentity candidateIdentity(Candidate candidate);
    public SelectedDecision selectedDecision(CandidateBatch batch, Candidate candidate);
    public byte[] encodeDecision(SelectedDecision decision);
    public Optional<SelectedDecision> decodeCompatibleDecision(
            CandidateBatch batch, byte[] encodedDecision);
    public PreparedExecution prepareTrial(CandidateBatch batch, Candidate candidate);
    public PreparedExecution prepareSelected(
            CandidateBatch batch, SelectedDecision decision);

    public static final class CandidateBatch implements BackendTuningCandidateBatch { ... }
    public static final class SelectedDecision implements BackendTuningDecision { ... }
    public static final class Candidate { ... }
    public static final class Compatibility { ... }
    public static final class CandidateIdentity { ... }
    public enum ReuseScope { SESSION, PERSISTENT }
}
```

The collaboration constructor is not public. `CpuBackendIntegration.localWorkloadTuning()` is the
sole supported acquisition path and returns the same retained instance. Nested value constructors
are not public. Public value methods are limited to immutable value semantics and these exact
accessors:

- `Compatibility.schemaVersion()`, `Compatibility.bytes()`, and
  `Compatibility.reuseScope()`;
- `CandidateIdentity.bytes()`;
- ordinary `equals(Object)`, `hashCode()`, and redacted `toString()` where required for the
  tools/tuning adapter and round-trip verification.

Every returned byte array is a [defensive copy](../../../../glossary.md#term-defensive-copy). Diagnostic text reports only [schema](../../../../glossary.md#term-schema), reuse [scope](../../../../glossary.md#term-scope),
and byte lengths. [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), candidate, and decision expose no accessor beyond ordinary association-
safe diagnostic/value behavior needed by tests; in particular they expose no route, thread,
representation, workload, plan, provider, or [artifacts](../../../../glossary.md#term-artifact) accessor. Batch and candidate equality is
exact [owner](../../../../glossary.md#term-owner)/association [identity](../../../../glossary.md#term-identity-meanings). Decision equality combines that exact owner/current-batch
association with the internal decision's structural value so a compatible decode against the same
batch satisfies tools/tuning's mandatory [round-trip](../../../../glossary.md#schema-serialization-and-checkpoint) equality check without making decisions from
different integrations equal.

An outer [Engine](../../../../glossary.md#term-engine) composition can implement its own private
`BackendWorkloadTuning<CandidateBatch, SelectedDecision, Candidate>` by delegating the six matching
operations and translating `Compatibility` and `CandidateIdentity` into the corresponding
tools-owned immutable byte values. That outward adapter belongs to Engine 0007, not this task.

## Exact [lifecycle](../../../../glossary.md#lifecycle) and [dataflow](../../../../glossary.md#term-dataflow)

1. `CpuBackendIntegration.open()` creates one composition, one retained
   `CpuLocalWorkloadTuning`, and one immutable session-compatibility nonce. No host query, cache
   read, or measurement is added.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [host glossary](../../../../glossary.md#term-host))</span>
2. `candidateHandoff(artifacts)` checks that the integration is open, rejects a null argument,
   and applies the same exact sole non-empty CPU-partition validation as ordinary preparation.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [partition glossary](../../../../glossary.md#partition); [validation glossary](../../../../glossary.md#term-validation))</span>
3. It derives fresh authoritative analysis inputs with no selected decision and runs CPU analysis.
   If analysis produces no 0010E batch, it returns empty. It does not manufacture a portable-only
   tuning workload from an ineligible plan.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
4. For an eligible workload, it creates one association [token](../../../../glossary.md#term-token) retaining the exact integration,
   `CompileArtifacts`, sole `PlannedPartition`, internal batch, and ordered internal candidates.
   The public batch and each public candidate retain that token. The handoff carries the exact
   partition and an empty decision.
5. Enumeration returns the complete immutable public candidate list in CPU 0010E encounter order.
   Compatibility and identity operations canonically encode the batch workload projection and
   candidate identity. No operation consults a clock, cache, or filesystem.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [operation glossary](../../../../glossary.md#operation))</span>
6. `selectedDecision(batch, candidate)` verifies the collaboration owner and exact batch
   association, then creates one opaque public decision around the existing CPU 0010E decision.
   A candidate from another batch or integration is rejected before preparation.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [opaque glossary](../../../../glossary.md#term-opaque))</span>
7. `encodeDecision` emits a bounded versioned representation. Persistent scope contains only the
   stable 0010E persistent workload projection and selected candidate identity. Session scope also
   binds the current collaboration nonce. The encoding contains no live object, provider handle,
   path, class name, native address, or measurement.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
8. `decodeCompatibleDecision(batch, bytes)` snapshots and bounds the input before parsing. Invalid,
   truncated, trailing, unsupported-schema, wrong-session, stale-workload, unknown-candidate, or
   otherwise incompatible data returns empty without changing backend state. Programmer errors
   such as a null input still fail normally. Successful decoding reconstructs a decision
   associated with the supplied current batch.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [backend glossary](../../../../glossary.md#backend))</span>
9. `prepareTrial(batch, candidate)` constructs the exact selected decision, derives new analysis
   inputs from the retained artifacts, repeats lowering and candidate generation, and requires the
   fresh batch to accept the exact decision and retain the exact selected identity. It then runs
   ordinary Graph preparation, source-only constant contribution, assignment, finalization, and
   schedule assembly once and returns the resulting <span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [Graph preparation glossary](../../../../glossary.md#graph-preparation--graphpreparation); [constant glossary](../../../../glossary.md#term-constant))</span>[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution).
10. `prepareSelected(batch, decision)` follows the same fresh path. It rejects a decision from a
    different batch/integration and rejects an ineligible, stale, or changed fresh batch rather
    than silently selecting the heuristic. Compatible cache bytes must first be decoded against
    the current batch, which produces a current-associated decision.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
11. The outer owner binds equivalent representative inputs, invokes Runtime once for each complete
    cold trial requested by tools/tuning, reads or discards publications as its later contract
    requires, and closes every run/result resource. None of that behavior enters CPU.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run))</span>
12. The final selected decision is prepared through step 10. If later Engine policy permits
    fallback after tuning failure, Engine separately invokes ordinary<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
    `CpuBackendIntegration.prepare(artifacts)`; [CPU](../../../../glossary.md#term-cpu) never chooses that policy through this adapter.
13. Closing `CpuBackendIntegration` atomically prevents new handoff, codec, and preparation work,
    then uses the existing coordinator close/quiescence/restoration behavior. Racing operations
    have the same admitted-before-close contract as existing CPU preparation. Returned recipes
    must not outlive the integration.<span data-glossary-navigation="true"> ([quiescence glossary](../../../../glossary.md#term-quiescence))</span>

Any failure during fresh analysis, assignment, [artifact](../../../../glossary.md#term-artifact) realization, provider configuration, or
schedule assembly propagates unchanged after existing local cleanup. No partial [handoff](../../../../glossary.md#term-handoff), decision,
or recipe is published. [Codec](../../../../glossary.md#term-codec) rejection has no [side effect](../../../../glossary.md#term-side-effect). CPU never deletes or mutates [cache](../../../../glossary.md#term-cache)
state because it never receives a cache path.

## Affected files

Expected production and test paths:

- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuLocalWorkloadTuning.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuBackendIntegration.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendComposition.java`
- add at most one package-private CPU-internal codec/helper under the existing OpenBLAS package if
  keeping canonical serialization out of the supported wrapper materially improves cohesion<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningBatch.java` only if a package-local [canonical](../../../../glossary.md#term-canonical-form) projection helper is required
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningDecision.java` only if a package-local canonical [codec](../../../../glossary.md#term-codec) helper is required
- add `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/spi/CpuLocalWorkloadTuningPublicTest.java`
- add at most one focused internal codec/coordination test under the existing mirrored OpenBLAS
  package
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendCompositionTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md` only if the supported CPU collaboration creates a reusable term not already
  covered by candidate batch, selected tuning decision, or tuning orchestration<span data-glossary-navigation="true"> ([candidate batch glossary](../../../../glossary.md#candidate-batch); [selected tuning decision glossary](../../../../glossary.md#selected-tuning-decision))</span>
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/modules/engine/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a concrete contradiction is found: `ARCHITECTURE.md`, focused
architecture explanations, ADRs, [Prepare](../../../../glossary.md#prepare) 0004 source/tests/docs, tools/tuning 0001 source/tests/docs,
Config 0006A, [Engine](../../../../glossary.md#term-engine) source/tests, [Compiler](../../../../glossary.md#term-compiler)/Planning/[Runtime](../../../../glossary.md#term-runtime)/[Backend](../../../../glossary.md#backend) Contract, [Gradle](../../../../glossary.md#term-gradle) files,
architecture tests, backend-[conformance](../../../../glossary.md#term-conformance) tests, integration tests, generated-[kernel](../../../../glossary.md#kernel) source/tests,
and optional [CPU](../../../../glossary.md#term-cpu) 0011–0017 plans.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 17 paths:

- 7 CPU production/Javadoc paths: the three required supported/composition paths plus no more than
  four narrowly justified internal codec/delegate paths;<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- 4 focused [CPU](../../../../glossary.md#term-cpu) test paths;
- 2 explanatory documentation paths; and
- 3 [planning](../../../../glossary.md#term-planning) paths in addition to this task.

The preferred implementation uses one supported top-level type and one package-private internal
[codec](../../../../glossary.md#term-codec)/helper at most. Do not add a [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), build path, shared-contract path,
architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) path, [conformance](../../../../glossary.md#term-conformance)/integration path, generated-[kernel](../../../../glossary.md#kernel) path, [Engine](../../../../glossary.md#term-engine) Java path, or
tools/tuning Java path. If fresh selected preparation cannot be expressed within this ceiling and
the existing composition, analysis-input, preparation, and finalization contracts, stop and
return the task to planning.

## Acceptance criteria

1. `CpuBackendIntegration.localWorkloadTuning()` returns one retained non-null supported
   collaboration while open; no public constructor, registry, discovery hook, or independent
   close surface exists.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>
2. The supported CPU package adds exactly one top-level tuning type. Its nested batch and decision
   implement the Prepare marker roles, and all collaboration signatures are fully typed without
   raw types, unchecked casts, arbitrary <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Prepare glossary](../../../../glossary.md#prepare); [signatures glossary](../../../../glossary.md#term-signature))</span>`Object` payloads, reflection, string [dispatch](../../../../glossary.md#term-dispatch), or maps.
3. A supported one-partition artifact receives a handoff only when fresh CPU analysis produces the
   complete current 0010E candidate batch. The handoff retains the exact partition and has no
   selected decision. A valid non-tunable artifact returns empty; zero/empty/non-CPU/mixed/multi-
   partition artifacts fail before candidate exposure.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [artifact glossary](../../../../glossary.md#term-artifact); [handoff glossary](../../../../glossary.md#term-handoff); [candidate batch glossary](../../../../glossary.md#candidate-batch))</span>
4. Candidate enumeration is immutable, complete, stable, portable-first, and exactly ordered like
   the internal batch. No candidate is invented, pruned, reordered, or exposed structurally.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
5. Compatibility and candidate identities are non-empty, canonical, defensively copied, bounded,
   and typed. Persistent scope exists only with a stable qualified binary projection; otherwise
   compatibility is session-only and bound to the current integration.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [scope glossary](../../../../glossary.md#term-scope); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
6. Selected-decision construction accepts only a candidate belonging to the exact supplied batch
   and collaboration. Cross-batch and cross-integration combinations fail before any analysis,
   finalization, provider mutation, or artifact realization.
7. Decision encoding is deterministic and versioned, contains no live/internal object, and uses
   fixed binary tags. Compatible decode rejects corruption, unsupported versions, trailing data,
   stale workloads, unknown candidates, and wrong session by returning empty. A successful
   persistent or same-session round trip returns a decision equal to the original public value.<span data-glossary-navigation="true"> ([round trip glossary](../../../../glossary.md#term-round-trip))</span>
8. Trial preparation selects exactly the requested candidate, repeats authoritative analysis,
   validates fresh compatibility and membership, contributes current source-only constants, and
   returns one complete <span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution). It never runs, times, compares, or caches it.
9. Final selected preparation repeats the same authoritative path. An eligibility change,
   incompatible/stale decision, or changed candidate set is rejected and never silently converted
   to heuristic success.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
10. Ordinary `prepare(CompileArtifacts)` retains its existing safe heuristic and exact observable
    behavior when tuning is absent or disabled. Failure or rejection through the tuning adapter
    does not mutate later ordinary preparation.
11. Provider qualification, count-one configuration, shared budget admission, prepared execution,
    source-only initializer, and coordinator close/restoration behavior remain owned by the
    existing composition. Portable-only operation requires no provider; native selected
    preparation uses only the exact live qualified coordinator.<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification); [admission glossary](../../../../glossary.md#term-admission); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [operation glossary](../../../../glossary.md#operation))</span>
12. Independent adapter calls are thread-safe while open. Close prevents new work, racing admitted
    work follows the existing lifecycle contract, repeated close is inert, and all public methods
    reject use after close. Artifacts, trial recipes, Runtime runs, results, and input storage are
    never closed by the collaboration.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [lifecycle glossary](../../../../glossary.md#lifecycle); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
13. CPU performs no measurement, cache I/O, model/profile orchestration, representative-input
    binding, Runtime selection, fallback-policy choice, or Engine work. CPU has no tools/tuning or
    Engine dependency.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [model glossary](../../../../glossary.md#term-model); [binding glossary](../../../../glossary.md#term-binding); [Engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency))</span>
14. CPU 0016 remains Draft and unchanged as later cross-route generalization; CPU 0011 remains
    Blocked and CPU 0012–0015/0017 retain their current optional statuses and ordering.
15. No generated kernel, generated-artifact identity, lowering semantic, numerical rule,
    performance threshold, architecture contract, ADR, Gradle file, conformance test, or
    integration test changes.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [identity glossary](../../../../glossary.md#term-identity-meanings); [lowering glossary](../../../../glossary.md#lowering); [architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test))</span>
16. A separate clean documentation-focused context independently reviews the implementation and
    tests, finalizes every affected Javadoc and supported-package explanation, updates the focused
    CPU guide/glossary only where needed, and records reasoned no-change conclusions for shared
    APIs, architecture, Config, Engine API, and Runtime documentation. It does not repeat a
    successful Java suite unless it changes executable Java or identifies stale evidence.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
17. Focused and full CPU tests, CPU Javadoc, public-shape/import/javap checks, codec malformed-input
    checks, changed-Markdown links/anchors/headings/fences/newlines, exact scope, and<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>
    `git diff --check` pass with recorded evidence.

## [Validation](../../../../glossary.md#term-validation)

Validation tier: focused [CPU](../../../../glossary.md#term-cpu) [module](../../../../glossary.md#term-module) validation plus full CPU module validation. Repository-wide
validation is not required because the task adds no [dependency](../../../../glossary.md#term-dependency), build, shared-contract,
architecture, or multi-module executable change.

Implementation context:

```bash
./gradlew :backends:cpu:test --tests '*CpuLocalWorkloadTuningPublicTest' \
  --tests '*CpuBackendCompositionTest' --tests '*CpuOpenBlas*Tuning*Test'
./gradlew :backends:cpu:test
./gradlew :backends:cpu:javadoc
git diff --check
```

Also inspect the compiled public surface with `javap`; verify imports and [bytecode](../../../../glossary.md#term-bytecode)/source contain
no tools/tuning, [Engine](../../../../glossary.md#term-engine), `.internal` type in a public [signature](../../../../glossary.md#term-signature), raw/unchecked bridge, reflection,
Java [serialization](../../../../glossary.md#term-serialization), [cache](../../../../glossary.md#term-cache)/path/I/O, timing, or string-[dispatch](../../../../glossary.md#term-dispatch) mechanism. Exercise malformed and
bounded [codec](../../../../glossary.md#term-codec) inputs, persistent and session-only round trips, wrong-session rejection, exact
association rejection, fresh-analysis stale rejection, ordinary [heuristic](../../../../glossary.md#term-heuristic) preservation,
source-only [constants](../../../../glossary.md#term-constant), provider/no-provider cases, [concurrency](../../../../glossary.md#term-concurrency), close races, and cleanup.

Documentation context:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Inspect rendered [Javadoc](../../../../glossary.md#term-javadoc) for every new or changed public/nested type and method. Check all changed
Markdown links and anchors, unique headings, balanced fences, LF endings, final newlines, exact
path count, and synchronized statuses. Reuse the implementation context's successful CPU tests
unless documentation changes executable Java.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0010E: Complete
- CPU 0010F: Complete
- CPU 0010H: Complete
- [Prepare](../../../../glossary.md#prepare) 0004: Complete
- tools/tuning 0001: Complete consumer contract; CPU does not depend on its [module](../../../../glossary.md#term-module)
- Config 0006A: Complete boundary evidence only; not an implementation [dependency](../../../../glossary.md#term-dependency)
- [Compiler](../../../../glossary.md#term-compiler) 0006B3 and Prepare 0003/0005: Complete [lifecycle](../../../../glossary.md#lifecycle) and source-only [handoff](../../../../glossary.md#term-handoff)
- [Runtime](../../../../glossary.md#term-runtime) 0010 and closure hardening: Complete recipe/[run](../../../../glossary.md#term-run) lifecycle
- [Engine](../../../../glossary.md#term-engine) 0007: remains Blocked and is not a dependency

## Documentation-focused pass

After implementation and focused/full [CPU](../../../../glossary.md#term-cpu) [validation](../../../../glossary.md#term-validation), a distinct clean documentation context
must read the final diff, this task, CPU 0010E/0010F/0010H, [Prepare](../../../../glossary.md#prepare) 0004, tuning 0001, Config
0006A, the documentation rules and General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc) profiles, the supported CPU package, and
the focused CPU guide/glossary entries. It must:

- finalize meaningful Javadocs for lifetime, association, byte snapshots, nullability, return
  semantics, rejection, failure, ownership, concurrency, and close behavior;<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [nullability glossary](../../../../glossary.md#term-nullability); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- explain that the supported CPU collaboration is an outer-composition building block, not a
  tuning runner or application facade;<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- preserve the CPU/tools/[Engine](../../../../glossary.md#term-engine)/[Runtime](../../../../glossary.md#term-runtime) [ownership](../../../../glossary.md#term-ownership-meanings) split and safe-[heuristic](../../../../glossary.md#term-heuristic) path;
- update the glossary only if existing terms cannot describe the new supported seam;
- record explicit no-change conclusions for architecture/ADRs, Prepare and tuning APIs, Config,
  Engine API, Runtime, builds, conformance/integration docs, and generated-code guidance; and<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- synchronize this task, CPU master plan, Engine master plan, and roadmap completion evidence only
  after implementation is actually complete.

## Follow-up

- Reassess Engine 0007 in a new clean planning context only after CPU 0010I is Complete and the
  separate Engine representative-execution/fallback contract is explicitly selected. That Engine
  task may privately adapt this collaboration to tools/tuning 0001 and map Config 0006A.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Do not create the representative-execution/fallback task in this change.
- Keep tools/tuning 0002 as later [graph](../../../../glossary.md#graph)/plan tuning.
- Keep CPU 0016 as later cross-route generalization after concrete vendor peer routes exist; it
  must reuse rather than duplicate this supported composition shape where applicable.

## Architecture impact

No architecture-contract, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [module](../../../../glossary.md#term-module)-boundary, or [dependency](../../../../glossary.md#term-dependency) change is intended. The task adds a
narrow supported [CPU](../../../../glossary.md#term-cpu) integration [SPI](../../../../glossary.md#term-service-provider-interface-spi) in the already supported root package and delegates to
existing CPU-owned analysis/finalization behavior. If implementation needs a shared marker method,
a tools/tuning or [Engine](../../../../glossary.md#term-engine) dependency, a [Runtime](../../../../glossary.md#term-runtime) candidate [API](../../../../glossary.md#term-application-programming-interface-api), multi-provider coordination, or a
new [ownership](../../../../glossary.md#term-ownership-meanings) rule, stop and return the task to [planning](../../../../glossary.md#term-planning).

## Local decisions and known limits

- The public collaboration is CPU-owned and tools-shaped, but it does not implement a tools-owned
  interface because that would reverse the dependency boundary.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Valid CPU artifacts with no 0010E batch return empty. A portable plan is not automatically a
  tunable batch; CPU 0010E intentionally creates a batch only when at least one eligible OpenBLAS
  peer exists.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Public batch/candidate/decision values are exact association handles, not portable domain DTOs.
  Only compatibility, identity, and encoded decision bytes cross invocation or persistence
  boundaries.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Persistent reuse remains available only for the existing qualified persistent-binary scope.
  Session-only values may deduplicate inside the current tuning invocation/integration but cannot
  become a cross-session cache hit.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scope glossary](../../../../glossary.md#term-scope); [cache hit glossary](../../../../glossary.md#cache-and-memoization))</span>
- Trial and final preparation use the current capacity-one/single-thread production composition.
  Multiple provider thread counts, overlapping differently configured prepared recipes, and
  vendor-peer coordination are deliberately not generalized here.
- [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) ownership and Runtime cleanup do not change. The collaboration creates a
  recipe; it does not own or close a later run/result.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run))</span>
- Representative inputs and failure fallback remain the separate unresolved Engine prerequisite.
  Their absence does not prevent CPU from exposing this bounded preparation seam.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>

## Implementation prompt

Work in `/Users/phujka/IdeaProjects/Synaptik`. Read `AGENTS.md`, `ARCHITECTURE.md`, the current
architecture plan, [planning](../../../../glossary.md#term-planning) guide/roadmap, documentation rules and General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc) profiles,
[CPU](../../../../glossary.md#term-cpu) master plan and tasks 0010E/0010F/0010H plus relevant 0010C/0010D, [Prepare](../../../../glossary.md#prepare) 0004 and its source
and tests, tools/tuning 0001 public contracts/tests, Config 0006A source, [Engine](../../../../glossary.md#term-engine) master plan and
current composition sources/tests, and every CPU source/test touched by this specification.

Implement task 0010I exactly within its allowlist and maximum [scope](../../../../glossary.md#term-scope). Add the single supported
`CpuLocalWorkloadTuning` top-level type and the retained acquisition method on
`CpuBackendIntegration`. Keep all nested values [opaque](../../../../glossary.md#term-opaque) and association-safe. Generate handoffs
only from fresh eligible 0010E analysis, expose the exact ordered candidates and [canonical](../../../../glossary.md#term-canonical-form) typed
compatibility/[identity](../../../../glossary.md#term-identity-meanings)/decision operations, implement bounded versioned encoding/compatible
decoding, and prepare trial/final selected recipes only after fresh authoritative analysis accepts
the exact choice. Preserve ordinary [heuristic](../../../../glossary.md#term-heuristic) preparation, source-only [constants](../../../../glossary.md#term-constant), current
provider/thread coordination, closure, and every [ownership](../../../../glossary.md#term-ownership-meanings) boundary. Add no measurement, [cache](../../../../glossary.md#term-cache)
I/O, Engine orchestration, [Runtime](../../../../glossary.md#term-runtime) selection, [dependency](../../../../glossary.md#term-dependency), generated-[kernel](../../../../glossary.md#kernel) behavior, or cross-
route generalization.

[Run](../../../../glossary.md#term-run) focused and full CPU tests, CPU Javadoc, public-surface/import/[bytecode](../../../../glossary.md#term-bytecode)/manual [codec](../../../../glossary.md#term-codec) checks,
Markdown/scope/status [validation](../../../../glossary.md#term-validation), and `git diff --check`. Then use the mandatory separate clean
documentation-focused context to finalize affected Javadocs and focused documentation without
repeating successful Java tests unless executable Java changes. Do not [commit](../../../../glossary.md#term-commit) or push. If the
existing contracts cannot support the exact fresh-analysis and [lifetime](../../../../glossary.md#term-lifetime) rules inside the scope
ceiling, stop and report the missing prerequisite instead of [widening](../../../../glossary.md#term-widening) the architecture.

## Completion summary

Implementation context `01a0b149-462c-7cc2-a326-1b34f5b518f3` completed the bounded supported
adapter. `CpuBackendIntegration` now retains one `CpuLocalWorkloadTuning`; the new collaboration
exposes exact eligible handoffs, portable-first [opaque](../../../../glossary.md#term-opaque) candidates, typed compatibility and
candidate [identities](../../../../glossary.md#term-identity-meanings), association-safe selected decisions, bounded versioned persistent/session
codecs, and fresh authoritative trial/final selected preparation. Ordinary
`prepare(CompileArtifacts)` retains its safe [heuristic](../../../../glossary.md#term-heuristic). No measurement, representative [binding](../../../../glossary.md#term-binding),
execution, winner selection, [cache](../../../../glossary.md#term-cache)/filesystem behavior, [Engine](../../../../glossary.md#term-engine)/tools-tuning [dependency](../../../../glossary.md#term-dependency),
reflection/[serialization](../../../../glossary.md#term-serialization), generated-code change, or architecture/dependency change was added.

Actual implementation-owned paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuBackendIntegration.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuLocalWorkloadTuning.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendComposition.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderPublicShapeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendCompositionTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/spi/CpuBackendIntegrationAndCpuPreparedScheduleAssemblerPublicTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/spi/CpuLocalWorkloadTuningPublicTest.java`

Actual documentation-owned paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0010i-supported-cpu-local-workload-tuning-composition-adapter.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/modules/engine/master-plan.md`
- `docs/planning/roadmap.md`

Documentation context `01a0b164-e6e6-78f2-9e15-c3c4fef144bf` finalized those paths. The glossary
gained no new term because its existing [candidate generator](../../../../glossary.md#candidate-generator), [selected tuning decision](../../../../glossary.md#selected-tuning-decision), [opaque
backend tuning handoff](../../../../glossary.md#opaque-backend-tuning-handoff), and [model](../../../../glossary.md#term-model)-[autotuning](../../../../glossary.md#autotuning--model-autotuning) definitions already cover the supported seam; only
their stale pre-0010I status text was corrected.

Implementation [validation](../../../../glossary.md#term-validation) passed 20/20 focused tests and the full [CPU](../../../../glossary.md#term-cpu) suite passed 1,016 tests
with 28 existing conditional skips and zero failures or errors. CPU [Javadoc](../../../../glossary.md#term-javadoc) passed with 94
pre-existing warnings in unchanged [lowering](../../../../glossary.md#lowering)/preparation-plan declarations and no warning from
0010I. Public-source, `javap`, and [bytecode](../../../../glossary.md#term-bytecode) checks found no `.internal` type in a supported
[signature](../../../../glossary.md#term-signature) and no forbidden dependency or behavior. The documentation pass preserved every dirty
Java/test path byte-for-byte, inspected the rendered supported page to confirm that the hidden
package-private construction seam does not expose `CpuBackendComposition`, and passed the focused
Markdown, status, exact-[scope](../../../../glossary.md#term-scope), and whitespace checks recorded for this task.

`ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, [Prepare](../../../../glossary.md#prepare) and
tools/tuning public contracts, Config, Engine and [Runtime](../../../../glossary.md#term-runtime) APIs, other guides, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance),
integration tests, generated-code guidance, and [Gradle](../../../../glossary.md#term-gradle) files require no change: [ownership](../../../../glossary.md#term-ownership-meanings),
[dependency direction](../../../../glossary.md#java-platform-and-build-tools), executable semantics, and cross-[module](../../../../glossary.md#term-module) contracts are unchanged. CPU 0010I
closes only the CPU prerequisite. Engine 0007 remains Blocked until a separate Engine-owned
representative-execution/input-binding/cleanup and fallback contract is specified and completed;
Engine 0008 remains Draft without a task specification.

```text
Status: Complete
```
