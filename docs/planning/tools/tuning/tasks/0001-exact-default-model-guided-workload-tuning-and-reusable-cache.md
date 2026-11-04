# Task 0001: Exact/Default [Model](../../../../glossary.md#term-model)-Guided Workload Tuning and Reusable [Cache](../../../../glossary.md#term-cache)

## Editorial reading key: the caller can be framework composition

The caller here is code that already connects a concrete backend to an execution lifecycle.
It calls `WorkloadTuning.tune` with the request, a backend-aware collaborator and an execution
action. Tuning calls that collaborator to enumerate and authenticate candidates, then calls the
action to perform complete trials. “Opaque” means tuning can pass the typed candidate onward
without interpreting backend-private settings. The composition still arranges comparable input
values and execution resources; a selected decision is not an executed result.

Request-owned byte sequences and collections are snapshots, while exact partition/batch
references remain references. This is not a blanket copy or cleanup transfer.
Read [local workload tuning](../../../../developer-guide/benchmarking.md#phase-1-local-workload-transaction)
and [candidate responsibility](../../../../architecture/performance-evidence-and-tuning.md#candidate-ownership-and-opaque-orchestration).
The original “no Engine facade” and future-integration statements below describe this earlier
task's frontier, not today's callable surface.


## Status

Complete

## Goal

Implement the first operational local-workload phase of [model autotuning](../../../../glossary.md#autotuning--model-autotuning) for the completed [CPU](../../../../glossary.md#term-cpu)
0010E and [Prepare](../../../../glossary.md#prepare) 0004 boundary without giving shared tooling knowledge of CPU candidate fields.
The workflow accepts complete [backend](../../../../glossary.md#backend)-owned candidate [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) through Prepare's method-free
[opaque](../../../../glossary.md#term-opaque) roles, reuses compatible results, measures [cache](../../../../glossary.md#term-cache) misses only, selects a deterministic
winner under one explicit elapsed-time [objective](../../../../glossary.md#term-objective) and bounded [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) policy, returns selected
backend-owned decisions, atomically persists a compact workload cache, and returns separate rich
in-memory evidence.

The repository has no [Engine](../../../../glossary.md#term-engine) [facade](../../../../glossary.md#term-facade) or general public [model](../../../../glossary.md#term-model)-execution path. This task therefore
defines the smallest operational caller-supplied cold collaboration. The caller that already
composes a [concrete backend](../../../../glossary.md#concrete-backend) supplies:

- stable model and representative-profile fingerprints;
- occurrences and their exact `BackendPartitionTuningHandoff<C, D>` values;
- candidate enumeration from each complete opaque batch;
- [canonical](../../../../glossary.md#term-canonical-form) local and persistent compatibility [identities](../../../../glossary.md#term-identity-meanings);
- stable candidate identities;
- selected-decision construction and decision encoding/compatible decoding; and
- one complete comparable execution action for each candidate.

`tools/tuning` supplies:

- [immutable](../../../../glossary.md#term-immutable) tool-local request [validation](../../../../glossary.md#term-validation);
- identical-workload deduplication with [occurrence](../../../../glossary.md#term-occurrence) [weights](../../../../glossary.md#term-weight) and contexts retained;
- cache loading before any measurement;
- miss-only [warmup](../../../../glossary.md#term-warmup), monotonic timing, [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) validation, and bounded measurement;
- minimum-[median](../../../../glossary.md#term-median) elapsed-time comparison with candidate encounter order as the exact tie break;
- compatible decision reuse and fresh decision selection;
- compact bounded cache encoding, corruption/incompatibility rejection, and atomic persistence;
  and<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- separate rich evidence containing raw miss samples and reused-entry summaries.

The collaboration is generic and typed:

```java
public interface BackendWorkloadTuning<
        C extends BackendTuningCandidateBatch,
        D extends BackendTuningDecision,
        K> {
    List<K> candidates(C candidateBatch);
    WorkloadTuningRequest.WorkloadCompatibility compatibility(C candidateBatch);
    WorkloadTuningRequest.CandidateIdentity candidateIdentity(K candidate);
    D selectedDecision(C candidateBatch, K candidate);
    byte[] encodeDecision(D decision);
    Optional<D> decodeCompatibleDecision(C candidateBatch, byte[] encodedDecision);
}

@FunctionalInterface
public interface ColdCandidateMeasurement<
        C extends BackendTuningCandidateBatch,
        K> {
    void execute(C candidateBatch, K candidate) throws Exception;
}

public final class WorkloadTuning {
    public static <C extends BackendTuningCandidateBatch,
            D extends BackendTuningDecision, K>
            WorkloadTuningResult<C, D> tune(
                    WorkloadTuningRequest<C, D> request,
                    BackendWorkloadTuning<C, D, K> backend,
                    ColdCandidateMeasurement<C, K> measurement) throws Exception;
}
```

The single checked failure boundary preserves cache I/O and caller execution failures without
wrapping or swallowing either. Implementation must not change the six collaboration operations,
their [ownership](../../../../glossary.md#term-ownership-meanings), or their typed generic association.

## [Scope](../../../../glossary.md#term-scope)

- Add the public `WorkloadTuning` stateless entry, `WorkloadTuningRequest`,
  `BackendWorkloadTuning`, `ColdCandidateMeasurement`, and `WorkloadTuningResult` under
  `io.github.pho001.synaptik.tools.tuning`.
- Delete the existing public placeholder `TuningToolModule` so the completed tuning API exposes
  exactly those five public production types.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Make `tools/tuning` expose its [Prepare](../../../../glossary.md#prepare) [dependency](../../../../glossary.md#term-dependency) as [Gradle](../../../../glossary.md#term-gradle) `api` because its public generic
  contracts use `BackendTuningCandidateBatch`, `BackendTuningDecision`, and
  `BackendPartitionTuningHandoff`. Do not add a concrete-[backend](../../../../glossary.md#backend), [Engine](../../../../glossary.md#term-engine), or [Runtime](../../../../glossary.md#term-runtime) dependency.
- Keep all request inputs tool-local. `WorkloadTuningRequest<C, D>` contains exactly:
  - a non-empty stable `ModelFingerprint` byte sequence with a positive caller [schema](../../../../glossary.md#term-schema) version;
  - a non-empty stable `ProfileFingerprint` byte sequence with a positive caller schema version;
  - an ordered non-empty [immutable](../../../../glossary.md#term-immutable) list of `Occurrence<C, D>` values;
  - [objective](../../../../glossary.md#term-objective) `MIN_MEDIAN_ELAPSED_NANOS`;
  - one `Budget`; and
  - one explicit workload-[cache](../../../../glossary.md#term-cache) `Path`.
- Define `Occurrence<C, D>` as one exact handoff, one positive occurrence weight, and one
  non-empty opaque context fingerprint. Require the input handoff's selected decision to be empty;
  output handoffs carry the selected backend decision. Retain exact partition and batch
  references and snapshot only request-owned collections and byte sequences.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [occurrence glossary](../../../../glossary.md#term-occurrence); [weight glossary](../../../../glossary.md#term-weight); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [carry glossary](../../../../glossary.md#term-carry); [partition glossary](../../../../glossary.md#partition); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Define `Budget` with positive maximum distinct cache misses, positive maximum candidates per
  miss, non-negative warmup count, and a positive odd timed-sample count. The task has no wall-
  clock cutoff: it validates the complete deduplicated work and every complete candidate count
  against the budget before executing the first candidate, so machine-speed variation cannot
  change which candidates are compared.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Define `WorkloadCompatibility` as a positive backend-owned key-schema version, a non-empty
  opaque canonical byte sequence, and <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>`SESSION` or `PERSISTENT` reuse scope. The tool compares
  exact values only. It neither parses nor constructs backend compatibility content.
- Define `CandidateIdentity` as a non-empty opaque canonical byte sequence. Require identities to
  be unique within the complete ordered candidate list. The tool compares exact values only and
  uses encounter order, not lexical identity, to break an equal-median tie.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [median glossary](../../../../glossary.md#term-median))</span>
- Deduplicate occurrences by exact `WorkloadCompatibility`, objective, and timed sampling policy.
  Sum weights with checked arithmetic and retain every original occurrence/context in request
  order. Session identities may reuse a result only within the current invocation. Persistent
  identities may additionally reuse compatible file entries across invocations and models.
- Load and validate the entire explicit cache before candidate enumeration or measurement.
  Reject malformed, oversized, truncated, checksum-invalid, duplicate-key, unsupported-artifact-
  schema, or structurally inconsistent files with an <span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum); [artifact glossary](../../../../glossary.md#term-artifact))</span>`IOException`; preserve the original file
  and perform no measurement or write after rejection.
- Treat a well-formed entry with no exact key/objective/[sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) match or whose [opaque](../../../../glossary.md#term-opaque) decision
  `decodeCompatibleDecision` returns empty as an ordinary cache miss. Never ask the tool to
  interpret why the backend rejected it.<span data-glossary-navigation="true"> ([cache miss glossary](../../../../glossary.md#cache-and-memoization))</span>
- On a compatible hit, reuse the decoded decision without enumerating or executing candidates.
  Retain the cache summary in rich evidence and mark the source as `CACHE_HIT`.
- On a miss, ask the collaboration once for the complete immutable candidate list. Reject null,
  empty, duplicate, null-containing, over-budget, or unstable identity results before executing
  the first candidate for that workload. Do not prune, synthesize, reorder, or interpret a
  candidate.
- For each candidate in encounter order, invoke the complete cold measurement action for the
  configured warmups and timed samples. The public path uses `System.nanoTime`; one package-
  private clock seam in `WorkloadTuning` permits deterministic tests. Reject negative or
  non-monotonic elapsed values and checked-overflow conditions. Propagate execution failure and
  persist nothing from the failed invocation.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- Compute each candidate's median from the already bounded odd sample count without floating-
  point conversion. Select the smallest median; retain the first candidate on a tie. Call
  `selectedDecision` exactly once for the winner and require a non-null decision.
- Call `encodeDecision` only for a persistent miss winner. Require a non-null, non-empty, bounded
  byte sequence. Verify round-trip compatibility by decoding it against the same batch and
  requiring an equal decision before making it eligible for cache publication.<span data-glossary-navigation="true"> ([round-trip glossary](../../../../glossary.md#schema-serialization-and-checkpoint); [publication glossary](../../../../glossary.md#term-publication))</span>
- Return selected handoffs for every original occurrence in request order. Identical occurrences
  reuse the same exact selected-decision reference from the one deduplicated result.
- Return immutable rich evidence with model/profile fingerprints, objective/budget, occurrence
  contexts and weights, cache-hit/miss source, candidate identities, raw elapsed samples for
  misses, compact min/median/max/sample-count summaries, winner identity, and rejection/failure
  only through the thrown exception. Evidence is never written into the compact workload cache.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Add one package-private `WorkloadCacheFile` implementation. Use a deterministic bounded binary
  encoding with a fixed magic, artifact schema version one, entry count, length-prefixed opaque
  compatibility/decision/winner bytes, fixed numeric objective and sampling fields, compact
  min/median/max/sample-count summary, deterministic unsigned compatibility-key order, and a
  SHA-256 checksum over all preceding bytes. Do not use Java object serialization, JSON maps,
  reflection, class names, enum names, or string dispatch.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [serialization glossary](../../../../glossary.md#term-serialization); [JSON glossary](../../../../glossary.md#term-json); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Bound the cache file, entry count, and every byte-array length before allocation. Use explicit
  big-endian fixed-width integers and reject trailing bytes, numeric overflow, and invalid
  summary ordering.<span data-glossary-navigation="true"> ([big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries))</span>
- Persist only when at least one persistent miss produced a new compatible entry. Merge with
  retained compatible and unrelated well-formed entries, write a uniquely named sibling
  temporary file through a `FileChannel`, force file contents, publish with
  `ATOMIC_MOVE` plus `REPLACE_EXISTING`, and best-effort force the parent directory where the JDK
  and platform support it. If atomic move is unsupported or publication fails, leave the prior
  cache intact as far as the filesystem permits, clean up the temporary file, and propagate the
  failure. Do not silently fall back to a non-atomic move.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [atomic move glossary](../../../../glossary.md#term-atomic-move))</span>
- Finalize public and package Javadocs, the focused tuning/Prepare explanatory boundary, and the
  affected tuning glossary entries in the mandatory separate clean documentation context.

## Out of [scope](../../../../glossary.md#term-scope)

- a public Engine facade, model compilation, graph preparation, prepared execution construction,
  caller-input binding, published-output access, or a general end-to-end model runner<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [facade glossary](../../../../glossary.md#term-facade); [model glossary](../../../../glossary.md#term-model); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [graph preparation glossary](../../../../glossary.md#graph-preparation--graphpreparation); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [binding glossary](../../../../glossary.md#term-binding))</span>
- any [dependency](../../../../glossary.md#term-dependency) on or import from `backends/cpu`, the OpenBLAS provider, another concrete
  backend, private backend internals, or technically public unsupported-internal CPU packages<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [concrete backend glossary](../../../../glossary.md#concrete-backend); [backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- direct construction, parsing, reflection, downcast, matching, or interpretation of
  `CpuOpenBlasTuningBatch`, `CpuOpenBlasTuningDecision`, candidate fields, CPU routes, thread
  counts, representations, resources, hardware, provider qualification, or workload signatures<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification); [signatures glossary](../../../../glossary.md#term-signature))</span>
- changing Prepare's method-free marker roles or handoff, CPU 0010E values, backend candidate
  generation, compatibility rules, decision matching, heuristic fallback, analysis, finalization,
  generated-artifact cache, executable behavior, or exact/default eligibility<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [handoff glossary](../../../../glossary.md#term-handoff); [heuristic glossary](../../../../glossary.md#term-heuristic); [artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache))</span>
- BFLOAT16, relaxed or fast-math candidates, Config 0004–0006A, provider 0004, CPU 0010D1, or a
  Config facade; the initial eligible batch remains exact/default FLOAT32/FLOAT64<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- graph/plan candidate tuning, fusion/ownership/layout/materialization search, task 0002, a model-
  plan cache, or prepared-executable serialization<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [fusion glossary](../../../../glossary.md#term-fusion); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- candidate correctness validation, semantic comparison, route-specific warmup, outlier removal,
  confidence intervals, adaptive sampling, parallel measurement, cancellation, wall-time cutoffs,
  energy/memory objectives, multi-objective ranking, or performance-proof claims<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [warmup glossary](../../../../glossary.md#term-warmup); [outlier glossary](../../../../glossary.md#term-outlier); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [cancellation glossary](../../../../glossary.md#term-cancellation); [objective glossary](../../../../glossary.md#term-objective))</span>
- report-only `tools/benchmarks`, benchmark cache population, Runtime profiling, trace DTOs, or
  Runtime hot-path measurement, selection, cache access, mutation, or graph inspection<span data-glossary-navigation="true"> ([Runtime profiling glossary](../../../../glossary.md#runtime-profiling); [trace glossary](../../../../glossary.md#trace); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- cache migration, remote/shared caches, locking across concurrently writing processes, eviction,
  expiry, compression, encryption, signatures, access-control management, or recovery of corrupt
  files
- hidden global state, static mutable caches, registry, service locator, plugin discovery,
  callback discovery, raw <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [plugin glossary](../../../../glossary.md#term-plugin); [callback glossary](../../../../glossary.md#term-callback))</span>`Object`, unchecked casts, generic maps, reflective annotations,
  arbitrary parameter bags, Java serialization, or string-keyed dispatch<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- changes to Architecture/ADRs, Model, Backend Contract, Config, Planning, Compiler, Runtime,
  Prepare production, Engine, concrete backends/providers, architecture tests, backend conformance,
  integration tests, shared build logic, or another task specification<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/prepare`, Concrete backend modules, Performance evidence and optimization tooling,
  Runtime service locator, Prepare lifecycle, and Dependency rules<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module); [Runtime glossary](../../../../glossary.md#term-runtime); [service locator glossary](../../../../glossary.md#term-service-locator); [Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle); [Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Tuning master plan](../master-plan.md)
- [CPU 0010E candidate and decision contract](../../../backends/cpu/tasks/0010e-float32-float64-openblas-tuning-candidates-and-compatible-decisions.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [Prepare 0004 opaque handoff](../../../modules/prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- [Config master plan](../../../modules/config/master-plan.md)
- [Engine master plan](../../../modules/engine/master-plan.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>

## Architecture constraints

- Tuning coordinates measurement, explicit cache state, comparison, selection, and evidence. It
  must not own graph semantics, Planning ownership policy, Prepare orchestration, backend lowering,
  route vocabulary, candidate generation, compatibility meaning, executable construction, or
  Runtime behavior.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [graph glossary](../../../../glossary.md#graph); [Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- The concrete backend remains the sole owner of complete valid candidate contents, their stable
  order, canonical compatibility, candidate identity, decision construction/encoding/decoding,
  and decision validation during later analysis. The explicit caller collaboration bridges these
  owned operations without exposing their fields to the tool.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [owner glossary](../../../../glossary.md#term-owner); [canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Prepare's batch and decision roles remain method-free. The generic handoff remains an exact-
  partition opaque transport and gains no enumeration, codec, matching, measurement, or cache
  method.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [handoff glossary](../../../../glossary.md#term-handoff); [partition glossary](../../../../glossary.md#partition); [opaque glossary](../../../../glossary.md#term-opaque); [codec glossary](../../../../glossary.md#term-codec))</span>
- The cold measurement action is supplied explicitly and runs outside Runtime's hot path. It is
  not <span data-glossary-navigation="true"> ([hot path glossary](../../../../glossary.md#term-hot-path))</span>[`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) discovery or Runtime profiling. The caller must arrange equivalent
  representative inputs and complete candidate execution before invoking tuning.<span data-glossary-navigation="true"> ([Runtime profiling glossary](../../../../glossary.md#runtime-profiling))</span>
- Stable model/profile fingerprints are tool-local caller assertions used for result evidence.
  They do not define canonical Model identity, affect workload-cache reuse, or create a Model,
  Compiler, Prepare, Runtime, or Engine API.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Workload and candidate compatibility bytes are opaque to tuning and supplied by the backend-
  aware collaborator. For CPU 0010E the collaborator must derive them from the authoritative
  exact/default FLOAT32/FLOAT64 CPU values; the tool must not share an interpretation of those
  fields.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- Model autotuning is optional for correctness. An absent, incompatible, corrupt, or failed cache
  must never grant eligibility. Ordinary cache-free heuristic preparation remains unchanged.<span data-glossary-navigation="true"> ([Model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Tuning completes before any selected decision reaches repeated cold backend analysis and before
  Runtime execution. Runtime never measures, searches, selects, reads, writes, or invalidates the
  cache.
- [Benchmarking](../../../../glossary.md#term-benchmarking) remains report-only and does not call this workflow to mutate production settings.
- No hidden global state, concrete-backend dependency, module ownership change, or architecture
  rule is authorized. Stop if implementation requires one.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.tools.tuning` — current tool module root; becomes the deliberate small
  public surface and contains its package-private cache implementation.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- `io.github.pho001.synaptik.prepare.analysis` — existing public method-free opaque roles and
  exact-partition handoff used only through bounded generics.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [partition glossary](../../../../glossary.md#partition); [handoff glossary](../../../../glossary.md#term-handoff))</span>

Packages added or changed:

- No Java package is added.
- The tuning root changes from a placeholder-only package to one cohesive local-workload tuning
  API plus package-private persistence mechanics.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [persistence glossary](../../../../glossary.md#term-persistence))</span>

Type placement:

- `io.github.pho001.synaptik.tools.tuning.WorkloadTuning` — stateless workflow entry and private
  orchestration; contains only the package-private monotonic-clock test seam.
- `io.github.pho001.synaptik.tools.tuning.WorkloadTuningRequest` — immutable tool-local request and
  its small nested <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>[`Objective`](../../../../glossary.md#term-objective), `Budget`, fingerprint, compatibility, candidate-identity, reuse-
  scope, and occurrence values.<span data-glossary-navigation="true"> ([fingerprint glossary](../../../../glossary.md#term-fingerprint); [identity glossary](../../../../glossary.md#term-identity-meanings); [scope glossary](../../../../glossary.md#term-scope); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `io.github.pho001.synaptik.tools.tuning.BackendWorkloadTuning` — typed caller collaboration for
  backend-owned enumeration, opaque identities, decision construction, and codec operations.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>
- `io.github.pho001.synaptik.tools.tuning.ColdCandidateMeasurement` — typed one-complete-candidate
  execution action, separate from Runtime and benchmarking.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [benchmarking glossary](../../../../glossary.md#term-benchmarking))</span>
- `io.github.pho001.synaptik.tools.tuning.WorkloadTuningResult` — immutable selected handoffs and
  rich nested occurrence/workload/candidate evidence values.
- `io.github.pho001.synaptik.tools.tuning.WorkloadCacheFile` — package-private bounded binary
  cache parsing, merging, deterministic encoding, checksum, and atomic publication.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [checksum glossary](../../../../glossary.md#term-checksum); [atomic publication glossary](../../../../glossary.md#term-atomic-publication))</span>

Tests mirror the production package because they must exercise package-private clock and [cache](../../../../glossary.md#term-cache)
seams. No `api`, `cache`, `evidence`, `internal`, `util`, `common`, or [backend](../../../../glossary.md#backend) package is added.

## Affected files

Expected production, build, and test paths:

- `tools/tuning/build.gradle.kts`
- delete `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/TuningToolModule.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/package-info.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/WorkloadTuning.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/WorkloadTuningRequest.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/BackendWorkloadTuning.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/ColdCandidateMeasurement.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/WorkloadTuningResult.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/WorkloadCacheFile.java`
- add `tools/tuning/src/test/java/io/github/pho001/synaptik/tools/tuning/WorkloadTuningTest.java`
- add `tools/tuning/src/test/java/io/github/pho001/synaptik/tools/tuning/WorkloadTuningCacheTest.java`
- add `tools/tuning/src/test/java/io/github/pho001/synaptik/tools/tuning/WorkloadTuningValidationTest.java`
- add `tools/tuning/src/test/java/io/github/pho001/synaptik/tools/tuning/TuningPublicShapeTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/developer-guide/benchmarking.md` — distinguish the operational tuning workflow from
  report-only benchmark evidence and explain the current caller-supplied cold boundary
- `docs/backend-guide/partition-preparer.md` — explain how an owning composition supplies the
  typed collaboration without widening the method-free Prepare roles<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>
- `docs/glossary.md` — update the tuning workflow/artifact contracts, measurement and persistent
  workload-cache status, and tuning-orchestration status from planned to the exact implemented
  task-0001 boundary<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache))</span>
- this task
- `docs/planning/tools/tuning/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a concrete contradiction is found: `ARCHITECTURE.md`, focused
architecture explanations and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0008, Config/[Engine](../../../../glossary.md#term-engine)/[Compiler](../../../../glossary.md#term-compiler)/Planning/[Prepare](../../../../glossary.md#prepare) master
plans, completed [CPU](../../../../glossary.md#term-cpu) 0010E and Prepare 0004 tasks, CPU/Prepare production and tests, [Runtime](../../../../glossary.md#term-runtime) [API](../../../../glossary.md#term-application-programming-interface-api),
other [Gradle](../../../../glossary.md#term-gradle) files, architecture tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) tests, and integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the 19 exact paths above:

- 9 tuning production/build paths;
- 4 tuning test paths;
- 3 focused explanatory-documentation paths (the two guides plus `docs/glossary.md`); and
- 3 tuning task/master/roadmap [planning](../../../../glossary.md#term-planning) paths.

The task remains one cohesive capability because measurement, compatible [cache](../../../../glossary.md#term-cache) hits, miss-only
selection, decision [codec](../../../../glossary.md#term-codec) round trips, evidence separation, and [atomic publication](../../../../glossary.md#term-atomic-publication) are one
observable [transaction](../../../../glossary.md#term-transaction) and share the same failure-before-mutation tests. Splitting cache or
measurement first would create an unconsumed format or an operational workflow that cannot meet
its reuse contract. If another path, production type, package, [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), or public
[operation](../../../../glossary.md#operation) is required, stop and return the task to planning.

## Acceptance criteria

1. The five public types and one package-private cache type listed in Package impact exist with
   the exact responsibilities and generic bounds specified; no additional public production type
   or package is added.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
2. Public signatures expose only JDK values, tool-local immutable values, and Prepare's method-
   free roles/handoff. The tuning module has a public Prepare dependency and no concrete-backend,
   Engine, or Runtime dependency.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [immutable glossary](../../../../glossary.md#term-immutable); [Prepare glossary](../../../../glossary.md#prepare); [handoff glossary](../../../../glossary.md#term-handoff); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
3. Request validation snapshots all byte sequences/lists, retains exact handoff references,
   rejects null/empty/invalid values and present input decisions, and locks the sole initial
   objective to <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [objective glossary](../../../../glossary.md#term-objective))</span>`MIN_MEDIAN_ELAPSED_NANOS`.
4. The backend collaboration is called only for its six declared operations. Tuning uses no
   reflection, downcast, raw type, unchecked cast, `Object`, map/string dispatch, route switch,
   CPU field access, or concrete-backend import.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [CPU glossary](../../../../glossary.md#term-cpu))</span>
5. Exact compatible occurrences deduplicate once while retaining every context and checked-sum
   weight. Session results reuse only in the invocation; persistent entries reuse across cache
   loads and models. Model/profile identity never changes workload-cache matching.<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight); [Model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
6. The complete cache is loaded and validated before candidate enumeration or execution. Every
   specified corruption/size/schema/duplicate/truncation/checksum/trailing-byte case fails without
   measurement or file mutation.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [checksum glossary](../../../../glossary.md#term-checksum))</span>
7. An exact well-formed compatible cache hit calls backend decoding once, executes and enumerates
   no candidate, returns the decoded selected decision for every occurrence, and records only its
   compact summary as <span data-glossary-navigation="true"> ([cache hit glossary](../../../../glossary.md#cache-and-memoization); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`CACHE_HIT` evidence.
8. A missing or backend-rejected entry is a miss. Only misses enumerate and measure candidates;
   incompatibility never grants eligibility or falls back to a stale decision.
9. Budget and complete-candidate validation finish before the first execution. Over-budget,
   empty/null/duplicate candidates, unstable or duplicate identities, and arithmetic overflow
   fail without partial measurement or persistence.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [persistence glossary](../../../../glossary.md#term-persistence))</span>
10. Deterministic clock-backed tests prove exact warmup/timed invocation counts, non-negative
    sample validation, integer median calculation, lowest-median selection, encounter-order tie
    breaking, checked failure propagation, and one decision construction for the winner.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [median glossary](../../../../glossary.md#term-median))</span>
11. Miss evidence retains every raw timed sample and compact min/median/max/count summary;
    cache entries retain only the summary. Evidence, request, result, and cache snapshots are
    immutable and leak no mutable byte array or collection.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>
12. Persistent decision encoding is non-empty/bounded and must decode back to an equal compatible
    decision before publication. Session results and failed/incompatible round trips are never
    persisted.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
13. Cache encoding is deterministic and bounded with the exact binary ingredients in Scope;
    Java serialization, JSON, class names, enum names, strings as dispatch, and executable bytes
    are absent.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [Scope glossary](../../../../glossary.md#term-scope); [serialization glossary](../../../../glossary.md#term-serialization); [JSON glossary](../../../../glossary.md#term-json))</span>
14. Same-directory forced temporary-file plus atomic replacement publishes a changed persistent
    cache. Tests prove no-write hits, stable deterministic bytes, retained unrelated entries,
    temporary cleanup, and preservation of the prior cache on injected pre-move and move failure.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
15. The workflow performs no report-only benchmark mutation, Runtime profiling/hot-path work,
    Prepare analysis/finalization, backend discovery, model compilation, graph/plan tuning, or
    performance-proof assertion.<span data-glossary-navigation="true"> ([Runtime profiling glossary](../../../../glossary.md#runtime-profiling); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [graph glossary](../../../../glossary.md#graph))</span>
16. CPU 0010E and Prepare 0004 remain Complete; Config 0006A remains Draft without a detailed
    task; provider 0004 and CPU 0010D1 remain blocked/deferred; tuning 0002–0003 remain Draft
    without detailed task specifications.
17. A separate clean documentation-focused context independently reviews implementation/tests,
    finalizes affected Javadocs/package prose, the two focused guides, and `docs/glossary.md`,
    records reasoned no-change conclusions for excluded areas, and does not repeat successful
    Java tests unless it changes executable behavior or records a concrete stale-evidence risk.
18. Focused/final tuning tests, tuning Javadoc and rendered-page inspection, root dependency/
    architecture validation, Markdown links/anchors/headings/fences/newlines/whitespace, exact
    path/public-shape/package/dependency/status/order checks, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [shape glossary](../../../../glossary.md#shape))</span>`git diff --check` pass before
    completion.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests while implementing. After executable Java stabilizes, run one final [module](../../../../glossary.md#term-module)
command:

```bash
./gradlew :tools:tuning:test
```

Tests use synthetic [immutable](../../../../glossary.md#term-immutable) implementations of the [Prepare](../../../../glossary.md#prepare) marker roles and a deterministic
package-private clock. They are filesystem-local and execute no [CPU](../../../../glossary.md#term-cpu)/[OpenBLAS](../../../../glossary.md#term-openblas)/[Runtime](../../../../glossary.md#term-runtime) code. They
must prove the complete operational workflow rather than only [DTO](../../../../glossary.md#data-transfer-object--dto) construction.

The documentation-focused context reuses that successful executable evidence unless it changes
Java behavior or tests, then runs:

```bash
./gradlew :tools:tuning:javadoc
git diff --check
git status --short -uall
```

Render and inspect the five public tuning declarations and package summary. Validate every
changed Markdown file for local [targets](../../../../glossary.md#term-target) and anchors, unique headings, balanced backtick/tilde
fences, LF/final newlines, trailing whitespace, terminology, exact [scope](../../../../glossary.md#term-scope), package placement,
public [shape](../../../../glossary.md#shape), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), and synchronized statuses.

After the documentation pass, run one repository-wide [checkpoint](../../../../glossary.md#term-checkpoint) because this task adds a public
Prepare [dependency](../../../../glossary.md#term-dependency) and a new persistent-[artifact](../../../../glossary.md#term-artifact) boundary:

```bash
./gradlew test
```

No real OpenBLAS library, performance threshold, benchmark run, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance)-specific run,
or integration-specific run is required. The root command supplies dependency and architecture-
test coverage; this task changes no backend or end-to-end execution behavior.

## Dependencies

- CPU 0010E — Complete and authoritative for the first exact/default FLOAT32/FLOAT64 complete
  candidate batches, compatibility facts, candidate identities, and selected decisions.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Prepare 0004 — Complete and authoritative for the method-free opaque batch/decision roles and
  exact-partition typed handoff.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque); [partition glossary](../../../../glossary.md#partition); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- JDK filesystem, channel, checksum/digest, monotonic-clock, and atomic-move APIs already available
  to the tool module.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [channel glossary](../../../../glossary.md#term-channel); [checksum glossary](../../../../glossary.md#term-checksum); [digest glossary](../../../../glossary.md#term-digest); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Explicit caller composition supplies stable model/profile fingerprints, the typed backend
  collaboration, and one complete comparable cold candidate execution action.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend))</span>

The last item is an input contract implemented and exercised with deterministic test
collaborators in this task, not an absent [module](../../../../glossary.md#term-module) prerequisite. [Engine](../../../../glossary.md#term-engine), Config 0006A, provider
0004, [CPU](../../../../glossary.md#term-cpu) 0010D1, [Runtime](../../../../glossary.md#term-runtime) changes, report-only benchmarks, and a public CPU adapter are not
dependencies.

## Follow-up tasks

- Tuning 0002 remains Draft. It later requires Compiler/Planning complete candidates, a complete
  Prepare candidate path, and operational Engine lifecycle measurement to tune bounded graph and
  plan alternatives without repeating local search.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [graph glossary](../../../../glossary.md#graph))</span>
- Tuning 0003 remains Draft and later adds cache/plan inspection after both artifact roles are
  stable.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Config 0006A remains Draft and may define immutable public request inputs only after this
  tool-local consumer is implemented and reviewed. Do not create its detailed specification here.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Engine 0001–0004 remain Draft. Later Engine composition may supply model identity and complete
  candidate execution through this task's generic collaboration; it must not be invented here.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- CPU 0016 later may generalize CPU-owned candidate/decision production across eligible peer
  routes. It does not move CPU vocabulary or an adapter into shared tuning.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- A supported out-of-the-box CPU integration, if later required by an actual public consumer,
  must be planned at the composition boundary without making tuning depend on unsupported CPU
  internals or making CPU depend on tooling.

## Architecture impact

Expected impact: None.

This task implements the explicit tuning [ownership](../../../../glossary.md#term-ownership-meanings) and [opaque](../../../../glossary.md#term-opaque) collaboration already authorized by
the architecture. Adding a public [Prepare](../../../../glossary.md#prepare) [dependency](../../../../glossary.md#term-dependency) from the outer tool follows the existing
direction and changes no inward [module](../../../../glossary.md#term-module). If implementation requires a concrete-[backend](../../../../glossary.md#backend) edge,
Prepare role methods, [Runtime](../../../../glossary.md#term-runtime) measurement, an [Engine](../../../../glossary.md#term-engine) [facade](../../../../glossary.md#term-facade), Config [API](../../../../glossary.md#term-application-programming-interface-api), hidden state, or another
architecture rule, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik tools/tuning task 0001. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the tuning master plan, and
docs/planning/tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md
in full, plus its directly referenced completed CPU 0010E and Prepare 0004 contracts and current
affected sources/tests. Implement exactly the Ready task within its 19-path ceiling. Stop for any
architecture, concrete-backend, Engine, Runtime, Config, Prepare-role, cache-safety, or scope
conflict instead of inventing a boundary.

After executable code and the final tuning test stabilize, hand the exact diff and evidence to a
distinct clean documentation-focused context in the same overall change. That context must follow
docs/developer-guide/documentation-rules.md, independently inspect implementation/tests, finalize
affected Javadocs/package documentation, the two focused guides, `docs/glossary.md`, planning
evidence, and documentation validation, and avoid repeating successful Java suites unless
executable behavior changes or a concrete stale-evidence risk is recorded. Do not mark 0001
Complete until that pass and every specified gate succeed.
```

## Local decisions

- Keep task 0001 Ready by making missing Engine/model-execution producers explicit caller inputs,
  not by inventing an Engine facade or claiming the placeholder module is operational.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [Engine glossary](../../../../glossary.md#term-engine); [model glossary](../../../../glossary.md#term-model); [facade glossary](../../../../glossary.md#term-facade); [module glossary](../../../../glossary.md#term-module))</span>
- Use a typed generic collaboration beside the tuning workflow rather than adding methods to
  Prepare's opaque markers. This gives the operational consumer exactly the six backend-owned
  operations it needs while preserving method-free transport and backend vocabulary ownership.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque); [backend glossary](../../../../glossary.md#backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Treat model/profile fingerprints as caller-supplied evidence identity. Workload-cache reuse is
  deliberately model-independent and keyed only by backend-supplied canonical compatibility plus
  objective/sampling semantics.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [cache glossary](../../../../glossary.md#term-cache); [canonical glossary](../../../../glossary.md#term-canonical-form); [objective glossary](../../../../glossary.md#term-objective); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- Measure every candidate in a complete miss batch. The budget must admit the full deduplicated
  work or fail before execution; a time cutoff or prefix search would make the winner dependent on
  machine speed or backend order beyond the explicit tie rule.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Use only minimum median elapsed nanoseconds in this first slice. Additional objectives or
  statistical policies require separate consumers and evidence rather than generic scoring maps.<span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median); [slice glossary](../../../../glossary.md#slice))</span>
- Keep timing in tuning by supplying an execution action, not a caller-computed duration. A
  package-private clock makes validation deterministic without exposing clock policy publicly.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Reject a corrupt cache as an explicit request failure and preserve it for diagnosis. A valid
  but backend-incompatible entry is a miss because only the backend-aware collaborator can decide
  compatibility.
- Require decision encode/decode equality before atomic publication. This prevents a codec from
  persisting a value that the same current batch cannot recover compatibly.<span data-glossary-navigation="true"> ([atomic publication glossary](../../../../glossary.md#term-atomic-publication); [codec glossary](../../../../glossary.md#term-codec))</span>
- Keep one cohesive 19-path task because cache-first reuse, miss measurement, deterministic
  selection, evidence separation, and atomic publication are one transactional capability and
  share failure/mutation invariants.

## Known limitations

- The task supplies an operational generic workflow and deterministic synthetic integration
  tests, not a supported public CPU adapter or public Engine entry. Current CPU candidate values
  remain technically public unsupported internals interpreted only by backend-aware composition.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend))</span>
- Only exact/default FLOAT32/FLOAT64 candidate batches are authorized for the first workflow.
  The tool itself does not inspect data type or numerical fields; eligibility is guaranteed by
  the supplying backend collaboration.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- The first objective is minimum median elapsed nanoseconds with fixed warmup and odd sample
  counts. There is no correctness oracle, noise model, adaptive sampling, confidence claim, or
  proof that the selected candidate is universally faster.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [median glossary](../../../../glossary.md#term-median); [warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [oracle glossary](../../../../glossary.md#term-test-oracle); [model glossary](../../../../glossary.md#term-model); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- Session-only workloads deduplicate only inside one invocation and are not persisted. Cross-
  process reuse requires backend-supplied persistent compatibility and a successful codec round
  trip.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [round trip glossary](../../../../glossary.md#term-round-trip))</span>
- Concurrent writers, cache migration/recovery/inspection, eviction, remote caches, graph/plan
  tuning, prepared artifacts, and Config/Engine facades remain deferred.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [graph glossary](../../../../glossary.md#graph); [artifacts glossary](../../../../glossary.md#term-artifact))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Documentation-focused review context:
  `01a091d6-247d-7f82-844a-7a2860541741`.
- Reused the clean executable-correction context's final
  `./gradlew :tools:tuning:test` result because this documentation pass changed no executable Java
  behavior or tests: 4 suites, 24 tests, 0 failures, 0 errors, 0 skipped. The four XML suites are
  `TuningPublicShapeTest` (3), `WorkloadTuningCacheTest` (8), `WorkloadTuningTest` (3), and
  `WorkloadTuningValidationTest` (10).
- `./gradlew :tools:tuning:javadoc` passed after the final [Javadoc](../../../../glossary.md#term-javadoc) edits with no warnings.
- Generated package summary and the five public top-level declaration pages were rendered to text
  with Pandoc and inspected. The nested `CandidateEvidence` and `WorkloadEvidence` pages were also
  inspected for the measured-only exact raw-sample invariant and empty cache-hit candidate list.
  An in-app/connected browser renderer was unavailable, and macOS Quick Look rendering was denied
  by the execution environment; generated HTML structure and rendered text were therefore the
  available rendering evidence.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [invariant glossary](../../../../glossary.md#term-invariant); [cache glossary](../../../../glossary.md#term-cache))</span>
- Package-private `WorkloadCacheFile` source and Javadocs were inspected for bounds, complete-file
  validation, deterministic encoding, immutable byte snapshots, same-directory forced temporary
  files, atomic replacement, prior-file preservation, cleanup, and absence of executable payloads.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Changed-Markdown validation passed for local file targets and heading anchors, unique headings,
  balanced backtick/tilde fences, LF endings, final newlines, trailing whitespace, and current-
  versus-planned terminology.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- Exact-scope inspection confirmed exactly 19 authorized status paths: 9 production/build paths,
  4 test paths, 3 explanatory documentation paths, and 3 planning paths. Public-shape tests and
  source inspection confirmed exactly five public top-level production types plus the package-
  private cache type. Gradle/source inspection confirmed a public Prepare dependency, private
  Planning dependency, and no Config, Engine, Runtime, or concrete-backend dependency.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [planning glossary](../../../../glossary.md#term-planning); [shape glossary](../../../../glossary.md#shape); [Gradle glossary](../../../../glossary.md#term-gradle); [Prepare glossary](../../../../glossary.md#prepare); [dependency glossary](../../../../glossary.md#term-dependency); [Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- Status/order inspection confirmed CPU 0010E and Prepare 0004 remain Complete; provider 0004 and
  CPU 0010D1 remain blocked/deferred; tuning 0002 and 0003 remain Draft without specifications;
  Config 0006A is the next Draft task.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- `./gradlew test` passed: `BUILD SUCCESSFUL`, 65 actionable tasks (1 executed, 64 up-to-date).
  Aggregating the current JUnit XML produced 468 suites, 3,032 tests, 0 failures, 0 errors, and 28
  skipped tests.<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit))</span>
- `git diff --check` passed on the final combined change.

## Implementation notes

- Replaced the placeholder module marker with five focused public API types and one package-private
  cache codec in the existing tuning package; no new package or public facade was introduced.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [cache glossary](../../../../glossary.md#term-cache); [codec glossary](../../../../glossary.md#term-codec); [facade glossary](../../../../glossary.md#term-facade))</span>
- The workflow loads and validates the explicit cache first, deduplicates exact compatibility in
  request order, validates every complete miss batch before execution, measures configured warmup
  and odd timed samples, selects minimum integer median with encounter-order ties, round-trips a
  persistent decision codec, and publishes only completed persistent misses atomically.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [warmup glossary](../../../../glossary.md#term-warmup); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [median glossary](../../../../glossary.md#term-median))</span>
- The executable correction finalized `CandidateEvidence` as measured-only evidence with a
  non-empty immutable encounter-order snapshot of non-null non-negative samples whose count,
  minimum, integer-middle median, and maximum exactly match its summary. Cache-hit<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
  `WorkloadEvidence` exposes an empty candidate-evidence list.
- The public surface remains a generic composition boundary. Callers supply stable identities,
  backend-owned compatibility/candidate identity and decision codec behavior, candidate
  enumeration, and complete candidate execution. There is no supported CPU adapter, Engine
  integration, Config facade, model extraction, or graph/plan tuning.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [identity glossary](../../../../glossary.md#term-identity-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph))</span>

## Completion summary

- Completed changes: implemented the exact/default generic workload-tuning transaction, reusable
  bounded persistent cache, selected handoffs, and separate rich evidence; finalized Javadocs,
  guides, glossary, and synchronized planning state.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction); [cache glossary](../../../../glossary.md#term-cache); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly the 19 paths listed under Affected files.
- Tests and validation: reused the final 24-test tuning-module result; new clean Javadoc,
  rendered-text/page inspection, Markdown/scope/dependency/status checks, repository-wide 3,032-
  test checkpoint, and final whitespace check passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [dependency glossary](../../../../glossary.md#term-dependency); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Documentation-agent review: this mandatory clean documentation-focused context independently
  audited all acceptance criteria and finalized the same overall uncommitted change.
- Documentation impact: benchmarking and partition-preparer guides now describe the operational
  generic caller-supplied tuner/cache and its unsupported integration boundaries.<span data-glossary-navigation="true"> ([benchmarking glossary](../../../../glossary.md#term-benchmarking); [partition glossary](../../../../glossary.md#partition))</span>
- Javadoc review: all five public types, their nested public values and members, package summary,
  and package-private cache contract were finalized; `CandidateEvidence` and cache-hit evidence
  invariants are explicit.
- Glossary impact: corrected the three stale claims about workflow/artifact contracts,
  measurement/persistent-cache use, and tuning orchestration while leaving the full two-phase
  roadmap planned.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Unresolved issues: None within task 0001 [scope](../../../../glossary.md#term-scope).
- Follow-up required: Config 0006A remains the next Draft task; tuning 0002/0003, supported
  CPU/Engine integration, model extraction, and graph/plan tuning remain planned.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph))</span>

Status: Complete
