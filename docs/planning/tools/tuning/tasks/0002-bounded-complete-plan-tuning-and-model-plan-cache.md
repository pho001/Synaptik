# Task 0002: Bounded Complete-Plan Tuning and [Model](../../../../glossary.md#term-model)-Plan [Cache](../../../../glossary.md#term-cache)

## Editorial reading key: who executes the candidate?

Composition code calls the generic complete-plan tool with a backend collaborator, correctness
actions and a measurement action. The tool controls ordering and trial counts. The collaborator
defines candidate meaning and compatibility; the execution actions handle representative input
values, fresh preparation, result copying and cleanup. The tool's opaque correctness reference is
temporary bookkeeping, not a transfer of live execution resources into a cache.

### Reading the trial-count arithmetic

For a measured cache miss, let $N\geq1$ be the integer number of complete candidates,
$W\geq0$ the integer warmups per candidate and $S\geq1$ an odd integer number of timed samples
per candidate. The execution count $E$, in complete candidate executions, is

$$
E=N(1+W+S).
$$

The one accounts for each candidate's correctness action. For example, $N=2$, $W=1$ and $S=5$
require $E=2(1+1+5)=14$ complete actions: two correctness, two warmup and ten timed actions.
This is integer counting, not floating-point evaluation or a duration estimate. Checked
implementation arithmetic and request ceilings can reject it before any action; a compatible
persistent cache hit performs none of these trials.

Read [the complete-plan transaction](../../../../developer-guide/benchmarking.md#phase-2-complete-plan-transaction)
and [candidate responsibility](../../../../architecture/performance-evidence-and-tuning.md#candidate-ownership-and-opaque-orchestration)
for those boundaries. The original formulas, later-Engine claims and execution receipts below
remain historical text.


## Status

Complete

## Goal

Implement one bounded, generic complete-plan tuning [transaction](../../../../glossary.md#term-transaction) in `tools/tuning`. A caller that
already composes the relevant [backend](../../../../glossary.md#backend) and execution [lifecycle](../../../../glossary.md#lifecycle) supplies one complete [opaque](../../../../glossary.md#term-opaque) [Prepare](../../../../glossary.md#prepare)
[candidate batch](../../../../glossary.md#candidate-batch), backend-owned [identity](../../../../glossary.md#term-identity-meanings)/[codec](../../../../glossary.md#term-codec) operations, exact correctness actions, and one fresh
complete-candidate execution action. The tool validates the whole transaction before execution,
runs correctness before timing, selects deterministically, returns the selected typed [handoff](../../../../glossary.md#term-handoff) plus
a compact selected-plan record and separate rich evidence, and conditionally reuses or atomically
publishes an explicit [model](../../../../glossary.md#term-model)-plan [cache](../../../../glossary.md#term-cache).

This task is the tools-only Phase-2 consumer foundation. It does not compose current [CPU](../../../../glossary.md#term-cpu) or [Engine](../../../../glossary.md#term-engine)
types. Complete CPU 0010J is the first truthful [producer](../../../../glossary.md#term-producer) and Complete Engine 0008A is the
correctness [owner](../../../../glossary.md#term-owner), but a later Engine task adapts both collaborations.

```text
caller-owned opaque batch + identities/policies/bounds + explicit cache path
  -> SESSION: skip all filesystem access
  -> PERSISTENT: load and authenticate a compatible compact decision, or miss
  -> miss: validate N and checked N * (1 + W + S) before execution
  -> capture/compare all N candidates exactly, with no timing
  -> only after every candidate matches: W warmups + S timed executions each
  -> lowest integer-middle median; first encounter wins ties
  -> selected typed handoff + compact selected-plan record + separate rich evidence
  -> PERSISTENT miss only: codec round trip and atomic compact-cache publication
```

## Readiness audit

The fresh post-[Engine](../../../../glossary.md#term-engine)-0008A source audit found no missing contract, inversion, or architecture
decision:

1. `tools/tuning` can own the transaction using its existing public Prepare dependency and
   tool-local generic callbacks. It needs no Engine, Runtime, Config, CPU, or backend-internal
   import.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction); [Prepare glossary](../../../../glossary.md#prepare); [dependency glossary](../../../../glossary.md#term-dependency); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
2. `CpuCompletePlanTuning` already supplies [immutable](../../../../glossary.md#term-immutable) `CandidateBatch`, `Candidate`,
   `SelectedDecision`, compatibility, identity, codec, and fresh trial/selected preparation
   operations. A later caller can adapt them without the tool inspecting a CPU value.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [codec glossary](../../../../glossary.md#term-codec))</span>
3. `RepresentativeExecutionSession` and `RepresentativePlanCorrectness` keep publication bytes,
   Runtime leases, descriptor preflight, aggregate byte enforcement, association, poisoning, and
   cleanup package-private. A later Engine adapter can present only one opaque reference and<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [leases glossary](../../../../glossary.md#term-lease); [descriptor glossary](../../../../glossary.md#term-descriptor); [opaque glossary](../../../../glossary.md#term-opaque))</span>
   `MATCH`/`MISMATCH` to this task's generic correctness collaboration.
4. `CpuCompletePlanTuning.prepareTrial(...)` and the representative session prove that one caller
   action can freshly prepare and execute a complete candidate on each invocation. The tool needs
   to own only invocation count and `System.nanoTime` measurement.
5. One correctness action per candidate fixes exact [preflight](../../../../glossary.md#term-preflight) arithmetic at
   `N * (1 + W + S)`. Positive candidate and total-execution ceilings, non-negative `W`, positive
   odd `S`, and checked addition/multiplication can all be validated before execution.
6. [CPU](../../../../glossary.md#term-cpu) 0010J truthfully reports `SESSION`, so the generic workflow can skip lookup/publication and
   still return an in-memory record and evidence. Synthetic <span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>`PERSISTENT` tests can validate the
   generic cache without claiming current CPU persistence.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence))</span>
7. A compact cache needs only opaque identities, encoded selected decisions, and compact timing
   summaries. Backend-compatible decoding authenticates a hit; fresh selected preparation remains
   outside this task. No executable or [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) is serialized.
8. Config 0006A's four Phase-1 budget meanings remain untouched. A later Config task adds distinct
   Phase-2 fields after this consumer stabilizes, and a later Engine task owns CPU/Engine
   adaptation, fresh production preparation, evidence translation, and fallback.

## [Scope](../../../../glossary.md#term-scope)

- Add one stateless public complete-plan tuning entry in
  `io.github.pho001.synaptik.tools.tuning`.
- Add an [immutable](../../../../glossary.md#term-immutable) tool-local request with:
  - non-empty versioned [model](../../../../glossary.md#term-model), representative-profile, and [target](../../../../glossary.md#term-target) fingerprints;
  - [objective](../../../../glossary.md#term-objective) `MIN_MEDIAN_ELAPSED_NANOS`;
  - correctness policy `EXACT_CANONICAL_BYTES`;
  - one non-empty versioned [opaque](../../../../glossary.md#term-opaque) constraint/policy [identity](../../../../glossary.md#term-identity-meanings);
  - a positive maximum plan-candidate count;
  - a non-negative [warmup](../../../../glossary.md#term-warmup) count;
  - a positive odd timed-[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) count;
  - a positive maximum total plan-execution count;
  - a non-negative maximum aggregate correctness-byte count carried to the caller collaboration;
  - one explicit model-plan-[cache](../../../../glossary.md#term-cache) `Path`; and
  - one exact decision-empty `BackendPartitionTuningHandoff<C, D>`.
- Add one typed generic backend collaboration over Prepare's method-free batch and decision roles.
  It supplies only the complete candidate list, opaque producer/codec identity, compatibility and
  reuse scope, candidate identities, decision construction, decision encoding, and compatible
  decision decoding. All returned collections and bytes are snapshotted or immutable at the tool
  boundary.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Prepare glossary](../../../../glossary.md#prepare); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [producer glossary](../../../../glossary.md#term-producer); [codec glossary](../../../../glossary.md#term-codec))</span>
- Add one generic exact-correctness collaboration. Its first operation captures an opaque
  caller-owned reference for the first candidate under the request's maximum aggregate
  correctness-byte bound; its second compares each later candidate and returns only <span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>`MATCH` or
  `MISMATCH`. The reference is a generic value that is never returned, persisted, inspected,
  compared, or exposed in evidence by the tool.
- Add one generic complete-candidate measurement action. Each invocation performs one complete
  freshly prepared and freshly executed candidate outside the tool. The tool owns only invocation
  count and monotonic elapsed-time measurement.
- Obtain compatibility before filesystem work. For `SESSION`, perform no filesystem query,
  create, read, write, temporary-file, move, or directory-force operation. Always run fresh
  correctness and measurement, then return an in-memory compact selected-plan record.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- For `PERSISTENT`, load and completely validate the explicit cache before candidate enumeration
  or execution. An exact compatible entry is a hit only when the backend decoder returns a
  compatible decision. A hit performs zero correctness, warmup, or timed executions.
- On a miss, snapshot and validate the complete ordered candidate set and every stable unique
  identity. Let <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>`N` be its size. Before the first correctness call, checked-compute
  `perCandidate = 1 + W + S` and `total = N * perCandidate`, then require `N` and `total` not to
  exceed their positive ceilings.
- Invoke correctness in candidate encounter order. The first candidate captures the opaque
  reference; each later candidate compares against it. All `N` actions must complete and match
  before the first warmup or timed action. A mismatch throws one unambiguous typed task-local
  failure with no payload bytes or backend value.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- After correctness succeeds, execute every candidate for exactly `W` warmups and `S` timed
  samples. Use `System.nanoTime` through one package-private clock seam, validate monotonic
  non-negative elapsed values, calculate the integer-middle median, choose the smallest median,
  and retain the first encountered candidate on a tie.<span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>
- Construct the selected backend decision once. Return a decision-present typed Prepare handoff,
  one immutable compact selected-plan record, and separate immutable rich evidence. Evidence
  retains ordered correctness outcomes, raw timed samples for misses, summaries, identities,
  source, and winner; compact state retains no raw samples.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- For a persistent miss, require a bounded non-empty encoded decision and compatible equal codec
  round trip before publication. Publish no cache entry for <span data-glossary-navigation="true"> ([round trip glossary](../../../../glossary.md#term-round-trip); [publication glossary](../../../../glossary.md#term-publication))</span>`SESSION`, incompatible decoding,
  mismatch, or any failure.
- Construct no externally visible record, evidence, or result until every required action and any
  persistent publication succeeds. Propagate caller execution/cleanup and cache failures through
  their exact declared boundary; do not convert them to correctness mismatch or move fallback
  policy into tools.
- Add one package-private model-plan cache implementation in the same style as task 0001: bounded
  versioned canonical big-endian binary data, length prefixes, fixed numeric tags, deterministic
  unsigned-key ordering, whole-file SHA-256 checksum, complete parsing before use, duplicate-key
  rejection, forced same-directory unique temporary file, atomic replacement only, best-effort
  directory force, and temporary cleanup.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [checksum glossary](../../../../glossary.md#term-checksum); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Cache keys include producer/decision-codec identity; model, representative-profile, and target
  fingerprints; backend compatibility schema/bytes; objective; correctness and constraint/policy
  identity; and sampling fields. Entries contain only selected candidate identity, encoded
  decision, and compact timing summary.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- Preserve all task-0001 behavior and public meanings. Share only narrowly package-private cache
  publication/checksum mechanics if implementation proves that reuse is clearer and safer than
  duplication; do not change Phase-1 public request, result, collaboration, or cache format.
- Finalize Javadocs, package documentation, focused tuning documentation, glossary impact, and
  planning evidence in the required separate documentation-focused context.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- importing or adapting [Engine](../../../../glossary.md#term-engine), [Runtime](../../../../glossary.md#term-runtime), Config, [CPU](../../../../glossary.md#term-cpu), [OpenBLAS](../../../../glossary.md#term-openblas), or [backend](../../../../glossary.md#backend)-internal types in
  `tools/tuning`
- public Engine Phase-2 integration, representative input binding, publication copying, reference
  byte ownership, result cleanup, fallback policy, or selected production preparation<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- changing Config 0006A or reinterpreting its Phase-1 maximum-miss, candidates-per-miss, warmup,
  or timed-sample fields as Phase-2 values<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Compiler graph-transformation candidates, Planning ownership/partition alternatives, multiple
  partitions, mixed backends, or more than the complete opaque batch supplied by the caller<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [Planning glossary](../../../../glossary.md#term-planning); [partition glossary](../../../../glossary.md#partition); [opaque glossary](../../../../glossary.md#term-opaque); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- persistent CPU complete-plan reuse or a claim that CPU 0010J has cross-session compatibility
- local workload extraction, occurrence weighting, Phase-1 route ranking, or any repeat of
  task-0001 candidate search<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- tolerance, relative/absolute error, ULP policies, NaN normalization, relaxed math, stochastic
  comparison, adaptive sampling, confidence claims, outlier removal, wall-clock cutoffs, parallel
  measurement, or objectives other than minimum median elapsed nanoseconds<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance); [absolute error glossary](../../../../glossary.md#term-absolute-error); [ULP glossary](../../../../glossary.md#term-unit-in-the-last-place-ulp); [NaN glossary](../../../../glossary.md#term-nan); [normalization glossary](../../../../glossary.md#term-normalization); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [outlier glossary](../../../../glossary.md#term-outlier); [median glossary](../../../../glossary.md#term-median))</span>
- serializing [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), an executable, Runtime state, Engine reference, publication
  bytes, backend candidate object, or live resource
- cache migration, recovery, locking across concurrent writers, eviction, remote caches,
  compression, encryption, signatures, or cache inspection UI<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [signatures glossary](../../../../glossary.md#term-signature))</span>
- changes to source or tests outside `tools/tuning`, module dependencies, Gradle, architecture
  tests, backend conformance tests, integration tests, or authoritative architecture/ADRs<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/prepare`, [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module), Performance evidence and optimization tooling,
  `modules/engine`, and [Dependency](../../../../glossary.md#term-dependency) rules
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Tuning master plan](../master-plan.md)
- [Task 0001 workload tuning and cache](0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- [Prepare 0004 opaque handoff](../../../modules/prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- [CPU 0010J complete-plan producer](../../../backends/cpu/tasks/0010j-supported-complete-plan-candidate-and-decision-producer.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [producer glossary](../../../../glossary.md#term-producer))</span>
- [Engine 0008A correctness oracle](../../../modules/engine/tasks/0008a-representative-complete-plan-correctness-oracle.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- [Config 0006A current request](../../../modules/config/tasks/0006a-model-autotuning-request-configuration.md)

## Architecture constraints

- Tuning owns bounded measurement, explicit compact cache coordination, deterministic comparison,
  selection, and rich evidence. It does not own the meaning or legality of any candidate.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- The concrete producer owns the complete candidate set, stable order, identities, compatibility,
  reuse scope, decisions, codecs, and later fresh preparation. Shared Prepare remains method-free
  opaque transport.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [scope glossary](../../../../glossary.md#term-scope); [Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- The caller owns correctness semantics, the opaque reference, maximum-correctness-byte
  enforcement, execution resources, preparation, Runtime invocation, publication materialization,
  and cleanup. The tool sees only <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [materialization glossary](../../../../glossary.md#materialization))</span>`MATCH` or `MISMATCH`.
- Correctness actions are separate from timing. A correctness execution is never reused as a
  warmup or sample.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Runtime receives only a freshly prepared selected recipe later. It never receives a tuning
  request, candidate, decision, cache entry, reference, or evidence and performs no selection.
- `SESSION` is a strict no-filesystem mode, not a [cache miss](../../../../glossary.md#cache-and-memoization) that happens to avoid [publication](../../../../glossary.md#term-publication).
- Persistent decoding authenticates a compact hit; it never grants backend eligibility. Later
  composition must freshly prepare the decoded decision.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- No dependency direction or architecture rule changes. If implementation needs an inward-module
  change, concrete-backend import, public Engine oracle, widened Prepare role, or executable
  serialization, stop and report the conflict.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [module glossary](../../../../glossary.md#term-module); [Engine glossary](../../../../glossary.md#term-engine); [oracle glossary](../../../../glossary.md#term-test-oracle); [serialization glossary](../../../../glossary.md#term-serialization))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.tools.tuning` — the existing deliberate small public tuning surface
  and package-private persistence mechanics.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
- `io.github.pho001.synaptik.prepare.analysis` — existing method-free batch/decision roles and
  exact-partition handoff, used only through bounded generics.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [partition glossary](../../../../glossary.md#partition); [handoff glossary](../../../../glossary.md#term-handoff))</span>

Packages added or changed:

- No Java package is added.
- The tuning root gains one cohesive complete-plan transaction beside the unchanged workload
  transaction.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction))</span>

Type placement:

- `io.github.pho001.synaptik.tools.tuning.CompletePlanTuning` — stateless transaction entry,
  package-private clock seam, and nested typed mismatch failure.
- `io.github.pho001.synaptik.tools.tuning.CompletePlanTuningRequest` — immutable tool-local
  fingerprints, policies, Phase-2 bounds, exact handoff, and cache path.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.tools.tuning.BackendCompletePlanTuning` — typed caller collaboration
  for complete candidates, opaque producer/compatibility/identity values, decisions, and codec.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [producer glossary](../../../../glossary.md#term-producer); [identity glossary](../../../../glossary.md#term-identity-meanings); [codec glossary](../../../../glossary.md#term-codec))</span>
- `io.github.pho001.synaptik.tools.tuning.CompletePlanCorrectness` — typed opaque-reference capture
  and exact two-valued comparison collaboration.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- `io.github.pho001.synaptik.tools.tuning.CompleteCandidateMeasurement` — one complete fresh
  candidate action used only for warmups and timed samples.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- `io.github.pho001.synaptik.tools.tuning.CompletePlanTuningResult` — selected typed handoff,
  compact selected-plan record, and separate rich evidence.
- `io.github.pho001.synaptik.tools.tuning.ModelPlanCacheFile` — package-private bounded parsing,
  canonical encoding, checksum, merge, and atomic publication.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [checksum glossary](../../../../glossary.md#term-checksum); [atomic publication glossary](../../../../glossary.md#term-atomic-publication))</span>

Tests mirror the production package for clock and [cache](../../../../glossary.md#term-cache) seams. No `api`, `cache`, `internal`,
`util`, `common`, [backend](../../../../glossary.md#backend), [Engine](../../../../glossary.md#term-engine), or [Runtime](../../../../glossary.md#term-runtime) package is added.

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths:

- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/CompletePlanTuning.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/CompletePlanTuningRequest.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/BackendCompletePlanTuning.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/CompletePlanCorrectness.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/CompleteCandidateMeasurement.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/CompletePlanTuningResult.java`
- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/ModelPlanCacheFile.java`
- `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/package-info.java`

Expected test paths:

- add `tools/tuning/src/test/java/io/github/pho001/synaptik/tools/tuning/CompletePlanTuningTest.java`
- add `tools/tuning/src/test/java/io/github/pho001/synaptik/tools/tuning/CompletePlanTuningCacheTest.java`
- add `tools/tuning/src/test/java/io/github/pho001/synaptik/tools/tuning/CompletePlanTuningValidationTest.java`
- add `tools/tuning/src/test/java/io/github/pho001/synaptik/tools/tuning/CompletePlanTuningPublicShapeTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/developer-guide/benchmarking.md` — distinguish the implemented Phase-1 transaction, this
  complete-plan transaction, and still-later Engine composition<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction))</span>
- `docs/glossary.md` only if implementation makes an existing reusable complete-plan or
  model-plan-cache term incomplete; otherwise record a reasoned no-change conclusion<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [cache glossary](../../../../glossary.md#term-cache))</span>
- this task
- `docs/planning/tools/tuning/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a concrete contradiction is found: `ARCHITECTURE.md`, ADRs,
the two focused architecture explanations, Config/[Engine](../../../../glossary.md#term-engine)/[CPU](../../../../glossary.md#term-cpu)/[Prepare](../../../../glossary.md#prepare) plans and contracts, all
task-0001 source/tests, other [modules](../../../../glossary.md#term-module), [Gradle](../../../../glossary.md#term-gradle) files, architecture tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) tests,
and integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 18 paths:

- 8 tuning production/[Javadoc](../../../../glossary.md#term-javadoc) paths;
- 4 tuning test paths;
- at most 2 explanatory documentation paths; and
- exactly 3 [planning](../../../../glossary.md#term-planning) paths.

The eighteenth path is intentionally unallocated. The capability remains cohesive because
[validation](../../../../glossary.md#term-validation), correctness ordering, timing, selection, [cache](../../../../glossary.md#term-cache) [authentication](../../../../glossary.md#term-authentication), compact record, rich
evidence, and failure-before-[publication](../../../../glossary.md#term-publication) are one observable [transaction](../../../../glossary.md#term-transaction). If another production
type, package, [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), build path, explanatory path beyond the allowance, or more than
18 total paths is required, stop and return the task to planning.

## Acceptance criteria

1. The six public top-level types and one package-private cache type listed under Package impact
   exist with no additional public top-level production type or package.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
2. Public signatures use only JDK values, tool-local immutable values, and Prepare's method-free
   roles/handoff. Source and dependency checks find no Engine, Runtime, Config, CPU, provider, or
   backend-internal import or dependency.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [immutable glossary](../../../../glossary.md#term-immutable); [Prepare glossary](../../../../glossary.md#prepare); [handoff glossary](../../../../glossary.md#term-handoff); [dependency glossary](../../../../glossary.md#term-dependency); [Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
3. Request construction snapshots all bytes/collections, retains the exact handoff references,
   requires an absent input decision, and validates every identity, policy, Phase-2 bound, and
   explicit path without filesystem access.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
4. Backend collaboration results are non-null, immutable/snapshotted, stable, complete, bounded,
   and identity-unique. The tool performs no reflection, downcast, raw or unchecked cast, route
   switch, backend-field access, generic map, or string dispatch.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
5. `SESSION` performs zero filesystem operations and always performs fresh correctness and timing
   on a miss-like transaction. It returns a compact in-memory selected-plan record and rich
   evidence but publishes no file.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction))</span>
6. `PERSISTENT` completely validates the cache before candidate enumeration or execution.
   Unsupported schema, oversize, truncation, checksum failure, duplicates, noncanonical order,
   structural inconsistency, or trailing bytes fail without execution or mutation.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [checksum glossary](../../../../glossary.md#term-checksum))</span>
7. An exact cache hit invokes compatible backend decoding, constructs the selected handoff, and
   performs zero candidate enumeration, correctness, warmup, and timed actions. Later fresh
   preparation remains caller-owned and is not invoked by this task.<span data-glossary-navigation="true"> ([cache hit glossary](../../../../glossary.md#cache-and-memoization); [warmup glossary](../../../../glossary.md#term-warmup))</span>
8. A missing entry or backend-rejected decision is a miss. No stale or incompatible value grants
   eligibility.
9. Miss [validation](../../../../glossary.md#term-validation) proves positive `N`, `N <= maximumPlanCandidates`, checked
   `N * (1 + W + S)`, and the positive total-execution ceiling before the first correctness call.
   Overflow or over-budget input performs zero actions and publishes nothing.<span data-glossary-navigation="true"> ([Overflow glossary](../../../../glossary.md#term-overflow))</span>
10. Deterministic tests prove exactly `N` correctness actions in encounter order, first-candidate
    capture, later comparison, and completion of all correctness actions before any measurement.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
11. A mismatch produces the exact task-local typed failure only after the caller action returns
    `MISMATCH`; it runs no warmup/sample, returns no result/evidence/record, and publishes no cache.
    Caller execution and cleanup failures propagate without being misreported as mismatch.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
12. Each candidate receives exactly `W` warmups and `S` timed actions after correctness succeeds.
    Tests prove monotonic timing, elapsed validation, raw sample retention, integer-middle median,
    lowest-median selection, and first-encounter ties.<span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>
13. The selected decision is constructed once. The result contains the decision-present typed
    handoff, compact record, and separate immutable evidence; no opaque reference, candidate
    object, publication byte, Runtime value, or executable escapes through record/evidence.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [publication glossary](../../../../glossary.md#term-publication))</span>
14. Persistent miss encoding is bounded and must decode compatibly to an equal decision before
    publication. Cache entries retain only opaque key/decision/winner bytes and compact summaries;
    rich correctness and raw timing evidence remains in memory only.
15. Canonical bounded cache bytes, duplicate safety, deterministic ordering, whole-file checksum,
    same-directory forced temporary publication, atomic replacement, prior-file preservation on
    injected failures, and cleanup are all tested with a synthetic <span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>`PERSISTENT` [producer](../../../../glossary.md#term-producer).
16. Workload tuning API, behavior, tests, and cache format remain unchanged. Any shared private
    mechanics preserve task 0001 byte-for-byte observable behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
17. Package/Javadocs explain ownership, bounds, units, nullability, snapshots, generic association,
    lifecycle, failure, <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [nullability glossary](../../../../glossary.md#term-nullability); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>`SESSION` no-I/O, persistent authentication, and current-versus-planned
    behavior. A separate clean documentation-focused context finalizes them and reviews glossary
    impact.<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication))</span>
18. Focused/final tuning tests, tuning Javadoc/rendered inspection, Markdown validation, exact
    scope/public-shape/import/status checks, empty staging, and both diff checks pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [shape glossary](../../../../glossary.md#shape))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests while implementing. After executable Java stabilizes, run once:

```bash
./gradlew :tools:tuning:test
```

The four new test [owners](../../../../glossary.md#term-owner) use synthetic method-free [Prepare](../../../../glossary.md#prepare) values, an [opaque](../../../../glossary.md#term-opaque) reference, a
deterministic clock, a filesystem-counting `SESSION` path, and a fake `PERSISTENT` [producer](../../../../glossary.md#term-producer). They
cover validation/order/budget/correctness/mismatch, SESSION no-I/O, persistent hit/miss/corrupt/
[atomic publication](../../../../glossary.md#term-atomic-publication), deterministic selection/evidence, public [shape](../../../../glossary.md#shape)/imports, and unchanged
task-0001 behavior through the complete existing [module](../../../../glossary.md#term-module) suite.

Documentation pass:

```bash
./gradlew :tools:tuning:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

Render and inspect every new public declaration and package summary. Validate changed Markdown
local [targets](../../../../glossary.md#term-target)/anchors, unique headings, balanced backtick/tilde fences, LF/final newlines,
trailing whitespace, terminology, exact [scope](../../../../glossary.md#term-scope), package placement, public shape, imports, and
synchronized task/master/roadmap status.

Repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration, [CPU](../../../../glossary.md#term-cpu), [Engine](../../../../glossary.md#term-engine), and Config Java
tests are unnecessary for this task. It changes one tool module, no [dependency](../../../../glossary.md#term-dependency)/build boundary,
no backend behavior, and no public Engine workflow. The later Engine Phase-2 composition
[checkpoint](../../../../glossary.md#term-checkpoint) owns cross-module and end-to-end validation; [CI](../../../../glossary.md#term-continuous-integration-ci) remains the repository-wide gate.

## Dependencies

- Tools/tuning 0001 — Complete; establishes the module, monotonic timing precedent, cache safety,
  deterministic median selection, evidence separation, and explicit persistent path policy.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [median glossary](../../../../glossary.md#term-median))</span>
- Prepare 0004 — Complete; supplies the unchanged method-free opaque batch/decision roles and
  typed handoff.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- CPU 0010J — Complete evidence for one truthful bounded producer and exact Phase-1 reuse. It is
  not imported or exercised by this task.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Engine 0008A — Complete evidence for an adaptable opaque exact correctness primitive with
  Engine-owned publication bytes, byte limits, lifecycle, and cleanup. It is not imported or
  exercised by this task.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- JDK clock, filesystem, channel, checksum/digest, and atomic-move APIs already available to the
  module.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [channel glossary](../../../../glossary.md#term-channel); [checksum glossary](../../../../glossary.md#term-checksum); [digest glossary](../../../../glossary.md#term-digest); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>

Caller-supplied generic collaborations are task inputs exercised by synthetic tests, not absent
[module](../../../../glossary.md#term-module) prerequisites. Config extension and [Engine](../../../../glossary.md#term-engine)/[CPU](../../../../glossary.md#term-cpu) composition are follow-ups, not dependencies.

## Follow-up tasks

- Add one Config Phase-2 extension only after this consumer is implemented. It adds distinct
  maximum-plan-candidate, maximum-total-plan-execution, maximum-correctness-byte, policy identity,
  target identity, and explicit model-plan-cache inputs without changing Config 0006A's four
  Phase-1 budget meanings.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [target glossary](../../../../glossary.md#term-target); [model glossary](../../../../glossary.md#term-model))</span>
- Add one later Engine-owned CPU composition after both consumer contracts stabilize. It adapts
  CPU 0010J and Engine 0008A, passes the exact Phase-1 selection, freshly prepares every action,
  stops on mismatch, freshly prepares the authenticated winner for production, translates rich
  evidence, and applies the existing public safe-fallback policy.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Tuning 0003 remains Draft for [cache](../../../../glossary.md#term-cache)/plan inspection after both [artifact](../../../../glossary.md#term-artifact) formats are stable.
- Persistent CPU compatibility, Compiler graph alternatives, Planning ownership alternatives,
  multiple partitions, mixed backends, tolerance policies, and executable serialization remain
  separate future work.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [backends glossary](../../../../glossary.md#backend); [tolerance glossary](../../../../glossary.md#term-tolerance); [serialization glossary](../../../../glossary.md#term-serialization))</span>

## Documentation, [Javadoc](../../../../glossary.md#term-javadoc), and glossary impact

Apply General plus [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc style to the Java contracts, General plus Developer-guide style to
the focused [benchmarking](../../../../glossary.md#term-benchmarking)/tuning explanation, and General plus [Planning](../../../../glossary.md#term-planning) style to this task and
status [synchronization](../../../../glossary.md#term-synchronization). Use Example style only for a new multi-step example and label planned
[Engine](../../../../glossary.md#term-engine) adaptation as planned.

Review `docs/glossary.md` for the existing [complete-plan candidate](../../../../glossary.md#complete-plan-candidate), [candidate batch](../../../../glossary.md#candidate-batch), [selected
tuning decision](../../../../glossary.md#selected-tuning-decision), and tuning [artifact](../../../../glossary.md#term-artifact) entries. Change it only if the implementation makes those
reusable definitions incomplete. A no-change conclusion must explain why the task composes
existing terms rather than changing their meaning.

## Architecture impact

Expected impact: None.

The task realizes the already authorized tools measurement/[cache](../../../../glossary.md#term-cache) [ownership](../../../../glossary.md#term-ownership-meanings) through the current
[opaque](../../../../glossary.md#term-opaque) [Prepare](../../../../glossary.md#prepare) boundary. It changes no [module](../../../../glossary.md#term-module) edge, ownership rule, [lifecycle](../../../../glossary.md#lifecycle) order, or public
[Engine](../../../../glossary.md#term-engine) behavior. If implementation requires an authoritative change, stop and report it; do not
edit `ARCHITECTURE.md`, an [ADR](../../../../glossary.md#term-architecture-decision-record-adr), or architecture tests within this task.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the isolated implementation agent for Synaptik tools/tuning task 0002. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/tools/tuning/master-plan.md, and
docs/planning/tools/tuning/tasks/0002-bounded-complete-plan-tuning-and-model-plan-cache.md in full,
plus the directly referenced task-0001, Prepare 0004, CPU 0010J, Engine 0008A, current tuning
source/tests, and documentation rules/profiles. Implement exactly the Ready specification within
its 18-path ceiling. Stop and report any architecture, dependency, public-surface, cache-safety,
or scope conflict; do not import or implement Engine/CPU/Config integration.

After executable code and the final tuning test stabilize, hand the same worktree and exact test
evidence to a distinct clean documentation-focused context. That context must independently
inspect implementation/tests, finalize Javadocs/package prose, focused documentation, glossary
impact, planning evidence, and documentation validation without repeating successful Java tests
unless executable behavior changes or a concrete stale-evidence risk is recorded.

Update this task's evidence, notes, completion summary, and status only after every acceptance
criterion and the documentation pass succeed.
```

## Local decisions

- Keep Phase 2 in new complete-plan request/result/collaboration types. Do not reinterpret or
  widen task-0001 Phase-1 public types.
- Use one generic opaque reference type in the correctness collaboration. The tool retains it only
  transiently and never asks it for equality, bytes, diagnostics, or serialization.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Use a nested typed mismatch failure because mismatch is a clean callback outcome that later
  Engine composition must distinguish from execution/cleanup failure without receiving payloads.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Treat `SESSION` as a strict no-filesystem branch while still constructing the same in-memory
  compact record shape used by `PERSISTENT` results.
- Keep one cohesive cache rather than splitting persistence into an unconsumed task. The fake
  persistent producer makes the generic format executable without overstating CPU capability.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence); [producer glossary](../../../../glossary.md#term-producer); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Require a decoder-authenticated hit and later caller-owned fresh preparation. Encoded bytes are
  never a prepared artifact.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Leave one path below the 18-path ceiling unallocated so documentation can choose a glossary
  update or no change without silently expanding implementation scope.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Known limitations

- The task supplies generic tools capability and synthetic integration only. No current public
  Engine method invokes it.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- Current CPU 0010J is session-scoped, so it will always take the no-filesystem fresh-measurement
  branch when later adapted.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- The first policy is exact canonical represented-byte comparison and the sole objective is
  minimum median elapsed nanoseconds.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [objective glossary](../../../../glossary.md#term-objective); [median glossary](../../../../glossary.md#term-median))</span>
- The first transaction tunes one caller-supplied complete batch. It does not combine candidates
  from multiple architecture owners, partitions, or backends.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [backends glossary](../../../../glossary.md#backend))</span>
- Cache inspection/migration, persistent CPU identity, relaxed correctness, and executable
  serialization remain deferred.<span data-glossary-navigation="true"> ([Cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [serialization glossary](../../../../glossary.md#term-serialization))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Stabilized executable validation was reused without rerunning Java tests because the independent
  documentation pass changed no executable Java token: <span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token))</span>`./gradlew :tools:tuning:test` passed 8
  suites and 51 tests with zero failures, zero errors, and zero skipped tests; the focused
  complete-plan run passed 4 suites and 27 tests. Public-shape/reflection and <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [shape glossary](../../../../glossary.md#shape))</span>`javap` inspection
  also passed.
- Final `./gradlew :tools:tuning:javadoc` passed. Generated package and public-type pages were
  inspected for the Phase-1/Phase-2 boundary, ownership, bounds, callback order, lifecycle,
  nullability, snapshotting, units, return semantics, and failure rendering; the final run emitted
  no Javadoc errors or warnings.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [callback glossary](../../../../glossary.md#term-callback); [lifecycle glossary](../../../../glossary.md#lifecycle); [nullability glossary](../../../../glossary.md#term-nullability); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Targeted Markdown validation passed for all five changed Markdown documents: every local target
  and fragment resolved, headings were unique, and backtick/tilde fences were balanced. LF/final-
  newline and trailing-whitespace checks passed.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Exact-scope validation found 17 changed paths: eight production/Javadoc paths, four unchanged
  test paths from the implementation, two explanatory documentation paths, and exactly three
  planning paths. Every path is permitted by this task and the 18-path ceiling is preserved.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [planning glossary](../../../../glossary.md#term-planning))</span>
- Non-comment source comparison confirmed that the documentation pass changed no executable Java
  token. `git diff --check` and `git diff --cached --check` passed, and the staging area remained
  empty.

## Implementation notes

- Added the six specified public complete-plan types plus package-private `ModelPlanCacheFile` in
  the existing tuning package, without another public top-level type or package. The request owns
  bounded immutable fingerprints and distinct Phase-2 ceilings; backend candidates and decisions
  remain opaque Prepare-role values.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [backend glossary](../../../../glossary.md#backend); [opaque glossary](../../../../glossary.md#term-opaque); [Prepare glossary](../../../../glossary.md#prepare))</span>
- The measured transaction snapshots and validates the complete stable candidate list, preflights
  checked <span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction))</span>`N * (1 + W + S)`, completes all exact correctness actions before timing, measures every
  candidate with the configured warmups and odd sample count, selects the smallest integer-middle
  median with encounter-order ties, and returns a decision-present exact handoff, compact record,
  and separate rich evidence.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [median glossary](../../../../glossary.md#term-median); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- `SESSION` performs no filesystem [operation](../../../../glossary.md#operation). `PERSISTENT` validates the complete bounded,
  canonical, checksummed cache before enumeration; authenticates hits through the current
  producer decoder; and publishes a miss only after bounded encoding and an equal compatible
  decode round trip. Cache entries contain compact opaque decision/winner state and a summary,
  never raw samples, correctness references, publication bytes, or executable state.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [cache glossary](../../../../glossary.md#term-cache); [producer glossary](../../../../glossary.md#term-producer); [round trip glossary](../../../../glossary.md#term-round-trip); [publication glossary](../../../../glossary.md#term-publication))</span>
- Finalized Javadocs and package documentation on all eight production paths. The benchmarking
  guide now explains both implemented generic transactions, caller/producer ownership,
  correctness-before-timing, the exact execution formula, session versus persistent behavior,
  compact cache versus rich evidence, and the still-planned Engine/Config Phase-2 composition.<span data-glossary-navigation="true"> ([benchmarking glossary](../../../../glossary.md#term-benchmarking); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Updated the reusable glossary terms because their former text incorrectly described generic
  complete-plan correctness, measurement, and model-plan caching as wholly future work. The
  correction explicitly preserves current CPU <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu))</span>`SESSION` scope and the absence of public Engine
  Phase-2 integration.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Independent review found no executable defect or architectural uncertainty. No change was
  needed in `ARCHITECTURE.md`, an ADR, the focused performance/tuning or Runtime/Prepare/backend
  architecture explanations, Config/Engine/CPU/Prepare plans or contracts, architecture tests,
  backend conformance, integration tests, Gradle/build files, or other modules. The task realizes
  the existing tools boundary and changes no dependency or architecture rule.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

Completed the bounded generic Phase-2 [transaction](../../../../glossary.md#term-transaction) and its compact [model](../../../../glossary.md#term-model)-plan [cache](../../../../glossary.md#term-cache) within the
planned 17-path [scope](../../../../glossary.md#term-scope). Production/[Javadoc](../../../../glossary.md#term-javadoc) changes are the package summary plus
`BackendCompletePlanTuning`, `CompleteCandidateMeasurement`, `CompletePlanCorrectness`,
`CompletePlanTuning`, `CompletePlanTuningRequest`, `CompletePlanTuningResult`, and
`ModelPlanCacheFile`. The four specified complete-plan test paths remain the executable evidence;
the documentation pass did not modify them. Explanatory changes are `benchmarking.md` and the
reusable glossary corrections; [planning](../../../../glossary.md#term-planning) changes are this task, the tuning master plan, and the
roadmap.

The generic tools capability is complete and validated. The next ordered tuning work is task 0003
cache/plan inspection, which remains Draft without a detailed specification. Before public
Phase-2 use, separately planned Config extension and [Engine](../../../../glossary.md#term-engine)-owned [CPU](../../../../glossary.md#term-cpu) adaptation must provide the
new request inputs, join CPU 0010J with Engine 0008A, freshly [prepare](../../../../glossary.md#prepare) the selected production
decision, translate evidence, and apply fallback. Persistent CPU compatibility remains deferred.
No unresolved issue remains in task 0002.

Status: Complete
