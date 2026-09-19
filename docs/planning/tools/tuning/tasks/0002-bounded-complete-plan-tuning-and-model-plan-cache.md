# Task 0002: Bounded Complete-Plan Tuning and Model-Plan Cache

## Status

Ready

## Goal

Implement one bounded, generic complete-plan tuning transaction in `tools/tuning`. A caller that
already composes the relevant backend and execution lifecycle supplies one complete opaque Prepare
candidate batch, backend-owned identity/codec operations, exact correctness actions, and one fresh
complete-candidate execution action. The tool validates the whole transaction before execution,
runs correctness before timing, selects deterministically, returns the selected typed handoff plus
a compact selected-plan record and separate rich evidence, and conditionally reuses or atomically
publishes an explicit model-plan cache.

This task is the tools-only Phase-2 consumer foundation. It does not compose current CPU or Engine
types. Complete CPU 0010J is the first truthful producer and Complete Engine 0008A is the
correctness owner, but a later Engine task adapts both collaborations.

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

The fresh post-Engine-0008A source audit found no missing contract, inversion, or architecture
decision:

1. `tools/tuning` can own the transaction using its existing public Prepare dependency and
   tool-local generic callbacks. It needs no Engine, Runtime, Config, CPU, or backend-internal
   import.
2. `CpuCompletePlanTuning` already supplies immutable `CandidateBatch`, `Candidate`,
   `SelectedDecision`, compatibility, identity, codec, and fresh trial/selected preparation
   operations. A later caller can adapt them without the tool inspecting a CPU value.
3. `RepresentativeExecutionSession` and `RepresentativePlanCorrectness` keep publication bytes,
   Runtime leases, descriptor preflight, aggregate byte enforcement, association, poisoning, and
   cleanup package-private. A later Engine adapter can present only one opaque reference and
   `MATCH`/`MISMATCH` to this task's generic correctness collaboration.
4. `CpuCompletePlanTuning.prepareTrial(...)` and the representative session prove that one caller
   action can freshly prepare and execute a complete candidate on each invocation. The tool needs
   to own only invocation count and `System.nanoTime` measurement.
5. One correctness action per candidate fixes exact preflight arithmetic at
   `N * (1 + W + S)`. Positive candidate and total-execution ceilings, non-negative `W`, positive
   odd `S`, and checked addition/multiplication can all be validated before execution.
6. CPU 0010J truthfully reports `SESSION`, so the generic workflow can skip lookup/publication and
   still return an in-memory record and evidence. Synthetic `PERSISTENT` tests can validate the
   generic cache without claiming current CPU persistence.
7. A compact cache needs only opaque identities, encoded selected decisions, and compact timing
   summaries. Backend-compatible decoding authenticates a hit; fresh selected preparation remains
   outside this task. No executable or `PreparedExecution` is serialized.
8. Config 0006A's four Phase-1 budget meanings remain untouched. A later Config task adds distinct
   Phase-2 fields after this consumer stabilizes, and a later Engine task owns CPU/Engine
   adaptation, fresh production preparation, evidence translation, and fallback.

## Scope

- Add one stateless public complete-plan tuning entry in
  `io.github.pho001.synaptik.tools.tuning`.
- Add an immutable tool-local request with:
  - non-empty versioned model, representative-profile, and target fingerprints;
  - objective `MIN_MEDIAN_ELAPSED_NANOS`;
  - correctness policy `EXACT_CANONICAL_BYTES`;
  - one non-empty versioned opaque constraint/policy identity;
  - a positive maximum plan-candidate count;
  - a non-negative warmup count;
  - a positive odd timed-sample count;
  - a positive maximum total plan-execution count;
  - a non-negative maximum aggregate correctness-byte count carried to the caller collaboration;
  - one explicit model-plan-cache `Path`; and
  - one exact decision-empty `BackendPartitionTuningHandoff<C, D>`.
- Add one typed generic backend collaboration over Prepare's method-free batch and decision roles.
  It supplies only the complete candidate list, opaque producer/codec identity, compatibility and
  reuse scope, candidate identities, decision construction, decision encoding, and compatible
  decision decoding. All returned collections and bytes are snapshotted or immutable at the tool
  boundary.
- Add one generic exact-correctness collaboration. Its first operation captures an opaque
  caller-owned reference for the first candidate under the request's maximum aggregate
  correctness-byte bound; its second compares each later candidate and returns only `MATCH` or
  `MISMATCH`. The reference is a generic value that is never returned, persisted, inspected,
  compared, or exposed in evidence by the tool.
- Add one generic complete-candidate measurement action. Each invocation performs one complete
  freshly prepared and freshly executed candidate outside the tool. The tool owns only invocation
  count and monotonic elapsed-time measurement.
- Obtain compatibility before filesystem work. For `SESSION`, perform no filesystem query,
  create, read, write, temporary-file, move, or directory-force operation. Always run fresh
  correctness and measurement, then return an in-memory compact selected-plan record.
- For `PERSISTENT`, load and completely validate the explicit cache before candidate enumeration
  or execution. An exact compatible entry is a hit only when the backend decoder returns a
  compatible decision. A hit performs zero correctness, warmup, or timed executions.
- On a miss, snapshot and validate the complete ordered candidate set and every stable unique
  identity. Let `N` be its size. Before the first correctness call, checked-compute
  `perCandidate = 1 + W + S` and `total = N * perCandidate`, then require `N` and `total` not to
  exceed their positive ceilings.
- Invoke correctness in candidate encounter order. The first candidate captures the opaque
  reference; each later candidate compares against it. All `N` actions must complete and match
  before the first warmup or timed action. A mismatch throws one unambiguous typed task-local
  failure with no payload bytes or backend value.
- After correctness succeeds, execute every candidate for exactly `W` warmups and `S` timed
  samples. Use `System.nanoTime` through one package-private clock seam, validate monotonic
  non-negative elapsed values, calculate the integer-middle median, choose the smallest median,
  and retain the first encountered candidate on a tie.
- Construct the selected backend decision once. Return a decision-present typed Prepare handoff,
  one immutable compact selected-plan record, and separate immutable rich evidence. Evidence
  retains ordered correctness outcomes, raw timed samples for misses, summaries, identities,
  source, and winner; compact state retains no raw samples.
- For a persistent miss, require a bounded non-empty encoded decision and compatible equal codec
  round trip before publication. Publish no cache entry for `SESSION`, incompatible decoding,
  mismatch, or any failure.
- Construct no externally visible record, evidence, or result until every required action and any
  persistent publication succeeds. Propagate caller execution/cleanup and cache failures through
  their exact declared boundary; do not convert them to correctness mismatch or move fallback
  policy into tools.
- Add one package-private model-plan cache implementation in the same style as task 0001: bounded
  versioned canonical big-endian binary data, length prefixes, fixed numeric tags, deterministic
  unsigned-key ordering, whole-file SHA-256 checksum, complete parsing before use, duplicate-key
  rejection, forced same-directory unique temporary file, atomic replacement only, best-effort
  directory force, and temporary cleanup.
- Cache keys include producer/decision-codec identity; model, representative-profile, and target
  fingerprints; backend compatibility schema/bytes; objective; correctness and constraint/policy
  identity; and sampling fields. Entries contain only selected candidate identity, encoded
  decision, and compact timing summary.
- Preserve all task-0001 behavior and public meanings. Share only narrowly package-private cache
  publication/checksum mechanics if implementation proves that reuse is clearer and safer than
  duplication; do not change Phase-1 public request, result, collaboration, or cache format.
- Finalize Javadocs, package documentation, focused tuning documentation, glossary impact, and
  planning evidence in the required separate documentation-focused context.

## Out of scope

- importing or adapting Engine, Runtime, Config, CPU, OpenBLAS, or backend-internal types in
  `tools/tuning`
- public Engine Phase-2 integration, representative input binding, publication copying, reference
  byte ownership, result cleanup, fallback policy, or selected production preparation
- changing Config 0006A or reinterpreting its Phase-1 maximum-miss, candidates-per-miss, warmup,
  or timed-sample fields as Phase-2 values
- Compiler graph-transformation candidates, Planning ownership/partition alternatives, multiple
  partitions, mixed backends, or more than the complete opaque batch supplied by the caller
- persistent CPU complete-plan reuse or a claim that CPU 0010J has cross-session compatibility
- local workload extraction, occurrence weighting, Phase-1 route ranking, or any repeat of
  task-0001 candidate search
- tolerance, relative/absolute error, ULP policies, NaN normalization, relaxed math, stochastic
  comparison, adaptive sampling, confidence claims, outlier removal, wall-clock cutoffs, parallel
  measurement, or objectives other than minimum median elapsed nanoseconds
- serializing `PreparedExecution`, an executable, Runtime state, Engine reference, publication
  bytes, backend candidate object, or live resource
- cache migration, recovery, locking across concurrent writers, eviction, remote caches,
  compression, encryption, signatures, or cache inspection UI
- changes to source or tests outside `tools/tuning`, module dependencies, Gradle, architecture
  tests, backend conformance tests, integration tests, or authoritative architecture/ADRs

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/prepare`, Concrete backend modules, Performance evidence and optimization tooling,
  `modules/engine`, and Dependency rules
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Tuning master plan](../master-plan.md)
- [Task 0001 workload tuning and cache](0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)
- [Prepare 0004 opaque handoff](../../../modules/prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)
- [CPU 0010J complete-plan producer](../../../backends/cpu/tasks/0010j-supported-complete-plan-candidate-and-decision-producer.md)
- [Engine 0008A correctness oracle](../../../modules/engine/tasks/0008a-representative-complete-plan-correctness-oracle.md)
- [Config 0006A current request](../../../modules/config/tasks/0006a-model-autotuning-request-configuration.md)

## Architecture constraints

- Tuning owns bounded measurement, explicit compact cache coordination, deterministic comparison,
  selection, and rich evidence. It does not own the meaning or legality of any candidate.
- The concrete producer owns the complete candidate set, stable order, identities, compatibility,
  reuse scope, decisions, codecs, and later fresh preparation. Shared Prepare remains method-free
  opaque transport.
- The caller owns correctness semantics, the opaque reference, maximum-correctness-byte
  enforcement, execution resources, preparation, Runtime invocation, publication materialization,
  and cleanup. The tool sees only `MATCH` or `MISMATCH`.
- Correctness actions are separate from timing. A correctness execution is never reused as a
  warmup or sample.
- Runtime receives only a freshly prepared selected recipe later. It never receives a tuning
  request, candidate, decision, cache entry, reference, or evidence and performs no selection.
- `SESSION` is a strict no-filesystem mode, not a cache miss that happens to avoid publication.
- Persistent decoding authenticates a compact hit; it never grants backend eligibility. Later
  composition must freshly prepare the decoded decision.
- No dependency direction or architecture rule changes. If implementation needs an inward-module
  change, concrete-backend import, public Engine oracle, widened Prepare role, or executable
  serialization, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.tools.tuning` — the existing deliberate small public tuning surface
  and package-private persistence mechanics.
- `io.github.pho001.synaptik.prepare.analysis` — existing method-free batch/decision roles and
  exact-partition handoff, used only through bounded generics.

Packages added or changed:

- No Java package is added.
- The tuning root gains one cohesive complete-plan transaction beside the unchanged workload
  transaction.

Type placement:

- `io.github.pho001.synaptik.tools.tuning.CompletePlanTuning` — stateless transaction entry,
  package-private clock seam, and nested typed mismatch failure.
- `io.github.pho001.synaptik.tools.tuning.CompletePlanTuningRequest` — immutable tool-local
  fingerprints, policies, Phase-2 bounds, exact handoff, and cache path.
- `io.github.pho001.synaptik.tools.tuning.BackendCompletePlanTuning` — typed caller collaboration
  for complete candidates, opaque producer/compatibility/identity values, decisions, and codec.
- `io.github.pho001.synaptik.tools.tuning.CompletePlanCorrectness` — typed opaque-reference capture
  and exact two-valued comparison collaboration.
- `io.github.pho001.synaptik.tools.tuning.CompleteCandidateMeasurement` — one complete fresh
  candidate action used only for warmups and timed samples.
- `io.github.pho001.synaptik.tools.tuning.CompletePlanTuningResult` — selected typed handoff,
  compact selected-plan record, and separate rich evidence.
- `io.github.pho001.synaptik.tools.tuning.ModelPlanCacheFile` — package-private bounded parsing,
  canonical encoding, checksum, merge, and atomic publication.

Tests mirror the production package for clock and cache seams. No `api`, `cache`, `internal`,
`util`, `common`, backend, Engine, or Runtime package is added.

## Affected files

Expected production/Javadoc paths:

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

Expected documentation and planning paths:

- `docs/developer-guide/benchmarking.md` — distinguish the implemented Phase-1 transaction, this
  complete-plan transaction, and still-later Engine composition
- `docs/glossary.md` only if implementation makes an existing reusable complete-plan or
  model-plan-cache term incomplete; otherwise record a reasoned no-change conclusion
- this task
- `docs/planning/tools/tuning/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a concrete contradiction is found: `ARCHITECTURE.md`, ADRs,
the two focused architecture explanations, Config/Engine/CPU/Prepare plans and contracts, all
task-0001 source/tests, other modules, Gradle files, architecture tests, backend conformance tests,
and integration tests.

## Maximum scope

At most 18 paths:

- 8 tuning production/Javadoc paths;
- 4 tuning test paths;
- at most 2 explanatory documentation paths; and
- exactly 3 planning paths.

The eighteenth path is intentionally unallocated. The capability remains cohesive because
validation, correctness ordering, timing, selection, cache authentication, compact record, rich
evidence, and failure-before-publication are one observable transaction. If another production
type, package, module dependency, build path, explanatory path beyond the allowance, or more than
18 total paths is required, stop and return the task to planning.

## Acceptance criteria

1. The six public top-level types and one package-private cache type listed under Package impact
   exist with no additional public top-level production type or package.
2. Public signatures use only JDK values, tool-local immutable values, and Prepare's method-free
   roles/handoff. Source and dependency checks find no Engine, Runtime, Config, CPU, provider, or
   backend-internal import or dependency.
3. Request construction snapshots all bytes/collections, retains the exact handoff references,
   requires an absent input decision, and validates every identity, policy, Phase-2 bound, and
   explicit path without filesystem access.
4. Backend collaboration results are non-null, immutable/snapshotted, stable, complete, bounded,
   and identity-unique. The tool performs no reflection, downcast, raw or unchecked cast, route
   switch, backend-field access, generic map, or string dispatch.
5. `SESSION` performs zero filesystem operations and always performs fresh correctness and timing
   on a miss-like transaction. It returns a compact in-memory selected-plan record and rich
   evidence but publishes no file.
6. `PERSISTENT` completely validates the cache before candidate enumeration or execution.
   Unsupported schema, oversize, truncation, checksum failure, duplicates, noncanonical order,
   structural inconsistency, or trailing bytes fail without execution or mutation.
7. An exact cache hit invokes compatible backend decoding, constructs the selected handoff, and
   performs zero candidate enumeration, correctness, warmup, and timed actions. Later fresh
   preparation remains caller-owned and is not invoked by this task.
8. A missing entry or backend-rejected decision is a miss. No stale or incompatible value grants
   eligibility.
9. Miss validation proves positive `N`, `N <= maximumPlanCandidates`, checked
   `N * (1 + W + S)`, and the positive total-execution ceiling before the first correctness call.
   Overflow or over-budget input performs zero actions and publishes nothing.
10. Deterministic tests prove exactly `N` correctness actions in encounter order, first-candidate
    capture, later comparison, and completion of all correctness actions before any measurement.
11. A mismatch produces the exact task-local typed failure only after the caller action returns
    `MISMATCH`; it runs no warmup/sample, returns no result/evidence/record, and publishes no cache.
    Caller execution and cleanup failures propagate without being misreported as mismatch.
12. Each candidate receives exactly `W` warmups and `S` timed actions after correctness succeeds.
    Tests prove monotonic timing, elapsed validation, raw sample retention, integer-middle median,
    lowest-median selection, and first-encounter ties.
13. The selected decision is constructed once. The result contains the decision-present typed
    handoff, compact record, and separate immutable evidence; no opaque reference, candidate
    object, publication byte, Runtime value, or executable escapes through record/evidence.
14. Persistent miss encoding is bounded and must decode compatibly to an equal decision before
    publication. Cache entries retain only opaque key/decision/winner bytes and compact summaries;
    rich correctness and raw timing evidence remains in memory only.
15. Canonical bounded cache bytes, duplicate safety, deterministic ordering, whole-file checksum,
    same-directory forced temporary publication, atomic replacement, prior-file preservation on
    injected failures, and cleanup are all tested with a synthetic `PERSISTENT` producer.
16. Workload tuning API, behavior, tests, and cache format remain unchanged. Any shared private
    mechanics preserve task 0001 byte-for-byte observable behavior.
17. Package/Javadocs explain ownership, bounds, units, nullability, snapshots, generic association,
    lifecycle, failure, `SESSION` no-I/O, persistent authentication, and current-versus-planned
    behavior. A separate clean documentation-focused context finalizes them and reviews glossary
    impact.
18. Focused/final tuning tests, tuning Javadoc/rendered inspection, Markdown validation, exact
    scope/public-shape/import/status checks, empty staging, and both diff checks pass.

## Tests / validation

Run focused tests while implementing. After executable Java stabilizes, run once:

```bash
./gradlew :tools:tuning:test
```

The four new test owners use synthetic method-free Prepare values, an opaque reference, a
deterministic clock, a filesystem-counting `SESSION` path, and a fake `PERSISTENT` producer. They
cover validation/order/budget/correctness/mismatch, SESSION no-I/O, persistent hit/miss/corrupt/
atomic publication, deterministic selection/evidence, public shape/imports, and unchanged
task-0001 behavior through the complete existing module suite.

Documentation pass:

```bash
./gradlew :tools:tuning:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

Render and inspect every new public declaration and package summary. Validate changed Markdown
local targets/anchors, unique headings, balanced backtick/tilde fences, LF/final newlines,
trailing whitespace, terminology, exact scope, package placement, public shape, imports, and
synchronized task/master/roadmap status.

Repository-wide, architecture, backend-conformance, integration, CPU, Engine, and Config Java
tests are unnecessary for this task. It changes one tool module, no dependency/build boundary,
no backend behavior, and no public Engine workflow. The later Engine Phase-2 composition
checkpoint owns cross-module and end-to-end validation; CI remains the repository-wide gate.

## Dependencies

- Tools/tuning 0001 — Complete; establishes the module, monotonic timing precedent, cache safety,
  deterministic median selection, evidence separation, and explicit persistent path policy.
- Prepare 0004 — Complete; supplies the unchanged method-free opaque batch/decision roles and
  typed handoff.
- CPU 0010J — Complete evidence for one truthful bounded producer and exact Phase-1 reuse. It is
  not imported or exercised by this task.
- Engine 0008A — Complete evidence for an adaptable opaque exact correctness primitive with
  Engine-owned publication bytes, byte limits, lifecycle, and cleanup. It is not imported or
  exercised by this task.
- JDK clock, filesystem, channel, checksum/digest, and atomic-move APIs already available to the
  module.

Caller-supplied generic collaborations are task inputs exercised by synthetic tests, not absent
module prerequisites. Config extension and Engine/CPU composition are follow-ups, not dependencies.

## Follow-up tasks

- Add one Config Phase-2 extension only after this consumer is implemented. It adds distinct
  maximum-plan-candidate, maximum-total-plan-execution, maximum-correctness-byte, policy identity,
  target identity, and explicit model-plan-cache inputs without changing Config 0006A's four
  Phase-1 budget meanings.
- Add one later Engine-owned CPU composition after both consumer contracts stabilize. It adapts
  CPU 0010J and Engine 0008A, passes the exact Phase-1 selection, freshly prepares every action,
  stops on mismatch, freshly prepares the authenticated winner for production, translates rich
  evidence, and applies the existing public safe-fallback policy.
- Tuning 0003 remains Draft for cache/plan inspection after both artifact formats are stable.
- Persistent CPU compatibility, Compiler graph alternatives, Planning ownership alternatives,
  multiple partitions, mixed backends, tolerance policies, and executable serialization remain
  separate future work.

## Documentation, Javadoc, and glossary impact

Apply General plus API/Javadoc style to the Java contracts, General plus Developer-guide style to
the focused benchmarking/tuning explanation, and General plus Planning style to this task and
status synchronization. Use Example style only for a new multi-step example and label planned
Engine adaptation as planned.

Review `docs/glossary.md` for the existing complete-plan candidate, candidate batch, selected
tuning decision, and tuning artifact entries. Change it only if the implementation makes those
reusable definitions incomplete. A no-change conclusion must explain why the task composes
existing terms rather than changing their meaning.

## Architecture impact

Expected impact: None.

The task realizes the already authorized tools measurement/cache ownership through the current
opaque Prepare boundary. It changes no module edge, ownership rule, lifecycle order, or public
Engine behavior. If implementation requires an authoritative change, stop and report it; do not
edit `ARCHITECTURE.md`, an ADR, or architecture tests within this task.

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
  transiently and never asks it for equality, bytes, diagnostics, or serialization.
- Use a nested typed mismatch failure because mismatch is a clean callback outcome that later
  Engine composition must distinguish from execution/cleanup failure without receiving payloads.
- Treat `SESSION` as a strict no-filesystem branch while still constructing the same in-memory
  compact record shape used by `PERSISTENT` results.
- Keep one cohesive cache rather than splitting persistence into an unconsumed task. The fake
  persistent producer makes the generic format executable without overstating CPU capability.
- Require a decoder-authenticated hit and later caller-owned fresh preparation. Encoded bytes are
  never a prepared artifact.
- Leave one path below the 18-path ceiling unallocated so documentation can choose a glossary
  update or no change without silently expanding implementation scope.

## Known limitations

- The task supplies generic tools capability and synthetic integration only. No current public
  Engine method invokes it.
- Current CPU 0010J is session-scoped, so it will always take the no-filesystem fresh-measurement
  branch when later adapted.
- The first policy is exact canonical represented-byte comparison and the sole objective is
  minimum median elapsed nanoseconds.
- The first transaction tunes one caller-supplied complete batch. It does not combine candidates
  from multiple architecture owners, partitions, or backends.
- Cache inspection/migration, persistent CPU identity, relaxed correctness, and executable
  serialization remain deferred.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
