# Task 0001: Exact/Default Model-Guided Workload Tuning and Reusable Cache

## Status

Complete

## Goal

Implement the first operational local-workload phase of model autotuning for the completed CPU
0010E and Prepare 0004 boundary without giving shared tooling knowledge of CPU candidate fields.
The workflow accepts complete backend-owned candidate batches through Prepare's method-free
opaque roles, reuses compatible results, measures cache misses only, selects a deterministic
winner under one explicit elapsed-time objective and bounded sampling policy, returns selected
backend-owned decisions, atomically persists a compact workload cache, and returns separate rich
in-memory evidence.

The repository has no Engine facade or general public model-execution path. This task therefore
defines the smallest operational caller-supplied cold collaboration. The caller that already
composes a concrete backend supplies:

- stable model and representative-profile fingerprints;
- occurrences and their exact `BackendPartitionTuningHandoff<C, D>` values;
- candidate enumeration from each complete opaque batch;
- canonical local and persistent compatibility identities;
- stable candidate identities;
- selected-decision construction and decision encoding/compatible decoding; and
- one complete comparable execution action for each candidate.

`tools/tuning` supplies:

- immutable tool-local request validation;
- identical-workload deduplication with occurrence weights and contexts retained;
- cache loading before any measurement;
- miss-only warmup, monotonic timing, sample validation, and bounded measurement;
- minimum-median elapsed-time comparison with candidate encounter order as the exact tie break;
- compatible decision reuse and fresh decision selection;
- compact bounded cache encoding, corruption/incompatibility rejection, and atomic persistence;
  and
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
their ownership, or their typed generic association.

## Scope

- Add the public `WorkloadTuning` stateless entry, `WorkloadTuningRequest`,
  `BackendWorkloadTuning`, `ColdCandidateMeasurement`, and `WorkloadTuningResult` under
  `io.github.pho001.synaptik.tools.tuning`.
- Delete the existing public placeholder `TuningToolModule` so the completed tuning API exposes
  exactly those five public production types.
- Make `tools/tuning` expose its Prepare dependency as Gradle `api` because its public generic
  contracts use `BackendTuningCandidateBatch`, `BackendTuningDecision`, and
  `BackendPartitionTuningHandoff`. Do not add a concrete-backend, Engine, or Runtime dependency.
- Keep all request inputs tool-local. `WorkloadTuningRequest<C, D>` contains exactly:
  - a non-empty stable `ModelFingerprint` byte sequence with a positive caller schema version;
  - a non-empty stable `ProfileFingerprint` byte sequence with a positive caller schema version;
  - an ordered non-empty immutable list of `Occurrence<C, D>` values;
  - objective `MIN_MEDIAN_ELAPSED_NANOS`;
  - one `Budget`; and
  - one explicit workload-cache `Path`.
- Define `Occurrence<C, D>` as one exact handoff, one positive occurrence weight, and one
  non-empty opaque context fingerprint. Require the input handoff's selected decision to be empty;
  output handoffs carry the selected backend decision. Retain exact partition and batch
  references and snapshot only request-owned collections and byte sequences.
- Define `Budget` with positive maximum distinct cache misses, positive maximum candidates per
  miss, non-negative warmup count, and a positive odd timed-sample count. The task has no wall-
  clock cutoff: it validates the complete deduplicated work and every complete candidate count
  against the budget before executing the first candidate, so machine-speed variation cannot
  change which candidates are compared.
- Define `WorkloadCompatibility` as a positive backend-owned key-schema version, a non-empty
  opaque canonical byte sequence, and `SESSION` or `PERSISTENT` reuse scope. The tool compares
  exact values only. It neither parses nor constructs backend compatibility content.
- Define `CandidateIdentity` as a non-empty opaque canonical byte sequence. Require identities to
  be unique within the complete ordered candidate list. The tool compares exact values only and
  uses encounter order, not lexical identity, to break an equal-median tie.
- Deduplicate occurrences by exact `WorkloadCompatibility`, objective, and timed sampling policy.
  Sum weights with checked arithmetic and retain every original occurrence/context in request
  order. Session identities may reuse a result only within the current invocation. Persistent
  identities may additionally reuse compatible file entries across invocations and models.
- Load and validate the entire explicit cache before candidate enumeration or measurement.
  Reject malformed, oversized, truncated, checksum-invalid, duplicate-key, unsupported-artifact-
  schema, or structurally inconsistent files with an `IOException`; preserve the original file
  and perform no measurement or write after rejection.
- Treat a well-formed entry with no exact key/objective/sampling match or whose opaque decision
  `decodeCompatibleDecision` returns empty as an ordinary cache miss. Never ask the tool to
  interpret why the backend rejected it.
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
  persist nothing from the failed invocation.
- Compute each candidate's median from the already bounded odd sample count without floating-
  point conversion. Select the smallest median; retain the first candidate on a tie. Call
  `selectedDecision` exactly once for the winner and require a non-null decision.
- Call `encodeDecision` only for a persistent miss winner. Require a non-null, non-empty, bounded
  byte sequence. Verify round-trip compatibility by decoding it against the same batch and
  requiring an equal decision before making it eligible for cache publication.
- Return selected handoffs for every original occurrence in request order. Identical occurrences
  reuse the same exact selected-decision reference from the one deduplicated result.
- Return immutable rich evidence with model/profile fingerprints, objective/budget, occurrence
  contexts and weights, cache-hit/miss source, candidate identities, raw elapsed samples for
  misses, compact min/median/max/sample-count summaries, winner identity, and rejection/failure
  only through the thrown exception. Evidence is never written into the compact workload cache.
- Add one package-private `WorkloadCacheFile` implementation. Use a deterministic bounded binary
  encoding with a fixed magic, artifact schema version one, entry count, length-prefixed opaque
  compatibility/decision/winner bytes, fixed numeric objective and sampling fields, compact
  min/median/max/sample-count summary, deterministic unsigned compatibility-key order, and a
  SHA-256 checksum over all preceding bytes. Do not use Java object serialization, JSON maps,
  reflection, class names, enum names, or string dispatch.
- Bound the cache file, entry count, and every byte-array length before allocation. Use explicit
  big-endian fixed-width integers and reject trailing bytes, numeric overflow, and invalid
  summary ordering.
- Persist only when at least one persistent miss produced a new compatible entry. Merge with
  retained compatible and unrelated well-formed entries, write a uniquely named sibling
  temporary file through a `FileChannel`, force file contents, publish with
  `ATOMIC_MOVE` plus `REPLACE_EXISTING`, and best-effort force the parent directory where the JDK
  and platform support it. If atomic move is unsupported or publication fails, leave the prior
  cache intact as far as the filesystem permits, clean up the temporary file, and propagate the
  failure. Do not silently fall back to a non-atomic move.
- Finalize public and package Javadocs, the focused tuning/Prepare explanatory boundary, and the
  affected tuning glossary entries in the mandatory separate clean documentation context.

## Out of scope

- a public Engine facade, model compilation, graph preparation, prepared execution construction,
  caller-input binding, published-output access, or a general end-to-end model runner
- any dependency on or import from `backends/cpu`, the OpenBLAS provider, another concrete
  backend, private backend internals, or technically public unsupported-internal CPU packages
- direct construction, parsing, reflection, downcast, matching, or interpretation of
  `CpuOpenBlasTuningBatch`, `CpuOpenBlasTuningDecision`, candidate fields, CPU routes, thread
  counts, representations, resources, hardware, provider qualification, or workload signatures
- changing Prepare's method-free marker roles or handoff, CPU 0010E values, backend candidate
  generation, compatibility rules, decision matching, heuristic fallback, analysis, finalization,
  generated-artifact cache, executable behavior, or exact/default eligibility
- BFLOAT16, relaxed or fast-math candidates, Config 0004–0006A, provider 0004, CPU 0010D1, or a
  Config facade; the initial eligible batch remains exact/default FLOAT32/FLOAT64
- graph/plan candidate tuning, fusion/ownership/layout/materialization search, task 0002, a model-
  plan cache, or prepared-executable serialization
- candidate correctness validation, semantic comparison, route-specific warmup, outlier removal,
  confidence intervals, adaptive sampling, parallel measurement, cancellation, wall-time cutoffs,
  energy/memory objectives, multi-objective ranking, or performance-proof claims
- report-only `tools/benchmarks`, benchmark cache population, Runtime profiling, trace DTOs, or
  Runtime hot-path measurement, selection, cache access, mutation, or graph inspection
- cache migration, remote/shared caches, locking across concurrently writing processes, eviction,
  expiry, compression, encryption, signatures, access-control management, or recovery of corrupt
  files
- hidden global state, static mutable caches, registry, service locator, plugin discovery,
  callback discovery, raw `Object`, unchecked casts, generic maps, reflective annotations,
  arbitrary parameter bags, Java serialization, or string-keyed dispatch
- changes to Architecture/ADRs, Model, Backend Contract, Config, Planning, Compiler, Runtime,
  Prepare production, Engine, concrete backends/providers, architecture tests, backend conformance,
  integration tests, shared build logic, or another task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/prepare`, Concrete backend modules, Performance evidence and optimization tooling,
  Runtime service locator, Prepare lifecycle, and Dependency rules
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Tuning master plan](../master-plan.md)
- [CPU 0010E candidate and decision contract](../../../backends/cpu/tasks/0010e-float32-float64-openblas-tuning-candidates-and-compatible-decisions.md)
- [Prepare 0004 opaque handoff](../../../modules/prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)
- [Config master plan](../../../modules/config/master-plan.md)
- [Engine master plan](../../../modules/engine/master-plan.md)

## Architecture constraints

- Tuning coordinates measurement, explicit cache state, comparison, selection, and evidence. It
  must not own graph semantics, Planning ownership policy, Prepare orchestration, backend lowering,
  route vocabulary, candidate generation, compatibility meaning, executable construction, or
  Runtime behavior.
- The concrete backend remains the sole owner of complete valid candidate contents, their stable
  order, canonical compatibility, candidate identity, decision construction/encoding/decoding,
  and decision validation during later analysis. The explicit caller collaboration bridges these
  owned operations without exposing their fields to the tool.
- Prepare's batch and decision roles remain method-free. The generic handoff remains an exact-
  partition opaque transport and gains no enumeration, codec, matching, measurement, or cache
  method.
- The cold measurement action is supplied explicitly and runs outside Runtime's hot path. It is
  not `PreparedExecutable` discovery or Runtime profiling. The caller must arrange equivalent
  representative inputs and complete candidate execution before invoking tuning.
- Stable model/profile fingerprints are tool-local caller assertions used for result evidence.
  They do not define canonical Model identity, affect workload-cache reuse, or create a Model,
  Compiler, Prepare, Runtime, or Engine API.
- Workload and candidate compatibility bytes are opaque to tuning and supplied by the backend-
  aware collaborator. For CPU 0010E the collaborator must derive them from the authoritative
  exact/default FLOAT32/FLOAT64 CPU values; the tool must not share an interpretation of those
  fields.
- Model autotuning is optional for correctness. An absent, incompatible, corrupt, or failed cache
  must never grant eligibility. Ordinary cache-free heuristic preparation remains unchanged.
- Tuning completes before any selected decision reaches repeated cold backend analysis and before
  Runtime execution. Runtime never measures, searches, selects, reads, writes, or invalidates the
  cache.
- Benchmarking remains report-only and does not call this workflow to mutate production settings.
- No hidden global state, concrete-backend dependency, module ownership change, or architecture
  rule is authorized. Stop if implementation requires one.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.tools.tuning` — current tool module root; becomes the deliberate small
  public surface and contains its package-private cache implementation.
- `io.github.pho001.synaptik.prepare.analysis` — existing public method-free opaque roles and
  exact-partition handoff used only through bounded generics.

Packages added or changed:

- No Java package is added.
- The tuning root changes from a placeholder-only package to one cohesive local-workload tuning
  API plus package-private persistence mechanics.

Type placement:

- `io.github.pho001.synaptik.tools.tuning.WorkloadTuning` — stateless workflow entry and private
  orchestration; contains only the package-private monotonic-clock test seam.
- `io.github.pho001.synaptik.tools.tuning.WorkloadTuningRequest` — immutable tool-local request and
  its small nested `Objective`, `Budget`, fingerprint, compatibility, candidate-identity, reuse-
  scope, and occurrence values.
- `io.github.pho001.synaptik.tools.tuning.BackendWorkloadTuning` — typed caller collaboration for
  backend-owned enumeration, opaque identities, decision construction, and codec operations.
- `io.github.pho001.synaptik.tools.tuning.ColdCandidateMeasurement` — typed one-complete-candidate
  execution action, separate from Runtime and benchmarking.
- `io.github.pho001.synaptik.tools.tuning.WorkloadTuningResult` — immutable selected handoffs and
  rich nested occurrence/workload/candidate evidence values.
- `io.github.pho001.synaptik.tools.tuning.WorkloadCacheFile` — package-private bounded binary
  cache parsing, merging, deterministic encoding, checksum, and atomic publication.

Tests mirror the production package because they must exercise package-private clock and cache
seams. No `api`, `cache`, `evidence`, `internal`, `util`, `common`, or backend package is added.

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

Expected documentation and planning paths:

- `docs/developer-guide/benchmarking.md` — distinguish the operational tuning workflow from
  report-only benchmark evidence and explain the current caller-supplied cold boundary
- `docs/backend-guide/partition-preparer.md` — explain how an owning composition supplies the
  typed collaboration without widening the method-free Prepare roles
- `docs/glossary.md` — update the tuning workflow/artifact contracts, measurement and persistent
  workload-cache status, and tuning-orchestration status from planned to the exact implemented
  task-0001 boundary
- this task
- `docs/planning/tools/tuning/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a concrete contradiction is found: `ARCHITECTURE.md`, focused
architecture explanations and ADR 0008, Config/Engine/Compiler/Planning/Prepare master
plans, completed CPU 0010E and Prepare 0004 tasks, CPU/Prepare production and tests, Runtime API,
other Gradle files, architecture tests, backend conformance tests, and integration tests.

## Maximum scope

At most the 19 exact paths above:

- 9 tuning production/build paths;
- 4 tuning test paths;
- 3 focused explanatory-documentation paths (the two guides plus `docs/glossary.md`); and
- 3 tuning task/master/roadmap planning paths.

The task remains one cohesive capability because measurement, compatible cache hits, miss-only
selection, decision codec round trips, evidence separation, and atomic publication are one
observable transaction and share the same failure-before-mutation tests. Splitting cache or
measurement first would create an unconsumed format or an operational workflow that cannot meet
its reuse contract. If another path, production type, package, module, dependency, or public
operation is required, stop and return the task to planning.

## Acceptance criteria

1. The five public types and one package-private cache type listed in Package impact exist with
   the exact responsibilities and generic bounds specified; no additional public production type
   or package is added.
2. Public signatures expose only JDK values, tool-local immutable values, and Prepare's method-
   free roles/handoff. The tuning module has a public Prepare dependency and no concrete-backend,
   Engine, or Runtime dependency.
3. Request validation snapshots all byte sequences/lists, retains exact handoff references,
   rejects null/empty/invalid values and present input decisions, and locks the sole initial
   objective to `MIN_MEDIAN_ELAPSED_NANOS`.
4. The backend collaboration is called only for its six declared operations. Tuning uses no
   reflection, downcast, raw type, unchecked cast, `Object`, map/string dispatch, route switch,
   CPU field access, or concrete-backend import.
5. Exact compatible occurrences deduplicate once while retaining every context and checked-sum
   weight. Session results reuse only in the invocation; persistent entries reuse across cache
   loads and models. Model/profile identity never changes workload-cache matching.
6. The complete cache is loaded and validated before candidate enumeration or execution. Every
   specified corruption/size/schema/duplicate/truncation/checksum/trailing-byte case fails without
   measurement or file mutation.
7. An exact well-formed compatible cache hit calls backend decoding once, executes and enumerates
   no candidate, returns the decoded selected decision for every occurrence, and records only its
   compact summary as `CACHE_HIT` evidence.
8. A missing or backend-rejected entry is a miss. Only misses enumerate and measure candidates;
   incompatibility never grants eligibility or falls back to a stale decision.
9. Budget and complete-candidate validation finish before the first execution. Over-budget,
   empty/null/duplicate candidates, unstable or duplicate identities, and arithmetic overflow
   fail without partial measurement or persistence.
10. Deterministic clock-backed tests prove exact warmup/timed invocation counts, non-negative
    sample validation, integer median calculation, lowest-median selection, encounter-order tie
    breaking, checked failure propagation, and one decision construction for the winner.
11. Miss evidence retains every raw timed sample and compact min/median/max/count summary;
    cache entries retain only the summary. Evidence, request, result, and cache snapshots are
    immutable and leak no mutable byte array or collection.
12. Persistent decision encoding is non-empty/bounded and must decode back to an equal compatible
    decision before publication. Session results and failed/incompatible round trips are never
    persisted.
13. Cache encoding is deterministic and bounded with the exact binary ingredients in Scope;
    Java serialization, JSON, class names, enum names, strings as dispatch, and executable bytes
    are absent.
14. Same-directory forced temporary-file plus atomic replacement publishes a changed persistent
    cache. Tests prove no-write hits, stable deterministic bytes, retained unrelated entries,
    temporary cleanup, and preservation of the prior cache on injected pre-move and move failure.
15. The workflow performs no report-only benchmark mutation, Runtime profiling/hot-path work,
    Prepare analysis/finalization, backend discovery, model compilation, graph/plan tuning, or
    performance-proof assertion.
16. CPU 0010E and Prepare 0004 remain Complete; Config 0006A remains Draft without a detailed
    task; provider 0004 and CPU 0010D1 remain blocked/deferred; tuning 0002–0003 remain Draft
    without detailed task specifications.
17. A separate clean documentation-focused context independently reviews implementation/tests,
    finalizes affected Javadocs/package prose, the two focused guides, and `docs/glossary.md`,
    records reasoned no-change conclusions for excluded areas, and does not repeat successful
    Java tests unless it changes executable behavior or records a concrete stale-evidence risk.
18. Focused/final tuning tests, tuning Javadoc and rendered-page inspection, root dependency/
    architecture validation, Markdown links/anchors/headings/fences/newlines/whitespace, exact
    path/public-shape/package/dependency/status/order checks, and `git diff --check` pass before
    completion.

## Tests / validation

Run focused tests while implementing. After executable Java stabilizes, run one final module
command:

```bash
./gradlew :tools:tuning:test
```

Tests use synthetic immutable implementations of the Prepare marker roles and a deterministic
package-private clock. They are filesystem-local and execute no CPU/OpenBLAS/Runtime code. They
must prove the complete operational workflow rather than only DTO construction.

The documentation-focused context reuses that successful executable evidence unless it changes
Java behavior or tests, then runs:

```bash
./gradlew :tools:tuning:javadoc
git diff --check
git status --short -uall
```

Render and inspect the five public tuning declarations and package summary. Validate every
changed Markdown file for local targets and anchors, unique headings, balanced backtick/tilde
fences, LF/final newlines, trailing whitespace, terminology, exact scope, package placement,
public shape, dependency direction, and synchronized statuses.

After the documentation pass, run one repository-wide checkpoint because this task adds a public
Prepare dependency and a new persistent-artifact boundary:

```bash
./gradlew test
```

No real OpenBLAS library, performance threshold, benchmark run, backend-conformance-specific run,
or integration-specific run is required. The root command supplies dependency and architecture-
test coverage; this task changes no backend or end-to-end execution behavior.

## Dependencies

- CPU 0010E — Complete and authoritative for the first exact/default FLOAT32/FLOAT64 complete
  candidate batches, compatibility facts, candidate identities, and selected decisions.
- Prepare 0004 — Complete and authoritative for the method-free opaque batch/decision roles and
  exact-partition typed handoff.
- JDK filesystem, channel, checksum/digest, monotonic-clock, and atomic-move APIs already available
  to the tool module.
- Explicit caller composition supplies stable model/profile fingerprints, the typed backend
  collaboration, and one complete comparable cold candidate execution action.

The last item is an input contract implemented and exercised with deterministic test
collaborators in this task, not an absent module prerequisite. Engine, Config 0006A, provider
0004, CPU 0010D1, Runtime changes, report-only benchmarks, and a public CPU adapter are not
dependencies.

## Follow-up tasks

- Tuning 0002 remains Draft. It later requires Compiler/Planning complete candidates, a complete
  Prepare candidate path, and operational Engine lifecycle measurement to tune bounded graph and
  plan alternatives without repeating local search.
- Tuning 0003 remains Draft and later adds cache/plan inspection after both artifact roles are
  stable.
- Config 0006A remains Draft and may define immutable public request inputs only after this
  tool-local consumer is implemented and reviewed. Do not create its detailed specification here.
- Engine 0001–0004 remain Draft. Later Engine composition may supply model identity and complete
  candidate execution through this task's generic collaboration; it must not be invented here.
- CPU 0016 later may generalize CPU-owned candidate/decision production across eligible peer
  routes. It does not move CPU vocabulary or an adapter into shared tuning.
- A supported out-of-the-box CPU integration, if later required by an actual public consumer,
  must be planned at the composition boundary without making tuning depend on unsupported CPU
  internals or making CPU depend on tooling.

## Architecture impact

Expected impact: None.

This task implements the explicit tuning ownership and opaque collaboration already authorized by
the architecture. Adding a public Prepare dependency from the outer tool follows the existing
direction and changes no inward module. If implementation requires a concrete-backend edge,
Prepare role methods, Runtime measurement, an Engine facade, Config API, hidden state, or another
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
  not by inventing an Engine facade or claiming the placeholder module is operational.
- Use a typed generic collaboration beside the tuning workflow rather than adding methods to
  Prepare's opaque markers. This gives the operational consumer exactly the six backend-owned
  operations it needs while preserving method-free transport and backend vocabulary ownership.
- Treat model/profile fingerprints as caller-supplied evidence identity. Workload-cache reuse is
  deliberately model-independent and keyed only by backend-supplied canonical compatibility plus
  objective/sampling semantics.
- Measure every candidate in a complete miss batch. The budget must admit the full deduplicated
  work or fail before execution; a time cutoff or prefix search would make the winner dependent on
  machine speed or backend order beyond the explicit tie rule.
- Use only minimum median elapsed nanoseconds in this first slice. Additional objectives or
  statistical policies require separate consumers and evidence rather than generic scoring maps.
- Keep timing in tuning by supplying an execution action, not a caller-computed duration. A
  package-private clock makes validation deterministic without exposing clock policy publicly.
- Reject a corrupt cache as an explicit request failure and preserve it for diagnosis. A valid
  but backend-incompatible entry is a miss because only the backend-aware collaborator can decide
  compatibility.
- Require decision encode/decode equality before atomic publication. This prevents a codec from
  persisting a value that the same current batch cannot recover compatibly.
- Keep one cohesive 19-path task because cache-first reuse, miss measurement, deterministic
  selection, evidence separation, and atomic publication are one transactional capability and
  share failure/mutation invariants.

## Known limitations

- The task supplies an operational generic workflow and deterministic synthetic integration
  tests, not a supported public CPU adapter or public Engine entry. Current CPU candidate values
  remain technically public unsupported internals interpreted only by backend-aware composition.
- Only exact/default FLOAT32/FLOAT64 candidate batches are authorized for the first workflow.
  The tool itself does not inspect data type or numerical fields; eligibility is guaranteed by
  the supplying backend collaboration.
- The first objective is minimum median elapsed nanoseconds with fixed warmup and odd sample
  counts. There is no correctness oracle, noise model, adaptive sampling, confidence claim, or
  proof that the selected candidate is universally faster.
- Session-only workloads deduplicate only inside one invocation and are not persisted. Cross-
  process reuse requires backend-supplied persistent compatibility and a successful codec round
  trip.
- Concurrent writers, cache migration/recovery/inspection, eviction, remote caches, graph/plan
  tuning, prepared artifacts, and Config/Engine facades remain deferred.

## Validation evidence

- Documentation-focused review context:
  `01a091d6-247d-7f82-844a-7a2860541741`.
- Reused the clean executable-correction context's final
  `./gradlew :tools:tuning:test` result because this documentation pass changed no executable Java
  behavior or tests: 4 suites, 24 tests, 0 failures, 0 errors, 0 skipped. The four XML suites are
  `TuningPublicShapeTest` (3), `WorkloadTuningCacheTest` (8), `WorkloadTuningTest` (3), and
  `WorkloadTuningValidationTest` (10).
- `./gradlew :tools:tuning:javadoc` passed after the final Javadoc edits with no warnings.
- Generated package summary and the five public top-level declaration pages were rendered to text
  with Pandoc and inspected. The nested `CandidateEvidence` and `WorkloadEvidence` pages were also
  inspected for the measured-only exact raw-sample invariant and empty cache-hit candidate list.
  An in-app/connected browser renderer was unavailable, and macOS Quick Look rendering was denied
  by the execution environment; generated HTML structure and rendered text were therefore the
  available rendering evidence.
- Package-private `WorkloadCacheFile` source and Javadocs were inspected for bounds, complete-file
  validation, deterministic encoding, immutable byte snapshots, same-directory forced temporary
  files, atomic replacement, prior-file preservation, cleanup, and absence of executable payloads.
- Changed-Markdown validation passed for local file targets and heading anchors, unique headings,
  balanced backtick/tilde fences, LF endings, final newlines, trailing whitespace, and current-
  versus-planned terminology.
- Exact-scope inspection confirmed exactly 19 authorized status paths: 9 production/build paths,
  4 test paths, 3 explanatory documentation paths, and 3 planning paths. Public-shape tests and
  source inspection confirmed exactly five public top-level production types plus the package-
  private cache type. Gradle/source inspection confirmed a public Prepare dependency, private
  Planning dependency, and no Config, Engine, Runtime, or concrete-backend dependency.
- Status/order inspection confirmed CPU 0010E and Prepare 0004 remain Complete; provider 0004 and
  CPU 0010D1 remain blocked/deferred; tuning 0002 and 0003 remain Draft without specifications;
  Config 0006A is the next Draft task.
- `./gradlew test` passed: `BUILD SUCCESSFUL`, 65 actionable tasks (1 executed, 64 up-to-date).
  Aggregating the current JUnit XML produced 468 suites, 3,032 tests, 0 failures, 0 errors, and 28
  skipped tests.
- `git diff --check` passed on the final combined change.

## Implementation notes

- Replaced the placeholder module marker with five focused public API types and one package-private
  cache codec in the existing tuning package; no new package or public facade was introduced.
- The workflow loads and validates the explicit cache first, deduplicates exact compatibility in
  request order, validates every complete miss batch before execution, measures configured warmup
  and odd timed samples, selects minimum integer median with encounter-order ties, round-trips a
  persistent decision codec, and publishes only completed persistent misses atomically.
- The executable correction finalized `CandidateEvidence` as measured-only evidence with a
  non-empty immutable encounter-order snapshot of non-null non-negative samples whose count,
  minimum, integer-middle median, and maximum exactly match its summary. Cache-hit
  `WorkloadEvidence` exposes an empty candidate-evidence list.
- The public surface remains a generic composition boundary. Callers supply stable identities,
  backend-owned compatibility/candidate identity and decision codec behavior, candidate
  enumeration, and complete candidate execution. There is no supported CPU adapter, Engine
  integration, Config facade, model extraction, or graph/plan tuning.

## Completion summary

- Completed changes: implemented the exact/default generic workload-tuning transaction, reusable
  bounded persistent cache, selected handoffs, and separate rich evidence; finalized Javadocs,
  guides, glossary, and synchronized planning state.
- Files changed or created: exactly the 19 paths listed under Affected files.
- Tests and validation: reused the final 24-test tuning-module result; new clean Javadoc,
  rendered-text/page inspection, Markdown/scope/dependency/status checks, repository-wide 3,032-
  test checkpoint, and final whitespace check passed.
- Documentation-agent review: this mandatory clean documentation-focused context independently
  audited all acceptance criteria and finalized the same overall uncommitted change.
- Documentation impact: benchmarking and partition-preparer guides now describe the operational
  generic caller-supplied tuner/cache and its unsupported integration boundaries.
- Javadoc review: all five public types, their nested public values and members, package summary,
  and package-private cache contract were finalized; `CandidateEvidence` and cache-hit evidence
  invariants are explicit.
- Glossary impact: corrected the three stale claims about workflow/artifact contracts,
  measurement/persistent-cache use, and tuning orchestration while leaving the full two-phase
  roadmap planned.
- Unresolved issues: None within task 0001 scope.
- Follow-up required: Config 0006A remains the next Draft task; tuning 0002/0003, supported
  CPU/Engine integration, model extraction, and graph/plan tuning remain planned.

Status: Complete
