# Task 0003: Read-Only [Cache](../../../../glossary.md#term-cache), Plan, and Evidence Inspection

## Status

Complete

## Goal

Add one bounded public `tools/tuning` inspection surface for the two implemented compact tuning
[artifact](../../../../glossary.md#term-artifact) formats and the two implemented rich in-memory evidence values. Inspection is strictly
observational: it snapshots and validates bytes, reports redacted [provenance](../../../../glossary.md#provenance) and selected-candidate
summaries, explains every mismatch provable from stored keys, and never executes a candidate,
decodes a [backend](../../../../glossary.md#backend) decision, prepares a plan, or mutates a file or [runtime](../../../../glossary.md#term-runtime) state.

```text
workload-cache bytes -> structural report + compact workload-entry summaries
model-plan bytes     -> structural report + compact selected-plan summaries
existing result evidence -> redacted rich measurement/correctness summaries
```

A structurally valid key match is not a compatibility verdict. Only the current backend-owned
decoder can authenticate [opaque](../../../../glossary.md#term-opaque) decision bytes. Inspection therefore reports
`KEY_MATCH_REQUIRES_BACKEND_DECODER`, never `COMPATIBLE`.

## Readiness audit

The source-backed audit found no missing [owner](../../../../glossary.md#term-owner), format decision, or architecture blocker:

1. `WorkloadCacheFile` and `ModelPlanCacheFile` are implemented package-private schema-1 codecs in
   the same package. Both formats are bounded to 16 MiB and 65,536 entries, use positive
   length-prefixed opaque values bounded to 1 MiB, fixed big-endian numeric fields, strict
   canonical entry order, and a trailing SHA-256 whole-payload checksum.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [opaque glossary](../../../../glossary.md#term-opaque); [big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [canonical glossary](../../../../glossary.md#term-canonical-form); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [checksum glossary](../../../../glossary.md#term-checksum))</span>
2. Both loaders already reject unsupported schema, oversize, truncation, file growth during the
   bounded read, checksum damage, malformed fields, invalid summaries, duplicate/noncanonical
   order, and trailing payload bytes. Inspection can reuse the same parsing authority without
   defining a third format.<span data-glossary-navigation="true"> ([truncation glossary](../../../../glossary.md#term-truncation-toward-zero))</span>
3. The workload cache stores compatibility schema/bytes, objective and sampling fields, encoded
   decision bytes, winner identity, and a compact timing summary. It does not persist reuse scope,
   model, representative-profile, occurrence context/weight, raw samples, or an executable.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [objective glossary](../../../../glossary.md#term-objective); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [identity glossary](../../../../glossary.md#term-identity-meanings); [scope glossary](../../../../glossary.md#term-scope); [model glossary](../../../../glossary.md#term-model); [occurrence glossary](../../../../glossary.md#term-occurrence); [weight glossary](../../../../glossary.md#term-weight); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
4. The model-plan cache stores producer, codec, model, representative-profile, target,
   compatibility, and policy identities; objective, correctness, and sampling fields; encoded
   decision and winner bytes; and a compact timing summary. It does not persist reuse scope,
   correctness actions/reference bytes, raw samples, a backend candidate, or executable state.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [codec glossary](../../../../glossary.md#term-codec); [target glossary](../../../../glossary.md#term-target); [backend glossary](../../../../glossary.md#backend))</span>
5. `WorkloadTuningResult.Evidence` and `CompletePlanTuningResult.Evidence` already retain the rich
   in-memory measurement evidence truthfully. Raw samples and complete-plan correctness actions
   exist there only and cannot be reconstructed from either compact file.
6. Exact cache-key comparison is tool-owned because the existing transactions already construct
   and compare those keys. Opaque decision interpretation remains producer/backend-owned. A
   redacted inspector can prove field mismatches while refusing to claim backend compatibility or
   selected-plan executability.
7. The existing root package is the deliberate public tuning surface and has no Engine, Runtime,
   Config, CPU, or private-backend import. Inspection needs no new module edge or package.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [module glossary](../../../../glossary.md#term-module))</span>
8. Both stable formats share ownership, bounds, redaction, checksum, read-only file semantics, and
   compatibility limitations, so inspecting both is one cohesive tools-only task.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
9. Rich evidence inspection remains only a view over already constructed result values. It
   performs no file lookup and does not imply that rich evidence was persisted.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [lookup glossary](../../../../glossary.md#term-lookup))</span>

The readiness audit established that the task required no Config/[Engine](../../../../glossary.md#term-engine)/[CPU](../../../../glossary.md#term-cpu) prerequisite or
architecture decision.

## [Scope](../../../../glossary.md#term-scope)

- Add one public final field-free `TuningInspection` namespace in
  `io.github.pho001.synaptik.tools.tuning`; it is the sole new public top-level production type.
- Add exactly these public operations, with Path and byte-array pairs having identical parsing and
  reporting semantics:
  - `WorkloadCacheInspection inspectWorkloadCache(Path)`;
  - `WorkloadCacheInspection inspectWorkloadCache(Path, WorkloadExpectation)`;
  - `WorkloadCacheInspection inspectWorkloadCache(byte[])`;
  - `WorkloadCacheInspection inspectWorkloadCache(byte[], WorkloadExpectation)`;
  - `ModelPlanCacheInspection inspectModelPlanCache(Path)`;
  - `ModelPlanCacheInspection inspectModelPlanCache(Path, ModelPlanExpectation)`;
  - `ModelPlanCacheInspection inspectModelPlanCache(byte[])`;
  - `ModelPlanCacheInspection inspectModelPlanCache(byte[], ModelPlanExpectation)`;
  - `WorkloadEvidenceInspection summarize(WorkloadTuningResult.Evidence)`; and
  - `CompletePlanEvidenceInspection summarize(CompletePlanTuningResult.Evidence)`.
- Keep all report, expectation, enum, opaque-summary, cache-entry, mismatch, and evidence-summary
  values as public immutable nested types of <span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [cache glossary](../../../../glossary.md#term-cache); [immutable glossary](../../../../glossary.md#term-immutable))</span>`TuningInspection`. Add no other top-level API type,
  extensible hierarchy, generic map, callback, registry, or service.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [callback glossary](../../../../glossary.md#term-callback); [registry glossary](../../../../glossary.md#term-registry))</span>
- The exact public nested type names are `ArtifactStatus`, `ArtifactSource`, `InvalidReason`,
  `CompatibilityStatus`, `MismatchReason`, `OpaqueSummary`, `VersionedOpaqueSummary`,
  `WorkloadExpectation`, `ModelPlanExpectation`, `WorkloadCacheInspection`,
  `WorkloadEntryInspection`, `ModelPlanCacheInspection`, `ModelPlanEntryInspection`,
  `OccurrenceEvidenceSummary`, `CandidateMeasurementSummary`, `WorkloadEvidenceSummary`,
  `WorkloadEvidenceInspection`, `CompleteCandidateMeasurementSummary`, and
  `CompletePlanEvidenceInspection`. Use records/enums unless a validated final class is required
  to snapshot an array/list before assignment. Add no public constructor to <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>`TuningInspection`.
- Define workload expectations from an existing `WorkloadCompatibility`, objective, warmup count,
  and positive odd timed-sample count. Preserve the compatibility's backend-declared reuse scope
  in the expectation although it is not stored in the file.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [backend glossary](../../../../glossary.md#backend))</span>
- Define model-plan expectations from existing producer/codec/model/profile/target/plan-
  compatibility/policy identity types, objective, correctness policy, warmup count, and positive
  odd timed-sample count. Preserve producer-declared reuse scope in the expectation although it is
  not stored in the file.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [codec glossary](../../../../glossary.md#term-codec); [target glossary](../../../../glossary.md#term-target); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Report [artifact](../../../../glossary.md#term-artifact) status as exactly `MISSING`, `VALID`, or `INVALID`. `MISSING` applies only when a
  Path did not exist at read-only open. Byte-array inspection never reports `MISSING`.
- Classify invalid artifacts with stable typed reasons for size-too-small, oversize, truncation,
  growth during read, checksum mismatch, invalid magic, unsupported schema, invalid entry count,
  invalid length, invalid key/value or summary structure, duplicate/noncanonical order, and
  trailing payload. Parser exception messages are not public classification.<span data-glossary-navigation="true"> ([truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [checksum glossary](../../../../glossary.md#term-checksum); [schema glossary](../../../../glossary.md#term-schema); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Propagate Path permission, directory, and ordinary read failures as `IOException`. Malformed
  content returns `INVALID`; existing tuning loaders retain their [fail-closed](../../../../glossary.md#term-fail-closed) `IOException`
  behavior.
- Report observed byte count, supported artifact schema when available, whole-file SHA-256, and
  entries in exact canonical file order. Missing reports have no digest/entries; invalid reports
  have no entries and only safely known metadata.<span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [canonical glossary](../../../../glossary.md#term-canonical-form); [digest glossary](../../../../glossary.md#term-digest); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Represent every opaque identity, compatibility, encoded decision, and winner as an immutable
  byte-length plus lowercase SHA-256 digest summary, including positive schema where supplied.
  Never return stored opaque bytes, decoded backend fields, `Object`, or implementation names.
- Workload entries report compatibility, numeric objective, sampling, selected-candidate and
  encoded-decision summaries, and compact min/median/max/count. They explicitly have no
  model/profile, occurrence, raw-sample, correctness, or executable facts.<span data-glossary-navigation="true"> ([sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [median glossary](../../../../glossary.md#term-median); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Model-plan entries report redacted producer, codec, model, profile, target, compatibility, and
  policy; numeric objective/correctness and sampling; selected-candidate and decision summaries;
  and compact timing. They expose no prepared recipe, decision meaning, correctness reference, or
  raw sample.
- Without an expectation, compatibility is `NOT_EVALUATED`. With an expectation, compare fields
  in documented deterministic order and return an immutable ordered mismatch list. An exact
  stored-key match is `KEY_MATCH_REQUIRES_BACKEND_DECODER` only when the corresponding workload
  or plan compatibility declares `PERSISTENT`. A workload or [model](../../../../glossary.md#term-model)-plan expectation declaring
  `SESSION` reports `SESSION_SCOPE_FORBIDS_PERSISTENT_REUSE` even if every stored key field matches.
- Mismatch reasons are bounded to schema/value mismatch for each stored opaque identity,
  objective/correctness mismatch, warmup mismatch, timed-sample mismatch, and the shared workload/
  model-plan session-scope reason. Do not infer hardware, provider, route, candidate legality,
  decision decodability, freshness, numerical compatibility, or executability.
- Path inspection opens read-only, obtains size from that channel, enforces 16 MiB before
  allocation, reads exactly the bounded snapshot, rejects early end, and probes one extra byte for
  growth. It never queries/creates a parent, writes, forces, locks, repairs, renames, moves,
  deletes, or creates a temporary file.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel))</span>
- Document the remaining filesystem race: same-length replacement/mutation is assessed only
  through bytes actually read and checksum; no stable-file or locking promise exists. Bytes
  appended after the final growth probe are outside the completed snapshot.<span data-glossary-navigation="true"> ([race glossary](../../../../glossary.md#term-race))</span>
- Byte-array inspection enforces 16 MiB before a second full allocation, snapshots before parse,
  never mutates the caller array, and retains neither supplied nor complete snapshot bytes.
- Preserve task-0001/task-0002 cache bytes, keys, ordering, public behavior, hit/miss behavior,
  atomic publication, bounds, and failure-before-execution. Refactor only narrowly shared
  package-private parsing/read mechanics.<span data-glossary-navigation="true"> ([atomic publication glossary](../../../../glossary.md#term-atomic-publication))</span>
- Summarize workload evidence with redacted model/profile/compatibility/context/candidate
  identities, objective/budget, source, occurrence counts/weights, measured raw samples, compact
  summaries, and winner. Cache hits retain no fabricated candidate measurements.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight))</span>
- Summarize complete-plan evidence with redacted model/profile/target/policy/candidate identities,
  objective, correctness policy, source, recorded correctness actions, measured raw samples, and
  compact winner. Cache hits retain no fabricated correctness or measurement rows.
- Evidence summarization performs no cache I/O, decoding, enumeration, correctness, measurement,
  preparation, or execution. Reports are detached immutable snapshots separate from cache reports.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results))</span>
- Finalize Javadocs/package prose, the focused benchmarking guide, glossary, and planning evidence
  in the mandatory separate clean documentation context.<span data-glossary-navigation="true"> ([benchmarking glossary](../../../../glossary.md#term-benchmarking); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [CLI](../../../../glossary.md#term-cli)/UI, [JSON](../../../../glossary.md#term-json)/text export, logging, remote service, or watch mode
- cache migration, repair, rewriting, deletion, eviction, expiry, merging, recovery, locking,
  trust/authentication, encryption, signatures, or concurrent-writer coordination<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication); [signatures glossary](../../../../glossary.md#term-signature))</span>
- streams, supplied channels, directories, globs, multiple paths, URLs, or memory mapping
- decoding/returning backend decision or candidate fields, invoking backend collaborations, or
  exposing opaque identity/decision bytes<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [opaque glossary](../../../../glossary.md#term-opaque); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- claiming key match means compatible, executable, current, faster, or safe on this machine
- deserializing, constructing, inspecting, or executing [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), Runtime state,
  publication bytes, correctness references, concrete plans, or generated artifacts<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Config/Engine composition, CPU/private-backend imports, fallback, selected production
  preparation, model extraction, or multiple-partition search<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [model glossary](../../../../glossary.md#term-model); [partition glossary](../../../../glossary.md#partition))</span>
- changing either [cache](../../../../glossary.md#term-cache) [schema](../../../../glossary.md#term-schema)/magic/[checksum](../../../../glossary.md#term-checksum)/order/bounds/bytes or existing public tuning types
- persisting rich evidence, reconstructing missing evidence, or joining cache and evidence reports
- Java [serialization](../../../../glossary.md#term-serialization), reflection-based interpretation, raw/unchecked casts,
  `Map<String,Object>`, string [dispatch](../../../../glossary.md#term-dispatch), hidden global state, [registry](../../../../glossary.md#term-registry), or [service locator](../../../../glossary.md#term-service-locator)
- Architecture/ADR, dependency, Gradle, other-module, architecture-test, conformance-test, or
  integration-test changes<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants, Performance
  evidence and optimization tooling, Runtime, Prepare, concrete backends, and Dependency rules<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [backends glossary](../../../../glossary.md#backend); [Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Tuning master plan](../master-plan.md)
- [Task 0001 workload tuning and cache](0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- [Task 0002 complete-plan tuning and cache](0002-bounded-complete-plan-tuning-and-model-plan-cache.md)
- [Benchmarking and tuning guide](../../../../developer-guide/benchmarking.md)<span data-glossary-navigation="true"> ([Benchmarking glossary](../../../../glossary.md#term-benchmarking))</span>
- [Tuning glossary terms](../../../../glossary.md#autotuning--model-autotuning)

## Architecture constraints

- `tools/tuning` owns bounded cache coordination, comparison, selection, and evidence. A cold
  read-only view of its own compact artifacts and result evidence remains in that owner.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [view glossary](../../../../glossary.md#term-view-meanings); [artifacts glossary](../../../../glossary.md#term-artifact); [owner glossary](../../../../glossary.md#term-owner))</span>
- Runtime hot-path code performs no tuning or inspection. This API imports no Runtime type and
  never receives or constructs runtime state.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Candidate legality, compatibility meaning, decision decoding, backend eligibility/selection,
  lowering, and route interpretation remain producer-owned. Exact stored-key match is only a
  proven byte/key match awaiting backend authentication.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [producer glossary](../../../../glossary.md#term-producer); [authentication glossary](../../../../glossary.md#term-authentication))</span>
- Inspection never mutates or repairs cache state. Existing cache publication remains solely in
  tuning transactions with unchanged atomic replacement.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Compact cache and rich evidence remain separate. A file report cannot claim non-persisted raw
  samples, correctness actions, occurrence context, or provenance.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [occurrence glossary](../../../../glossary.md#term-occurrence); [provenance glossary](../../../../glossary.md#provenance))</span>
- No Engine, CPU/private-backend, Config, Runtime, Compiler, or concrete-backend dependency/import
  is authorized. Existing Prepare marker exposure remains unchanged.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [Compiler glossary](../../../../glossary.md#term-compiler); [dependency glossary](../../../../glossary.md#term-dependency); [Prepare glossary](../../../../glossary.md#prepare))</span>
- No Java serialization, reflection-based interpretation, generic object map, hidden state,
  discovery, or service locator is authorized.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- If implementation requires schema change, backend decoder, cross-module contract, file
  mutation, or architecture update, stop and return the task to planning.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [module glossary](../../../../glossary.md#term-module); [planning glossary](../../../../glossary.md#term-planning))</span>

## Package impact / type placement

Existing package used:

- `io.github.pho001.synaptik.tools.tuning` — owns both formats, both evidence models, and the new
  read-only inspection surface.

Packages added or changed:

- No Java package is added.
- The tuning root gains one public namespace and package-private parsing support in the existing
  codecs.

Type placement:

- `io.github.pho001.synaptik.tools.tuning.TuningInspection` — public final field-free namespace
  containing the ten exact operations and every immutable public report/expectation value.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.tools.tuning.WorkloadCacheFile` — remains package-private and owns the
  authoritative workload parser and internal typed diagnostics.
- `io.github.pho001.synaptik.tools.tuning.ModelPlanCacheFile` — remains package-private and owns
  the authoritative model-plan parser and internal typed diagnostics.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

Tests mirror production for package-private format seams. No `inspection`, [`cache`](../../../../glossary.md#term-cache), `evidence`,
`internal`, `util`, or [backend](../../../../glossary.md#backend) package is added.

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths:

- add `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/TuningInspection.java`
- `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/WorkloadCacheFile.java`
- `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/ModelPlanCacheFile.java`
- `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/package-info.java`

Expected test paths:

- add `tools/tuning/src/test/java/io/github/pho001/synaptik/tools/tuning/TuningInspectionTest.java`
- add `tools/tuning/src/test/java/io/github/pho001/synaptik/tools/tuning/TuningInspectionValidationTest.java`
- add `tools/tuning/src/test/java/io/github/pho001/synaptik/tools/tuning/TuningInspectionPublicShapeTest.java`

Expected explanatory and [planning](../../../../glossary.md#term-planning) paths:

- `docs/developer-guide/benchmarking.md`
- `docs/glossary.md`
- this task
- `docs/planning/tools/tuning/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless contradicted: architecture/ADRs, current tuning contracts and
tests, Config/[Engine](../../../../glossary.md#term-engine)/[CPU](../../../../glossary.md#term-cpu)/[Prepare](../../../../glossary.md#prepare), [Gradle](../../../../glossary.md#term-gradle), architecture tests, [conformance](../../../../glossary.md#term-conformance)/integration, and other
task specifications.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the 12 exact paths above: 4 production/[Javadoc](../../../../glossary.md#term-javadoc), 3 test, 2 explanatory, and exactly 3
[planning](../../../../glossary.md#term-planning) paths. The seven source/test paths meet the 3–12 guardrail. Both formats are coordinated
compact tuning [artifacts](../../../../glossary.md#term-artifact) with one read-only/redaction/compatibility contract; evidence summaries
complete the compact-versus-rich distinction. If another top-level type, package, [module](../../../../glossary.md#term-module) edge,
[schema](../../../../glossary.md#term-schema) change, test [owner](../../../../glossary.md#term-owner), explanatory path, or thirteenth path is needed, stop and re-plan.

## Acceptance criteria

1. `TuningInspection` is the only new public top-level type and exposes exactly the ten Scope
   operations. Every nested public value is final or a record/enum, immutable, and documented.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [immutable glossary](../../../../glossary.md#term-immutable))</span>
2. Path and byte-array inspection of identical bytes returns equal semantic content except source
   kind. Arrays, lists, digests, samples, and nested values leak no mutable state.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [mutable glossary](../../../../glossary.md#term-mutable))</span>
3. Missing Path returns `MISSING`; ordinary read failures throw `IOException`; malformed bytes
   return `INVALID` with one exact typed category and no partial entries.
4. Read-only-directory and absent-target tests prove no target/temp/directory entry, mtime, or byte
   mutation. Source inspection finds no write/create/force/move/delete in inspection.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
5. The channel reader enforces size before allocation, detects early EOF and one-byte growth, and
   documents TOCTOU limits. Deterministic seams prove truncation/growth classification.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [EOF glossary](../../../../glossary.md#term-eof); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero))</span>
6. Byte-array inspection rejects oversize before a second full copy, snapshots before parse,
   ignores later mutation, and retains no supplied full array.
7. Both formats classify too-small, oversize, truncation, checksum, magic, schema, entry count,
   length/structure/summary, duplicate/order, and trailing-byte failures with no partial entries.<span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum); [schema glossary](../../../../glossary.md#term-schema))</span>
8. Valid reports preserve file order and redact every opaque value to schema when present,
   length, and lowercase SHA-256. Raw opaque values never appear in reports or <span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [SHA-256 glossary](../../../../glossary.md#term-sha-256))</span>`toString()`.
9. Workload and model-plan reports expose only their actually stored compact fields and make
   absent provenance/evidence explicit.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
10. No-expectation entries are `NOT_EVALUATED`; mismatch lists use deterministic field order;
    exact `PERSISTENT` key matches require decoder. Exact-key workload and model-plan expectations
    declaring `SESSION` both report `SESSION_SCOPE_FORBIDS_PERSISTENT_REUSE` and never appear
    persistently usable.
11. Tests prove every mismatch independently, including the workload and [model](../../../../glossary.md#term-model)-plan `SESSION`
    cases, and prove inspection invokes no decoder, candidate enumeration/decision construction,
    preparation, measurement, execution, or Runtime work.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
12. Workload evidence summaries preserve redacted provenance/source/context/weights, measured raw
    samples, summaries, and winner; hits fabricate no candidate rows.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight))</span>
13. Complete-plan evidence summaries preserve redacted provenance/source/correctness actions,
    measured raw samples, summaries, and winner; hits fabricate no correctness/candidate rows.
14. Evidence summarization performs zero filesystem/collaboration calls and returns detached
    immutable views.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [views glossary](../../../../glossary.md#term-view-meanings))</span>
15. Golden pre-refactor bytes still load/inspect; identical entries encode byte-for-byte
    identically; all existing task-0001/task-0002 tests pass unchanged.
16. Public-shape/import checks find no forbidden layers, reflection, serialization, raw/unchecked
    casts, generic object map, service locator, registry, or hidden static mutable state.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [serialization glossary](../../../../glossary.md#term-serialization); [service locator glossary](../../../../glossary.md#term-service-locator); [registry glossary](../../../../glossary.md#term-registry))</span>
17. Javadocs specify bounds, snapshots, redaction, units, status/failure semantics, I/O, mismatch
    order, decoder limitation, missing provenance, and side effects. A separate clean documentation
    context finalizes Javadocs, package prose, guide, glossary, and planning evidence.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
18. Final tuning tests, tuning Javadoc/rendered inspection, Markdown validation, exact 12-path/
    public-shape/import/status checks, empty staging, and both diff checks pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [validation glossary](../../../../glossary.md#term-validation))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

After focused development, [run](../../../../glossary.md#term-run) one final executable command:

```bash
./gradlew :tools:tuning:test
```

The three new test [owners](../../../../glossary.md#term-owner) cover both formats/input forms, invalid reasons, missing/I/O/read-only/
[snapshot](../../../../glossary.md#term-snapshot)/growth/[truncation](../../../../glossary.md#term-truncation-toward-zero) behavior, redaction/mismatch order, both rich evidence models,
immutability, unchanged bytes, and exact public/import [shape](../../../../glossary.md#shape). Existing eight suites remain the
task-0001/task-0002 [regression](../../../../glossary.md#term-regression) gate.

The mandatory separate clean documentation-focused context reuses that evidence unless it changes
executable Java or tests, then runs:

```bash
./gradlew :tools:tuning:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

Render/inspect `TuningInspection` and package summary. Validate Markdown [targets](../../../../glossary.md#term-target)/anchors, unique
headings, fences, LF/final newlines, whitespace, terminology, exact [scope](../../../../glossary.md#term-scope)/type placement/public
shape/imports, and synchronized task/master/roadmap status.

Repository-wide and other-[module](../../../../glossary.md#term-module) suites are deferred to [CI](../../../../glossary.md#term-continuous-integration-ci)/owning [checkpoints](../../../../glossary.md#term-checkpoint): this changes one
tool module and no [dependency](../../../../glossary.md#term-dependency), build, [backend](../../../../glossary.md#backend), architecture, or end-to-end boundary.

## Dependencies

- Tools/tuning 0001 — Complete; stable workload format, compact summary, and rich evidence.
- Tools/tuning 0002 — Complete; stable [model](../../../../glossary.md#term-model)-plan format, selected record, and rich evidence.
- [JDK](../../../../glossary.md#term-java-development-kit-jdk) read-only `FileChannel`, bounded arrays, and [SHA-256](../../../../glossary.md#term-sha-256) already used by the [module](../../../../glossary.md#term-module).

No Config, [Engine](../../../../glossary.md#term-engine), [CPU](../../../../glossary.md#term-cpu), [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare) change, [backend](../../../../glossary.md#backend) decoder, or public Phase-2 composition is
a prerequisite.

## Follow-up tasks

- Create no later tuning specification here. After 0003, separately planned Config Phase-2
  extension and Engine-owned CPU Phase-2 composition remain operational follow-ups.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- A later `tools/cli` diagnostic task may render these reports when its frontier reaches this API;
  it must not decode, repair, or move inspection into Runtime.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Migration/repair, persistent CPU compatibility, trust, remote caches, concurrent writers,
  relaxed correctness, and executable serialization remain future decisions.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>

## Documentation / [Javadoc](../../../../glossary.md#term-javadoc) / glossary impact

Apply General plus [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc style to Java, General plus Developer-guide style to [benchmarking](../../../../glossary.md#term-benchmarking),
and General plus [Planning](../../../../glossary.md#term-planning) style here. Use Example style only for a multi-step example.

Update benchmarking for Path/byte inspection, redaction, key-match-versus-decoder [authentication](../../../../glossary.md#term-authentication),
typed invalid reasons, evidence-only rich data, and no [CLI](../../../../glossary.md#term-cli)/[Engine](../../../../glossary.md#term-engine)/[Runtime](../../../../glossary.md#term-runtime) behavior. Update the
existing tuning glossary with a reusable **[tuning inspection](../../../../glossary.md#tuning-inspection)** term and correct the statement that
inspection is deferred. Do not claim a new [schema](../../../../glossary.md#term-schema) or public Phase-2 composition.

The documentation context records reasoned no-change conclusions for architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0008,
focused architecture pages, Config/Engine/[CPU](../../../../glossary.md#term-cpu)/[Prepare](../../../../glossary.md#prepare) guides/APIs, architecture tests,
[conformance](../../../../glossary.md#term-conformance)/integration, and other glossary terms.

## Architecture impact

Expected impact: None. This is a read-only [view](../../../../glossary.md#term-view-meanings) of state already owned by `tools/tuning`; it
changes no edge, [owner](../../../../glossary.md#term-owner), [lifecycle](../../../../glossary.md#lifecycle), [schema](../../../../glossary.md#term-schema), [Runtime](../../../../glossary.md#term-runtime) behavior, or [backend](../../../../glossary.md#backend) authority. Stop if an
authoritative change is required.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the isolated implementation agent for Synaptik tools/tuning task 0003. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the tuning master plan, and
docs/planning/tools/tuning/tasks/0003-read-only-cache-plan-and-evidence-inspection.md in full, plus
tasks 0001/0002, current tuning source/tests, documentation rules/profiles, benchmarking, and
relevant glossary terms. Implement exactly the Ready specification within its 12-path ceiling.
Preserve both cache formats and all existing behavior. Stop for any schema, architecture,
dependency, decoder, mutation, public-surface, or scope conflict.

After the final tuning test stabilizes, hand the same worktree, diff, and evidence to a distinct
clean documentation-focused context. It independently finalizes Javadocs/package prose,
benchmarking, glossary, planning evidence, and documentation validation without repeating Java
tests unless executable behavior changes or a concrete stale-evidence risk is recorded.

Update this task's decisions, limitations, evidence, notes, completion summary, and status only
after every criterion and documentation pass succeed.
```

## Local decisions

- Inspect both formats together because they share owner, bounds, checksum, read-only semantics,
  redaction policy, and decoder limitation.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [checksum glossary](../../../../glossary.md#term-checksum))</span>
- Use one public field-free namespace with nested [immutable](../../../../glossary.md#term-immutable) values to keep the public root small.
- Support both Path and `byte[]`: Path is operational; bytes support already-loaded diagnostics
  and deterministic tests. Both inspect detached bounded snapshots.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results))</span>
- Represent malformed content as `INVALID`, reserving `IOException` for snapshot acquisition;
  operational loaders remain fail-closed exceptions.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Redact opaque values to length/SHA-256/schema. Defensive raw copies would still disclose private
  identity and decision payloads unnecessarily.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Compare only stored key fields plus the expectation's declared reuse [scope](../../../../glossary.md#term-scope). A `PERSISTENT` key
  match requires decoder and never means compatible; either kind of `SESSION` expectation is
  explicitly ineligible for file reuse.
- Keep rich evidence as a separate value-to-value summary; files cannot supply its raw data.
- Preserve [canonical](../../../../glossary.md#term-canonical-form) file order rather than map iteration order.
- Do not add repository-wide tests because no shared boundary changes.

## Known limitations

- [SHA-256](../../../../glossary.md#term-sha-256) reports equality/difference but does not authenticate an untrusted writer.
- Same-length concurrent rewrite is assessed only through read bytes/checksum; no lock or stable
  filesystem snapshot is promised.<span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Key matches still require current decoder and fresh later preparation; inspection cannot prove
  legality, freshness, or executability.
- Workload files omit model/profile and occurrence provenance. Both files omit reuse scope and raw
  samples; the model-plan file also omits correctness actions.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [occurrence glossary](../../../../glossary.md#term-occurrence); [provenance glossary](../../../../glossary.md#provenance); [scope glossary](../../../../glossary.md#term-scope); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Reports are Java values only; rendering, CLI, migration, repair, and remote inspection remain
  outside scope.<span data-glossary-navigation="true"> ([CLI glossary](../../../../glossary.md#term-cli))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning)-only evidence at [Ready](../../../../glossary.md#term-ready) creation:

- Read the authoritative architecture, focused tuning and Runtime/Prepare explanations, planning
  guide/roadmap/master plan, completed tasks 0001/0002, documentation rules/profiles,
  benchmarking, relevant glossary entries, tuning build, and current tuning production/tests.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [benchmarking glossary](../../../../glossary.md#term-benchmarking))</span>
- Source audit confirmed both schema-1 formats, bounds, fields, order, checksum/publication,
  loader TOCTOU checks, evidence contents, public dependencies, and decoder ownership summarized
  above.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [checksum glossary](../../../../glossary.md#term-checksum); [publication glossary](../../../../glossary.md#term-publication); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Local Markdown target/fragment validation passed for all three changed files; headings are
  unique and backtick/tilde fences are balanced.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- LF/final-newline and trailing-whitespace validation passed for all three changed files.
- The exact changed-path allowlist contains only this task, the tuning master plan, and the
  roadmap. The staging area is empty.
- `git diff --check` and `git diff --cached --check` passed; the new untracked task separately
  passed the trailing-whitespace/final-newline checks because ordinary `git diff --check` does not
  include untracked files.
- Task, master-plan row/current-status prose, and roadmap frontier/status references all report
  task 0003 as `Ready`; no stale tuning-0003 `Draft` reference remains.

Implementation and documentation evidence:

- Implementation context `01a0b9d1-23ad-7a20-b797-bdf734f09468` added the exact field-free
  `TuningInspection` namespace, its ten public operations and nineteen public nested values, the
  shared authoritative parser diagnostics, and the three focused test owners.
- Its initial final `./gradlew :tools:tuning:test` run passed 11 suites and 68 tests with zero
  failures, errors, or skips. Documentation context <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`01a0b9e2-dd7a-7771-926f-7b109443e5ca`
  performed the initial independent pass using that evidence.
- After that pass, the implementation context corrected the malformed-input report invariant so a
  checksum-valid non-positive raw artifact schema is reported as zero rather than passed into an
  invalid public report value. It added negative-schema regression assertions for both cache
  formats and reran <span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant); [artifact glossary](../../../../glossary.md#term-artifact); [regression glossary](../../../../glossary.md#term-regression); [cache glossary](../../../../glossary.md#term-cache))</span>`./gradlew :tools:tuning:test`: 11 suites and 68 tests passed with zero
  failures, errors, or skips.
- Documentation context `01a0b9e2-dd7a-7771-926f-7b109443e5ca` then performed this final targeted
  re-review, clarified the public schema-report Javadoc, and reused the fresh post-correction test
  evidence without duplicating the Java suite.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The documentation context independently reviewed the implementation and tests, finalized the
  General + API/Javadoc, Developer-guide, Planning, and applicable Example-profile content, and
  generated final module Javadoc with <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>`./gradlew :tools:tuning:javadoc`; the final run completed
  successfully without warnings.
- Generated `TuningInspection` nested-type pages and the tuning package summary were inspected for
  bounds, nullable status invariants, redaction, decoder-required and `SESSION` semantics,
  nanosecond units, missing-file/`IOException` behavior, detachment, and non-mutation boundaries.
- Final targeted validation checked Markdown local targets and fragments, unique headings,
  balanced fences, LF/final newlines, trailing whitespace, terminology, synchronized task/master/
  roadmap status, exact 12-path allowlist, ten-operation/nineteen-type public shape, imports,
  empty staging, <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [shape glossary](../../../../glossary.md#shape))</span>`git diff --check`, and `git diff --cached --check`.
- Comment-aware review confirmed that final documentation-context Java edits changed no executable
  token and that the documentation context did not modify either corrected regression test. The
  fresh post-correction 68-test evidence therefore remains current.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token))</span>
- No architecture/ADR/focused-architecture update is needed because inspection remains a read-only
  view inside the existing tuning owner and changes no dependency, lifecycle, schema, or backend
  authority. Config, Engine, CPU, and Prepare APIs/guides remain accurate because the API composes
  with none of them and adds no public Phase-2 orchestration. Gradle and architecture tests need no
  change because no module edge or build rule changed; conformance and integration tests need no
  change because no backend or end-to-end behavior changed. Other task specifications and unrelated
  glossary terms remain accurate and outside this capability's scope.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [view glossary](../../../../glossary.md#term-view-meanings); [owner glossary](../../../../glossary.md#term-owner); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [scope glossary](../../../../glossary.md#term-scope))</span>

## Implementation notes

- Kept the two schema-1 cache formats byte-for-byte stable while routing operational loading and
  read-only inspection through one parser per format. Operational loaders continue to fail closed
  with <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>`IOException`; inspection returns typed invalid reports for malformed bounded snapshots.
- Added Path and caller-owned-byte snapshot inspection, exact stored-key comparisons, explicit
  decoder-required and session-ineligible results, redacted summaries, and detached rich-evidence
  summaries without mutation, preparation, measurement, or execution.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [detached glossary](../../../../glossary.md#detached-model-results))</span>
- Documentation review expanded public record/canonical-constructor contracts, enum values,
  package-private read seam/parser diagnostics, package prose, benchmarking guidance, and the
  glossary's compact-cache-versus-rich-evidence distinction.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [benchmarking glossary](../../../../glossary.md#term-benchmarking))</span>
- Known limitations remain as documented: inspection does not authenticate writers or decisions,
  cannot provide a stable concurrent-filesystem snapshot, cannot recover omitted provenance or rich
  evidence, and provides no renderer, repair, migration, or runtime composition.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Completion summary

Implemented and documented the bounded read-only inspection capability within the exact 12-path
[scope](../../../../glossary.md#term-scope). Both compact formats retain their bytes and [fail-closed](../../../../glossary.md#term-fail-closed) operational behavior; the public [API](../../../../glossary.md#term-application-programming-interface-api)
reports only truthful structural/key facts and [detached](../../../../glossary.md#detached-model-results) redacted evidence. Focused tests, final
[Javadoc](../../../../glossary.md#term-javadoc), rendered-output inspection, documentation checks, public-[shape](../../../../glossary.md#shape)/import checks, scope and
status [synchronization](../../../../glossary.md#term-synchronization), staging, and diff [validation](../../../../glossary.md#term-validation) all passed. No unresolved issue or required
follow-up remains for task 0003; later [CLI](../../../../glossary.md#term-cli) rendering and public Phase-2 composition remain separate
planned capabilities.

Status: Complete
