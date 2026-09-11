# Task 0004: Opaque Backend Candidate-Batch and Selected-Decision Handoff

## Status

Complete

## Goal

Define the smallest backend-neutral Prepare contract that can carry one concrete backend's
complete candidate batch out of cold partition analysis and carry one already selected decision
back toward a repeated cold analysis without exposing or interpreting backend-private fields.

The first producer is the completed CPU 0010E exact/default FLOAT32/FLOAT64 OpenBLAS slice. Its
existing `CpuOpenBlasTuningBatch` and `CpuOpenBlasTuningDecision` remain the authoritative typed
values. Prepare adds only nominal opaque roles and one immutable partition-associated handoff:

```java
package io.github.pho001.synaptik.prepare.analysis;

public interface BackendTuningCandidateBatch {}

public interface BackendTuningDecision {}

public record BackendPartitionTuningHandoff<
        C extends BackendTuningCandidateBatch,
        D extends BackendTuningDecision>(
        PlannedPartition partition,
        C candidateBatch,
        Optional<D> selectedDecision) {}
```

The type parameters preserve the concrete backend's candidate and decision types for a caller
that already owns that typed collaboration. Shared Prepare and later tuning orchestration may
transport the values but may not inspect, downcast, compare, serialize, select, or otherwise
interpret them. The concrete backend remains responsible for constructing a complete valid batch
and for validating any supplied decision against a freshly generated batch during analysis.

The mental model is:

```text
concrete backend cold analysis
  -> complete backend-owned candidate batch
  -> Prepare-owned typed opaque handoff associated with one exact partition
  -> later tuning tooling measures and selects outside this task
  -> same handoff may carry the backend-owned selected decision
  -> concrete backend repeats cold analysis and validates or rejects it as a miss
```

## Scope

- Add the exact two method-free marker roles and one generic immutable handoff record shown in
  Goal under `io.github.pho001.synaptik.prepare.analysis`.
- Require `BackendTuningCandidateBatch` implementations to be immutable, complete, backend-owned
  candidate batches. The marker exposes no candidate collection, schema, signature, identity,
  route, thread, representation, resource, measurement, cache, or persistence method.
- Require `BackendTuningDecision` implementations to be immutable backend-owned selection
  references. The marker exposes no compatibility, matching, decoding, or application method.
- Make `BackendPartitionTuningHandoff` retain the exact non-null `PlannedPartition` and candidate-
  batch references, and a non-null `Optional` containing either no decision or one exact decision
  reference. It acquires no ownership and performs no backend work.
- Preserve the exact generic association between a concrete candidate-batch type and decision
  type without raw types, unchecked casts, `Object`, a generic map, or a string-keyed envelope.
- Make the handoff an immutable point-in-time transport value only. It does not mutate
  `PrepareContext`, `BackendPartitionAnalysis`, `BackendPreparationPlan`, or analysis inputs.
- Update the existing technically public unsupported-internal CPU 0010E values so
  `CpuOpenBlasTuningBatch` implements `BackendTuningCandidateBatch` and
  `CpuOpenBlasTuningDecision` implements `BackendTuningDecision` without changing their record
  components, schema, equality, validation, candidate generation, matching, or selection.
- Add focused Prepare tests for exact public shape, generic bounds, null validation, exact
  reference retention, optional-decision behavior, equality, and the absence of shared
  interpretation methods.
- Add focused native-free CPU compile/contract coverage proving the existing batch and decision
  can populate the handoff with their exact types and that no CPU behavior or value shape changed.
- Finalize affected Javadocs, package documentation, focused Prepare/backend explanation,
  glossary impact, and planning evidence in the mandatory separate clean documentation context.

## Out of scope

- measurement, benchmarking, timing samples, warmup, objective or constraint comparison,
  candidate ranking, winner selection, or search-budget enforcement
- workload extraction, occurrence weighting, canonical-signature deduplication, or a model-wide
  tuning session
- cache access, artifact decoding or encoding, schema migration, compatibility or corruption
  recovery, invalidation, serialization, file formats, atomic persistence, or rich evidence
- candidate or decision compatibility interpretation in Prepare; CPU continues to validate its
  decision against a freshly generated CPU batch and treats incompatibility as a miss
- adding a shared candidate identity, workload signature, route, thread count, representation,
  carrier, copy mask, resource geometry, provider, hardware, numerical, or determinism model
- exposing candidate enumeration or decision construction through the marker roles
- a backend registry, tuning registry, service locator, plugin discovery mechanism, callback
  framework, visitor, reflection, downcast, generic parameter language, or `Map<String, ?>`
- changing `PrepareContext`, `BackendPartitionPreparer`, `BackendPartitionAnalysis`,
  `BackendPreparationPlan`, finalization, slot assignment, `GraphPreparation`, schedule assembly,
  prepared execution, or Runtime contracts
- host or provider discovery, qualification, native queries, physical allocation, executable
  construction, generated-artifact cache access, or any hot-path work
- changing CPU route eligibility, heuristic selection, candidate order or contents, candidate or
  decision schema, selected route, resources, finalization, generated code, or execution
- BFLOAT16 OpenBLAS work, provider 0004, or CPU 0010D1; both remain blocked/deferred
- tools/tuning 0001 or Config 0006A implementation or detailed task specifications; both remain
  Draft after this task
- changes to Backend Contract, Config, Planning, Compiler, Runtime, Engine, OpenBLAS provider,
  build files, module dependencies, architecture rules, ADRs, backend conformance, or integration
  behavior

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/prepare`, Concrete backend modules, Performance evidence and optimization tooling,
  Prepare lifecycle, and Dependency rules
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Prepare master plan](../master-plan.md)
- completed Prepare tasks [0001](0001-backend-partition-analysis-and-resource-declaration.md),
  [0002](0002-backend-partition-finalization-handoff.md),
  [0003](0003-prepare-orchestration-and-validation.md), and
  [0003A](0003a-immutable-partition-local-dag-analysis-projection.md)
- [CPU 0010E candidate and decision contract](../../../backends/cpu/tasks/0010e-float32-float64-openblas-tuning-candidates-and-compatible-decisions.md)
- [Backend Contract master plan](../../backend-contract/master-plan.md)
- [Tuning master plan](../../../tools/tuning/master-plan.md)
- [Config master plan](../../config/master-plan.md)

## Architecture constraints

- `modules/prepare` owns the backend-neutral opaque transport contract and must not depend on the
  concrete CPU module or any other concrete backend.
- Concrete backends own candidate vocabulary, complete candidate generation, compatibility facts,
  route/resource validity, and validation and application of selected decisions.
- Prepare may associate and transport the exact values only. It must not inspect CPU route,
  thread, representation, candidate identity, workload signature, provider, or hardware fields.
- Tools/tuning later owns measurement, comparison, selection, persistence, corruption rejection,
  and rich evidence. This task must not pre-empt that consumer or invent its request/cache schema.
- Ordinary cache-free and heuristic preparation remains correct. A handoff and selected decision
  are optional to the lifecycle and cannot create eligibility or weaken numerical/determinism
  requirements.
- Backend analysis remains deterministic from explicit immutable inputs. Any selected decision is
  validated by the concrete backend against a fresh complete batch before it can affect the plan.
- The handoff is cold preparation state. Runtime and the execution hot path must never receive or
  inspect candidate or decision values.
- Backend Contract remains closed and receives no tuning or Prepare collaboration.
- No module dependency, architecture rule, or prepared lifecycle order changes. Stop if the
  implementation needs one.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.prepare.analysis` — public backend-neutral cold analysis contracts.
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas` — existing
  technically public unsupported-internal CPU candidate and decision values.

Packages added or changed:

- No package is added.
- `prepare.analysis` gains the two nominal opaque roles and the immutable typed handoff.
- The CPU OpenBLAS package changes only to adopt those roles on its existing values.

Type placement:

- `io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch` — method-free nominal
  role for one backend-owned immutable complete candidate batch.
- `io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision` — method-free nominal role
  for one backend-owned immutable selected-decision value.
- `io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff` — immutable exact-
  partition association of one typed opaque batch and an optional same-collaboration decision.
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasTuningBatch`
  — remains CPU-owned and adopts only the candidate-batch role.
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasTuningDecision`
  — remains CPU-owned and adopts only the selected-decision role.

Prepare tests mirror `prepare.analysis`. CPU tests remain in the existing mirrored OpenBLAS
package. No root-package facade or generic `tuning`, `util`, `common`, or registry package is
added.

## Affected files

Expected production and test paths:

- add `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/BackendTuningCandidateBatch.java`
- add `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/BackendTuningDecision.java`
- add `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/BackendPartitionTuningHandoff.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/package-info.java`
- add up to two focused tests under
  `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/analysis/`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningBatch.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningDecision.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/package-info.java`
- add at most one focused assertion owner under the existing mirrored CPU OpenBLAS test package;
  prefer extending `CpuOpenBlasTuningBatchTest`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`

Expected documentation and planning paths:

- `docs/architecture/runtime-prepare-backend-boundary.md` — focused current-status and opaque-
  handoff explanation only
- `docs/backend-guide/partition-preparer.md`
- `docs/backend-guide/cpu-backend.md` only if the marker adoption makes its current 0010E
  handoff explanation incomplete
- `docs/glossary.md` only if existing candidate-generator and selected-tuning-decision entries do
  not cover the final reusable distinction
- this task
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a concrete contradiction is found: `ARCHITECTURE.md`, ADRs,
Backend Contract/Config/tuning master plans, completed Prepare and CPU 0010E task history,
Runtime/Engine APIs, Gradle files, architecture tests, conformance tests, and integration tests.

## Maximum scope

At most 18 paths:

- 7 Prepare production/test paths;
- 5 CPU production/test paths;
- 3 explanatory documentation paths; and
- 3 Prepare task/master/roadmap planning paths.

The glossary may replace, but not increase, the three-path explanatory-documentation allowance.
Do not modify more than one existing CPU focused test owner. If another production type, module,
dependency, build path, public collaboration, or lifecycle mutation is required, stop and return
the task to planning.

## Acceptance criteria

1. The exact two method-free marker interfaces and three-component generic handoff record from
   Goal exist in `prepare.analysis`; no additional public production type is added.
2. The handoff's generic bounds accept only a `BackendTuningCandidateBatch` and
   `BackendTuningDecision`, retain the exact supplied references, snapshot no backend state, and
   reject a null partition, batch, optional container, or contained decision.
3. The handoff permits an absent decision and one present typed decision. Record equality,
   `hashCode`, and `toString` follow exactly the three declared components.
4. The marker roles expose no method and shared Prepare exposes no candidate collection,
   identity, compatibility, schema, matching, selection, serialization, or route/resource field.
5. `CpuOpenBlasTuningBatch` and `CpuOpenBlasTuningDecision` implement only their corresponding
   roles. Their record components, constructors, methods, constants, nested types, equality,
   schema values, and validation remain unchanged.
6. A native-free CPU test constructs a typed `BackendPartitionTuningHandoff` from the existing
   CPU batch and absent/present CPU decision and proves exact reference retention. No CPU
   reflection, downcast, raw type, or unchecked cast is introduced.
7. CPU candidate generation, portable-first order, all eligible OpenBLAS representation/thread
   candidates, exact decision matching/miss behavior, heuristic fallback, retained selected plan,
   resource declarations, finalization, and execution remain unchanged.
8. Prepare performs no measurement, selection, persistence, corruption recovery, cache access,
   host/provider discovery, qualification, allocation, executable work, or Runtime work.
9. No `Object` payload, generic map, string-keyed attributes, reflection, registry, service
   locator, plugin discovery, callback framework, or public switch over concrete backends exists.
10. `modules/prepare` has no concrete-backend dependency, and Backend Contract, Config, Planning,
    Compiler, Runtime, Engine, provider, Gradle, architecture, conformance, and integration
    contracts remain unchanged.
11. Provider 0004 and CPU 0010D1 remain blocked/deferred; CPU 0010E and Prepare 0004 are
    Complete; tools/tuning 0001 is the next Draft planning frontier, and Config 0006A remains
    Draft. Neither downstream task has a detailed task specification.
12. A separate clean documentation-focused context independently reviews the implementation and
    tests, finalizes affected Javadocs/package prose and focused explanations, records reasoned
    no-change conclusions for excluded documentation, and does not repeat successful Java tests
    unless executable behavior changes or a concrete stale-evidence risk is recorded.
13. Focused/final module tests, Prepare and CPU Javadocs, rendered-page inspection, changed-
    Markdown links/anchors/headings/fences/newlines/whitespace, exact path/public-shape/package/
    dependency/status/order checks, and `git diff --check` pass before completion.

## Tests / validation

Implementation-focused tests while developing:

```bash
./gradlew :modules:prepare:test \
  --tests 'io.github.pho001.synaptik.prepare.analysis.*Tuning*Test'
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasTuningBatchTest \
  --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest
```

After executable Java stabilizes, run one final affected-module command:

```bash
./gradlew :modules:prepare:test :backends:cpu:test
```

The documentation-focused context reuses that successful executable evidence unless it changes
Java behavior or tests, then runs:

```bash
./gradlew :modules:prepare:javadoc :backends:cpu:javadoc
git diff --check
git status --short -uall
```

Render and inspect the three new Prepare declarations and the two affected CPU types. Validate
every changed Markdown file for local targets and anchors, unique headings, balanced backtick and
tilde fences, LF/final newlines, trailing whitespace, terminology, exact scope, package placement,
public shape, dependency direction, and synchronized statuses. Inspect the compiled public shape
through automated tests; do not add manual reflection or `javap` steps when those tests cover it.

Repository-wide validation, including architecture tests, is required once after the final
documentation pass because this task adds a shared public Prepare contract consumed by a concrete
backend. Run:

```bash
./gradlew test
```

Backend-conformance and integration-specific suites need no separate command because the task
changes no backend behavior or end-to-end execution; the root test command remains the shared-
contract checkpoint. No real OpenBLAS library, measurement run, cache, or native checkpoint is
required.

## Dependencies

- Prepare 0001–0003A — Complete.
- CPU 0010E — Complete and the authoritative first typed candidate/decision producer and
  consumer.
- Backend Contract 0001–0004 — Complete and closed; no new dependency or contract is needed.
- The existing Prepare-to-Planning and CPU-to-Prepare dependency directions already permit the
  exact `PlannedPartition` association and marker adoption.

Provider 0004 and CPU 0010D1 are not dependencies. Tools/tuning 0001, Config 0006A, Engine, and an
operational measurement path remain downstream.

## Follow-up tasks

- Tools/tuning 0001 remains the next Draft tuning task after Prepare 0004. A later clean planning
  pass may make it Ready only when stable model identity and an operational cold measurement path
  also exist. It owns workload extraction/deduplication, measurement, selection, explicit cache
  compatibility and persistence, corruption rejection, and rich evidence.
- Config 0006A remains Draft after an implemented stable tuning consumer and continues to own
  only immutable public request inputs.
- CPU 0016 later generalizes the CPU-owned candidate/decision contract across implemented peer
  routes after tools/tuning 0001. It does not move CPU vocabulary into Prepare.
- Any need for multi-batch partition aggregation, candidate enumeration, measurement callbacks,
  or Engine composition is a separate downstream contract justified by the first operational
  consumer; do not add it speculatively here.

## Architecture impact

Expected impact: None.

This task implements the future narrow opaque-candidate orchestration boundary already authorized
by the architecture. It changes neither ownership nor dependency direction. If implementation
requires Prepare to inspect backend fields, a concrete-backend dependency from Prepare, a tuning
service or registry, Runtime state, or another architecture rule, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik Prepare task 0004. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the Prepare master plan, and
docs/planning/modules/prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md
in full, plus its directly referenced completed Prepare and CPU 0010E contracts and current
affected source/tests. Implement exactly the approved task within its 18-path ceiling. Stop for any
architecture, dependency, shared interpretation, tuning, persistence, Runtime, or scope conflict.

After executable code and the final affected-module tests stabilize, hand the exact diff and test
evidence to a distinct clean documentation-focused context in the same overall change. That
context must follow docs/developer-guide/documentation-rules.md, independently inspect the
implementation/tests, finalize affected Javadocs/package documentation, focused explanations,
glossary impact, task/master/roadmap evidence, and documentation validation, and avoid repeating
successful Java suites unless executable behavior changes or a concrete stale-evidence risk is
recorded. Do not mark 0004 Complete until that pass and every specified gate succeed.
```

## Local decisions

- Use two method-free nominal roles rather than a shared candidate or decision model. This keeps
  CPU's versioned schema, workload signature, candidate identities, compatibility, and matching
  logic in CPU while giving shared code a type-safe boundary that is not `Object`.
- Use one generic handoff record rather than changing `PrepareContext`,
  `BackendPartitionAnalysis`, or `BackendPreparationPlan`. The current CPU slice has one optional
  eligible batch per analyzed partition, and a broader aggregation contract has no stable
  operational consumer yet.
- Associate the handoff with the exact `PlannedPartition` reference because partition identity is
  already a stable Prepare-visible fact. Do not add a tuning-specific identifier or infer
  workload identity in shared code.
- Retain an `Optional<D>` decision so the same immutable shape represents candidate discovery and
  a later selected-decision return. Absence is ordinary and cannot make tuning necessary for
  correctness.
- Do not encode a type token, `Class<?>`, visitor, matcher, decoder, or factory. The caller that
  composes a concrete backend retains the generic types; the backend validates meaning.
- Limit CPU adoption to `implements` clauses and documentation/tests. Existing CPU 0010E plan and
  input components already retain and consume the authoritative values, so duplicating them in a
  shared plan or mutating analysis inputs would create two sources of truth.

## Known limitations

- The first handoff carries exactly one candidate batch for one exact partition. It does not
  aggregate multiple tunable workloads or model occurrences.
- The marker roles deliberately provide no candidate enumeration or decision-construction API.
  The operational measurement owner must justify any later collaboration rather than Prepare
  speculating about CPU fields.
- No model identity, objective, budget, representative profile, cache schema, file format,
  measurement evidence, or Engine facade exists in this task.
- Only CPU 0010E's exact/default FLOAT32/FLOAT64 bare rank-two MATMUL values adopt the roles.
  Other routes and BFLOAT16 remain outside this handoff until their owning tasks complete.

## Validation evidence

- Implementation-focused validation supplied by the implementation context passed before the
  documentation review: the focused Prepare and native-free CPU contract tests passed, followed
  by one final `./gradlew :modules:prepare:test :backends:cpu:test` run with 46 Prepare tests and
  992 CPU tests, zero failures, zero errors, and 28 expected CPU skips. This documentation review
  changed no executable Java behavior or tests, so it reused that evidence as required.
- `./gradlew :modules:prepare:javadoc :backends:cpu:javadoc` passed. Prepare emitted no Javadoc
  warnings. CPU emitted 94 pre-existing warnings in unrelated preparation-plan types plus the two
  expected incubating-module warnings; none named the five affected tuning types.
- Generated HTML for `BackendTuningCandidateBatch`, `BackendTuningDecision`,
  `BackendPartitionTuningHandoff`, `CpuOpenBlasTuningBatch`, and
  `CpuOpenBlasTuningDecision` was rendered to visible text and inspected. The pages show the two
  method-free roles, bounded type parameters, three record components, exact-reference and
  no-ownership wording, null/failure semantics, and CPU-owned compatibility semantics. Local
  Javadoc links resolve. No graphical in-app browser was available in documentation context
  `01a0919b-21a4-7e61-bc1d-1eab58956c4d`; generated HTML and rendered text were inspected
  directly instead.
- Every changed Markdown file passed local target and anchor checks, unique-heading checks,
  balanced backtick/tilde-fence checks, LF/final-newline and trailing-whitespace checks, and a
  focused terminology/status/order/dependency review. The final diff contains 16 paths, including
  exactly three explanatory documentation paths, and no tools/tuning task specification.
- `git diff --check` passed. The changed-path audit found no architecture contract, ADR, build,
  dependency, backend-conformance, integration, Runtime, Engine, Config, Compiler, Planning,
  Backend Contract, or provider change.
- The single shared public-contract checkpoint `./gradlew test` passed. The resulting 464 XML
  suites report 3,008 tests, zero failures, zero errors, and 28 skips.

## Implementation notes

- Added exactly two public method-free roles and one public three-component record in
  `prepare.analysis`. Reflection-backed shape tests lock the empty declared method sets, exact
  generic bounds, exact `PlannedPartition`/candidate/optional-decision components, and absence of
  additional public analysis declarations.
- The handoff constructor rejects every null reference, retains the exact supplied partition,
  batch, and present decision references, accepts an absent decision, and otherwise relies only on
  ordinary record value methods over its three components. Prepare never inspects or downcasts a
  backend value.
- CPU adoption is nominal only: the existing batch and decision implement their corresponding
  roles. Their components, constructors, nested types, constants, methods, schemas, generation,
  compatibility matching, fallback, preparation, finalization, and execution behavior are
  unchanged. The existing CPU focused test now also proves typed native-free handoff construction
  and exact reference retention.
- Finalized the affected Prepare and CPU type/package Javadocs. The focused runtime/Prepare
  boundary, partition-preparer guide, and glossary now describe the current opaque transport while
  preserving measurement, selection, caching, persistence, and model orchestration as downstream
  work.
- Reviewed `ARCHITECTURE.md`, ADR impact, performance-evidence guidance, the CPU backend guide,
  completed Prepare and CPU task contracts, other module plans/APIs, architecture/conformance/
  integration tests, and build/dependency files. No changes were needed: the architecture already
  authorizes the narrow opaque boundary; the performance guide correctly keeps operational tuning
  planned; the CPU guide already documents CPU 0010E semantics; no decision or dependency changed;
  and no executable backend or end-to-end behavior changed.
- The glossary did require a correction because its prior entries still described shared opaque
  transport as future work. It replaces, rather than increases, the three-path explanatory-
  documentation allowance.
- Prepare 0004 is Complete. Tools/tuning 0001 is the next Draft planning frontier; no task
  specification or tuning/configuration implementation was created.

## Completion summary

Completed changes:

- introduced the minimal typed opaque Prepare transport and adopted its roles on the existing CPU
  OpenBLAS batch and decision values without changing backend schemas or behavior;
- added focused public-shape, null/reference, record-value, and typed CPU handoff coverage;
- finalized affected Javadocs, package prose, three focused explanatory documents, and synchronized
  Prepare/roadmap planning status.

Changed or created paths (16):

- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/BackendTuningCandidateBatch.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/BackendTuningDecision.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/BackendPartitionTuningHandoff.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/package-info.java`
- `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/analysis/BackendPartitionTuningHandoffTest.java`
- `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/analysis/AnalysisPublicShapeTest.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningBatch.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningDecision.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/package-info.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningBatchTest.java`
- `docs/architecture/runtime-prepare-backend-boundary.md`
- `docs/backend-guide/partition-preparer.md`
- `docs/glossary.md`
- `docs/planning/modules/prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md`
- `docs/planning/modules/prepare/master-plan.md`
- `docs/planning/roadmap.md`

Validation performed: focused and final affected-module evidence was reused unchanged; Prepare and
CPU Javadocs passed and all five affected generated pages were inspected; all changed Markdown and
scope/status checks passed; `git diff --check` passed; and the one root `./gradlew test` checkpoint
passed with 3,008 tests, zero failures, zero errors, and 28 skips across 464 XML suites.

Unresolved issues: none for Prepare 0004. The unavailable graphical browser did not prevent direct
generated-HTML and rendered-text inspection. Provider 0004 and CPU 0010D1 remain intentionally
blocked/deferred. Tools/tuning 0001 remains the required next Draft planning frontier; its task
specification and implementation are follow-up work outside this change.

Documentation review context: `01a0919b-21a4-7e61-bc1d-1eab58956c4d`.

Status: Complete
