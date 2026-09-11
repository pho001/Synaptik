# Task 0010E: FLOAT32/FLOAT64 OpenBLAS Tuning Candidates and Compatible Decisions

## Status

Complete

## Goal

Make the concrete CPU backend the authoritative producer and consumer of one typed, versioned
candidate contract for the completed exact/default FLOAT32/FLOAT64 OpenBLAS MATMUL route. For one
eligible workload, CPU must expose the complete safe portable candidate and every complete valid
OpenBLAS thread/representation candidate, together with canonical compatibility facts. CPU may
consume one explicitly supplied selected decision only when its workload, candidate schema, and
exact candidate identity still match; otherwise ordinary preparation uses its current safe
heuristic path.

This is the first executable stage of the repaired tuning sequence because all of its inputs are
already owned by completed CPU tasks. Prepare 0004 cannot define an opaque handoff until a
concrete producer exists, and tools/tuning 0001 cannot define persistent compatibility for values
whose owning backend has not stabilized their meaning.

The mental model is:

```text
completed CPU lowering + qualification + route/resource facts
  -> one versioned canonical workload signature
  -> complete portable and OpenBLAS candidates
  -> optional exact selected decision
       matching: select that already-valid candidate
       absent or incompatible: retain current safe heuristic selection
  -> immutable CPU preparation plan
```

## Scope

- Add one technically public, unsupported-internal immutable CPU candidate batch in
  `internal.route.nativeblas.openblas`. It owns the candidate-schema version, exact workload
  signature, deterministic candidate order, and complete candidate values required by later
  opaque orchestration.
- Add one technically public, unsupported-internal immutable selected-decision value containing
  the exact candidate-schema version, workload signature, and selected candidate identity. It
  contains no timing sample, objective, cache path, serialized bytes, executable, provider,
  segment, workspace instance, or Runtime state.
- Cover only the current positive rank-two, same-type FLOAT32 or FLOAT64, exact/default bare
  MATMUL boundary from CPU 0010–0010D. Preserve all eight 0010B representation masks and every
  eligible 0010C positive OpenBLAS thread candidate.
- Include the complete safe portable alternative in every emitted batch. A batch is not valid if
  it exposes only native choices.
- Define canonical typed compatibility facts for operation kind and fixed attributes; input,
  accumulation, and output types; exact Shapes and resolved layouts; carrier and copy-mask facts;
  numerical and determinism mode; qualified target and provider/binary reuse scope; CPU hardware
  identity supplied as immutable cold input; CPU concurrency capacity; portable strategy and
  worker facts; OpenBLAS thread counts; expected-run/workload-cohort facts; and every CPU route,
  generated-artifact, cost-policy, qualification, and candidate-schema version that can change
  candidate meaning.
- Represent persistent compatibility only when CPU 0010D supplies a persistent binary identity.
  A session-only qualification may produce and consume a decision within that exact session but
  must expose no persistent compatibility projection.
- Derive candidate identities deterministically from typed CPU-owned fields. Identity and order
  must not depend on object identity, hash-map iteration, native address, live thread state,
  timing, or platform inference.
- Generate and validate candidates during cold CPU analysis from the same already-derived
  lowering, representation, resource, qualification, thread, and cost facts used by the current
  selector. Candidate generation must not create a second graph interpreter or disagree with
  actual route eligibility.
- Accept an optional decision through CPU analysis inputs. Validate it against the freshly
  generated batch before selecting its candidate. A missing or incompatible decision is an
  explicit miss and uses the unchanged safe heuristic; malformed CPU values fail construction.
- Retain the selected typed compatibility facts in the CPU preparation plan far enough for tests
  and later Prepare transport to prove the exact choice. Finalization and Runtime receive no
  search, cache, measurement, or reselection responsibility.
- Add deterministic native-free tests for field completeness, immutability, ordering, equality,
  session/persistent scope, candidate generation, selected-decision hits and misses, safe
  heuristic fallback, resource agreement, and unchanged prepared execution semantics.
- Finalize affected Javadocs, OpenBLAS package documentation, CPU backend guide, glossary impact,
  this task, CPU master plan, and roadmap in the mandatory separate documentation-focused pass.

## Out of scope

- BFLOAT16 OpenBLAS candidates or CPU 0010D1. Provider 0004 and CPU 0010D1 remain a
  blocked/deferred optional side branch. This task must not add a placeholder BFLOAT16 capability
  bit that could later be mistaken for proof.
- `cblas_sbgemm`, conversion staging, BFLOAT16 input widening, a temporary FLOAT32 output matrix,
  result conversion, platform-based BFLOAT16 inference, or any relaxation of Model MATMUL
  semantics.
- Prepare 0004, tools/tuning 0001, Config 0004–0006A, Engine, or any shared/public handoff. This
  task changes only the CPU module plus its focused explanatory and planning documentation.
- Measurement, benchmarking, objective/budget comparison, candidate timing, workload
  deduplication across model occurrences, cache loading, serialization, corruption recovery,
  atomic persistence, or rich evidence. Those belong to tools/tuning.
- A public user request, public CPU builder, Config facade, automatic composition, or Engine
  lifecycle.
- Relaxed or fast-math candidates. The complete task is exact/default only and has no dependency
  on Config 0006.
- Planning ownership or cost changes. Planning continues to select only `BackendId("cpu")` and
  never sees route, representation, thread, or candidate vocabulary.
- New OpenBLAS symbols, provider API/source/test changes, discovery, qualification probes, binary
  inspection, thread coordination, provider invocation, MATMUL representation expansion, or new
  execution behavior.
- Batch, broadcast, rank-one, empty, zero-contraction, epilogue, packing, persistent weights,
  mixed type, integral type, sparse, quantized, or complex MATMUL.
- Generated bytecode, emitter, Class-File schema, generated-artifact persistence, performance
  evidence, module dependencies, Gradle, architecture, ADR, backend-conformance, or integration
  changes.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially concrete backend candidate
  ownership, opaque tuning orchestration, staged preparation, CPU routes, and Runtime prohibitions
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU master plan](../master-plan.md)
- [CPU 0010 narrow OpenBLAS route](0010-narrow-openblas-blas-compatible-native-route.md)
- [CPU 0010B bounded representations](0010b-bounded-openblas-matmul-representation-expansion.md)
- [CPU 0010C coordinated thread candidates](0010c-coordinated-openblas-thread-candidates-and-shared-cpu-thread-budget.md)
- [CPU 0010D qualification and target fingerprinting](0010d-installed-openblas-qualification-and-target-fingerprinting.md)
- [Blocked provider 0004 direct BFLOAT16 proof](../../openblas-provider/tasks/0004-optional-direct-bfloat16-output-gemm-capability.md)
- [Prepare master plan](../../../modules/prepare/master-plan.md)
- [Tuning master plan](../../../tools/tuning/master-plan.md)
- [Config master plan](../../../modules/config/master-plan.md)

## Architecture constraints

- CPU owns the typed candidate vocabulary, canonical backend/workload compatibility facts,
  candidate validity, deterministic ordering, and validation/consumption of a selected decision.
- Shared Prepare may later transport the complete batch and decision opaquely, but it must not
  interpret CPU fields or select a route.
- Tools/tuning owns measurement, selection against objective and budget, persistent artifact
  encoding/loading/mutation, corruption and incompatibility rejection, and rich evidence. CPU
  must not implement those responsibilities here.
- Backend analysis remains deterministic from explicit immutable facts. It performs no native
  query, host discovery, measurement, file/cache access, or mutation.
- A selected decision can choose only a candidate in the freshly generated complete batch. It
  cannot create eligibility, bypass Model semantics, weaken determinism, change declarations, or
  authorize a late fallback.
- Safe heuristic and cache-free preparation remains correct. Runtime executes only the prepared
  result and never sees candidate, decision, cache, or measurement vocabulary.
- Portable BFLOAT16 support remains unchanged. No native BFLOAT16 candidate exists until the
  separate provider and CPU proof branch completes under its exact semantic gates.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.prepare`
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas`

Packages added or changed:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas` — adds the narrow
  typed candidate-batch and selected-decision contracts beside their owning route.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — accepts the optional cold decision
  and retains the exact selected compatibility facts; it does not gain tuning policy.

Type placement:

- `CpuOpenBlasTuningBatch` — technically public unsupported-internal immutable batch, with nested
  typed workload, hardware, reuse-scope, candidate-identity, and complete-candidate values.
- `CpuOpenBlasTuningDecision` — technically public unsupported-internal immutable exact selection
  reference validated only by CPU.
- `CpuPartitionAnalysisInputs` — owns explicit cold hardware/workload-cohort facts and the optional
  decision because these are inputs to deterministic CPU analysis.
- `CpuOpenBlasRouteSelector` — remains the sole owner that derives eligibility/cost candidates and
  now exposes the exact batch before applying either a matching decision or its safe heuristic.
- `CpuPartitionPreparationPlan` — retains only the selected CPU facts required to prove the
  prepared choice; it does not become a cache or evidence object.

Tests mirror the production packages. No new package is added.

## Affected files

Expected production and Javadoc paths:

- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningBatch.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningDecision.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelector.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`

Expected focused test paths:

- add `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningBatchTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelectorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`

Expected documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md` only if the final reusable candidate/decision distinction needs an entry
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the 17 exact paths above: eight production/Javadoc paths,
four CPU test paths, two explanatory-documentation paths, and three planning paths. It adds
exactly two production types and one test type, with no new package.

The final combined diff has 21 paths rather than changing that implementation ceiling:
CpuPartitionPreparerTest needed no edit, while the earlier clean planning pass intentionally
updated the five existing provider, Prepare, tuning, and Config planning records needed to repair
the acyclic sequence. Those planning-only reconciliations are part of the approved handoff; they
add no task specification, source, test, dependency, or behavior.

If truthful implementation needs Prepare, tuning, Config, Engine, provider, shared-module, build,
architecture, conformance, integration, generated-code, or another source/test path, stop and
return the task to planning.

## Acceptance criteria

1. One immutable versioned batch contains one exact canonical workload signature, the safe
   portable candidate, and every eligible OpenBLAS thread/representation candidate in stable
   deterministic order.
2. Candidate generation reuses the current authoritative CPU lowering, representation, resource,
   qualification, thread, and checked-cost facts; no second graph or eligibility interpretation
   exists.
3. The workload signature contains every typed compatibility field listed in Scope. Automated
   tests change each field independently and prove inequality or incompatibility.
4. Persistent compatibility is present only for an exact path-qualified persistent binary
   identity. Session-only decisions are bound to the exact session and cannot be projected or
   reused persistently.
5. Candidate identities distinguish portable strategy/resource facts and every OpenBLAS thread,
   representation, resource, and cost-policy meaning without native addresses or live objects.
6. A decision selects only the exact matching candidate in a freshly generated matching batch.
   Schema, workload, scope, session, or candidate mismatch is an explicit miss that retains the
   existing safe heuristic; malformed CPU-owned values fail construction.
7. The no-decision path selects byte-for-byte equivalent route/representation/thread facts to the
   current heuristic for the same inputs. Ineligible OpenBLAS workloads remain portable and emit
   no false native candidate.
8. Analysis and preparation perform no measurement, objective comparison, cache access,
   serialization, corruption handling, persistence, native query, or host/platform discovery.
9. Finalization and execution behavior, exact resource declarations, all eight representation
   masks, provider coordination, qualification, and no-late-fallback rules remain unchanged.
10. No BFLOAT16 OpenBLAS candidate, placeholder proof flag, `cblas_sbgemm`, conversion/staging
    path, relaxed semantic, or platform-derived native eligibility is added. Portable BFLOAT16 is
    unchanged.
11. Public supported CPU API, provider API, shared Prepare/Runtime, Planning, Config, Engine,
    dependencies, Gradle, generated schema/code, backend conformance, integration, and
    architecture remain unchanged.
12. A separate clean documentation-focused context independently reviews the implementation and
    tests, finalizes all affected Javadocs/package prose, CPU guide, glossary impact, task/master/
    roadmap evidence, and records reasoned no-change conclusions for excluded areas.
13. Focused/final CPU tests, CPU Javadoc and rendered-page inspection, Markdown links/anchors,
    unique headings, balanced fences, final newlines, trailing whitespace, exact path/public-
    internal surface/status/order checks, and `git diff --check` pass before completion.

## Tests / validation

Run focused tests while implementing. After executable Java stabilizes, run one final module
command:

```bash
./gradlew :backends:cpu:test
```

Ordinary tests are native-free and deterministic. No real OpenBLAS checkpoint is required because
the task changes candidate description and cold decision consumption, not qualification,
provider invocation, representation execution, or numerical behavior.

The documentation-focused context reuses the successful CPU result unless it changes executable
Java or tests, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git status --short -uall
```

It renders and inspects changed Javadocs; validates local Markdown files and anchors, unique
headings, balanced fences, LF/final newlines, terminology, exact scope, package placement, and
public/internal inventory; and confirms CPU 0010E is `Complete` only after all gates, CPU 0010D1
remains Blocked/deferred, Prepare 0004 is the next Draft stage, tools/tuning 0001 follows Prepare
0004, and neither later task has a detailed specification.

Repository-wide validation is deferred to the Prepare 0004 shared-contract checkpoint or CI.
This task changes one concrete backend module and no dependency, build, shared contract, or
architecture rule.

## Dependencies

- Complete CPU 0010, 0010A, 0010B, 0010C, and
  [0010D](0010d-installed-openblas-qualification-and-target-fingerprinting.md).
- Complete portable MATMUL, typed portable route, representation, generated-artifact identity,
  qualification, thread-candidate, cost, staged analysis/finalization, and package-inventory
  contracts already present in the CPU module.
- Complete OpenBLAS provider tasks 0001–0003 for the existing FLOAT32/FLOAT64 leaf.

All dependencies are complete. Provider 0004, CPU 0010D1, Prepare 0004, tools/tuning 0001, Config
0004–0006A, and Engine are not dependencies of this CPU-owned producer/consumer capability.

## Follow-up tasks

- Prepare 0004 is the next staged row. It transports CPU-owned batches and selected decisions
  opaquely and performs no measurement, compatibility interpretation, or persistence.
- Tools/tuning 0001 follows Prepare 0004. It owns exact/default workload extraction and
  deduplication, measurement, objective/budget selection, persistent cache encoding/loading/
  mutation, corruption/incompatibility rejection, and separate rich evidence.
- Config 0006A may later expose immutable public request inputs after the tuning consumer is
  stable. Config 0004–0006 are not forced through this first exact/default slice.
- CPU 0010D1 remains a blocked/deferred optional side branch. If it later completes, a versioned
  CPU follow-up must extend candidate compatibility with direct-BFLOAT16 proof facts explicitly.
- CPU 0016 later generalizes the proven contract across whichever peer routes are implemented.

## Architecture impact

Expected impact: None.

This task implements the concrete backend's already-authorized candidate and compatibility
ownership. If it requires shared Prepare interpretation, tuning persistence inside CPU, Runtime
selection, a provider change, public configuration, a module edge, or an architecture rule,
stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0010E. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, and
docs/planning/backends/cpu/tasks/0010e-float32-float64-openblas-tuning-candidates-and-compatible-decisions.md
in full, plus its directly referenced completed CPU/provider contracts and current affected
source/tests. Implement exactly the Ready task within its 17-path ceiling. Stop for any
architecture, Prepare, tuning, Config, provider, BFLOAT16, generated-code, public-API, or scope
conflict instead of inventing a boundary.

After executable code and the final CPU test stabilize, hand the exact diff and evidence to a
distinct clean documentation-focused agent in the same overall change. That agent must follow
docs/developer-guide/documentation-rules.md, independently inspect implementation/tests, finalize
affected Javadocs/package documentation, CPU guide, glossary impact, task/master/roadmap evidence,
and documentation validation, and avoid repeating successful Java suites unless executable
behavior changes or a concrete stale-evidence risk is recorded. Do not mark 0010E Complete until
that pass and every specified gate succeed.
```

## Local decisions

- Confirm the earlier CPU-first direction, but narrow the first task to the concrete backend's
  owned producer/consumer contract. Making Prepare or tuning first would require them to invent or
  interpret CPU candidate meaning.
- Keep the initial schema exact/default and FLOAT32/FLOAT64 only. This avoids a false dependency
  on relaxed Config and does not weaken or bypass the blocked BFLOAT16 semantic proof.
- Treat an incompatible external decision as a cache-style miss, not an analysis failure or
  permission to use a stale candidate. Malformed CPU-owned construction still fails immediately.
- Include the portable candidate in every batch so optional tuning cannot become necessary for
  correctness.
- Keep rich measurements and persistent encoding out of CPU. CPU supplies typed semantic values;
  tools/tuning later owns how evidence and compact artifacts are stored.

## Known limitations

- Only the current positive rank-two exact/default FLOAT32/FLOAT64 bare MATMUL route participates.
- The task creates an unsupported-internal CPU contract, not a public tuning or Engine API.
- No model-wide occurrence deduplication, measurement, persistence, or end-to-end plan selection
  exists until Prepare 0004 and tools/tuning 0001 complete.
- Session-only qualification can support same-session decisions but cannot create persistent
  compatibility.
- Native BFLOAT16 MATMUL remains unavailable; portable BFLOAT16 remains supported.

## Validation evidence

- Implementation context 01a09107-730a-7993-ac6d-7e970a92c35d reported focused
  candidate/selector/inventory tests passing, followed by the final CPU module test passing in 2
  minutes 33 seconds: 194 suites, 988 tests, 28 skips, zero failures, and zero errors. It also
  reported the Git whitespace check passing.
- Documentation context 01a0911b-25e4-7480-bf48-a353084e3f49 independently inspected every
  changed production and test source. The implementation is portable-first, enumerates 8
  realizable masks by every fitting configured thread count, restricts eligibility to positive
  rank-two same-type exact/default FLOAT32/FLOAT64 MATMUL, uses immutable graph-ID-free typed
  workload/candidate values, distinguishes session-only and persistent-binary qualification,
  treats absent or incompatible decisions as heuristic misses, and retains selected-plan resource
  agreement. No measurement, cache/persistence, host discovery, provider query/change, shared
  Prepare/Runtime interpretation, generated-code change, or native BFLOAT16 path was found.
- The same independent review found acceptance criterion 3 incomplete:
  CpuOpenBlasTuningBatchTest changes hardware compatibility and separately exercises schema and
  candidate misses, but no automated matrix changes every WorkloadSignature component
  independently and proves inequality or decision incompatibility. Repository search found no
  other construction or mutation of WorkloadSignature in tests.
- Documentation context 01a0911b-25e4-7480-bf48-a353084e3f49 finalized affected Javadocs,
  OpenBLAS package prose, the CPU backend guide, and glossary terminology without changing Java
  behavior or tests.
- The CPU Javadoc task passed in 3 seconds. It emitted 94 pre-existing warnings in unrelated or
  legacy constructor documentation; no warning names the new tuning batch, decision, selector, or
  the added canonical-constructor tags. Rendered pages for CpuOpenBlasTuningBatch and its nested
  values, CpuOpenBlasTuningDecision, CpuOpenBlasRouteSelector, CpuPartitionAnalysisInputs,
  CpuPartitionPreparationPlan, and the OpenBLAS package summary were inspected successfully.
- Correction implementation context 01a09122-987e-7023-874b-10d0969d7c16 added the complete
  field-by-field compatibility matrix requested by the first documentation review. It reported a
  focused three-class run passing 3 suites and 20 tests with zero skips, failures, or errors,
  followed by one final `./gradlew :backends:cpu:test` run passing 194 suites and 991 tests with
  28 skips and zero failures or errors. `git diff --check` also passed.
- Final documentation review context 01a0912d-4901-7bc3-8da3-73a452e3c1c6 first identified two
  remaining fail-closed evidence omissions: no exact `BoundaryStorageFact` component inventory and
  no explicit rejection of a non-three-entry boundary list. The correction added only those two
  assertions and reran the same focused and final CPU commands with the same final counts.
- Resumed final review in context 01a0912d-4901-7bc3-8da3-73a452e3c1c6 inspected both assertions
  and the complete surrounding matrix. Every mutable `WorkloadSignature` component and meaningful
  nested compatibility fact now has independent inequality plus decision-miss/safe-fallback
  coverage, while closed-schema or relationally coupled facts have explicit rejection evidence.
  Exact record-component inventories make additions fail closed. No executable Java changed after
  the final CPU run, and the prior CPU Javadoc/rendered-page evidence remains current because both
  corrections changed tests only.
- The repository Markdown validator passed all 10 changed Markdown files, including local targets
  and anchors, unique headings, balanced fences, LF endings, final newlines, and trailing
  whitespace. The final Git diff whitespace check passed.
- Exact inventory and order checks found 21 combined paths for the documented reason in Maximum
  scope; both new technically public types remain under the unsupported internal OpenBLAS package;
  provider 0004 and CPU 0010D1 remain blocked/deferred; Prepare 0004 is the next Draft planning
  frontier with no detailed task file; tools/tuning 0001 and Config 0006A remain staged Draft work
  with no detailed task files; and Runtime has no candidate-selection change.

## Implementation notes

- Added CpuOpenBlasTuningBatch and CpuOpenBlasTuningDecision beside the OpenBLAS route, and
  extended CPU analysis inputs, route selection, and the retained preparation plan to produce and
  consume the typed cold decision.
- Candidate enumeration reuses the existing portable plan, common access bindings, representation
  planner, qualification, thread candidates, resource declarations, and checked cost facts. The
  batch begins with portable and orders native candidates by representation enum order and
  configured-thread order.
- The safe heuristic is applied only after the complete batch is built. A matching decision may
  select any member, including portable or a native candidate not favored by the heuristic;
  incompatible values cannot create eligibility.
- Documentation no-change conclusions: ARCHITECTURE.md and focused architecture explanations
  remain accurate because the implementation realizes their existing concrete-backend ownership
  and Runtime prohibition; provider source/API and shared Prepare/Runtime, Config, Engine,
  Planning, generated code/schema, dependency/build, backend-conformance, and integration paths
  require no update because none of their contracts or behavior changed. The intentional planning
  edits keep provider 0004 and CPU 0010D1 blocked/deferred, retain Prepare 0004 then tools/tuning
  0001 as Draft, and place Config 0006A after the stable tuning consumer.

## Completion summary

- Completed changes: implemented the CPU-owned typed candidate batch and compatible-decision
  consumption; finalized its Javadocs, OpenBLAS package explanation, CPU backend guide, glossary
  terms, and the already-staged acyclic planning sequence.
- Files changed or created: eight permitted production/Javadoc paths, three changed/created
  focused test paths (the planned CpuPartitionPreparerTest needed no edit), the CPU backend guide,
  glossary, this task, CPU master plan, roadmap, and the intentional provider/Prepare/tuning/Config
  planning reconciliation paths already present in the handoff.
- Tests and validation: reused the final successful CPU module run from implementation context
  01a09122-987e-7023-874b-10d0969d7c16: 194 suites, 991 tests, 28 skips, zero failures/errors;
  its focused three-class run passed 3 suites and 20 tests with zero skips/failures/errors. CPU
  Javadoc passed and affected rendered pages were inspected in documentation context
  01a0911b-25e4-7480-bf48-a353084e3f49; resumed final context
  01a0912d-4901-7bc3-8da3-73a452e3c1c6 reused that current evidence and passed changed-Markdown,
  exact-scope/status/order/package, and final Git whitespace checks.
- Documentation-agent review: finalized independently in contexts
  01a0911b-25e4-7480-bf48-a353084e3f49 and
  01a0912d-4901-7bc3-8da3-73a452e3c1c6 after the separate correction context closed every
  identified executable-evidence gap.
- Documentation impact: affected Javadocs/package prose and the CPU guide now explain complete
  portable-first enumeration, exact matching/miss behavior, compatibility scope, ownership, and
  Runtime boundaries.
- Javadoc review: affected constructors and methods now document inputs, results, failures,
  nullability, immutability, ownership, and lifecycle where applicable.
- Glossary impact: updated Canonical workload signature and Candidate generator, and added
  Selected tuning decision to distinguish exact selection references from measurements and
  caches.
- Unresolved issues: None.
- Follow-up required: None for CPU 0010E.
- Next planning frontier: Prepare 0004 remains Draft; create its detailed task specification
  before considering it Ready. Tools/tuning 0001 and Config 0006A remain staged Draft work, while
  provider 0004 and CPU 0010D1 remain blocked/deferred.

Status: Complete
