# Task 0010E: [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [OpenBLAS](../../../../glossary.md#term-openblas) Tuning Candidates and Compatible Decisions

## Reader context — historical integration boundary

This milestone defines CPU-owned alternatives before any outer tuning runner measures them. CPU
analysis produces the complete portable candidate and eligible OpenBLAS thread/copy candidates; a
selected decision names one exact candidate for one compatible workload. It is neither an executable
nor timing evidence. Shared Prepare may transport that value without interpreting native-route
meaning, and Runtime receives only the finalized choice. An incompatible optional decision is a miss
at this internal layer, not authority to run stale work. The later supported collaboration has its
own stricter fresh selected-preparation boundary, described in the current reference.

Current reading: [Current OpenBLAS route](../../../../backend-guide/cpu-backend.md#current-narrow-openblas-matmul-route); [Local-workload tuning collaboration](../../../../backend-guide/cpu-backend.md#cold-local-workload-tuning-collaboration).

## Status

Complete

## Goal

Make the concrete [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) the authoritative [producer](../../../../glossary.md#term-producer) and consumer of one typed, versioned
candidate contract for the completed exact/default [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [OpenBLAS](../../../../glossary.md#term-openblas) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) route. For one
eligible workload, CPU must expose the complete safe portable candidate and every complete valid
OpenBLAS thread/representation candidate, together with [canonical](../../../../glossary.md#term-canonical-form) compatibility facts. CPU may
consume one explicitly supplied selected decision only when its workload, candidate [schema](../../../../glossary.md#term-schema), and
exact candidate [identity](../../../../glossary.md#term-identity-meanings) still match; otherwise ordinary preparation uses its current safe
[heuristic](../../../../glossary.md#term-heuristic) path.

This is the first executable stage of the repaired tuning sequence because all of its inputs are
already owned by completed CPU tasks. [Prepare](../../../../glossary.md#prepare) 0004 cannot define an [opaque](../../../../glossary.md#term-opaque) [handoff](../../../../glossary.md#term-handoff) until a
concrete producer exists, and tools/tuning 0001 cannot define persistent compatibility for values
whose owning backend has not stabilized their meaning.

The [mental model](../../../../glossary.md#term-mental-model) is:

```text
completed CPU lowering + qualification + route/resource facts
  -> one versioned canonical workload signature
  -> complete portable and OpenBLAS candidates
  -> optional exact selected decision
       matching: select that already-valid candidate
       absent or incompatible: retain current safe heuristic selection
  -> immutable CPU preparation plan
```

## [Scope](../../../../glossary.md#term-scope)

- Add one technically public, unsupported-internal [immutable](../../../../glossary.md#term-immutable) [CPU](../../../../glossary.md#term-cpu) [candidate batch](../../../../glossary.md#candidate-batch) in
  `internal.route.nativeblas.openblas`. It owns the candidate-schema version, exact workload
  signature, deterministic candidate order, and complete candidate values required by later
  opaque orchestration.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [signature glossary](../../../../glossary.md#term-signature); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- Add one technically public, unsupported-internal immutable selected-decision value containing
  the exact candidate-schema version, workload signature, and selected candidate identity. It
  contains no timing sample, objective, cache path, serialized bytes, executable, provider,
  segment, workspace instance, or Runtime state.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [timing sample glossary](../../../../glossary.md#benchmark-measurement-vocabulary); [objective glossary](../../../../glossary.md#term-objective); [cache glossary](../../../../glossary.md#term-cache); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Cover only the current positive rank-two, same-type FLOAT32 or FLOAT64, exact/default bare
  MATMUL boundary from CPU 0010–0010D. Preserve all eight 0010B representation masks and every
  eligible 0010C positive OpenBLAS thread candidate.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [masks glossary](../../../../glossary.md#term-boolean-mask); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Include the complete safe portable alternative in every emitted batch. A batch is not valid if
  it exposes only native choices.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Define canonical typed compatibility facts for operation kind and fixed attributes; input,
  accumulation, and output types; exact Shapes and resolved layouts; carrier and copy-mask facts;
  numerical and determinism mode; qualified target and provider/binary reuse scope; CPU hardware
  identity supplied as immutable cold input; CPU concurrency capacity; portable strategy and
  worker facts; OpenBLAS thread counts; expected-run/workload-cohort facts; and every CPU route,
  generated-artifact, cost-policy, qualification, and candidate-schema version that can change
  candidate meaning.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [operation glossary](../../../../glossary.md#operation); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier); [target glossary](../../../../glossary.md#term-target); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [concurrency glossary](../../../../glossary.md#term-concurrency); [run glossary](../../../../glossary.md#term-run); [artifact glossary](../../../../glossary.md#term-artifact); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Represent persistent compatibility only when CPU 0010D supplies a persistent binary identity.
  A session-only qualification may produce and consume a decision within that exact session but
  must expose no persistent compatibility projection.
- Derive candidate identities deterministically from typed CPU-owned fields. Identity and order
  must not depend on object identity, hash-map iteration, native address, live thread state,
  timing, or platform inference.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash); [inference glossary](../../../../glossary.md#term-inference))</span>
- Generate and validate candidates during cold CPU analysis from the same already-derived
  lowering, representation, resource, qualification, thread, and cost facts used by the current
  selector. Candidate generation must not create a second graph interpreter or disagree with
  actual route eligibility.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [graph glossary](../../../../glossary.md#graph))</span>
- Accept an optional decision through CPU analysis inputs. Validate it against the freshly
  generated batch before selecting its candidate. A missing or incompatible decision is an
  explicit miss and uses the unchanged safe heuristic; malformed CPU values fail construction.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Retain the selected typed compatibility facts in the CPU preparation plan far enough for tests
  and later Prepare transport to prove the exact choice. Finalization and Runtime receive no
  search, cache, measurement, or reselection responsibility.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- Add deterministic native-free tests for field completeness, immutability, ordering, equality,
  session/persistent scope, candidate generation, selected-decision hits and misses, safe
  heuristic fallback, resource agreement, and unchanged prepared execution semantics.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- Finalize affected Javadocs, OpenBLAS package documentation, CPU backend guide, glossary impact,
  this task, CPU master plan, and roadmap in the mandatory separate documentation-focused pass.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- BFLOAT16 OpenBLAS candidates or CPU 0010D1. Provider 0004 and CPU 0010D1 remain a
  blocked/deferred optional side branch. This task must not add a placeholder BFLOAT16 capability
  bit that could later be mistaken for proof.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- `cblas_sbgemm`, conversion staging, BFLOAT16 input widening, a temporary FLOAT32 output matrix,
  result conversion, platform-based BFLOAT16 inference, or any relaxation of Model MATMUL
  semantics.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [FLOAT32 glossary](../../../../glossary.md#term-float32); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [inference glossary](../../../../glossary.md#term-inference); [Model glossary](../../../../glossary.md#term-model); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Prepare 0004, tools/tuning 0001, Config 0004–0006A, Engine, or any shared/public handoff. This
  task changes only the CPU module plus its focused explanatory and planning documentation.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [handoff glossary](../../../../glossary.md#term-handoff); [module glossary](../../../../glossary.md#term-module))</span>
- Measurement, benchmarking, objective/budget comparison, candidate timing, workload
  deduplication across model occurrences, cache loading, serialization, corruption recovery,
  atomic persistence, or rich evidence. Those belong to tools/tuning.<span data-glossary-navigation="true"> ([benchmarking glossary](../../../../glossary.md#term-benchmarking); [objective glossary](../../../../glossary.md#term-objective); [cache glossary](../../../../glossary.md#term-cache); [serialization glossary](../../../../glossary.md#term-serialization); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- A public user request, public CPU builder, Config facade, automatic composition, or Engine
  lifecycle.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Relaxed or fast-math candidates. The complete task is exact/default only and has no dependency
  on Config 0006.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Planning](../../../../glossary.md#term-planning) [ownership](../../../../glossary.md#term-ownership-meanings) or cost changes. Planning continues to select only `BackendId("cpu")` and
  never sees route, representation, thread, or candidate vocabulary.
- New OpenBLAS symbols, provider API/source/test changes, discovery, qualification probes, binary
  inspection, thread coordination, provider invocation, MATMUL representation expansion, or new
  execution behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [qualification glossary](../../../../glossary.md#term-qualification); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Batch, broadcast, rank-one, empty, zero-contraction, epilogue, packing, persistent weights,
  mixed type, integral type, sparse, quantized, or complex MATMUL.<span data-glossary-navigation="true"> ([Batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [contraction glossary](../../../../glossary.md#contraction-dimension); [epilogue glossary](../../../../glossary.md#term-epilogue); [weights glossary](../../../../glossary.md#term-weight))</span>
- Generated bytecode, emitter, Class-File schema, generated-artifact persistence, performance
  evidence, module dependencies, Gradle, architecture, ADR, backend-conformance, or integration
  changes.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [schema glossary](../../../../glossary.md#term-schema); [artifact glossary](../../../../glossary.md#term-artifact); [Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially concrete backend candidate
  ownership, opaque tuning orchestration, staged preparation, CPU routes, and Runtime prohibitions<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [opaque glossary](../../../../glossary.md#term-opaque); [CPU glossary](../../../../glossary.md#term-cpu); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU master plan](../master-plan.md)
- [CPU 0010 narrow OpenBLAS route](0010-narrow-openblas-blas-compatible-native-route.md)<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- [CPU 0010B bounded representations](0010b-bounded-openblas-matmul-representation-expansion.md)
- [CPU 0010C coordinated thread candidates](0010c-coordinated-openblas-thread-candidates-and-shared-cpu-thread-budget.md)
- [CPU 0010D qualification and target fingerprinting](0010d-installed-openblas-qualification-and-target-fingerprinting.md)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification); [target glossary](../../../../glossary.md#term-target))</span>
- [Blocked provider 0004 direct BFLOAT16 proof](../../openblas-provider/tasks/0004-optional-direct-bfloat16-output-gemm-capability.md)<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- [Prepare master plan](../../../modules/prepare/master-plan.md)
- [Tuning master plan](../../../tools/tuning/master-plan.md)
- [Config master plan](../../../modules/config/master-plan.md)

## Architecture constraints

- CPU owns the typed candidate vocabulary, canonical backend/workload compatibility facts,
  candidate validity, deterministic ordering, and validation/consumption of a selected decision.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [canonical glossary](../../../../glossary.md#term-canonical-form); [backend glossary](../../../../glossary.md#backend); [validation glossary](../../../../glossary.md#term-validation))</span>
- Shared Prepare may later transport the complete batch and decision opaquely, but it must not
  interpret CPU fields or select a route.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Tools/tuning owns measurement, selection against objective and budget, persistent artifact
  encoding/loading/mutation, corruption and incompatibility rejection, and rich evidence. CPU
  must not implement those responsibilities here.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Backend analysis remains deterministic from explicit immutable facts. It performs no native
  query, host discovery, measurement, file/cache access, or mutation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [host glossary](../../../../glossary.md#term-host); [cache glossary](../../../../glossary.md#term-cache))</span>
- A selected decision can choose only a candidate in the freshly generated complete batch. It
  cannot create eligibility, bypass Model semantics, weaken determinism, change declarations, or
  authorize a late fallback.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Safe heuristic and cache-free preparation remains correct. Runtime executes only the prepared
  result and never sees candidate, decision, cache, or measurement vocabulary.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Portable BFLOAT16 support remains unchanged. No native BFLOAT16 candidate exists until the
  separate provider and CPU proof branch completes under its exact semantic gates.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.prepare`
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas`

Packages added or changed:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas` — adds the narrow
  typed candidate-batch and selected-decision contracts beside their owning route.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — accepts the optional cold decision
  and retains the exact selected compatibility facts; it does not gain tuning policy.

Type placement:

- `CpuOpenBlasTuningBatch` — technically public unsupported-internal immutable batch, with nested
  typed workload, hardware, reuse-scope, candidate-identity, and complete-candidate values.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [scope glossary](../../../../glossary.md#term-scope); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `CpuOpenBlasTuningDecision` — technically public unsupported-internal immutable exact selection
  reference validated only by CPU.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- `CpuPartitionAnalysisInputs` — owns explicit cold hardware/workload-cohort facts and the optional
  decision because these are inputs to deterministic CPU analysis.
- `CpuOpenBlasRouteSelector` — remains the sole owner that derives eligibility/cost candidates and
  now exposes the exact batch before applying either a matching decision or its safe heuristic.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- `CpuPartitionPreparationPlan` — retains only the selected CPU facts required to prove the
  prepared choice; it does not become a cache or evidence object.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>

Tests mirror the production packages. No new package is added.

## Affected files

Expected production and [Javadoc](../../../../glossary.md#term-javadoc) paths:

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

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md` only if the final reusable candidate/decision distinction needs an entry
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the 17 exact paths above: eight production/[Javadoc](../../../../glossary.md#term-javadoc) paths,
four [CPU](../../../../glossary.md#term-cpu) test paths, two explanatory-documentation paths, and three [planning](../../../../glossary.md#term-planning) paths. It adds
exactly two production types and one test type, with no new package.

The final combined diff has 21 paths rather than changing that implementation ceiling:
CpuPartitionPreparerTest needed no edit, while the earlier clean planning pass intentionally
updated the five existing provider, [Prepare](../../../../glossary.md#prepare), tuning, and Config planning records needed to repair
the acyclic sequence. Those planning-only reconciliations are part of the approved [handoff](../../../../glossary.md#term-handoff); they
add no task specification, source, test, [dependency](../../../../glossary.md#term-dependency), or behavior.

If truthful implementation needs Prepare, tuning, Config, [Engine](../../../../glossary.md#term-engine), provider, shared-[module](../../../../glossary.md#term-module), build,
architecture, [conformance](../../../../glossary.md#term-conformance), integration, generated-code, or another source/test path, stop and
return the task to planning.

## Acceptance criteria

1. One immutable versioned batch contains one exact canonical workload signature, the safe
   portable candidate, and every eligible OpenBLAS thread/representation candidate in stable
   deterministic order.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [canonical workload signature glossary](../../../../glossary.md#canonical-workload-signature))</span>
2. Candidate generation reuses the current authoritative CPU lowering, representation, resource,
   qualification, thread, and checked-cost facts; no second graph or eligibility interpretation
   exists.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [qualification glossary](../../../../glossary.md#term-qualification); [graph glossary](../../../../glossary.md#graph))</span>
3. The workload signature contains every typed compatibility field listed in Scope. Automated
   tests change each field independently and prove inequality or incompatibility.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [Scope glossary](../../../../glossary.md#term-scope))</span>
4. Persistent compatibility is present only for an exact path-qualified persistent binary
   identity. Session-only decisions are bound to the exact session and cannot be projected or
   reused persistently.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
5. Candidate identities distinguish portable strategy/resource facts and every OpenBLAS thread,
   representation, resource, and cost-policy meaning without native addresses or live objects.
6. A decision selects only the exact matching candidate in a freshly generated matching batch.
   Schema, workload, scope, session, or candidate mismatch is an explicit miss that retains the
   existing safe heuristic; malformed CPU-owned values fail construction.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
7. The no-decision path selects byte-for-byte equivalent route/representation/thread facts to the
   current heuristic for the same inputs. Ineligible OpenBLAS workloads remain portable and emit
   no false native candidate.
8. Analysis and preparation perform no measurement, objective comparison, cache access,
   serialization, corruption handling, persistence, native query, or host/platform discovery.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [cache glossary](../../../../glossary.md#term-cache); [serialization glossary](../../../../glossary.md#term-serialization); [persistence glossary](../../../../glossary.md#term-persistence); [host glossary](../../../../glossary.md#term-host))</span>
9. Finalization and execution behavior, exact resource declarations, all eight representation
   masks, provider coordination, qualification, and no-late-fallback rules remain unchanged.<span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask))</span>
10. No [BFLOAT16](../../../../glossary.md#term-bfloat16) [OpenBLAS](../../../../glossary.md#term-openblas) candidate, placeholder proof flag, `cblas_sbgemm`, conversion/staging
    path, relaxed semantic, or platform-derived native eligibility is added. Portable BFLOAT16 is
    unchanged.
11. Public supported CPU API, provider API, shared Prepare/Runtime, Planning, Config, Engine,
    dependencies, Gradle, generated schema/code, backend conformance, integration, and
    architecture remain unchanged.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Planning glossary](../../../../glossary.md#term-planning); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
12. A separate clean documentation-focused context independently reviews the implementation and
    tests, finalizes all affected Javadocs/package prose, CPU guide, glossary impact, task/master/
    roadmap evidence, and records reasoned no-change conclusions for excluded areas.
13. Focused/final CPU tests, CPU Javadoc and rendered-page inspection, Markdown links/anchors,
    unique headings, balanced fences, final newlines, trailing whitespace, exact path/public-
    internal surface/status/order checks, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` pass before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests while implementing. After executable Java stabilizes, run one final [module](../../../../glossary.md#term-module)
command:

```bash
./gradlew :backends:cpu:test
```

Ordinary tests are native-free and deterministic. No real [OpenBLAS](../../../../glossary.md#term-openblas) [checkpoint](../../../../glossary.md#term-checkpoint) is required because
the task changes candidate description and cold decision consumption, not [qualification](../../../../glossary.md#term-qualification),
provider invocation, representation execution, or numerical behavior.

The documentation-focused context reuses the successful [CPU](../../../../glossary.md#term-cpu) result unless it changes executable
Java or tests, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git status --short -uall
```

It renders and inspects changed Javadocs; validates local Markdown files and anchors, unique
headings, balanced fences, LF/final newlines, terminology, exact [scope](../../../../glossary.md#term-scope), package placement, and
public/internal inventory; and confirms CPU 0010E is `Complete` only after all gates, CPU 0010D1
remains Blocked/deferred, [Prepare](../../../../glossary.md#prepare) 0004 is the next Draft stage, tools/tuning 0001 follows Prepare
0004, and neither later task has a detailed specification.

Repository-wide validation is deferred to the Prepare 0004 shared-contract checkpoint or [CI](../../../../glossary.md#term-continuous-integration-ci).
This task changes one [concrete backend](../../../../glossary.md#concrete-backend) module and no [dependency](../../../../glossary.md#term-dependency), build, shared contract, or
architecture rule.

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0010, 0010A, 0010B, 0010C, and
  [0010D](0010d-installed-openblas-qualification-and-target-fingerprinting.md).
- Complete portable MATMUL, typed portable route, representation, generated-artifact identity,
  qualification, thread-candidate, cost, staged analysis/finalization, and package-inventory
  contracts already present in the CPU module.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [qualification glossary](../../../../glossary.md#term-qualification); [module glossary](../../../../glossary.md#term-module))</span>
- Complete [OpenBLAS](../../../../glossary.md#term-openblas) provider tasks 0001–0003 for the existing [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [leaf](../../../../glossary.md#term-leaf).

All dependencies are complete. Provider 0004, CPU 0010D1, [Prepare](../../../../glossary.md#prepare) 0004, tools/tuning 0001, Config
0004–0006A, and [Engine](../../../../glossary.md#term-engine) are not dependencies of this CPU-owned [producer](../../../../glossary.md#term-producer)/consumer capability.

## Follow-up tasks

- Prepare 0004 is the next staged row. It transports CPU-owned batches and selected decisions
  opaquely and performs no measurement, compatibility interpretation, or persistence.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Tools/tuning 0001 follows Prepare 0004. It owns exact/default workload extraction and
  deduplication, measurement, objective/budget selection, persistent cache encoding/loading/
  mutation, corruption/incompatibility rejection, and separate rich evidence.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [cache glossary](../../../../glossary.md#term-cache))</span>
- Config 0006A may later expose immutable public request inputs after the tuning consumer is
  stable. Config 0004–0006 are not forced through this first exact/default slice.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [slice glossary](../../../../glossary.md#slice))</span>
- CPU 0010D1 remains a blocked/deferred optional side branch. If it later completes, a versioned
  CPU follow-up must extend candidate compatibility with direct-BFLOAT16 proof facts explicitly.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- [CPU](../../../../glossary.md#term-cpu) 0016 later generalizes the proven contract across whichever peer routes are implemented.

## Architecture impact

Expected impact: None.

This task implements the [concrete backend](../../../../glossary.md#concrete-backend)'s already-authorized candidate and compatibility
[ownership](../../../../glossary.md#term-ownership-meanings). If it requires shared [Prepare](../../../../glossary.md#prepare) interpretation, tuning [persistence](../../../../glossary.md#term-persistence) inside [CPU](../../../../glossary.md#term-cpu), [Runtime](../../../../glossary.md#term-runtime)
selection, a provider change, public configuration, a [module](../../../../glossary.md#term-module) edge, or an architecture rule,
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
  interpret CPU candidate meaning.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [concrete backend glossary](../../../../glossary.md#concrete-backend); [producer glossary](../../../../glossary.md#term-producer); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Keep the initial schema exact/default and FLOAT32/FLOAT64 only. This avoids a false dependency
  on relaxed Config and does not weaken or bypass the blocked BFLOAT16 semantic proof.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [dependency glossary](../../../../glossary.md#term-dependency); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Treat an incompatible external decision as a cache-style miss, not an analysis failure or
  permission to use a stale candidate. Malformed CPU-owned construction still fails immediately.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Include the portable candidate in every batch so optional tuning cannot become necessary for
  correctness.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Keep rich measurements and persistent encoding out of CPU. CPU supplies typed semantic values;
  tools/tuning later owns how evidence and compact artifacts are stored.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact))</span>

## Known limitations

- Only the current positive [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) exact/default [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) bare [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) route participates.
- The task creates an unsupported-internal [CPU](../../../../glossary.md#term-cpu) contract, not a public tuning or [Engine](../../../../glossary.md#term-engine) [API](../../../../glossary.md#term-application-programming-interface-api).
- No model-wide occurrence deduplication, measurement, persistence, or end-to-end plan selection
  exists until Prepare 0004 and tools/tuning 0001 complete.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [occurrence glossary](../../../../glossary.md#term-occurrence); [persistence glossary](../../../../glossary.md#term-persistence); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Session-only qualification can support same-session decisions but cannot create persistent
  compatibility.<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>
- Native [BFLOAT16](../../../../glossary.md#term-bfloat16) MATMUL remains unavailable; portable BFLOAT16 remains supported.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context 01a09107-730a-7993-ac6d-7e970a92c35d reported focused
  candidate/selector/inventory tests passing, followed by the final CPU module test passing in 2
  minutes 33 seconds: 194 suites, 988 tests, 28 skips, zero failures, and zero errors. It also
  reported the Git whitespace check passing.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [module glossary](../../../../glossary.md#term-module))</span>
- Documentation context 01a0911b-25e4-7480-bf48-a353084e3f49 independently inspected every
  changed production and test source. The implementation is portable-first, enumerates 8
  realizable masks by every fitting configured thread count, restricts eligibility to positive
  rank-two same-type exact/default FLOAT32/FLOAT64 MATMUL, uses immutable graph-ID-free typed
  workload/candidate values, distinguishes session-only and persistent-binary qualification,
  treats absent or incompatible decisions as heuristic misses, and retains selected-plan resource
  agreement. No measurement, cache/persistence, host discovery, provider query/change, shared
  Prepare/Runtime interpretation, generated-code change, or native BFLOAT16 path was found.<span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [qualification glossary](../../../../glossary.md#term-qualification); [heuristic glossary](../../../../glossary.md#term-heuristic); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence); [host glossary](../../../../glossary.md#term-host); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- The same independent review found acceptance criterion 3 incomplete:
  CpuOpenBlasTuningBatchTest changes hardware compatibility and separately exercises schema and
  candidate misses, but no automated matrix changes every WorkloadSignature component
  independently and proves inequality or decision incompatibility. Repository search found no
  other construction or mutation of WorkloadSignature in tests.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Documentation context 01a0911b-25e4-7480-bf48-a353084e3f49 finalized affected Javadocs,
  OpenBLAS package prose, the CPU backend guide, and glossary terminology without changing Java
  behavior or tests.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [backend glossary](../../../../glossary.md#backend))</span>
- The CPU Javadoc task passed in 3 seconds. It emitted 94 pre-existing warnings in unrelated or
  legacy constructor documentation; no warning names the new tuning batch, decision, selector, or
  the added canonical-constructor tags. Rendered pages for CpuOpenBlasTuningBatch and its nested
  values, CpuOpenBlasTuningDecision, CpuOpenBlasRouteSelector, CpuPartitionAnalysisInputs,
  CpuPartitionPreparationPlan, and the OpenBLAS package summary were inspected successfully.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Correction implementation context 01a09122-987e-7023-874b-10d0969d7c16 added the complete
  field-by-field compatibility matrix requested by the first documentation review. It reported a
  focused three-class run passing 3 suites and 20 tests with zero skips, failures, or errors,
  followed by one final <span data-glossary-navigation="true"> ([compatibility matrix glossary](../../../../glossary.md#review-tables-and-support-matrices))</span><span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`./gradlew :backends:cpu:test` run passing 194 suites and 991 tests with
  28 skips and zero failures or errors. `git diff --check` also passed.
- Final documentation review context 01a0912d-4901-7bc3-8da3-73a452e3c1c6 first identified two
  remaining fail-closed evidence omissions: no exact <span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>`BoundaryStorageFact` component inventory and
  no explicit rejection of a non-three-entry boundary list. The correction added only those two
  assertions and reran the same focused and final CPU commands with the same final counts.
- Resumed final review in context 01a0912d-4901-7bc3-8da3-73a452e3c1c6 inspected both assertions
  and the complete surrounding matrix. Every mutable <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>`WorkloadSignature` component and meaningful
  nested compatibility fact now has independent inequality plus decision-miss/safe-fallback
  coverage, while closed-schema or relationally coupled facts have explicit rejection evidence.
  Exact record-component inventories make additions fail closed. No executable Java changed after
  the final CPU run, and the prior CPU Javadoc/rendered-page evidence remains current because both
  corrections changed tests only.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- The repository Markdown validator passed all 10 changed Markdown files, including local targets
  and anchors, unique headings, balanced fences, LF endings, final newlines, and trailing
  whitespace. The final Git diff whitespace check passed.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- Exact inventory and order checks found 21 combined paths for the documented reason in Maximum
  scope; both new technically public types remain under the unsupported internal OpenBLAS package;
  provider 0004 and CPU 0010D1 remain blocked/deferred; Prepare 0004 is the next Draft planning
  frontier with no detailed task file; tools/tuning 0001 and Config 0006A remain staged Draft work
  with no detailed task files; and Runtime has no candidate-selection change.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [planning glossary](../../../../glossary.md#term-planning))</span>

## Implementation notes

- Added CpuOpenBlasTuningBatch and CpuOpenBlasTuningDecision beside the OpenBLAS route, and
  extended CPU analysis inputs, route selection, and the retained preparation plan to produce and
  consume the typed cold decision.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Candidate enumeration reuses the existing portable plan, common access bindings, representation
  planner, qualification, thread candidates, resource declarations, and checked cost facts. The
  batch begins with portable and orders native candidates by representation enum order and
  configured-thread order.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [qualification glossary](../../../../glossary.md#term-qualification); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- The safe heuristic is applied only after the complete batch is built. A matching decision may
  select any member, including portable or a native candidate not favored by the heuristic;
  incompatible values cannot create eligibility.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Documentation no-change conclusions: ARCHITECTURE.md and focused architecture explanations
  remain accurate because the implementation realizes their existing concrete-backend ownership
  and Runtime prohibition; provider source/API and shared Prepare/Runtime, Config, Engine,
  Planning, generated code/schema, dependency/build, backend-conformance, and integration paths
  require no update because none of their contracts or behavior changed. The intentional planning
  edits keep provider 0004 and CPU 0010D1 blocked/deferred, retain Prepare 0004 then tools/tuning
  0001 as Draft, and place Config 0006A after the stable tuning consumer.<span data-glossary-navigation="true"> ([backend ownership glossary](../../../../glossary.md#backend-ownership); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [Planning glossary](../../../../glossary.md#term-planning); [schema glossary](../../../../glossary.md#term-schema); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Completion summary

- Completed changes: implemented the CPU-owned typed candidate batch and compatible-decision
  consumption; finalized its Javadocs, OpenBLAS package explanation, CPU backend guide, glossary
  terms, and the already-staged acyclic planning sequence.<span data-glossary-navigation="true"> ([candidate batch glossary](../../../../glossary.md#candidate-batch); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: eight permitted production/Javadoc paths, three changed/created
  focused test paths (the planned CpuPartitionPreparerTest needed no edit), the CPU backend guide,
  glossary, this task, CPU master plan, roadmap, and the intentional provider/Prepare/tuning/Config
  planning reconciliation paths already present in the handoff.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Prepare glossary](../../../../glossary.md#prepare); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Tests and validation: reused the final successful CPU module run from implementation context
  01a09122-987e-7023-874b-10d0969d7c16: 194 suites, 991 tests, 28 skips, zero failures/errors;
  its focused three-class run passed 3 suites and 20 tests with zero skips/failures/errors. CPU
  Javadoc passed and affected rendered pages were inspected in documentation context
  01a0911b-25e4-7480-bf48-a353084e3f49; resumed final context
  01a0912d-4901-7bc3-8da3-73a452e3c1c6 reused that current evidence and passed changed-Markdown,
  exact-scope/status/order/package, and final Git whitespace checks.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [run glossary](../../../../glossary.md#term-run); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: finalized independently in contexts
  01a0911b-25e4-7480-bf48-a353084e3f49 and
  01a0912d-4901-7bc3-8da3-73a452e3c1c6 after the separate correction context closed every
  identified executable-evidence gap.
- Documentation impact: affected Javadocs/package prose and the CPU guide now explain complete
  portable-first enumeration, exact matching/miss behavior, compatibility scope, ownership, and
  Runtime boundaries.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: affected constructors and methods now document inputs, results, failures,
  nullability, immutability, ownership, and lifecycle where applicable.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Glossary impact: updated Canonical workload signature and Candidate generator, and added
  Selected tuning decision to distinguish exact selection references from measurements and
  caches.<span data-glossary-navigation="true"> ([Canonical workload signature glossary](../../../../glossary.md#canonical-workload-signature); [Candidate generator glossary](../../../../glossary.md#candidate-generator); [Selected tuning decision glossary](../../../../glossary.md#selected-tuning-decision))</span>
- Unresolved issues: None.
- Follow-up required: None for [CPU](../../../../glossary.md#term-cpu) 0010E.
- Next planning frontier: Prepare 0004 remains Draft; create its detailed task specification
  before considering it Ready. Tools/tuning 0001 and Config 0006A remain staged Draft work, while
  provider 0004 and CPU 0010D1 remain blocked/deferred.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

Status: Complete
