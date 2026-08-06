# Task 0005C: Vector and Parallel Portable Strategies

## Status

Complete

## Goal

Add the three remaining portable execution strategies—Vector API, parallel-scalar, and
parallel-vector—to the exact fully static FLOAT64 `ADD -> exact GELU -> MUL` whole-partition
topology completed by CPU 0005B. All four strategies consume the same route-independent
`CpuKernelIr`, the same five access state machines, the same four-boundary ordered carrier-pattern
specialization, and the same cold `start`/`end` binding contract.

CPU analysis makes every eligibility and selection decision before shared slot assignment.
Finalization realizes the already-selected scalar or vector generated class and, for a parallel
plan, verifies one explicitly supplied CPU-private worker group. Runtime cold binding creates
direct calls for deterministic disjoint logical-output chunks. Generated hot loops perform no
route, compute-strategy, orchestration, carrier, access-regime, or operation dispatch and allocate
nothing per element or vector lane.

The mental model is:

```text
0005B canonical IR + access bindings + carrier pattern
  -> cold vector eligibility + bounded parallelism + strategy selection
  -> one scalar or exact-species vector generated artifact
  -> single call, or CPU-private deterministic chunk orchestration
  -> joined success or one joined failure
```

CPU 0005C is complete only for this exact proving slice. CPU 0005D and later rows remain Draft
without detailed specifications.

## Scope

- Preserve the exact three-node, one-unit, fully static FLOAT64 `ADD -> exact GELU -> MUL`
  topology, exact/default numerical mode, fixed instruction order, two virtual intermediates,
  four boundary declarations, producer/provenance rules, alias and span decisions, and one
  partition-level `PreparedExecutable` established by 0005A and 0005B.
- Preserve `CpuAccessPlan` as the sole normalized access family and preserve its five state
  machines without adding a vector-specific broadcast/layout planner:
  `DENSE_LINEAR`, `SCALAR_ALL_ZERO`, `LAST_AXIS_BIAS`, `BLOCK_OUTER`, and
  `GENERAL_ODOMETER`.
- Add one typed CPU-private portable execution configuration to
  `CpuPartitionAnalysisInputs`. It records:
  - compute preference: `SCALAR` or `VECTOR_IF_ELIGIBLE`;
  - positive configured maximum parallelism;
  - positive available-parallelism snapshot supplied by CPU composition/platform discovery; and
  - positive minimum elements per worker.
- Preserve `CpuPartitionAnalysisInputs.DEFAULT` as manifest-disabled, four ordered
  `MEMORY_SEGMENT` boundaries, scalar preference, configured/available parallelism of one, and a
  positive minimum range. Existing default preparation therefore remains scalar/single-thread.
- Bound usable parallelism during analysis to
  `min(configuredMaximumParallelism, availableParallelism)`. For a non-empty full range, bound
  selected range count further to
  `min(usableParallelism, ceil(elementCount / minimumElementsPerWorker))`. Parallel
  orchestration is eligible only when this count is at least two.
- Use the Java 26 preferred FLOAT64 Vector API species as the sole 0005C species candidate. Capture
  its exact vector bit size and lane count on the cold analysis path, require more than one lane,
  and include its bit size in specialization, compatibility metadata, generated binary identity,
  artifact verification, and lowering manifest whenever vector compute is selected. Scalar
  specializations contain no vector species.
- Select the strategy deterministically during CPU analysis:
  1. zero elements select scalar/single-thread;
  2. determine vector eligibility under the exact table below;
  3. determine parallel eligibility from the bounded range count above;
  4. choose parallel-vector when vector and parallel are eligible;
  5. otherwise choose vector when vector alone is eligible;
  6. otherwise choose parallel-scalar when parallel alone is eligible; and
  7. otherwise choose scalar.
- Treat `VECTOR_IF_ELIGIBLE` as a preference with scalar fallback, not a required route. An
  admitted `GENERAL_ODOMETER` binding or too-small contiguous run therefore selects scalar or
  parallel-scalar rather than failing an otherwise supported partition.
- Vectorize exactly the regimes and carrier forms described in the eligibility table below. Do
  not implement Vector API gather, scatter, index maps, lane maps, or a general-stride vector
  approximation.
- Add a package-private `CpuVectorEmitter` beside the existing scalar/carrier/loop emitters. It
  emits the current exact GELU error-function polynomial with FLOAT64 Vector API arithmetic,
  `EXP`, comparisons, masks, and blends. It must preserve the current branch thresholds,
  coefficient order, exact GELU formula, NaN/infinity/signed-zero classifications, and numerical
  tolerance. It must not substitute `GELU_TANH_APPROXIMATION`, call scalar GELU once per lane,
  reassociate, contract, or enable relaxed math.
- Generate unmasked full-vector bodies only. Within each contiguous run, begin at the exact
  arbitrary cold-bound start address, execute as many complete preferred-species vectors as fit
  before either the logical `end` or the earliest participating access-plan contiguous-run
  boundary, then execute the remaining elements through the existing scalar body. This scalar
  prologue/tail rule also handles a non-zero start, a worker chunk ending mid-block, and runs
  shorter than one vector. No masked tail is added in 0005C.
- Use direct `DoubleVector.fromArray` / `intoArray` and
  `DoubleVector.fromMemorySegment` / `intoMemorySegment` forms selected at generation time for the
  existing ordered heap/segment pattern. Every segment vector load/store passes
  `ByteOrder.nativeOrder()`, preserving the current scalar carrier contract. Mixed carriers
  remain supported without a generated carrier switch.
- Keep vector and scalar code shaping cold. A generated class contains the one selected body and
  one exact direct entry signature; it does not contain a runtime branch between scalar and vector
  strategies. Parallel-scalar reuses the scalar generated body; parallel-vector reuses the vector
  generated body. Parallel orchestration remains outside generated code.
- Reintroduce one minimal CPU-private `CpuWorkerGroup` in `internal.executable`. It owns a fixed
  positive number of named daemon platform workers for its explicit caller-managed lifetime. It
  is not a public executor facade, general task system, common pool, virtual-thread scheduler,
  registry, or Runtime service.
- Make worker ownership explicit: composition/test code constructs and closes the group;
  `CpuPartitionFinalizer`, `CpuPreparedExecutable`, and bound invocations borrow it and never close
  it. The default finalizer remains valid for single-thread plans. Finalizing a parallel plan
  requires an open supplied group whose worker count is at least the plan's selected range count;
  otherwise finalization fails before constructing the executable.
- Keep `CpuPreparedExecutable` immutable and concurrently bindable. For a parallel plan, cold
  binding computes the exact ranged bindings, geometry arrays, direct handle calls, and
  quotient/remainder chunk boundaries once per bound invocation. It proves every selected segment
  accessible to every worker and retains no generic carrier array or Runtime lookup in worker
  execution.
- Partition any requested non-empty `[start, end)` into
  `min(selectedRangeCount, ceil((end - start) / minimumElementsPerWorker))` contiguous non-empty
  chunks. Quotient/remainder division occurs only during cold binding. Chunks appear in ascending
  range-index order, cover the requested range exactly once, and never overlap logically.
  Existing output injectivity then proves their physical output writes disjoint. Input ranges may
  overlap because they are read-only.
- Execute a zero range without worker submission or carrier access. Execute a non-empty range that
  reduces to one chunk directly on the invoking thread. Submit two or more chunks synchronously to
  the borrowed worker group and return only after every started chunk has quiesced.
- Reject nested submission from one of the same group's owned workers with
  `IllegalStateException("CPU worker must not submit parallel work")` before a nested chunk starts.
  This rejection applies only when an invocation would submit two or more chunks. A zero-range
  invocation remains a no-op and a one-chunk invocation remains inline even when invoked by an
  owned worker. Distinct concurrent external submissions remain isolated and share only bounded
  worker capacity.
- Define joined failure and cancellation exactly:
  - the first detected unchecked worker failure requests cancellation so no unclaimed chunk
    starts;
  - already-started chunks run to completion and all workers are joined;
  - after joining, the failure at the lowest completed failing range index is rethrown unchanged;
  - later distinct failures are suppressed in ascending range-index order, skipping the exact
    primary object to avoid self-suppression;
  - no write rollback is claimed, and Runtime's existing executable failure transition leaves all
    declared output copies invalid.
- If the invoking thread is interrupted while joining, request cancellation, join already-started
  work, restore interrupt status, and throw the CPU-private
  `CpuParallelExecutionException("CPU parallel execution interrupted", cause)`. Suppress distinct
  completed worker failures in ascending range-index order.
- Make `CpuWorkerGroup.close()` thread-safe and idempotent. It rejects new work, cancels unclaimed
  chunks, joins started chunks, wakes and joins every owned worker, and returns only when none
  survives. Closing from an owned worker fails with
  `IllegalStateException("CPU worker must not close its worker group")`. A close racing a
  submission surfaces the earlier worker failure when one exists; otherwise the submission fails
  with `CpuParallelExecutionException("CPU parallel execution cancelled by worker-group close")`.
- Preserve generated-class process-local reuse and optional persistence. Strategy and exact vector
  species are compatibility facts; configured/available parallelism, selected range count,
  minimum range size, worker-group identity, chunk boundaries, extents, carriers, slots, and run
  identity remain prepared/cold instance facts unless they alter emitted code.
- Update detailed Javadoc and package documentation for every changed or new CPU contract. A
  separate clean documentation-focused pass after Java stabilization must finalize the CPU guide,
  glossary, and planning/status records in the same overall change.

## Vector eligibility and fallback table

The table is evaluated for all four ordered materialized boundaries after 0005B normalization.
`contiguous run` means the checked product of the plan's trailing contiguous suffix for the cold
binding.

| Access regime | 0005C vector behavior | Eligibility condition |
|---|---|---|
| `DENSE_LINEAR` | Direct contiguous vector load/store. | Eligible when the requested unit contains at least one full species. |
| `SCALAR_ALL_ZERO` | Load one scalar and broadcast it to all lanes. Writes never use this regime for a repeated address. | Eligible for read inputs. |
| `LAST_AXIS_BIAS` | Direct vectors within the final-axis run; scalar tail before reset. | Eligible when the final-axis extent is at least one species. |
| `BLOCK_OUTER` | Direct vectors within each trailing contiguous block; scalar tail before outer carry. | Eligible when the contiguous-run length is at least one species. |
| `GENERAL_ODOMETER` | No gather and no vector body. | Vector-ineligible; select scalar or parallel-scalar fallback. |

Whole-unit vector eligibility requires every non-scalar boundary to satisfy its row, the output to
remain an admitted injective direct store, and the full element count to contain at least one
species. Heap, `MemorySegment`, and every mixed four-boundary carrier pattern have the same regime
eligibility. Carrier provenance does not change the decision.

## Out of scope

- Any operation, topology, or data type beyond the exact completed FLOAT64
  `ADD -> exact GELU -> MUL` proving chain.
- Vector gather/scatter, index-map allocation, general-stride Vector API access, or a promise that
  every 0005B regime is vectorized.
- CPU-internal materialization, copy units, direct-versus-contiguous cost comparison, fixed-shape
  specialization, unrolling, persistence benchmarks, tuning, or autotuning. CPU 0005D owns those
  decisions.
- A new public/general CPU configuration API. The 0005C configuration is typed, CPU-private
  analysis input only; shared Config remains unchanged.
- Another planner, candidate registry, route registry, executor facade, generic task abstraction,
  common/general thread pool, `ForkJoinPool.commonPool()`, `parallelStream`, virtual threads, or
  Runtime/Engine service lookup.
- Worker ownership by `PreparedExecution`, Runtime, shared Prepare, Engine, a global singleton, or
  a generated class. No shared lifecycle contract changes in this task.
- Dynamic or symbolic Shape binding, unresolved layouts, materialization, native or vendor routes,
  OpenBLAS integration, broader operation families, reductions/scans, relaxed math, tuning, or
  benchmark-driven selection.
- Changes to Model, Compiler, Planning, shared Prepare, Runtime, Config, Backend Contract, Trace,
  Engine, architecture contracts, dependencies, Gradle configuration, architecture tests,
  backend-conformance tests, or integration tests.
- Changes to completed CPU 0005A or 0005B task history, or creation of a detailed CPU 0005D or
  later task specification.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Runtime,
  Prepare, concrete backend ownership, CPU routes, and run lifecycle.
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Planning Guide](../../../planning-guide.md).
- [CPU backend master plan](../master-plan.md).
- [Completed CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md).
- [Completed CPU 0005B](0005b-universal-access-plans-and-right-aligned-broadcasting.md).
- [CPU backend guide](../../../../backend-guide/cpu-backend.md#atomic-partition-kernel-reset).
- [Glossary: CPU portable execution strategy](../../../../glossary.md#cpu-portable-execution-strategy).

## Architecture constraints

- Planning selects only CPU ownership. CPU analysis owns Vector API eligibility, bounded
  parallelism, strategy selection, generated specialization, and exact declarations.
- Shared Prepare sees the selected CPU plan opaquely and assigns only the unchanged four buffer
  declarations. It learns no species, lane, chunk, worker, or strategy policy.
- Runtime receives one immutable partition-level executable and one cold-bound invocation. It does
  not select a route or strategy, create workers, split ranges, or interpret graph/IR/access facts.
- Backend analysis remains deterministic from its explicit static partition facts, typed CPU
  inputs, and captured Java 26 preferred-species/platform facts. Finalization may verify but not
  change the selected strategy or species.
- Prepared recipes remain immutable and reusable. Concurrent runs use distinct Runtime
  `RunState` objects and distinct bound chunk-call state; the explicitly borrowed worker group is
  the only shared CPU orchestration resource.
- The generated hot path sees no `Operation`, `CompiledNode`, Runtime slot, route lookup, carrier
  classification, worker scheduling, reflection, map, cursor, or generic task object.
- Exact/default numerical semantics remain the eligibility boundary. Vector execution must satisfy
  the existing GELU oracle and special-value contract; hardware availability never grants relaxed
  permission.
- Any need for shared Prepare/Runtime/Engine lifecycle ownership, a new dependency, or an
  authoritative architecture change is a stop condition.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — typed configuration, cold strategy
  selection, selected parallelism/species facts, finalization verification.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — immutable selected portable
  realization facts without graph interpretation.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — strategy/species compatibility and
  generated schema identity.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — scalar/vector direct carrier and
  loop emission.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — borrowed worker ownership, cold
  chunk-call binding, synchronous orchestration, and failure joining.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — unchanged canonical IR and access plans.
- `io.github.pho001.synaptik.backend.cpu.internal.memory` — unchanged direct heap/segment argument
  and shared-arena accessibility contracts.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — unchanged scalar conformance oracle.

Packages added or changed:

- No package is added and no responsibility moves between packages.

Type placement:

- `CpuPartitionAnalysisInputs.PortableExecutionConfig` — nested immutable CPU-private configuration
  because it is consumed only with the existing analysis inputs and does not justify a public
  Config contract.
- `CpuPartitionPreparationPlan.ExecutionStrategy` — retains the existing four-name Cartesian
  vocabulary and gains exact selected range-count/minimum-range/species facts in the surrounding
  preparation/portable plan rather than another strategy hierarchy.
- `CpuVectorEmitter` — package-private beside the scalar/carrier/loop emitters because it emits
  the one vector body from already-selected canonical IR.
- `CpuWorkerGroup` — technically public only below unsupported `internal.executable`, because
  `internal.prepare.CpuPartitionFinalizer` must borrow it across Java package boundaries. It owns
  workers and synchronous range coordination only.
- `CpuWorkerGroup.CpuParallelExecutionException` — nested CPU-private unchecked coordination
  failure; it does not become a Runtime or public API exception.
- `CpuPreparedExecutable` — retains the selected direct artifact and optionally borrows the exact
  worker group; it owns cold chunk calls, not worker lifetime.

## Affected files

Expected production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLoopEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorEmitter.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuWorkerGroup.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`

Expected test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFusedGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedKernelShapeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuShapePolymorphicArtifactTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuWorkerGroupTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected explanatory and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0005c-vector-and-parallel-portable-strategies.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized by default.

## Maximum scope

This task may create or modify at most 35 paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| CPU production | 18 | Sixteen existing production/package paths plus two new types |
| CPU tests | 12 | Eleven existing tests plus one new worker-group test |
| Explanatory documentation | 2 | CPU backend guide and glossary |
| Planning/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **35** | **18 + 12 + 2 + 3** |

If implementation needs a path outside this map, another production abstraction, or any shared
module change, stop and revise the specification before coding. Do not spend the ceiling on
unrelated cleanup.

## Acceptance criteria

- CPU 0005A and 0005B remain `Complete`; their exact topology, provenance, virtuality, access
  plans, declarations, alias/span/resource rules, carrier-pattern specialization, and generated
  cache identity boundaries remain unchanged except for the new strategy/species compatibility
  facts.
- The four strategy names remain exactly scalar, vector, parallel-scalar, and parallel-vector.
  Focused selection tests exercise each strategy plus vector-preferred scalar fallback.
- Default analysis remains scalar/single-thread and preserves the exact existing four-segment
  compatibility behavior.
- Configured and available parallelism are positive immutable snapshots. Usable and selected
  counts follow the exact formulas in Scope; no process-wide property, environment variable,
  common pool, or hot lookup changes them.
- Vector specialization uses one exact Java 26 preferred FLOAT64 species with more than one lane.
  Species bit size changes compatibility identity; concrete extents and parallel chunk parameters
  do not.
- Vector code executes exact ADD, the current error-function-polynomial GELU formula, MUL, and
  direct output stores. Generated/reference differential tests cover ordinary values, both
  polynomial branches and their threshold neighborhoods, tails, subnormals, NaN, infinities, and
  signed zero under the existing tolerance.
- `DENSE_LINEAR`, `SCALAR_ALL_ZERO`, `LAST_AXIS_BIAS`, and `BLOCK_OUTER` follow the exact vector
  eligibility table across heap, segment, and mixed carriers. Any `GENERAL_ODOMETER` boundary
  selects scalar fallback; no gather bytecode or gather promise exists.
- Arbitrary legal non-zero starts, ends, block boundaries, lane counts, and chunk boundaries
  produce correct results. Every full vector stays inside all participating contiguous runs and
  every remainder uses the scalar body exactly once.
- Scalar tails are the only 0005C tail policy. No masked vector tail, out-of-range lane access,
  carrier touch for an empty range, or per-lane scalar GELU call exists.
- Every generated class has one selected direct static entry. Scalar/vector choice is absent from
  generated runtime control flow; parallel orchestration is absent from generated code.
- All sixteen current ordered heap/segment patterns work for scalar, vector, parallel-scalar, and
  parallel-vector whenever their access geometry is eligible. Mixed carriers neither force a
  copy nor change strategy eligibility, and segment vectors retain native byte order.
- Parallel chunks are deterministic, contiguous, non-empty, disjoint, ascending-indexed, exactly
  cover the requested range, and are bounded by configured, available, selected, and
  minimum-elements constraints.
- Zero elements select scalar/single-thread and execute without worker submission. A one-chunk
  ranged invocation executes inline. Nested same-group submission fails before nested writes.
- `CpuWorkerGroup` owns exactly its fixed daemon platform workers and has the specified explicit
  borrow/close boundary, concurrent-submission isolation, cancellation, interruption restoration,
  deterministic failure joining/suppression, idempotent shutdown, and no surviving worker after a
  normally returning close.
- Cold binding rejects a closed/undersized worker group and any segment not accessible to every
  selected worker before execution. A worker or coordination failure surfaces through the one
  partition invocation and leaves Runtime output validity unchanged from existing failure rules.
- Non-empty generated scalar/vector worker bodies perform no allocation per element or vector
  lane, cursor construction, virtual callback, division/modulo per element, semantic dispatch,
  storage discovery, route/strategy switch, map lookup, reflection, registry, or service lookup.
- Generated-artifact hit/miss/incompatibility tests cover the schema change, scalar/vector
  distinction, exact species, and exclusion of extents, chunk configuration, worker identity, and
  run state.
- No Java or build change occurs outside `backends/cpu`; no shared contract, architecture,
  dependency, Config, Prepare, Runtime, Engine, conformance, or integration behavior changes.
- A separate clean documentation-focused pass finalizes affected Javadocs, package summaries, CPU
  guide, glossary, and planning/status evidence without repeating a successful CPU test suite
  unless executable behavior changes afterward.
- CPU 0005C becomes `Complete` only after every implementation, documentation, and validation gate
  passes. CPU 0005D and all later tasks remain `Draft` without detailed specifications.

## Tests / validation

Run focused tests while implementing, including:

```bash
./gradlew :backends:cpu:test --tests '*CpuPartitionPreparerTest' --tests '*CpuKernelSpecializationTest'
./gradlew :backends:cpu:test --tests '*CpuFusedGeneratedKernelTest' --tests '*CpuPreparedExecutableTest'
./gradlew :backends:cpu:test --tests '*CpuWorkerGroupTest'
```

After executable Java stabilizes, the implementation context runs exactly one final CPU module
suite:

```bash
./gradlew :backends:cpu:test
```

That final suite must record suite/test counts and cover:

- all four strategies, fallback ordering, default compatibility, configured/available/minimum-
  range bounding, exact preferred species, and compatibility identity;
- the four vector-admitted regimes plus general-odometer scalar fallback across arbitrary starts,
  block boundaries, full vectors, scalar tails, zero/singleton ranges, and all carrier patterns;
- generated/reference numerical agreement and special-value classifications;
- deterministic disjoint chunk coverage, inline one-chunk behavior, worker accessibility, nested
  rejection, concurrent submissions, cancellation, lowest-range-index primary failure,
  suppression identity/order, interrupt restoration, racing close, idempotent close, and worker
  termination;
- artifact schema/hit/miss/incompatibility and direct one-entry generated-class shape; and
- absence of vector gather, masked tails, hot dispatch, per-element/lane allocation, and Runtime or
  shared-service orchestration.

The clean documentation-focused context reuses the successful test evidence unless it changes
executable behavior, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also records exact commands and results for:

- local Markdown targets and explicit heading anchors in the five authorized Markdown files;
- balanced Markdown fences, final newlines, and trailing whitespace;
- exact 18-production/12-test inventory and 35-path ceiling;
- no Java path outside `backends/cpu`, no new package, no native/vendor placeholder, and no shared
  module/build change;
- no generated/hot-path forbidden type, service, dispatch, gather, masked-tail, cursor,
  per-element division/modulo, or allocation vocabulary outside documented cold/test code;
- CPU 0005A/0005B `Complete`, 0005C synchronized `Complete` after implementation, and 0005D/later
  `Draft` with no detailed specification; and
- documentation wording that distinguishes completed 0005B scalar access from planned/implemented
  0005C strategy behavior according to the task status at review time.

Repository-wide validation is deferred to the portable generated-coverage closure checkpoint and
CI because implementation remains within the CPU module and changes no dependency or shared
contract. Architecture, backend-conformance, and integration suites are not run because 0005C
changes no dependency rule, cross-backend conformance claim, or end-to-end Engine behavior.

## Dependencies

- [CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md) is `Complete`.
- [CPU 0005B](0005b-universal-access-plans-and-right-aligned-broadcasting.md) is `Complete`.
- Current Java 26 `jdk.incubator.vector` setup in `backends/cpu/build.gradle.kts`, including compile,
  test, and Javadoc module flags.
- Current CPU whole-partition lowering, canonical IR, five access plans, ordered carrier
  specialization, scalar/reference generation, artifact store, finalization, buffer arguments,
  and prepared executable.
- Current Runtime cold-binding, per-run isolation, output-validity, and failure-cleanup contracts.

## Follow-up tasks

- CPU 0005D remains the next `Draft` row for direct-versus-contiguous materialization,
  specialization budgets, and persistence evidence. It may reconsider gather only with a separate
  cost and correctness proof; 0005C makes no gather commitment.
- CPU 0005E and later rows remain `Draft` for broader types, operations, and families.
- A future dynamic/symbolic execution task still requires an explicit shared exact-binding
  contract and is not implied here.

## Architecture impact

Expected impact: None.

This task implements CPU-private route and orchestration behavior already permitted by the
architecture contract. If implementation requires a shared worker lifecycle, Runtime/Prepare/
Engine change, new dependency, or architecture rule, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean coding context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, completed CPU tasks 0005A and 0005B, and this CPU task
0005C in full. Inspect the complete current CPU production/test/build inventory and the directly
relevant Runtime binding/failure contracts.

Implement task 0005C exactly as specified within its exact path map and 35-path ceiling. Preserve
0005A/0005B history and boundaries; do not implement gather, materialization, another operation or
type, shared-module/architecture/build changes, later detailed specs, commits, or pushes. Stop if
an architecture or scope conflict appears.

After executable Java stabilizes and the one final CPU suite passes, hand the diff and exact test
evidence to a separate clean documentation-focused context. That pass follows the documentation
rules, finalizes affected Javadocs plus the authorized CPU guide/glossary/planning files, runs CPU
Javadoc and documentation validation, and updates this task's evidence, notes, summary, and final
status. It does not repeat successful Java tests unless executable behavior changes or a concrete
risk is recorded.
```

## Local decisions

- Java 26 preferred FLOAT64 species is the only vector candidate; species selection is cold and
  exact, not a hot lookup or tunable candidate family in 0005C.
- Exact GELU is vectorized through the current error-function polynomial with Vector API masks and
  `EXP`. Per-lane scalar GELU and tanh substitution are rejected.
- Vector execution uses unmasked full vectors plus the existing scalar body for every tail.
- General odometer access selects scalar fallback. Vector gather is neither implemented nor
  promised.
- The worker group is explicit caller-owned CPU-private infrastructure borrowed by finalized
  executables. This avoids adding lifecycle to `PreparedExecution`, Runtime, Prepare, or Engine.
- Parallel failure choice is deterministic by lowest completed failing range index after all
  started work joins; detection still cancels unclaimed work promptly.

## Known limitations

- Semantic coverage remains only the exact fully static resolved-layout FLOAT64
  `ADD -> exact GELU -> MUL` topology.
- Vectorization is limited to direct contiguous runs and scalar broadcast. General positive-
  strided odometers use scalar fallback; there is no gather or materialization.
- Only the Java 26 preferred FLOAT64 species and scalar tails are implemented. There is no species
  candidate search, masked tail, unroll, tile, or benchmark claim.
- Parallel orchestration requires explicit CPU-private worker-group composition. The repository
  still has no public Engine composition path for it.
- No performance, tuning, persistence-speed, backend-conformance, integration, native-route,
  relaxed-math, dynamic-shape, or broader-family claim is made.

## Validation evidence

The implementation context supplied the final executable evidence after correcting the CPU test
suite:

- `./gradlew :backends:cpu:test` passed 18 suites and 49 tests with zero failures, errors, or
  skips. No executable Java or test changed afterward. The documentation pass changed Javadoc
  comments only and therefore did not rerun this successful suite.
- The final suite covers all four strategies and vector-preferred scalar fallback; preferred
  species and specialization identity; direct array, segment, and mixed-carrier vector access;
  admitted contiguous regimes, arbitrary bounds, block boundaries, scalar tails, zero/one-chunk
  behavior, and general-odometer fallback; deterministic disjoint chunking; nested submission,
  failure ordering/suppression, interruption, racing close, idempotent shutdown, and worker
  accessibility; artifact compatibility; and the preserved exact fused topology.

Clean documentation context `/root/cpu_0005c_docs` independently reviewed the complete diff,
final source/tests, Java 26 CPU build setup, architecture/planning contracts, completed CPU 0005A
and 0005B, and the General, API/Javadoc, Backend Guide, Planning, and Example profiles. It
finalized every affected permitted Javadoc and package summary plus the CPU guide, glossary, this
task, CPU master plan, and roadmap. Its final validation established:

- `./gradlew :backends:cpu:javadoc` passed after the final Javadoc edits with `BUILD SUCCESSFUL`;
  Gradle reported 11 actionable tasks, 2 executed and 9 up-to-date. Remaining warnings were the
  incubating Vector API module and pre-existing default-constructor/unchanged scalar-emitter
  documentation warnings; there was no Javadoc error.
- A repository-local read-only validator checked all 691 local Markdown targets and 291 explicit
  heading anchors in the five authorized Markdown files with zero errors. It also confirmed
  balanced fences, final newlines, and no trailing whitespace.
- The exact authorized inventory contains 18 production paths, 12 test paths, and five Markdown
  paths under the 35-path ceiling. The final change uses 31 paths: all 18 production paths, eight
  test paths, and all five Markdown paths, with zero paths outside the map and zero Java paths
  outside `backends/cpu`.
- Package/native/vendor/shared/build checks found no new package, native or vendor placeholder,
  Java outside the CPU backend, shared-module change, or Gradle change. The Java 26 Vector module
  flags in `backends/cpu/build.gradle.kts` remain unchanged and sufficient for compile, test, and
  Javadoc.
- The generated/hot-path vocabulary audit found no `Operation`, `CompiledNode`, service locator,
  registry, gather/scatter, index/lane map, masked tail, or Runtime/Engine worker orchestration.
  Division/remainder occurs only in cold species/chunk calculation; map/queue allocation occurs
  only in caller-owned worker coordination; Class-File lookup/reflection vocabulary is confined
  to cold generation/definition; generated scalar/vector bodies retain no per-element/lane
  allocation, division/modulo, storage discovery, or strategy dispatch.
- Status/history checks confirmed CPU 0005A and 0005B remain `Complete`, CPU 0005C is synchronized
  `Complete`, CPU 0005D and every later task remain `Draft`, and no detailed 0005D-or-later task
  specification exists.
- `git diff --check` passed after final documentation and status edits.

## Implementation notes

CPU analysis now snapshots a typed compute preference, configured/available parallelism, and
minimum elements per worker. It selects all four strategies deterministically and records the Java
26 preferred FLOAT64 species only for vector compute. Scalar artifacts contain no species;
parallel-scalar and parallel-vector reuse their corresponding single-thread generated bodies.

Generated vector entries directly load/store `double[]` or `MemorySegment` carriers selected by
the ordered specialization, with native byte order for segments. Dense, scalar-broadcast,
last-axis, and block/outer regimes use complete unmasked vectors and the existing scalar body for
remainders. General odometer and too-short contiguous runs select scalar compute. There is no
gather, masked tail, per-lane scalar GELU, carrier switch, or generated parallel scheduler.

`CpuWorkerGroup` owns a fixed caller-managed lifetime of named daemon platform workers.
Finalization verifies and borrows an open sufficiently large group for parallel plans; prepared
executables never close it. Cold binding computes direct calls, geometry, and quotient/remainder
chunk boundaries. Execution preserves the specified zero/inline/multi-chunk behavior, nested
rejection, deterministic failure selection/suppression, interrupt restoration, racing-close
cancellation, and idempotent full shutdown.

The exact three-node topology remains one execution unit and one partition executable. The ADD
and GELU results remain two virtual graph/IR values, and the three inputs plus final output remain
the only four declared boundaries. No Model, Compiler, Planning, shared Prepare, Runtime, Config,
Trace, Backend Contract, Engine, architecture, dependency, build, native/provider,
backend-conformance, or integration change was required.

## Completion summary

Completed and documented preferred-species FLOAT64 vector, parallel-scalar, and parallel-vector
execution for the existing exact fused proving topology. CPU 0005A and 0005B remain Complete;
CPU 0005D and every later task remain Draft without detailed specifications.

- Completed changes: cold four-strategy selection, direct exact-species vector generation with
  scalar tails and scalar fallback, explicit borrowed worker-group orchestration, deterministic
  chunk/failure/interrupt/close behavior, and strategy/species artifact compatibility.
- Files changed or created: the authorized 18 CPU production/package paths, eight of the 12
  authorized CPU test paths, and five documentation/planning paths; 31 total under the 35-path
  ceiling.
- Tests and validation: reused the final corrected 18-suite/49-test CPU pass; CPU Javadoc,
  Markdown link/anchor/formatting, exact-scope, excluded-path, vocabulary, status, and whitespace
  gates passed.
- Documentation-agent review: `/root/cpu_0005c_docs` independently finalized affected Javadocs,
  package summaries, backend-guide explanation, glossary terms, and synchronized planning records.
- Documentation impact: the guide now explains current selection, direct carrier/species access,
  scalar fallback/tails, worker ownership, chunks, failures, interruption, close, and limitations.
- Javadoc review: changed internal contracts now document input constraints, results, ownership,
  lifecycle, concurrency, failure, species, and compatibility semantics.
- Glossary impact: portable route, generated kernel, specialization, preparation/executable,
  access-plan, execution-strategy, and new worker-group entries describe the implemented boundary.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
