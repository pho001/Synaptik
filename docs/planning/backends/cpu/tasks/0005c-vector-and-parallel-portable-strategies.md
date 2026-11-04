# Task 0005C: [Vector](../../../../glossary.md#term-vector) and Parallel Portable Strategies

## Reader context

Compute strategy and work distribution are separate decisions. CPU analysis chose scalar
instructions or Java Vector API lane operations, then chose one range or several disjoint output
ranges. Parallel-scalar and parallel-vector plans reused the corresponding generated body;
neither required a second graph evaluator or a scheduler inside the class.

CPU finalization borrowed an explicitly supplied `CpuWorkerGroup`, and Runtime binding captured
the direct carriers and each range’s primitive geometry. The group’s external owner retained its
close duty; reusable prepared code did not acquire that ownership. Joining all started ranges
before return is why synchronous failure handling matters for storage lifetime. The exact
FLOAT64 topology and vector eligibility below qualify this checkpoint only.

Current reading alongside this historical record:

- [CPU portable route plans](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalrouteportable)
  — separates compute form, range orchestration, and invocation-private carriers.
- [CPU binding, validation, and ranges](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalexecutable)
  — identifies current-value validation, direct invocation state, and worker/resource lifetimes.

## Status

Complete

## Goal

Add the three remaining portable execution strategies—[Vector API](../../../../glossary.md#term-vector-api), parallel-[scalar](../../../../glossary.md#term-scalar), and
parallel-[vector](../../../../glossary.md#term-vector)—to the exact fully static [FLOAT64](../../../../glossary.md#term-float64) `ADD -> exact GELU -> MUL` whole-[partition](../../../../glossary.md#partition)
[topology](../../../../glossary.md#term-topology) completed by [CPU](../../../../glossary.md#term-cpu) 0005B. All four strategies consume the same route-independent
`CpuKernelIr`, the same five access state machines, the same four-boundary ordered [carrier](../../../../glossary.md#term-storage-carrier)-pattern
[specialization](../../../../glossary.md#term-specialization), and the same cold `start`/`end` [binding](../../../../glossary.md#term-binding) contract.

CPU analysis makes every eligibility and selection decision before shared slot assignment.
Finalization realizes the already-selected scalar or vector generated class and, for a parallel
plan, verifies one explicitly supplied CPU-private worker group. [Runtime](../../../../glossary.md#term-runtime) cold binding creates
direct calls for deterministic disjoint logical-output chunks. Generated hot loops perform no
route, compute-strategy, orchestration, carrier, access-regime, or [operation](../../../../glossary.md#operation) [dispatch](../../../../glossary.md#term-dispatch) and allocate
nothing per [element](../../../../glossary.md#term-element) or [vector lane](../../../../glossary.md#simd).

The [mental model](../../../../glossary.md#term-mental-model) is:

```text
0005B canonical IR + access bindings + carrier pattern
  -> cold vector eligibility + bounded parallelism + strategy selection
  -> one scalar or exact-species vector generated artifact
  -> single call, or CPU-private deterministic chunk orchestration
  -> joined success or one joined failure
```

CPU 0005C is complete only for this exact proving [slice](../../../../glossary.md#slice). CPU 0005D and later rows remain Draft
without detailed specifications.

## [Scope](../../../../glossary.md#term-scope)

- Preserve the exact three-[node](../../../../glossary.md#node), one-unit, fully static [FLOAT64](../../../../glossary.md#term-float64) `ADD -> exact GELU -> MUL`
  topology, exact/default numerical mode, fixed instruction order, two virtual intermediates,
  four boundary declarations, producer/provenance rules, alias and span decisions, and one
  partition-level <span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>[`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) established by 0005A and 0005B.
- Preserve `CpuAccessPlan` as the sole normalized access family and preserve its five state
  machines without adding a vector-specific broadcast/layout planner:<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
  `DENSE_LINEAR`, `SCALAR_ALL_ZERO`, `LAST_AXIS_BIAS`, `BLOCK_OUTER`, and
  `GENERAL_ODOMETER`.
- Add one typed [CPU](../../../../glossary.md#term-cpu)-private portable execution configuration to
  `CpuPartitionAnalysisInputs`. It records:
  - compute preference: [`SCALAR`](../../../../glossary.md#term-scalar) or `VECTOR_IF_ELIGIBLE`;
  - positive configured maximum parallelism;
  - positive available-parallelism [snapshot](../../../../glossary.md#term-snapshot) supplied by CPU composition/platform discovery; and
  - positive minimum [elements](../../../../glossary.md#term-element) per worker.
- Preserve `CpuPartitionAnalysisInputs.DEFAULT` as [manifest](../../../../glossary.md#term-manifest)-disabled, four ordered
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
  specializations contain no vector species.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api); [Capture glossary](../../../../glossary.md#term-capture-meanings); [lane glossary](../../../../glossary.md#term-lane); [specialization glossary](../../../../glossary.md#term-specialization); [metadata glossary](../../../../glossary.md#term-metadata); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings); [artifact glossary](../../../../glossary.md#term-artifact); [lowering glossary](../../../../glossary.md#lowering); [vector species glossary](../../../../glossary.md#term-vector-species))</span>
- Select the strategy deterministically during CPU analysis:
  1. zero elements select scalar/single-thread;
  2. determine [vector](../../../../glossary.md#term-vector) eligibility under the exact table below;
  3. determine parallel eligibility from the bounded range count above;
  4. choose parallel-vector when vector and parallel are eligible;
  5. otherwise choose vector when vector alone is eligible;
  6. otherwise choose parallel-scalar when parallel alone is eligible; and
  7. otherwise choose scalar.
- Treat `VECTOR_IF_ELIGIBLE` as a preference with scalar fallback, not a required route. An
  admitted `GENERAL_ODOMETER` binding or too-small contiguous run therefore selects scalar or
  parallel-scalar rather than failing an otherwise supported partition.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Vectorize exactly the regimes and carrier forms described in the eligibility table below. Do
  not implement Vector API gather, scatter, index maps, lane maps, or a general-stride vector
  approximation.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [gather glossary](../../../../glossary.md#gather); [stride glossary](../../../../glossary.md#term-stride))</span>
- Add a package-private `CpuVectorEmitter` beside the existing scalar/carrier/loop emitters. It
  emits the current exact GELU error-function polynomial with FLOAT64 Vector API arithmetic,<span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu))</span>
  [`EXP`](../../../../glossary.md#term-expx), comparisons, masks, and blends. It must preserve the current branch thresholds,
  coefficient order, exact GELU formula, NaN/infinity/signed-zero classifications, and numerical
  tolerance. It must not substitute <span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>`GELU_TANH_APPROXIMATION`, call scalar GELU once per lane,
  reassociate, contract, or enable relaxed math.
- Generate unmasked full-vector bodies only. Within each contiguous run, begin at the exact
  arbitrary cold-bound start address, execute as many complete preferred-species vectors as fit
  before either the logical `end` or the earliest participating access-plan contiguous-run
  boundary, then execute the remaining elements through the existing scalar body. This scalar
  prologue/tail rule also handles a non-zero start, a worker chunk ending mid-block, and runs
  shorter than one vector. No masked tail is added in 0005C.
- Use direct `DoubleVector.fromArray` / `intoArray` and
  `DoubleVector.fromMemorySegment` / `intoMemorySegment` forms selected at generation time for the
  existing ordered heap/segment pattern. Every segment vector load/store passes<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
  `ByteOrder.nativeOrder()`, preserving the current scalar carrier contract. Mixed carriers
  remain supported without a generated carrier switch.
- Keep vector and scalar code shaping cold. A generated class contains the one selected body and
  one exact direct entry signature; it does not contain a runtime branch between scalar and vector
  strategies. Parallel-scalar reuses the scalar generated body; parallel-vector reuses the vector
  generated body. Parallel orchestration remains outside generated code.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Reintroduce one minimal CPU-private [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) in `internal.executable`. It owns a fixed
  positive number of named daemon platform workers for its explicit caller-managed lifetime. It
  is not a public executor facade, general task system, common pool, virtual-thread scheduler,
  registry, or Runtime service.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry))</span>
- Make worker [ownership](../../../../glossary.md#term-ownership-meanings) explicit: composition/test code constructs and closes the group;
  `CpuPartitionFinalizer`, `CpuPreparedExecutable`, and bound invocations borrow it and never close
  it. The default finalizer remains valid for single-thread plans. Finalizing a parallel plan
  requires an open supplied group whose worker count is at least the plan's selected range count;
  otherwise finalization fails before constructing the executable.
- Keep `CpuPreparedExecutable` immutable and concurrently bindable. For a parallel plan, cold
  binding computes the exact ranged bindings, geometry arrays, direct handle calls, and
  quotient/remainder chunk boundaries once per bound invocation. It proves every selected segment
  accessible to every worker and retains no generic carrier array or Runtime lookup in worker
  execution.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [Partition](../../../../glossary.md#partition) any requested non-empty `[start, end)` into
  `min(selectedRangeCount, ceil((end - start) / minimumElementsPerWorker))` contiguous non-empty
  chunks. Quotient/remainder division occurs only during cold binding. Chunks appear in ascending
  range-index order, cover the requested range exactly once, and never overlap logically.
  Existing output injectivity then proves their physical output writes disjoint. Input ranges may
  overlap because they are read-only.<span data-glossary-navigation="true"> ([injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Execute a zero range without worker submission or carrier access. Execute a non-empty range that
  reduces to one chunk directly on the invoking thread. Submit two or more chunks synchronously to
  the borrowed worker group and return only after every started chunk has quiesced.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Reject nested submission from one of the same group's owned workers with
  `IllegalStateException("CPU worker must not submit parallel work")` before a nested chunk starts.
  This rejection applies only when an invocation would submit two or more chunks. A zero-range
  invocation remains a no-op and a one-chunk invocation remains inline even when invoked by an
  owned worker. Distinct concurrent external submissions remain isolated and share only bounded
  worker capacity.
- Define joined failure and [cancellation](../../../../glossary.md#term-cancellation) exactly:
  - the first detected unchecked worker failure requests cancellation so no unclaimed chunk
    starts;
  - already-started chunks [run](../../../../glossary.md#term-run) to completion and all workers are joined;
  - after joining, the failure at the lowest completed failing range index is rethrown unchanged;
  - later distinct failures are suppressed in ascending range-index order, skipping the exact
    primary object to avoid self-suppression;
  - no write rollback is claimed, and Runtime's existing executable failure transition leaves all
    declared output copies invalid.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- If the invoking thread is interrupted while joining, request cancellation, join already-started
  work, restore interrupt status, and throw the CPU-private
  `CpuParallelExecutionException("CPU parallel execution interrupted", cause)`. Suppress distinct
  completed worker failures in ascending range-index order.
- Make `CpuWorkerGroup.close()` thread-safe and idempotent. It rejects new work, cancels unclaimed
  chunks, joins started chunks, wakes and joins every owned worker, and returns only when none
  survives. Closing from an owned worker fails with<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [idempotent glossary](../../../../glossary.md#term-idempotent))</span>
  `IllegalStateException("CPU worker must not close its worker group")`. A close racing a
  submission surfaces the earlier worker failure when one exists; otherwise the submission fails
  with `CpuParallelExecutionException("CPU parallel execution cancelled by worker-group close")`.
- Preserve generated-class process-local reuse and optional persistence. Strategy and exact vector
  species are compatibility facts; configured/available parallelism, selected range count,
  minimum range size, worker-group identity, chunk boundaries, extents, carriers, slots, and run
  identity remain prepared/cold instance facts unless they alter emitted code.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
- Update detailed Javadoc and package documentation for every changed or new CPU contract. A
  separate clean documentation-focused pass after Java stabilization must finalize the CPU guide,
  glossary, and planning/status records in the same overall change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>

## [Vector](../../../../glossary.md#term-vector) eligibility and fallback table

The table is evaluated for all four ordered materialized boundaries after 0005B [normalization](../../../../glossary.md#term-normalization).
`contiguous run` means the checked product of the plan's trailing [contiguous](../../../../glossary.md#term-contiguous) suffix for the cold
[binding](../../../../glossary.md#term-binding).

| Access regime | 0005C vector behavior | Eligibility condition |
|---|---|---|
| `DENSE_LINEAR` | Direct contiguous vector load/store. | Eligible when the requested unit contains at least one full species. |
| `SCALAR_ALL_ZERO` | Load one [scalar](../../../../glossary.md#term-scalar) and broadcast it to all [lanes](../../../../glossary.md#term-lane). Writes never use this regime for a repeated address. | Eligible for read inputs. |
| `LAST_AXIS_BIAS` | Direct vectors within the final-[axis](../../../../glossary.md#term-axis) [run](../../../../glossary.md#term-run); [scalar tail](../../../../glossary.md#term-scalar-tail) before reset. | Eligible when the final-axis [extent](../../../../glossary.md#term-extent) is at least one species. |
| `BLOCK_OUTER` | Direct vectors within each trailing contiguous block; scalar tail before outer [carry](../../../../glossary.md#term-carry). | Eligible when the contiguous-run length is at least one species. |
| `GENERAL_ODOMETER` | No [gather](../../../../glossary.md#gather) and no vector body. | Vector-ineligible; select scalar or parallel-scalar fallback. |

Whole-unit vector eligibility requires every non-scalar boundary to satisfy its row, the output to
remain an admitted [injective](../../../../glossary.md#term-injective) direct store, and the full [element](../../../../glossary.md#term-element) count to contain at least one
species. [Heap](../../../../glossary.md#term-heap), [`MemorySegment`](../../../../glossary.md#term-memorysegment), and every mixed four-boundary [carrier](../../../../glossary.md#term-storage-carrier) pattern have the same regime
eligibility. Carrier [provenance](../../../../glossary.md#provenance) does not change the decision.

## Out of [scope](../../../../glossary.md#term-scope)

- Any [operation](../../../../glossary.md#operation), [topology](../../../../glossary.md#term-topology), or [data type](../../../../glossary.md#data-type--datatype) beyond the exact completed [FLOAT64](../../../../glossary.md#term-float64)
  `ADD -> exact GELU -> MUL` proving chain.
- Vector gather/scatter, index-map allocation, general-stride Vector API access, or a promise that
  every 0005B regime is vectorized.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [gather glossary](../../../../glossary.md#gather); [stride glossary](../../../../glossary.md#term-stride); [Vector API glossary](../../../../glossary.md#term-vector-api))</span>
- CPU-internal materialization, copy units, direct-versus-contiguous cost comparison, fixed-shape
  specialization, unrolling, persistence benchmarks, tuning, or autotuning. CPU 0005D owns those
  decisions.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [materialization glossary](../../../../glossary.md#materialization); [contiguous glossary](../../../../glossary.md#term-contiguous); [shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization); [unrolling glossary](../../../../glossary.md#term-unrolling); [persistence glossary](../../../../glossary.md#term-persistence); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- A new public/general CPU configuration API. The 0005C configuration is typed, CPU-private
  analysis input only; shared Config remains unchanged.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Another planner, candidate registry, route registry, executor facade, generic task abstraction,
  common/general thread pool, <span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade))</span>`ForkJoinPool.commonPool()`, `parallelStream`, virtual threads, or
  Runtime/Engine service lookup.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Worker [ownership](../../../../glossary.md#term-ownership-meanings) by [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), Runtime, shared Prepare, Engine, a global singleton, or
  a generated class. No shared lifecycle contract changes in this task.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [singleton glossary](../../../../glossary.md#term-singleton); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Dynamic or symbolic Shape binding, unresolved layouts, materialization, native or vendor routes,
  OpenBLAS integration, broader operation families, reductions/scans, relaxed math, tuning, or
  benchmark-driven selection.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [layouts glossary](../../../../glossary.md#layout); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Changes to Model, Compiler, Planning, shared Prepare, Runtime, Config, Backend Contract, Trace,
  Engine, architecture contracts, dependencies, Gradle configuration, architecture tests,
  backend-conformance tests, or integration tests.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Changes to completed CPU 0005A or 0005B task history, or creation of a detailed CPU 0005D or
  later task specification.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Runtime,
  Prepare, concrete backend ownership, CPU routes, and run lifecycle.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [concrete backend glossary](../../../../glossary.md#concrete-backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [run glossary](../../../../glossary.md#term-run); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.
- [CPU backend master plan](../master-plan.md)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>.
- [Completed CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md).
- [Completed CPU 0005B](0005b-universal-access-plans-and-right-aligned-broadcasting.md).
- [CPU backend guide](../../../../backend-guide/cpu-backend.md#atomic-partition-kernel-reset).
- [Glossary: CPU portable execution strategy](../../../../glossary.md#cpu-portable-execution-strategy).

## Architecture constraints

- Planning selects only CPU ownership. CPU analysis owns Vector API eligibility, bounded
  parallelism, strategy selection, generated specialization, and exact declarations.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Vector API glossary](../../../../glossary.md#term-vector-api); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Shared Prepare sees the selected CPU plan opaquely and assigns only the unchanged four buffer
  declarations. It learns no species, lane, chunk, worker, or strategy policy.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [buffer glossary](../../../../glossary.md#term-buffer); [lane glossary](../../../../glossary.md#term-lane))</span>
- Runtime receives one immutable partition-level executable and one cold-bound invocation. It does
  not select a route or strategy, create workers, split ranges, or interpret graph/IR/access facts.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Backend analysis remains deterministic from its explicit static partition facts, typed CPU
  inputs, and captured Java 26 preferred-species/platform facts. Finalization may verify but not
  change the selected strategy or species.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>
- Prepared recipes remain [immutable](../../../../glossary.md#term-immutable) and reusable. Concurrent runs use distinct [Runtime](../../../../glossary.md#term-runtime)
  [`RunState`](../../../../glossary.md#run-state--runstate) objects and distinct bound chunk-call state; the explicitly borrowed worker group is
  the only shared CPU orchestration resource.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- The generated [hot path](../../../../glossary.md#term-hot-path) sees no [`Operation`](../../../../glossary.md#operation), `CompiledNode`, Runtime slot, route lookup, carrier
  classification, worker scheduling, reflection, map, cursor, or generic task object.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Exact/default numerical semantics remain the eligibility boundary. Vector execution must satisfy
  the existing GELU oracle and special-value contract; hardware availability never grants relaxed
  permission.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Any need for shared Prepare/Runtime/Engine lifecycle ownership, a new dependency, or an
  authoritative architecture change is a stop condition.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — typed configuration, cold strategy
  selection, selected parallelism/species facts, finalization verification.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — immutable selected portable
  realization facts without graph interpretation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — strategy/species compatibility and
  generated schema identity.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — scalar/vector direct carrier and
  loop emission.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — borrowed worker ownership, cold
  chunk-call binding, synchronous orchestration, and failure joining.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [binding glossary](../../../../glossary.md#term-binding); [synchronous glossary](../../../../glossary.md#term-synchronous))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — unchanged [canonical](../../../../glossary.md#term-canonical-form) [IR](../../../../glossary.md#term-intermediate-representation-ir) and access plans.
- `io.github.pho001.synaptik.backend.cpu.internal.memory` — unchanged direct heap/segment argument
  and shared-arena accessibility contracts.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [argument glossary](../../../../glossary.md#term-argument); [arena glossary](../../../../glossary.md#term-arena))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — unchanged [scalar](../../../../glossary.md#term-scalar) [conformance](../../../../glossary.md#term-conformance) [oracle](../../../../glossary.md#term-test-oracle).

Packages added or changed:

- No package is added and no responsibility moves between packages.

Type placement:

- `CpuPartitionAnalysisInputs.PortableExecutionConfig` — nested immutable CPU-private configuration
  because it is consumed only with the existing analysis inputs and does not justify a public
  Config contract.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- `CpuPartitionPreparationPlan.ExecutionStrategy` — retains the existing four-name Cartesian
  vocabulary and gains exact selected range-count/minimum-range/species facts in the surrounding
  preparation/portable plan rather than another strategy hierarchy.
- `CpuVectorEmitter` — package-private beside the scalar/carrier/loop emitters because it emits
  the one vector body from already-selected canonical IR.
- [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) — technically public only below unsupported `internal.executable`, because
  `internal.prepare.CpuPartitionFinalizer` must borrow it across Java package boundaries. It owns
  workers and synchronous range coordination only.
- `CpuWorkerGroup.CpuParallelExecutionException` — nested CPU-private unchecked coordination
  failure; it does not become a Runtime or public API exception.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `CpuPreparedExecutable` — retains the selected direct artifact and optionally borrows the exact
  worker group; it owns cold chunk calls, not worker lifetime.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>

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

Expected explanatory and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0005c-vector-and-parallel-portable-strategies.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized by default.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 35 paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production | 18 | Sixteen existing production/package paths plus two new types |
| CPU tests | 12 | Eleven existing tests plus one new worker-group test |
| Explanatory documentation | 2 | CPU [backend](../../../../glossary.md#backend) guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **35** | **18 + 12 + 2 + 3** |

If implementation needs a path outside this map, another production abstraction, or any shared
[module](../../../../glossary.md#term-module) change, stop and revise the specification before coding. Do not spend the ceiling on
unrelated cleanup.

## Acceptance criteria

- [CPU](../../../../glossary.md#term-cpu) 0005A and 0005B remain `Complete`; their exact topology, provenance, virtuality, access
  plans, declarations, alias/span/resource rules, carrier-pattern specialization, and generated
  cache identity boundaries remain unchanged except for the new strategy/species compatibility
  facts.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [provenance glossary](../../../../glossary.md#provenance); [alias glossary](../../../../glossary.md#aliasing-and-arena); [carrier glossary](../../../../glossary.md#term-storage-carrier); [specialization glossary](../../../../glossary.md#term-specialization); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- The four strategy names remain exactly scalar, vector, parallel-scalar, and parallel-vector.
  Focused selection tests exercise each strategy plus vector-preferred scalar fallback.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector))</span>
- Default analysis remains scalar/single-thread and preserves the exact existing four-segment
  compatibility behavior.
- Configured and available parallelism are positive immutable snapshots. Usable and selected
  counts follow the exact formulas in Scope; no process-wide property, environment variable,
  common pool, or hot lookup changes them.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Scope glossary](../../../../glossary.md#term-scope); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Vector specialization uses one exact Java 26 preferred FLOAT64 species with more than one lane.
  Species bit size changes compatibility identity; concrete extents and parallel chunk parameters
  do not.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [lane glossary](../../../../glossary.md#term-lane); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- Vector code executes exact ADD, the current error-function-polynomial GELU formula, MUL, and
  direct output stores. Generated/reference differential tests cover ordinary values, both
  polynomial branches and their threshold neighborhoods, tails, subnormals, NaN, infinities, and
  signed zero under the existing tolerance.<span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [subnormals glossary](../../../../glossary.md#term-subnormal); [NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- `DENSE_LINEAR`, `SCALAR_ALL_ZERO`, `LAST_AXIS_BIAS`, and `BLOCK_OUTER` follow the exact vector
  eligibility table across heap, segment, and mixed carriers. Any <span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>`GENERAL_ODOMETER` boundary
  selects scalar fallback; no gather bytecode or gather promise exists.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Arbitrary legal non-zero starts, ends, block boundaries, lane counts, and chunk boundaries
  produce correct results. Every full vector stays inside all participating contiguous runs and
  every remainder uses the scalar body exactly once.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Scalar tails are the only 0005C tail policy. No masked vector tail, out-of-range lane access,
  carrier touch for an empty range, or per-lane scalar GELU call exists.
- Every generated class has one selected direct static entry. Scalar/vector choice is absent from
  generated runtime control flow; parallel orchestration is absent from generated code.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- All sixteen current ordered heap/segment patterns work for scalar, vector, parallel-scalar, and
  parallel-vector whenever their access geometry is eligible. Mixed carriers neither force a
  copy nor change strategy eligibility, and segment vectors retain native byte order.<span data-glossary-navigation="true"> ([byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries))</span>
- Parallel chunks are deterministic, contiguous, non-empty, disjoint, ascending-indexed, exactly
  cover the requested range, and are bounded by configured, available, selected, and
  minimum-elements constraints.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>
- Zero elements select scalar/single-thread and execute without worker submission. A one-chunk
  ranged invocation executes inline. Nested same-group submission fails before nested writes.
- [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) owns exactly its fixed daemon platform workers and has the specified explicit
  borrow/close boundary, concurrent-submission isolation, cancellation, interruption restoration,
  deterministic failure joining/suppression, idempotent shutdown, and no surviving worker after a
  normally returning close.<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [idempotent glossary](../../../../glossary.md#term-idempotent))</span>
- Cold binding rejects a closed/undersized worker group and any segment not accessible to every
  selected worker before execution. A worker or coordination failure surfaces through the one
  partition invocation and leaves Runtime output validity unchanged from existing failure rules.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [partition glossary](../../../../glossary.md#partition))</span>
- Non-empty generated scalar/vector worker bodies perform no allocation per element or vector
  lane, cursor construction, virtual callback, division/modulo per element, semantic dispatch,
  storage discovery, route/strategy switch, map lookup, reflection, registry, or service lookup.<span data-glossary-navigation="true"> ([vector lane glossary](../../../../glossary.md#simd); [callback glossary](../../../../glossary.md#term-callback); [dispatch glossary](../../../../glossary.md#term-dispatch); [registry glossary](../../../../glossary.md#term-registry))</span>
- Generated-artifact hit/miss/incompatibility tests cover the schema change, scalar/vector
  distinction, exact species, and exclusion of extents, chunk configuration, worker identity, and
  run state.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [schema glossary](../../../../glossary.md#term-schema); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- No Java or build change occurs outside `backends/cpu`; no shared contract, architecture,
  dependency, Config, Prepare, Runtime, Engine, conformance, or integration behavior changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- A separate clean documentation-focused pass finalizes affected Javadocs, package summaries, CPU
  guide, glossary, and planning/status evidence without repeating a successful CPU test suite
  unless executable behavior changes afterward.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- CPU 0005C becomes `Complete` only after every implementation, documentation, and validation gate
  passes. CPU 0005D and all later tasks remain <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`Draft` without detailed specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests while implementing, including:

```bash
./gradlew :backends:cpu:test --tests '*CpuPartitionPreparerTest' --tests '*CpuKernelSpecializationTest'
./gradlew :backends:cpu:test --tests '*CpuFusedGeneratedKernelTest' --tests '*CpuPreparedExecutableTest'
./gradlew :backends:cpu:test --tests '*CpuWorkerGroupTest'
```

After executable Java stabilizes, the implementation context runs exactly one final [CPU](../../../../glossary.md#term-cpu) [module](../../../../glossary.md#term-module)
suite:

```bash
./gradlew :backends:cpu:test
```

That final suite must record suite/test counts and cover:

- all four strategies, fallback ordering, default compatibility, configured/available/minimum-
  range bounding, exact preferred species, and compatibility identity;<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- the four vector-admitted regimes plus general-odometer scalar fallback across arbitrary starts,
  block boundaries, full vectors, scalar tails, zero/singleton ranges, and all carrier patterns;<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [scalar glossary](../../../../glossary.md#term-scalar); [singleton glossary](../../../../glossary.md#term-singleton); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- generated/reference numerical agreement and special-value classifications;
- deterministic disjoint chunk coverage, inline one-chunk behavior, worker accessibility, nested
  rejection, concurrent submissions, cancellation, lowest-range-index primary failure,
  suppression identity/order, interrupt restoration, racing close, idempotent close, and worker
  termination;<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [idempotent glossary](../../../../glossary.md#term-idempotent))</span>
- [artifact](../../../../glossary.md#term-artifact) [schema](../../../../glossary.md#term-schema)/hit/miss/incompatibility and direct one-entry generated-class shape; and
- absence of vector gather, masked tails, hot dispatch, per-element/lane allocation, and Runtime or
  shared-service orchestration.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [dispatch glossary](../../../../glossary.md#term-dispatch); [element glossary](../../../../glossary.md#term-element); [lane glossary](../../../../glossary.md#term-lane); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

The clean documentation-focused context reuses the successful test evidence unless it changes
executable behavior, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also records exact commands and results for:

- local Markdown [targets](../../../../glossary.md#term-target) and explicit heading anchors in the five authorized Markdown files;
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

Repository-wide validation is deferred to the portable generated-coverage closure [checkpoint](../../../../glossary.md#term-checkpoint) and
[CI](../../../../glossary.md#term-continuous-integration-ci) because implementation remains within the CPU module and changes no [dependency](../../../../glossary.md#term-dependency) or shared
contract. Architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration suites are not run because 0005C
changes no dependency rule, cross-backend conformance claim, or end-to-end [Engine](../../../../glossary.md#term-engine) behavior.

## Dependencies

- [CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> is `Complete`.
- [CPU 0005B](0005b-universal-access-plans-and-right-aligned-broadcasting.md) is `Complete`.
- Current Java 26 `jdk.incubator.vector` setup in `backends/cpu/build.gradle.kts`, including compile,
  test, and Javadoc module flags.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [Javadoc glossary](../../../../glossary.md#term-javadoc); [module glossary](../../../../glossary.md#term-module))</span>
- Current CPU whole-partition lowering, canonical IR, five access plans, ordered carrier
  specialization, scalar/reference generation, artifact store, finalization, buffer arguments,
  and prepared executable.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [canonical glossary](../../../../glossary.md#term-canonical-form); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [carrier glossary](../../../../glossary.md#term-storage-carrier); [specialization glossary](../../../../glossary.md#term-specialization); [scalar glossary](../../../../glossary.md#term-scalar); [artifact glossary](../../../../glossary.md#term-artifact); [buffer glossary](../../../../glossary.md#term-buffer); [arguments glossary](../../../../glossary.md#term-argument); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>
- Current [Runtime](../../../../glossary.md#term-runtime) cold-[binding](../../../../glossary.md#term-binding), per-[run](../../../../glossary.md#term-run) isolation, output-validity, and failure-cleanup contracts.

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0005D remains the next `Draft` row for direct-versus-contiguous materialization,
  specialization budgets, and persistence evidence. It may reconsider gather only with a separate
  cost and correctness proof; 0005C makes no gather commitment.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [materialization glossary](../../../../glossary.md#materialization); [specialization glossary](../../../../glossary.md#term-specialization); [persistence glossary](../../../../glossary.md#term-persistence); [gather glossary](../../../../glossary.md#gather))</span>
- CPU 0005E and later rows remain `Draft` for broader types, operations, and families.
- A future dynamic/symbolic execution task still requires an explicit shared exact-binding
  contract and is not implied here.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>

## Architecture impact

Expected impact: None.

This task implements [CPU](../../../../glossary.md#term-cpu)-private route and orchestration behavior already permitted by the
[architecture contract](../../../../glossary.md#architecture-contract). If implementation requires a shared worker [lifecycle](../../../../glossary.md#lifecycle), [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/
[Engine](../../../../glossary.md#term-engine) change, new [dependency](../../../../glossary.md#term-dependency), or architecture rule, stop and report the conflict.

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
  exact, not a hot lookup or tunable candidate family in 0005C.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Exact [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) is vectorized through the current error-function polynomial with [Vector API](../../../../glossary.md#term-vector-api) [masks](../../../../glossary.md#term-boolean-mask) and
  [`EXP`](../../../../glossary.md#term-expx). Per-[lane](../../../../glossary.md#term-lane) [scalar](../../../../glossary.md#term-scalar) GELU and [tanh](../../../../glossary.md#term-tanh) substitution are rejected.
- [Vector](../../../../glossary.md#term-vector) execution uses unmasked full vectors plus the existing scalar body for every tail.
- General odometer access selects scalar fallback. Vector gather is neither implemented nor
  promised.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather))</span>
- The worker group is explicit caller-owned CPU-private infrastructure borrowed by finalized
  executables. This avoids adding lifecycle to <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), or [Engine](../../../../glossary.md#term-engine).
- Parallel failure choice is deterministic by lowest completed failing range index after all
  started work joins; detection still cancels unclaimed work promptly.

## Known limitations

- Semantic coverage remains only the exact fully static resolved-[layout](../../../../glossary.md#layout) [FLOAT64](../../../../glossary.md#term-float64)
  `ADD -> exact GELU -> MUL` [topology](../../../../glossary.md#term-topology).
- Vectorization is limited to direct contiguous runs and scalar broadcast. General positive-
  strided odometers use scalar fallback; there is no gather or materialization.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [scalar glossary](../../../../glossary.md#term-scalar); [gather glossary](../../../../glossary.md#gather); [materialization glossary](../../../../glossary.md#materialization))</span>
- Only the Java 26 preferred FLOAT64 species and scalar tails are implemented. There is no species
  candidate search, masked tail, unroll, tile, or benchmark claim.
- Parallel orchestration requires explicit CPU-private worker-group composition. The repository
  still has no public Engine composition path for it.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
- No performance, tuning, persistence-speed, backend-conformance, integration, native-route,
  relaxed-math, dynamic-shape, or broader-family claim is made.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [shape glossary](../../../../glossary.md#shape))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context supplied the final executable evidence after correcting the [CPU](../../../../glossary.md#term-cpu) test
suite:

- `./gradlew :backends:cpu:test` passed 18 suites and 49 tests with zero failures, errors, or
  skips. No executable Java or test changed afterward. The documentation pass changed Javadoc
  comments only and therefore did not rerun this successful suite.
- The final suite covers all four strategies and vector-preferred scalar fallback; preferred
  species and specialization identity; direct array, segment, and mixed-carrier vector access;
  admitted contiguous regimes, arbitrary bounds, block boundaries, scalar tails, zero/one-chunk
  behavior, and general-odometer fallback; deterministic disjoint chunking; nested submission,
  failure ordering/suppression, interruption, racing close, idempotent shutdown, and worker
  accessibility; artifact compatibility; and the preserved exact fused topology.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [scalar glossary](../../../../glossary.md#term-scalar); [specialization glossary](../../../../glossary.md#term-specialization); [identity glossary](../../../../glossary.md#term-identity-meanings); [carrier glossary](../../../../glossary.md#term-storage-carrier); [contiguous glossary](../../../../glossary.md#term-contiguous); [idempotent glossary](../../../../glossary.md#term-idempotent); [artifact glossary](../../../../glossary.md#term-artifact); [fused glossary](../../../../glossary.md#specialization-and-fusion); [topology glossary](../../../../glossary.md#term-topology))</span>

Clean documentation context `/root/cpu_0005c_docs` independently reviewed the complete diff,
final source/tests, Java 26 CPU build setup, architecture/[planning](../../../../glossary.md#term-planning) contracts, completed CPU 0005A
and 0005B, and the General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend) Guide, Planning, and Example profiles. It
finalized every affected permitted Javadoc and package summary plus the CPU guide, glossary, this
task, CPU master plan, and roadmap. Its final validation established:

- `./gradlew :backends:cpu:javadoc` passed after the final Javadoc edits with `BUILD SUCCESSFUL`;
  Gradle reported 11 actionable tasks, 2 executed and 9 up-to-date. Remaining warnings were the
  incubating Vector API module and pre-existing default-constructor/unchanged scalar-emitter
  documentation warnings; there was no Javadoc error.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [Vector API glossary](../../../../glossary.md#term-vector-api); [module glossary](../../../../glossary.md#term-module))</span>
- A repository-local read-only validator checked all 691 local Markdown targets and 291 explicit
  heading anchors in the five authorized Markdown files with zero errors. It also confirmed
  balanced fences, final newlines, and no trailing whitespace.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- The exact authorized inventory contains 18 production paths, 12 test paths, and five Markdown
  paths under the 35-path ceiling. The final change uses 31 paths: all 18 production paths, eight
  test paths, and all five Markdown paths, with zero paths outside the map and zero Java paths
  outside `backends/cpu`.
- Package/native/vendor/shared/build checks found no new package, native or vendor placeholder,
  Java outside the CPU backend, shared-module change, or Gradle change. The Java 26 Vector module
  flags in `backends/cpu/build.gradle.kts` remain unchanged and sufficient for compile, test, and
  Javadoc.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- The generated/hot-path vocabulary audit found no [`Operation`](../../../../glossary.md#operation), `CompiledNode`, service locator,
  registry, gather/scatter, index/lane map, masked tail, or Runtime/Engine worker orchestration.
  Division/remainder occurs only in cold species/chunk calculation; map/queue allocation occurs
  only in caller-owned worker coordination; Class-File lookup/reflection vocabulary is confined
  to cold generation/definition; generated scalar/vector bodies retain no per-element/lane
  allocation, division/modulo, storage discovery, or strategy dispatch.<span data-glossary-navigation="true"> ([service locator glossary](../../../../glossary.md#term-service-locator); [registry glossary](../../../../glossary.md#term-registry); [gather glossary](../../../../glossary.md#gather); [lane glossary](../../../../glossary.md#term-lane); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [lookup glossary](../../../../glossary.md#term-lookup); [element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Status/history checks confirmed CPU 0005A and 0005B remain `Complete`, CPU 0005C is synchronized
  `Complete`, CPU 0005D and every later task remain `Draft`, and no detailed 0005D-or-later task
  specification exists.
- `git diff --check` passed after final documentation and status edits.

## Implementation notes

[CPU](../../../../glossary.md#term-cpu) analysis now snapshots a typed compute preference, configured/available parallelism, and
minimum [elements](../../../../glossary.md#term-element) per worker. It selects all four strategies deterministically and records the Java
26 preferred [FLOAT64](../../../../glossary.md#term-float64) species only for [vector](../../../../glossary.md#term-vector) compute. [Scalar](../../../../glossary.md#term-scalar) [artifacts](../../../../glossary.md#term-artifact) contain no species;
parallel-scalar and parallel-vector reuse their corresponding single-thread generated bodies.

Generated vector entries directly load/store `double[]` or [`MemorySegment`](../../../../glossary.md#term-memorysegment) carriers selected by
the ordered [specialization](../../../../glossary.md#term-specialization), with native [byte order](../../../../glossary.md#text-encoding-and-file-boundaries) for segments. Dense, scalar-broadcast,
last-[axis](../../../../glossary.md#term-axis), and block/outer regimes use complete unmasked vectors and the existing scalar body for
remainders. General odometer and too-short [contiguous](../../../../glossary.md#term-contiguous) runs select scalar compute. There is no
[gather](../../../../glossary.md#gather), masked tail, per-[lane](../../../../glossary.md#term-lane) scalar [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu), [carrier](../../../../glossary.md#term-storage-carrier) switch, or generated parallel scheduler.

[`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) owns a fixed caller-managed [lifetime](../../../../glossary.md#term-lifetime) of named daemon platform workers.
Finalization verifies and borrows an open sufficiently large group for parallel plans; prepared
executables never close it. Cold [binding](../../../../glossary.md#term-binding) computes direct calls, geometry, and quotient/remainder
chunk boundaries. Execution preserves the specified zero/inline/multi-chunk behavior, nested
rejection, deterministic failure selection/suppression, interrupt restoration, racing-close
[cancellation](../../../../glossary.md#term-cancellation), and [idempotent](../../../../glossary.md#term-idempotent) full shutdown.

The exact three-[node](../../../../glossary.md#node) [topology](../../../../glossary.md#term-topology) remains one execution unit and one [partition](../../../../glossary.md#partition) executable. The ADD
and GELU results remain two virtual [graph](../../../../glossary.md#graph)/[IR](../../../../glossary.md#term-intermediate-representation-ir) values, and the three inputs plus final output remain
the only four declared boundaries. No [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [Planning](../../../../glossary.md#term-planning), shared [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), Config,
[Trace](../../../../glossary.md#trace), [Backend](../../../../glossary.md#backend) Contract, [Engine](../../../../glossary.md#term-engine), architecture, [dependency](../../../../glossary.md#term-dependency), build, native/provider,
backend-[conformance](../../../../glossary.md#term-conformance), or integration change was required.

## Completion summary

Completed and documented preferred-species [FLOAT64](../../../../glossary.md#term-float64) [vector](../../../../glossary.md#term-vector), parallel-[scalar](../../../../glossary.md#term-scalar), and parallel-vector
execution for the existing exact [fused](../../../../glossary.md#specialization-and-fusion) proving [topology](../../../../glossary.md#term-topology). [CPU](../../../../glossary.md#term-cpu) 0005A and 0005B remain Complete;
CPU 0005D and every later task remain Draft without detailed specifications.

- Completed changes: cold four-strategy selection, direct exact-species vector generation with
  scalar tails and scalar fallback, explicit borrowed worker-group orchestration, deterministic
  chunk/failure/interrupt/close behavior, and strategy/species artifact compatibility.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Files changed or created: the authorized 18 CPU production/package paths, eight of the 12
  authorized CPU test paths, and five documentation/planning paths; 31 total under the 35-path
  ceiling.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the final corrected 18-suite/49-test CPU pass; CPU Javadoc,
  Markdown link/anchor/formatting, exact-scope, excluded-path, vocabulary, status, and whitespace
  gates passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: `/root/cpu_0005c_docs` independently finalized affected Javadocs,
  package summaries, backend-guide explanation, glossary terms, and synchronized planning records.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Documentation impact: the guide now explains current selection, direct carrier/species access,
  scalar fallback/tails, worker ownership, chunks, failures, interruption, close, and limitations.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Javadoc review: changed internal contracts now document input constraints, results, ownership,
  lifecycle, concurrency, failure, species, and compatibility semantics.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Glossary impact: portable route, generated kernel, specialization, preparation/executable,
  access-plan, execution-strategy, and new worker-group entries describe the implemented boundary.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
