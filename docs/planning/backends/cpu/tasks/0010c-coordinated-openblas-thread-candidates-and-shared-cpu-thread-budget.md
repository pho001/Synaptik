# Task 0010C: Coordinated [OpenBLAS](../../../../glossary.md#term-openblas) Thread Candidates and Shared [CPU](../../../../glossary.md#term-cpu) Thread Budget

## Status

Complete

## Goal

Add the first explicitly composed [CPU](../../../../glossary.md#term-cpu) [concurrency](../../../../glossary.md#term-concurrency) boundary for the current portable and [OpenBLAS](../../../../glossary.md#term-openblas)
routes. CPU analysis generates and selects complete positive OpenBLAS thread-count candidates;
CPU finalization installs the selected count through one composition-owned coordinator; and every
coordinated portable or native invocation consumes its fixed demand from one fair [CPU concurrency
budget](../../../../glossary.md#cpu-concurrency-budget--cpuconcurrencybudget).

The coordinator owns exclusion between OpenBLAS calls and thread-count writers, retains the
provider [lifetime](../../../../glossary.md#term-lifetime) transferred from the current discovery session, and restores the positive
pre-configuration count through that still-open [owner](../../../../glossary.md#term-owner) when it closes. It is not a [singleton](../../../../glossary.md#term-singleton) and
cannot control provider use that bypasses it. Existing deployments that do not explicitly compose
and share this coordinator retain the current single-thread and external-exclusion contract.

The [mental model](../../../../glossary.md#term-mental-model) is:

```text
immutable analysis facts
  -> complete positive thread candidates
  -> one selected route plan with fixed permit demand
  -> finalization installs that count while calls are quiescent
  -> each run acquires fixed shared permits, calls, and releases in finally
  -> coordinator close quiesces calls, restores the captured count, then closes the provider
```

## [Scope](../../../../glossary.md#term-scope)

- Add one CPU-private positive-capacity concurrency budget with fair, interruptible acquisition and
  an idempotent lease that releases exactly the acquired permits.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [concurrency glossary](../../../../glossary.md#term-concurrency); [idempotent glossary](../../../../glossary.md#term-idempotent); [lease glossary](../../../../glossary.md#term-lease); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Integrate that same explicit budget with existing portable execution. Inline portable work
  acquires one permit. [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) acquires the exact participant count
  `min(workerCount, submittedRangeCount)` once around each submission and releases it only after
  every started range quiesces.
- Add one CPU-private OpenBLAS coordinator. A loaded discovery session transfers its owned
  invocation and close action to the coordinator exactly once; the session remains an immutable
  metadata view and no longer owns the transferred resource.<span data-glossary-navigation="true"> ([OpenBLAS coordinator glossary](../../../../glossary.md#openblas-coordinator--cpuopenblascoordinator); [immutable glossary](../../../../glossary.md#term-immutable); [metadata glossary](../../../../glossary.md#term-metadata); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Extend the CPU-private provider-free invocation seam with only the already-supported positive
  thread-count setter needed by the coordinator. The low-level OpenBLAS provider API is unchanged.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Replace the closed `SINGLE_THREAD` analysis value with immutable positive thread-count candidate
  facts while preserving the current single-thread factories as compatibility conveniences.
- Generate one complete native route candidate for every unique configured positive thread-count
  candidate that fits the explicit analysis-time CPU capacity. Filter eligibility and calculate
  complete costs before deterministic selection.
- Carry the selected positive provider count and identical native permit demand in the immutable
  OpenBLAS route plan. Preserve all eight CPU 0010B representation masks and exact resources.<span data-glossary-navigation="true"> ([Carry glossary](../../../../glossary.md#term-carry); [masks glossary](../../../../glossary.md#term-boolean-mask))</span>
- Configure the coordinator during CPU finalization, before constructing the prepared native
  recipe. Finalization may not change the selected count or route and must fail if its live budget
  capacity disagrees with the analysis snapshot.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Execute the selected input-copy/GEMM/output-copy sequence under one native budget lease and one
  coordinator call admission. Copies remain inside the lease because they are part of the selected
  CPU plan and must not overlap unrelated portable demand beyond the shared capacity.<span data-glossary-navigation="true"> ([GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [admission glossary](../../../../glossary.md#term-admission))</span>
- Add native-free state-machine, permit, portable/native overlap, failure, interruption, close,
  restoration, candidate, finalization, binding, and conformance tests using fake invocation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Extend the explicit real-native checkpoint for one OpenBLAS 0.3.34 binary. It must exercise two
  positive selected counts when the supplied budget permits, overlapping compatible independent
  calls, exclusion of a configuration writer while a call is active, and restoration in<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
  `finally` through the still-open [owner](../../../../glossary.md#term-owner).
- After executable Java stabilizes, require a distinct clean documentation-focused agent to
  finalize meaningful Javadocs, package documentation, the CPU backend guide, glossary impact,
  this task, CPU master plan, and roadmap in the same overall change.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Automatic installed-binary qualification, resolved-path recovery, build/version inspection,
  target or binary fingerprinting, or CPU 0010D.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [qualification glossary](../../../../glossary.md#term-qualification); [target glossary](../../../../glossary.md#term-target); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Persistent workload or plan caches, measurement, benchmarking, model autotuning, compatible
  tuning evidence, or CPU 0010E.<span data-glossary-navigation="true"> ([benchmarking glossary](../../../../glossary.md#term-benchmarking); [model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- Public Engine composition, public CPU construction API, a public generic thread pool, a public
  Config knob, supported user-facing lifecycle, or automatic global coordinator discovery.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Any OpenBLAS-provider API, source, test, build, symbol, ABI, or documentation change. The CPU
  adapter delegates to existing <span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [symbol glossary](../../../../glossary.md#term-symbol); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>`OpenBlasLibrary.threadCount()` and `setThreadCount(int)`.
- A hidden singleton, static mutable registry, service locator, class-loader-wide lock, native
  interposition, or claim that arbitrary external handles and native callers are controlled.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [mutable glossary](../../../../glossary.md#term-mutable); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- Runtime route choice, runtime candidate generation, per-GEMM thread query or setter, late
  fallback, or provider discovery during analysis, finalization, binding, or execution.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [binding glossary](../../../../glossary.md#term-binding))</span>
- A shared Prepare or Runtime resource/lease contract. The budget and coordinator are borrowed
  CPU-private finalization resources retained directly by prepared CPU recipes.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lease glossary](../../../../glossary.md#term-lease); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Changes to [Planning](../../../../glossary.md#term-planning) [ownership](../../../../glossary.md#term-ownership-meanings) or cost scoring. Planning still selects only
  `BackendId("cpu")` and never sees routes, thread counts, or permits.
- Rank-one, batched, broadcast, empty, zero-contraction, promoted, mixed-type, sparse, quantized,
  or complex MATMUL; epilogues; provider transpose flags; packing; persistent weights; or new
  representation forms.<span data-glossary-navigation="true"> ([Rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [contraction glossary](../../../../glossary.md#contraction-dimension); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [transpose glossary](../../../../glossary.md#term-transpose); [weights glossary](../../../../glossary.md#term-weight))</span>
- Changes to the positive rank-two same-type FLOAT32/FLOAT64 bare MATMUL boundary, the eight 0010B
  copy masks, portable completeness, generated affine-copy semantics, or provider numerical
  contract.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [masks glossary](../../../../glossary.md#term-boolean-mask))</span>
- Generated bytecode, emitter, generated schema, specialization identity, or generated evidence
  changes. If generated code proves necessary, stop and replan; only then does the clean-Java
  semantic/hot-loop/overhead oracle and proportionate Class-File/performance validation apply.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [schema glossary](../../../../glossary.md#term-schema); [specialization glossary](../../../../glossary.md#term-specialization); [identity glossary](../../../../glossary.md#term-identity-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle); [validation glossary](../../../../glossary.md#term-validation))</span>
- Relaxed math, other native vendors, CPU 0011 or later CPU routes, Model/Tensor/Compiler/Training
  APIs, architecture-contract edits, ADRs, module dependencies, Gradle changes, or new testing
  project dependencies.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [Compiler glossary](../../../../glossary.md#term-compiler); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially staged preparation, Runtime cold
  binding, concrete-backend candidate ownership, CPU routes, OpenBLAS provider boundaries, and
  explicit Engine composition<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding); [backend glossary](../../../../glossary.md#backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [Engine glossary](../../../../glossary.md#term-engine))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU 0010 narrow OpenBLAS route](0010-narrow-openblas-blas-compatible-native-route.md)
- [CPU 0010A discovery and internal composition](0010a-automatic-openblas-discovery-and-internal-composition-foundation.md)
- [CPU 0010B bounded representations](0010b-bounded-openblas-matmul-representation-expansion.md)
- [OpenBLAS provider master plan](../../openblas-provider/master-plan.md)
- [Provider 0001 loading and lifetime](../../openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md)<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- [Provider 0002 GEMM invocation](../../openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md)<span data-glossary-navigation="true"> ([GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- [Provider 0003 thread control and checkpoint](../../openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [Runtime master plan](../../../modules/runtime/master-plan.md)
- [Prepare master plan](../../../modules/prepare/master-plan.md)
- [Config master plan](../../../modules/config/master-plan.md)
- [Tuning master plan](../../../tools/tuning/master-plan.md)
- [Engine master plan](../../../modules/engine/master-plan.md)

## Architecture constraints

- CPU preparation owns typed OpenBLAS candidates, filtering, complete route/representation cost,
  selection, provider coordination, and the concrete permit demand. Shared Planning, Prepare, and
  Runtime do not interpret these facts.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Analysis remains deterministic from immutable supplied facts and performs no provider call,
  host probing, benchmark, cache access, or measurement. Provider query/set work is cold
  coordinator/finalization work only.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [host glossary](../../../../glossary.md#term-host); [cache glossary](../../../../glossary.md#term-cache))</span>
- Prepared recipes remain immutable and reusable. A bound invocation retains direct references to
  the budget/coordinator and performs no route selection, count discovery, setter call, map lookup,
  reflection, or graph inspection.<span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [lookup glossary](../../../../glossary.md#term-lookup); [graph glossary](../../../../glossary.md#graph))</span>
- The provider remains a low-level leaf. CPU may adapt its existing direct setter but must not add
  provider policy, a provider-owned route plan, a scoped provider lease, or reverse dependency.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [lease glossary](../../../../glossary.md#term-lease); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- The coordinator is an explicit composition resource, not a run-owned Runtime resource,
  workspace, or hidden global. It owns the transferred provider lifetime; prepared work borrows it.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- The same budget object must be used by every portable worker group, inline portable recipe, and
  OpenBLAS coordinator that claims coordinated execution. Capacity equality is validated at cold
  boundaries; object identity is validated where two live resources must share one budget.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([OpenBLAS coordinator glossary](../../../../glossary.md#openblas-coordinator--cpuopenblascoordinator))</span>
- A selected native failure after analysis is an error. Provider closure, installed-count drift,
  permit interruption, configuration failure, or invocation failure never selects portable late.
- If implementation requires a shared public lifecycle/resource contract, an Engine dependency,
  provider expansion, generated-code change, or architecture update, stop and report the exact
  conflict instead of authorizing it in planning notes.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Current-code evidence

- [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) owns a fixed worker array, rejects submission from its own workers, computes
  participants as `min(workerCount, calls.length)`, and synchronously waits for every started
  participant. One acquisition around that existing submission therefore represents truthful
  portable demand without worker-side nested acquisition or deadlock.<span data-glossary-navigation="true"> ([deadlock glossary](../../../../glossary.md#term-deadlock))</span>
- `CpuPreparedExecutable.Invocation` has exactly two execution forms: one inline `KernelCall`, or
  one `CpuWorkerGroup.execute(calls)` submission. The inline form can acquire one permit, while the
  group owns acquisition for the parallel form. `CpuPartialReductionExecution` also delegates its
  multi-call phase to the same group and needs no second lease.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease))</span>
- `CpuPartitionFinalizer` already receives borrowed CPU-private workers and an optional borrowed
  OpenBLAS invocation. It is the existing cold point that validates worker capacity and provider
  state before immutable executable construction.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- `CpuOpenBlasDiscoverySession` currently owns one invocation/close pair in an atomic reference and
  exposes immutable result metadata independently. An atomic one-time ownership transfer can
  preserve the result while giving the coordinator exclusive close/restoration responsibility.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [metadata glossary](../../../../glossary.md#term-metadata); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `CpuOpenBlasInvocation` currently exposes `isOpen`, `threadCount`, `sgemm`, and `dgemm`; its
  production adapter strongly retains [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary). Adding [CPU](../../../../glossary.md#term-cpu)-private `setThreadCount` needs
  no provider surface change.
- `CpuOpenBlasPreparedExecutable` currently queries for count one during every cold bind and then
  invokes GEMM directly. Coordinated mode can replace that non-atomic check/use pair with one
  coordinator admission around the complete prepared sequence; uncoordinated compatibility mode
  retains the existing check and external-exclusion limitation.<span data-glossary-navigation="true"> ([GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [admission glossary](../../../../glossary.md#term-admission))</span>
- `OpenBlasRouteConfig` currently has one `SINGLE_THREAD` enum value and one [OpenBLAS](../../../../glossary.md#term-openblas) cost triple.
  `CpuOpenBlasRouteSelector` already calculates exact whole-plan cost and fails closed on missing,
  overflowing, tied, or insufficient-benefit facts. The smallest extension is a bounded immutable
  list of `(positive count, complete OpenBLAS cost terms)` candidates and stable selection across
  otherwise identical 0010B representation facts.
- Shared Prepare assigns only buffer/workspace slots, and Runtime binds/executes backend recipes
  without concrete-backend knowledge. No shared contract is needed for a CPU-private borrowed
  concurrency resource.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [buffer glossary](../../../../glossary.md#term-buffer); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>

## Exact design and invariants

### Explicit [CPU concurrency budget](../../../../glossary.md#cpu-concurrency-budget--cpuconcurrencybudget)

Add [`CpuConcurrencyBudget`](../../../../glossary.md#cpu-concurrency-budget--cpuconcurrencybudget) under `internal.executable`. It has one [immutable](../../../../glossary.md#term-immutable) positive capacity and
one fair `Semaphore`. Its package-visible acquisition validates demand in `[1, capacity]`, waits
interruptibly, and returns an [idempotent](../../../../glossary.md#term-idempotent) [lease](../../../../glossary.md#term-lease) containing that exact demand. Interruption restores
the thread interrupt flag and throws one stable [CPU](../../../../glossary.md#term-cpu)-private [runtime](../../../../glossary.md#term-runtime) coordination exception without
leaking permits. Every success [releases](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) exactly once in `finally`, including unchecked failure,
`Error`, [cancellation](../../../../glossary.md#term-cancellation), and [OpenBLAS](../../../../glossary.md#term-openblas) failure. The budget owns no threads and has no `close()`.

Existing constructors remain available and create or retain the current uncoordinated behavior.
New explicit constructors accept a budget:

- a coordinated [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) borrows it for its whole [lifetime](../../../../glossary.md#term-lifetime);
- a coordinated `CpuPartitionFinalizer` borrows it and passes it to every finalized portable
  recipe, including single-thread recipes; and
- [`CpuOpenBlasCoordinator`](../../../../glossary.md#openblas-coordinator--cpuopenblascoordinator) borrows the same object and exposes its identity/capacity for cold
  validation.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>

For portable inline work, demand is one. For worker work, demand is the existing exact participant
count, not submitted range count, configured maximum, or invoking thread: the invoking thread only
waits and performs no range body. Acquisition occurs before queue [publication](../../../../glossary.md#term-publication); no worker acquires
another permit. The group releases only after all started ranges quiesce. This preserves its
current failure ordering, cancellation, close, and interruption behavior and prevents nested
oversubscription.

### Coordinator [ownership](../../../../glossary.md#term-ownership-meanings) and state machine

Add [`CpuOpenBlasCoordinator`](../../../../glossary.md#openblas-coordinator--cpuopenblascoordinator) beside the [OpenBLAS](../../../../glossary.md#term-openblas) route. A loaded
`CpuOpenBlasDiscoverySession` creates it only through an explicit package-private transfer method.
The method atomically removes the session's `OwnedResource`; a second transfer, disabled or
unavailable session, or already-closed session fails. After transfer:

- the session still exposes [immutable](../../../../glossary.md#term-immutable) discovery [metadata](../../../../glossary.md#term-metadata);
- session `close()` is a no-op for the transferred resource;
- the coordinator exclusively owns the invocation and close action; and
- composition must close the coordinator only after all prepared borrowers are [quiescent](../../../../glossary.md#term-quiescent).

The coordinator uses one fair `ReentrantLock` and one condition with these states:

```text
OPEN_UNCONFIGURED -> OPEN_CONFIGURED -> CLOSING -> CLOSED
          ^                 |          ^
          +-- restored -----+          |
                    FAILED -----------+
```

State guarded by the lock is: state, writer-active flag, active admitted-call count, optional
captured original positive count, optional installed positive count, and a terminal `FAILED` flag
when a failed transition cannot prove restoration. There is never more than
one writer. No native query/set/call occurs while holding the Java lock; the writer flag prevents
new call [admission](../../../../glossary.md#term-admission) while the lock is temporarily released for native control calls.

`configure(count)` is interruptible while waiting for `activeCalls == 0` and no writer. It rejects
a non-positive or over-capacity count, closing/closed state, or closed provider. On the first
transition it queries and validates the positive previous count exactly once. It sets the selected
count, queries it back, and publishes `OPEN_CONFIGURED` only on exact equality. Later cold
reconfiguration uses the same original restore [target](../../../../glossary.md#term-target), waits for all calls, and installs one new
count atomically with respect to coordinated calls and writers.

If installation or verification fails, the writer immediately attempts to restore and verify the
captured original count before clearing writer state. A proved restoration returns to
`OPEN_UNCONFIGURED`; an unproved restoration records a terminal failed state accepted only by
`close()`. Restoration failure is suppressed on the original failure. No prepared route is
constructed from a failed configuration.

Call admission first acquires the fixed shared-budget demand, then waits under the coordinator
lock for no writer. It requires `OPEN_CONFIGURED`, an installed count equal to the plan's selected
count, an open provider, and demand equal to that count; it increments `activeCalls` before
releasing the lock. It does not query the provider per call. The complete input-copy/[GEMM](../../../../glossary.md#general-matrix-multiplication--gemm)/
output-copy sequence executes after admission. `finally` decrements `activeCalls`, signals a
waiting writer/closer, and [releases](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) the budget [lease](../../../../glossary.md#term-lease).

`close()` is [idempotent](../../../../glossary.md#term-idempotent) and uninterruptibly completes once won: it marks `CLOSING`, wakes queued
call admissions so they fail and [release](../../../../glossary.md#term-release) permits, waits for the active count and writer to reach
zero, restores and verifies the captured original positive count through the still-open invocation
when configuration was attempted, then invokes the transferred close action. It restores its
caller's interrupt status after cleanup. Provider close failure is suppressed on a primary
restoration failure; state becomes `CLOSED` on every exit. Close invoked from an admitted [callback](../../../../glossary.md#term-callback)
is rejected to avoid self-[deadlock](../../../../glossary.md#term-deadlock). The fake tests must prove these transitions and failure
precedence rather than use timing-only assertions.

Coordinated exclusion covers only calls and writers routed through this exact coordinator.
Separate engines/coordinators, other Java handles, other class loaders, and arbitrary [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture)
remain outside it. They must explicitly share the coordinator or retain provider 0003's external
[quiescence](../../../../glossary.md#term-quiescence)/exclusion requirement.

### Typed candidates and deterministic cost

Replace the one-value `ThreadConfiguration` with a typed [immutable](../../../../glossary.md#term-immutable) candidate carrying:

- positive `threadCount`;
- complete non-negative [OpenBLAS](../../../../glossary.md#term-openblas) fixed/per-output/per-MAC cost terms for that count; and
- stable input order used only after lower-cost and lower-count tie-breaks.

`OpenBlasRouteConfig` snapshots at most 32 candidates, rejects duplicate counts and null entries,
and retains the current `qualifiedSingleThread(...)` factories by constructing exactly one count-1
candidate. A complete qualified config requires at least one candidate and all existing portable,
representation, threshold, [qualification](../../../../glossary.md#term-qualification), and expected-storage facts. The list is explicit
[heuristic](../../../../glossary.md#term-heuristic) input, not measured evidence or a [cache](../../../../glossary.md#term-cache) [artifact](../../../../glossary.md#term-artifact).

Analysis derives its [concurrency](../../../../glossary.md#term-concurrency) capacity from the existing positive
`PortableExecutionConfig.availableParallelism` [snapshot](../../../../glossary.md#term-snapshot). It generates one complete OpenBLAS route
candidate per configured candidate whose count is no greater than that capacity, after the exact
0010B semantic/storage/representation eligibility succeeds. For candidate `t`, use its own
OpenBLAS cost triple in the unchanged checked formula:

```text
portable = R * (portableFixed + portablePerOutput * O + portablePerMac * W)

openblas(t) = R * (openblasFixed(t) + openblasPerOutput(t) * O
                   + openblasPerMac(t) * W + complete 0010B representation cost)
```

There is no assumed [linear](../../../../glossary.md#term-linear-meanings) speedup, division by thread count, benchmark [lookup](../../../../glossary.md#term-lookup), or hidden hardware
[model](../../../../glossary.md#term-model). Each candidate must independently beat portable and both existing thresholds. [Overflow](../../../../glossary.md#term-overflow),
missing terms, no fitting candidate, equality with portable, or insufficient benefit retains the
portable plan. Among eligible native candidates select lower complete cost, then lower thread
count, then stable configured order. This tie rule conserves permits without claiming performance.

The selected `CpuOpenBlasRoutePlan` records the exact candidate, positive selected count, identical
permit demand, analysis capacity, costs, and unchanged representation/resources. Constructor
invariants require `permitDemand == threadCount <= analysisCapacity`. Candidate generation and
selection occur only in [CPU](../../../../glossary.md#term-cpu) analysis; finalization validates and installs that selected fact and
never searches or substitutes a count.

### Finalization, [binding](../../../../glossary.md#term-binding), and execution sequence

For explicitly coordinated construction:

1. Composition creates one `CpuConcurrencyBudget(capacity)`.
2. Composition creates every worker group with that same budget.
3. A loaded discovery session transfers its owned resource into one coordinator using that budget.
4. [CPU](../../../../glossary.md#term-cpu) analysis receives [immutable](../../../../glossary.md#term-immutable) candidate facts and the matching
   `availableParallelism == capacity` [snapshot](../../../../glossary.md#term-snapshot), retains portable, and may select one native plan.
5. CPU finalization validates budget identity/capacity and worker capacity. For a native plan it
   calls <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>`coordinator.configure(selectedThreadCount)` before constructing the recipe.
6. Cold Runtime binding validates buffers, workspaces, copies, provider/coordinator openness, and
   immutable plan agreement, but performs no thread query/set and no permit acquisition.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [buffers glossary](../../../../glossary.md#term-buffer))</span>
7. A portable run acquires one inline permit or exact worker participants. A native run admits its
   fixed demand through the coordinator, performs copies/GEMM/copy-out, and releases on every exit.<span data-glossary-navigation="true"> ([GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
8. After every prepared use is quiescent, composition closes the coordinator, which restores and
   verifies the original count before closing the transferred provider owner; it then closes the
   worker group.<span data-glossary-navigation="true"> ([quiescent glossary](../../../../glossary.md#term-quiescent); [owner glossary](../../../../glossary.md#term-owner))</span>

Independent native calls may overlap only if their fixed demands fit the fair budget and both
plans require the coordinator's currently installed count. For example, capacity eight with count
two admits at most four independent two-permit GEMMs; a fifth waits. A count-four prepared call
cannot [run](../../../../glossary.md#term-run) while count two is installed and fails rather than reconfiguring itself. Explicit cold
reconfiguration to four waits for all admitted count-two calls; afterward old count-two recipes
fail until composition explicitly reconfigures again.

Uncoordinated compatibility constructors remain limited to count one, keep the current bind-time
positive count-one query, and document the existing external exclusion/restoration obligation.
They do not participate in the shared-budget or overlap guarantee.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.executable`
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas`

Packages added or changed:

- `internal.executable` — adds the route-neutral CPU-private permit budget and integrates existing
  worker/inline portable execution.
- `internal.prepare` — carries immutable candidate facts and validates explicitly composed live
  resources during finalization.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `internal.route.nativeblas.openblas` — owns provider-state coordination, candidate/plan facts,
  session ownership transfer, and coordinated prepared execution.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget` — one explicit
  route-neutral CPU permit capacity shared by portable orchestration and native calls.
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasCoordinator`
  — OpenBLAS-specific provider lifetime, writer/call exclusion, installation, and restoration.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Existing `CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate` — nested immutable
  candidate because it is inseparable from the current CPU-private route configuration facts.

Both new top-level types are technically public only where cross-package [CPU](../../../../glossary.md#term-cpu) internals require
access and are unsupported internal [API](../../../../glossary.md#term-application-programming-interface-api). `CpuCapabilityProvider` remains the sole supported public
CPU type. No new package is added.

## Affected files

Expected production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuConcurrencyBudget.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuWorkerGroup.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasInvocation.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasDiscoverySession.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasCoordinator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelector.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/package-info.java`

Expected test and [checkpoint](../../../../glossary.md#term-checkpoint) paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuConcurrencyBudgetTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuWorkerGroupTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasDiscoveryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasCoordinatorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelectorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasNativeCheckpoint.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuOpenBlasRouteConformanceTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) paths during implementation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`, only if the documentation pass finds a reusable term changed or introduced
- `docs/planning/backends/cpu/tasks/0010c-coordinated-openblas-thread-candidates-and-shared-cpu-thread-budget.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless an in-[scope](../../../../glossary.md#term-scope) [Javadoc](../../../../glossary.md#term-javadoc) statement is directly stale:

- `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibrary.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecutable.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/BoundInvocation.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunState.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/BackendPartitionAnalysis.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalization.java`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 29 paths, all within the categories and exact expected paths
above. It may add exactly the two top-level production types and two matching focused test types.
No provider, shared [module](../../../../glossary.md#term-module), public [API](../../../../glossary.md#term-application-programming-interface-api), build, architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), generated-code/[schema](../../../../glossary.md#term-schema), Config,
[Engine](../../../../glossary.md#term-engine), other vendor, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) build, architecture-test, or integration-test path may
change. If another path or top-level type is required, stop and return the task to [planning](../../../../glossary.md#term-planning).

## Acceptance criteria

1. One explicit positive-capacity fair budget accounts for coordinated inline portable demand,
   exact worker participants, and selected OpenBLAS demand, with interrupt-safe and failure-safe
   release and no nested acquisition.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [release glossary](../../../../glossary.md#term-release))</span>
2. Worker-group concurrency, deterministic failure selection, cancellation, close behavior, and
   no-worker-submission rule remain unchanged. Native-free tests prove total active portable plus
   native demand never exceeds capacity.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
3. A loaded discovery session transfers its owned invocation/close pair exactly once. The
   coordinator becomes the sole close/restoration owner while immutable discovery metadata remains
   readable. Disabled, unavailable, closed, or repeated transfer fails deterministically.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [immutable glossary](../../../../glossary.md#term-immutable); [metadata glossary](../../../../glossary.md#term-metadata))</span>
4. The coordinator implements the exact state, fair lock/condition, writer, active-call,
   interruption, failure, reconfiguration, and close protocol above. No query/set/call occurs
   under the Java lock and no thread setter occurs per GEMM.<span data-glossary-navigation="true"> ([GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
5. Every successful first configuration captures one positive original count, installs and
   verifies one selected positive count while calls/writers are excluded, and close restores and
   verifies the original through the still-open owner before provider close. Failure cleanup and
   suppression are proved with fakes.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span>
6. Candidate facts are typed, immutable, unique by positive count, and bounded to 32. Deliberately
   incomplete cost terms remain representable so the complete configuration fails closed without
   guessing. Analysis filters against exact capacity and current 0010B eligibility and generates
   no native candidate from missing, overflowing, unqualified, or ineligible facts.
7. Candidate cost uses the exact per-count terms and unchanged complete representation cost.
   Tests cover multiple counts, over-capacity filtering, all missing terms, overflow, thresholds,
   portable ties, native ties, lower-count tie-break, and stable-order final tie-break without a
   benchmark or assumed scaling formula.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
8. The selected plan records one fixed positive thread count and identical permit demand. The
   existing count-one factories and uncoordinated path preserve current behavior, all eight 0010B
   representations, portable completeness, and fail-closed/no-late-fallback rules.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
9. Coordinated finalization validates exact budget capacity/identity, installs only the selected
   count, and constructs no recipe after a failed transition. Cold binding performs no thread
   query/set or permit acquisition.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [binding glossary](../../../../glossary.md#term-binding))</span>
10. Native execution admits once around copy-in/GEMM/copy-out, releases on every outcome, and never
    reconfigures. Independent fake GEMMs overlap only when demands fit and installed counts agree;
    a writer/close waits for admitted calls, and a mismatched prepared count fails.<span data-glossary-navigation="true"> ([releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
11. The backend-conformance test remains Engine-free, provider-free, and native-free and exercises
    analysis, assignment, coordinated finalization, cold binding, execution, permit release, and
    restoration with at least one count greater than one while retaining direct and expanded
    representation coverage.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>
12. The explicit checkpoint uses one caller-supplied exact OpenBLAS 0.3.34 path, verifies positive
    original state, configured counts one and two (or fails if capacity two is unavailable),
    compatible overlapping count-one calls under capacity two, writer exclusion, FLOAT32/FLOAT64
    route results, and restoration in <span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>`finally` before provider close. It never discovers,
    substitutes, skips, benchmarks, or qualifies the binary automatically.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
13. Public CPU shape, provider API, shared Prepare/Runtime, Planning, Config, Engine, generated
    code/schema, module dependencies, and architecture remain unchanged. Source/API inventory
    confirms no singleton, registry, public generic pool, hot setter/query, or hidden fallback.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [shape glossary](../../../../glossary.md#shape); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Planning glossary](../../../../glossary.md#term-planning); [schema glossary](../../../../glossary.md#term-schema); [module glossary](../../../../glossary.md#term-module); [singleton glossary](../../../../glossary.md#term-singleton); [registry glossary](../../../../glossary.md#term-registry))</span>
14. Every changed Java declaration has meaningful Javadoc covering ownership, lifetime,
    concurrency, parameters, results, nullability, mutation, interruption, failures, and cleanup as
    applicable. A distinct clean documentation-focused agent independently reviews the final diff,
    finalizes Javadocs/package docs/CPU guide/glossary impact/planning evidence, and records
    reasoned no-change conclusions for every review-only and excluded area.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime); [parameters glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability))</span>
15. Focused tests, final affected-project tests, repository-wide validation, the mandatory native
    checkpoint, CPU Javadoc/render inspection, Markdown/scope/status/order checks, and whitespace
    validation pass before 0010C becomes <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [scope glossary](../../../../glossary.md#term-scope))</span>`Complete`.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, use focused tests as needed. After executable Java and tests stabilize,
[run](../../../../glossary.md#term-run) once:

```bash
./gradlew :backends:cpu:test :testing:backend-conformance:test
```

Ordinary tests must be native-free and deterministic. Use latches/barriers and fake invocation
state to prove [admission](../../../../glossary.md#term-admission), exclusion, overlap, waiting, [release](../../../../glossary.md#term-release), interruption, reconfiguration, and
restoration; do not rely only on sleeps or wall-clock ordering.

[Compile](../../../../glossary.md#compile) and run the mandatory native [checkpoint](../../../../glossary.md#term-checkpoint) with the caller-supplied exact 0.3.34 [binary](../../../../glossary.md#term-binary-descriptor-and-metadata):

```bash
./gradlew :backends:cpu:testClasses
java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector \
  -cp backends/cpu/build/classes/java/test:backends/cpu/build/classes/java/main:backends/openblas-provider/build/classes/java/main:modules/model/build/classes/java/main:modules/config/build/classes/java/main:modules/planning/build/classes/java/main:modules/runtime/build/classes/java/main:modules/prepare/build/classes/java/main:modules/backend-contract/build/classes/java/main:modules/trace/build/classes/java/main \
  io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasNativeCheckpoint \
  <ABSOLUTE_OPENBLAS_0_3_34_LIBRARY>
```

The checkpoint must restore in `finally` even when an intermediate assertion fails and report the
verified restored count. Absence of the exact compatible binary leaves the task incomplete.

Because this changes coordinated behavior spanning [CPU](../../../../glossary.md#term-cpu) and [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), the implementation
context then runs one repository-wide gate:

```bash
./gradlew test
```

The final root run may serve as final CPU/conformance evidence if it follows all executable/test
changes. No provider suite, benchmark, tuning workflow, generated evidence regeneration, or
[integration test](../../../../glossary.md#term-integration-test) is required. Architecture tests need no change because dependencies and shared
boundaries do not change; the root suite remains the proportional [regression](../../../../glossary.md#term-regression) gate.

The distinct documentation-focused context reuses successful Java/native evidence unless it
changes executable behavior or tests. After final documentation/[Javadoc](../../../../glossary.md#term-javadoc) edits it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git status --short -uall
```

It renders and inspects every changed generated Javadoc page and validates local Markdown links
and anchors, heading order and uniqueness, balanced fences, conceptual-example labels, terminology,
LF/final newlines, trailing whitespace, exact path [scope](../../../../glossary.md#term-scope), package/public inventory, and status/order.
It must confirm 0010C is `Complete` only after all evidence, 0010D and 0010E remain master-plan-only
`Draft` rows with no detailed files, and CPU 0011 remains `Blocked`.

## Dependencies

- Complete [CPU 0010B](0010b-bounded-openblas-matmul-representation-expansion.md), including all
  eight representation masks and the current OpenBLAS 0.3.34 checkpoint.<span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Complete CPU 0010 and 0010A for the provider-free invocation and transferable internal discovery
  lifetime.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Complete OpenBLAS provider tasks 0001-0003 for exact loading, GEMM, positive thread query/set,
  caller-owned lifetime, and external shared-state contract.<span data-glossary-navigation="true"> ([GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- Stable [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup), portable selected range counts, staged CPU finalization, immutable
  prepared recipes, cold binding, and Runtime run-state isolation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [binding glossary](../../../../glossary.md#term-binding); [Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run))</span>

All prerequisites are present. Current code supports the bounded [CPU](../../../../glossary.md#term-cpu)-private implementation above
without changing an architecture or shared contract.

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0010D remains a master-plan-only `Draft` row and owns automatic installed-binary
  qualification and invalidatable target/binary identity.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [qualification glossary](../../../../glossary.md#term-qualification); [target glossary](../../../../glossary.md#term-target); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- CPU 0010E remains a master-plan-only `Draft` row and owns versioned tuning candidates,
  compatibility signatures, and explicit selected-evidence consumption with measurement/cache
  mutation in <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [cache glossary](../../../../glossary.md#term-cache))</span>`tools/tuning`.
- A future CPU composition/API task may expose a supported facade after Config and Engine contracts
  exist. It must wrap this internal ownership model rather than expose internal packages.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [facade glossary](../../../../glossary.md#term-facade); [Engine glossary](../../../../glossary.md#term-engine); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [model glossary](../../../../glossary.md#term-model))</span>
- CPU 0011 remains `Blocked` after 0010E and still requires a concrete Intel use case and supported
  oneMKL ABI evidence.<span data-glossary-navigation="true"> ([oneMKL glossary](../../../../glossary.md#term-onemkl); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Batch, broadcast, epilogue, packing, persistent representations, relaxed math, and other vendor
  routes are not implied follow-ups.<span data-glossary-navigation="true"> ([Batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [epilogue glossary](../../../../glossary.md#term-epilogue))</span>

## Architecture impact

Expected architecture impact: None. The task implements concrete-[CPU](../../../../glossary.md#term-cpu) candidate, coordination,
finalization, and execution behavior already assigned to the CPU [backend](../../../../glossary.md#backend). The provider remains a
[leaf](../../../../glossary.md#term-leaf), shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) contracts remain backend-neutral, [Planning](../../../../glossary.md#term-planning) sees no route vocabulary,
and [Engine](../../../../glossary.md#term-engine) remains the future supported [composition root](../../../../glossary.md#term-composition-root).

If implementation reveals that explicit coordination cannot be retained as a CPU-private [borrowed](../../../../glossary.md#ownership-borrowing-and-lease)
resource, or requires a shared [lifecycle](../../../../glossary.md#lifecycle) [API](../../../../glossary.md#term-application-programming-interface-api), provider expansion, new [dependency](../../../../glossary.md#term-dependency), or architecture
rule, stop and report the exact uncertainty before editing outside this specification.

## Implementation prompt

Use this prompt in a distinct clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0010C. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, and
docs/planning/backends/cpu/tasks/0010c-coordinated-openblas-thread-candidates-and-shared-cpu-thread-budget.md
in full, plus its directly referenced contracts and current affected source/tests. Implement the
task exactly within its 29-path ceiling. Stop for any architecture, provider, shared-contract,
generated-code, public-API, or scope conflict instead of inventing a boundary.

After executable code and Java/native validation stabilize, hand the exact diff, affected behavior,
architecture constraints, drafted documentation, and recorded evidence to a distinct clean
documentation-focused agent in the same overall change. That agent must follow
documentation-rules.md, independently inspect the implementation/tests, finalize affected Javadocs,
package documentation, CPU guide, glossary impact, task/master/roadmap evidence, and documentation
validation, and must not repeat successful Java suites unless executable behavior changes or a
concrete stale-evidence risk is recorded. Do not mark 0010C Complete until that pass and every
specified gate succeed.
```

## Local decisions

- Use one route-neutral fair permit budget and one OpenBLAS-specific coordinator. Combining them
  would make portable worker orchestration depend on one vendor route; putting provider mutation in
  the worker group would give an execution primitive provider policy it does not own.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Count only actual computing participants. The current parallel invoking thread waits and executes
  no range, so charging it would overstate demand. A native demand equals the installed OpenBLAS
  count, including its invoking/native team semantics as one fixed provider demand.
- Acquire worker permits once before queue publication. Worker-side acquisition could deadlock when
  a job owns some permits while queued workers wait for the remainder.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [deadlock glossary](../../../../glossary.md#term-deadlock))</span>
- Keep copies inside native admission. They are part of the complete selected plan, and excluding
  them would permit extra portable work beyond the explicit CPU budget during the native recipe.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Accept explicit per-count heuristic cost triples. Inferring linear speedup from count would be an
  unsupported performance claim; requiring measured evidence belongs to 0010E.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- Permit cold reconfiguration but never automatic per-call reconfiguration. Prepared plans remain
  fixed; a mismatched installed count fails until composition explicitly transitions while calls
  are quiescent.<span data-glossary-navigation="true"> ([quiescent glossary](../../../../glossary.md#term-quiescent))</span>
- Preserve the old uncoordinated count-one path as a compatibility boundary. Its limitations stay
  explicit, so this task does not pretend that internal coordination controls external consumers.

## Known limitations

- Coordination covers only work explicitly sharing the exact budget and coordinator object.
- There is no supported public Config or Engine composition surface yet; the implementation remains
  CPU-internal and is exercised through focused/conformance/checkpoint paths.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Heuristic per-count costs are caller-supplied immutable facts, not benchmark evidence. Count one
  remains the compatibility default.<span data-glossary-navigation="true"> ([Heuristic glossary](../../../../glossary.md#term-heuristic); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- A prepared plan is usable only while its selected count is installed. Reconfiguration can make an
  older plan temporarily ineligible; execution fails rather than changing provider state itself.
- Fair semaphore/lock admission prevents barging but does not promise operating-system scheduling
  order or performance proportional to permits.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>
- Arbitrary external OpenBLAS users can still race this coordinator and invalidate its guarantees;
  they require external exclusion or explicit adoption of the same coordinator.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [race glossary](../../../../glossary.md#term-race))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context `01a0901c-f6bc-76d1-848e-521e1fb0644b` completed executable work from clean
baseline `69e21ec4` without [commit](../../../../glossary.md#term-commit) or push. The final affected-project command
`./gradlew :backends:cpu:test :testing:backend-conformance:test` passed: [CPU](../../../../glossary.md#term-cpu) reported 973 tests,
28 skipped, zero failures/errors; [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) reported eight tests with zero skips,
failures, or errors. An earlier full affected [run](../../../../glossary.md#term-run) exposed the worker-close [publication](../../../../glossary.md#term-publication) [race](../../../../glossary.md#term-race); after
the [lifecycle](../../../../glossary.md#lifecycle)-monitor correction, its focused rerun and the final full affected rerun passed.

`./gradlew :backends:cpu:testClasses` passed. The required [checkpoint](../../../../glossary.md#term-checkpoint) path
`/opt/homebrew/opt/openblas/lib/libopenblas.dylib` resolved to
`/opt/homebrew/Cellar/openblas/0.3.34/lib/libopenblasp-r0.3.34.dylib`. The exact prescribed Java
checkpoint command passed for selected counts one and two, compatible overlapping count-one calls,
writer exclusion, [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) results across all eight representation [masks](../../../../glossary.md#term-boolean-mask), and restoration
to the captured original count 16. The final repository-wide `./gradlew test` gate passed.

Implementation [scope](../../../../glossary.md#term-scope) was 20 paths including this task. Documentation context
`01a09049-3b8e-7ed2-8554-6d6362a73f9b` independently inspected the implementation and tests, then
finalized the affected [API](../../../../glossary.md#term-application-programming-interface-api) and package Javadocs, CPU backend guide, glossary, this task, CPU master
plan, and roadmap without changing executable Java behavior or tests. Final scope is 24 paths,
below the 29-path ceiling, with exactly two new top-level production types and their two matching
new tests. Source inventory and `javap -public` confirmed that both new types remain in unsupported
internal packages and that `CpuCapabilityProvider` remains the only supported top-level CPU API.

`./gradlew :backends:cpu:javadoc` passed after the final [Javadoc](../../../../glossary.md#term-javadoc) edits. It reported 94 existing
missing-`@param` warnings in unchanged `CpuPartitionLowering.LoweredPartition` and
`CpuPartitionPreparationPlan` declarations, plus the existing incubating-[module](../../../../glossary.md#term-module) warnings; no
warning originated in a declaration changed by this task. The documentation context rendered and
inspected every changed generated Javadoc page: [`CpuConcurrencyBudget`](../../../../glossary.md#cpu-concurrency-budget--cpuconcurrencybudget) and its nested exception and
[lease](../../../../glossary.md#term-lease), `CpuPreparedExecutable`, [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup), `CpuPartitionAnalysisInputs` and its changed
nested route records, `CpuPartitionFinalizer`, `CpuOpenBlasDiscoverySession`,
`CpuOpenBlasInvocation`, `CpuOpenBlasPreparedExecutable`, `CpuOpenBlasRoutePlan`,
`CpuOpenBlasRouteSelector`, [`CpuOpenBlasCoordinator`](../../../../glossary.md#openblas-coordinator--cpuopenblascoordinator), and the [OpenBLAS](../../../../glossary.md#term-openblas) package summary.

The repository Markdown validator passed for the five changed Markdown files, covering local
links and anchors, duplicate headings, balanced fences, LF/final-newline/trailing-whitespace
hygiene, and changed-document terminology. Heading order and new conceptual diagrams/examples were
also inspected: the guide's new flow is explicitly an internal lifecycle [model](../../../../glossary.md#term-model), not a supported API
example. Final inventory checks found no detailed CPU 0010D or 0010E task file and confirmed the
ordered state: 0010C `Complete`, 0010D the next master-plan-only `Draft` frontier, 0010E a
master-plan-only `Draft` behind it, and CPU 0011 `Blocked`.

Because the documentation pass changed no executable behavior or tests, it reused the successful
affected-project, repository-wide, and exact native checkpoint evidence above rather than rerunning
those gates. Final `git diff --check`, exact path-scope, newline/whitespace, and
`git status --short -uall` checks passed.

No architecture rule or [dependency](../../../../glossary.md#term-dependency) boundary changed, so `ARCHITECTURE.md`, the current
architecture plan, ADRs, and architecture tests required no update. The OpenBLAS provider API and
provider documentation still accurately describe caller-owned handle [lifetime](../../../../glossary.md#term-lifetime) and externally
coordinated process-global thread state, so they required no change. Shared [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime),
[Planning](../../../../glossary.md#term-planning), Config, and [Engine](../../../../glossary.md#term-engine) contracts also required no change: candidate selection, finalization,
coordination, and execution remain CPU-private, while Engine remains the future supported
[composition root](../../../../glossary.md#term-composition-root). Generated code, generator [schema](../../../../glossary.md#term-schema)/[fingerprint](../../../../glossary.md#term-fingerprint)/evidence, build files, integration
tests, other backends, benchmarks, and tuning [artifacts](../../../../glossary.md#term-artifact) were unaffected. The glossary did require
a focused update because the [CPU concurrency budget](../../../../glossary.md#cpu-concurrency-budget--cpuconcurrencybudget) and [OpenBLAS coordinator](../../../../glossary.md#openblas-coordinator--cpuopenblascoordinator) are reusable terms;
it adds only their current CPU-private meanings and does not imply a public construction surface.

## Implementation notes

Added a fair explicit [CPU](../../../../glossary.md#term-cpu) permit budget and integrated it with inline portable execution and exact
worker participants. Added transferable discovery [ownership](../../../../glossary.md#term-ownership-meanings) and one fair-lock [OpenBLAS coordinator](../../../../glossary.md#openblas-coordinator--cpuopenblascoordinator)
whose guarded state, writer exclusion, admitted calls, verified configuration, failure [rollback](../../../../glossary.md#term-rollback),
restoration, and close ordering remain CPU-private. Provider query/set operations occur outside the
Java lock and execution performs no per-[GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) thread mutation.

Replaced the closed count-one analysis representation with bounded, [immutable](../../../../glossary.md#term-immutable), unique positive
thread candidates carrying per-count complete costs. Selection filters by the analysis capacity
and uses complete cost, lower count, then stable input order. The route plan fixes identical thread
and permit demand. Coordinated finalization validates live budget [identity](../../../../glossary.md#term-identity-meanings)/capacity and installs the
selected count before [artifact](../../../../glossary.md#term-artifact) construction; compatibility constructors retain count-one behavior.
Prepared native execution admits the complete copy-in/GEMM/copy-out sequence once.

The implementation preserved the eight 0010B representation [masks](../../../../glossary.md#term-boolean-mask) and changed no provider,
shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime), [Planning](../../../../glossary.md#term-planning), Config, [Engine](../../../../glossary.md#term-engine), architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [dependency](../../../../glossary.md#term-dependency), [Gradle](../../../../glossary.md#term-gradle),
generated-code/[schema](../../../../glossary.md#term-schema), benchmark, integration-test, or other-[backend](../../../../glossary.md#backend) path. The separate
documentation context finalized the affected Javadocs, [OpenBLAS](../../../../glossary.md#term-openblas) package description, CPU guide,
glossary, and planning records and validated their rendered and source forms.

## Completion summary

Executable implementation and [validation](../../../../glossary.md#term-validation) are complete. The mandatory separate documentation pass
is also complete: affected contracts and user-facing explanations are finalized, implementation
evidence is retained, all documentation gates pass, and no unresolved issue or follow-up is
required for [CPU](../../../../glossary.md#term-cpu) 0010C.

```text
Status: Complete
```
