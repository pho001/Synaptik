# Task 0010C: Coordinated OpenBLAS Thread Candidates and Shared CPU Thread Budget

## Status

Ready

## Goal

Add the first explicitly composed CPU concurrency boundary for the current portable and OpenBLAS
routes. CPU analysis generates and selects complete positive OpenBLAS thread-count candidates;
CPU finalization installs the selected count through one composition-owned coordinator; and every
coordinated portable or native invocation consumes its fixed demand from one fair CPU concurrency
budget.

The coordinator owns exclusion between OpenBLAS calls and thread-count writers, retains the
provider lifetime transferred from the current discovery session, and restores the positive
pre-configuration count through that still-open owner when it closes. It is not a singleton and
cannot control provider use that bypasses it. Existing deployments that do not explicitly compose
and share this coordinator retain the current single-thread and external-exclusion contract.

The mental model is:

```text
immutable analysis facts
  -> complete positive thread candidates
  -> one selected route plan with fixed permit demand
  -> finalization installs that count while calls are quiescent
  -> each run acquires fixed shared permits, calls, and releases in finally
  -> coordinator close quiesces calls, restores the captured count, then closes the provider
```

## Scope

- Add one CPU-private positive-capacity concurrency budget with fair, interruptible acquisition and
  an idempotent lease that releases exactly the acquired permits.
- Integrate that same explicit budget with existing portable execution. Inline portable work
  acquires one permit. `CpuWorkerGroup` acquires the exact participant count
  `min(workerCount, submittedRangeCount)` once around each submission and releases it only after
  every started range quiesces.
- Add one CPU-private OpenBLAS coordinator. A loaded discovery session transfers its owned
  invocation and close action to the coordinator exactly once; the session remains an immutable
  metadata view and no longer owns the transferred resource.
- Extend the CPU-private provider-free invocation seam with only the already-supported positive
  thread-count setter needed by the coordinator. The low-level OpenBLAS provider API is unchanged.
- Replace the closed `SINGLE_THREAD` analysis value with immutable positive thread-count candidate
  facts while preserving the current single-thread factories as compatibility conveniences.
- Generate one complete native route candidate for every unique configured positive thread-count
  candidate that fits the explicit analysis-time CPU capacity. Filter eligibility and calculate
  complete costs before deterministic selection.
- Carry the selected positive provider count and identical native permit demand in the immutable
  OpenBLAS route plan. Preserve all eight CPU 0010B representation masks and exact resources.
- Configure the coordinator during CPU finalization, before constructing the prepared native
  recipe. Finalization may not change the selected count or route and must fail if its live budget
  capacity disagrees with the analysis snapshot.
- Execute the selected input-copy/GEMM/output-copy sequence under one native budget lease and one
  coordinator call admission. Copies remain inside the lease because they are part of the selected
  CPU plan and must not overlap unrelated portable demand beyond the shared capacity.
- Add native-free state-machine, permit, portable/native overlap, failure, interruption, close,
  restoration, candidate, finalization, binding, and conformance tests using fake invocation.
- Extend the explicit real-native checkpoint for one OpenBLAS 0.3.34 binary. It must exercise two
  positive selected counts when the supplied budget permits, overlapping compatible independent
  calls, exclusion of a configuration writer while a call is active, and restoration in
  `finally` through the still-open owner.
- After executable Java stabilizes, require a distinct clean documentation-focused agent to
  finalize meaningful Javadocs, package documentation, the CPU backend guide, glossary impact,
  this task, CPU master plan, and roadmap in the same overall change.

## Out of scope

- Automatic installed-binary qualification, resolved-path recovery, build/version inspection,
  target or binary fingerprinting, or CPU 0010D.
- Persistent workload or plan caches, measurement, benchmarking, model autotuning, compatible
  tuning evidence, or CPU 0010E.
- Public Engine composition, public CPU construction API, a public generic thread pool, a public
  Config knob, supported user-facing lifecycle, or automatic global coordinator discovery.
- Any OpenBLAS-provider API, source, test, build, symbol, ABI, or documentation change. The CPU
  adapter delegates to existing `OpenBlasLibrary.threadCount()` and `setThreadCount(int)`.
- A hidden singleton, static mutable registry, service locator, class-loader-wide lock, native
  interposition, or claim that arbitrary external handles and native callers are controlled.
- Runtime route choice, runtime candidate generation, per-GEMM thread query or setter, late
  fallback, or provider discovery during analysis, finalization, binding, or execution.
- A shared Prepare or Runtime resource/lease contract. The budget and coordinator are borrowed
  CPU-private finalization resources retained directly by prepared CPU recipes.
- Changes to Planning ownership or cost scoring. Planning still selects only
  `BackendId("cpu")` and never sees routes, thread counts, or permits.
- Rank-one, batched, broadcast, empty, zero-contraction, promoted, mixed-type, sparse, quantized,
  or complex MATMUL; epilogues; provider transpose flags; packing; persistent weights; or new
  representation forms.
- Changes to the positive rank-two same-type FLOAT32/FLOAT64 bare MATMUL boundary, the eight 0010B
  copy masks, portable completeness, generated affine-copy semantics, or provider numerical
  contract.
- Generated bytecode, emitter, generated schema, specialization identity, or generated evidence
  changes. If generated code proves necessary, stop and replan; only then does the clean-Java
  semantic/hot-loop/overhead oracle and proportionate Class-File/performance validation apply.
- Relaxed math, other native vendors, CPU 0011 or later CPU routes, Model/Tensor/Compiler/Training
  APIs, architecture-contract edits, ADRs, module dependencies, Gradle changes, or new testing
  project dependencies.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially staged preparation, Runtime cold
  binding, concrete-backend candidate ownership, CPU routes, OpenBLAS provider boundaries, and
  explicit Engine composition
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU 0010 narrow OpenBLAS route](0010-narrow-openblas-blas-compatible-native-route.md)
- [CPU 0010A discovery and internal composition](0010a-automatic-openblas-discovery-and-internal-composition-foundation.md)
- [CPU 0010B bounded representations](0010b-bounded-openblas-matmul-representation-expansion.md)
- [OpenBLAS provider master plan](../../openblas-provider/master-plan.md)
- [Provider 0001 loading and lifetime](../../openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md)
- [Provider 0002 GEMM invocation](../../openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md)
- [Provider 0003 thread control and checkpoint](../../openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md)
- [Runtime master plan](../../../modules/runtime/master-plan.md)
- [Prepare master plan](../../../modules/prepare/master-plan.md)
- [Config master plan](../../../modules/config/master-plan.md)
- [Tuning master plan](../../../tools/tuning/master-plan.md)
- [Engine master plan](../../../modules/engine/master-plan.md)

## Architecture constraints

- CPU preparation owns typed OpenBLAS candidates, filtering, complete route/representation cost,
  selection, provider coordination, and the concrete permit demand. Shared Planning, Prepare, and
  Runtime do not interpret these facts.
- Analysis remains deterministic from immutable supplied facts and performs no provider call,
  host probing, benchmark, cache access, or measurement. Provider query/set work is cold
  coordinator/finalization work only.
- Prepared recipes remain immutable and reusable. A bound invocation retains direct references to
  the budget/coordinator and performs no route selection, count discovery, setter call, map lookup,
  reflection, or graph inspection.
- The provider remains a low-level leaf. CPU may adapt its existing direct setter but must not add
  provider policy, a provider-owned route plan, a scoped provider lease, or reverse dependency.
- The coordinator is an explicit composition resource, not a run-owned Runtime resource,
  workspace, or hidden global. It owns the transferred provider lifetime; prepared work borrows it.
- The same budget object must be used by every portable worker group, inline portable recipe, and
  OpenBLAS coordinator that claims coordinated execution. Capacity equality is validated at cold
  boundaries; object identity is validated where two live resources must share one budget.
- A selected native failure after analysis is an error. Provider closure, installed-count drift,
  permit interruption, configuration failure, or invocation failure never selects portable late.
- If implementation requires a shared public lifecycle/resource contract, an Engine dependency,
  provider expansion, generated-code change, or architecture update, stop and report the exact
  conflict instead of authorizing it in planning notes.

## Current-code evidence

- `CpuWorkerGroup` owns a fixed worker array, rejects submission from its own workers, computes
  participants as `min(workerCount, calls.length)`, and synchronously waits for every started
  participant. One acquisition around that existing submission therefore represents truthful
  portable demand without worker-side nested acquisition or deadlock.
- `CpuPreparedExecutable.Invocation` has exactly two execution forms: one inline `KernelCall`, or
  one `CpuWorkerGroup.execute(calls)` submission. The inline form can acquire one permit, while the
  group owns acquisition for the parallel form. `CpuPartialReductionExecution` also delegates its
  multi-call phase to the same group and needs no second lease.
- `CpuPartitionFinalizer` already receives borrowed CPU-private workers and an optional borrowed
  OpenBLAS invocation. It is the existing cold point that validates worker capacity and provider
  state before immutable executable construction.
- `CpuOpenBlasDiscoverySession` currently owns one invocation/close pair in an atomic reference and
  exposes immutable result metadata independently. An atomic one-time ownership transfer can
  preserve the result while giving the coordinator exclusive close/restoration responsibility.
- `CpuOpenBlasInvocation` currently exposes `isOpen`, `threadCount`, `sgemm`, and `dgemm`; its
  production adapter strongly retains `OpenBlasLibrary`. Adding CPU-private `setThreadCount` needs
  no provider surface change.
- `CpuOpenBlasPreparedExecutable` currently queries for count one during every cold bind and then
  invokes GEMM directly. Coordinated mode can replace that non-atomic check/use pair with one
  coordinator admission around the complete prepared sequence; uncoordinated compatibility mode
  retains the existing check and external-exclusion limitation.
- `OpenBlasRouteConfig` currently has one `SINGLE_THREAD` enum value and one OpenBLAS cost triple.
  `CpuOpenBlasRouteSelector` already calculates exact whole-plan cost and fails closed on missing,
  overflowing, tied, or insufficient-benefit facts. The smallest extension is a bounded immutable
  list of `(positive count, complete OpenBLAS cost terms)` candidates and stable selection across
  otherwise identical 0010B representation facts.
- Shared Prepare assigns only buffer/workspace slots, and Runtime binds/executes backend recipes
  without concrete-backend knowledge. No shared contract is needed for a CPU-private borrowed
  concurrency resource.

## Exact design and invariants

### Explicit CPU concurrency budget

Add `CpuConcurrencyBudget` under `internal.executable`. It has one immutable positive capacity and
one fair `Semaphore`. Its package-visible acquisition validates demand in `[1, capacity]`, waits
interruptibly, and returns an idempotent lease containing that exact demand. Interruption restores
the thread interrupt flag and throws one stable CPU-private runtime coordination exception without
leaking permits. Every success releases exactly once in `finally`, including unchecked failure,
`Error`, cancellation, and OpenBLAS failure. The budget owns no threads and has no `close()`.

Existing constructors remain available and create or retain the current uncoordinated behavior.
New explicit constructors accept a budget:

- a coordinated `CpuWorkerGroup` borrows it for its whole lifetime;
- a coordinated `CpuPartitionFinalizer` borrows it and passes it to every finalized portable
  recipe, including single-thread recipes; and
- `CpuOpenBlasCoordinator` borrows the same object and exposes its identity/capacity for cold
  validation.

For portable inline work, demand is one. For worker work, demand is the existing exact participant
count, not submitted range count, configured maximum, or invoking thread: the invoking thread only
waits and performs no range body. Acquisition occurs before queue publication; no worker acquires
another permit. The group releases only after all started ranges quiesce. This preserves its
current failure ordering, cancellation, close, and interruption behavior and prevents nested
oversubscription.

### Coordinator ownership and state machine

Add `CpuOpenBlasCoordinator` beside the OpenBLAS route. A loaded
`CpuOpenBlasDiscoverySession` creates it only through an explicit package-private transfer method.
The method atomically removes the session's `OwnedResource`; a second transfer, disabled or
unavailable session, or already-closed session fails. After transfer:

- the session still exposes immutable discovery metadata;
- session `close()` is a no-op for the transferred resource;
- the coordinator exclusively owns the invocation and close action; and
- composition must close the coordinator only after all prepared borrowers are quiescent.

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
new call admission while the lock is temporarily released for native control calls.

`configure(count)` is interruptible while waiting for `activeCalls == 0` and no writer. It rejects
a non-positive or over-capacity count, closing/closed state, or closed provider. On the first
transition it queries and validates the positive previous count exactly once. It sets the selected
count, queries it back, and publishes `OPEN_CONFIGURED` only on exact equality. Later cold
reconfiguration uses the same original restore target, waits for all calls, and installs one new
count atomically with respect to coordinated calls and writers.

If installation or verification fails, the writer immediately attempts to restore and verify the
captured original count before clearing writer state. A proved restoration returns to
`OPEN_UNCONFIGURED`; an unproved restoration records a terminal failed state accepted only by
`close()`. Restoration failure is suppressed on the original failure. No prepared route is
constructed from a failed configuration.

Call admission first acquires the fixed shared-budget demand, then waits under the coordinator
lock for no writer. It requires `OPEN_CONFIGURED`, an installed count equal to the plan's selected
count, an open provider, and demand equal to that count; it increments `activeCalls` before
releasing the lock. It does not query the provider per call. The complete input-copy/GEMM/
output-copy sequence executes after admission. `finally` decrements `activeCalls`, signals a
waiting writer/closer, and releases the budget lease.

`close()` is idempotent and uninterruptibly completes once won: it marks `CLOSING`, wakes queued
call admissions so they fail and release permits, waits for the active count and writer to reach
zero, restores and verifies the captured original positive count through the still-open invocation
when configuration was attempted, then invokes the transferred close action. It restores its
caller's interrupt status after cleanup. Provider close failure is suppressed on a primary
restoration failure; state becomes `CLOSED` on every exit. Close invoked from an admitted callback
is rejected to avoid self-deadlock. The fake tests must prove these transitions and failure
precedence rather than use timing-only assertions.

Coordinated exclusion covers only calls and writers routed through this exact coordinator.
Separate engines/coordinators, other Java handles, other class loaders, and arbitrary native code
remain outside it. They must explicitly share the coordinator or retain provider 0003's external
quiescence/exclusion requirement.

### Typed candidates and deterministic cost

Replace the one-value `ThreadConfiguration` with a typed immutable candidate carrying:

- positive `threadCount`;
- complete non-negative OpenBLAS fixed/per-output/per-MAC cost terms for that count; and
- stable input order used only after lower-cost and lower-count tie-breaks.

`OpenBlasRouteConfig` snapshots at most 32 candidates, rejects duplicate counts and null entries,
and retains the current `qualifiedSingleThread(...)` factories by constructing exactly one count-1
candidate. A complete qualified config requires at least one candidate and all existing portable,
representation, threshold, qualification, and expected-storage facts. The list is explicit
heuristic input, not measured evidence or a cache artifact.

Analysis derives its concurrency capacity from the existing positive
`PortableExecutionConfig.availableParallelism` snapshot. It generates one complete OpenBLAS route
candidate per configured candidate whose count is no greater than that capacity, after the exact
0010B semantic/storage/representation eligibility succeeds. For candidate `t`, use its own
OpenBLAS cost triple in the unchanged checked formula:

```text
portable = R * (portableFixed + portablePerOutput * O + portablePerMac * W)

openblas(t) = R * (openblasFixed(t) + openblasPerOutput(t) * O
                   + openblasPerMac(t) * W + complete 0010B representation cost)
```

There is no assumed linear speedup, division by thread count, benchmark lookup, or hidden hardware
model. Each candidate must independently beat portable and both existing thresholds. Overflow,
missing terms, no fitting candidate, equality with portable, or insufficient benefit retains the
portable plan. Among eligible native candidates select lower complete cost, then lower thread
count, then stable configured order. This tie rule conserves permits without claiming performance.

The selected `CpuOpenBlasRoutePlan` records the exact candidate, positive selected count, identical
permit demand, analysis capacity, costs, and unchanged representation/resources. Constructor
invariants require `permitDemand == threadCount <= analysisCapacity`. Candidate generation and
selection occur only in CPU analysis; finalization validates and installs that selected fact and
never searches or substitutes a count.

### Finalization, binding, and execution sequence

For explicitly coordinated construction:

1. Composition creates one `CpuConcurrencyBudget(capacity)`.
2. Composition creates every worker group with that same budget.
3. A loaded discovery session transfers its owned resource into one coordinator using that budget.
4. CPU analysis receives immutable candidate facts and the matching
   `availableParallelism == capacity` snapshot, retains portable, and may select one native plan.
5. CPU finalization validates budget identity/capacity and worker capacity. For a native plan it
   calls `coordinator.configure(selectedThreadCount)` before constructing the recipe.
6. Cold Runtime binding validates buffers, workspaces, copies, provider/coordinator openness, and
   immutable plan agreement, but performs no thread query/set and no permit acquisition.
7. A portable run acquires one inline permit or exact worker participants. A native run admits its
   fixed demand through the coordinator, performs copies/GEMM/copy-out, and releases on every exit.
8. After every prepared use is quiescent, composition closes the coordinator, which restores and
   verifies the original count before closing the transferred provider owner; it then closes the
   worker group.

Independent native calls may overlap only if their fixed demands fit the fair budget and both
plans require the coordinator's currently installed count. For example, capacity eight with count
two admits at most four independent two-permit GEMMs; a fifth waits. A count-four prepared call
cannot run while count two is installed and fails rather than reconfiguring itself. Explicit cold
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
  resources during finalization.
- `internal.route.nativeblas.openblas` — owns provider-state coordination, candidate/plan facts,
  session ownership transfer, and coordinated prepared execution.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget` — one explicit
  route-neutral CPU permit capacity shared by portable orchestration and native calls.
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasCoordinator`
  — OpenBLAS-specific provider lifetime, writer/call exclusion, installation, and restoration.
- Existing `CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate` — nested immutable
  candidate because it is inseparable from the current CPU-private route configuration facts.

Both new top-level types are technically public only where cross-package CPU internals require
access and are unsupported internal API. `CpuCapabilityProvider` remains the sole supported public
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

Expected test and checkpoint paths:

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

Expected documentation and planning paths during implementation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`, only if the documentation pass finds a reusable term changed or introduced
- `docs/planning/backends/cpu/tasks/0010c-coordinated-openblas-thread-candidates-and-shared-cpu-thread-budget.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless an in-scope Javadoc statement is directly stale:

- `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibrary.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecutable.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/BoundInvocation.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunState.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/BackendPartitionAnalysis.java`
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalization.java`

## Maximum scope

This task may create or modify at most 29 paths, all within the categories and exact expected paths
above. It may add exactly the two top-level production types and two matching focused test types.
No provider, shared module, public API, build, architecture, ADR, generated-code/schema, Config,
Engine, other vendor, backend-conformance build, architecture-test, or integration-test path may
change. If another path or top-level type is required, stop and return the task to planning.

## Acceptance criteria

1. One explicit positive-capacity fair budget accounts for coordinated inline portable demand,
   exact worker participants, and selected OpenBLAS demand, with interrupt-safe and failure-safe
   release and no nested acquisition.
2. Worker-group concurrency, deterministic failure selection, cancellation, close behavior, and
   no-worker-submission rule remain unchanged. Native-free tests prove total active portable plus
   native demand never exceeds capacity.
3. A loaded discovery session transfers its owned invocation/close pair exactly once. The
   coordinator becomes the sole close/restoration owner while immutable discovery metadata remains
   readable. Disabled, unavailable, closed, or repeated transfer fails deterministically.
4. The coordinator implements the exact state, fair lock/condition, writer, active-call,
   interruption, failure, reconfiguration, and close protocol above. No query/set/call occurs
   under the Java lock and no thread setter occurs per GEMM.
5. Every successful first configuration captures one positive original count, installs and
   verifies one selected positive count while calls/writers are excluded, and close restores and
   verifies the original through the still-open owner before provider close. Failure cleanup and
   suppression are proved with fakes.
6. Candidate facts are typed, immutable, unique by positive count, bounded to 32, and complete.
   Analysis filters against exact capacity and current 0010B eligibility and generates no native
   candidate from missing, overflowing, unqualified, or ineligible facts.
7. Candidate cost uses the exact per-count terms and unchanged complete representation cost.
   Tests cover multiple counts, over-capacity filtering, all missing terms, overflow, thresholds,
   portable ties, native ties, lower-count tie-break, and stable-order final tie-break without a
   benchmark or assumed scaling formula.
8. The selected plan records one fixed positive thread count and identical permit demand. The
   existing count-one factories and uncoordinated path preserve current behavior, all eight 0010B
   representations, portable completeness, and fail-closed/no-late-fallback rules.
9. Coordinated finalization validates exact budget capacity/identity, installs only the selected
   count, and constructs no recipe after a failed transition. Cold binding performs no thread
   query/set or permit acquisition.
10. Native execution admits once around copy-in/GEMM/copy-out, releases on every outcome, and never
    reconfigures. Independent fake GEMMs overlap only when demands fit and installed counts agree;
    a writer/close waits for admitted calls, and a mismatched prepared count fails.
11. The backend-conformance test remains Engine-free, provider-free, and native-free and exercises
    analysis, assignment, coordinated finalization, cold binding, execution, permit release, and
    restoration with at least one count greater than one while retaining direct and expanded
    representation coverage.
12. The explicit checkpoint uses one caller-supplied exact OpenBLAS 0.3.34 path, verifies positive
    original state, configured counts one and two (or fails if capacity two is unavailable),
    compatible overlapping count-one calls under capacity two, writer exclusion, FLOAT32/FLOAT64
    route results, and restoration in `finally` before provider close. It never discovers,
    substitutes, skips, benchmarks, or qualifies the binary automatically.
13. Public CPU shape, provider API, shared Prepare/Runtime, Planning, Config, Engine, generated
    code/schema, module dependencies, and architecture remain unchanged. Source/API inventory
    confirms no singleton, registry, public generic pool, hot setter/query, or hidden fallback.
14. Every changed Java declaration has meaningful Javadoc covering ownership, lifetime,
    concurrency, parameters, results, nullability, mutation, interruption, failures, and cleanup as
    applicable. A distinct clean documentation-focused agent independently reviews the final diff,
    finalizes Javadocs/package docs/CPU guide/glossary impact/planning evidence, and records
    reasoned no-change conclusions for every review-only and excluded area.
15. Focused tests, final affected-project tests, repository-wide validation, the mandatory native
    checkpoint, CPU Javadoc/render inspection, Markdown/scope/status/order checks, and whitespace
    validation pass before 0010C becomes `Complete`.

## Tests / validation

During implementation, use focused tests as needed. After executable Java and tests stabilize,
run once:

```bash
./gradlew :backends:cpu:test :testing:backend-conformance:test
```

Ordinary tests must be native-free and deterministic. Use latches/barriers and fake invocation
state to prove admission, exclusion, overlap, waiting, release, interruption, reconfiguration, and
restoration; do not rely only on sleeps or wall-clock ordering.

Compile and run the mandatory native checkpoint with the caller-supplied exact 0.3.34 binary:

```bash
./gradlew :backends:cpu:testClasses
java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector \
  -cp backends/cpu/build/classes/java/test:backends/cpu/build/classes/java/main:backends/openblas-provider/build/classes/java/main:modules/model/build/classes/java/main:modules/config/build/classes/java/main:modules/planning/build/classes/java/main:modules/runtime/build/classes/java/main:modules/prepare/build/classes/java/main:modules/backend-contract/build/classes/java/main:modules/trace/build/classes/java/main \
  io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasNativeCheckpoint \
  <ABSOLUTE_OPENBLAS_0_3_34_LIBRARY>
```

The checkpoint must restore in `finally` even when an intermediate assertion fails and report the
verified restored count. Absence of the exact compatible binary leaves the task incomplete.

Because this changes coordinated behavior spanning CPU and backend conformance, the implementation
context then runs one repository-wide gate:

```bash
./gradlew test
```

The final root run may serve as final CPU/conformance evidence if it follows all executable/test
changes. No provider suite, benchmark, tuning workflow, generated evidence regeneration, or
integration test is required. Architecture tests need no change because dependencies and shared
boundaries do not change; the root suite remains the proportional regression gate.

The distinct documentation-focused context reuses successful Java/native evidence unless it
changes executable behavior or tests. After final documentation/Javadoc edits it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git status --short -uall
```

It renders and inspects every changed generated Javadoc page and validates local Markdown links
and anchors, heading order and uniqueness, balanced fences, conceptual-example labels, terminology,
LF/final newlines, trailing whitespace, exact path scope, package/public inventory, and status/order.
It must confirm 0010C is `Complete` only after all evidence, 0010D and 0010E remain master-plan-only
`Draft` rows with no detailed files, and CPU 0011 remains `Blocked`.

## Dependencies

- Complete [CPU 0010B](0010b-bounded-openblas-matmul-representation-expansion.md), including all
  eight representation masks and the current OpenBLAS 0.3.34 checkpoint.
- Complete CPU 0010 and 0010A for the provider-free invocation and transferable internal discovery
  lifetime.
- Complete OpenBLAS provider tasks 0001-0003 for exact loading, GEMM, positive thread query/set,
  caller-owned lifetime, and external shared-state contract.
- Stable `CpuWorkerGroup`, portable selected range counts, staged CPU finalization, immutable
  prepared recipes, cold binding, and Runtime run-state isolation.

All prerequisites are present. Current code supports the bounded CPU-private implementation above
without changing an architecture or shared contract.

## Follow-up tasks

- CPU 0010D remains a master-plan-only `Draft` row and owns automatic installed-binary
  qualification and invalidatable target/binary identity.
- CPU 0010E remains a master-plan-only `Draft` row and owns versioned tuning candidates,
  compatibility signatures, and explicit selected-evidence consumption with measurement/cache
  mutation in `tools/tuning`.
- A future CPU composition/API task may expose a supported facade after Config and Engine contracts
  exist. It must wrap this internal ownership model rather than expose internal packages.
- CPU 0011 remains `Blocked` after 0010E and still requires a concrete Intel use case and supported
  oneMKL ABI evidence.
- Batch, broadcast, epilogue, packing, persistent representations, relaxed math, and other vendor
  routes are not implied follow-ups.

## Architecture impact

Expected architecture impact: None. The task implements concrete-CPU candidate, coordination,
finalization, and execution behavior already assigned to the CPU backend. The provider remains a
leaf, shared Prepare/Runtime contracts remain backend-neutral, Planning sees no route vocabulary,
and Engine remains the future supported composition root.

If implementation reveals that explicit coordination cannot be retained as a CPU-private borrowed
resource, or requires a shared lifecycle API, provider expansion, new dependency, or architecture
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
  the worker group would give an execution primitive provider policy it does not own.
- Count only actual computing participants. The current parallel invoking thread waits and executes
  no range, so charging it would overstate demand. A native demand equals the installed OpenBLAS
  count, including its invoking/native team semantics as one fixed provider demand.
- Acquire worker permits once before queue publication. Worker-side acquisition could deadlock when
  a job owns some permits while queued workers wait for the remainder.
- Keep copies inside native admission. They are part of the complete selected plan, and excluding
  them would permit extra portable work beyond the explicit CPU budget during the native recipe.
- Accept explicit per-count heuristic cost triples. Inferring linear speedup from count would be an
  unsupported performance claim; requiring measured evidence belongs to 0010E.
- Permit cold reconfiguration but never automatic per-call reconfiguration. Prepared plans remain
  fixed; a mismatched installed count fails until composition explicitly transitions while calls
  are quiescent.
- Preserve the old uncoordinated count-one path as a compatibility boundary. Its limitations stay
  explicit, so this task does not pretend that internal coordination controls external consumers.

## Known limitations

- Coordination covers only work explicitly sharing the exact budget and coordinator object.
- There is no supported public Config or Engine composition surface yet; the implementation remains
  CPU-internal and is exercised through focused/conformance/checkpoint paths.
- Heuristic per-count costs are caller-supplied immutable facts, not benchmark evidence. Count one
  remains the compatibility default.
- A prepared plan is usable only while its selected count is installed. Reconfiguration can make an
  older plan temporarily ineligible; execution fails rather than changing provider state itself.
- Fair semaphore/lock admission prevents barging but does not promise operating-system scheduling
  order or performance proportional to permits.
- Arbitrary external OpenBLAS users can still race this coordinator and invalidate its guarantees;
  they require external exclusion or explicit adoption of the same coordinator.

## Validation evidence

Empty until implemented. Record exact implementation and documentation context IDs, commands,
test counts/results, native binary/path/version evidence, restored count, reused evidence, Javadoc
inspection, Markdown checks, scope/status checks, and reasoned no-change conclusions.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented. Do not mark `Complete` before the distinct implementation and
documentation-focused contexts and every required validation gate finish.
