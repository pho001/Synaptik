# Task 0010: Prepared-Handle [Ownership](../../../../glossary.md#term-ownership-meanings) and Closure

## Status

Complete

## Goal

Complete [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0013's outward [lifecycle](../../../../glossary.md#lifecycle) by making each ordinary and advanced [Engine](../../../../glossary.md#term-engine) prepared
handle the explicit closeable [owner](../../../../glossary.md#term-owner) of its one inward [Runtime](../../../../glossary.md#term-runtime) [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution). Close every
temporary one-shot, representative trial, correctness, measurement, loser, fallback, selected,
and failed-[publication](../../../../glossary.md#term-publication) preparation exactly once, and make `AdvancedEngine.close()` close every
retained preparation before [backend](../../../../glossary.md#backend) integration shutdown.

```text
Prepare transactional handoff
  -> one Runtime PreparedExecution owns identity-unique PreparedResource values
  -> one Engine prepared handle owns that Runtime PreparedExecution
  -> explicit handle close or Engine close rejects later runs
  -> an already leased synchronous run may finish
  -> persistent resources close after the last Runtime lease
  -> Engine closes results, then retained preparations, then backend composition
```

The task changes no Runtime [lease](../../../../glossary.md#term-lease) algorithm, [Prepare](../../../../glossary.md#prepare) [transaction](../../../../glossary.md#term-transaction), backend resource contract,
backend selection, or hot-path [graph](../../../../glossary.md#graph) representation. It completes the currently missing Engine
owner and cleanup layer over the already completed Runtime 0016 and Prepare 0006 contracts.

## Pre-implementation source-backed [lifecycle](../../../../glossary.md#lifecycle) audit

This audit records the lifecycle gaps observed before task 0010 implementation. Later sections
describe the completed behavior and final [validation](../../../../glossary.md#term-validation).

1. [Runtime](../../../../glossary.md#term-runtime) [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) is already the unique inward [owner](../../../../glossary.md#term-owner) of persistent
   [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource) values. Its close transition atomically rejects new run leases, does not
   wait, and defers physical resource release until the last acquired run lease is returned.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [leases glossary](../../../../glossary.md#term-lease); [release glossary](../../../../glossary.md#term-release))</span>
2. [Prepare](../../../../glossary.md#prepare) [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) already performs transactional finalization: it rolls back every
   successfully acquired identity-unique resource on a later failure and transfers successful
   ownership exactly once to Runtime. Engine must not duplicate or bypass that transaction.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Engine glossary](../../../../glossary.md#term-engine); [transaction glossary](../../../../glossary.md#term-transaction))</span>
3. `PreparedExecution` and `AdvancedPreparedExecution` retained inward Runtime prepared
   executions but exposed no close operation. <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>`AdvancedEngine` tracked open run results only, so
   a successful prepared recipe remained live until backend composition shutdown.<span data-glossary-navigation="true"> ([backend composition glossary](../../../../glossary.md#backend-composition))</span>
4. Ordinary `compute(...)` and `backward(...)` one-shot paths freshly prepared and closed their
   run result, but did not close the temporary prepared recipe on success or failure.<span data-glossary-navigation="true"> ([run result glossary](../../../../glossary.md#run-result--runresult))</span>
5. `RepresentativeExecutionSession` received a fresh inward preparation for each trial,
   correctness capture/comparison, warmup, and timed sample. It closed the result but left each
   preparation live. Those trial preparations were never production handles.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
6. Tuning created a fresh selected or safe-fallback preparation only after representative
   cleanup. That preparation became the one returned ordinary handle. Every non-selected trial
   remained temporary; no trial or loser could be promoted to production.
7. Existing final-publication logic prevented a newly completed public value from escaping after
   Engine close won, but it did not close a freshly prepared inward execution or wrapper when
   publication failed. The prepared-specific completion path therefore needed to roll it back.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
8. `ModelAutotuningPreparation` contains one ordinary `PreparedExecution`. It is result metadata
   around that sole outward owner, not a second owner and not another closeable facade.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [facade glossary](../../../../glossary.md#term-facade))</span>
9. Existing Engine admission makes shutdown wait for admitted synchronous compile, prepare, run,
   materialization, one-shot, and tuning operations. Existing Runtime run leases decide the
   direct prepared-handle close-versus-run race; Engine needs no second run-lease mechanism.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [synchronous glossary](../../../../glossary.md#term-synchronous); [compile glossary](../../../../glossary.md#compile); [materialization glossary](../../../../glossary.md#materialization); [race glossary](../../../../glossary.md#term-race))</span>
10. `AdvancedRunResult` already supplies the Engine-local close coordination pattern: one caller
    performs cleanup, concurrent close callers wait uninterruptibly and restore interruption,
    unregistering occurs in `finally`, and a cleanup failure is retained and rethrown. Prepared
    wrappers should use the same observable close discipline while delegating physical lifetime
    to Runtime.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime))</span>

## [Scope](../../../../glossary.md#term-scope)

- Make public [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) and `AdvancedPreparedExecution` implement `AutoCloseable`.
  Each remains owner-bound and owns exactly one inward Runtime prepared execution.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- Add public `boolean isClosed()` and public `void close()` to both prepared handles. The close
  transition is idempotent in effects, thread-safe, and visible immediately; <span data-glossary-navigation="true"> ([idempotent glossary](../../../../glossary.md#term-idempotent); [thread-safe glossary](../../../../glossary.md#term-thread-safe))</span>`isClosed()` means
  that the Engine-facing handle rejects new work, not that deferred Runtime resource release has
  necessarily completed.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release))</span>
- Preserve ordinary `PreparedExecution.compiledGraph()` and the advanced handle's otherwise
  opaque surface. Add no public delegate, resource, backend, recipe, or owner accessor.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [backend glossary](../../../../glossary.md#backend))</span>
- Track every successfully published ordinary or advanced prepared handle in its owning
  `AdvancedEngine`, in successful-publication order. Explicit handle close unregisters it even
  when inward cleanup throws.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- At Engine close, after admitted operations quiesce, close open run results in reverse
  successful-run order, then retained prepared handles in reverse successful-prepare order, then
  close backend composition. Attempt every cleanup and preserve the failure rules below.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [prepare glossary](../../../../glossary.md#prepare); [backend composition glossary](../../../../glossary.md#backend-composition))</span>
- Introduce prepared-specific final publication/rollback. If Engine close wins publication, or
  wrapper/result construction or evidence translation fails after inward preparation succeeds,
  close the newly created inward execution or owner-bound wrapper exactly once before propagating.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- In ordinary one-shot compute and backward paths, close the temporary result first and the
  temporary ordinary prepared handle second on success and every failure. The preparation must
  remain owned by the operation until both cleanup stages finish; it is never registered as a
  retained public handle.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Make `RepresentativeExecutionSession` own each fresh inward trial preparation passed to its
  execution, correctness-reference, or correctness-comparison action. Close the run result first
  and that trial preparation second before the action returns, including mismatch, copy,
  materialization, execution, and cleanup failures.<span data-glossary-navigation="true"> ([run result glossary](../../../../glossary.md#run-result--runresult); [materialization glossary](../../../../glossary.md#materialization))</span>
- Cover every Phase-1 and Phase-2 tuning lifecycle: candidate/trial, correctness reference,
  correctness comparison, warmup, timed sample, losing candidate, authenticated winner, safe
  fallback, selected/fallback construction failure, session poisoning, and close-race rollback.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [race glossary](../../../../glossary.md#term-race))</span>
- Retain only the one freshly prepared authenticated winner or one freshly prepared safe fallback
  inside the returned ordinary handle. Close it by explicit handle close or later Engine close.
- Keep `ModelAutotuningPreparation` non-`AutoCloseable`; document that callers close its exact
  `preparedExecution()` handle and that [Engine](../../../../glossary.md#term-engine) remains the final retained-[owner](../../../../glossary.md#term-owner) safety boundary.
- Complete a distinct clean documentation-focused pass in the same implementation change after
  executable behavior and tests stabilize.

## Public compatibility decision

- This is an additive source/API change to both existing final handle classes: no constructor,
  method, record component, result type, or existing signature is removed or reinterpreted.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [signature glossary](../../../../glossary.md#term-signature))</span>
- Implementing `AutoCloseable` changes class metadata intentionally and adds the exact two public
  methods on each handle. <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>`close()` declares no checked exception.
- Handle object identity, owner validation, compiled-handle association, and existing equality
  behavior remain unchanged.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [validation glossary](../../../../glossary.md#term-validation))</span>
- A closed handle is terminal. It cannot be reopened, republished, or transferred to another
  Engine. Run through that handle fails with the existing closed-preparation state failure when
  Runtime cannot acquire a lease.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- Closing a preparation after a [synchronous](../../../../glossary.md#term-synchronous) [run](../../../../glossary.md#term-run) returns does not invalidate a [detached](../../../../glossary.md#detached-model-results)
  [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue) or an open [`RunResult`](../../../../glossary.md#run-result--runresult); the result owns its separate [Runtime](../../../../glossary.md#term-runtime) result [lease](../../../../glossary.md#term-lease).
- `ModelAutotuningPreparation` remains the same public result and gains no close method or
  duplicate lifecycle state.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## [Ownership](../../../../glossary.md#term-ownership-meanings) state machine

```text
fresh inward Runtime preparation
  -> temporary Engine operation owner
       -> result cleanup -> preparation close -> terminal
  -> public wrapper construction
       -> publication failure -> wrapper/inward close -> terminal
       -> OPEN + registered with owning AdvancedEngine
            -> explicit handle close claims cleanup
            -> Engine close claims cleanup
                 -> CLOSED + unregistered
                 -> inward Runtime close called exactly once
                 -> retained immediate cleanup failure rethrown to close callers

Runtime delegate beneath CLOSED
  no active run lease  -> persistent resources close during Runtime close
  active run lease     -> Runtime close returns without waiting; last lease releases resources
```

Exactly one [Engine](../../../../glossary.md#term-engine) caller claims wrapper cleanup. Concurrent or repeated close callers wait for
that claim to finish, restore interruption after uninterruptible waiting, and observe the same
retained immediate cleanup failure when one occurred. If [Runtime](../../../../glossary.md#term-runtime) deferred resource [release](../../../../glossary.md#term-release) to an
already active [run](../../../../glossary.md#term-run), any release failure is reported by that run's [lease](../../../../glossary.md#term-lease) release under Runtime's
existing contract; the already completed Engine-wrapper close does not invent or replay it.

## Failure and suppression precedence

1. Preserve the primary work failure. Cleanup failures are suppressed in cleanup encounter order
   when they are not the exact primary object; never self-suppress that primary instance.
2. If work succeeds and cleanup fails, the first cleanup failure becomes primary. Attempt all
   remaining required cleanup and suppress later distinct failures in encounter order.
3. One-shot cleanup order is run result, then temporary prepared handle. Trial action cleanup
   order is run result, then the exact trial preparation, followed by existing session-owned
   borrowed input cleanup when the session itself terminates.<span data-glossary-navigation="true"> ([run result glossary](../../../../glossary.md#run-result--runresult); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
4. Representative-session aggregation additionally preserves its current identity de-duplication:
   do not add the exact same object twice if it is already the primary or already suppressed.
   Other Engine cleanup helpers preserve their current per-callback encounter semantics rather
   than introducing a new global throwable registry.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [Engine glossary](../../../../glossary.md#term-engine); [callback glossary](../../../../glossary.md#term-callback); [registry glossary](../../../../glossary.md#term-registry))</span>
5. An execution, copy, result-count, materialization, result-close, or trial-preparation-close
   failure inside <span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>`RepresentativeExecutionSession` is an execution-path failure: preserve its
   exact identity, poison the session under the existing rules, and forbid safe fallback.
6. Failure before a trial enters session execution retains its existing recoverable/non-
   recoverable classification. Closing a partially created trial during rollback must not turn a
   non-fallback path into fallback or vice versa.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
7. After selected/fallback inward preparation succeeds, evidence translation, public-result or
   wrapper construction, and final-publication failure remain primary; preparation rollback
   failure is suppressed when distinct. If rollback is the only failure, propagate it.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
8. Explicit prepared-handle close unregisters in `finally`. An immediate inward close failure is
   retained and rethrown unchanged to concurrent and repeated close callers; Engine close does
   not retry a wrapper whose close was already claimed.
9. Engine shutdown attempts open results in reverse successful-run order, retained preparations
   in reverse successful-publication order, and backend composition last. The first failure is
   primary; each later callback failure other than that exact primary is suppressed in encounter
   order, matching the existing Engine shutdown helper.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [backend composition glossary](../../../../glossary.md#backend-composition))</span>
10. Preserve `RuntimeException` and `Error` identity. Do not introduce checked wrapping, broad
   catch-and-ignore behavior, logging-only cleanup, or a second failure aggregate.

## [Concurrency](../../../../glossary.md#term-concurrency) and [admission](../../../../glossary.md#term-admission) behavior

- Existing Engine admission remains the outer gate. A call admitted before Engine close finishes
  all synchronous work and its temporary cleanup before admission is released. A call arriving
  after close begins fails under the existing Engine-closed rule.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [synchronous glossary](../../../../glossary.md#term-synchronous))</span>
- Engine close waits for the existing active-operation count to reach zero. It then closes run
  results, prepared handles, and backend composition; no persistent backend resource callback can
  race a still-admitted Engine operation at this shutdown boundary.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [run glossary](../../../../glossary.md#term-run); [backend composition glossary](../../../../glossary.md#backend-composition); [callback glossary](../../../../glossary.md#term-callback))</span>
- A direct prepared-handle close may [race](../../../../glossary.md#term-race) `run(...)`. If Runtime acquires the prepared lease
  first, close is non-waiting and the run finishes before the last lease releases resources. If
  close wins first, lease acquisition fails and no run begins.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Result materialization continues to synchronize against result/Engine close through the
  existing result lease. Prepared-handle close adds no materialization lock and does not revoke
  an already returned result.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- A public prepare completion racing Engine close either publishes and registers exactly one
  handle before close snapshots it, or rolls back the unpublished preparation. No handle or
  persistent resource may fall between those outcomes.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>
- Add no blocking wait to [Runtime](../../../../glossary.md#term-runtime) close, [asynchronous](../../../../glossary.md#term-asynchronous) [API](../../../../glossary.md#term-application-programming-interface-api), [cancellation](../../../../glossary.md#term-cancellation), timeout, finalizer,
  `Cleaner`, global [registry](../../../../glossary.md#term-registry), [service locator](../../../../glossary.md#term-service-locator), or [backend](../../../../glossary.md#backend)-global [lifecycle](../../../../glossary.md#lifecycle) state.

## Out of [scope](../../../../glossary.md#term-scope)

- changing [Runtime](../../../../glossary.md#term-runtime) [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner), [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource), lease
  acquisition/release, or cleanup algorithms<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [release glossary](../../../../glossary.md#term-release))</span>
- changing [Prepare](../../../../glossary.md#prepare) [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), finalizer/result contracts, transactional handoff, or
  resource identity rules<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- concrete CPU/Metal/CUDA production code, backend resource implementation, capability,
  selection, compilation, execution, or shutdown behavior<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [backend glossary](../../../../glossary.md#backend); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- making compiled handles, [run](../../../../glossary.md#term-run) results, [host](../../../../glossary.md#term-host) values, requests, evidence, or
  `ModelAutotuningPreparation` new [owners](../../../../glossary.md#term-owner) of prepared resources
- retaining or promoting a tuning trial, caching a prepared execution, cross-Engine transfer,
  shared prepared ownership, reference counting in Engine, or persistent executable reuse<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [Engine glossary](../../../../glossary.md#term-engine); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [operation](../../../../glossary.md#operation), `CompiledNode`, graph, Tensor, backend storage, representation, schedule, slot, or
  resource exposure on the Runtime hot path or public prepared handles<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [Tensor glossary](../../../../glossary.md#tensor); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- dependency, Gradle, package, build, architecture-rule, ADR, backend-conformance, serialization,
  tracing, Training, NN, or Tensor changes<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [serialization glossary](../../../../glossary.md#term-serialization); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Engine, Prepare, Runtime,
  concrete-backend, ownership, lifecycle, and dependency-direction rules<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span> and [roadmap](../../../roadmap.md)
- [ADR 0013: Prepared execution persistent-resource lifecycle](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- [Metal backend strategy](../../../../design/notes/metal-backend-strategy.md)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- [Engine master plan](../master-plan.md) and [Engine](../../../../glossary.md#term-engine)
  [0001](0001-advanced-composition-and-representation-level-lifecycle-foundation.md),
  [0002](0002-standard-built-in-composition.md),
  [0003](0003-typed-logical-input-binding-and-published-result-access.md),
  [0005](0005-one-shot-forward-convenience.md),
  [0006A](0006a-representative-tuning-execution-and-safe-fallback.md),
  [0007](0007-optional-model-autotuning-composition.md),
  [0008A](0008a-representative-complete-plan-correctness-oracle.md), and
  [0009](0009-public-complete-plan-autotuning-composition.md)
- [Runtime master plan](../../runtime/master-plan.md) and
  [Runtime 0016](../../runtime/tasks/0016-persistent-prepared-resource-lifecycle.md)
- [Prepare master plan](../../prepare/master-plan.md) and
  [Prepare 0006](../../prepare/tasks/0006-persistent-prepared-resource-finalization-transaction.md)
- [Metal master plan](../../../backends/metal/master-plan.md)

## Architecture constraints

- Engine remains the outer composition root and the only owner of outward handle registration,
  temporary execution cleanup, public lifecycle admission, and backend composition shutdown.<span data-glossary-navigation="true"> ([composition root glossary](../../../../glossary.md#term-composition-root); [lifecycle glossary](../../../../glossary.md#lifecycle); [admission glossary](../../../../glossary.md#term-admission); [backend composition glossary](../../../../glossary.md#backend-composition))</span>
- [Runtime](../../../../glossary.md#term-runtime) [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) remains the unique [owner](../../../../glossary.md#term-owner) of persistent [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource) values
  and the sole authority for close/run lease arbitration.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [lease glossary](../../../../glossary.md#term-lease))</span>
- Prepare remains the transactional transfer boundary. Engine accepts one completed Runtime
  owner and never enumerates, deduplicates, releases, or interprets its resource aggregate.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Concrete [backends](../../../../glossary.md#backend) do not depend on [Engine](../../../../glossary.md#term-engine). Runtime does not depend on a [concrete backend](../../../../glossary.md#concrete-backend).
- The Runtime [hot path](../../../../glossary.md#term-hot-path) gains no [`Operation`](../../../../glossary.md#operation), `CompiledNode`, Tensor graph, backend-selection,
  tuning, reflection, string dispatch, map lookup, or extra per-element/per-node work.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [dispatch glossary](../../../../glossary.md#term-dispatch); [lookup glossary](../../../../glossary.md#term-lookup); [element glossary](../../../../glossary.md#term-element); [node glossary](../../../../glossary.md#node))</span>
- Engine lifecycle registries are per-Engine private state, not static/global registries or
  service locators. They retain only outward wrappers needed for deterministic shutdown.
- If implementation requires changing a Runtime or Prepare contract, adding a reverse dependency,
  exposing an inward type publicly, waiting in Runtime close, or changing backend shutdown
  authority, stop and return the task to planning.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [planning glossary](../../../../glossary.md#term-planning))</span>

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.engine` — owns both public closeable prepared handles, per-Engine
  retention, temporary/session cleanup, and prepared-specific final publication.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>

No package, [module](../../../../glossary.md#term-module), or [dependency](../../../../glossary.md#term-dependency) is added. Inward [Runtime](../../../../glossary.md#term-runtime) types remain behind [Engine](../../../../glossary.md#term-engine) wrappers.

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths:

- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/PreparedExecution.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedPreparedExecution.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSession.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/ModelAutotuningPreparation.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/EngineBackendComposition.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/package-info.java`

Expected test paths:

- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/AdvancedEngineLifecycleTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineTypedLifecycleTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSessionTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/AdvancedEnginePublicShapeTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/EngineTypedPublicShapeTest.java`

Expected explanatory documentation paths:

- `docs/api/public-api.md`
- `docs/api/runtime-api.md`
- `docs/architecture/lifecycle.md`
- `docs/backend-guide/writing-a-backend.md`
- `docs/glossary.md`

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task specification
- `docs/planning/modules/engine/master-plan.md`
- `docs/planning/roadmap.md`
- `docs/planning/backends/metal/master-plan.md` — planning-only lifecycle-prerequisite correction
  while retaining Metal 0002 as Blocked<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

Review without modification unless contradicted: `ARCHITECTURE.md`, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0013, other architecture
pages, [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) source and tests, [Gradle](../../../../glossary.md#term-gradle) files, architecture-test source,
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) source, integration-test source, and unrelated documentation or plans.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 22 paths: 8 [Engine](../../../../glossary.md#term-engine) production/[Javadoc](../../../../glossary.md#term-javadoc) paths, 5 Engine test paths, 5 focused explanatory
documentation paths, and exactly 4 [planning](../../../../glossary.md#term-planning) paths. The count exceeds the normal planning
guardrail because one indivisible [lifecycle](../../../../glossary.md#lifecycle) protocol crosses both public prepared facades, both
one-shot paths, the shared representative session, tuned/fallback [publication](../../../../glossary.md#term-publication), Engine shutdown,
their existing focused tests, and the same-change documentation required for the public [API](../../../../glossary.md#term-application-programming-interface-api).
Splitting any one category would temporarily leave a current Engine preparation path leaking or
would publish a close contract without complete [ownership](../../../../glossary.md#term-ownership-meanings) behavior. If another production, test,
documentation, build, or planning path is required, stop and revise this task and master plan
before implementation rather than silently expanding it.

## Acceptance criteria

1. Ordinary and advanced prepared handles implement `AutoCloseable`, expose only the specified
   additive `isClosed()` and no-checked-exception `close()` lifecycle, and preserve every existing
   public signature and opacity boundary.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [signature glossary](../../../../glossary.md#term-signature))</span>
2. Each public handle owns one inward Runtime preparation. Exactly one explicit handle close or
   Engine close claims wrapper cleanup, unregisters it, and calls inward close exactly once.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine))</span>
3. Concurrent/repeated prepared-handle closes follow the specified waiting, interruption,
   retained-failure, and no-retry behavior. `isClosed()` becomes true at the terminal outward
   transition without claiming that deferred Runtime resource release has completed.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release))</span>
4. Engine close waits for admitted operations, closes results in reverse run-publication order,
   prepared handles in reverse prepare-publication order, and backend composition last. It
   attempts all cleanup and preserves deterministic primary/suppressed throwable identity.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication); [prepare glossary](../../../../glossary.md#prepare); [backend composition glossary](../../../../glossary.md#backend-composition); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
5. The run-versus-handle-close race delegates to Runtime leases: a leased run completes and owns
   deferred release; a close-first run is rejected. Engine adds no second lease or waiting close.<span data-glossary-navigation="true"> ([race glossary](../../../../glossary.md#term-race); [leases glossary](../../../../glossary.md#term-lease))</span>
6. Every ordinary compute/backward one-shot preparation closes after its result on success and
   every compile/bind/run/materialization/cleanup failure, with exact-once effects and specified
   failure precedence.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [materialization glossary](../../../../glossary.md#materialization))</span>
7. Every Phase-1/Phase-2 trial, correctness, warmup, timed, loser, mismatch, and poisoned-session
   preparation closes after its result before the action returns. No trial becomes production.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup))</span>
8. The freshly prepared authenticated winner or safe fallback is the sole retained preparation
   in the returned handle. Explicit handle close or Engine close releases it; evidence and<span data-glossary-navigation="true"> ([releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
   `ModelAutotuningPreparation` do not duplicate [ownership](../../../../glossary.md#term-ownership-meanings).
9. Construction, evidence translation, selected/fallback preparation, final publication, and
   Engine-close races leave either one registered outward owner or one fully rolled-back
   preparation, never an escaped closed handle or unowned inward preparation.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
10. Focused fake-resource tests prove exact close counts/order, reverse retained order, result-
    before-preparation-before-composition shutdown, all success/failure paths, self-suppression
    avoidance, `RuntimeException`/`Error` identity, repeated failure observation, and race
    outcomes with deterministic latches.
11. Current standard CPU integration tests continue to pass without CPU production changes.
    Resource-free current CPU behavior is compatibility evidence, not proof substituted for the
    fake persistent-resource lifecycle tests.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
12. Javadocs document owner, terminal state, nullability, failure, concurrency, result
    independence, shutdown, and caller responsibilities. Focused docs distinguish Engine handle
    ownership from Runtime resource ownership and Prepare transfer; glossary impact is finalized.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
13. No finalizer, `Cleaner`, global registry, service locator, cache, reference count, new public
    facade, Runtime/Prepare/backend change, dependency/build change, architecture-test change,
    backend-conformance change, or integration-test source change is introduced.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [cache glossary](../../../../glossary.md#term-cache); [facade glossary](../../../../glossary.md#term-facade); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance))</span>
14. Focused Engine tests, existing focused integration tests, Engine Javadoc, Markdown/manual
    links, public-shape/import/scope checks, and diff validation pass within the exact 22-path
    ceiling.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [shape glossary](../../../../glossary.md#shape); [scope glossary](../../../../glossary.md#term-scope); [validation glossary](../../../../glossary.md#term-validation))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) the focused [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle) and public-[shape](../../../../glossary.md#shape) tests, then the full
affected [module](../../../../glossary.md#term-module):

```bash
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.AdvancedEngineLifecycleTest \
  --tests io.github.pho001.synaptik.engine.EngineTypedLifecycleTest \
  --tests io.github.pho001.synaptik.engine.RepresentativeExecutionSessionTest \
  --tests io.github.pho001.synaptik.engine.api.AdvancedEnginePublicShapeTest \
  --tests io.github.pho001.synaptik.engine.api.EngineTypedPublicShapeTest
./gradlew :modules:engine:test
```

Run the existing public [CPU](../../../../glossary.md#term-cpu) compatibility fixtures without changing them:

```bash
./gradlew :testing:integration-tests:test \
  --tests io.github.pho001.synaptik.testing.integration.EngineAdvancedLifecycleIntegrationTest \
  --tests io.github.pho001.synaptik.testing.integration.EngineTypedLifecycleIntegrationTest \
  --tests io.github.pho001.synaptik.testing.integration.EngineModelAutotuningIntegrationTest
```

This is a normal single-module lifecycle task with unchanged dependencies, architecture rules,
build, and [backend](../../../../glossary.md#backend) contracts. Repository-wide, architecture-test, backend-[conformance](../../../../glossary.md#term-conformance), [Runtime](../../../../glossary.md#term-runtime),
[Prepare](../../../../glossary.md#prepare), and CPU suite reruns are not required absent a concrete failure; the next [Metal](../../../../glossary.md#term-metal)
capability [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci) owns broader validation.

After executable Java stabilizes, the mandatory clean documentation context reuses successful
Java evidence unless it changes executable behavior, then runs:

```bash
./gradlew :modules:engine:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
git diff --cached --check
git status --short -uall
```

If the repository Markdown helper is absent, manually validate every changed local [target](../../../../glossary.md#term-target) and
anchor, heading uniqueness, balanced fences, LF/final newlines, and trailing whitespace and
record that substitution. Render and inspect both prepared-handle pages, both Engine pages,
`ModelAutotuningPreparation`, and the package page. Inspect compiled public shape, forbidden
imports, exact changed paths, no Java/build file outside the authorized list, no staged files,
and synchronized task/master-plan/roadmap/Metal-blocked status.

## Dependencies

- Runtime 0016 — Complete; supplies the unique prepared-resource owner and non-waiting close/run
  lease.<span data-glossary-navigation="true"> ([resource owner glossary](../../../../glossary.md#ownership-borrowing-and-lease); [run glossary](../../../../glossary.md#term-run); [lease glossary](../../../../glossary.md#term-lease))</span>
- Prepare 0006 — Complete; supplies transactional persistent-resource acquisition, rollback, and
  ownership transfer to Runtime.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- Engine 0001–0009 — Complete; supply every ordinary, advanced, one-shot, representative,
  correctness, tuning, fallback, publication, and retained-result lifecycle covered here.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [publication glossary](../../../../glossary.md#term-publication); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0013 — Accepted; already resolves the [ownership](../../../../glossary.md#term-ownership-meanings) and close semantics.

No new architecture decision, [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare) change, [backend](../../../../glossary.md#backend) resource implementation, [module](../../../../glossary.md#term-module)
[dependency](../../../../glossary.md#term-dependency), or build prerequisite is required.

## Follow-up tasks

- [Metal 0002](../../../backends/metal/master-plan.md)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span> remains `Blocked` after this task completes;
  lifecycle readiness does not make a reusable MPSGraph executable current capability.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- A fresh Metal audit may detail only the first truthful MPSGraph
  prepared-execution route against the completed outward lifecycle. Do not create that task here.
- No further [Engine](../../../../glossary.md#term-engine) task is made [Ready](../../../../glossary.md#term-ready) by this [planning](../../../../glossary.md#term-planning) change.

## Architecture impact

Expected impact: None.

This task implements the [ownership](../../../../glossary.md#term-ownership-meanings) chain already required by `ARCHITECTURE.md` and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0013:
[Prepare](../../../../glossary.md#prepare) transfers to [Runtime](../../../../glossary.md#term-runtime), Runtime uniquely owns persistent resources, and [Engine](../../../../glossary.md#term-engine) owns outward
public handles and composition shutdown. It changes neither authority nor [dependency direction](../../../../glossary.md#java-platform-and-build-tools).
No architecture-contract, ADR, architecture-test, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), or integration-test source
update is required. If implementation discovers that any such contract must change, stop and
return to [planning](../../../../glossary.md#term-planning) instead of inventing a design.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik Engine task 0010. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/engine/tasks/0010-prepared-handle-ownership-and-closure.md in full, then read
the task's directly required architecture, planning, Runtime, Prepare, Engine, and documentation
references. Inspect current source/tests; do not rely on remembered context.

Implement task 0010 exactly within its 22-path ceiling. Stop for any architecture, dependency,
inward-contract, public-surface, or scope conflict. After executable validation, hand the exact
diff and evidence to a distinct clean documentation-focused context in the same overall change,
following docs/developer-guide/documentation-rules.md. Do not mark 0010 Complete until that pass
and every specified gate succeed. Keep Metal 0002 Blocked until then.
```

## Local decisions

- Use the existing [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle) [owner](../../../../glossary.md#term-owner) and `AdvancedRunResult` close coordination as the
  design oracle for outward wrapper state; do not create a generic lifecycle framework.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Keep Runtime as physical-resource authority. Engine wrapper state exists only for public
  rejection, deterministic close claiming, failure replay, and owner registration.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Register only preparations that are successfully published as public handles. Temporary
  one-shot and representative preparations remain lexical operation/session owners.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Keep `ModelAutotuningPreparation` non-closeable so there is exactly one outward owner: its
  contained ordinary prepared handle.
- Keep run results independent after synchronous run completion. Result-before-preparation order
  at Engine shutdown is deterministic ownership hygiene, not a new claim that result storage is a
  persistent prepared resource.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [synchronous glossary](../../../../glossary.md#term-synchronous); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepared resource glossary](../../../../glossary.md#prepared-resource--preparedresource))</span>
- One cohesive task is required because every current preparation path must obey the public close
  contract before Metal can rely on it; no lifecycle path is deferred silently.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>

## Known limitations

- Current CPU preparation has no persistent backend resource requiring visible physical release;
  fake Engine compositions provide deterministic resource and failure evidence until Metal 0002.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [release glossary](../../../../glossary.md#term-release); [Engine glossary](../../../../glossary.md#term-engine); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Close is synchronous only for immediately releasable resources. It intentionally does not wait
  for an already leased run, expose completion, or support cancellation.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [run glossary](../../../../glossary.md#term-run); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- Engine retains published prepared handles until explicit handle close or Engine close. It adds
  no weak ownership, leak detector, finalizer, <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`Cleaner`, or process-global recovery.
- The current supported composition remains CPU-only and one non-empty maximal partition; this
  task adds no mixed-backend execution or reusable native executable.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- An earlier clean implementation correction checkpoint ran the exact focused
  lifecycle/public-shape command from this task with 82 focused tests and no failures; full<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [lifecycle glossary](../../../../glossary.md#lifecycle); [shape glossary](../../../../glossary.md#shape))</span>
  `./gradlew :modules:engine:test` then passed with 93 tests and no failures. The final follow-up
  verification, after seven tests were added to close the ordinary [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution)
  race/interruption and <span data-glossary-navigation="true"> ([race glossary](../../../../glossary.md#term-race))</span>`RepresentativeExecutionSession` early-rejection cleanup evidence, passed
  the focused five-class command with 89 of 89 tests and the full Engine suite with 100 of 100
  tests. Both final runs had no failures, errors, or skips.
- The exact three-[fixture](../../../../glossary.md#term-fixture) [CPU](../../../../glossary.md#term-cpu) compatibility command from this task passed with no failures:
  `EngineAdvancedLifecycleIntegrationTest`, `EngineTypedLifecycleIntegrationTest`, and
  `EngineModelAutotuningIntegrationTest`; it recorded 10 tests and one environment-dependent
  skip.
- The mandatory fresh documentation-focused correction context independently inspected the
  corrected implementation, focused tests, Runtime owner/lease contracts, Prepare transaction,
  affected APIs, documentation, glossary, and planning state. Its environment exposed no clean
  context UUID. It changed documentation and Javadoc comments only; no executable Java behavior
  changed after the implementation-owned Java evidence above, so those suites were not repeated.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [owner glossary](../../../../glossary.md#term-owner); [lease glossary](../../../../glossary.md#term-lease); [Prepare glossary](../../../../glossary.md#prepare); [transaction glossary](../../../../glossary.md#term-transaction); [planning glossary](../../../../glossary.md#term-planning))</span>
- Final `./gradlew :modules:engine:javadoc` passed after [Javadoc](../../../../glossary.md#term-javadoc) finalization. Generated pages for
  [`Engine`](../../../../glossary.md#term-engine), `AdvancedEngine`, both prepared handles, `ModelAutotuningPreparation`, and the Engine
  package were present and inspected for the finalized ownership, close/run race, selected-owner,
  and shutdown language.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [run glossary](../../../../glossary.md#term-run))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed for the five changed explanatory documents
  and four planning documents: 9 Markdown files validated, including local targets, anchors,
  heading uniqueness, fences, final newlines, and whitespace.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- `javap` confirmed that both final prepared handle classes implement `AutoCloseable`, expose
  only public `isClosed()` and no-checked-exception `close()` additions, and keep inward Runtime
  access package-private. The focused reflection tests independently passed the same public-shape
  boundary. A focused import scan found no new public inward exposure.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Final `git diff --check`, `git diff --cached --check`, exact allowlist/status inspection, and
  generated-artifact checks passed. The unstaged worktree contains exactly 22 authorized paths:
  eight Engine production/Javadoc paths, five Engine test paths, five explanatory documentation
  paths, and three tracked plus one new planning path. There are no staged files, tracked
  generated artifacts, or Java/build paths outside the allowlist.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Reasoned no-change conclusions: `ARCHITECTURE.md` and ADR 0013 already require the implemented
  ownership chain, so no authority change was needed. Prepare API/docs and Runtime executable code
  already implement the transactional transfer and unique lease-aware owner; Compile, Tensor,
  Training, CPU/Metal production, Gradle/dependencies, architecture tests, backend conformance,
  and integration source neither expose nor own this outward Engine wrapper lifecycle. Existing
  integration source was executed unchanged. Metal 0002 remains <span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [Metal glossary](../../../../glossary.md#term-metal); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>`Blocked` pending its separate
  fresh owning audit/specification even though its Engine lifecycle prerequisite is satisfied.

## Implementation notes

- Both public prepared wrappers now use the existing `AdvancedRunResult` coordination pattern:
  the first close marks the outward handle terminal, one caller closes the exact inward Runtime
  owner, concurrent/repeated callers wait uninterruptibly and replay its retained immediate
  failure, and unregistration occurs in <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [owner glossary](../../../../glossary.md#term-owner))</span>`finally`.
- `AdvancedEngine` retains successfully published ordinary and advanced prepared wrappers in one
  private per-Engine publication-order list. Final shutdown snapshots and closes results in
  reverse run order, preparations in reverse prepare order, and the composition last.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [publication glossary](../../../../glossary.md#term-publication); [run glossary](../../../../glossary.md#term-run); [prepare glossary](../../../../glossary.md#prepare))</span>
- Public prepare and tuned/fallback final-publication paths use prepared-specific publication.
  Engine-close races and construction/evidence failures close the unpublished wrapper or inward
  execution exactly once before propagating the primary failure.
- One-shot compute/backward cleanup now always attempts the temporary result first and temporary
  preparation second, including run, validation, materialization, result-close, and
  preparation-close failures.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [materialization glossary](../../../../glossary.md#materialization))</span>
- Every representative completion, correctness, warmup, and timed action now owns its exact
  trial preparation and closes it after its Runtime result. Preparation-close failure is an
  execution-path failure that poisons the session and forbids fallback.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup))</span>
- Focused fake persistent-resource tests cover ordinary and advanced explicit close, retained
  failure replay, final shutdown ordering, one-shot cleanup, representative cleanup, selected
  retention, and deterministic final-publication races. Public-shape tests lock the exact
  additive <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>`AutoCloseable`, `isClosed()`, and no-checked-exception `close()` surface.

## Executable correction evidence

- The correction context closed the outward-close/inward-close admission gap in both wrappers.
  Package-private delegate retrieval now shares the wrapper monitor and rejects an outward-closed
  handle, while the winning close retains that monitor through Runtime's close transition. The
  monitor is released before any complete synchronous run begins; Runtime remains the sole lease
  authority and close remains non-waiting for an already acquired lease.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [Runtime glossary](../../../../glossary.md#term-runtime); [synchronous glossary](../../../../glossary.md#term-synchronous); [run glossary](../../../../glossary.md#term-run); [lease glossary](../../../../glossary.md#term-lease))</span>
- `AdvancedEngineLifecycleTest` now deterministically holds the inward Runtime monitor before its
  close transition, observes the wrapper close blocked while retaining the outward monitor,
  observes a concurrent run blocked at outward retrieval, then releases the gate and proves run
  rejection plus exact-once resource cleanup. The existing lease-first race test remains.<span data-glossary-navigation="true"> ([releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- `RepresentativeExecutionSession` now takes ownership of every supplied non-null fresh trial at
  action entry. <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`execute`, correctness capture, and correctness comparison close the trial after
  session-state, null/reference, or preflight rejection without double-closing after Runtime
  execution ownership proceeds. Focused tests cover null and foreign references, all three
  actions after session close, exact-once cleanup, suppression identity, and poisoning when
  rejected-trial cleanup fails.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [preflight glossary](../../../../glossary.md#term-preflight); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- At this executable-correction checkpoint, the exact focused five-class Engine command passed
  with 82 tests and no failures, and full <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>`./gradlew :modules:engine:test` passed with 93 tests and
  no failures. The later final follow-up added seven tests covering the ordinary
  [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) [race](../../../../glossary.md#term-race)/interruption and `RepresentativeExecutionSession` early-rejection
  cleanup; its focused five-class verification passed 89 of 89 tests and its full Engine
  verification passed 100 of 100 tests, both with no failures, errors, or skips. The exact
  three-fixture integration command passed with 10 tests, one environment-dependent skip, and no
  failures. <span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>`./gradlew :modules:engine:javadoc` passed after the executable [Javadoc](../../../../glossary.md#term-javadoc) corrections.
- `javap` reconfirmed the two prepared handles' exact public shape and package-private synchronized
  inward access. Generated prepared-handle pages contain the corrected terminal observation and
  inward-transition wording. Exact 22-path status, no-staged-file, <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>`git diff --check`, and
  `git diff --cached --check` checks passed at this correction [checkpoint](../../../../glossary.md#term-checkpoint).
- No Runtime, Prepare, backend, build, dependency, architecture-test, backend-conformance, or
  integration-source file changed. Explanatory documentation was intentionally not finalized in
  that executable correction. At that checkpoint Engine 0010, its master-plan row, and the
  roadmap were <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance))</span>`Review needed`; [Metal](../../../../glossary.md#term-metal) 0002 remained `Blocked`.

## Completion summary

- Completed changes: Implemented both public prepared-handle lifecycles, exact owner
  registration/unregistration, temporary one-shot and representative-trial cleanup, retained
  result/preparation/composition shutdown ordering, tuned/fallback ownership, and publication
  rollback. Independently finalized all directly affected Javadocs and lifecycle documentation.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [publication glossary](../../../../glossary.md#term-publication); [rollback glossary](../../../../glossary.md#term-rollback); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Production/[Javadoc](../../../../glossary.md#term-javadoc) files changed:
  `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java`,
  `AdvancedEngine.java`, `PreparedExecution.java`, `AdvancedPreparedExecution.java`,
  `RepresentativeExecutionSession.java`, `ModelAutotuningPreparation.java`,
  `EngineBackendComposition.java`, and `package-info.java`.
- Tests changed:
  `AdvancedEngineLifecycleTest.java`, `EngineTypedLifecycleTest.java`,
  `RepresentativeExecutionSessionTest.java`, `api/AdvancedEnginePublicShapeTest.java`, and
  `api/EngineTypedPublicShapeTest.java`. They cover exact resource close counts/order,
  result-before-preparation-before-composition shutdown, close failure replay, explicit-close
  unregistration, result independence, temporary/trial cleanup, selected retention, and
  deterministic close/publication races.
- Explanatory documentation changed: `docs/api/public-api.md`, `docs/api/runtime-api.md`,
  `docs/architecture/lifecycle.md`, `docs/backend-guide/writing-a-backend.md`, and
  `docs/glossary.md`. They now distinguish Engine handle ownership, Runtime physical-resource
  ownership, Prepare transfer, temporary cleanup, tuned-owner uniqueness, and shutdown order.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Planning changed: this task, the Engine master plan, and roadmap now record Engine 0010 as
  Complete. Metal 0002 remains Blocked; no Metal task was implemented or detailed here.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Validation performed: The current executable correction passed the exact focused five-class
  Engine command, full Engine tests, the exact focused three-class integration command, Engine
  Javadoc, generated-page inspection, <span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation))</span>`javap`, exact 22-path [scope](../../../../glossary.md#term-scope), staged-file inspection,
  `git diff --check`, and `git diff --cached --check`.
- Architecture/ADR/test-suite conclusions: No architecture, ADR, dependency, build, Runtime,
  Prepare, CPU/Metal production, architecture-test, backend-conformance, or integration-test
  source change was needed. Existing focused integration source was executed unchanged.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Documentation-agent review: This fresh correction pass finalized affected Javadoc,
  explanatory documentation, glossary impact, and planning evidence after independently checking
  the corrected executable paths and tests. It removed the public shutdown-order contradiction,
  corrected stale Prepare finalization wording, and documented early-rejection ownership and
  failure precedence.
- Unresolved issues: None for [Engine](../../../../glossary.md#term-engine) 0010.
- Required follow-up: Metal 0002 stays Blocked pending its separate fresh owning
  audit/specification; no Metal implementation or detailed task belongs to this change.
- Implementation context: current mandatory clean implementation context; the execution
  environment did not expose a separate context UUID.
- Documentation review context: no context identifier was exposed by this environment.

Status: Complete
