# Advanced [Engine](../../glossary.md#term-engine) and [prepared execution](../../glossary.md#prepared-execution--preparedexecution)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

A prepared execution owns reusable recipes and any persistent resources acquired for them.
A [run state](../../glossary.md#run-state--runstate) instead owns one invocation's [mutable](../../glossary.md#term-mutable) resources. A **[run](../../glossary.md#term-run) [lease](../../glossary.md#term-lease)** temporarily protects
the preparation needed by an admitted run; it is different from the returned result's lease over
completed outputs. Closing preparation stops new runs without revoking an already returned result.

## Two closeable owners, two lifetimes

The application calls `AdvancedEngine.prepare`; Engine calls shared preparation and retains the
returned Runtime owner behind its outward prepared handle. On each run, Engine lends input
representations to Runtime, which creates a new state and returns a completed-state result lease.
Closing the preparation prevents new runs; closing a result releases that invocation's resources.
Neither closes caller input storage. Borrowed storage must remain usable through result close,
not merely until synchronous execution returns. The [ordinary Engine API](engine-boundary.md)
adds typed Tensor binding and host copies without replacing these inner ownership rules.

<a id="current-advanced-engine-prepare-and-run-boundary"></a>
## Current advanced [Engine](../../glossary.md#term-engine) [prepare](../../glossary.md#prepare) and [run](../../glossary.md#term-run) boundary

`AdvancedEngine.prepare(...)` accepts only an `AdvancedCompiledGraph` created by that exact open
Engine. The [CPU](../../glossary.md#term-cpu)-only composition requires exactly one non-empty maximal [partition](../../glossary.md#partition) owned by its
CPU [backend](../../glossary.md#backend); zero-[node](../../glossary.md#node), mixed-[owner](../../glossary.md#term-owner), and multiple-partition [artifacts](../../glossary.md#term-artifact) [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation). On success it
returns an [opaque](../../glossary.md#term-opaque), [immutable](../../glossary.md#term-immutable), closeable `AdvancedPreparedExecution` bound to the same owner. The
handle owns exactly one inward [Runtime](../../glossary.md#term-runtime) [prepared execution](../../glossary.md#prepared-execution--preparedexecution) and may be shared across concurrent runs
because each run creates its own [`RunState`](../../glossary.md#run-state--runstate).

`AdvancedEngine.run(...)` accepts the prepared handle and an ordered list of already-created
`BufferRepresentation` inputs. `AdvancedEngine.borrow(...)` creates the supported non-owning CPU
wrapper around caller-owned `MemorySegmentStorage`; closing that wrapper does not close the
storage or its backing [arena](../../glossary.md#term-arena). Both must remain valid until the advanced result closes. Every
admitted run performs the current [synchronous](../../glossary.md#term-synchronous) Runtime [lifecycle](../../glossary.md#lifecycle) and returns `AdvancedRunResult`, whose
`resultCount()` reports [publication](../../glossary.md#term-publication) [cardinality](../../glossary.md#term-cardinality) but exposes no value or storage.

The result exclusively owns its completed `RunState` until its [idempotent](../../glossary.md#term-idempotent) close completes. The
Engine also tracks open results. Once Engine closure begins, a newly attempted prepare or run
fails with `IllegalStateException("advanced engine is closed")` before null, owner, or inward
[validation](../../glossary.md#term-validation). Explicit handle close rejects later runs but does not close an already returned
result. The wrapper synchronizes delegate retrieval with Runtime's close transition, [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) its
monitor before a complete run, and adds no second [lease](../../glossary.md#term-lease) or waiting protocol. Runtime leases decide
any later close/run [race](../../glossary.md#term-race). Engine close waits for admitted work, closes remaining results in reverse run order,
retained preparations in reverse publication order, and finally the exact CPU integration it took
into [ownership](../../glossary.md#term-ownership-meanings). Handles cease to be usable after their owner closes. This path does not provide
typed logical [binding](../../glossary.md#term-binding), typed publication
[metadata](../../glossary.md#term-metadata) or materialized results, backend discovery, mixed-backend transfers, public one-shot
execution, tuning, or ordinary backward convenience. Ordinary Engine privately uses the same
lifecycle [admission](../../glossary.md#term-admission) for its one-shot compute convenience without adding an advanced method.

<a id="current-prepared-execution"></a>
## Current [prepared execution](../../glossary.md#prepared-execution--preparedexecution)

[`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) is the final [identity](../../glossary.md#term-identity-meanings)-bearing [Runtime](../../glossary.md#term-runtime) [owner](../../glossary.md#term-owner) for one [immutable](../../glossary.md#term-immutable) reusable recipe
and its persistent prepared resources. It retains one exact [`PreparedMemoryPlan`](../../glossary.md#prepared-memory-plan--preparedmemoryplan) and one exact
[`PreparedSchedule`](../../glossary.md#prepared-schedule--preparedschedule). Construction requires both references to be non-null and requires
`schedule.memoryPlan() == memoryPlan`; a structurally equal plan created separately is not the
same prepared context.

The two-[argument](../../glossary.md#term-argument) constructor is the current resource-free compatibility path. The three-argument
constructor additionally snapshots an ordered `List<? extends PreparedResource>` into private
storage and transfers [ownership](../../glossary.md#term-ownership-meanings) only after all [validation](../../glossary.md#term-validation) succeeds. Entries must be non-null and
unique by [reference identity](../../glossary.md#term-reference-identity); `equals` is not consulted. The class exposes neither that aggregate
nor a resource count, [lookup](../../glossary.md#term-lookup), or physical value. Unlike the former record, it deliberately uses
`Object` identity equality, hashing, and text: [lifecycle](../../glossary.md#lifecycle)-bearing owners with equal recipes can own
different physical resources and are not values.

```java
// An empty schedule isolates owner/lease rules; it is not a general Engine zero-node execution promise.
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.List;

PreparedSchedule schedule = new PreparedSchedule(plan, List.of());
PreparedExecution execution = new PreparedExecution(plan, schedule);
```

Here `execution.memoryPlan()` is exactly `plan`, and `execution.schedule()` is exactly
`schedule`. The compatibility constructor owns no persistent resource, but the resulting owner
still has the same explicit close and [lease](../../glossary.md#term-lease) lifecycle. A null component reports `memoryPlan` or
`schedule` in validation order. A schedule for a different plan reference fails with
`IllegalArgumentException("schedule memory plan does not match prepared execution memory plan")`.

`acquireRunLease()` admits one [synchronous](../../glossary.md#term-synchronous) [run](../../glossary.md#term-run) only while the owner is open. `close()` marks the
owner closed before prepared-owner cleanup and never waits: it cleans immediately when no lease is
active, or the last admitted execution lease performs cleanup. New leases then fail with
`IllegalStateException(\"prepared execution is closed\")`. Cleanup invokes every resource once in
reverse constructor-list order outside the lifecycle monitor. A concrete resource close
relinquishes the prepared-owner reference; physical [release](../../glossary.md#term-release) may remain deferred only under an
already-issued run-owned read-only child [binding](../../glossary.md#term-binding). The first unchecked failure or error is primary;
later distinct failures are suppressed in encounter order, and the same exact primary object is
skipped to avoid self-suppression. Repeated execution or lease closure is a no-op and never replays
an earlier cleanup failure. `isClosed()` reports [admission](../../glossary.md#term-admission) closure, not whether a [backend](../../glossary.md#backend) child
lease has completed physical cleanup.


---

[Previous topic](engine-boundary.md) · [Next topic](memory.md)
