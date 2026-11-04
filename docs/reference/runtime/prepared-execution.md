# Advanced Engine and prepared execution

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

<a id="current-advanced-engine-prepare-and-run-boundary"></a>
## Current advanced Engine prepare and run boundary

`AdvancedEngine.prepare(...)` accepts only an `AdvancedCompiledGraph` created by that exact open
Engine. The CPU-only composition requires exactly one non-empty maximal partition owned by its
CPU backend; zero-node, mixed-owner, and multiple-partition artifacts fail closed. On success it
returns an opaque, immutable, closeable `AdvancedPreparedExecution` bound to the same owner. The
handle owns exactly one inward Runtime prepared execution and may be shared across concurrent runs
because each run creates its own `RunState`.

`AdvancedEngine.run(...)` accepts the prepared handle and an ordered list of already-created
`BufferRepresentation` inputs. `AdvancedEngine.borrow(...)` creates the supported non-owning CPU
wrapper around caller-owned `MemorySegmentStorage`; closing that wrapper does not close the
storage or its backing arena. Both must remain valid until the advanced result closes. Every
admitted run performs the current synchronous Runtime lifecycle and returns `AdvancedRunResult`, whose
`resultCount()` reports publication cardinality but exposes no value or storage.

The result exclusively owns its completed `RunState` until its idempotent close completes. The
Engine also tracks open results. Once Engine closure begins, a newly attempted prepare or run
fails with `IllegalStateException("advanced engine is closed")` before null, owner, or inward
validation. Explicit handle close rejects later runs but does not close an already returned
result. The wrapper synchronizes delegate retrieval with Runtime's close transition, releases its
monitor before a complete run, and adds no second lease or waiting protocol. Runtime leases decide
any later close/run race. Engine close waits for admitted work, closes remaining results in reverse run order,
retained preparations in reverse publication order, and finally the exact CPU integration it took
into ownership. Handles cease to be usable after their owner closes. This path does not provide
typed logical binding, typed publication
metadata or materialized results, backend discovery, mixed-backend transfers, public one-shot
execution, tuning, or ordinary backward convenience. Ordinary Engine privately uses the same
lifecycle admission for its one-shot compute convenience without adding an advanced method.

<a id="current-prepared-execution"></a>
## Current prepared execution

`PreparedExecution` is the final identity-bearing Runtime owner for one immutable reusable recipe
and its persistent prepared resources. It retains one exact `PreparedMemoryPlan` and one exact
`PreparedSchedule`. Construction requires both references to be non-null and requires
`schedule.memoryPlan() == memoryPlan`; a structurally equal plan created separately is not the
same prepared context.

The two-argument constructor is the current resource-free compatibility path. The three-argument
constructor additionally snapshots an ordered `List<? extends PreparedResource>` into private
storage and transfers ownership only after all validation succeeds. Entries must be non-null and
unique by reference identity; `equals` is not consulted. The class exposes neither that aggregate
nor a resource count, lookup, or physical value. Unlike the former record, it deliberately uses
`Object` identity equality, hashing, and text: lifecycle-bearing owners with equal recipes can own
different physical resources and are not values.

```java
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.List;

PreparedSchedule schedule = new PreparedSchedule(plan, List.of());
PreparedExecution execution = new PreparedExecution(plan, schedule);
```

Here `execution.memoryPlan()` is exactly `plan`, and `execution.schedule()` is exactly
`schedule`. The compatibility constructor owns no persistent resource, but the resulting owner
still has the same explicit close and lease lifecycle. A null component reports `memoryPlan` or
`schedule` in validation order. A schedule for a different plan reference fails with
`IllegalArgumentException("schedule memory plan does not match prepared execution memory plan")`.

`acquireRunLease()` admits one synchronous run only while the owner is open. `close()` marks the
owner closed before prepared-owner cleanup and never waits: it cleans immediately when no lease is
active, or the last admitted execution lease performs cleanup. New leases then fail with
`IllegalStateException(\"prepared execution is closed\")`. Cleanup invokes every resource once in
reverse constructor-list order outside the lifecycle monitor. A concrete resource close
relinquishes the prepared-owner reference; physical release may remain deferred only under an
already-issued run-owned read-only child binding. The first unchecked failure or error is primary;
later distinct failures are suppressed in encounter order, and the same exact primary object is
skipped to avoid self-suppression. Repeated execution or lease closure is a no-op and never replays
an earlier cleanup failure. `isClosed()` reports admission closure, not whether a backend child
lease has completed physical cleanup.


---

[Previous topic](engine-boundary.md) · [Next topic](memory.md)
