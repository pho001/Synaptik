# Advanced Engine lifecycle

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)

The advanced facade serves callers composing lower-level lifecycle objects explicitly.
A **composition root** is the place where opened backend integrations are assembled and owned.
It does not discover implementations. A borrowed representation wraps caller storage without
taking responsibility for freeing that storage.

<a id="advanced-engine-lifecycle"></a>
### Advanced Engine lifecycle

`AdvancedEngine` is the current low-level composition root. Construction accepts the exact
`CpuBackendIntegration` returned by `CpuBackendIntegration.open()` and takes ownership of it.
The lifecycle is deliberately explicit:

```text
Tensor outputs + compile leaves -> AdvancedCompiledGraph
AdvancedCompiledGraph           -> AdvancedPreparedExecution
prepared execution + borrowed BufferRepresentation inputs -> AdvancedRunResult
```

`compile(...)` requires a non-empty output list and accepts the four standalone compile inputs
plus an optional `FunctionalGradientRequest`. The returned `AdvancedCompiledGraph` is an opaque,
immutable handle bound to the exact open Engine that created it. `prepare(...)` accepts only such
a handle and currently succeeds only when CPU owns one non-empty maximal partition. A zero-node,
mixed-owner, or multiple-partition artifact therefore fails during this CPU-only preparation
step; it is not silently repartitioned or routed elsewhere. The returned
`AdvancedPreparedExecution` is another immutable owner-bound, closeable handle. It owns exactly
one inward Runtime preparation and may be shared by concurrent callers because every `run(...)`
creates isolated mutable Runtime state. Callers should close it when reuse ends; closing it does
not close an already returned `AdvancedRunResult`.

Inputs to `run(...)` are already-created Runtime `BufferRepresentation` values in graph-input
order. `borrow(...)` wraps one caller-owned `MemorySegmentStorage` for this purpose. Neither the
wrapper nor the Engine owns or closes the storage or its backing arena. The wrapper must remain
open, and its storage must remain valid, through advanced-result closure, including after the
synchronous run returns. Closing the wrapper ends only that borrow. Compile and run do not retain
or mutate their caller lists.

`AdvancedRunResult` owns the completed run state until it or the Engine closes it. It currently
exposes only `resultCount()`, not published values. Its idempotent `close()` releases that state;
if a close is already in progress, another close waits uninterruptibly and restores interruption
before returning. Closing the Engine rejects new work with
`IllegalStateException("advanced engine is closed")` before argument, owner, or inward validation,
and waits for admitted work. Engine then closes still-open results in reverse run-publication
order, retained prepared handles in reverse prepare-publication order, and its owned CPU
integration. Prepared handle close uses the same outward synchronization and inward Runtime lease
arbitration described for ordinary handles; it never holds the wrapper monitor over a complete
run. Concurrent/repeated close callers replay the first immediate wrapper-cleanup failure by
identity; a deferred resource failure belongs to the lease-releasing run path.

This advanced surface does not provide mixed-backend composition, backend discovery, typed
logical input binding, typed publication metadata, host materialization, public one-shot
execution, backward convenience, tuning integration, or ordinary Engine exception translation. Supplying a
gradient request to `compile(...)` invokes the existing compiler mode/request contract; it does
not add those later conveniences or guarantee that the current CPU-only prepare step can lower
the resulting artifact.


---

[Previous topic](../nn/model-builder.md) · [Next topic](composition.md)
