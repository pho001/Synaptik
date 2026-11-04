# Advanced [Engine](../../glossary.md#term-engine) [lifecycle](../../glossary.md#lifecycle)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)

The advanced [facade](../../glossary.md#term-facade) serves callers composing lower-level lifecycle objects explicitly.
A **[composition root](../../glossary.md#term-composition-root)** is the place where opened [backend](../../glossary.md#backend) integrations are assembled and owned.
It does not discover implementations. A [borrowed](../../glossary.md#ownership-borrowing-and-lease) representation wraps caller storage without
taking responsibility for freeing that storage.

## Why this boundary takes representations

The application using Advanced Engine is also the low-level composition author: it explicitly
borrows CPU-compatible storage and orders the resulting Runtime representations for `run`.
Advanced Engine calls Runtime with those exact objects rather than discovering Tensor leaves or
returning host values. The borrowed wrapper and backing storage must remain usable until the
advanced result closes. The result owns its internal state, not the original storage. Prefer
[ordinary Engine](lifecycle.md) for descriptor-aware Tensor binding and detached copies;
[Runtime's inward lease](../runtime/publication.md) explains why representations are not values.

<a id="advanced-engine-lifecycle"></a>
### Advanced [Engine](../../glossary.md#term-engine) [lifecycle](../../glossary.md#lifecycle)

`AdvancedEngine` is the current low-level [composition root](../../glossary.md#term-composition-root). Construction accepts the exact
`CpuBackendIntegration` returned by `CpuBackendIntegration.open()` and takes [ownership](../../glossary.md#term-ownership-meanings) of it.
The lifecycle is deliberately explicit:

```text
Tensor outputs + compile leaves -> AdvancedCompiledGraph
AdvancedCompiledGraph           -> AdvancedPreparedExecution
prepared execution + borrowed BufferRepresentation inputs -> AdvancedRunResult
```

`compile(...)` requires a non-empty output list and accepts the four standalone [compile](../../glossary.md#compile) inputs
plus an optional `FunctionalGradientRequest`. The returned `AdvancedCompiledGraph` is an [opaque](../../glossary.md#term-opaque),
[immutable](../../glossary.md#term-immutable) handle bound to the exact open Engine that created it. `prepare(...)` accepts only such
a handle and currently succeeds only when [CPU](../../glossary.md#term-cpu) owns one non-empty maximal [partition](../../glossary.md#partition). A zero-[node](../../glossary.md#node),
mixed-[owner](../../glossary.md#term-owner), or multiple-partition [artifact](../../glossary.md#term-artifact) therefore fails during this CPU-only preparation
step; it is not silently repartitioned or routed elsewhere. The returned
`AdvancedPreparedExecution` is another immutable owner-bound, closeable handle. It owns exactly
one inward [Runtime](../../glossary.md#term-runtime) preparation and may be shared by concurrent callers because every `run(...)`
creates isolated [mutable](../../glossary.md#term-mutable) Runtime state. Callers should close it when reuse ends; closing it does
not close an already returned `AdvancedRunResult`.

Inputs to `run(...)` are already-created Runtime `BufferRepresentation` values in [graph](../../glossary.md#graph)-input
order. `borrow(...)` wraps one caller-owned `MemorySegmentStorage` for this purpose. Neither the
wrapper nor the Engine owns or closes the storage or its backing [arena](../../glossary.md#term-arena). The wrapper must remain
open, and its storage must remain valid, through advanced-result closure, including after the
[synchronous](../../glossary.md#term-synchronous) [run](../../glossary.md#term-run) returns. Closing the wrapper ends only that borrow. Compile and run do not retain
or mutate their caller lists.

`AdvancedRunResult` owns the completed [run state](../../glossary.md#run-state--runstate) until it or the Engine closes it. It currently
exposes only `resultCount()`, not published values. Its [idempotent](../../glossary.md#term-idempotent) `close()` [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) that state;
if a close is already in progress, another close waits uninterruptibly and restores interruption
before returning. Closing the Engine rejects new work with
`IllegalStateException("advanced engine is closed")` before [argument](../../glossary.md#term-argument), owner, or inward [validation](../../glossary.md#term-validation),
and waits for admitted work. Engine then closes still-open results in reverse run-[publication](../../glossary.md#term-publication)
order, retained prepared handles in reverse [prepare](../../glossary.md#prepare)-publication order, and its owned CPU
integration. Prepared handle close uses the same outward [synchronization](../../glossary.md#term-synchronization) and inward Runtime [lease](../../glossary.md#term-lease)
arbitration described for ordinary handles; it never holds the wrapper monitor over a complete
run. Concurrent/repeated close callers replay the first immediate wrapper-cleanup failure by
[identity](../../glossary.md#term-identity-meanings); a deferred resource failure belongs to the lease-releasing run path.

This advanced surface does not provide mixed-[backend composition](../../glossary.md#backend-composition), [backend](../../glossary.md#backend) discovery, typed
logical input [binding](../../glossary.md#term-binding), typed publication [metadata](../../glossary.md#term-metadata), [host](../../glossary.md#term-host) [materialization](../../glossary.md#materialization), public one-shot
execution, backward convenience, tuning integration, or ordinary Engine exception translation. Supplying a
[gradient](../../glossary.md#term-gradient) request to `compile(...)` invokes the existing [compiler](../../glossary.md#term-compiler) mode/request contract; it does
not add those later conveniences or guarantee that the current CPU-only prepare step can lower
the resulting artifact.


---

[Previous topic](../nn/model-builder.md) · [Next topic](composition.md)
