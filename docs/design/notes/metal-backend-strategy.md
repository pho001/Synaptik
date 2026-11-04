# [Metal](../../glossary.md#term-metal) [backend](../../glossary.md#backend) strategy


## Editorial reading context

This is explicitly a pre-implementation direction, not a current installation or
pooling recipe. Its deferred tasks and proposed buffer-pool/alias behavior remain
historical; do not infer present permission to reuse outputs or workspace.
The [current Metal guide](../../backend-guide/metal-backend.md) gives supported
setup and qualified routes.

Planning assigns a region to Metal; Metal preparation constructs its reusable
implementation; Runtime invokes it with run-specific representations. The
application requesting detached host output receives independent copied values,
not a view whose validity depends on a device-buffer lease. Conversely, synchronous
completion does not release the output resources retained by an open run result.
[ADR 0020](../decisions/0020-synchronous-single-default-device-metal-execution.md)
explains this completion-versus-lifetime distinction without turning concurrency
into a promise of GPU overlap.

## Purpose and authority

This pre-implementation note explains the intended [Metal](../../glossary.md#term-metal) [residency](../../glossary.md#residency), preparation, and
[synchronization](../../glossary.md#term-synchronization) direction. [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) remains authoritative;
this note is non-authoritative and does not change it. The direction below makes no [operation](../../glossary.md#operation)
capability, platform-version, performance, or native [application binary interface](../../glossary.md#term-application-binary-interface-abi) ([ABI](../../glossary.md#application-binary-interface-and-native-bridge)) promise.

## [Mental model](../../glossary.md#term-mental-model)

Read the flow from left to right. [Planning](../../glossary.md#term-planning) selects only [Metal](../../glossary.md#term-metal) [ownership](../../glossary.md#term-ownership-meanings); Metal preparation then
builds reusable native work, and [Runtime](../../glossary.md#term-runtime) executes the already-prepared boundary.

```text
maximal supported PlannedPartition region owned by Metal
  -> prepare-time Metal lowering and static-Shape specialization
  -> persistent MPSGraphExecutable + assigned MTLBuffer bindings
  -> synchronous region execution
  -> wait once at the region boundary or before CPU/host consumption
```

An `MTLBuffer` is a Metal [device](../../glossary.md#term-device)-visible byte [buffer](../../glossary.md#term-buffer). It is the intended primary resident
representation for values participating in Metal work. An `MPSGraphExecutable` is the reusable
[shape](../../glossary.md#shape)-specialized executable produced by compiling the supported region during [prepare](../../glossary.md#prepare), not
during a [run](../../glossary.md#term-run).

## Preparation and persistent [ownership](../../glossary.md#term-ownership-meanings)

[Planning](../../glossary.md#term-planning) first selects `owner = Metal` from declarative capability and scoring facts. [Metal](../../glossary.md#term-metal)
[prepare](../../glossary.md#prepare) then lowers the maximal supported region within that owned [partition](../../glossary.md#partition) and compiles one
[shape](../../glossary.md#shape)-specialized `MPSGraphExecutable`. The compiled executable is persistent prepared state,
while ordinary input, output, and scratch [buffers](../../glossary.md#term-buffer) keep their existing [Runtime](../../glossary.md#term-runtime) ownership roles.

[Immutable](../../glossary.md#term-immutable) [constants](../../glossary.md#term-constant) may be uploaded to their assigned Metal buffers once during prepare when the
later executable task proves the exact representation and [lifetime](../../glossary.md#term-lifetime). Prepare 0006 will [carry](../../glossary.md#term-carry) the
compiled executable or another persistent native object through the [backend](../../glossary.md#backend) finalization result;
[`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) will become its sole Runtime [owner](../../glossary.md#term-owner) only after the complete preparation
[transaction](../../glossary.md#term-transaction) succeeds.

## Resident execution and transfers

`MPSGraphTensorData` is the [Metal](../../glossary.md#term-metal) [graph](../../glossary.md#graph) [binding](../../glossary.md#term-binding) object for [tensor](../../glossary.md#tensor) storage. The intended route
binds assigned input and output `MTLBuffer` instances through `MPSGraphTensorData` and writes each
result directly into its preallocated Synaptik destination. It does not compute into a hidden
temporary result and copy that result merely to satisfy the prepared output slot.

Values remain resident across adjacent Metal work. Upload or download occurs only at an explicit
[CPU](../../glossary.md#term-cpu)/Metal boundary, before CPU consumption, or when the caller requests [detached](../../glossary.md#detached-model-results) [host](../../glossary.md#term-host) output.
Detached host output is a copy with no [lifetime](../../glossary.md#term-lifetime) [alias](../../glossary.md#aliasing-and-arena) to the Metal [buffer](../../glossary.md#term-buffer). Repeated schedule
occurrences do not create persistent-resource [ownership](../../glossary.md#term-ownership) entries.

## [Synchronization](../../glossary.md#term-synchronization) direction

No command-[buffer](../../glossary.md#term-buffer) wait belongs inside a compiled [Metal](../../glossary.md#term-metal) region. The current [Runtime](../../glossary.md#term-runtime) call is
[synchronous](../../glossary.md#term-synchronous), so the first executable route waits once after submitting the region, at the end of
that region or before a later [CPU](../../glossary.md#term-cpu) step or [host](../../glossary.md#term-host) [materialization](../../glossary.md#materialization) must consume its result. General
[asynchronous](../../glossary.md#term-asynchronous) execution, deferred completion, cross-[run](../../glossary.md#term-run) overlap, and a public future or event [model](../../glossary.md#term-model)
remain deferred.

## Storage direction and deferred optimization

The intended storage layer uses a bounded [buffer](../../glossary.md#term-buffer) pool rather than unbounded process-[lifetime](../../glossary.md#term-lifetime)
caching. A [host](../../glossary.md#term-host)-visible [alias](../../glossary.md#aliasing-and-arena) over `MTLBuffer.contents` is exposed, if a later task proves it safe,
only as a [lease](../../glossary.md#term-lease)-bound [`MemorySegment`](../../glossary.md#term-memorysegment); the alias cannot outlive the buffer lease and is not a
[detached](../../glossary.md#detached-model-results) host result.

Custom [Metal](../../glossary.md#term-metal) [kernels](../../glossary.md#kernel) and whole-plan [autotuning](../../glossary.md#autotuning--model-autotuning) are follow-ups. They may share resident
`MTLBuffer` representations, but they remain Metal-owned routes and must not move selection into
[Planning](../../glossary.md#term-planning) or [Runtime](../../glossary.md#term-runtime). Task 0002 remains blocked and without a detailed specification until [Ready](../../glossary.md#term-ready)
[Prepare](../../glossary.md#prepare) 0006 and Draft [Engine](../../glossary.md#term-engine) 0010 are both Complete.

## Training boundary

Training owns [optimizer](../../glossary.md#term-optimizer) algorithms. If a later [Metal](../../glossary.md#term-metal) route recognizes and fuses an optimizer
update, that executable remains a Metal [backend](../../glossary.md#backend) concern. The training extension must not depend
on Metal or introduce a `MetalOptimizerBridge`.

## Related documentation

- [Metal backend guide](../../backend-guide/metal-backend.md)<span data-glossary-navigation="true"> ([Metal glossary](../../glossary.md#term-metal); [backend glossary](../../glossary.md#backend))</span>
- [Backend-owned lowering ADR](../decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([Backend-owned lowering glossary](../../glossary.md#backend-owned-lowering); [ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>
- [Persistent prepared-resource lifecycle ADR](../decisions/0013-prepared-execution-persistent-resource-lifecycle.md)<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Metal master plan](../../planning/backends/metal/master-plan.md)
