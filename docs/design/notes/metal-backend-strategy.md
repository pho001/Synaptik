# Metal backend strategy

## Purpose and authority

This pre-implementation note explains the intended Metal residency, preparation, and
synchronization direction. [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) remains authoritative;
this note is non-authoritative and does not change it. The direction below makes no operation
capability, platform-version, performance, or native application binary interface (ABI) promise.

## Mental model

Read the flow from left to right. Planning selects only Metal ownership; Metal preparation then
builds reusable native work, and Runtime executes the already-prepared boundary.

```text
maximal supported PlannedPartition region owned by Metal
  -> prepare-time Metal lowering and static-Shape specialization
  -> persistent MPSGraphExecutable + assigned MTLBuffer bindings
  -> synchronous region execution
  -> wait once at the region boundary or before CPU/host consumption
```

An `MTLBuffer` is a Metal device-visible byte buffer. It is the intended primary resident
representation for values participating in Metal work. An `MPSGraphExecutable` is the reusable
shape-specialized executable produced by compiling the supported region during prepare, not
during a run.

## Preparation and persistent ownership

Planning first selects `owner = Metal` from declarative capability and scoring facts. Metal
prepare then lowers the maximal supported region within that owned partition and compiles one
shape-specialized `MPSGraphExecutable`. The compiled executable is persistent prepared state,
while ordinary input, output, and scratch buffers keep their existing Runtime ownership roles.

Immutable constants may be uploaded to their assigned Metal buffers once during prepare when the
later executable task proves the exact representation and lifetime. Prepare 0006 will carry the
compiled executable or another persistent native object through the backend finalization result;
`PreparedExecution` will become its sole Runtime owner only after the complete preparation
transaction succeeds.

## Resident execution and transfers

`MPSGraphTensorData` is the Metal graph binding object for tensor storage. The intended route
binds assigned input and output `MTLBuffer` instances through `MPSGraphTensorData` and writes each
result directly into its preallocated Synaptik destination. It does not compute into a hidden
temporary result and copy that result merely to satisfy the prepared output slot.

Values remain resident across adjacent Metal work. Upload or download occurs only at an explicit
CPU/Metal boundary, before CPU consumption, or when the caller requests detached host output.
Detached host output is a copy with no lifetime alias to the Metal buffer. Repeated schedule
occurrences do not create persistent-resource ownership entries.

## Synchronization direction

No command-buffer wait belongs inside a compiled Metal region. The current Runtime call is
synchronous, so the first executable route waits once after submitting the region, at the end of
that region or before a later CPU step or host materialization must consume its result. General
asynchronous execution, deferred completion, cross-run overlap, and a public future or event model
remain deferred.

## Storage direction and deferred optimization

The intended storage layer uses a bounded buffer pool rather than unbounded process-lifetime
caching. A host-visible alias over `MTLBuffer.contents` is exposed, if a later task proves it safe,
only as a lease-bound `MemorySegment`; the alias cannot outlive the buffer lease and is not a
detached host result.

Custom Metal kernels and whole-plan autotuning are follow-ups. They may share resident
`MTLBuffer` representations, but they remain Metal-owned routes and must not move selection into
Planning or Runtime. Task 0002 remains blocked and without a detailed specification until Ready
Prepare 0006 and Draft Engine 0010 are both Complete.

## Training boundary

Training owns optimizer algorithms. If a later Metal route recognizes and fuses an optimizer
update, that executable remains a Metal backend concern. The training extension must not depend
on Metal or introduce a `MetalOptimizerBridge`.

## Related documentation

- [Metal backend guide](../../backend-guide/metal-backend.md)
- [Backend-owned lowering ADR](../decisions/0002-backend-owned-lowering.md)
- [Persistent prepared-resource lifecycle ADR](../decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [Metal master plan](../../planning/backends/metal/master-plan.md)
