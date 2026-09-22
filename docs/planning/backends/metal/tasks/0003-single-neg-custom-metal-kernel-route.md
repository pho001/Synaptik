# Task 0003: Single-NEG Custom Metal Kernel Route

## Status

Complete

## Goal

Prove a second backend-private Metal execution route without changing Metal occurrence
capability. A whole Metal partition uses a direct custom Metal compute kernel only when it has
exactly one supported unary `NEG` occurrence, one unique feed, one unique target, and a checked
positive element count no greater than `UINT32_MAX`. The feed may be either a caller input or the
already-supported positive-rank `FLOAT32` splat represented by an `InitializedBuffer`. Every
other currently supported Metal NEG partition—including an otherwise matching singleton above
that custom 32-bit index domain—continues to use the task-0002 Metal Performance Shaders Graph
(`MPSGraph`) route.

```text
unchanged occurrence capability -> Planning chooses owner = metal
  -> Metal analysis sees the complete maximal partition
     -> exactly 1 NEG + 1 feed + 1 target + elementCount <= UINT32_MAX:
        direct custom NEG pipeline
     -> every other supported NEG partition: existing MPSGraph executable
  -> shared slot assignment
  -> route-specific persistent resource construction during finalization
  -> route-specific cold binding
  -> one synchronous native downcall into the assigned output buffer
```

This is a bounded second-route proof. It establishes that Metal analysis can select between two
private implementations before declaring resources without adding a generic route framework,
changing partitioning, or claiming that the custom kernel is faster. Metal 0004 remains the Draft
owner of complete typed route candidates, target/cache compatibility, and tuning integration.

## Pre-implementation audit

### Current repository surface

- `MetalCapabilityProvider` admits exactly positive fully static rank-`1..16`, resolved
  dense-contiguous, non-view, zero-offset `FLOAT32` unary `NEG` occurrences with equal input and
  output Shapes and equal `requiresGrad` flags. Task 0003 does not modify this file.
- Planning asks capability per occurrence and then creates maximal same-owner partitions. It
  neither sees nor selects MPSGraph or a custom kernel.
- `MetalNegPartitionPreparer` currently validates the complete supported NEG partition, derives
  stable unique feeds and targets, declares every boundary buffer, and always declares one
  MPSGraph address workspace.
- `MetalNegPreparationPlan` currently retains the complete stable MPSGraph lowering arrays plus
  feed/target facts and the address-workspace declaration. Those facts already cover the exact
  singleton case and checked element geometry; no new graph abstraction is needed.
- `MetalNegPartitionFinalizer` currently validates assignments and constructs one
  `MetalMpsGraphExecutableResource`. `MetalNegPreparedExecutable` then cold-marshals feed/target
  addresses into a run-owned workspace before its one hot native call.
- `MetalNegPreparedScheduleAssembler` already distinguishes caller feeds from exact `FLOAT32`
  splats and preserves caller borrowing, per-run splat/output ownership, publication order, and
  rollback. The custom route needs the same buffer recipes but no address-array workspace.
- `PreparedExecution` and shared Prepare already provide the required identity-unique persistent
  resource ownership, transactional finalization handoff, reverse cleanup, suppression rules,
  close/run leases, and concurrent-run isolation.
- The public Metal surface remains only `MetalCapabilityProvider`; current public Engine
  composition remains CPU-only.

The audit therefore selects an in-place extension of the existing NEG-specific owners. It does
not rename or split the preparer, finalizer, prepared executable, or schedule assembler, and it
does not create a generic Metal route, executable-handle, or pipeline registry.

### Installed SDK evidence

The planning audit used the installed macOS SDK at
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk` on the current arm64 macOS host. The relevant
Metal contracts are:

- `MTLDevice.newLibraryWithSource:options:error:` synchronously compiles Metal Shading Language
  (`MSL`) source into an `MTLLibrary` and reports failure with `NSError` or a nil result;
- `MTLLibrary.newFunctionWithName:` resolves one named compute function;
- `MTLDevice.newComputePipelineStateWithFunction:error:` synchronously creates a reusable
  `MTLComputePipelineState` and reports failure with `NSError` or a nil result;
- `MTLComputePipelineState.threadExecutionWidth` and
  `maxTotalThreadsPerThreadgroup` provide the legal one-dimensional threadgroup bound;
- `MTLCommandQueue.commandBuffer`, `MTLCommandBuffer.computeCommandEncoder`,
  `setComputePipelineState:`, `setBuffer:offset:atIndex:`, `endEncoding`, `commit`, and
  `waitUntilCompleted` provide the direct synchronous compute path;
- `dispatchThreads:threadsPerThreadgroup:` accepts an arbitrarily sized grid and non-uniform
  final threadgroup on supported targets, so an exact grid contains the static element count and
  does not require a rounded-grid bounds branch; and
- after `waitUntilCompleted`, `MTLCommandBuffer.status` must be
  `MTLCommandBufferStatusCompleted` and `error` must be nil for success.

Apple's official compute examples use a 32-bit unsigned `thread_position_in_grid` index. The
unchanged Metal capability accepts checked positive `FLOAT32` element counts beyond
`UINT32_MAX`, so topology alone cannot select this kernel truthfully. Task 0003 therefore treats
`elementCount <= UINT32_MAX` as a Metal-private implementation-domain predicate during analysis.
It is not a capability predicate: an otherwise supported singleton above the limit stays on
MPSGraph. Within the custom domain, exact non-uniform `dispatchThreads` keeps the branch-free
oracle valid. A bounds guard would not repair 32-bit index wrap above `UINT32_MAX` and is not the
selected design. No throughput or superiority conclusion follows from these API contracts.

## Scope

- Keep `MetalCapabilityProvider.java` byte-for-byte unchanged. Capability remains exactly the
  task-0002 domain and remains independent of route, native availability, source provenance, and
  partition topology.
- Extend `MetalNegPartitionPreparer` to select one of exactly two NEG-specific private routes
  before constructing `BackendPartitionAnalysis`:
  - `CUSTOM_SINGLE_NEG` when the complete partition has exactly one NEG node, exactly one unique
    feed, exactly one unique target, a checked positive element count no greater than
    `UINT32_MAX`, and valid native `MTLSize`/`NSUInteger`/pipeline geometry facts; or
  - `MPSGRAPH` for every other partition already accepted by task 0002.
- Treat caller and exact positive-rank `FLOAT32` splat feeds identically for route eligibility.
  Source provenance changes only the existing schedule buffer recipe, not capability or route
  selection.
- Extend `MetalNegPreparationPlan` with one package-private closed route discriminator and an
  optional address-workspace declaration. Preserve all stable feed, target, descriptor, geometry,
  splat, and MPSGraph arrays already owned by this plan. The custom route has exactly two buffer
  declarations and no workspace declaration; the MPSGraph route retains its existing boundary
  declarations and exactly one address workspace.
- Keep `MetalNegPartitionPreparer`, `MetalNegPartitionFinalizer`,
  `MetalNegPreparedExecutable`, and `MetalNegPreparedScheduleAssembler` under their current names
  and responsibilities. Extend them with explicit two-route branches at analysis, finalization,
  and cold binding. Do not introduce a generic route interface, route registry, generic
  executable handle, string dispatch, reflection, or parameter bag.
- Add one package-private `MetalNegKernelPipelineResource` implementing `PreparedResource`. It
  owns exactly one opaque custom-NEG native pipeline handle and one adopted context child lease.
  It is not an MPSGraph handle and is never passed to an MPSGraph release or run function.
- During custom-route finalization, compile the exact direct `FLOAT32` MSL kernel and reusable
  `MTLComputePipelineState` once, after shared buffer assignment. Return the typed pipeline
  resource exactly once in acquisition order.
- Preserve the existing MPSGraph finalization path and `MetalMpsGraphExecutableResource`
  semantics unchanged for every supported partition outside the custom implementation domain,
  including an otherwise matching singleton whose element count is greater than `UINT32_MAX`.
- Construct a route-specific `MetalNegPreparedExecutable` recipe:
  - the MPSGraph form retains the existing feed/target selections plus one address-workspace
    selection and cold address marshalling; and
  - the custom form has exactly one read-only buffer selection, one write-only buffer selection,
    no workspace selection, and a cold-bound invocation retaining direct typed references to the
    custom pipeline resource and the two compatible `MetalBufferRepresentation` values.
- Extend the schedule assembler only enough to omit the address workspace for the custom route.
  Preserve the existing caller input, per-run initialized splat, fresh per-run output, execution,
  and publication recipes.
- Evolve the same native library additively to exact ABI version `3` as defined below. Preserve
  all ten ABI-v2 symbols and their signatures and semantics.
- Add deterministic fake-native tests for route selection, exact declarations, both finalization
  branches, typed-handle separation, rollback and suppression, cold binding, one-downcall hot
  execution, status mapping, and MPSGraph regression.
- Add real-device execution that proves the custom route for both caller and splat feeds, direct
  assigned output, raw `FLOAT32` NEG semantics, pipeline reuse, and per-run isolation, plus an
  explicit multi-node case that proves MPSGraph remains selected.
- Finalize affected Javadocs, the Metal backend guide, native README, glossary impact, and
  planning evidence through the mandatory separate clean documentation-focused pass.

## Exact route boundary

### Capability and partition closure

The capability predicate remains byte-for-byte semantically unchanged:

1. operation kind is unary `NEG` with `NoOperationAttrs.INSTANCE`;
2. there is exactly one input and one output descriptor;
3. both data types are `FLOAT32`;
4. both Shapes are fully static, equal, rank `1..16`, and strictly positive in every dimension;
5. checked element and byte geometry succeeds;
6. both layouts are resolved dense-contiguous, non-view, and zero-offset; and
7. both `requiresGrad` flags are equal, with either shared value accepted.

Task 0003 adds no operation, type, Shape, layout, gradient, availability, or device predicate to
the provider. Existing provider and backend-conformance tests must pass unchanged.

Per-occurrence capability plus maximal same-owner partitioning is the reason `ADD` and every
other new operation are excluded. Advertising `ADD` would allow Planning to form arbitrary
maximal Metal partitions interleaving supported NEG and ADD occurrences. A truthful capability
change would therefore require complete mixed-partition lowering, not merely one binary shader.
This task instead proves a second implementation route inside the already-closed NEG partition
domain.

### Analysis-time route selection

Analysis first performs the existing complete partition validation and derives the existing
stable value/feed/target facts. It then applies this deterministic initial heuristic:

```text
node count == 1
and unique feed count == 1
and unique target count == 1
and 1 <= checked element count <= UINT32_MAX
and native MTLSize/NSUInteger/custom-pipeline geometry is valid
  -> CUSTOM_SINGLE_NEG
otherwise
  -> MPSGRAPH
```

The heuristic is a correctness-safe initial policy, not a measurement result or performance
claim. It runs before resource declarations are finalized. No compatible cache, benchmark,
runtime observation, buffer address, current residency, or caller/splat distinction participates.
The element-count and native-geometry checks are route-selection implementation-domain
predicates. They do not alter capability, request repartitioning, constitute fallback, or compare
performance. Metal 0004 may later replace this fixed choice with typed complete candidates and
compatible decisions without changing Planning ownership.

The custom plan declares the feed and target buffers only. The MPSGraph plan preserves the
existing feed-then-target buffers plus one exact address workspace. Finalization must reject a
route/declaration mismatch; it may not change the selected route, add a workspace, or fall back
from custom to MPSGraph after assignment.

## Direct MSL oracle and dispatch

The clean optimal source oracle is exactly one thread per logical element:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void synaptik_neg_f32(
        device const float *input [[buffer(0)]],
        device float *output [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
    output[index] = -input[index];
}
```

The production source may differ only in formatting needed by the native string literal. It must
not add another operation, bounds branch, temporary buffer, scalar loop, vector-width promise,
lookup, or hidden copy.

Preferred dispatch uses `dispatchThreads` with a one-dimensional grid width equal to the checked
static element count and a legal positive threadgroup width derived from
`threadExecutionWidth` and `maxTotalThreadsPerThreadgroup`. On a target proven to support
non-uniform threadgroups, Metal launches exactly that grid, including a possibly smaller last
threadgroup, so no bounds branch is present. Analysis never selects the custom route above
`UINT32_MAX`; those supported singleton partitions remain MPSGraph. Native create independently
rejects zero or greater-than-`UINT32_MAX` element counts with `UNSUPPORTED_SHAPE`, then validates
all `MTLSize`, `NSUInteger`, pipeline-width, and exact-dispatch geometry before publishing a
pipeline. If exact non-uniform dispatch cannot be proved for a target, analysis must select
MPSGraph for that partition or preparation must fail closed before pipeline publication; a
rounded guarded kernel is not an alternative because it cannot repair 32-bit index wrap.

## Exact native ABI version 3

ABI version is the unsigned 32-bit value `3`. The version function keeps its name. All six
foundation functions and all three MPSGraph functions keep their ABI-v2 names, signatures,
ownership, validation order, statuses, and behavior. Version `3` adds exactly these three typed
custom-NEG symbols:

```c
int32_t synaptik_metal_neg_kernel_pipeline_create(
        void *context,
        uint64_t element_count,
        void **out_pipeline);

int32_t synaptik_metal_neg_kernel_pipeline_release(void *pipeline);

int32_t synaptik_metal_neg_kernel_pipeline_run(
        void *pipeline,
        void *input_buffer,
        void *output_buffer);
```

The dylib therefore exports exactly thirteen `synaptik_metal_*` symbols. Java validates version
`3`, then eagerly resolves all thirteen symbols in this order before context creation:

1. version;
2. the existing six context/buffer functions in their ABI-v1 order;
3. the existing three MPSGraph functions in their ABI-v2 order; and
4. custom pipeline create, release, then run.

There is no optional symbol, alias, lazy route binding, generic executable create/release/run
symbol, function-name parameter, source parameter, raw pipeline accessor, or ABI fallback.

### Statuses and fail-closed mapping

Statuses `0..11` retain their exact ABI-v2 meanings. Version `3` appends one status:

| Value | C name | Meaning |
|---:|---|---|
| `12` | `SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED` | MSL source compilation, named function lookup, target-support validation, or compute-pipeline creation failed. |

The custom functions reuse existing statuses as follows:

- `INVALID_ARGUMENT` (`1`) covers a null required pointer/handle, malformed success/failure
  output-cell protocol, or another non-shape scalar precondition failure;
- `ALLOCATION_FAILED` (`4`) covers native custom-pipeline box allocation;
- `INTERNAL_ERROR` (`7`) covers a contained Objective-C exception;
- `UNSUPPORTED_SHAPE` (`8`) covers zero, a count greater than `UINT32_MAX`, checked
  `element_count * sizeof(float)` overflow, or a count not safely representable by the required
  MSL grid/`MTLSize`/`NSUInteger` strategy;
- `INCOMPATIBLE_RESOURCE` (`10`) covers wrong-device input/output buffers, insufficient logical
  extent, or input/output handle aliasing;
- `EXECUTION_FAILED` (`11`) covers nil command buffer or encoder, an unusable pipeline dispatch
  geometry, command-buffer completion status other than `Completed`, or a non-nil completion
  error; and
- `KERNEL_COMPILATION_FAILED` (`12`) covers nil/error library, function, pipeline, or required
  non-uniform-dispatch target proof.

Unknown integers remain unknown, retain the raw value in the Java failure, and fail closed without
retry, remapping, or MPSGraph fallback.

### Create, handle, and ownership contract

- `out_pipeline` is validated first and set to null before validating the context or element
  count. Every failure leaves it null.
- `context` must be one live native context handle supplied by its owning Java wrapper.
  `element_count` must be in the inclusive range `1..UINT32_MAX`; zero or a greater value returns
  `UNSUPPORTED_SHAPE`. Checked byte extent must fit `uint64_t`, `NSUInteger`, and the supported
  Metal buffer/dispatch geometry.
- Native code compiles only the fixed in-source `synaptik_neg_f32` function. It accepts no source,
  function name, macro, library path, metallib, or specialization input from Java.
- Creation synchronously calls `newLibraryWithSource`, resolves exactly
  `synaptik_neg_f32`, creates one `MTLComputePipelineState`, validates positive legal pipeline
  thread widths, and stores the checked element count, byte extent, selected dispatch geometry,
  pipeline, and exact context prerequisite in one retained opaque box.
- On success, `out_pipeline` receives exactly one retained handle. The box strongly retains the
  pipeline state and context needed for its device and command queue. Temporary library,
  function, compile options, and `NSError` values do not cross the ABI.
- Java acquires one provisional context child lease before native create. Successful wrapper
  construction adopts it. Native-create failure releases the lease; wrapper-construction failure
  releases the native pipeline first and the lease second. Primary and distinct suppressed
  failures follow the existing task-0002 order exactly.
- `synaptik_metal_neg_kernel_pipeline_release` consumes one live non-null custom-pipeline handle
  exactly once. Java supplies synchronized idempotency and never retries a failed native release.
- A custom-pipeline handle is never passed to an MPSGraph function, and an MPSGraph executable
  handle is never passed to a custom-pipeline function. Java fake seams and native tests must
  prove this typed separation.

### Run validation, encoding, and synchronization

- Java cold binding validates exactly one live context-local input and one live context-local
  output, each with logical bytes at least the pipeline's retained required extent, and rejects
  Java representation identity alias before constructing the bound invocation.
- Native run validates non-null live handles, exact device identity, sufficient retained logical
  extents, and distinct native buffer handles before creating command objects.
- Native run creates one command buffer and one compute encoder, binds the retained pipeline,
  binds input at buffer index `0` and output at index `1` with zero offsets, dispatches the
  retained static element count through exact non-uniform `dispatchThreads`, ends encoding,
  commits, and calls `waitUntilCompleted` exactly once.
- Success requires `MTLCommandBufferStatusCompleted` and a nil `error`. Any other completed state
  or error maps to `EXECUTION_FAILED`.
- Success writes the caller-supplied assigned output `MTLBuffer` directly. Synaptik performs no
  explicit intermediate/output copy, host staging, MPSGraph call, fallback, retry, or second
  synchronization.
- A native autorelease pool and bounded command objects per invocation are permitted. There is no
  allocation proportional to element count.

## Java lifecycle and hot-path semantics

### Analysis and finalization

`MetalNegPreparationPlan` gains one nested/package-private route enum with exactly
`CUSTOM_SINGLE_NEG` and `MPSGRAPH`. It is not a tuning candidate schema and is not exposed to
shared Prepare. The plan retains the same exact `MetalDeviceContext` identity and boundary facts.
Its address-workspace requirement is present only for `MPSGRAPH`.

`MetalNegPartitionFinalizer` validates assignment count by route. The custom route must have two
buffer assignments and no workspace assignment. The MPSGraph route retains its existing buffer
plus workspace rules. Finalization branches once on the plan route, creates the matching typed
resource, and constructs the matching prepared recipe. A route/resource mismatch fails before
return; no route changes or fallback occur.

Each resource is returned exactly once as one `PreparedResource`. Finalizer-local rollback,
shared Prepare rollback, successful ownership transfer, reverse cleanup, primary/suppressed
failure order, self-suppression avoidance, and context-child lease rules remain unchanged.

### Schedule and per-run ownership

The schedule assembler preserves exact context identity and sole-partition validation. For both
routes:

- a caller feed is borrowed and never closed by Runtime;
- a splat feed is one fresh run-owned initialized Metal buffer per run with exact raw `FLOAT32`
  bits;
- the output is one fresh run-owned Metal buffer per run;
- execution writes that output directly; and
- publication transfers/leasing follows existing Runtime behavior.

Only the MPSGraph route creates the address workspace. The custom route creates no workspace,
pointer array, scalar buffer, hidden intermediate, or binding owner.

### Cold binding and hot invocation

`MetalNegPreparedExecutable` remains one NEG-specific recipe class with two explicit construction
paths. It must not store a raw or generic resource supertype that requires a hot cast or switch.
Cold binding performs the route-specific branch and returns a route-specific bound invocation:

- MPSGraph retains the existing direct resource plus pre-marshalled input/output address slices;
  and
- custom retains the typed `MetalNegKernelPipelineResource`, input
  `MetalBufferRepresentation`, and output `MetalBufferRepresentation` directly.

The custom `executeBound()` performs exactly one call to the typed resource, which performs
exactly one native downcall. From Java hot execution through the downcall there is no allocation,
boxing, reflection, collection/map lookup, slot lookup, representation lookup, address-array
marshalling, unsafe cast, string dispatch, operation inspection, graph traversal, route branch,
kernel selection, retry, fallback, upload, download, or Java synchronization beyond the narrow
persistent-resource invocation gate already used to serialize safe native use.

Concurrent runs may reuse the immutable pipeline resource but have different `RunState`, splat,
output, and publication resources. No overlap or throughput claim is made.

## Out of scope

- any change to `MetalCapabilityProvider`, Planning capability, maximal same-owner partitioning,
  or logical memory planning
- `ADD`, `RELU`, another operation, mixed NEG/other partitions, pointwise fusion, or broader
  custom-kernel lowering
- FLOAT16, BFLOAT16, FLOAT64, integer, BOOL, scalar-rank, zero-extent, dynamic Shape, unresolved
  layout, view, offset, strided, broadcast, or multi-output capability
- generic route interfaces, route registries, executable handles, pipeline caches, shader-source
  registries, generic parameter bags, reflection, or string-selected kernels
- Metal 0004 typed candidates, compatibility fingerprints, workload cache, model tuning,
  benchmarking, or heuristic-performance evidence
- public Metal adapter/factory/integration, Engine source, standard `compute`, automatic host
  binding, mixed-owner schedule composition, CPU fallback, or backend discovery
- asynchronous APIs, futures, events, cancellation, timeout, cross-run overlap promises, or
  exposed command buffers
- buffer pooling, aliasing, persistent prepared splat buffers, public device storage, host-visible
  leases, or hidden coherence
- metallib compilation, packaging, lookup, discovery, extraction, signing, notarization, or
  distribution; the fixed MSL source is compiled during finalization
- architecture authority, ADR, shared production module, production dependency, Gradle,
  architecture-test, backend-conformance, or integration-test changes
- a detailed Metal 0004 task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/planning`, `modules/runtime`, `modules/prepare`, Concrete backend modules, Metal
  backend, Prepare lifecycle, Run lifecycle, and dependency rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [Metal strategy note](../../../../design/notes/metal-backend-strategy.md)
- [Metal master plan](../master-plan.md)
- [Metal task 0001](0001-metal-capability-storage-and-native-foundation.md)
- [Metal task 0002](0002-mpsgraph-prepared-execution-route.md)

## Architecture constraints

- Planning selects only `owner = metal`. Route selection is deterministic backend analysis over
  the complete already-owned partition and never leaks into Planning or Runtime.
- The existing occurrence capability must remain closed under the actual maximal same-owner
  partitioner. The second route does not reject, split, or repartition a supported partition;
  every supported partition outside the private custom domain remains on MPSGraph, including an
  exact singleton with more than `UINT32_MAX` elements.
- Backend analysis selects the route and exact declarations before shared assignment.
  Finalization constructs only the already-selected route and adds no resource requirement.
- Runtime receives an immutable prepared recipe and typed cold-bound invocation. It sees no
  `Operation`, `CompiledNode`, route selection, native handle, MSL source, MPSGraph type, or
  backend lookup on the hot path.
- `PreparedExecution` remains the unique owner of the pipeline or MPSGraph resource. Executable
  and schedule references create no ownership occurrence.
- Caller borrowing, per-run ownership, direct output residency, publication, close/run leases,
  rollback, suppression, and concurrent-run isolation retain current Runtime/Prepare semantics.
- Metal remains independent of Engine. No shared contract or dependency direction changes.
- Planning documents remain non-authoritative. `ARCHITECTURE.md` and explanatory architecture
  documentation are unchanged.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.metal` — unchanged public capability provider only.
- `io.github.pho001.synaptik.backend.metal.internal` — existing package-private native ABI,
  context/storage owners, NEG analysis/finalization, route-specific persistent resources,
  prepared executable, and sole-partition schedule assembly.

Packages added or changed:

- No package is added. The existing internal package gains one focused resource type. The
  deferred `.internal.prepare` and `.internal.route.mpsgraph` boundaries remain deferred because
  two small routes still share one NEG-specific analysis/finalization/binding lifecycle and no
  reusable cross-package seam has been proved.

Type placement:

- `io.github.pho001.synaptik.backend.metal.internal.MetalNegKernelPipelineResource` — new
  package-private typed owner for exactly one custom NEG compute pipeline and context lease.
- `MetalDeviceContext` — extended in place with the custom pipeline's provisional-child-lease
  create/adopt/rollback operation, parallel to but typed separately from its MPSGraph creation
  operation.
- `MetalNegPreparationPlan` — extended in place with the closed two-value route decision and
  optional MPSGraph workspace; not renamed or split.
- `MetalNegPartitionPreparer` — extended in place because complete partition validation and
  route selection are one backend-analysis responsibility.
- `MetalNegPartitionFinalizer` — extended in place because assignment validation, typed resource
  construction, and local rollback remain one NEG finalization responsibility.
- `MetalNegPreparedExecutable` — extended in place with explicit MPSGraph/custom construction and
  route-specific cold-bound invocation; not generalized.
- `MetalNegPreparedScheduleAssembler` — extended in place only to create the workspace required
  by the selected plan; feed/output/publication ownership remains shared.

The Metal master-plan package map does not change because every affected and new type stays in the
already-defined `.internal` responsibility. No type moves, no visibility widens, and no generic
route package is introduced.

## Affected files

Exact implementation allowlist:

Production and native paths:

1. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNativeApi.java`
2. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalDeviceContext.java`
3. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegKernelPipelineResource.java` (new)
4. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparationPlan.java`
5. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPartitionPreparer.java`
6. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPartitionFinalizer.java`
7. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparedExecutable.java`
8. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparedScheduleAssembler.java`
9. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/package-info.java`
10. `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
11. `native/metal-macos-arm64/README.md`

Test paths:

12. `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/internal/MetalFoundationTest.java`
13. `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparedExecutionTest.java`

Documentation and planning paths:

14. `docs/backend-guide/metal-backend.md`
15. `docs/glossary.md`
16. this task
17. `docs/planning/backends/metal/master-plan.md`
18. `docs/planning/roadmap.md`

Explicitly excluded and unchanged include `MetalCapabilityProvider.java`, its tests, backend
conformance, architecture tests, Engine, shared modules, Gradle files, `ARCHITECTURE.md`, and all
architecture/ADR documents.

## Maximum scope

This task may create or modify at most the exact 18 paths above: nine Metal production/Javadoc
paths, two native documentation/source paths, two existing Metal test paths, two explanatory
documentation paths, and three planning paths.

This cohesive scope is larger than the usual source-file target because the additive ABI,
analysis-time choice, typed persistent owner, finalization, cold binding, real/fake execution
evidence, and documentation must agree before the second route is truthful. It adds only one
production type and reuses the established NEG lifecycle owners. If another path, type, package,
public API, dependency, test module, status, or framework is needed, stop and return this task to
planning rather than widening it silently.

## Acceptance criteria

- `MetalCapabilityProvider.java` is byte-for-byte unchanged; all existing provider and
  capability/partition conformance tests retain their exact meaning and pass.
- The custom route is selected exactly for a complete supported partition with one NEG node, one
  unique feed, one unique target, a checked element count in `1..UINT32_MAX`, and valid native
  direct-dispatch geometry, for both caller and splat feeds. Every other task-0002 supported
  partition selects MPSGraph without repartitioning or fallback.
- Cold analysis tests prove the index-domain boundary without allocating giant buffers: safely
  constructed descriptor/geometry facts at exactly `UINT32_MAX` select `CUSTOM_SINGLE_NEG`, while
  the first representable checked singleton count above it selects `MPSGRAPH`. These tests stop
  before physical representation creation or native allocation.
- Analysis makes the route decision before returning declarations. Custom declares exactly feed
  and target buffers and zero workspaces; MPSGraph preserves its existing declarations and one
  address workspace.
- The plan retains the exact context, feed/target order, Shapes, byte extents, splat facts, and
  route. Finalization rejects route/declaration/assignment mismatches and never changes route.
- ABI version is exactly `3`; the dylib exports exactly thirteen named symbols; all ten ABI-v2
  signatures and semantics remain unchanged; the three additions, output-cell protocol,
  statuses, typed handles, validation order, concurrency, and cleanup match this task.
- Java validates version `3` and resolves all thirteen symbols eagerly in the specified order.
  Malformed native success/failure output cells are cleaned up exactly once and fail closed.
- The native pipeline compiles the fixed branch-free MSL source once during finalization, retains one
  `MTLComputePipelineState`, and does not compile or select a kernel during a run.
- The implementation proves exact non-uniform dispatch support within the `UINT32_MAX` custom
  domain and uses the branch-free `uint` oracle. It has no rounded-grid guarded variant; larger
  supported singletons remain MPSGraph because a bounds guard cannot repair 32-bit index wrap.
- Native custom create defensively returns `UNSUPPORTED_SHAPE` for zero and
  `UINT32_MAX + 1`, even though analysis never sends either count to custom finalization. Fake
  seam tests prove both rejections without allocating buffers.
- Custom cold binding accepts only the exact two compatible live context-local buffers, validates
  extents, rejects Java and native aliasing, and creates a direct typed bound invocation with no
  workspace or address array.
- Custom Java hot execution performs one native downcall with direct typed resource and buffer
  handles and no allocation, boxing, reflection, lookup, address-array marshalling, cast,
  operation dispatch, route branch, retry, fallback, upload, or download.
- Native custom execution binds the assigned input/output `MTLBuffer` values directly, submits
  one compute command, waits once at the route boundary, checks completed status/error, and writes
  the supplied output without an explicit copy or MPSGraph call.
- Caller input remains borrowed. Every splat and output remains fresh and run-owned per run.
  Publication leases the direct output; cleanup and failure never close a caller buffer or leak a
  run-owned/persistent resource.
- The pipeline resource owns exactly one native pipeline handle and one adopted context lease.
  It is idempotent, single-consumption, reusable across admitted runs, and isolated from the
  MPSGraph resource/symbol family.
- Finalizer-local and shared rollback preserve the exact preparation primary failure and attach
  later distinct close failures in cleanup encounter order without self-suppression. Successful
  preparation transfers the one selected persistent resource only to `PreparedExecution`.
- Close/run races retain Runtime 0016 behavior: new runs fail after close begins, admitted runs
  may finish, physical release is deferred when needed, and concurrent runs have isolated mutable
  state and buffers.
- Real-device assertions cover finite values, infinities, NaN sign irrelevance, and signed zero
  (`+0.0 -> -0.0`, `-0.0 -> +0.0`) with raw-bit checks where applicable.
- One prepared custom pipeline executes at least twice without recompilation and with fresh
  outputs. A separate supported multi-node partition executes through MPSGraph, proving regression
  preservation and the second-route boundary.
- Deterministic fake/native-seam tests cover nil/error library, function, pipeline, command
  buffer, encoder, completion status/error, incompatible buffers, aliasing, unknown status,
  Objective-C exception, and malformed create results. No failure retries or switches routes.
- Source and compiled-shape inspection confirms the MSL oracle, one compile site, typed handle
  separation, route-specific bound invocations, exact one-downcall custom hot path, and absence
  of a generic route framework or new public Metal type.
- Public Engine composition remains CPU-only; no Engine, mixed-owner, CPU fallback, public
  storage, or standard-compute claim appears in code or documentation.
- The Metal guide and native README clearly distinguish the custom singleton route from the
  retained whole-partition MPSGraph route, describe ABI version `3`, and make no performance
  superiority claim.
- Glossary review updates only reusable project terminology whose meaning changes; otherwise it
  records a reasoned no-change conclusion in the task evidence.
- A separate clean documentation-focused agent pass finalizes affected Javadocs, explanatory
  documentation, glossary impact, task evidence, master plan, and roadmap in the same change.
- Focused/final Java tests, real-device execution, native build/export/framework checks,
  Javadoc, Markdown links/anchors/fences/newlines/whitespace, exact 18-path scope, package/type
  placement, status/order synchronization, no-staged-files check, and `git diff --check` pass.

## Performance constraints and inspection

- Route choice, element-count calculation, source choice, source compilation, function lookup,
  pipeline construction, threadgroup geometry selection, and persistent acquisition occur before
  runtime execution.
- The MSL source is inspected against the branch-free clean oracle above. No bounds guard or
  other algorithmic deviation is accepted inside the `UINT32_MAX` custom domain.
- Native source inspection confirms exactly one element load, one unary negation, and one element
  store per valid custom thread, with no temporary buffer or second dispatch.
- Java source plus `javap -c -p` inspection confirms a route-specific custom bound invocation
  whose `executeBound()` directly calls the typed resource and whose resource directly invokes
  the custom native run handle once.
- No latency, throughput, occupancy, bandwidth, power, or superiority threshold is required.
  The heuristic is intentionally unmeasured until Metal 0004.

## Tests / validation

Implementation development may run focused test methods. After Java/native behavior stabilizes,
the implementation context runs one focused class and one final ordinary Metal suite:

```bash
./gradlew :backends:metal:test \
  --tests '*MetalNegPreparedExecutionTest'
./gradlew :backends:metal:test
```

The focused tests must use the injected fake native seam to prove both route branches,
declaration timing, assignment validation, exact status mapping, typed-handle separation,
resource rollback, cold binding, hot-call counts, caller/splat ownership, direct destination,
reuse, and concurrent-run isolation. Cold analysis-only fixtures must prove exactly
`UINT32_MAX -> CUSTOM_SINGLE_NEG` and `UINT32_MAX + 1 -> MPSGRAPH` from safely constructed static
descriptor/geometry facts without creating physical buffers. The native fake seam must separately
prove custom create returns `UNSUPPORTED_SHAPE` for zero and `UINT32_MAX + 1`. Existing
capability/provider tests are part of the final module suite and must pass without source changes.

Build and inspect the native library:

```bash
./native/metal-macos-arm64/build.sh
file native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
otool -L native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
```

`nm` must show exactly the thirteen specified `synaptik_metal_*` exports. `otool` must continue
to show Foundation, Metal, and MetalPerformanceShadersGraph; no new framework is required. The
build script should remain unchanged unless exact build evidence contradicts this plan, in which
case stop because it is outside the allowlist.

On the current eligible macOS-arm64 host, run the exact existing foundation case and the expanded
prepared execution case with the freshly built library:

```bash
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test \
  --tests '*MetalFoundationTest.nativeFoundationRoundTrip' \
  --tests '*MetalNegPreparedExecutionTest.nativePreparedNegRoundTripAndReuse'
```

The prepared native case must prove custom caller and splat execution/reuse/direct destinations
and a multi-node MPSGraph regression. The task cannot become Complete by skipping it on this
eligible host. If the sandbox hides the Metal device, rerun the exact command with approved
unsandboxed device access and record both outcomes. If custom pipeline compilation, safe
dispatch, assigned output, or exact NEG semantics cannot be proved, keep task 0003 incomplete and
return it to planning; do not change capability or silently use MPSGraph for the exact case.

No backend-conformance test is rerun solely for this task because the provider and partitioning
behavior do not change. No integration test applies because supported Engine composition remains
CPU-only. No architecture test applies because no dependency/build/module boundary changes. If
any such path becomes necessary, stop and replan before editing it.

Inspect source and compiled Java shape:

```bash
javap -classpath backends/metal/build/classes/java/main -c -p \
  io.github.pho001.synaptik.backend.metal.internal.MetalNegPreparedExecutable \
  io.github.pho001.synaptik.backend.metal.internal.MetalNegKernelPipelineResource
```

Manually confirm the exact MSL source, branch/guard decision, one pipeline compile site, direct
buffer indices `0` and `1`, one dispatch, one completion wait, one custom Java downcall, no hot
route branch/allocation/marshalling, no cross-family handle use, no public Metal type, and no
Engine import. Recurring semantic and lifecycle invariants belong in automated tests; this manual
inspection addresses generated/native/hot-path shape that ordinary assertions cannot fully prove.

Documentation-focused pass, after Javadocs and explanatory text stabilize:

```bash
./gradlew :backends:metal:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  native/metal-macos-arm64/README.md \
  docs/backend-guide/metal-backend.md docs/glossary.md \
  docs/planning/backends/metal/tasks/0003-single-neg-custom-metal-kernel-route.md \
  docs/planning/backends/metal/master-plan.md docs/planning/roadmap.md
git diff --check
git diff --cached --check
git status --short -uall
```

If the temporary validator is absent, use an equivalent tool outside the repository. Validate
local links and heading anchors, unique effective anchors, balanced fences, final newlines,
  trailing whitespace, exact 18-path implementation scope, package/type placement, ABI version,
status values, exact symbols, capability-source exclusion, task/master/roadmap status and order,
no detailed Metal 0004 task, and no staged files.

Repository-wide validation is deferred to the final Metal checkpoint or continuous integration.
The implementation is one backend-local behavior change with no capability, shared contract,
production edge, Gradle, or architecture change. The documentation pass reuses stabilized Java,
native, and real-device evidence and does not repeat successful Java suites unless it changes
executable behavior or records a concrete stale-evidence risk.

## Dependencies

- Metal 0001 — Complete; native context, buffer ownership, host byte access, child leases, and
  foundation statuses.
- Metal 0002 — Complete; unchanged NEG capability, whole-partition MPSGraph route, stable
  feed/target analysis, per-run splat/output/address ownership, ABI version `2`, and fake/real
  test seams to extend.
- Runtime 0016 — Complete; persistent-resource ownership and close/run leases.
- Prepare 0006 — Complete; finalizer-local responsibility, transactional rollback, and ownership
  transfer.
- Engine 0010 — Complete lifecycle prerequisite, but no Engine source is in scope.
- Compiler 0006B7 — Complete; final descriptor closure required by existing truthful capability.
- JDK 26 FFM and installed macOS-arm64 Foundation, Metal, and
  MetalPerformanceShadersGraph frameworks.

No Model 0026, tuning, cache, public composition, new capability, or shared-contract dependency is
introduced.

## Follow-up tasks

- Metal 0004 remains a concise Draft master-plan row. It will own typed complete MPSGraph/custom
  route candidates, candidate/cache schema compatibility, target/workload fingerprints, safe
  heuristic evolution, and tuning integration. This task must not create its detailed spec.
- Broader operations, pointwise fusion, mixed precision, public Metal composition, mixed-owner
  schedules, pooling, packaging, and asynchronous execution remain separately planned future
  work and do not weaken this task's acceptance criteria.

## Architecture impact

Expected impact: None.

This task implements custom-kernel route selection, compilation, persistent ownership, and
execution already assigned to the Metal backend. It changes no architecture authority, ownership
boundary, dependency direction, shared API, Planning meaning, Runtime meaning, or public surface.
If implementation requires any such change, stop and report the exact conflict instead of editing
architecture or expanding this task.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik Metal task 0003. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/metal/master-plan.md, and
docs/planning/backends/metal/tasks/0003-single-neg-custom-metal-kernel-route.md in full. Read
Metal tasks 0001–0002, ADRs 0002 and 0013, the Metal strategy, current Metal Java/native/tests,
directly affected shared contracts, documentation rules/profiles, and installed SDK Metal
headers needed to verify the exact ABI, pipeline, dispatch, and completion contracts.

Implement exactly the task specification and its 18-path allowlist. Preserve the byte-for-byte
unchanged capability, exact singleton-plus-UINT32_MAX custom implementation domain, MPSGraph route
for every other supported NEG partition, analysis-before-declaration choice, direct typed custom
handle, branch-free uint MSL oracle with exact dispatchThreads, ABI version 3,
persistent/per-run ownership, rollback/suppression, direct output, one-wait/one-downcall hot path,
and fail-closed behavior.
Do not add Engine work, mixed-owner composition, CPU fallback, another operation/type/layout,
generic route abstractions, async APIs, pooling, metallib packaging, tuning, or Metal 0004 detail.
Stop for an architecture, shared-contract, package, scope, ABI, target-support, index/dispatch, or
correctness conflict rather than inventing a boundary or silently changing capability.

Run and record the focused, final, native, real-device, ABI/framework, source/javap, and scope
validation. Then hand the stabilized diff and exact evidence to a distinct clean
documentation-focused agent/thread. That pass must follow documentation-rules.md, apply the
General/API-Javadoc/Backend-Guide/Planning profiles, finalize affected Javadocs/docs/glossary and
planning evidence, validate links/anchors/fences/newlines/whitespace and exact scope, and avoid
repeating stable Java tests unless executable behavior changes or a concrete risk requires it.
Mark task 0003 Complete only after every acceptance gate and the documentation pass succeed.
```

## Stop conditions

Stop and return the task to planning if:

- analysis cannot keep every singleton above `UINT32_MAX` on MPSGraph before declarations, or
  custom native create does not reject zero and greater-than-`UINT32_MAX` counts fail closed;
- the target cannot compile/reuse the fixed MSL pipeline or write the supplied output directly;
- custom execution requires a workspace, address-array marshalling, per-element allocation,
  hidden host copy, multiple native calls, multiple synchronization waits, or hot route dispatch;
- analysis cannot select the route and exact requirements before shared assignment;
- finalization cannot return exactly one typed persistent resource with existing transactional
  ownership and rollback semantics;
- implementation needs to change capability, repartition, add fallback, reinterpret an MPSGraph
  handle, add a generic executable handle/route framework, or cross a package/public boundary;
- Engine/shared production/Gradle/architecture/conformance/integration source or a nineteenth
  path is required; or
- the eligible real-device test cannot prove custom caller/splat execution, direct output,
  pipeline reuse, exact NEG semantics, and retained MPSGraph multi-node execution.

## Local decisions

- Extend the existing NEG-specific plan, preparer, finalizer, prepared executable, and schedule
  assembler in place. Their responsibilities remain cohesive; splitting or renaming them would
  add cross-type plumbing without a reusable second operation or route family.
- Add only `MetalNegKernelPipelineResource` as a new production type so native custom-pipeline
  ownership cannot be confused with `MetalMpsGraphExecutableResource`.
- Use a closed two-value plan discriminator rather than a generic route abstraction. It is an
  implementation fact, not a public API, tuning schema, or cache key.
- Keep splat materialization unchanged and per-run. The custom route changes only persistent
  executable state and the absence of MPSGraph address-array workspace.
- Restrict the custom implementation domain to checked positive counts no greater than
  `UINT32_MAX`, then use exact non-uniform `dispatchThreads` and the branch-free `uint` MSL oracle.
  A guard is not an alternative for larger counts because it cannot repair index wrap; those
  supported singletons remain MPSGraph.
- Reuse statuses `1`, `4`, `7`, `8`, `10`, and `11` for their existing meanings and add only
  status `12` for custom MSL/function/pipeline/target compilation failure.
- Do not modify the master-plan package map because the one new type fits its existing internal
  native/resource responsibility.

## Known limitations

- The custom route covers only an exact one-node/one-feed/one-target partition with checked
  element count in `1..UINT32_MAX` inside the unchanged `FLOAT32` NEG capability domain. Every
  larger/multi-boundary partition and every supported singleton above that count remains
  MPSGraph.
- The selection heuristic is deterministic but unmeasured. This task makes no speed, latency,
  bandwidth, occupancy, or energy claim.
- Execution remains synchronous and may serialize native use through the persistent resource.
  No asynchronous or concurrent-overlap guarantee is made.
- Current standard Engine composition remains CPU-only; the route is proved through the existing
  backend-local typed preparation/runtime integration.
- MSL is compiled from fixed source during each preparation finalization. There is no metallib,
  pipeline cache, binary archive, serialization, package, discovery, or tuning integration.
- All task-0002 type/Shape/layout/public-composition and platform limitations remain.

## Validation evidence

The implementation context recorded these stabilized results, which the distinct documentation
context `/root/metal_0003_docs` reused without repeating successful Java suites:

- focused `MetalFoundationTest`: 14 total, 13 passed, one expected native opt-in skip;
- final ordinary Metal suite: 45 total, 42 passed, three expected native opt-in skips, no failures
  or errors;
- earlier focused `MetalNegPreparedExecutionTest`: 28 total, 26 passed, two expected native
  opt-in skips before the final test-only/native mapping refinement; the final full suite
  supersedes this focused result;
- native dylib build passed; `file` reported Mach-O arm64; `nm` reported exactly thirteen
  `synaptik_metal_*` exports; `otool` reported the expected Foundation, Metal, and
  MetalPerformanceShadersGraph linkage;
- both `nativeFoundationRoundTrip` and `nativePreparedNegRoundTripAndReuse` passed on the real
  device against the final rebuilt dylib;
- generated Javadoc and source/`javap` hot-shape inspection passed before the documentation pass;
  the inspection confirmed the route-specific bound invocation, one custom native call, typed
  pipeline ownership, fixed branch-free MSL, direct buffer indices, one dispatch, and one wait;
- final source-contract assertions cover every specified native fail-closed create, release, and
  run branch and the `UNSUPPORTED_SHAPE`, `KERNEL_COMPILATION_FAILED`, and `EXECUTION_FAILED`
  mappings; and
- implementation review blockers were fixed, independent re-review reported no blocking
  findings, `git diff` checks passed, and the staged path count remained zero.

The documentation pass ran the task-prescribed Metal Javadoc, Markdown link/anchor/fence/final-
newline/whitespace checks, exact-path and no-detailed-0004 checks, `git diff --check`, and
`git diff --cached --check`. Its final command results are recorded in the completion summary.
No performance benchmark was run or required.

## Implementation notes

Analysis now selects `CUSTOM_SINGLE_NEG` exactly for one NEG node, one feed, one target, and a
checked positive count no greater than `UINT32_MAX`; it declares two buffers and no workspace.
The first larger supported singleton and all other supported NEG partitions select `MPSGRAPH`
with the existing address workspace. The unchanged capability provider remains occurrence-level,
and neither route selection nor a route failure changes ownership, retries, or repartitions work.

ABI version `3` eagerly binds exactly thirteen symbols and keeps the MPSGraph and custom handle
families separate. Custom creation receives `element_count` through a `uint64_t` carrier but
accepts only `1..UINT32_MAX`. The fixed MSL entry uses `uint thread_position_in_grid` and contains
one load, unary negation, and store with no bounds branch. Native creation validates grid
representation separately from threadgroup usability: the former is unsupported shape, while the
latter is execution failure. Source/library/function/pipeline and target-family proof failures use
the new kernel-compilation status `12`.

Finalization compiles exactly one selected typed persistent resource. Custom cold binding retains
the input and assigned output buffers directly; hot execution makes one typed native call. Native
execution uses one command buffer, one compute encoder, one exact dispatch, and one completion
wait, writing the assigned output `MTLBuffer` without an explicit host-staging or intermediate-
copy step. Caller inputs remain borrowed, while splats and outputs remain fresh and run-owned.
Context child leases, concurrent admitted-run isolation, deferred close, malformed create-output
cleanup, status preservation, and finalizer/shared rollback remain aligned with the established
Runtime and Prepare ownership contracts.

## Completion summary

Completed changes:

- Added the exact singleton/count route boundary before shared declarations, retaining MPSGraph
  for every other supported partition and leaving `MetalCapabilityProvider` unchanged.
- Added private ABI-v3 custom-pipeline create/release/run operations, fixed branch-free FLOAT32
  NEG MSL, typed persistent ownership, direct assigned-output execution, and fail-closed status
  and malformed-output handling.
- Preserved caller borrowing, per-run splat/output ownership, isolated admitted runs, deferred
  close, context child leases, deterministic rollback, and separate MPSGraph/custom handle
  families.
- Finalized affected Javadocs, the native README, Metal backend guide, glossary terminology, this
  task, the Metal master plan, and the roadmap in clean documentation context
  `/root/metal_0003_docs`.

Exact files changed or created:

1. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNativeApi.java`
2. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalDeviceContext.java`
3. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegKernelPipelineResource.java`
4. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparationPlan.java`
5. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPartitionPreparer.java`
6. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPartitionFinalizer.java`
7. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparedExecutable.java`
8. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparedScheduleAssembler.java`
9. `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/package-info.java`
10. `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/internal/MetalFoundationTest.java`
11. `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparedExecutionTest.java`
12. `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
13. `native/metal-macos-arm64/README.md`
14. `docs/backend-guide/metal-backend.md`
15. `docs/glossary.md`
16. `docs/planning/backends/metal/tasks/0003-single-neg-custom-metal-kernel-route.md`
17. `docs/planning/backends/metal/master-plan.md`
18. `docs/planning/roadmap.md`

Validation:

- Focused `MetalFoundationTest`: 14 total, 13 passed, one expected native opt-in skip.
- Final Metal suite: 45 total, 42 passed, three expected native opt-in skips, no failures or
  errors. The earlier 28-total focused prepared-execution result was superseded as recorded above.
- Native build, Mach-O arm64 inspection, exact thirteen-export inspection, expected framework
  linkage, both final-dylib real-device cases, source-contract assertions, generated hot-shape
  inspection, and independent implementation re-review all passed.
- Documentation validation passed `./gradlew :backends:metal:javadoc`, the targeted six-file
  Markdown validator, final newline/whitespace checks, `git diff --check`,
  `git diff --cached --check`, exact 18-path scope, zero staged paths, and the no-detailed-0004
  check.

No-change conclusions:

- `ARCHITECTURE.md`, focused architecture explanations, and ADRs remain accurate because route
  choice, lowering, and persistent ownership stay inside the existing Metal Prepare boundary.
- Capability, backend conformance, integration, public Engine, shared modules, Gradle, build
  structure, and native build script require no change. The provider and public API do not expand,
  and standard Engine composition remains CPU-only.
- The glossary required a targeted change because its physical-representation and MPSGraph-only
  prepared-executable text was stale; no new cross-project abstraction or public term was added.
- Existing Javadocs that were not changed remain accurate after review. The affected route, ABI,
  resource, lifecycle, and direct-output contracts were finalized in the already changed
  production files.

Unresolved issues: none for task 0003.

Follow-up: Metal 0004 remains Draft, without a detailed task, for typed route candidates, target
and cache compatibility, and tuning integration. Broader kernels, types, public Metal Engine
composition, packaging, and asynchronous execution remain separately planned.

Status: Complete
