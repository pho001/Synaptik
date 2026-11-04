# Task 0003: Single-NEG Custom [Metal](../../../../glossary.md#term-metal) [Kernel](../../../../glossary.md#kernel) Route

## Reader context

The second route changed how an already-supported negation ran, not which backend owned it.
Metal analysis selected either its own compiled shader pipeline or MPSGraph before declaring
resources. A larger eligible negation choosing MPSGraph was planned selection, not recovery after
a failed custom invocation. The unsigned grid-index limit explained the custom route's bound.

The prepared execution owned the reusable pipeline; Runtime borrowed input buffers and owned
the run's outputs. Only the MPSGraph alternative needed the address workspace. Shapes and element
counts selected the route, not the input's numerical contents.
The [current route-selection explanation](../../../../explanation/packages/metal.md#2-lower-once-declare-resources-then-compile)
places this singleton optimization beside whole-program routes. Nothing in the historical
completion record below is a measured speed comparison.

## Status

Complete

## Goal

Prove a second [backend](../../../../glossary.md#backend)-private [Metal](../../../../glossary.md#term-metal) execution route without changing Metal [occurrence](../../../../glossary.md#term-occurrence)
capability. A whole Metal [partition](../../../../glossary.md#partition) uses a direct custom Metal compute [kernel](../../../../glossary.md#kernel) only when it has
exactly one supported [unary](../../../../glossary.md#term-unary) `NEG` occurrence, one unique feed, one unique [target](../../../../glossary.md#term-target), and a checked
positive [element](../../../../glossary.md#term-element) count no greater than `UINT32_MAX`. The feed may be either a caller input or the
already-supported positive-[rank](../../../../glossary.md#term-rank) [`FLOAT32`](../../../../glossary.md#term-float32) splat represented by an `InitializedBuffer`. Every
other currently supported Metal NEG partition—including an otherwise matching [singleton](../../../../glossary.md#term-singleton) above
that custom 32-bit index domain—continues to use the task-0002 [Metal Performance Shaders Graph](../../../../glossary.md#mps-mpsgraph-and-openblas)
([`MPSGraph`](../../../../glossary.md#term-mpsgraph)) route.

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
[owner](../../../../glossary.md#term-owner) of complete typed route candidates, target/[cache](../../../../glossary.md#term-cache) compatibility, and tuning integration.

## Pre-implementation audit

### Current repository surface

- `MetalCapabilityProvider` admits exactly positive fully static [rank](../../../../glossary.md#term-rank)-`1..16`, resolved
  dense-contiguous, non-view, zero-offset <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [view glossary](../../../../glossary.md#term-view-meanings))</span>[`FLOAT32`](../../../../glossary.md#term-float32) [unary](../../../../glossary.md#term-unary) `NEG` occurrences with equal input and
  output Shapes and equal <span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape))</span>`requiresGrad` flags. Task 0003 does not modify this file.
- Planning asks capability per occurrence and then creates maximal same-owner partitions. It
  neither sees nor selects MPSGraph or a custom kernel.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [occurrence glossary](../../../../glossary.md#term-occurrence); [owner glossary](../../../../glossary.md#term-owner); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [kernel glossary](../../../../glossary.md#kernel))</span>
- `MetalNegPartitionPreparer` currently validates the complete supported NEG partition, derives
  stable unique feeds and targets, declares every boundary buffer, and always declares one
  MPSGraph address workspace.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [targets glossary](../../../../glossary.md#term-target); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- `MetalNegPreparationPlan` currently retains the complete stable MPSGraph lowering arrays plus
  feed/target facts and the address-workspace declaration. Those facts already cover the exact
  singleton case and checked element geometry; no new graph abstraction is needed.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [singleton glossary](../../../../glossary.md#term-singleton); [element glossary](../../../../glossary.md#term-element); [graph glossary](../../../../glossary.md#graph))</span>
- `MetalNegPartitionFinalizer` currently validates assignments and constructs one
  `MetalMpsGraphExecutableResource`. `MetalNegPreparedExecutable` then cold-marshals feed/target
  addresses into a run-owned workspace before its one hot native call.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `MetalNegPreparedScheduleAssembler` already distinguishes caller feeds from exact `FLOAT32`
  splats and preserves caller borrowing, per-run splat/output ownership, publication order, and
  rollback. The custom route needs the same buffer recipes but no address-array workspace.<span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [publication glossary](../../../../glossary.md#term-publication); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) and shared Prepare already provide the required identity-unique persistent
  resource ownership, transactional finalization handoff, reverse cleanup, suppression rules,
  close/run leases, and concurrent-run isolation.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [identity glossary](../../../../glossary.md#term-identity-meanings); [handoff glossary](../../../../glossary.md#term-handoff); [leases glossary](../../../../glossary.md#term-lease))</span>
- At task 0003's completion frontier, the public [Metal](../../../../glossary.md#term-metal) surface was only
  `MetalCapabilityProvider`, and public Engine composition was CPU-only. Engine task 0015 later
  added public Metal configuration/integration plus explicit single-owner Metal Engine execution;
  it did not change this task's route, capability, or mixed-owner exclusions.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu))</span>

The audit therefore selects an in-place extension of the existing NEG-specific owners. It does
not rename or split the preparer, finalizer, [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable), or schedule assembler, and it
does not create a generic Metal route, executable-handle, or pipeline [registry](../../../../glossary.md#term-registry).

### Installed [SDK](../../../../glossary.md#term-sdk) evidence

The [planning](../../../../glossary.md#term-planning) audit used the installed macOS SDK at
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk` on the current [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) macOS [host](../../../../glossary.md#term-host). The relevant
[Metal](../../../../glossary.md#term-metal) contracts are:

- `MTLDevice.newLibraryWithSource:options:error:` synchronously compiles Metal Shading Language
  (<span data-glossary-navigation="true"> ([Metal Shading Language glossary](../../../../glossary.md#device-submission-synchronization-and-execution-context))</span>[`MSL`](../../../../glossary.md#term-metal-shading-language-msl)) source into an `MTLLibrary` and reports failure with `NSError` or a nil result;
- `MTLLibrary.newFunctionWithName:` resolves one named compute function;
- `MTLDevice.newComputePipelineStateWithFunction:error:` synchronously creates a reusable
  `MTLComputePipelineState` and reports failure with `NSError` or a nil result;
- `MTLComputePipelineState.threadExecutionWidth` and
  `maxTotalThreadsPerThreadgroup` provide the legal one-dimensional [threadgroup](../../../../glossary.md#term-threadgroup) bound;
- `MTLCommandQueue.commandBuffer`, `MTLCommandBuffer.computeCommandEncoder`,
  `setComputePipelineState:`, `setBuffer:offset:atIndex:`, `endEncoding`, [`commit`](../../../../glossary.md#term-commit), and
  `waitUntilCompleted` provide the direct [synchronous](../../../../glossary.md#term-synchronous) compute path;
- `dispatchThreads:threadsPerThreadgroup:` accepts an arbitrarily sized grid and non-uniform
  final threadgroup on supported targets, so an exact grid contains the static element count and
  does not require a rounded-grid bounds branch; and<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- after `waitUntilCompleted`, `MTLCommandBuffer.status` must be
  `MTLCommandBufferStatusCompleted` and `error` must be nil for success.

Apple's official compute examples use a 32-bit unsigned `thread_position_in_grid` index. The
unchanged Metal capability accepts checked positive [`FLOAT32`](../../../../glossary.md#term-float32) [element](../../../../glossary.md#term-element) counts beyond
`UINT32_MAX`, so [topology](../../../../glossary.md#term-topology) alone cannot select this [kernel](../../../../glossary.md#kernel) truthfully. Task 0003 therefore treats
`elementCount <= UINT32_MAX` as a Metal-private implementation-domain predicate during analysis.
It is not a capability predicate: an otherwise supported [singleton](../../../../glossary.md#term-singleton) above the limit stays on
[MPSGraph](../../../../glossary.md#term-mpsgraph). Within the custom domain, exact non-uniform `dispatchThreads` keeps the branch-free
[oracle](../../../../glossary.md#term-test-oracle) valid. A bounds [guard](../../../../glossary.md#term-guard) would not repair 32-bit index wrap above `UINT32_MAX` and is not the
selected design. No [throughput](../../../../glossary.md#term-throughput) or superiority conclusion follows from these [API](../../../../glossary.md#term-application-programming-interface-api) contracts.

## [Scope](../../../../glossary.md#term-scope)

- Keep `MetalCapabilityProvider.java` byte-for-byte unchanged. Capability remains exactly the
  task-0002 domain and remains independent of route, native availability, source provenance, and
  partition topology.<span data-glossary-navigation="true"> ([source provenance glossary](../../../../glossary.md#term-source-provenance); [topology glossary](../../../../glossary.md#term-topology))</span>
- Extend `MetalNegPartitionPreparer` to select one of exactly two NEG-specific private routes
  before constructing [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis):
  - `CUSTOM_SINGLE_NEG` when the complete partition has exactly one NEG node, exactly one unique
    feed, exactly one unique target, a checked positive element count no greater than<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [target glossary](../../../../glossary.md#term-target); [element glossary](../../../../glossary.md#term-element))</span>
    `UINT32_MAX`, and valid native `MTLSize`/`NSUInteger`/pipeline geometry facts; or
  - [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph) for every other [partition](../../../../glossary.md#partition) already accepted by task 0002.
- Treat caller and exact positive-[rank](../../../../glossary.md#term-rank) [`FLOAT32`](../../../../glossary.md#term-float32) splat feeds identically for route eligibility.
  Source provenance changes only the existing schedule buffer recipe, not capability or route
  selection.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- Extend `MetalNegPreparationPlan` with one package-private closed route discriminator and an
  optional address-workspace declaration. Preserve all stable feed, target, descriptor, geometry,
  splat, and MPSGraph arrays already owned by this plan. The custom route has exactly two buffer
  declarations and no workspace declaration; the MPSGraph route retains its existing boundary
  declarations and exactly one address workspace.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Keep `MetalNegPartitionPreparer`, `MetalNegPartitionFinalizer`,
  `MetalNegPreparedExecutable`, and `MetalNegPreparedScheduleAssembler` under their current names
  and responsibilities. Extend them with explicit two-route branches at analysis, finalization,
  and cold binding. Do not introduce a generic route interface, route registry, generic
  executable handle, string dispatch, reflection, or parameter bag.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Add one package-private `MetalNegKernelPipelineResource` implementing [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource). It
  owns exactly one opaque custom-NEG native pipeline handle and one adopted context child lease.
  It is not an MPSGraph handle and is never passed to an MPSGraph release or run function.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [lease glossary](../../../../glossary.md#term-lease); [release glossary](../../../../glossary.md#term-release); [run glossary](../../../../glossary.md#term-run))</span>
- During custom-route finalization, [compile](../../../../glossary.md#compile) the exact direct `FLOAT32` [MSL](../../../../glossary.md#term-metal-shading-language-msl) [kernel](../../../../glossary.md#kernel) and reusable
  `MTLComputePipelineState` once, after shared buffer assignment. Return the typed pipeline
  resource exactly once in acquisition order.
- Preserve the existing MPSGraph finalization path and `MetalMpsGraphExecutableResource`
  semantics unchanged for every supported partition outside the custom implementation domain,
  including an otherwise matching singleton whose element count is greater than <span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>`UINT32_MAX`.
- Construct a route-specific `MetalNegPreparedExecutable` recipe:
  - the MPSGraph form retains the existing feed/target selections plus one address-workspace
    selection and cold address marshalling; and<span data-glossary-navigation="true"> ([marshalling glossary](../../../../glossary.md#term-marshalling))</span>
  - the custom form has exactly one read-only buffer selection, one write-only buffer selection,
    no workspace selection, and a cold-bound invocation retaining direct typed references to the
    custom pipeline resource and the two compatible <span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>`MetalBufferRepresentation` values.
- Extend the schedule assembler only enough to omit the address workspace for the custom route.
  Preserve the existing caller input, per-run initialized splat, fresh per-run output, execution,
  and publication recipes.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- Evolve the same [native library](../../../../glossary.md#term-native-library) additively to exact [ABI](../../../../glossary.md#term-application-binary-interface-abi) version `3` as defined below. Preserve
  all ten ABI-v2 symbols and their signatures and semantics.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- Add deterministic fake-native tests for route selection, exact declarations, both finalization
  branches, typed-handle separation, rollback and suppression, cold binding, one-downcall hot
  execution, status mapping, and MPSGraph regression.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [downcall glossary](../../../../glossary.md#term-downcall); [regression glossary](../../../../glossary.md#term-regression))</span>
- Add real-device execution that proves the custom route for both caller and splat feeds, direct
  assigned output, raw <span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>`FLOAT32` NEG semantics, pipeline reuse, and per-run isolation, plus an
  explicit multi-node case that proves MPSGraph remains selected.
- Finalize affected Javadocs, the Metal backend guide, native README, glossary impact, and
  planning evidence through the mandatory separate clean documentation-focused pass.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>

## Exact route boundary

### Capability and [partition](../../../../glossary.md#partition) closure

The capability predicate remains byte-for-byte semantically unchanged:

1. [operation](../../../../glossary.md#operation) kind is [unary](../../../../glossary.md#term-unary) `NEG` with `NoOperationAttrs.INSTANCE`;
2. there is exactly one input and one output [descriptor](../../../../glossary.md#term-descriptor);
3. both data types are [`FLOAT32`](../../../../glossary.md#term-float32);
4. both [Shapes](../../../../glossary.md#shape) are fully static, equal, [rank](../../../../glossary.md#term-rank) `1..16`, and strictly positive in every [dimension](../../../../glossary.md#dimension);
5. checked [element](../../../../glossary.md#term-element) and byte geometry succeeds;
6. both [layouts](../../../../glossary.md#layout) are resolved dense-[contiguous](../../../../glossary.md#term-contiguous), non-[view](../../../../glossary.md#term-view-meanings), and zero-offset; and
7. both `requiresGrad` flags are equal, with either shared value accepted.

Task 0003 adds no operation, type, Shape, layout, [gradient](../../../../glossary.md#term-gradient), availability, or [device](../../../../glossary.md#term-device) predicate to
the provider. Existing provider and [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) tests must pass unchanged.

Per-[occurrence](../../../../glossary.md#term-occurrence) capability plus maximal same-[owner](../../../../glossary.md#term-owner) partitioning is the reason `ADD` and every
other new operation are excluded. Advertising `ADD` would allow [Planning](../../../../glossary.md#term-planning) to form arbitrary
maximal [Metal](../../../../glossary.md#term-metal) partitions interleaving supported NEG and ADD occurrences. A truthful capability
change would therefore require complete mixed-partition [lowering](../../../../glossary.md#lowering), not merely one [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [shader](../../../../glossary.md#term-shader).
This task instead proves a second implementation route inside the already-closed NEG partition
domain.

### Analysis-time route selection

Analysis first performs the existing complete [partition](../../../../glossary.md#partition) [validation](../../../../glossary.md#term-validation) and derives the existing
stable value/feed/[target](../../../../glossary.md#term-target) facts. It then applies this deterministic initial [heuristic](../../../../glossary.md#term-heuristic):

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
claim. It runs before resource declarations are finalized. No compatible [cache](../../../../glossary.md#term-cache), benchmark,
[runtime](../../../../glossary.md#term-runtime) observation, [buffer](../../../../glossary.md#term-buffer) address, current [residency](../../../../glossary.md#residency), or caller/splat distinction participates.
The [element](../../../../glossary.md#term-element)-count and native-geometry checks are route-selection implementation-domain
predicates. They do not alter capability, request repartitioning, constitute fallback, or compare
performance. [Metal](../../../../glossary.md#term-metal) 0004 may later replace this fixed choice with typed complete candidates and
compatible decisions without changing [Planning](../../../../glossary.md#term-planning) [ownership](../../../../glossary.md#term-ownership-meanings).

The custom plan declares the feed and target buffers only. The [MPSGraph](../../../../glossary.md#term-mpsgraph) plan preserves the
existing feed-then-target buffers plus one exact address workspace. Finalization must reject a
route/declaration mismatch; it may not change the selected route, add a workspace, or fall back
from custom to MPSGraph after assignment.

## Direct [MSL](../../../../glossary.md#term-metal-shading-language-msl) [oracle](../../../../glossary.md#term-test-oracle) and [dispatch](../../../../glossary.md#term-dispatch)

The clean optimal source oracle is exactly one thread per logical [element](../../../../glossary.md#term-element):

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
not add another [operation](../../../../glossary.md#operation), bounds branch, temporary [buffer](../../../../glossary.md#term-buffer), [scalar](../../../../glossary.md#term-scalar) loop, [vector](../../../../glossary.md#term-vector)-width promise,
[lookup](../../../../glossary.md#term-lookup), or hidden copy.

Preferred dispatch uses `dispatchThreads` with a one-dimensional grid width equal to the checked
static element count and a legal positive [threadgroup](../../../../glossary.md#term-threadgroup) width derived from
`threadExecutionWidth` and `maxTotalThreadsPerThreadgroup`. On a [target](../../../../glossary.md#term-target) proven to support
non-uniform threadgroups, [Metal](../../../../glossary.md#term-metal) launches exactly that grid, including a possibly smaller last
threadgroup, so no bounds branch is present. Analysis never selects the custom route above
`UINT32_MAX`; those supported [singleton](../../../../glossary.md#term-singleton) partitions remain [MPSGraph](../../../../glossary.md#term-mpsgraph). Native create independently
rejects zero or greater-than-`UINT32_MAX` element counts with `UNSUPPORTED_SHAPE`, then validates
all `MTLSize`, `NSUInteger`, pipeline-width, and exact-dispatch geometry before publishing a
pipeline. If exact non-uniform dispatch cannot be proved for a target, analysis must select
MPSGraph for that [partition](../../../../glossary.md#partition) or preparation must [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) before pipeline [publication](../../../../glossary.md#term-publication); a
rounded guarded [kernel](../../../../glossary.md#kernel) is not an alternative because it cannot repair 32-bit index wrap.

## Exact native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 3

ABI version is the unsigned 32-bit value `3`. The version function keeps its name. All six
foundation functions and all three [MPSGraph](../../../../glossary.md#term-mpsgraph) functions keep their ABI-v2 names, [signatures](../../../../glossary.md#term-signature),
[ownership](../../../../glossary.md#term-ownership-meanings), [validation](../../../../glossary.md#term-validation) order, statuses, and behavior. Version `3` adds exactly these three typed
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

The [dylib](../../../../glossary.md#term-dylib) therefore exports exactly thirteen `synaptik_metal_*` symbols. Java validates version
`3`, then eagerly resolves all thirteen symbols in this order before context creation:

1. version;
2. the existing six context/[buffer](../../../../glossary.md#term-buffer) functions in their ABI-v1 order;
3. the existing three MPSGraph functions in their ABI-v2 order; and
4. custom pipeline create, [release](../../../../glossary.md#term-release), then [run](../../../../glossary.md#term-run).

There is no optional [symbol](../../../../glossary.md#term-symbol), [alias](../../../../glossary.md#aliasing-and-arena), [lazy](../../../../glossary.md#term-lazy-evaluation) route [binding](../../../../glossary.md#term-binding), generic executable create/release/run
symbol, function-name [parameter](../../../../glossary.md#term-parameter), source parameter, raw pipeline accessor, or ABI fallback.

### Statuses and [fail-closed](../../../../glossary.md#term-fail-closed) mapping

Statuses `0..11` retain their exact [ABI](../../../../glossary.md#term-application-binary-interface-abi)-v2 meanings. Version `3` appends one status:

| Value | C name | Meaning |
|---:|---|---|
| `12` | `SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED` | [MSL](../../../../glossary.md#term-metal-shading-language-msl) source [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), named function [lookup](../../../../glossary.md#term-lookup), [target](../../../../glossary.md#term-target)-support [validation](../../../../glossary.md#term-validation), or compute-pipeline creation failed. |

The custom functions reuse existing statuses as follows:

- `INVALID_ARGUMENT` (`1`) covers a null required pointer/handle, malformed success/failure
  output-cell protocol, or another non-shape scalar precondition failure;<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- `ALLOCATION_FAILED` (`4`) covers native custom-pipeline box allocation;
- `INTERNAL_ERROR` (`7`) covers a contained [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) exception;
- `UNSUPPORTED_SHAPE` (`8`) covers zero, a count greater than `UINT32_MAX`, checked
  `element_count * sizeof(float)` overflow, or a count not safely representable by the required
  MSL grid/<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>`MTLSize`/`NSUInteger` strategy;
- `INCOMPATIBLE_RESOURCE` (`10`) covers wrong-device input/output buffers, insufficient logical
  extent, or input/output handle aliasing;<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [buffers glossary](../../../../glossary.md#term-buffer); [extent glossary](../../../../glossary.md#term-extent); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- `EXECUTION_FAILED` (`11`) covers nil command buffer or encoder, an unusable pipeline dispatch
  geometry, command-buffer completion status other than <span data-glossary-navigation="true"> ([command buffer glossary](../../../../glossary.md#term-command-buffer); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>`Completed`, or a non-nil completion
  error; and
- `KERNEL_COMPILATION_FAILED` (`12`) covers nil/error library, function, pipeline, or required
  non-uniform-dispatch target proof.

Unknown integers remain unknown, retain the raw value in the Java failure, and [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) without
retry, remapping, or [MPSGraph](../../../../glossary.md#term-mpsgraph) fallback.

### Create, handle, and [ownership](../../../../glossary.md#term-ownership-meanings) contract

- `out_pipeline` is validated first and set to null before validating the context or element
  count. Every failure leaves it null.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- `context` must be one live native context handle supplied by its owning Java wrapper.
  `element_count` must be in the inclusive range `1..UINT32_MAX`; zero or a greater value returns
  `UNSUPPORTED_SHAPE`. Checked byte [extent](../../../../glossary.md#term-extent) must fit [`uint64_t`](../../../../glossary.md#index-mappings-and-cost-dominance), `NSUInteger`, and the supported
  Metal buffer/dispatch geometry.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [buffer glossary](../../../../glossary.md#term-buffer); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- [Native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture) compiles only the fixed in-source `synaptik_neg_f32` function. It accepts no source,
  function name, macro, library path, metallib, or specialization input from Java.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- Creation synchronously calls `newLibraryWithSource`, resolves exactly
  `synaptik_neg_f32`, creates one `MTLComputePipelineState`, validates positive legal pipeline
  thread widths, and stores the checked element count, byte extent, selected dispatch geometry,
  pipeline, and exact context prerequisite in one retained opaque box.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- On success, `out_pipeline` receives exactly one retained handle. The box strongly retains the
  pipeline state and context needed for its device and command queue. Temporary library,
  function, compile options, and <span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [command queue glossary](../../../../glossary.md#term-command-queue); [compile glossary](../../../../glossary.md#compile))</span>`NSError` values do not cross the [ABI](../../../../glossary.md#term-application-binary-interface-abi).
- Java acquires one provisional context child lease before native create. Successful wrapper
  construction adopts it. Native-create failure releases the lease; wrapper-construction failure
  releases the native pipeline first and the lease second. Primary and distinct suppressed
  failures follow the existing task-0002 order exactly.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- `synaptik_metal_neg_kernel_pipeline_release` consumes one live non-null custom-pipeline handle
  exactly once. Java supplies synchronized idempotency and never retries a failed native release.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release))</span>
- A custom-pipeline handle is never passed to an MPSGraph function, and an MPSGraph executable
  handle is never passed to a custom-pipeline function. Java fake seams and native tests must
  prove this typed separation.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>

### [Run](../../../../glossary.md#term-run) [validation](../../../../glossary.md#term-validation), encoding, and [synchronization](../../../../glossary.md#term-synchronization)

- Java cold binding validates exactly one live context-local input and one live context-local
  output, each with logical bytes at least the pipeline's retained required extent, and rejects
  Java representation identity alias before constructing the bound invocation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [extent glossary](../../../../glossary.md#term-extent); [identity glossary](../../../../glossary.md#term-identity-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
- Native run validates non-null live handles, exact device identity, sufficient retained logical
  extents, and distinct native buffer handles before creating command objects.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Native run creates one command buffer and one compute encoder, binds the retained pipeline,
  binds input at buffer index <span data-glossary-navigation="true"> ([command buffer glossary](../../../../glossary.md#term-command-buffer))</span>`0` and output at index `1` with zero offsets, dispatches the
  retained static element count through exact non-uniform <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>`dispatchThreads`, ends encoding,
  commits, and calls <span data-glossary-navigation="true"> ([commits glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>`waitUntilCompleted` exactly once.
- Success requires `MTLCommandBufferStatusCompleted` and a nil `error`. Any other completed state
  or error maps to `EXECUTION_FAILED`.
- Success writes the caller-supplied assigned output `MTLBuffer` directly. Synaptik performs no
  explicit intermediate/output copy, host staging, MPSGraph call, fallback, retry, or second
  synchronization.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- A native autorelease pool and bounded command objects per invocation are permitted. There is no
  allocation proportional to element count.<span data-glossary-navigation="true"> ([autorelease pool glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>

## Java [lifecycle](../../../../glossary.md#lifecycle) and hot-path semantics

### Analysis and finalization

`MetalNegPreparationPlan` gains one nested/package-private route enum with exactly
`CUSTOM_SINGLE_NEG` and [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph). It is not a [tuning candidate](../../../../glossary.md#candidate-batch) [schema](../../../../glossary.md#term-schema) and is not exposed to
shared [Prepare](../../../../glossary.md#prepare). The plan retains the same exact `MetalDeviceContext` [identity](../../../../glossary.md#term-identity-meanings) and boundary facts.
Its address-workspace requirement is present only for `MPSGRAPH`.

`MetalNegPartitionFinalizer` validates assignment count by route. The custom route must have two
[buffer](../../../../glossary.md#term-buffer) assignments and no workspace assignment. The MPSGraph route retains its existing buffer
plus workspace rules. Finalization branches once on the plan route, creates the matching typed
resource, and constructs the matching prepared recipe. A route/resource mismatch fails before
return; no route changes or fallback occur.

Each resource is returned exactly once as one [`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource). Finalizer-local [rollback](../../../../glossary.md#term-rollback),
shared Prepare rollback, successful [ownership](../../../../glossary.md#term-ownership-meanings) transfer, reverse cleanup, primary/suppressed
failure order, self-suppression avoidance, and context-child [lease](../../../../glossary.md#term-lease) rules remain unchanged.

### Schedule and per-[run](../../../../glossary.md#term-run) [ownership](../../../../glossary.md#term-ownership-meanings)

The schedule assembler preserves exact context [identity](../../../../glossary.md#term-identity-meanings) and sole-[partition](../../../../glossary.md#partition) [validation](../../../../glossary.md#term-validation). For both
routes:

- a caller feed is [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) and never closed by [Runtime](../../../../glossary.md#term-runtime);
- a splat feed is one fresh run-owned initialized [Metal](../../../../glossary.md#term-metal) [buffer](../../../../glossary.md#term-buffer) per run with exact raw [`FLOAT32`](../../../../glossary.md#term-float32)
  bits;
- the output is one fresh run-owned Metal buffer per run;
- execution writes that output directly; and
- [publication](../../../../glossary.md#term-publication) transfers/leasing follows existing Runtime behavior.

Only the [MPSGraph](../../../../glossary.md#term-mpsgraph) route creates the address workspace. The custom route creates no workspace,
pointer array, [scalar](../../../../glossary.md#term-scalar) buffer, hidden intermediate, or [binding](../../../../glossary.md#term-binding) [owner](../../../../glossary.md#term-owner).

### Cold [binding](../../../../glossary.md#term-binding) and hot invocation

`MetalNegPreparedExecutable` remains one NEG-specific recipe class with two explicit construction
paths. It must not store a raw or generic resource supertype that requires a hot cast or switch.
Cold binding performs the route-specific branch and returns a route-specific [bound invocation](../../../../glossary.md#bound-invocation--boundinvocation):

- MPSGraph retains the existing direct resource plus pre-marshalled input/output address slices;
  and<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- custom retains the typed `MetalNegKernelPipelineResource`, input
  `MetalBufferRepresentation`, and output `MetalBufferRepresentation` directly.

The custom `executeBound()` performs exactly one call to the typed resource, which performs
exactly one native [downcall](../../../../glossary.md#term-downcall). From Java hot execution through the downcall there is no allocation,
boxing, reflection, collection/map [lookup](../../../../glossary.md#term-lookup), slot lookup, representation lookup, address-array
[marshalling](../../../../glossary.md#term-marshalling), unsafe cast, string [dispatch](../../../../glossary.md#term-dispatch), [operation](../../../../glossary.md#operation) inspection, [graph](../../../../glossary.md#graph) traversal, route branch,
[kernel](../../../../glossary.md#kernel) selection, retry, fallback, upload, download, or Java [synchronization](../../../../glossary.md#term-synchronization) beyond the narrow
persistent-resource invocation gate already used to serialize safe native use.

Concurrent runs may reuse the [immutable](../../../../glossary.md#term-immutable) pipeline resource but have different [`RunState`](../../../../glossary.md#run-state--runstate), splat,
output, and [publication](../../../../glossary.md#term-publication) resources. No overlap or [throughput](../../../../glossary.md#term-throughput) claim is made.

## Out of [scope](../../../../glossary.md#term-scope)

- any change to `MetalCapabilityProvider`, Planning capability, maximal same-owner partitioning,
  or logical memory planning<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [owner glossary](../../../../glossary.md#term-owner))</span>
- `ADD`, `RELU`, another operation, mixed NEG/other partitions, pointwise fusion, or broader
  custom-kernel lowering<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [pointwise glossary](../../../../glossary.md#term-pointwise); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel); [lowering glossary](../../../../glossary.md#lowering))</span>
- FLOAT16, BFLOAT16, FLOAT64, integer, BOOL, scalar-rank, zero-extent, dynamic Shape, unresolved
  layout, view, offset, strided, broadcast, or multi-output capability<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT64 glossary](../../../../glossary.md#term-float64); [BOOL glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [extent glossary](../../../../glossary.md#term-extent); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- generic route interfaces, route registries, executable handles, pipeline caches, shader-source
  registries, generic parameter bags, reflection, or string-selected kernels<span data-glossary-navigation="true"> ([shader glossary](../../../../glossary.md#term-shader); [parameter glossary](../../../../glossary.md#term-parameter); [kernels glossary](../../../../glossary.md#kernel))</span>
- Metal 0004 typed candidates, compatibility fingerprints, workload cache, model tuning,
  benchmarking, or heuristic-performance evidence<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [model glossary](../../../../glossary.md#term-model); [benchmarking glossary](../../../../glossary.md#term-benchmarking); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- public [Metal](../../../../glossary.md#term-metal) adapter/factory/integration, [Engine](../../../../glossary.md#term-engine) source, standard `compute`, automatic host
  binding, mixed-owner schedule composition, CPU fallback, or backend discovery<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [binding glossary](../../../../glossary.md#term-binding); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
- asynchronous APIs, futures, events, cancellation, timeout, cross-run overlap promises, or
  exposed command buffers<span data-glossary-navigation="true"> ([asynchronous glossary](../../../../glossary.md#term-asynchronous); [cancellation glossary](../../../../glossary.md#term-cancellation); [run glossary](../../../../glossary.md#term-run); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- buffer pooling, aliasing, persistent prepared splat buffers, public device storage, host-visible
  leases, or hidden coherence<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [aliasing glossary](../../../../glossary.md#term-aliasing); [device glossary](../../../../glossary.md#term-device); [leases glossary](../../../../glossary.md#term-lease))</span>
- metallib compilation, packaging, lookup, discovery, extraction, signing, notarization, or
  distribution; the fixed MSL source is compiled during finalization<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [lookup glossary](../../../../glossary.md#term-lookup); [notarization glossary](../../../../glossary.md#term-notarization); [distribution glossary](../../../../glossary.md#term-distribution); [MSL glossary](../../../../glossary.md#term-metal-shading-language-msl))</span>
- architecture authority, ADR, shared production module, production dependency, Gradle,
  architecture-test, backend-conformance, or integration-test changes<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- a detailed Metal 0004 task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/planning`, `modules/runtime`, `modules/prepare`, Concrete backend modules, Metal
  backend, Prepare lifecycle, Run lifecycle, and dependency rules<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module); [Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend); [Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle); [Run glossary](../../../../glossary.md#term-run); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Partition scoring](../../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition))</span>
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Backend-owned lowering glossary](../../../../glossary.md#backend-owned-lowering))</span>
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [Metal strategy note](../../../../design/notes/metal-backend-strategy.md)
- [Metal master plan](../master-plan.md)
- [Metal task 0001](0001-metal-capability-storage-and-native-foundation.md)
- [Metal task 0002](0002-mpsgraph-prepared-execution-route.md)

## Architecture constraints

- [Planning](../../../../glossary.md#term-planning) selects only `owner = metal`. Route selection is deterministic backend analysis over
  the complete already-owned partition and never leaks into Planning or Runtime.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [partition glossary](../../../../glossary.md#partition))</span>
- The existing occurrence capability must remain closed under the actual maximal same-owner
  partitioner. The second route does not reject, split, or repartition a supported partition;
  every supported partition outside the private custom domain remains on MPSGraph, including an
  exact singleton with more than <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [owner glossary](../../../../glossary.md#term-owner); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [singleton glossary](../../../../glossary.md#term-singleton))</span>`UINT32_MAX` [elements](../../../../glossary.md#term-element).
- Backend analysis selects the route and exact declarations before shared assignment.
  Finalization constructs only the already-selected route and adds no resource requirement.
- [Runtime](../../../../glossary.md#term-runtime) receives an [immutable](../../../../glossary.md#term-immutable) prepared recipe and typed cold-[bound invocation](../../../../glossary.md#bound-invocation--boundinvocation). It sees no
  [`Operation`](../../../../glossary.md#operation), `CompiledNode`, route selection, native handle, MSL source, MPSGraph type, or
  backend lookup on the hot path.<span data-glossary-navigation="true"> ([MSL glossary](../../../../glossary.md#term-metal-shading-language-msl); [lookup glossary](../../../../glossary.md#term-lookup); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) remains the unique owner of the pipeline or MPSGraph resource. Executable
  and schedule references create no ownership occurrence.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Caller borrowing, per-run ownership, direct output residency, publication, close/run leases,
  rollback, suppression, and concurrent-run isolation retain current Runtime/Prepare semantics.<span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing); [run glossary](../../../../glossary.md#term-run); [residency glossary](../../../../glossary.md#residency); [publication glossary](../../../../glossary.md#term-publication); [leases glossary](../../../../glossary.md#term-lease); [rollback glossary](../../../../glossary.md#term-rollback); [Prepare glossary](../../../../glossary.md#prepare))</span>
- [Metal](../../../../glossary.md#term-metal) remains independent of [Engine](../../../../glossary.md#term-engine). No shared contract or [dependency direction](../../../../glossary.md#java-platform-and-build-tools) changes.
- Planning documents remain non-authoritative. `ARCHITECTURE.md` and explanatory architecture
  documentation are unchanged.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.metal` — unchanged public capability provider only.
- `io.github.pho001.synaptik.backend.metal.internal` — existing package-private native ABI,
  context/storage owners, NEG analysis/finalization, route-specific persistent resources,
  prepared executable, and sole-partition schedule assembly.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [partition glossary](../../../../glossary.md#partition))</span>

Packages added or changed:

- No package is added. The existing internal package gains one focused resource type. The
  deferred `.internal.prepare` and `.internal.route.mpsgraph` boundaries remain deferred because
  two small routes still share one NEG-specific analysis/finalization/binding lifecycle and no
  reusable cross-package seam has been proved.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

Type placement:

- `io.github.pho001.synaptik.backend.metal.internal.MetalNegKernelPipelineResource` — new
  package-private typed owner for exactly one custom NEG compute pipeline and context lease.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [compute pipeline glossary](../../../../glossary.md#term-compute-pipeline); [lease glossary](../../../../glossary.md#term-lease))</span>
- `MetalDeviceContext` — extended in place with the custom pipeline's provisional-child-lease
  create/adopt/rollback operation, parallel to but typed separately from its MPSGraph creation
  operation.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [operation glossary](../../../../glossary.md#operation); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- `MetalNegPreparationPlan` — extended in place with the closed two-value route decision and
  optional MPSGraph workspace; not renamed or split.
- `MetalNegPartitionPreparer` — extended in place because complete partition validation and
  route selection are one backend-analysis responsibility.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend))</span>
- `MetalNegPartitionFinalizer` — extended in place because assignment validation, typed resource
  construction, and local rollback remain one NEG finalization responsibility.
- `MetalNegPreparedExecutable` — extended in place with explicit MPSGraph/custom construction and
  route-specific cold-bound invocation; not generalized.<span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
- `MetalNegPreparedScheduleAssembler` — extended in place only to create the workspace required
  by the selected plan; feed/output/publication ownership remains shared.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

The [Metal](../../../../glossary.md#term-metal) master-plan package map does not change because every affected and new type stays in the
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

Documentation and [planning](../../../../glossary.md#term-planning) paths:

14. `docs/backend-guide/metal-backend.md`
15. `docs/glossary.md`
16. this task
17. `docs/planning/backends/metal/master-plan.md`
18. `docs/planning/roadmap.md`

Explicitly excluded and unchanged include `MetalCapabilityProvider.java`, its tests, [backend](../../../../glossary.md#backend)
[conformance](../../../../glossary.md#term-conformance), architecture tests, [Engine](../../../../glossary.md#term-engine), shared [modules](../../../../glossary.md#term-module), [Gradle](../../../../glossary.md#term-gradle) files, `ARCHITECTURE.md`, and all
architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) documents.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the exact 18 paths above: nine [Metal](../../../../glossary.md#term-metal) production/[Javadoc](../../../../glossary.md#term-javadoc)
paths, two native documentation/source paths, two existing Metal test paths, two explanatory
documentation paths, and three [planning](../../../../glossary.md#term-planning) paths.

This cohesive scope is larger than the usual source-file [target](../../../../glossary.md#term-target) because the additive [ABI](../../../../glossary.md#term-application-binary-interface-abi),
analysis-time choice, typed persistent [owner](../../../../glossary.md#term-owner), finalization, cold [binding](../../../../glossary.md#term-binding), real/fake execution
evidence, and documentation must agree before the second route is truthful. It adds only one
production type and reuses the established NEG [lifecycle](../../../../glossary.md#lifecycle) owners. If another path, type, package,
public [API](../../../../glossary.md#term-application-programming-interface-api), [dependency](../../../../glossary.md#term-dependency), test [module](../../../../glossary.md#term-module), status, or framework is needed, stop and return this task to
planning rather than [widening](../../../../glossary.md#term-widening) it silently.

## Acceptance criteria

- `MetalCapabilityProvider.java` is byte-for-byte unchanged; all existing provider and
  capability/partition conformance tests retain their exact meaning and pass.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The custom route is selected exactly for a complete supported partition with one NEG node, one
  unique feed, one unique target, a checked element count in <span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [target glossary](../../../../glossary.md#term-target); [element glossary](../../../../glossary.md#term-element))</span>`1..UINT32_MAX`, and valid native
  direct-dispatch geometry, for both caller and splat feeds. Every other task-0002 supported
  partition selects MPSGraph without repartitioning or fallback.
- Cold analysis tests prove the index-domain boundary without allocating giant buffers: safely
  constructed descriptor/geometry facts at exactly <span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>`UINT32_MAX` select `CUSTOM_SINGLE_NEG`, while
  the first representable checked singleton count above it selects <span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>[`MPSGRAPH`](../../../../glossary.md#term-mpsgraph). These tests stop
  before physical representation creation or native allocation.
- Analysis makes the route decision before returning declarations. Custom declares exactly feed
  and target buffers and zero workspaces; MPSGraph preserves its existing declarations and one
  address workspace.
- The plan retains the exact context, feed/target order, Shapes, byte extents, splat facts, and
  route. Finalization rejects route/declaration/assignment mismatches and never changes route.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape))</span>
- [ABI](../../../../glossary.md#term-application-binary-interface-abi) version is exactly `3`; the dylib exports exactly thirteen named symbols; all ten ABI-v2
  signatures and semantics remain unchanged; the three additions, output-cell protocol,
  statuses, typed handles, validation order, concurrency, and cleanup match this task.<span data-glossary-navigation="true"> ([dylib glossary](../../../../glossary.md#term-dylib); [signatures glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Java validates version `3` and resolves all thirteen symbols eagerly in the specified order.
  Malformed native success/failure output cells are cleaned up exactly once and fail closed.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- The native pipeline compiles the fixed branch-free [MSL](../../../../glossary.md#term-metal-shading-language-msl) source once during finalization, retains one
  `MTLComputePipelineState`, and does not [compile](../../../../glossary.md#compile) or select a [kernel](../../../../glossary.md#kernel) during a [run](../../../../glossary.md#term-run).
- The implementation proves exact non-uniform [dispatch](../../../../glossary.md#term-dispatch) support within the `UINT32_MAX` custom
  domain and uses the branch-free `uint` oracle. It has no rounded-grid guarded variant; larger
  supported singletons remain MPSGraph because a bounds guard cannot repair 32-bit index wrap.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [guard glossary](../../../../glossary.md#term-guard))</span>
- Native custom create defensively returns `UNSUPPORTED_SHAPE` for zero and
  `UINT32_MAX + 1`, even though analysis never sends either count to custom finalization. Fake
  seam tests prove both rejections without allocating buffers.
- Custom cold binding accepts only the exact two compatible live context-local buffers, validates
  extents, rejects Java and native aliasing, and creates a direct typed bound invocation with no
  workspace or address array.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [aliasing glossary](../../../../glossary.md#term-aliasing); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
- Custom Java hot execution performs one native downcall with direct typed resource and buffer
  handles and no allocation, boxing, reflection, lookup, address-array marshalling, cast,
  operation dispatch, route branch, retry, fallback, upload, or download.<span data-glossary-navigation="true"> ([downcall glossary](../../../../glossary.md#term-downcall); [lookup glossary](../../../../glossary.md#term-lookup); [marshalling glossary](../../../../glossary.md#term-marshalling); [operation glossary](../../../../glossary.md#operation))</span>
- Native custom execution binds the assigned input/output `MTLBuffer` values directly, submits
  one compute command, waits once at the route boundary, checks completed status/error, and writes
  the supplied output without an explicit copy or MPSGraph call.
- Caller input remains borrowed. Every splat and output remains fresh and run-owned per run.
  Publication leases the direct output; cleanup and failure never close a caller buffer or leak a
  run-owned/persistent resource.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Publication glossary](../../../../glossary.md#term-publication); [leases glossary](../../../../glossary.md#term-lease))</span>
- The pipeline resource owns exactly one native pipeline handle and one adopted context lease.
  It is idempotent, single-consumption, reusable across admitted runs, and isolated from the
  MPSGraph resource/symbol family.<span data-glossary-navigation="true"> ([idempotent glossary](../../../../glossary.md#term-idempotent); [symbol glossary](../../../../glossary.md#term-symbol))</span>
- Finalizer-local and shared rollback preserve the exact preparation primary failure and attach
  later distinct close failures in cleanup encounter order without self-suppression. Successful
  preparation transfers the one selected persistent resource only to <span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution).
- Close/run races retain Runtime 0016 behavior: new runs fail after close begins, admitted runs
  may finish, physical release is deferred when needed, and concurrent runs have isolated mutable
  state and buffers.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [release glossary](../../../../glossary.md#term-release); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Real-device assertions cover finite values, infinities, NaN sign irrelevance, and signed zero
  (<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [finite glossary](../../../../glossary.md#term-finite); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>`+0.0 -> -0.0`, `-0.0 -> +0.0`) with raw-bit checks where applicable.
- One prepared custom pipeline executes at least twice without recompilation and with fresh
  outputs. A separate supported multi-node partition executes through MPSGraph, proving regression
  preservation and the second-route boundary.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- Deterministic fake/native-seam tests cover nil/error library, function, pipeline, command
  buffer, encoder, completion status/error, incompatible buffers, aliasing, unknown status,
  Objective-C exception, and malformed create results. No failure retries or switches routes.<span data-glossary-navigation="true"> ([command buffer glossary](../../../../glossary.md#term-command-buffer); [Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>
- Source and compiled-shape inspection confirms the MSL oracle, one compile site, typed handle
  separation, route-specific bound invocations, exact one-downcall custom hot path, and absence
  of a generic route framework or new public Metal type.<span data-glossary-navigation="true"> ([hot path glossary](../../../../glossary.md#term-hot-path); [Metal glossary](../../../../glossary.md#term-metal))</span>
- At task 0003's completion frontier, public Engine composition was CPU-only; this task's
  acceptance scope permitted no Engine, mixed-owner, CPU fallback, public storage, or
  standard-compute claim in code or documentation.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [scope glossary](../../../../glossary.md#term-scope); [owner glossary](../../../../glossary.md#term-owner))</span>
- The Metal guide and native README clearly distinguish the custom singleton route from the
  retained whole-partition MPSGraph route, describe ABI version `3`, and make no performance
  superiority claim.
- Glossary review updates only reusable project terminology whose meaning changes; otherwise it
  records a reasoned no-change conclusion in the task evidence.
- A separate clean documentation-focused agent pass finalizes affected Javadocs, explanatory
  documentation, glossary impact, task evidence, master plan, and roadmap in the same change.
- Focused/final Java tests, real-device execution, native build/export/framework checks,
  Javadoc, Markdown links/anchors/fences/newlines/whitespace, exact 18-path scope, package/type
  placement, status/order synchronization, no-staged-files check, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>`git diff --check` pass.

## Performance constraints and inspection

- Route choice, element-count calculation, source choice, source compilation, function lookup,
  pipeline construction, threadgroup geometry selection, and persistent acquisition occur before
  runtime execution.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [lookup glossary](../../../../glossary.md#term-lookup); [threadgroup glossary](../../../../glossary.md#term-threadgroup); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- The MSL source is inspected against the branch-free clean oracle above. No bounds guard or
  other algorithmic deviation is accepted inside the <span data-glossary-navigation="true"> ([MSL glossary](../../../../glossary.md#term-metal-shading-language-msl); [oracle glossary](../../../../glossary.md#term-test-oracle); [guard glossary](../../../../glossary.md#term-guard))</span>`UINT32_MAX` custom domain.
- Native source inspection confirms exactly one element load, one unary negation, and one element
  store per valid custom thread, with no temporary buffer or second dispatch.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [buffer glossary](../../../../glossary.md#term-buffer); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Java source plus `javap -c -p` inspection confirms a route-specific custom bound invocation
  whose <span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>`executeBound()` directly calls the typed resource and whose resource directly invokes
  the custom native run handle once.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- No latency, throughput, occupancy, bandwidth, power, or superiority threshold is required.
  The heuristic is intentionally unmeasured until Metal 0004.<span data-glossary-navigation="true"> ([latency glossary](../../../../glossary.md#term-latency); [throughput glossary](../../../../glossary.md#term-throughput); [heuristic glossary](../../../../glossary.md#term-heuristic); [Metal glossary](../../../../glossary.md#term-metal))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation development may [run](../../../../glossary.md#term-run) focused test methods. After Java/native behavior stabilizes,
the implementation context runs one focused class and one final ordinary [Metal](../../../../glossary.md#term-metal) suite:

```bash
./gradlew :backends:metal:test \
  --tests '*MetalNegPreparedExecutionTest'
./gradlew :backends:metal:test
```

The focused tests must use the injected fake native seam to prove both route branches,
declaration timing, assignment validation, exact status mapping, typed-handle separation,
resource [rollback](../../../../glossary.md#term-rollback), cold [binding](../../../../glossary.md#term-binding), hot-call counts, caller/splat [ownership](../../../../glossary.md#term-ownership-meanings), direct destination,
reuse, and concurrent-run isolation. Cold analysis-only fixtures must prove exactly
`UINT32_MAX -> CUSTOM_SINGLE_NEG` and `UINT32_MAX + 1 -> MPSGRAPH` from safely constructed static
[descriptor](../../../../glossary.md#term-descriptor)/geometry facts without creating physical [buffers](../../../../glossary.md#term-buffer). The native fake seam must separately
prove custom create returns `UNSUPPORTED_SHAPE` for zero and `UINT32_MAX + 1`. Existing
capability/provider tests are part of the final [module](../../../../glossary.md#term-module) suite and must pass without source changes.

Build and inspect the [native library](../../../../glossary.md#term-native-library):

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

On the current eligible macOS-[arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) [host](../../../../glossary.md#term-host), run the exact existing foundation case and the expanded
[prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) case with the freshly built library:

```bash
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test \
  --tests '*MetalFoundationTest.nativeFoundationRoundTrip' \
  --tests '*MetalNegPreparedExecutionTest.nativePreparedNegRoundTripAndReuse'
```

The prepared native case must prove custom caller and splat execution/reuse/direct destinations
and a multi-[node](../../../../glossary.md#node) [MPSGraph](../../../../glossary.md#term-mpsgraph) [regression](../../../../glossary.md#term-regression). The task cannot become Complete by skipping it on this
eligible host. If the sandbox hides the Metal [device](../../../../glossary.md#term-device), rerun the exact command with approved
unsandboxed device access and record both outcomes. If custom pipeline [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), safe
[dispatch](../../../../glossary.md#term-dispatch), assigned output, or exact NEG semantics cannot be proved, keep task 0003 incomplete and
return it to [planning](../../../../glossary.md#term-planning); do not change capability or silently use MPSGraph for the exact case.

No [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) test was rerun solely for this task because the provider and partitioning
behavior did not change. Under the completion-time [CPU](../../../../glossary.md#term-cpu)-only [Engine](../../../../glossary.md#term-engine) frontier, no [integration test](../../../../glossary.md#term-integration-test)
applied. No [architecture test](../../../../glossary.md#term-architecture-test) applied because no [dependency](../../../../glossary.md#term-dependency)/build/module boundary changed. If
any such path had become necessary, the task required replanning before editing it.

Inspect source and compiled Java [shape](../../../../glossary.md#shape):

```bash
javap -classpath backends/metal/build/classes/java/main -c -p \
  io.github.pho001.synaptik.backend.metal.internal.MetalNegPreparedExecutable \
  io.github.pho001.synaptik.backend.metal.internal.MetalNegKernelPipelineResource
```

Manually confirm the exact [MSL](../../../../glossary.md#term-metal-shading-language-msl) source, branch/[guard](../../../../glossary.md#term-guard) decision, one pipeline [compile](../../../../glossary.md#compile) site, direct
buffer indices `0` and `1`, one dispatch, one completion wait, one custom Java [downcall](../../../../glossary.md#term-downcall), no hot
route branch/allocation/[marshalling](../../../../glossary.md#term-marshalling), no cross-family handle use, no public Metal type added by
task 0003, and no Engine import. Recurring semantic and [lifecycle](../../../../glossary.md#lifecycle) invariants belong in automated
tests; this manual
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
no detailed Metal 0004 task, and no staged files.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>

Repository-wide validation is deferred to the final Metal [checkpoint](../../../../glossary.md#term-checkpoint) or [continuous integration](../../../../glossary.md#term-continuous-integration-ci).
The implementation is one backend-local behavior change with no capability, shared contract,
production edge, [Gradle](../../../../glossary.md#term-gradle), or architecture change. The documentation pass reuses stabilized Java,
native, and real-device evidence and does not repeat successful Java suites unless it changes
executable behavior or records a concrete stale-evidence risk.

## Dependencies

- Metal 0001 — Complete; native context, buffer ownership, host byte access, child leases, and
  foundation statuses.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span><span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [buffer glossary](../../../../glossary.md#term-buffer); [host glossary](../../../../glossary.md#term-host))</span>
- Metal 0002 — Complete; unchanged NEG capability, whole-partition MPSGraph route, stable
  feed/target analysis, per-run splat/output/address ownership, ABI version <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [target glossary](../../../../glossary.md#term-target); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>`2`, and fake/real
  test seams to extend.
- [Runtime](../../../../glossary.md#term-runtime) 0016 — Complete; persistent-resource [ownership](../../../../glossary.md#term-ownership) and close/[run](../../../../glossary.md#term-run) [leases](../../../../glossary.md#term-lease).
- Prepare 0006 — Complete; finalizer-local responsibility, transactional rollback, and ownership
  transfer.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- [Engine](../../../../glossary.md#term-engine) 0010 — Complete [lifecycle](../../../../glossary.md#lifecycle) prerequisite, but no Engine source is in [scope](../../../../glossary.md#term-scope).
- [Compiler](../../../../glossary.md#term-compiler) 0006B7 — Complete; final [descriptor](../../../../glossary.md#term-descriptor) closure required by existing truthful capability.
- JDK 26 FFM and installed macOS-arm64 Foundation, Metal, and
  MetalPerformanceShadersGraph frameworks.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [FFM glossary](../../../../glossary.md#term-ffm); [arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>

No [Model](../../../../glossary.md#term-model) 0026, tuning, [cache](../../../../glossary.md#term-cache), public composition, new capability, or shared-contract [dependency](../../../../glossary.md#term-dependency) is
introduced.

## Follow-up tasks

- Metal 0004 remains a concise Draft master-plan row. It will own typed complete MPSGraph/custom
  route candidates, candidate/cache schema compatibility, target/workload fingerprints, safe
  heuristic evolution, and tuning integration. This task must not create its detailed spec.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [cache glossary](../../../../glossary.md#term-cache); [schema glossary](../../../../glossary.md#term-schema); [target glossary](../../../../glossary.md#term-target); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Broader operations, pointwise fusion, mixed precision, public Metal composition, mixed-owner
  schedules, pooling, packaging, and asynchronous execution remain separately planned future
  work and do not weaken this task's acceptance criteria.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [fusion glossary](../../../../glossary.md#term-fusion); [mixed precision glossary](../../../../glossary.md#term-mixed-precision); [owner glossary](../../../../glossary.md#term-owner); [pooling glossary](../../../../glossary.md#term-pooling); [asynchronous glossary](../../../../glossary.md#term-asynchronous))</span>

## Architecture impact

Expected impact: None.

This task implements custom-[kernel](../../../../glossary.md#kernel) route selection, [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), persistent [ownership](../../../../glossary.md#term-ownership-meanings), and
execution already assigned to the [Metal](../../../../glossary.md#term-metal) [backend](../../../../glossary.md#backend). It changes no architecture authority, ownership
boundary, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), shared [API](../../../../glossary.md#term-application-programming-interface-api), [Planning](../../../../glossary.md#term-planning) meaning, [Runtime](../../../../glossary.md#term-runtime) meaning, or public surface.
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

Stop and return the task to [planning](../../../../glossary.md#term-planning) if:

- analysis cannot keep every [singleton](../../../../glossary.md#term-singleton) above `UINT32_MAX` on MPSGraph before declarations, or
  custom native create does not reject zero and greater-than-<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>`UINT32_MAX` counts [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation);
- the [target](../../../../glossary.md#term-target) cannot [compile](../../../../glossary.md#compile)/reuse the fixed [MSL](../../../../glossary.md#term-metal-shading-language-msl) pipeline or write the supplied output directly;
- custom execution requires a workspace, address-array marshalling, per-element allocation,
  hidden host copy, multiple native calls, multiple synchronization waits, or hot route dispatch;<span data-glossary-navigation="true"> ([marshalling glossary](../../../../glossary.md#term-marshalling); [element glossary](../../../../glossary.md#term-element); [host glossary](../../../../glossary.md#term-host); [synchronization glossary](../../../../glossary.md#term-synchronization); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- analysis cannot select the route and exact requirements before shared assignment;
- finalization cannot return exactly one typed persistent resource with existing transactional
  ownership and rollback semantics;<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- implementation needs to change capability, repartition, add fallback, reinterpret an MPSGraph
  handle, add a generic executable handle/route framework, or cross a package/public boundary;
- Engine/shared production/Gradle/architecture/conformance/integration source or a nineteenth
  path is required; or<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- the eligible real-device test cannot prove custom caller/splat execution, direct output,
  pipeline reuse, exact NEG semantics, and retained MPSGraph multi-node execution.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [node glossary](../../../../glossary.md#node))</span>

## Local decisions

- Extend the existing NEG-specific plan, preparer, finalizer, prepared executable, and schedule
  assembler in place. Their responsibilities remain cohesive; splitting or renaming them would
  add cross-type plumbing without a reusable second operation or route family.<span data-glossary-navigation="true"> ([prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [operation glossary](../../../../glossary.md#operation))</span>
- Add only `MetalNegKernelPipelineResource` as a new production type so native custom-pipeline
  ownership cannot be confused with <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`MetalMpsGraphExecutableResource`.
- Use a closed two-value plan discriminator rather than a generic route abstraction. It is an
  implementation fact, not a public API, tuning schema, or cache key.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [schema glossary](../../../../glossary.md#term-schema); [cache key glossary](../../../../glossary.md#term-cache-key))</span>
- Keep splat materialization unchanged and per-run. The custom route changes only persistent
  executable state and the absence of MPSGraph address-array workspace.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [run glossary](../../../../glossary.md#term-run); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- Restrict the custom implementation domain to checked positive counts no greater than
  `UINT32_MAX`, then use exact non-uniform `dispatchThreads` and the branch-free `uint` MSL oracle.
  A guard is not an alternative for larger counts because it cannot repair index wrap; those
  supported singletons remain MPSGraph.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [guard glossary](../../../../glossary.md#term-guard))</span>
- Reuse statuses `1`, `4`, `7`, `8`, `10`, and `11` for their existing meanings and add only
  status `12` for custom [MSL](../../../../glossary.md#term-metal-shading-language-msl)/function/pipeline/[target](../../../../glossary.md#term-target) [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) failure.
- Do not modify the master-plan package map because the one new type fits its existing internal
  native/resource responsibility.

## Known limitations

- The custom route covers only an exact one-node/one-feed/one-target partition with checked
  element count in <span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [target glossary](../../../../glossary.md#term-target); [partition glossary](../../../../glossary.md#partition); [element glossary](../../../../glossary.md#term-element))</span>`1..UINT32_MAX` inside the unchanged [`FLOAT32`](../../../../glossary.md#term-float32) NEG capability domain. Every
  larger/multi-boundary partition and every supported singleton above that count remains
  MPSGraph.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- The selection heuristic is deterministic but unmeasured. This task makes no speed, latency,
  bandwidth, occupancy, or energy claim.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic); [latency glossary](../../../../glossary.md#term-latency))</span>
- Execution remains synchronous and may serialize native use through the persistent resource.
  No asynchronous or concurrent-overlap guarantee is made.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [asynchronous glossary](../../../../glossary.md#term-asynchronous))</span>
- At task 0003's completion frontier, the ordinary Engine lifecycle was CPU-only; this task proved
  the route through backend-local typed preparation/runtime integration. Engine task 0015 later
  added explicit single-owner Metal execution while leaving <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [owner glossary](../../../../glossary.md#term-owner); [Metal glossary](../../../../glossary.md#term-metal))</span>`Engine.standard()` [CPU](../../../../glossary.md#term-cpu)-only.
- MSL is compiled from fixed source during each preparation finalization. There is no metallib,
  pipeline cache, binary archive, serialization, package, discovery, or tuning integration.<span data-glossary-navigation="true"> ([MSL glossary](../../../../glossary.md#term-metal-shading-language-msl); [cache glossary](../../../../glossary.md#term-cache); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- All task-0002 type/[Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/public-composition and platform limitations remain.

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context recorded these stabilized results, which the distinct documentation
context `/root/metal_0003_docs` reused without repeating successful Java suites:

- focused `MetalFoundationTest`: 14 total, 13 passed, one expected native opt-in skip;
- final ordinary Metal suite: 45 total, 42 passed, three expected native opt-in skips, no failures
  or errors;
- earlier focused `MetalNegPreparedExecutionTest`: 28 total, 26 passed, two expected native
  opt-in skips before the final test-only/native mapping refinement; the final full suite
  supersedes this focused result;
- native [dylib](../../../../glossary.md#term-dylib) build passed; `file` reported [Mach-O](../../../../glossary.md#term-mach-o) [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary); `nm` reported exactly thirteen
  `synaptik_metal_*` exports; `otool` reported the expected Foundation, Metal, and
  MetalPerformanceShadersGraph linkage;
- both `nativeFoundationRoundTrip` and `nativePreparedNegRoundTripAndReuse` passed on the real
  device against the final rebuilt dylib;<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- generated [Javadoc](../../../../glossary.md#term-javadoc) and source/`javap` hot-shape inspection passed before the documentation pass;
  the inspection confirmed the route-specific bound invocation, one custom native call, typed
  pipeline ownership, fixed branch-free MSL, direct buffer indices, one dispatch, and one wait;<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [MSL glossary](../../../../glossary.md#term-metal-shading-language-msl); [buffer glossary](../../../../glossary.md#term-buffer); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- final source-contract assertions cover every specified native fail-closed create, release, and
  run branch and the <span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [release glossary](../../../../glossary.md#term-release))</span>`UNSUPPORTED_SHAPE`, `KERNEL_COMPILATION_FAILED`, and `EXECUTION_FAILED`
  mappings; and
- implementation review blockers were fixed, independent re-review reported no blocking
  findings, `git diff` checks passed, and the staged path count remained zero.

The documentation pass ran the task-prescribed [Metal](../../../../glossary.md#term-metal) Javadoc, Markdown link/anchor/fence/final-
newline/whitespace checks, exact-path and no-detailed-0004 checks, `git diff --check`, and
`git diff --cached --check`. Its final command results are recorded in the completion summary.
No performance benchmark was [run](../../../../glossary.md#term-run) or required.

## Implementation notes

Analysis now selects `CUSTOM_SINGLE_NEG` exactly for one NEG [node](../../../../glossary.md#node), one feed, one [target](../../../../glossary.md#term-target), and a
checked positive count no greater than `UINT32_MAX`; it declares two [buffers](../../../../glossary.md#term-buffer) and no workspace.
The first larger supported [singleton](../../../../glossary.md#term-singleton) and all other supported NEG partitions select [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph)
with the existing address workspace. The unchanged capability provider remains [occurrence](../../../../glossary.md#term-occurrence)-level,
and neither route selection nor a route failure changes [ownership](../../../../glossary.md#term-ownership-meanings), retries, or repartitions work.

[ABI](../../../../glossary.md#term-application-binary-interface-abi) version `3` eagerly binds exactly thirteen symbols and keeps the MPSGraph and custom handle
families separate. Custom creation receives `element_count` through a [`uint64_t`](../../../../glossary.md#index-mappings-and-cost-dominance) [carrier](../../../../glossary.md#term-storage-carrier) but
accepts only `1..UINT32_MAX`. The fixed [MSL](../../../../glossary.md#term-metal-shading-language-msl) entry uses `uint thread_position_in_grid` and contains
one load, [unary](../../../../glossary.md#term-unary) negation, and store with no bounds branch. Native creation validates grid
representation separately from [threadgroup](../../../../glossary.md#term-threadgroup) usability: the former is unsupported [shape](../../../../glossary.md#shape), while the
latter is execution failure. Source/library/function/pipeline and target-family proof failures use
the new [kernel](../../../../glossary.md#kernel)-[compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) status `12`.

Finalization compiles exactly one selected typed persistent resource. Custom cold [binding](../../../../glossary.md#term-binding) retains
the input and assigned output buffers directly; hot execution makes one typed native call. Native
execution uses one [command buffer](../../../../glossary.md#term-command-buffer), one compute encoder, one exact [dispatch](../../../../glossary.md#term-dispatch), and one completion
wait, writing the assigned output `MTLBuffer` without an explicit [host](../../../../glossary.md#term-host)-staging or intermediate-
copy step. Caller inputs remain [borrowed](../../../../glossary.md#ownership-borrowing-and-lease), while splats and outputs remain fresh and [run](../../../../glossary.md#term-run)-owned.
Context child [leases](../../../../glossary.md#term-lease), concurrent admitted-run isolation, deferred close, malformed create-output
cleanup, status preservation, and finalizer/shared [rollback](../../../../glossary.md#term-rollback) remain aligned with the established
[Runtime](../../../../glossary.md#term-runtime) and [Prepare](../../../../glossary.md#prepare) ownership contracts.

## Completion summary

Completed changes:

- Added the exact singleton/count route boundary before shared declarations, retaining MPSGraph
  for every other supported partition and leaving <span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [partition glossary](../../../../glossary.md#partition))</span>`MetalCapabilityProvider` unchanged.
- Added private ABI-v3 custom-pipeline create/release/run operations, fixed branch-free FLOAT32
  NEG MSL, typed persistent ownership, direct assigned-output execution, and fail-closed status
  and malformed-output handling.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [release glossary](../../../../glossary.md#term-release); [run glossary](../../../../glossary.md#term-run); [FLOAT32 glossary](../../../../glossary.md#term-float32); [MSL glossary](../../../../glossary.md#term-metal-shading-language-msl); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Preserved caller borrowing, per-run splat/output ownership, isolated admitted runs, deferred
  close, context child leases, deterministic rollback, and separate MPSGraph/custom handle
  families.<span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing); [leases glossary](../../../../glossary.md#term-lease); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Finalized affected Javadocs, the native README, Metal backend guide, glossary terminology, this
  task, the Metal master plan, and the roadmap in clean documentation context<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
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

[Validation](../../../../glossary.md#term-validation):

- Focused `MetalFoundationTest`: 14 total, 13 passed, one expected native opt-in skip.
- Final Metal suite: 45 total, 42 passed, three expected native opt-in skips, no failures or
  errors. The earlier 28-total focused prepared-execution result was superseded as recorded above.
- Native build, Mach-O arm64 inspection, exact thirteen-export inspection, expected framework
  linkage, both final-dylib real-device cases, source-contract assertions, generated hot-shape
  inspection, and independent implementation re-review all passed.<span data-glossary-navigation="true"> ([Mach-O glossary](../../../../glossary.md#term-mach-o); [arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [dylib glossary](../../../../glossary.md#term-dylib); [device glossary](../../../../glossary.md#term-device); [shape glossary](../../../../glossary.md#shape))</span>
- Documentation validation passed `./gradlew :backends:metal:javadoc`, the targeted six-file
  Markdown validator, final newline/whitespace checks, `git diff --check`,
  `git diff --cached --check`, exact 18-path scope, zero staged paths, and the no-detailed-0004
  check.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

No-change conclusions:

- `ARCHITECTURE.md`, focused architecture explanations, and ADRs remain accurate because route
  choice, lowering, and persistent ownership stay inside the existing Metal Prepare boundary.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Capability, backend conformance, integration, public Engine, shared modules, Gradle, build
  structure, and native build script required no change for task 0003. Its provider and public API
  did not expand, and the Engine composition available at that completion frontier was CPU-only.
  Engine task 0015 later superseded that public-composition status.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- The glossary required a targeted change because its physical-representation and MPSGraph-only
  prepared-executable text was stale; no new cross-project abstraction or public term was added.
- Existing Javadocs that were not changed remain accurate after review. The affected route, ABI,
  resource, lifecycle, and direct-output contracts were finalized in the already changed
  production files.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

Unresolved issues: none for task 0003.

Follow-up: [Metal](../../../../glossary.md#term-metal) 0004 remains Draft, without a detailed task, for typed route candidates, [target](../../../../glossary.md#term-target)
and [cache](../../../../glossary.md#term-cache) compatibility, and tuning integration. Broader [kernels](../../../../glossary.md#kernel), types, public Metal [Engine](../../../../glossary.md#term-engine)
composition, packaging, and [asynchronous](../../../../glossary.md#term-asynchronous) execution remain separately planned.

Status: Complete
