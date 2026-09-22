# Metal native bridge for macOS arm64

## Purpose

This directory builds the local application binary interface (ABI) used by the Synaptik Metal
backend on Apple-silicon macOS. ABI version 3 retains the context, shared-storage buffer, and
Metal Performance Shaders Graph (MPSGraph) contracts and adds one typed custom compute-pipeline
route for a bounded singleton `FLOAT32` unary `NEG` partition.

```text
Java analysis -> choose custom singleton or MPSGraph route -> declare exact resources
Java prepare  -> compile one typed persistent resource     -> retain opaque handle
Java run      -> pass assigned MTLBuffer handles           -> synchronous direct output
Java close    -> release persistent/run resources, context, and FFM lookup
```

The bridge is backend-private. Its handles are process-local ownership tokens, not caller-visible
addresses or serialization values.

## Build prerequisites and output

Build on an Apple-silicon macOS host with Xcode Command Line Tools:

```bash
./native/metal-macos-arm64/build.sh
```

The script targets arm64, enables automatic reference counting (ARC), and links Foundation,
Metal, and MetalPerformanceShadersGraph. It writes only
`native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib`. The ignored binary is not a
packaged, signed, notarized, or published artifact; Java callers supply its absolute path.

## ABI version 3

The dylib exports exactly these thirteen symbols:

```text
synaptik_metal_foundation_abi_version
synaptik_metal_context_create
synaptik_metal_context_release
synaptik_metal_buffer_create
synaptik_metal_buffer_release
synaptik_metal_buffer_upload
synaptik_metal_buffer_download
synaptik_metal_mpsgraph_neg_executable_create
synaptik_metal_mpsgraph_executable_release
synaptik_metal_mpsgraph_executable_run
synaptik_metal_neg_kernel_pipeline_create
synaptik_metal_neg_kernel_pipeline_release
synaptik_metal_neg_kernel_pipeline_run
```

The version function returns unsigned value `3`. All other functions return a signed 32-bit
status. Context, buffer, MPSGraph-executable, and custom-pipeline values cross the boundary as
separate opaque `void *` handle families. Buffer sizes and offsets are unsigned 64-bit values.
MPSGraph value, node, feed, and target counts, ranks, and value indices are unsigned 32-bit
values. Custom-pipeline `element_count` is carried as `uint64_t`, but custom NEG creation accepts
only `1..UINT32_MAX`; it returns unsupported shape outside that domain rather than narrowing the
value. Created handles use caller-supplied output cells, which remain null on failure.

The complete executable-create and run signatures, padded rank-16 dimension table, stable
feed/target ordering, and pointer preconditions are recorded in
[Metal task 0003](../../docs/planning/backends/metal/tasks/0003-single-neg-custom-metal-kernel-route.md#exact-native-abi-version-3).

### Status values

| Value | C name | Meaning |
|---:|---|---|
| 0 | `SYNAPTIK_METAL_STATUS_OK` | The operation completed successfully. |
| 1 | `SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT` | A required handle, pointer, count, index, topology fact, or output cell is invalid. |
| 2 | `SYNAPTIK_METAL_STATUS_NO_DEVICE` | No default Metal device is available. |
| 3 | `SYNAPTIK_METAL_STATUS_NO_COMMAND_QUEUE` | The device could not create a command queue. |
| 4 | `SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED` | Native owner or buffer allocation failed. |
| 5 | `SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS` | A requested buffer copy is outside its logical extent. |
| 6 | `SYNAPTIK_METAL_STATUS_COPY_FAILED` | Shared buffer contents were unavailable for a requested copy. |
| 7 | `SYNAPTIK_METAL_STATUS_INTERNAL_ERROR` | An Objective-C exception crossed an internal implementation boundary. |
| 8 | `SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE` | Rank, dimensions, checked `FLOAT32` byte geometry, custom element count, or dispatch-grid representation is unsupported. |
| 9 | `SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED` | Graph construction or compilation produced no usable executable. |
| 10 | `SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE` | A buffer has the wrong device or extent, or an input aliases an output. |
| 11 | `SYNAPTIK_METAL_STATUS_EXECUTION_FAILED` | Synchronous execution, unusable threadgroup geometry, completion reporting, or returned-result validation failed. |
| 12 | `SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED` | Fixed custom-kernel compilation, function lookup, target validation, or pipeline creation failed. |

Unknown integers remain unknown and fail closed on the Java side with the raw status retained.
The deterministic Java fake/native seam is the accepted error-matrix evidence; real-device tests
exercise successful execution and do not manufacture framework failures.

## Prepared NEG execution

Creation consumes a validated, topologically ordered whole-partition description. Native code
creates fixed-shape `FLOAT32` placeholders, lowers each node to MPSGraph negation, and compiles one
shape-specialized executable. The executable owner retains ordered feed and target shapes, byte
extents, stable-to-framework permutations, and the originating context.

Each run binds ordered input and supplied output `MTLBuffer` objects through
`MPSGraphTensorData`. It sets `waitUntilCompleted = YES`, submits the executable once on the
context command queue, checks the completion error and returned-result list, and returns only
after the supplied destinations are usable. Synaptik requests no result-synchronization blit and
performs no explicit post-execution device or host copy. MPSGraph may still use internal temporary
storage.

Java validates live typed handles and readable pointer arrays before native entry. The ABI cannot
prove that an arbitrary non-null raw pointer is live, type-correct, or sufficiently sized;
stale, wrong-type, or undersized raw pointers are outside its preconditions.

## Custom singleton NEG execution

An exact one-node, one-feed, one-target supported partition whose checked element count is in
`1..UINT32_MAX` uses a distinct custom pipeline handle. Creation compiles only the fixed,
branch-free `synaptik_neg_f32` Metal Shading Language source, validates the target and pipeline
geometry for exact non-uniform dispatch, and retains the element count and selected threadgroup
width. A grid that cannot represent the element count maps to unsupported shape; unusable
threadgroup geometry maps to execution failure. Java does not supply source text, a function
name, or a route identifier.

Each invocation binds the direct input `MTLBuffer` at index `0` and assigned output `MTLBuffer`
at index `1`, creates one command buffer and one compute encoder, dispatches exactly the retained
element count with `dispatchThreads`, and waits once for successful completion. The assigned
output is written directly; the bridge performs no explicit host staging or intermediate output
copy. Every other already-supported NEG partition remains on MPSGraph. The route is selected once
during analysis and is never retried, replaced, or repartitioned during finalization or execution.
This private implementation-domain boundary is not capability narrowing, tuning, fallback, or a
performance claim.

## Ownership and concurrency

A successful create transfers one retained handle to Java, and its matching release consumes that
handle once. Malformed create results fail closed: success with a null output is rejected, while a
failure that publishes a non-null handle releases that handle once and preserves cleanup failure
as suppressed evidence. Buffer wrappers and the selected persistent resource retain context child
leases, so context owner close rejects new work but defers physical context release until the
final child closes. `PreparedExecution` owns the selected resource exactly once; each run
separately owns its initialized constant buffers and outputs. MPSGraph runs additionally own one
native-address workspace; custom runs own none. Caller input buffers are borrowed.

The prepared recipe and selected resource are reusable. Concurrent admitted logical runs have
isolated mutable buffers and any route-specific workspace. Closing the context or prepared owner
rejects new work and defers physical release until admitted uses finish. The current persistent
resource serializes its synchronous invocation boundary; no throughput, asynchronous overlap,
cancellation, timeout, or event-chaining contract is claimed.

## Validation

Build and inspect the local library with:

```bash
./native/metal-macos-arm64/build.sh
file native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
otool -L native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
```

The symbol list must contain exactly the thirteen names above, and the link list must contain
Foundation, Metal, and MetalPerformanceShadersGraph.

Run the two opt-in real-device cases against the freshly built dylib:

```bash
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test \
  --tests '*MetalFoundationTest.nativeFoundationRoundTrip' \
  --tests '*MetalNegPreparedExecutionTest.nativePreparedNegRoundTripAndReuse'
```

The environment variable is required; ordinary sandboxed runs skip native-device cases. The
foundation case proves context/buffer ownership and bounded shared-memory copies. The prepared
case proves custom caller and splat execution with pipeline reuse and direct supplied outputs,
then real `GraphCompilationPort`, shared `GraphPreparation`, Runtime, and retained multi-node
MPSGraph execution. Positive-rank compile-time splat construction uses backend-local typed
`PrepareContext` because the public compilation port cannot express that forward constant.

## Boundaries

The bridge implements no library discovery, packaging, public Engine composition, mixed-owner
schedule, CPU fallback, general custom-kernel framework, asynchronous API, buffer pool,
persistent constant buffer, executable serialization, tuning, FLOAT16, BFLOAT16, or performance
claim. The public Metal surface remains only `MetalCapabilityProvider`.
