# Metal native bridge for macOS arm64

## Purpose

This directory builds the local application binary interface (ABI) used by the Synaptik Metal
backend on Apple-silicon macOS. ABI version 2 retains the context and shared-storage buffer
foundation and adds one reusable Metal Performance Shaders Graph (MPSGraph) route for whole
partitions of supported `FLOAT32` unary `NEG` operations.

```text
Java prepare -> compile one MPSGraph executable -> retain opaque executable handle
Java run     -> pass ordered MTLBuffer handles  -> synchronous execution into supplied outputs
Java close   -> release executable, buffers, context, and FFM lookup through their owners
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

## ABI version 2

The dylib exports exactly these ten symbols:

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
```

The version function returns unsigned value `2`. All other functions return a signed 32-bit
status. Context, buffer, and executable values cross the boundary as opaque `void *` handles.
Sizes and offsets are unsigned 64-bit values; counts, ranks, and value indices are unsigned
32-bit values. Created handles use caller-supplied output cells, which remain null on failure.

The complete executable-create and run signatures, padded rank-16 dimension table, stable
feed/target ordering, and pointer preconditions are recorded in
[Metal task 0002](../../docs/planning/backends/metal/tasks/0002-mpsgraph-prepared-execution-route.md#exact-native-abi-version-2).

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
| 8 | `SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE` | Rank, dimensions, or checked `FLOAT32` byte geometry are unsupported. |
| 9 | `SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED` | Graph construction or compilation produced no usable executable. |
| 10 | `SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE` | A buffer has the wrong device or extent, or an input aliases an output. |
| 11 | `SYNAPTIK_METAL_STATUS_EXECUTION_FAILED` | Synchronous execution, completion reporting, or returned-result validation failed. |

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

## Ownership and concurrency

A successful create transfers one retained handle to Java, and its matching release consumes that
handle once. Buffer wrappers and the persistent executable resource retain context child leases,
so context owner close rejects new work but defers physical context release until the final child
closes. `PreparedExecution` owns the executable resource exactly once; each run separately owns
its initialized constant buffers, outputs, and native-address workspace. Caller input buffers are
borrowed.

The prepared recipe and executable are reusable. Concurrent logical runs have isolated mutable
buffers and workspaces. The current native resource serializes its synchronous invocation
boundary; no throughput, asynchronous overlap, cancellation, timeout, or event-chaining contract
is claimed.

## Validation

Build and inspect the local library with:

```bash
./native/metal-macos-arm64/build.sh
file native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
otool -L native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
```

The symbol list must contain exactly the ten names above, and the link list must contain
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
case proves real `GraphCompilationPort` caller-input compilation, shared `GraphPreparation`,
Runtime execution, executable reuse, direct supplied outputs, publication, and host observation.
Positive-rank compile-time splat initialization is tested separately through backend-local typed
`PrepareContext` construction because the public compilation port cannot express an explicit
positive-rank forward constant.

## Boundaries

The bridge implements no library discovery, packaging, public Engine composition, mixed-owner
schedule, CPU fallback, custom Metal kernel, asynchronous API, buffer pool, persistent constant
buffer, executable serialization, tuning, FLOAT16, BFLOAT16, or performance claim. The public
Metal surface remains only `MetalCapabilityProvider`.
