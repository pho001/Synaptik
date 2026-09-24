# Metal native bridge for macOS arm64

## Purpose

This directory builds the local application binary interface (ABI) used by the Synaptik Metal
backend on Apple-silicon macOS. ABI version 4 retains context, shared-storage buffer, executable,
and bounded custom singleton-`NEG` ownership. Its versioned typed whole-partition MPSGraph program
supports `NEG`, `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE`. Operation wires
`2..5`, formerly used for `ADD`, `SUB`, `MUL`, and `DIV`, are withdrawn and rejected: real-device
audit proved that MPSGraph arithmetic does not preserve required `FLOAT32` subnormal semantics.

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

## ABI version 4

The dylib exports exactly these thirteen symbols:

```text
synaptik_metal_foundation_abi_version
synaptik_metal_context_create
synaptik_metal_context_release
synaptik_metal_buffer_create
synaptik_metal_buffer_release
synaptik_metal_buffer_upload
synaptik_metal_buffer_download
synaptik_metal_mpsgraph_executable_create
synaptik_metal_mpsgraph_executable_release
synaptik_metal_mpsgraph_executable_run
synaptik_metal_neg_kernel_pipeline_create
synaptik_metal_neg_kernel_pipeline_release
synaptik_metal_neg_kernel_pipeline_run
```

The version function returns unsigned value `4`. The removed
`synaptik_metal_mpsgraph_neg_executable_create` symbol is not exported. All other functions return
a signed 32-bit status. Context, buffer, MPSGraph-executable, and custom-pipeline values cross the
boundary as separate opaque `void *` handle families. Buffer sizes and offsets are unsigned
64-bit values. Counts, ranks, value indices, node fields, and the node-schema version are unsigned
32-bit values. Custom-pipeline `element_count` is carried as `uint64_t`, but custom NEG creation
accepts only `1..UINT32_MAX`; it returns unsupported shape outside that domain rather than
narrowing the value. Created handles use caller-supplied output cells, which remain null on
failure.

The graph creator accepts node schema `3` and a bounded fixed-width table:

```c
typedef struct {
    uint32_t operation;       /* NEG=1 or one of affine operations 6..10 */
    uint32_t attribute_kind;  /* NONE=0, TARGET_SHAPE=1, PERMUTATION=2, AXIS=3 */
    uint32_t first_input;
    uint32_t second_input;    /* UINT32_MAX for every accepted node */
    uint32_t output;
    uint32_t attribute_count;
    uint32_t axis;            /* normalized axis or UINT32_MAX */
    uint32_t reserved;        /* zero */
    uint64_t attribute_values[16];
} SynaptikMetalMpsGraphNodeV3; /* exactly 160 bytes; payload begins at byte 32 */
```

Its exact signature is:

```c
int32_t synaptik_metal_mpsgraph_executable_create(
    void *context, uint32_t node_schema_version,
    uint32_t value_count, const uint32_t *value_ranks,
    const uint64_t *value_dimensions,
    uint32_t node_count, const SynaptikMetalMpsGraphNodeV3 *nodes,
    uint32_t feed_count, const uint32_t *feed_indices,
    uint32_t target_count, const uint32_t *target_indices,
    void **out_executable);
```

The dimension table has `value_count * 16` cells with used positive axes followed by zero padding.
Feeds are unique and available before node zero; nodes are topological and produce fresh values;
targets are unique produced values. No-attribute nodes require zero attribute count/payload and the
axis sentinel. Target Shapes and complete permutations use `attribute_count` payload cells; axis
forms use count one, the normalized `axis`, and a zero payload. Every other cell is zero or its
required sentinel. Native validation checks exact operation/attribute pairing, ranks `1..16`,
positive dimensions, target Shapes, permutations, axes, `NEG` equality, and affine result
geometry. An affine result cannot feed another node. Unknown operations, including withdrawn wires
`2..5`, wrong sentinels, incompatible Shapes, unused values, malformed indices or payloads, and
wrong schema versions fail closed.

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

## Prepared NEG and affine execution

Creation consumes a validated, topologically ordered whole-partition description. Native code
creates fixed-shape `FLOAT32` placeholders and lowers typed nodes to MPSGraph negation,
`reshapeTensor:withShape:name:`, `broadcastTensor:toShape:name:`,
`transposeTensor:permutation:name:`, `expandDimsOfTensor:axis:name:`, or
`squeezeTensor:axis:name:`. It verifies each affine result Shape and compiles one shape-specialized
executable. The executable owner retains ordered feed and target Shapes, byte extents,
stable-to-framework permutations, and the originating context.

Each create and run call opens a local Objective-C `@autoreleasepool` inside its exception
boundary. The executable box crosses the pool only through `__bridge_retained`; every success and
early-failure return drains temporary framework objects before returning through the C ABI.
Each run binds ordered input and supplied output `MTLBuffer` objects through
`MPSGraphTensorData`. It sets `waitUntilCompleted = YES`, submits the executable once on the
context command queue, checks the completion error and returned-result list, and returns only
after the supplied destinations are usable. Synaptik requests no result-synchronization blit and
performs no explicit post-execution device or host copy. MPSGraph may still use internal temporary
storage.

Affine targets are supplied full positive logical byte extents and receive canonical logical
coordinate order. That dense physical choice is backend-private: logical view strides, offsets,
and `isView` metadata remain unchanged, and it does not imply aliasing with the source.

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
copy. Every other supported partition, including every affine occurrence, uses the typed MPSGraph
route. The route is selected once during analysis and is never retried, replaced, or repartitioned
in finalization or execution.
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

Run the opt-in real-device cases against the freshly built dylib:

```bash
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test --tests '*Metal*' --rerun-tasks
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :testing:integration-tests:test \
  --tests '*EngineExplicitCompositionMetalIntegrationTest' --rerun-tasks
```

The environment variable is required; ordinary sandboxed runs skip native-device cases. The
foundation coverage proves context/buffer ownership and bounded shared-memory copies. Prepared
coverage proves custom caller and splat execution, backend-local typed logical splats, stable
multi-feed/multi-target ordering, direct supplied outputs, and 5,000 consecutive NEG MPSGraph runs
through one retained executable. Affine coverage exercises all five selectors, adversarial raw
bits, Shapes through rank sixteen, mixed NEG-prefix fan-out, direct dense targets, repeated runs,
exact host publication, and unchanged canonical-only cross-owner transfer. Public Metal-only
Engine cases prove supported NEG and affine execution. A focused subnormal binary graph proves
that a Metal-only Engine rejects ownership before native preparation and that an explicitly
registered CPU owns and executes the graph instead.

## Boundaries

The bridge itself implements no library discovery, packaging, Engine composition, mixed-owner
schedule, CPU fallback, general custom-kernel framework, asynchronous API, buffer pool,
persistent constant buffer, executable serialization, FLOAT16, BFLOAT16, reduction, unary algebra,
MATMUL, or performance claim. Affine lowering adds no custom kernel. The public Java Metal surface
is `MetalCapabilityProvider`, `MetalBackendConfiguration`, and `MetalBackendIntegration`; Engine
accepts an explicitly opened integration through `Engine.builder()`.
