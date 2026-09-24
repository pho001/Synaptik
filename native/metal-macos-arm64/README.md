# Metal native bridge for macOS arm64

## Purpose

This directory builds the local application binary interface (ABI) used by the Synaptik Metal
backend on Apple-silicon macOS. ABI version 4 retains context, shared-storage buffer, executable,
and bounded custom singleton-`NEG` ownership. Its versioned typed whole-partition MPSGraph program
uses node schema 10. Under Java's profile-qualified preflight, both profiles support exact canonical
`NEG`, `ABS`, `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, the explicit
`CONTIGUOUS` canonicalization barrier, canonical positive-rank `FLOAT32` data `GATHER` with
canonical `INT32` indices, canonical positive-rank `INT32`-to-`BOOL` `ONE_HOT`, and canonical
positive-rank `FLOAT32`/`INT32`/`FLOAT32` `SCATTER_ELEMENTS` replacement. `ACCELERATOR`
additionally supports tensor `ADD`, `SUB`, `MUL`, and `DIV`; canonical `FLOAT32` `SUM`, `MEAN`,
and binding-resolved `SUM_TO_SHAPE`; and positive static rank-two `FLOAT32` `MATMUL` with exact
authenticated local rank-two transpose operands. Strict `MATMUL` remains unsupported. No symbol
or ABI-signature change was required.

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

The graph creator accepts node schema `10` and a bounded fixed-width table:

```c
typedef struct {
    uint32_t operation;       /* NEG=1, binaries=2..5, affine=6..10,
                                 CONTIGUOUS=11, ABS=12, SUM=13, MEAN=14,
                                 MATMUL=15, GATHER=16, ONE_HOT=17,
                                 SCATTER_ELEMENTS=18 */
    uint32_t attribute_kind;  /* NONE=0, TARGET_SHAPE=1, PERMUTATION=2,
                                 AXIS=3, REDUCTION=4, DEPTH=5 */
    uint32_t first_input;
    uint32_t second_input;    /* ordered binary/MATMUL/GATHER/scatter index */
    uint32_t output;
    uint32_t attribute_count;
    uint32_t axis;            /* normalized axis, reduction form, or UINT32_MAX */
    uint32_t auxiliary;       /* reduction keep-dimensions or scatter updates input */
    uint64_t attribute_values[16];
} SynaptikMetalMpsGraphNodeV10; /* exactly 160 bytes; payload begins at byte 32 */
```

Its exact signature is:

```c
int32_t synaptik_metal_mpsgraph_executable_create(
    void *context, uint32_t node_schema_version,
    uint32_t value_count, const uint32_t *value_ranks,
    const uint64_t *value_dimensions,
    uint32_t node_count, const SynaptikMetalMpsGraphNodeV10 *nodes,
    uint32_t feed_count, const uint32_t *feed_indices,
    uint32_t target_count, const uint32_t *target_indices,
    void **out_executable);
```

The dimension table has `value_count * 16` cells with used positive axes followed by zero padding.
Value data types are inferred unambiguously from typed node roles: ordinary floating paths remain
`FLOAT32`; `GATHER` indices, `ONE_HOT` input, and the `SCATTER_ELEMENTS` indices role are `INT32`;
`ONE_HOT` output is `BOOL`; and scatter data, updates, and output are `FLOAT32`. A declared value
may be rank zero only when it is a locally produced reduction target. Feeds are
unique positive-rank canonical values available before node zero; nodes are topological, take
positive-rank inputs, and produce fresh values; targets are unique produced values. Native
validation walks explicit unavailable, canonical, and affine-view states. `NEG` and `ABS` accept
one canonical input, require equal input and output Shapes, and produce canonical state. Binary
nodes accept two ordered canonical inputs, require exact right-aligned broadcasting to the declared
output Shape, and produce canonical state. `SUM` and `MEAN` accept one canonical `FLOAT32` input.
Their typed reduction form encodes full, one normalized axis, ordered distinct normalized axes
including an empty identity list, or `SUM`-only sum-to-Shape dimensions, plus exact
keep-dimensions state. Native validation derives and checks the exact output Shape and a positive
term count; a rank-zero result must be a direct target and cannot feed another node. `MATMUL`
accepts positive rank-two canonical operands or exact local `PERMUTE [1,0]` views of canonical
sources, requires exact `[M,K] @ [K,N] -> [M,N]` geometry, and produces canonical state. `GATHER`
accepts canonical positive-rank `FLOAT32` data and `INT32` indices, replaces the selected data axis
with the complete indices Shape, and produces canonical `FLOAT32`. `ONE_HOT` accepts canonical
positive-rank `INT32`, appends its positive depth, and produces canonical `BOOL` with exact byte
values zero and one. `SCATTER_ELEMENTS` accepts ordered canonical `FLOAT32` data, `INT32` indices,
and `FLOAT32` updates of equal positive rank, requires exact indices/update Shape and data agreement
away from its normalized axis, and produces a canonical data-shaped `FLOAT32` replacement result.
Local transpose authentication constrains only an affine operand actually consumed by MATMUL; that
view may also be a target or have another valid affine consumer. Affine nodes accept canonical or
prior affine-view state and produce affine-view state.
`CONTIGUOUS` accepts either available state and produces canonical state. No-attribute nodes
require zero attribute count/payload and the axis sentinel. Target Shapes and complete
permutations use `attribute_count` payload cells; axis forms use count one, the normalized `axis`,
and a zero payload; scatter additionally carries the updates value index in `auxiliary`; depth uses
count one and its positive payload value. Every other cell is zero or its required sentinel. Native
value types, ranks `0..16` under those role restrictions, positive dimensions, target Shapes,
permutations, axes, unary/binary/reduction/MATMUL/indexing/`CONTIGUOUS` Shape rules, and affine
result geometry. Unknown operations, type conflicts, wrong sentinels, incompatible Shapes,
unavailable or invalid value states, unused values, malformed indices or payloads, and wrong
schema versions fail closed. Java separately authenticates the numerical profile and rejects
every profile-incompatible program before the native create call.

### Status values

| Value | C name | Meaning |
|---:|---|---|
| 0 | `SYNAPTIK_METAL_STATUS_OK` | The operation completed successfully. |
| 1 | `SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT` | A required handle, pointer, count, index, topology fact, or output cell is invalid. |
| 2 | `SYNAPTIK_METAL_STATUS_NO_DEVICE` | No default Metal device is available. |
| 3 | `SYNAPTIK_METAL_STATUS_NO_COMMAND_QUEUE` | The device could not create a command queue. |
| 4 | `SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED` | Native owner or buffer allocation failed. |
| 5 | `SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS` | A requested buffer copy is outside its logical extent, or an indexing operand fails its declared extent/depth. |
| 6 | `SYNAPTIK_METAL_STATUS_COPY_FAILED` | Shared buffer contents were unavailable for a requested copy. |
| 7 | `SYNAPTIK_METAL_STATUS_INTERNAL_ERROR` | An Objective-C exception crossed an internal implementation boundary. |
| 8 | `SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE` | Rank, dimensions, checked typed byte geometry, custom element count, or dispatch-grid representation is unsupported. |
| 9 | `SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED` | Graph construction or compilation produced no usable executable. |
| 10 | `SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE` | A buffer has the wrong device or extent, or an input aliases an output. |
| 11 | `SYNAPTIK_METAL_STATUS_EXECUTION_FAILED` | Synchronous execution, unusable threadgroup geometry, completion reporting, or returned-result validation failed. |
| 12 | `SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED` | Fixed custom-kernel compilation, function lookup, target validation, or pipeline creation failed. |

Unknown integers remain unknown and fail closed on the Java side with the raw status retained.
The deterministic Java fake/native seam is the accepted error-matrix evidence; real-device tests
exercise successful execution and do not manufacture framework failures.

## Prepared profile-qualified whole-partition execution

Creation consumes a validated, topologically ordered whole-partition description. Native code
creates fixed-shape typed placeholders and lowers typed nodes to MPSGraph negation, absolute
value, ordered addition, subtraction, multiplication, division, reduction sum, reduction mean,
`reshapeTensor:withShape:name:`, `broadcastTensor:toShape:name:`,
`transposeTensor:permutation:name:`, `expandDimsOfTensor:axis:name:`,
`squeezeTensor:axis:name:`, `gatherWithUpdatesTensor:indicesTensor:axis:batchDimensions:name:`,
`oneHotWithIndicesTensor:depth:dataType:onValue:offValue:name:`,
`scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:` with
`MPSGraphScatterModeSet`, or `matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:`.
`GATHER` uses zero batch dimensions; `ONE_HOT` uses exact `BOOL` scalar constants one and zero.
`CONTIGUOUS` and empty-axis reduction identities use same-Shape
`reshapeTensor:withShape:name:`; keep-dimensions and sum-to-Shape results
are reshaped to the validated declared output. Compilation explicitly sets and reads back
`reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathNone` for MATMUL and rejects native
creation unless that control is available. It verifies every result Shape and compiles one
shape-specialized executable. The executable owner retains ordered feed and target Shapes, inferred
data types, byte extents, stable-to-framework permutations, ordered indexing-domain checks,
checked coordinate geometry and bounded primitive uniqueness scratch for scatter, and the
originating context.

Each create and run call opens a local Objective-C `@autoreleasepool` inside its exception
boundary. The executable box crosses the pool only through `__bridge_retained`; every success and
early-failure return drains temporary framework objects before returning through the C ABI.
Each run validates every supplied input and output resource, then scans every `INT32` indexing
element in stable node order and row-major logical ordinal before constructing tensor data,
dispatching MPSGraph, or writing any target. Scatter completes its full bounds pass before deriving
and sorting complete target coordinates in executable-owned scratch; any duplicate target returns
range status. Bounds also return range status. Java rescans the same retained input buffers to
publish the Model's exact bounds or smallest-later/smallest-earlier duplicate diagnostic. Only
after every descriptor passes validation does the run bind ordered input and supplied output
`MTLBuffer` objects through `MPSGraphTensorData`.
It sets `waitUntilCompleted = YES`, submits the executable
once on the context command queue, checks the completion error and returned-result list, and
returns only after the supplied destinations are usable. Synaptik requests no
result-synchronization blit and performs no explicit post-execution device or host copy. MPSGraph
may still use internal temporary storage.

Affine-view targets are supplied full positive logical byte extents and receive canonical logical
coordinate order. That dense physical choice is backend-private: logical view strides, offsets,
and `isView` metadata remain unchanged, and it does not imply aliasing with the source.
`CONTIGUOUS` targets use the ordinary canonical materialization path. A rank-zero reduction target
uses exactly one `FLOAT32` element and may be downloaded locally as exactly four canonical bytes;
canonical caller ingress accepts exact `FLOAT32` and `INT32`, local canonical `BOOL` targets publish
exact one-byte elements, and CPU/Metal transfer remains positive-rank canonical `FLOAT32` only.

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
copy. Every ABS partition and every other supported partition uses the typed MPSGraph route unless
it is an eligible singleton NEG using the custom route under either profile. The route is selected
once during analysis and is never retried, replaced, or
repartitioned in finalization or execution. This private implementation-domain boundary is not
capability narrowing, tuning, fallback, or a performance claim.

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
through one retained executable. Affine coverage exercises all five selectors plus same-Shape
`CONTIGUOUS`, adversarial raw bits, Shapes through rank sixteen, chained local views, fan-out,
intermediate and final direct dense targets, repeated/concurrent runs, exact host publication, and
unchanged canonical-only cross-owner transfer. ABS coverage executes both profiles across signed
zeros, subnormal and normal boundaries, ordinary finite values, maximum finite values, infinities,
and multiple signed quiet/signaling NaNs; it also covers strict/accelerator composition,
intermediate/direct targets, reuse, concurrency, independent contexts, input preservation, and
close rejection. Binary coverage executes each of `ADD`, `SUB`, `MUL`, and `DIV` separately with
row, column, and scalar-tensor broadcasting, operand reversal, chains, fan-out, repeated operands,
published intermediates, repeated runs, exact input preservation, and an independent bounded
DAZ/FTZ/signed-zero/NaN raw-bit oracle. Reduction coverage executes full, single-axis,
multi-axis, empty-axis identity, keep-dimensions, and binding-resolved sum-to-Shape forms; it
checks scalar four-byte publication, direct targets, positive-rank ABS/binary composition,
repeated and concurrent sessions, exact copy identities, and strict-profile rejection. MATMUL
coverage executes direct, left-transposed, right-transposed, and both-transposed rank-two forms
through reusable executables, exact outputs, direct targets, and input preservation. Public
CPU-free Metal-only Engine cases additionally exercise direct MATMUL, no-bias linear's visible
right transpose, and explicitly seeded gradients for both operands without a CPU owner.

## Boundaries

The bridge itself implements no library discovery, packaging, Engine composition, mixed-owner
schedule, CPU fallback, general custom-kernel framework, asynchronous API, buffer pool,
persistent constant buffer, executable serialization, FLOAT16, BFLOAT16, masked/extrema/product
reductions, unary algebra beyond exact profile-independent `NEG` and `ABS`, binary
comparison/logical/scalar forms, strict/vector/batched/broadcast MATMUL, general backward
execution, alias promise, or performance claim. Accelerator arithmetic, reductions, and rank-two
MATMUL do not imply strict IEEE subnormal preservation: Model-owned DAZ/FTZ applies at its declared
boundaries; finite arithmetic may reassociate and use corresponding FMA choices; and NaN
payload/sign are unspecified. Only a final exact-zero SUM/SUM_TO_SHAPE root with at least two
terms, the mandatory final MEAN quotient, or publication of a complete nonempty exact-zero MATMUL
contraction may choose either zero sign. Empty-axis identities and one-term SUM/SUM_TO_SHAPE
identities remain exact copies. MEAN always performs the mandatory positive-count FLOAT32 quotient
and retains the row's DAZ/FTZ/NaN freedoms, even when the selected count is one. Every MATMUL term
must occur exactly once; products and intermediates receive no publication-only zero-sign freedom.
ABS receives none of those relaxations. Affine composition adds no custom kernel. The only current
backward execution is the Compiler-generated explicitly seeded rank-two MATMUL formula, not a
general training route.
The public Java Metal surface is `MetalCapabilityProvider`, `MetalBackendConfiguration`, and
`MetalBackendIntegration`; Engine accepts an explicitly opened integration through
`Engine.builder()`.
