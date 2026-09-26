# Metal native bridge for macOS arm64

## Purpose

This directory builds the local application binary interface (ABI) used by the Synaptik Metal
backend on Apple-silicon macOS. ABI version 5 retains the same thirteen context, shared-storage
buffer, executable, and bounded custom singleton-`NEG` exports. Its graph creator accepts one
bounded schema-14 program image. The image carries an explicit fixed route plus value types and
complete variable-cardinality operation, attribute, reference, dimension, and gradient metadata;
no native type or shape inference is part of the boundary.

The schema registry reserves operation wires `1..115` and attribute wires `0..41`. The native graph
creator has 62 structural recipes: wires `1..34`, `38`, `40..54`, `56..63`, and `65..68`.
Structural-only raw fixtures do not widen capability. Java production capability is exactly 45
operations: wires `1..34`, FLOAT32 classification wires `40..42`, BOOL logic wires `43..45`,
FLOAT32 `WHERE` wire `51`, and exact raw discrete unary wires `60..63`, subject to exact profile,
type, Shape, and topology restrictions. The remaining 70 production rows fail closed before native
creation. A structurally valid registered operation without a native recipe returns the dedicated
unsupported-operation status rather than masquerading as malformed input.
Candidate and route identity are version 15. Java owns exactly three prepared-route identities:
custom singleton NEG wire 1, MPSGraph wire 2, and shared custom-program wire 3. Schema 14 embeds
wire 2 or 3 in each graph image; schema 13 and every other route value fail closed. The exhaustive
Java structural catalog adds no native route selection, capability, autotuning, fallback,
telemetry, or performance authority.

```text
Java analysis -> choose fixed whole-partition route -> declare every exact resource
Java prepare  -> compile one typed persistent resource -> retain opaque handle
Java run      -> pass assigned MTLBuffer handles       -> one synchronous native call
Java close    -> release persistent/run resources, context, and FFM lookup
```

The bridge is backend-private. Its handles are process-local ownership tokens, not caller-visible
addresses or serialization values.

## Build prerequisites and local package

Build on an Apple-silicon macOS host with Xcode Command Line Tools:

```bash
./native/metal-macos-arm64/build.sh
```

The script targets exactly arm64 and macOS 26.0, enables automatic reference counting (ARC), sets
the install name to `@rpath/libsynaptik_metal_foundation.dylib`, adds no rpath, and links
Foundation, Metal, and MetalPerformanceShadersGraph. It requires a real local `build/` directory,
rejects symlink or non-directory destinations and any existing non-regular output, compiles in a
private directory, and atomically replaces only
`native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib` after success.

Explicitly replace the linker's incidental signature with the fixed local ad-hoc identity, then
package and independently verify the final signed bytes:

```bash
/usr/bin/codesign --force --sign - --timestamp=none \
  --identifier io.github.pho001.synaptik.metal.foundation \
  native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh \
  native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh \
  native/metal-macos-arm64/build/package-v1/macos-arm64
```

The ignored `build/package-v1/macos-arm64/` directory contains exactly the signed dylib,
`manifest.json`, and `SHA256SUMS`. The canonical schema-1 manifest records the final dylib's
relative name, size, SHA-256, platform, architecture, macOS 26.0 minimum, install name, empty
rpath set, ABI 5, node schema 14, required frameworks, and fixed ad-hoc identifier. It contains no
time, host, absolute path, source revision, product version, SDK version, Team ID, notarization,
provenance, or release field. Packaging the same exact signed input produces byte-identical
manifest and checksum files.

The packager applies the same real-directory checks to `build/` and its package staging parent.
The verifier rejects redundant terminal separators or components before checking the package root
without following a symlink. It parses every raw `otool -L` dependency line as one complete path
plus the exact compatibility/current-version suffix and rejects malformed, control-bearing, or
ambiguous records before applying the system-dependency allowlist.

The checksums detect corruption; because they travel with the artifact, they do not authenticate a
hostile replacement. The ad-hoc signature verifies internal Mach-O integrity but supplies no
publisher identity or Apple trust. This package and its optional local ZIP remain verified local
development artifacts, not Developer-ID-signed, notarized, authenticated, published, or public
release artifacts. Developer ID signing, secure credentials/keychain handling, a notarizable
distribution container, versioning, provenance, and publication require separate release planning
and real externally supplied credentials.

## Optional verified local Gradle archive

Supply the canonical package directory explicitly as an absolute path. The property has no
default, environment fallback, or discovery:

```bash
PACKAGE="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64"
./gradlew :backends:metal:verifyMetalNativePackage \
  -PsynaptikMetalNativePackage="$PACKAGE"
./gradlew :backends:metal:metalNativeLocalZip \
  -PsynaptikMetalNativePackage="$PACKAGE"
```

Both tasks are opt-in. Ordinary `build`, `assemble`, `check`, `jar`, and `test` do not evaluate the
property or run native verification. The ZIP task first runs the existing verifier every time; it
never builds, signs, repairs, installs, extracts, discovers, or caches a library. Blank, relative,
redundantly terminated, and symlink-root package paths fail closed.

The fixed output is
`backends/metal/build/distributions/synaptik-metal-macos-arm64-local.zip`. It is unversioned and
contains only the exact verified package beneath `macos-arm64/`, with reproducible order and
timestamps and modes `0755` for the directory/dylib and `0644` for the metadata. It contains no
source path, product version, provenance, license placeholder, or publication coordinates.

Extraction and lifetime remain caller-owned. Use a permission-preserving extractor, rerun the
same verifier, and pass the extracted dylib's explicit absolute path to
`MetalBackendConfiguration`:

```bash
DEST="$(mktemp -d)"
/usr/bin/ditto -x -k \
  backends/metal/build/distributions/synaptik-metal-macos-arm64-local.zip \
  "$DEST"
./native/metal-macos-arm64/verify-package.sh "$DEST/macos-arm64"
LIB="$DEST/macos-arm64/libsynaptik_metal_foundation.dylib"
```

Keep or remove `DEST` under the caller's own lifecycle; Synaptik performs no automatic install,
extraction, cleanup, classpath lookup, or runtime discovery. `SYNAPTIK_METAL_TEST_LIBRARY` remains
test-only. The archive does not add authentication or redistribution rights, and it deliberately
does not copy the CPU-only `THIRD_PARTY_NOTICES.md`.

## ABI version 5

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

The version function returns unsigned value `5`. All other functions return a signed 32-bit
status. Context, buffer, MPSGraph-executable, and custom-pipeline values cross the boundary as
separate opaque `void *` handle families. Buffer sizes and offsets are unsigned 64-bit values.
Created handles use caller-supplied output cells, which remain null on failure. The graph creator's
exact signature is:

```c
int32_t synaptik_metal_mpsgraph_executable_create(
    void *context,
    const uint8_t *program,
    uint32_t program_bytes,
    void **out_executable);
```

`program` is one canonical little-endian schema-14 image of at most `INT32_MAX` bytes:

```text
64-byte header
value_count × 16-byte value descriptors
node_count × 32-byte node descriptors
dimension_count × u64 dimensions
reference_count × u32 value references
zero u32 alignment word when required
attribute_word_count × u64 attribute words
```

The 16 header words are retained magic `SM13` (`0x33314d53`), schema `14`, total byte count, value
count, node count, feed count, target count, dimension count, reference count, attribute-word
count, fixed route (`2=MPSGRAPH` or `3=CUSTOM_PROGRAM`), and five reserved zero words. A value
descriptor is `{type, rank, dimension_offset, flags}`. Stable type wires are `1=FLOAT32`,
`2=INT32`, `3=BOOL`, `4=FLOAT64`, `5=BFLOAT16`, and `6=INT64`; rank is `0..16`; flag bit zero is
`requiresGrad`; all other flag bits are zero. A node descriptor is
`{operation, attribute_kind, input_offset, input_count, output_offset, output_count,
attribute_offset, attribute_word_count}`.

References are ordered feeds, targets, then each node's inputs and outputs. Descriptor offsets must
name exactly those contiguous subranges. Dimensions and attributes are likewise contiguous with no
gaps, overlap, or trailing data. The optional alignment word must be zero. Native validation checks
all checked arithmetic, section bounds, reserved bits, registered operation cardinality, registered
attribute pairing and word count, type/rank/dimension rules, topological availability, feed and
target uniqueness, and the existing executable operations' exact Shape and state contracts before
graph construction. Unknown wires, malformed sections, unavailable values, incompatible Shapes,
and wrong schema versions fail closed as invalid arguments. A well-formed registered operation
outside current execution capability returns status 13. Java separately authenticates numerical
profile compatibility and rejects every profile-incompatible program before native creation.

The production operation domain is wires `1..34`, `40..45`, `51`, and `60..63`: the existing
arithmetic, affine, indexing, reduction, MATMUL, comparison/extrema/scan, classification,
BOOL/WHERE domain plus exact raw `FLOOR`, `CEIL`, `SIGN`, and `RELU`. Existing positive-rank
requirements still apply where operation semantics require them; the four new exact operations
require canonical FLOAT32 rank `1..16`, equal input/output Shape and gradient eligibility, and no
attributes.

Raw structural fixtures additionally create wires `38`, `46..50`, `52..54`, `56..59`, and
`65..68` only under ACCELERATOR. They exercise direct power, exact raw scalar-data splats with
ordered arithmetic, direct unary selectors, and the finite `LOG1P`, `EXPM1`, `GELU`,
`GELU_TANH_APPROXIMATION`, and `SILU` compositions. These recipes are not production capability:
power and elementary/recursive result sets remain unproved, while scalar basic arithmetic and
`RECIPROCAL` cannot satisfy the current public primitive gradient contract. Java production
preparation rejects them before native creation.

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
| 13 | `SYNAPTIK_METAL_STATUS_UNSUPPORTED_OPERATION` | The image is structurally valid but names a registered operation outside current execution capability. |

Unknown integers remain unknown and fail closed on the Java side with the raw status retained.
The deterministic Java fake/native seam is the accepted error-matrix evidence; real-device tests
exercise successful execution and do not manufacture framework failures.

## Prepared profile-qualified whole-partition execution

Creation consumes a validated, topologically ordered whole-partition description. Native code
creates fixed-shape typed placeholders and lowers typed nodes to MPSGraph negation, absolute
value, ordered addition, subtraction, multiplication, division, reduction sum, reduction mean,
`reshapeTensor:withShape:name:`, `broadcastTensor:toShape:name:`,
`transposeTensor:permutation:name:`, `expandDimsOfTensor:axis:name:`,
`squeezeTensor:axis:name:`, `sliceTensor:starts:ends:strides:name:`,
`concatTensors:dimension:name:`,
`gatherWithUpdatesTensor:indicesTensor:axis:batchDimensions:name:`,
`oneHotWithIndicesTensor:depth:dataType:onValue:offValue:name:`,
`scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:` with
`MPSGraphScatterModeSet`, `isFiniteWithTensor:name:`, `isNaNWithTensor:name:`,
`isInfiniteWithTensor:name:`, `logicalANDWithPrimaryTensor:secondaryTensor:name:`,
`logicalORWithPrimaryTensor:secondaryTensor:name:`, `logicalNOTWithTensor:name:`,
`selectWithPredicateTensor:truePredicateTensor:falsePredicateTensor:name:`, or
`matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:`.
UNFOLD_AXIS creates size-many strided slices in ascending window-offset order, appends a singleton
final dimension to each, and concatenates that ordered list along the final dimension. `GATHER`
uses zero batch dimensions; `ONE_HOT` uses exact `BOOL` scalar constants one and zero.
`CONTIGUOUS` and empty-axis reduction identities use same-Shape
`reshapeTensor:withShape:name:`; keep-dimensions and sum-to-Shape results are reshaped to the
validated declared output. Compilation explicitly sets and reads back
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
`CONTIGUOUS` targets use the ordinary canonical materialization path. Canonical host ingress,
download, and CPU/Metal prepared transfer accept ranks `0..16` for all six public data types:
`FLOAT32`, `FLOAT64`, `BFLOAT16`, `INT32`, `INT64`, and `BOOL`. Exact big-endian canonical byte
widths are retained at the public boundary; native shared storage is byte-preserving. Every BOOL
upload and downloaded/publicized BOOL byte is validated as exactly `0` or `1`.

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
copy. Outside the shared custom-program domain below, ABS and other supported partitions use the
typed MPSGraph route, except an eligible singleton NEG may use its dedicated custom route under
either profile. The route is selected once during analysis and is never retried, replaced, or
repartitioned in finalization or execution. This private implementation-domain boundary is not
capability narrowing, tuning, fallback, or a performance claim.

## Shared exact custom whole-program execution

Any schema-14 program containing a Task-0052 wire, one of wires `40..45` and `51`, or one of exact
raw wires `60..63` uses one retained custom-program handle. Creation compiles only the 26 fixed
reviewed Metal kernels with `MTLMathModeSafe`, creates one immutable pipeline and metadata buffer
per custom node, and cold-compiles each interleaved existing node as a typed one-node MPSGraph
executable. Java declares and assigns a run-owned buffer for every logical intermediate and a
native-address workspace for the stable value table plus direct target aliases. The fixed route
crosses in the authenticated schema image; no source text, function name, hidden intermediate, or
input-dependent choice crosses the ABI.

One Java/native run call authenticates the complete value table and exact direct targets, rejects
one physical buffer reused by distinct live value entries, and preserves each target's required
output alias to its own table entry. Before any dispatch or write, it scans every caller BOOL feed
consumed by a BOOL-domain node and rejects every byte other than zero or one. The hot native route
consumes supplied handles directly without allocating a mirror collection, executes stable program
order, and submits consecutive custom nodes through one framework command buffer. Interleaved
existing nodes execute their already-compiled resource internally; Java performs no per-node
downcall. Classification and exact discrete unary kernels inspect raw FLOAT32 words, logic writes
exact zero/one bytes, exact unary kernels preserve the Model-required raw classes, and WHERE copies
the selected branch word without floating arithmetic. Internal BOOL producers are closed: custom
comparison/classification/logic kernels write exact zero or one, while the owned nested ONE_HOT
selector is constructed with typed BOOL zero/one constants. Arbitrary framework BOOL producers
cannot enter the recipe. Direct candidates for all eleven custom-domain nodes remain
package-private structural regressions and never replace the production custom route. There is no
host staging, retry, fallback, or hot compilation.

## Task-0053 proof-gated source

`src/task0053_candidate.metal` and `src/task0053_integer_core.h` retain the proposed raw-word
integer/fixed-point `EXP` and stable `SIGMOID` kernels. The latter is the single algorithm source
included directly by the C checker model and deterministically expanded by
`generate-task0053-header.py --check` into the embedded NSString. The host has the corresponding
private wire/function/dispatch metadata, but it keeps
`task0053_domain_approved` false and returns status `13` for either structurally valid operation.
The unapproved source is therefore not appended to the active shared exact custom library, no Java capability
or route identity includes it, and ordinary preparation cannot compile or execute it. This dormant
integration is evidence for Task 0053, not current execution capability.

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

After the build, explicit ad-hoc signing, and package commands above, inspect the packaged dylib:

```bash
LIB=native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib
file "$LIB"
xcrun lipo -archs "$LIB"
xcrun vtool -show-build "$LIB"
otool -hv "$LIB"
otool -D "$LIB"
otool -l "$LIB"
otool -L "$LIB"
nm -gUj "$LIB"
/usr/bin/codesign --verify --strict --verbose=4 "$LIB"
/usr/bin/codesign --display --verbose=4 "$LIB"
(cd native/metal-macos-arm64/build/package-v1/macos-arm64 && \
  /usr/bin/shasum -a 256 -c SHA256SUMS)
```

The verifier requires exactly one arm64 Mach-O 64-bit `DYLIB`, minimum macOS 26.0, install name
`@rpath/libsynaptik_metal_foundation.dylib`, no `LC_RPATH`, only Apple system dependencies, all
three required framework links, and exactly the thirteen exports above. It also requires a strict
valid ad-hoc signature with identifier `io.github.pho001.synaptik.metal.foundation` and no Team ID,
plus the exact three-file schema-1 package, modes, canonical manifest, and checksums. It fails
closed rather than signing, repairing, normalizing, or accepting an ambiguous package.

Run the opt-in real-device cases against the packaged dylib:

```bash
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/$LIB" \
  ./gradlew :backends:metal:test --tests '*Metal*' --rerun-tasks
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/$LIB" \
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
close rejection. `FLOOR`, `CEIL`, `SIGN`, and `RELU` coverage executes both profiles through the
production custom route across the full boundary corpus, forces each direct MPSGraph candidate
independently, mixes raw and nested MPSGraph nodes in one recipe, and checks repeated/concurrent
public Engine sessions, direct publication, input preservation, pre-invocation recovery, and close
rejection. The retained exhaustive checker separately covers all `2^32` binary32 words with zero
transform or partition failures. Binary coverage executes each of `ADD`, `SUB`, `MUL`, and `DIV`
separately with row, column, and scalar-tensor broadcasting, operand reversal, chains, fan-out,
repeated operands, published intermediates, repeated runs, exact input preservation, and an
independent bounded DAZ/FTZ/signed-zero/NaN raw-bit oracle. Reduction coverage executes full,
single-axis, multi-axis, empty-axis identity, keep-dimensions, and binding-resolved sum-to-Shape
forms; it checks scalar four-byte publication, direct targets, positive-rank ABS/binary
composition, repeated and concurrent sessions, exact copy identities, and strict-profile
rejection. MATMUL coverage executes direct, left-transposed, right-transposed, and both-transposed
rank-two forms through reusable executables, exact outputs, direct targets, and input preservation.
Public CPU-free Metal-only Engine cases additionally exercise direct MATMUL, no-bias linear's
visible right transpose, and explicitly seeded gradients for both operands without a CPU owner.

## Boundaries

The bridge itself implements no library discovery, package selection or extraction, Engine
composition, mixed-owner schedule, CPU fallback, general custom-kernel framework, asynchronous
API, buffer pool,
persistent constant buffer, executable serialization, FLOAT16, BFLOAT16, masked/extrema/product
reductions, unary algebra beyond exact profile-independent `NEG`, `ABS`, `FLOOR`, `CEIL`, `SIGN`,
and `RELU`, binary comparison/logical/scalar forms, strict/vector/batched/broadcast MATMUL, general
backward execution, alias promise, or performance claim. Accelerator arithmetic, reductions, and
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
