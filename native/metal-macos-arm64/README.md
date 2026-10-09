# Metal native bridge for macOS arm64

## Purpose

This directory builds the local application binary interface (ABI) used by the Synaptik Metal
backend on Apple-silicon macOS. ABI version 7 exports thirteen context, shared-storage buffer,
executable, and bounded custom singleton-`NEG` functions. Its graph creator accepts one bounded
schema-20 program image. The image carries an explicit fixed route, value types, and complete
variable-cardinality operation, attribute,
reference, dimension, gradient, optional storage-layout metadata, and a route-specific
authenticated execution extension; no native type, shape, layout, or plan inference is part of the
boundary.

The schema registry reserves operation wires `1..115` and attribute wires `0..41`. The native graph
can structurally execute exactly 101 operation kinds; production capability is exactly 86 kinds.
Task 0059 adds exact movement/indexing rows and complete positive-stride storage geometry.
Task 0060 adds replacement/fold rows `72`, `76`, `80`, `82`, and `84` plus exact aggregate rows
`106..108`. Task 0061 widens existing `MATMUL=15` without adding a wire: the selected capability
admits no-gradient INT32/INT64 ordered pairs and every positive-static homogeneous
FLOAT32/BFLOAT16/FLOAT16 vector, matrix, batched, and broadcast geometry plus qualified
one-low-plus-FLOAT32 widening pairs. Existing all-FLOAT32 rank-two matrix products retain
MPSGraph; every low or new-geometry MATMUL form selects the fixed custom program. Exact local
identity-prefix last-two-axis transpose inputs retain their physical source, offset, and strides.
Task 0062 implements `MEAN_SQUARED_ERROR=85` as `SUB(prediction,target)`,
`MUL(delta,delta)`, and optional full `SUM` or `MEAN`; P9 admits homogeneous
FLOAT32/BFLOAT16/FLOAT16 selected occurrences and their generated gradients.

Task 0063 adds custom-only wires `94=SORT`, `95=ARGSORT`, `96=TOP_K`, `109=ARG_MAX`, and
`110=ARG_MIN`. The first three admit all seven carriers and the arg operations admit six numeric
carriers. Ranks are in `1..16`; every dispatch dimension, count, stride, selected extent, K, and
derived logical index is bounded to unsigned 32 bits before resource creation. The integer-only
comparator preserves raw selected words, stable order, NaNs-last sorting, signed-zero order, top-K
pair order, and NaN-preferred arg ties.

Task 0064 adds custom-only wires `36=CONV2D`, `37=CONV3D`, `97=MAX_POOL2D`,
`98=AVERAGE_POOL2D`, `99=MAX_POOL3D`, and `100=AVERAGE_POOL3D`. The selected capability admits qualified
homogeneous FLOAT32/BFLOAT16/FLOAT16 convolution and pooling; maximum pooling also
retains FLOAT64. All dimensions, counts, strides, dilations, paddings, spans, logical coordinates,
and one-dimensional grids are checked before resource creation. Pooling caps the kernel-position
product at 65,536. Exact local singleton-height affine inputs authenticate Conv1d and Pool1d.
Convolution preserves grouped term/bias order. Maximum pooling selects raw words by NaN-first,
positive-zero-over-negative-zero, first-logical-winner order with negative-infinity padding.
Average pooling uses the full kernel-position divisor and conceptual positive-zero padding.
Non-overlapping maximum-pool backward closes; overlap accumulation and Conv3d backward remain
fail-closed.

Task 0065 adds custom-only `101=DROPOUT` and `102=INITIAL_STATE`. INITIAL_STATE writes one
canonical raw INT64[2] key/counter output. Homogeneous
FLOAT32/BFLOAT16/FLOAT16 dropout writes the value, one-byte canonical BOOL saved mask, and raw next
state. The fixed SplitMix64 V1 kernel maps each logical ordinal, compares the top 53 bits with the
exact host threshold, and advances the counter modulo $2^{64}$. Dropped output is raw positive
zero; kept low output retains FLOAT32 computation until one final narrowing. State has one writer,
all live outputs use ordinary run-local buffers, and direct MPSGraph creation rejects both wires.

Task 0069 Slices 1 through 3 add source-owned custom-only `114=L1_NORM`, `70=SCATTER_ADD`,
and `112=VARIANCE` for narrowly qualified FLOAT32 occurrences. L1 accepts one canonical positive-static rank-one
no-gradient input, ordered
multi-axis `[0]`, and a canonical scalar or retained `[1]` output. Its one output thread raw-clears
every contributor sign bit, initializes from ordinal zero, performs exactly `N-1` source-ordered
safe binary32 additions, and serializes one final raw word; `N=1` performs no addition.

ScatterAdd accepts positive canonical rank-one base/result and index/update extents, axis zero, no
gradient flags, and a materialized canonical INT32 or INT64 index partition feed. The compiler
places explicit `CONTIGUOUS` after its generated-zero expand. Native execution completely scans
`[0,D)` before creating a command buffer or mutating output. One thread owns each target, raw-loads
the base once, applies the sole safe addition to every matching update ordinal in source order,
and stores once. Duplicate targets are retained, unaddressed cells copy the raw base word, and no
atomic operation exists.

Variance accepts only a canonical no-gradient rank-one `[1]` input, statistical axis `[0]`,
correction zero, and a canonical scalar or retained `[1]` output. One dispatched thread evaluates
exactly `DIV(x,+1)`, `SUB(x,mean)`, `MUL(difference,difference)`, and `DIV(square,+1)` using the
metadata-owned raw binary32 `+1`, then stores the sole canonical cell once. Every finite input,
including either zero and every subnormal under all admitted DAZ/FTZ behavior, produces exact
positive zero; NaN and either infinity sign produce NaN class. There is no classifier, aggregate
add, reciprocal, FMA, clamp, tolerance, fallback, retry, timing, or alternate publication path.

The runtime compiler fixes `MTLMathModeSafe` and `MTLLanguageVersion3_2`. The exact assembled
production source passes the pinned Xcode 27.0/Metal 32023.921/macOS SDK 27.0 compiled-MSL/AIR
audit with `metal3.2`, no-fast-math, warnings as errors, and the explicit SDK isysroot. AIR exposes
one unflagged `fadd` in each L1/ScatterAdd kernel; the VARIANCE kernel contains exactly two
unflagged `fdiv`, one `fsub`, one `fmul`, no `fadd` or FMA, and one static store. Scatter contains
no atomics.

Task 0071 historically added anchor-epilogue steps under ACCELERATOR while the bridge was ABI 5 with thirteen
exports. An eligible MATMUL suffix is optional literal scalar multiplication, at most one ordinary
right-aligned tensor ADD, and optional terminal RELU or no-gradient CLAMP. An eligible Conv2d
suffix is at most one external ADD followed by optional RELU or no-gradient CLAMP; Conv2d never
absorbs scalar multiplication. Intrinsic rank-one `[C]` bias remains only the third Conv2d input.
An external rank-one addend is `[W]`, while `[1,C,1,1]` expresses channel broadcasting. Every
suffix intermediate is private, single-consumer, non-target, canonical, and absent from the
materialized slot table. Each admitted anchor executes as one safe-math kernel dispatch and one
final store, with no native retry or fallback. The pinned AIR audit covers all new FMA, multiply,
add, and final-store sites plus NaN, signed-zero, infinity, subnormal DAZ/FTZ, and clamp boundaries.

The remaining 29 production rows fail closed before native creation. A structurally valid
registered operation without a native recipe returns the dedicated unsupported-operation status
rather than masquerading as malformed input. Candidate and route identity are version 30. Java
owns exactly three prepared-route identities: custom singleton NEG wire 1, MPSGraph wire 2, and
shared custom-program wire 3. Schema 20 embeds wire 2 or 3 in each graph image; every other
schema or route value fails closed. The exhaustive Java
structural catalog adds no native route selection, capability, autotuning, fallback, telemetry, or
performance authority.

BFLOAT16 and FLOAT16 (`7`) occurrences use only authenticated custom Metal source. This includes
exact homogeneous no-gradient raw-preserving RESHAPE, simple PERMUTE, materializing CONTIGUOUS,
SLICE, CONCAT, and TILE. The native parser rejects any MPSGraph-route image containing either low
type before operation lowering. No low occurrence is qualified through MPSGraph, classic MPS, MPP,
pointwise source generation, or anchor-epilogue fusion, and execution has no retry or CPU fallback.
The complete frozen, formerly ACCELERATOR Metal FLOAT32 target set has a homogeneous BFLOAT16 and FLOAT16
recipe. Schema 20, route wires 2/3, identity 30,
native ABI 7, and the existing operation and attribute wires retain their allocations.


Raw storage is always an unsigned 16-bit word. Exact kernels copy or select that word, predicates
classify it as an integer encoding, and casts use the declared bit-exact conversion. Arithmetic
kernels integer-decode each BFLOAT16 or FLOAT16 input to its exact FLOAT32 value, retain FLOAT32
working values and accumulators (including corresponding safe-mode FMA), and integer-encode each
observable low result exactly once with round-to-nearest, ties-to-even. No input, contributor,
accumulator, public value, or saved value is pre-narrowed. Sorting, extrema, maximum pooling, masks,
indices, RNG state, and selected values remain exact rather than taking arithmetic tolerance.

The low custom library is compiled with `MTLMathModeSafe`. A represented input is widened exactly;
at a FLOAT32 arithmetic site a FLOAT32 subnormal operand may be consumed either exactly or as
same-sign zero (DAZ), and a FLOAT32 subnormal site result may remain exact or become same-sign zero
(FTZ). Integer final narrowing itself does not flush representable low subnormals. Arithmetic NaNs
retain NaN class and domain behavior, but final low arithmetic publication canonicalizes payload
and sign; exact selection/order/extrema paths retain the selected raw NaN. Primitive/FMA signed
zero is retained through final narrowing and may take either sign when the exact arithmetic result
is zero. Exact paths preserve zero bits; MIN chooses negative zero, MAX chooses positive zero,
SIGN preserves a zero's sign, RELU maps negative zero to positive zero, and average pooling
preserves negative zero only when every contributor is negative zero and none is padding.

For admitted nodes, the version-30 workload signature binds operation wire, source/target carrier
types and widths, every Shape, normalized axis/batch/tuple fact, complete raw attributes, exact
scalar bits, variadic input/output order and count, complete encoded logical storage-layout
geometry, independently safe physical materialization, and the canonical schema-20 execution
extension. The schema-20 and identity-30 cutover has no compatibility reader or migration alias;
every other schema or identity fails closed.

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
rpath set, ABI 7, node schema 20, required frameworks, and fixed ad-hoc identifier. It contains no
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

## ABI version 7

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

The version function returns unsigned value `7`. All other functions return a signed 32-bit
status. Context, buffer, program-executable, and custom-pipeline values cross the boundary as
separate opaque `void *` handle families. Buffer sizes and offsets are unsigned 64-bit values.
Created handles use caller-supplied output cells, which remain null on failure. ABI 7 removes the
former `synaptik_metal_context_certification_environment` export.

The graph creator's exact signature is:

```c
int32_t synaptik_metal_mpsgraph_executable_create(
    void *context,
    const uint8_t *program,
    uint32_t program_bytes,
    void **out_executable);
```

`program` is one canonical little-endian schema-20 image of at most `INT32_MAX` bytes:

```text
124-byte header
value_count × 40-byte value descriptors
node_count × 32-byte node descriptors
dimension_count × u64 dimensions
stride_count × u64 element strides
reference_count × u32 value references (immediately followed by attributes; no padding)
attribute_word_count × u64 attribute words
CUSTOM_PROGRAM only:
  step_count × 40-byte step records
  member_count × u32 node positions
  binding_count × 24-byte binding records
  materialized_count × u32 program-value indices
  instruction_count × 64-byte typed execution instruction records
  canonical ASCII manifest
  32-byte SHA-256 manifest digest
```

The 31 header words are magic `SM20` (`0x30324d53`), schema `20`, header bytes `124`, total
bytes, fixed route, generator schema/extension flag, the eight core
counts, the five execution-record counts, manifest/digest byte counts, generated-unit count,
generated/fixed/total source byte counts, first rejected node, cap reason, and the per-function,
generated-source, and total-source caps. Route `2=MPSGRAPH` requires generator schema, extension
flag, all extension counts, source sizes, and caps to be zero and physically omits every extension
section. Route `3=CUSTOM_PROGRAM` requires generator schema `2`, extension flag `1`, a nonempty
canonical manifest with a 32-byte digest, and parser-recomputed records, source sizes, and caps.
Schema 19 and its profile-bearing header fail before native resource creation or mutation.

A value descriptor is `{type, rank, dimension_offset, stride_offset, flags, layout_kind,
storage_offset, referenced_span}`; the final two fields are unsigned 64-bit element counts. Stable
type wires are `1=FLOAT32`, `2=INT32`, `3=BOOL`, `4=FLOAT64`, `5=BFLOAT16`, `6=INT64`, and
`7=FLOAT16`; rank is `0..16`. Flag bits are `requiresGrad`, layout-present, view, and dense-physical.
A missing layout uses stride offset `UINT32_MAX`, kind/offset/span zero, and no layout flags. A present layout names
one rank-sized contiguous stride-pool range and kind `1=DENSE_CONTIGUOUS`,
`2=DENSE_WITH_OFFSET`, `3=STRIDED`, or `4=BROADCAST_ZERO_STRIDE`. A node descriptor is
`{operation, attribute_kind, input_offset, input_count, output_offset, output_count,
attribute_offset, attribute_word_count}`.

References are ordered feeds, targets, then each node's inputs and outputs. Descriptor offsets must
name exactly those contiguous subranges. Dimensions, strides, references, attributes, and any
extension sections are contiguous with no gaps, overlap, or trailing data. The optional alignment
word must be zero.

Feed count may be zero only when topology produces every value, as in Task-0065 INITIAL_STATE.
The Java FFM boundary then passes no fabricated feed pointer; target and node reference sections
remain ordinary nonempty image data. Execution binds the produced value buffer normally.
Native validation checks all arithmetic, section bounds, reserved bits, layout kind/span
reconstruction, registered operation cardinality, attribute pairing and word count,
type/rank/dimension rules, topological availability, feed and target uniqueness, and executable
operations' exact Shape and state contracts before graph construction. Unknown wires, malformed
sections, unavailable values, incompatible Shapes, and wrong schema versions fail closed as invalid
arguments. Scatter indices must be a feed and
receive their complete bounds scan before command encoding. A well-formed registered operation
outside current execution capability returns status 13. Java independently authenticates the
profile-free program before native creation.

Task 0066 broadens selected existing wires without changing schema, ABI, or operation counts.
All 49 ordered CAST pairs use integer-defined Model conversion, including direct ties-to-even
BFLOAT16/FLOAT16 rounding, signed zero, gradual underflow, infinity overflow, deterministic NaN
handling, saturating floating-to-signed conversion, modular INT64-to-INT32 narrowing, and exact BOOL
mapping. Four-carrier floating classification and the fourteen non-mixed-low promoted floating
WHERE signatures are bit-defined. GATHER, ONE_HOT, replacement scatter, and ND/index movement
accept INT32 or INT64 indices.

All seven carriers enter exact affine movement and canonicalization. Logical zero-stride
descendants are admitted only from authenticated local EXPAND provenance, while every separately
bound physical descriptor stays safe and dense where materialized. External unsafe-stride,
overlapping, empty, dynamic, and unresolved layouts fail closed. Selected occurrences use one
deterministic custom whole-program route; bounds and destination uniqueness are proved before
mutation, and there is no retry, fallback, timing, autotuning, or host repair. UNFOLD_AXIS accepts
every positive static window that fits the selected extent.

Task 0060 custom replacement and signed slice update preserve all seven carriers. Scatter accepts
canonical INT32/INT64 indices and preflights every tuple's bounds plus global destination
uniqueness before copying or replacement. FOLD_AXIS accepts every non-BOOL carrier; FOLD2D/FOLD3D
accept FLOAT64/FLOAT32/BFLOAT16/FLOAT16 when every stride is at least the effective dilated kernel.
Their output-centric kernels give each output cell one writer, copy its sole in-bounds contributor,
skip conceptual padding, and otherwise write exact carrier zero. INT32/INT64 PROD uses stable
row-major modular multiplication; BOOL ALL/ANY uses exact zero/one identities. Empty axes are point
identities and never imply zero-extent support.

Task 0064's six kernels are output-cell-owned and require no scratch, atomics, host calculation,
repair, retry, or fallback. Convolution iterates groups and every kernel-position/channel term in
fixed logical order, injects positive-zero padding as an ordinary multiplicand, and uses the
optional bias as the initial accumulator. Mixed BFLOAT16/FLOAT32 inputs convert each source term
to FLOAT32 before multiplication. Maximum pooling compares integer carrier keys, prefers every NaN
over every number, positive zero over negative zero, and the first logical coordinate on ties, then
copies the selected raw bits. Average pooling includes every padded kernel position in the fixed
divisor, uses FLOAT32 accumulation and one final division, and preserves the specified all-negative-
zero result. All writes remain one-writer and direct.

Task 0065 kernels require no scratch, host payload inspection, host output repair, retry, fallback,
or mutable generator state. Threshold and complement words are fixed at cold lowering from the raw
binary64 probability. The mask remains a live ordinary output until every generated WHERE consumer
finishes; target subsets do not elide sibling mask or next-state production. Exact BOOL WHERE
accepts authenticated local affine views. Ordinary nested MPSGraph affine steps materialize into
dense assigned buffers, so custom WHERE derives canonical physical strides and zero offset from
the representation actually bound at runtime rather than reapplying declared logical-view
geometry. Repeated runs, sessions, and concurrent invocations therefore share only immutable
executable metadata, never a counter.

The prior FLOAT32 scalar `ADD/SUB/MUL/DIV` and `RECIPROCAL` domain remains canonical rank `1..16`
with equal input/output Shape and no-gradient input and output. Each scalar arithmetic recipe
creates one exact four-byte raw FLOAT32 MPSGraph constant with Shape `[1]` and applies exactly one
corresponding binary primitive with the input primary and scalar secondary. `RECIPROCAL` creates
exact raw `+1.0f` with Shape `[1]` and applies exactly one division with one primary and the input
secondary. MPSGraph broadcasting supplies the output Shape; no full-tensor constant is
materialized. The four exact raw discrete unaries require canonical FLOAT32 rank `1..16`, equal
input/output Shape and gradient eligibility, and no attributes.

Raw structural fixtures additionally create prior wires `38`, `50`, `53..54`, `56..59`, and
`65..68` and all Task-0059 wires `39` and `69..84` under the selected capability,
production-exact convolution/pooling wires `36`, `37`, and `97..100`, Task-0065 custom wires
`101..102`, aggregate wires `106..108`, and structural-only wires `111..113` and `115`. Task-0059
recipes cover direct cast/index/pad/slice/concat/tile/im2col/col2im selectors and explicit stack,
fold-axis, and 3D-window compositions. Task-0060 adds stable log-sum-exp, correction-aware
variance/standard-deviation, and L2-norm structural compositions. Wires `70` and `114`, plus the
exact wire-112 singleton `[1]`/axis-`[0]`/correction-zero occurrence, reject direct MPSGraph
creation and use their fixed Task-0069 custom kernels. Non-domain VARIANCE retains its existing
direct MPSGraph structural recipe. Task-0061 retains only all-FLOAT32 rank-two MATMUL in the
MPSGraph recipe; typed custom forms are rejected by that route. Task-0064 direct/composed MPSGraph
family metadata remains structural only. Its six production rows, both Task-0065 random rows, and
all three exact Task-0069 occurrences always select the custom program. Structural creation and
execution do not widen production capability.

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
| 10 | `SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE` | A buffer has the wrong device or extent, an input aliases an output, or distinct output slots alias one resource. |
| 11 | `SYNAPTIK_METAL_STATUS_EXECUTION_FAILED` | Synchronous execution, unusable threadgroup geometry, completion reporting, or returned-result validation failed. |
| 12 | `SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED` | Fixed custom-kernel compilation, function lookup, target validation, or pipeline creation failed. |
| 13 | `SYNAPTIK_METAL_STATUS_UNSUPPORTED_OPERATION` | The image is structurally valid but names a registered operation outside current execution capability. |

Unknown integers remain unknown and fail closed on the Java side with the raw status retained.
The deterministic Java fake/native seam is the accepted error-matrix evidence; real-device tests
exercise successful execution and do not manufacture framework failures.

## Prepared whole-partition execution

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
validated declared output. Direct all-FLOAT32 rank-two MATMUL compilation explicitly sets and reads
back `reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathNone` and rejects native creation
unless that control is available. It verifies every result Shape and compiles one
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
download, and CPU/Metal prepared transfer accept ranks `0..16` for all seven public data types:
`FLOAT32`, `FLOAT64`, `BFLOAT16`, `FLOAT16`, `INT32`, `INT64`, and `BOOL`. Exact big-endian
canonical byte widths are retained at the public boundary; native shared storage is byte-preserving.
Every BOOL upload and downloaded/publicized BOOL byte is validated as exactly `0` or `1`.

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
typed MPSGraph route, except an eligible singleton NEG may use its dedicated custom route. The
route is selected once during analysis and is never retried, replaced, or
repartitioned in finalization or execution. This private implementation-domain boundary is not
capability narrowing, tuning, fallback, or a performance claim.

## Shared exact custom whole-program execution

Any schema-20 program containing an exact custom node or a MATMUL outside the retained
all-FLOAT32 rank-two MPSGraph slice uses one retained custom-program handle. Creation authenticates
the frozen SHA-256 values of all ten reviewed fixed-kernel components and their ordered
84,541-byte total. The sole production pointwise emitter first traverses a structured
no-allocation count sink, validates unit, instruction, per-function, generated, and total caps,
allocates the exact byte count, traverses the same emitter once for compiler input, independently
authenticates the assembled source, and only then compiles with `MTLMathModeSafe` and
`MTLLanguageVersion3_2`. It creates one immutable pipeline and metadata buffer per custom step and
cold-compiles each interleaved existing node as a typed one-node MPSGraph executable. Task-0061
contributes seven typed MATMUL kernels: four modular INT32/INT64 signatures, FLOAT32/FLOAT32, and
both ordered BFLOAT16/FLOAT32 mixed signatures.

Java declares and assigns buffers only for the sorted compact materialized-value set. Every step
binding names both its compact slot and authenticated program value; direct targets resolve through
the same map. The per-run native-address workspace therefore has one entry per materialized value,
not one per logical value. The fixed route, ordered steps/members/bindings, generated instructions,
materialized set, canonical manifest, integer source byte counts, and manifest SHA-256 cross in the
authenticated schema image; no source text, source hash, function name, hidden intermediate, or
input-dependent choice crosses the ABI. Fixed, generated, and assembled source ownership remains
native.

An eligible maximal linear chain of canonical no-gradient FLOAT32 `FLOOR`, `CEIL`, `SIGN`, or
`RELU` nodes has fully static positive rank `1..16` and element count `1..UINT32_MAX`, and is
divided deterministically into generated units of length `2..8`; when an eight-node take would
leave one node, the preceding unit takes seven. Each generated kernel loads one raw word, applies
the frozen helpers in order, stores one raw word, and uses a 32-byte, eight-aligned `PointMeta`
`{u64 elementCount, u64 gridWidth, u64 gridHeight, u32 scalar, u32 reserved}`. Native reconstructs
exact cap precedence and first-rejected-node state. Creation uses
`MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo` and accepts exactly named
`input`, `output`, and `meta` bindings. Xcode reflects `input` and `meta` read-only but the output
device pointer read-write (compiled AIR independently proves the store-only use), so validation
requires that actual strongest runtime contract. The `PointMeta` pointer and struct are read-only,
size 32/alignment eight, and expose exactly the five ordered members and offsets above. The plan
caps are 32 generated units, 256 instructions, 16,384 UTF-8 bytes per function, 262,144 generated
bytes, and 1,048,576 total bytes. Barriers never fuse, and all non-generated fixed custom and
MPSGraph steps retain their established behavior.

Anchor-epilogue steps use generator-schema-2 typed instructions and bind suffix source order, ADD
side and external role, raw scalar/clamp words, anchor family, types, Shapes, layouts, and
gradient metadata in the authenticated manifest/image digest. Native independently reconstructs
the same grammar and broadcasts. The 6,096-byte/eight-aligned `AnchorEpilogueMeta` embeds the
existing `DataMeta`, right-aligned add strides/offset, raw scalar and clamp words, and flags. The
fixed source contains six MATMUL type/add variants and two Conv2d add/no-add variants. A test-only
observer immediately before `dispatchThreads` proves each fused step issues one physical dispatch;
no suffix logical intermediate owns a native slot or store.

One Java/native run call authenticates the complete compact materialized-slot table and exact
direct targets, rejects one physical buffer reused by distinct live slots, and preserves each
target's required output alias to its own slot. Before any dispatch or write, it scans every caller
BOOL feed
consumed by a BOOL-domain node and rejects every byte other than zero or one. SCATTER_ND likewise
preflights the complete index tensor for bounds and duplicate destinations before its copy stage.
The hot native route consumes supplied handles directly without allocating a mirror collection,
executes stable program order, and submits consecutive custom nodes through one framework command
buffer. Interleaved existing nodes execute their already-compiled resource internally; Java
performs no per-node downcall. Classification and exact discrete unary kernels inspect raw FLOAT32
words, logic writes exact zero/one bytes, exact unary kernels preserve the Model-required raw
classes, replacement and non-overlap fold kernels preserve carrier words, and WHERE copies the
selected branch word without floating arithmetic. Internal BOOL producers are closed: custom
comparison/classification/logic/reduction kernels write exact zero or one, while the owned nested
ONE_HOT selector is constructed with typed BOOL zero/one constants. Arbitrary framework BOOL
producers cannot enter the recipe. Direct candidates for custom-domain nodes remain
package-private structural regressions and never replace the production custom route. There is no
host staging, retry, fallback, hot compilation, atomic update, or multi-writer output cell.

## Task-0053 evidence-only source

`src/task0053_candidate.metal`, `src/task0053_integer_core.h`, and the deterministic header
generator remain evidence artifacts for the proposed raw-word integer/fixed-point `EXP` and stable
`SIGMOID` algorithms. Production does not import the candidate header, carry a Task-0053 gate,
append that source to the runtime library, or expose Task-0053 function/dispatch metadata. Java
capability and route availability remain false, so ordinary preparation cannot compile or execute
the candidate. Reintroducing it requires a separately approved production cutover rather than a
runtime boolean gate.

## Ownership and concurrency

A successful create transfers one retained handle to Java, and its matching release consumes that
handle once. Malformed create results fail closed: success with a null output is rejected, while a
failure that publishes a non-null handle releases that handle once and preserves cleanup failure
as suppressed evidence. Buffer wrappers and the selected persistent resource retain context child
leases, so context owner close rejects new work but defers physical context release until the
final child closes. `PreparedExecution` owns the selected resource exactly once; each run
separately owns its initialized constant buffers and outputs. MPSGraph and custom-program runs
additionally own one native-address workspace; singleton custom runs own none. Caller input buffers
are borrowed.

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
  --tests '*MetalIntegrationTest' --rerun-tasks
```

The environment variable is required; ordinary sandboxed runs skip native-device cases. The
foundation coverage proves context/buffer ownership and bounded shared-memory copies. Prepared
coverage proves custom caller and splat execution, backend-local typed logical splats, stable
multi-feed/multi-target ordering, direct supplied outputs, and 5,000 consecutive NEG MPSGraph runs
through one retained executable. Affine coverage exercises all five selectors plus same-Shape
`CONTIGUOUS`, adversarial raw bits, Shapes through rank sixteen, chained local views, fan-out,
intermediate and final direct dense targets, repeated/concurrent runs, exact host publication, and
unchanged canonical-only cross-owner transfer. ABS coverage executes the selected capability across signed
zeros, subnormal and normal boundaries, ordinary finite values, maximum finite values, infinities,
and multiple signed quiet/signaling NaNs; it also covers custom/MPSGraph composition,
intermediate/direct targets, reuse, concurrency, independent contexts, input preservation, and
close rejection. `FLOOR`, `CEIL`, `SIGN`, and `RELU` coverage executes the selected capability through the
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
composition, repeated and concurrent sessions, and exact copy identities. MATMUL coverage executes vector dot, vector/matrix, matrix/vector, rank-two, batched,
broadcast, rank-16, and exact local-transpose forms. It covers all four integral promotions with
modular extremes, both ordered mixed BFLOAT16 signatures with exact widening, FLOAT32 special
classes and FMA/zero-sign certificate cases, physical offset/stride canaries, direct targets,
immutable inputs, reuse, independent sessions, and concurrency. Public CPU-free Metal-only Engine
cases additionally exercise integral execution and general selected
FLOAT32 forward/generated-gradient and mixed-carrier execution.

MSE coverage executes `NONE`, `SUM`, and `MEAN` through direct targets, repeated inputs, retained
executable reuse, isolated sessions, immutable prediction/target buffers, and the same nested
MPSGraph composition inside a custom program. Its independent source-derived oracle recursively
enumerates permitted DAZ/FTZ choices at the subtraction and self-multiplication sites, every
all-contributors-once binary reduction tree, final exact-count division, signed-zero freedoms, and
NaN classification over zeros, subnormal/normal boundaries, maximum finite values, infinities, and
quiet/signaling NaNs. Java/native malformed-image parity rejects wrong reductions, types, ranks,
Shapes, and gradient flags. The fixed source composition against the already qualified primitive
and full-reduction domains is the authorization proof; these executions corroborate it rather than
grant capability by sampling. The packaged CPU-free Engine proof covers sole Metal ownership, all
three publications including four-byte scalars, direct and nested-custom-program execution, reused
and independent sessions, input preservation, excluded-domain and all eight neighboring
normalization/loss-family rejection.

Convolution/pooling coverage checks grouped biased and unbiased Conv2d/Conv3d, exact Conv1d and
Pool1d singleton-height compositions, dilation/stride/padding/ceil geometry, homogeneous low and
widening execution, maximum raw NaN/signed-zero/tie/infinity ordering, fixed average divisors, and
all-padding windows. Java/native malformed-image parity rejects bad types, ranks, attributes,
Shapes, gradient metadata, external affine layouts, and one-past unsigned-32-bit facts before resource creation. CPU-free public Engine evidence covers forward reuse,
separate and joint Conv2d cotangents, non-overlapping average-pool cotangents, and
FLOAT32/BFLOAT16/FLOAT16 non-overlapping maximum-pool cotangents. Overlap accumulation,
Conv3d backward, attention, and convolution transpose reject before
native execution.

## Boundaries

The bridge itself implements no library discovery, package selection or extraction, Engine
composition, asynchronous API, buffer pool, executable serialization, implicit seeding, alias
promise, or performance claim. It executes only capability-admitted fixed routes. Homogeneous
BFLOAT16/FLOAT16 kernels decode inputs exactly, retain FLOAT32 working values and accumulators, and
narrow each observable low result once. Direct BFLOAT16/FLOAT16 mixed-low operations have no
kernel; explicit FLOAT32 casts provide that boundary.

Selected floating MATMUL contracts each scalar product through one increasing-order loop.
Homogeneous low output narrows once; widening rows publish FLOAT32. Authenticated affine operands
address their physical source directly. There is no tiling, atomics, autotuning, runtime route
selection, retry, or fallback. Generated floating gradients use the same admitted
domain without implying unrestricted training.

Conv3d backward, overlap accumulation, attention,
convolution transpose, asymmetric padding, unsafe layouts, dynamic/empty geometry, scratch,
atomics, and host repair remain unavailable. Portable/configurable RNG, entropy or
random-quality claims, host/eager distribution, and recurrent execution also remain unavailable.
RNN, GRU, and LSTM reject both directions, bias forms, zero-length/no-work cases, and every
gradient.

The public Java Metal surface is `MetalCapabilityProvider`, `MetalBackendConfiguration`, and
`MetalBackendIntegration`; Engine accepts an explicitly opened integration through
`Engine.builder()`.
