# Metal native bridge for macOS arm64

## Purpose

This directory builds the native ABI used by Synaptik Metal on Apple-silicon macOS.
ABI 7 retains thirteen context, shared-storage buffer, executable and singleton-NEG exports.
The current graph image is schema 22 (`SM22`) with a 124-byte header, operation wires1..116,
attribute wires0..42, carrier wires1..7 and fixed graph route2 or3.
Coordinated Java candidate/workload/policy/compatibility/codec identities are version42; stale inputs
have no compatibility reader or migration alias.

Java and native code independently validate declared types, shapes, attributes, gradient bits,
canonical result roles, physical layouts/spans and the authenticated execution plan.
The bridge consumes that fixed plan; it does not discover a backend, choose ownership, retry
a route or fall back to CPU. Source/package authentication remains a safety boundary.

Current capabilities and independently exercised numerical/training domains are defined in the
[Metal guide](../../../../docs/backend-guide/metal-backend.md) and
[production support ledger](../../../../docs/backend-guide/production-support.md#metal-production-target).
The ordinary inventory has 115 kinds and 137 signatures; general Scan is separate. Historical
structural-recipe counts are not current capability authority.

The shared FLOAT32/BFLOAT16/FLOAT16 implementation includes ordinary elementary/reduction/
statistical/masked, normalization/loss/attention, convolution/pooling, arithmetic scatter and
selected Compiler gradients. Original storage carriers and per-role POD metadata retain each
input/output type, gradient bit and canonical sibling role; there is no FLOAT32 tensor proxy.
Conv3d/Pool3d train through shared grouped/window/fold owners. Three-family FOLD_AXIS/
FOLD2D/FOLD3D overlap has one writer per output, without atomics or host repair; FLOAT64 remains
non-overlapping exact movement only. Maximum pooling preserves exact winner rules; average
pooling uses the full kernel divisor. Geometry and contributor guards remain operation-specific.

Exact raw operations retain original FLOAT64/integral/BOOL/low carriers without software FP64.
All 49 casts and qualified movement/index/ordering/selection paths preserve their Model semantics.
Seven-carrier signed-step SLICE materializes canonical storage rather than negative physical
strides. INT32/INT64 cumulative arithmetic is exact and modular. Read-only broadcast views
retain authenticated physical strides; writable outputs remain injective. CONCAT/STACK use
dynamic operand references and one disjoint direct-copy child per operand in the shared ordinary/
Scan owner, not N-wide argument binding or intermediate pairwise tensors. There is no
sixteen-operand cap: 17/33/257-input three-family cases exercise views and last-operand-only AD.
Rank zero is one stored value. Qualified empty source/EXPAND/normalization/Scan and zero-output
CONCAT/STACK cotangent paths detect zero extents without zero-grid loads or dispatch.

General Scan uses wire116/attribute42 for an authenticated bounded body/binder plan within the
existing native executable. A true native time loop reuses once-prepared ordinary kernels and
run-private control/storage. Traversal, carry/history ports, original demanded output roles and
per-operand structural activity remain explicit. Inactive contributors are skipped before loads
or index validation; present zero is not absence. Actual Compiler reverse bodies use the same
controlled ordinary owners. There is no Java body interpreter, T-fold unrolling or copied CPU
evaluator. Explicit NN GeneralScan exercises FLOAT32/BFLOAT16/FLOAT16, both directions, optional
bias, changing lengths and inactive NaNs against an independent ordinary RnnCell unroll and
parameter AD, including save/load/fresh SGD. Fixed RNN/GRU/LSTM first AD separately exercises
every floating input/state/weight/bias role, lengths 0/1/3, both directions and optional bias
against independent FLOAT32 finite differences and typed scalar adjoints. Fixed-cell working/
publication rounding remains distinct from the explicit ordinary-cell Scan path.
Owned MLP and Conv3d/overlap AvgPool3d/MaxPool3d journeys exercise three-family forward/MSE,
input/parameter AD, SGD and save/close/load/rebind/fresh training. Cold NN decoding accepts
Model-legal per-role floating types/casts without changing public checkpoint wire or loosening
strict rebind descriptors.

Low-containing partitions have only the fixed CUSTOM_PROGRAM whole-partition route. Their
BFLOAT16/FLOAT16-valued steps use the same typed numerical owners as FLOAT32, never implicit
MPSGraph low execution. Approved FLOAT32-only internal boundary steps do not change that route.
The separate low-precision library is retired; `synaptik_numeric_codec.h` supplies the shared
codec. Model-legal floating promotion is validated per operation; other mixes require explicit casts.
CPU low arithmetic is unchanged; there is no CPU fallback, FP64 arithmetic or FP64 emulation.

Low inputs decode into FLOAT32 working/accumulator sites; each observable output publishes in
its declared carrier, with low results rounded to nearest, ties to even. Exact selection/order/
extrema preserve stored words; arithmetic special values and named DAZ/FTZ sites follow Model.
No blanket tolerance or zero-sign freedom is added. Dedicated EXP/SIGMOID device oracles
remain separate from Runtime policy; admitted scalar/gradient occurrences use all three families.

The fixed corpus authenticates 23 components and their ordered 187,750-byte UTF-8 source before
safe-math Metal 3.2 compilation; the current device compile exposes 165 functions. Generated
pointwise/anchor source retains exact count, digest and cap validation. Low-containing
partitions do not use those fusions. The native body, metadata, pipelines and resources are
prepared once; invocation has no route lookup, compilation, retry or resource-plan rebuilding.

Workload signatures bind exact operation/carrier/shape/attribute/gradient, logical and physical
geometry, original boundary roles, constants, body/control plans and current identity meaning.
The local native package remains deterministic and source-build/ad-hoc signed, not a trusted
public prebuilt. Final distribution acceptance requires its separate frozen source and installed
consumer receipts; a historical task probe does not certify the current binary. The refreshed
operation catalogs have matched three-family coverage; final root and installed-consumer receipts
are recorded separately in the production ledger.

```text
Java analysis -> choose fixed whole-partition route -> declare every exact resource
Java prepare  -> compile one typed persistent resource -> retain opaque handle
Java run      -> pass assigned MTLBuffer handles       -> one synchronous native call
Java close    -> release persistent/run resources, context, and FFM lookup
```

The bridge is backend-private. Its handles are process-local ownership tokens, not caller-visible
addresses or serialization values.

## Build prerequisites and local package

Build on an Apple-silicon macOS host with full Xcode installed at `/Applications/Xcode.app`.
The build explicitly selects its `Contents/Developer` directory; standalone Command Line Tools
are not sufficient for this supported build path.

```bash
./modules/backend/metal/macos-arm64/build.sh
```

The script targets exactly arm64 and macOS 26.0, enables automatic reference counting (ARC), sets
the install name to `@rpath/libsynaptik_metal_foundation.dylib`, adds no rpath, and links
Foundation, Metal, and MetalPerformanceShadersGraph. It requires a real local `build/` directory,
rejects symlink or non-directory destinations and any existing non-regular output, compiles in a
private directory, and atomically replaces only
`modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib` after success.

Explicitly replace the linker's incidental signature with the fixed local ad-hoc identity, then
package and independently verify the final signed bytes:

```bash
/usr/bin/codesign --force --sign - --timestamp=none \
  --identifier io.github.pho001.synaptik.metal.foundation \
  modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib
./modules/backend/metal/macos-arm64/package-local.sh \
  modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib
./modules/backend/metal/macos-arm64/verify-package.sh \
  modules/backend/metal/macos-arm64/build/package-v1/macos-arm64
```

The ignored `build/package-v1/macos-arm64/` directory contains exactly the signed dylib,
`manifest.json`, and `SHA256SUMS`. The canonical schema-1 manifest records the final dylib's
relative name, size, SHA-256, platform, architecture, macOS 26.0 minimum, install name, empty
rpath set, ABI7, node schema22, required frameworks and fixed ad-hoc identifier. It contains no
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
PACKAGE="$PWD/modules/backend/metal/macos-arm64/build/package-v1/macos-arm64"
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
`modules/backend/metal/build/distributions/synaptik-metal-macos-arm64-local.zip`. It is unversioned and
contains only the exact verified package beneath `macos-arm64/`, with reproducible order and
timestamps and modes `0755` for the directory/dylib and `0644` for the metadata. It contains no
source path, product version, provenance, license placeholder, or publication coordinates.

Extraction and lifetime remain caller-owned. Use a permission-preserving extractor, rerun the
same verifier, and pass the extracted dylib's explicit absolute path to
`MetalBackendConfiguration`:

```bash
DEST="$(mktemp -d)"
/usr/bin/ditto -x -k \
  modules/backend/metal/build/distributions/synaptik-metal-macos-arm64-local.zip \
  "$DEST"
./modules/backend/metal/macos-arm64/verify-package.sh "$DEST/macos-arm64"
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

`program` is one canonical little-endian schema22 image of at most `INT32_MAX` bytes:

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

The 31 header words are magic `SM22` (`0x32324d53`), schema `22`, header bytes `124`, total
bytes, fixed route, generator schema/extension flag, the eight core
counts, the five execution-record counts, manifest/digest byte counts, generated-unit count,
generated/fixed/total source byte counts, first rejected node, cap reason, and the per-function,
generated-source, and total-source caps. Route `2=MPSGRAPH` requires generator schema, extension
flag, all extension counts, source sizes, and caps to be zero and physically omits every extension
section. Route `3=CUSTOM_PROGRAM` requires generator schema `2`, extension flag `1`, a nonempty
canonical manifest with a 32-byte digest, and parser-recomputed records, source sizes, and caps.
Older schemas fail before native resource creation or mutation.

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
arguments. Caller-feed controls/indices are validated before dispatch; produced controls are
validated at their consuming step before that operation writes. Preceding completed producers
are not rolled back. A well-formed operation outside current execution capability returns
status13. Java independently authenticates the same current program before native creation.

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
uniqueness before copying or replacement. FOLD_AXIS accepts every non-BOOL carrier with
non-overlapping windows. FOLD2D/FOLD3D accept FLOAT64/FLOAT32/BFLOAT16/FLOAT16 when every
stride is at least the effective dilated kernel; only FLOAT32 also admits overlap. Non-overlap
copies the sole in-bounds contributor or exact carrier zero. Overlapping FLOAT32 image folds
instead give each output cell one writer, start at positive zero and deterministically sum its
in-range column contributors, excluding conceptual padding. No atomics, scratch, host fallback,
repair or software FLOAT64 are introduced. INT32/INT64 PROD uses stable row-major modular
multiplication; BOOL ALL/ANY uses exact zero/one identities. Empty axes are point identities
and never imply zero-extent support.

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
| 10 | `SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE` | A buffer has the wrong creating context, device, or extent, an input aliases an output, or distinct output slots alias one resource. |
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
Each run validates supplied input/output resources and stable node/logical index order.
Replacement scatter completes bounds and destination-uniqueness checks before its own copy or
write; arithmetic modes retain collisions in declared order. Produced controls/indices are
checked at their consuming step, after prior producers complete. Invalid consumption preserves
that operation's targets, not all earlier requested producer outputs. Public prepared Java
execution reproduces Model exception classes; the raw ABI reports typed status.
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

Any schema22 program containing a custom-only node or native Scan uses its retained custom-
program handle. Creation authenticates 23 reviewed fixed-kernel components and their ordered
187,750-byte total. The sole production pointwise emitter first traverses a structured
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

Native contexts and their buffer wrappers share a retained, private identity object. Every custom
program, MPSGraph, and singleton-NEG invocation checks exact identity before encoding or target
mutation, even when two contexts use the same `MTLDevice`. A buffer retains only this identity,
not the context or command queue; therefore a surviving buffer cannot match a reused context
address, and native context release lifetime is unchanged. Executables keep their existing strong
context reference. Upload/download and matching releases remain valid for surviving buffer handles.

Host upload/download own local Objective-C autorelease pools; callers do not supply a pool.
Each existing exception handler is inside its pool so mapped failures also drain temporary
Foundation objects before returning. Explicit buffer/context handle ownership is unchanged.
The source-matched [Task0082 qualification](../../../../docs/planning/backends/metal/tasks/0082-host-copy-autorelease-boundaries.md#result)
records unchanged public training/raw workflows and finite category-specific before/after memory
observations. It is not a universal native-allocation or public-prebuilt trust claim.

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

Engine0020 relocates paths only. Historical evidence under `../evidence/` retains original-revision
manifests, hashes, and outcomes; repaired tool paths do not renew those claims. In particular,
`../evidence/0053/tools/write_manifest.py --check` compares against the archived pre-relocation
manifest and is intentionally not a current-layout acceptance check. Do not regenerate that
manifest or run the full historical `../evidence/0053/run-proof.sh` for a layout change. The
[migration guide](../../../../docs/developer-guide/repository-layout.md#backend-path-migration-and-historical-navigation)
distinguishes historical paths from current commands. The build/package commands above and a
bounded opt-in device smoke below provide separate source-matched current evidence.

For a layout-only smoke, select just the existing typed execution case rather than the complete
native corpus:

```bash
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/modules/backend/metal/macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test \
  --tests '*MetalNegPreparedExecutionTest.realDeviceTypedMpsGraphInitializesLogicalSplatsPerRun' \
  --max-workers=2 --console=plain
```

This checks actual source-matched loading, preparation, and execution; it is not numerical
recertification, exhaustive proof, or performance evidence.

After the build, explicit ad-hoc signing, and package commands above, inspect the packaged dylib:

```bash
LIB=modules/backend/metal/macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib
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
(cd modules/backend/metal/macos-arm64/build/package-v1/macos-arm64 && \
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
FLOAT32/BFLOAT16/FLOAT16 non-overlapping maximum-pool cotangents. The Conv3d extension
additionally passed static FLOAT32 all-role and selected weight-only finite differences,
including grouped biased anisotropic geometry/overlap and unbiased depthwise cases, and real
NN training/save/close/load/fresh training without any CPU backend. See the
[bounded evidence](../../../../docs/backend-guide/production-support.md#metal-production-target),
including the initial reserved-identifier shader failure and corrected source-matched rebuild.
Qualified FLOAT32/BFLOAT16/FLOAT16 attention and generated gradients are supported; convolution transpose is not.

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

FLOAT64 overlapping numerical folds, convolution transpose, asymmetric padding, unsafe layouts
and unqualified dynamic/empty geometry remain unavailable. Common rank-16, geometry and
workspace bounds remain, but CONCAT/STACK have no sixteen-operand bound. Three-family
Conv3d/Pool3d, fixed-cell first AD and native general Scan training do not imply arbitrary
layouts, higher-order derivatives or general FLOAT64 arithmetic. Portable/configurable RNG,
entropy/random-quality guarantees, generic buffer pools, autotuning, atomics and host repair
are not added.

The public Java Metal surface is `MetalCapabilityProvider`, `MetalBackendConfiguration`, and
`MetalBackendIntegration`; Engine accepts an explicitly opened integration through
`Engine.builder()`.
