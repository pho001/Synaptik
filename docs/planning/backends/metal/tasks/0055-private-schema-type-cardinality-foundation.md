# Task 0055: Private Metal Schema, Type, and Cardinality Foundation

## Status

Ready

This planning-only task is the sole authorized Metal frontier after Task 0052. It starts from clean
base `2fcfefeb3a1fe90f0c4b4519404a8682a18ea471`. No production, test, native, package, generated,
capability, identity, or device change has been made by this planning commit.

Implementation is one atomic, zero-new-kind foundation cutover. It replaces the current private
fixed-node transport with one bounded program image, represents every current Model signature and
attribute family, carries every current Model data type explicitly, admits legal scalar values,
generalizes CPU/Metal typed transfer, and migrates every caller without retaining a schema-12 or
ABI-4 decoder. The public Model API, shared Runtime policy, and exact thirteen native export names
remain unchanged.

## Change class

Class C — the future implementation changes the private Java/native program schema, one existing
native function signature, native ABI and package metadata, workload identities, typed transfer
plumbing, and backend-private validation. It adds no Model operation capability and selects no new
numerical or performance route.

The planning commit changes only this task, the Metal master plan, and the repository roadmap. It
does not authorize a partial implementation. Once production work begins, the cutover must land as
one reviewable change whose Java, native, package, tuning, transfer, test, and documentation callers
all agree.

## Goal and invariant

Create the last shared structural prerequisite needed before the remaining current 115-kind Metal
workstreams can proceed independently:

- ordered node inputs in `0..N` and outputs in `1..N`, including zero-input `INITIAL_STATE`,
  variadic `CONCAT`/`STACK`, and all current multi-output signatures;
- explicit per-value data type, rank, dimensions, and gradient flag;
- all six existing Model carriers: `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, and `BOOL`;
- graph-local BOOL and integral consumers rather than terminal-only special cases;
- canonical typed CPU/Metal transfer in both directions for all six carriers;
- rank zero wherever the operation and descriptor contracts permit it;
- one closed, typed attribute vocabulary covering every current `OperationSignature`; and
- independent Java and native rejection of malformed, overflowing, non-canonical, or unsupported
  inputs before resource publication.

The capability invariant is unchanged: **schema vocabulary is not execution support**. The current
34 bounded capability-true kinds remain the only true kinds after this foundation; `EXP` and
`SIGMOID` remain Task-0053 false, and the other 79 kinds remain false. Thus the current accounting
is `34 + 2 + 79 = 115`, and Task 0055 contributes zero new kinds.

## Exact current Model signature inventory

The source of truth is the 40 current `OperationKind` enums and their `OperationSignature` values.
`a..b -> c..d` means inclusive input and output counts. `Integer.MAX_VALUE` is the Model's variadic
upper bound, not permission to omit the program-image byte bound defined later.

| Operation kind family / exact kinds | Exact attribute variants | Inputs -> outputs |
|---|---|---:|
| `ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION` | `ScaledDotProductAttentionAttrs` | `3..4 -> 1..2` |
| `Conv2dKind.CONV2D` | `Conv2dAttrs` | `2..3 -> 1` |
| `Conv3dKind.CONV3D` | `Conv3dAttrs` | `2..3 -> 1` |
| all seven `BinaryArithmeticKind` values | `NoOperationAttrs` | `2 -> 1` |
| `CastKind.CAST` | `CastAttrs` | `1 -> 1` |
| all three `FloatingClassificationKind` values | `NoOperationAttrs` | `1 -> 1` |
| all six `BinaryComparisonKind` values | `NoOperationAttrs` | `2 -> 1` |
| `BooleanLogicalKind.AND`, `OR` | `NoOperationAttrs` | `2 -> 1` |
| `BooleanLogicalKind.NOT` | `NoOperationAttrs` | `1 -> 1` |
| scalar `ADD`, `SUB`, `MUL`, `DIV`, `POW`, `MIN`, `MAX` | `ScalarValueAttrs` | `1 -> 1` |
| `ScalarElementwiseKind.CLAMP` | `ClampRangeAttrs` | `1 -> 1` |
| `WhereSelectionKind.WHERE` | `NoOperationAttrs` | `3 -> 1` |
| all nineteen `UnaryElementwiseKind` values | `NoOperationAttrs` | `1 -> 1` |
| both `AxisGatherKind` values | `IndexAxisAttrs` | `2 -> 1` |
| `AxisScatterKind.SCATTER_ELEMENTS` | `ScatterElementsAttrs` | `3 -> 1` |
| `AxisScatterKind.SCATTER_ADD` | `IndexAxisAttrs` | `3 -> 1` |
| `GatherNdKind.GATHER_ND` | `GatherNdAttrs` | `2 -> 1` |
| `OneHotKind.ONE_HOT` | `OneHotAttrs` | `1 -> 1` |
| `ScatterNdKind.SCATTER_ND` | `ScatterNdAttrs` | `3 -> 1` |
| `SelectKind.SELECT` | `SelectAttrs` | `1 -> 1` |
| `AxisTransformKind.PERMUTE` | `PermutationAttrs` | `1 -> 1` |
| `AxisTransformKind.EXPAND_DIMS`, `SQUEEZE` | `AxisTransformAttrs` | `1 -> 1` |
| `ContiguousKind.CONTIGUOUS` | `NoOperationAttrs` | `1 -> 1` |
| `PadKind.PAD` | `PadAttrs` | `1 -> 1` |
| both `ShapeTransformKind` values | `TargetShapeAttrs` | `1 -> 1` |
| `SliceKind.SLICE` | `SliceAttrs` or `CropToShapeAttrs` | `1 -> 1` |
| `SliceKind.SLICE_UPDATE` | `SliceAttrs` or `CropToShapeAttrs` | `2 -> 1` |
| `TensorCompositionKind.CONCAT`, `STACK` | `CompositionAxisAttrs` | `1..Integer.MAX_VALUE -> 1` |
| `TileKind.TILE` | `TileAttrs` | `1 -> 1` |
| `WindowTransformKind.UNFOLD_AXIS` | `UnfoldAxisAttrs` | `1 -> 1` |
| `WindowTransformKind.FOLD_AXIS` | `FoldAxisAttrs` | `1 -> 1` |
| `WindowTransformKind.UNFOLD2D` | `Window2dAttrs` or `Unfold2dAttrs` | `1 -> 1` |
| `WindowTransformKind.FOLD2D` | `Fold2dAttrs` | `1 -> 1` |
| `WindowTransformKind.UNFOLD3D` | `Window3dAttrs` or `Unfold3dAttrs` | `1 -> 1` |
| `WindowTransformKind.FOLD3D` | `Fold3dAttrs` | `1 -> 1` |
| `MatmulKind.MATMUL` | `NoOperationAttrs` | `2 -> 1` |
| `LossKind.MEAN_SQUARED_ERROR` | `MeanSquaredErrorAttrs` | `2 -> 1` |
| dense categorical cross entropy | `DenseCategoricalCrossEntropyWithLogitsAttrs` | `2 -> 1` |
| index categorical cross entropy | `IndexCategoricalCrossEntropyWithLogitsAttrs` | `2 -> 1` |
| `BatchNormKind.BATCH_NORM_INFERENCE` | `BatchNormInferenceAttrs` | `5 -> 1` |
| `BatchNormKind.BATCH_NORM_TRAINING` | `BatchNormTrainingAttrs` | `5 -> 5` |
| `LayerNormKind.LAYER_NORM` | `LayerNormAttrs` or `AffineLayerNormAttrs` | `1 -> 1` or `3 -> 1`, respectively |
| `RmsNormKind.RMS_NORM` | `RmsNormAttrs` | `1..2 -> 1` |
| both `SoftmaxKind` values | `SoftmaxAttrs` | `1 -> 1` |
| both `OrderingKind` values | `SortAttrs` | `1 -> 1` |
| `TopKKind.TOP_K` | `TopKAttrs` | `1 -> 2` |
| `Pool2dKind.MAX_POOL2D`, `AVERAGE_POOL2D` | matching `MaxPool2dAttrs` or `AveragePool2dAttrs` | `1 -> 1` |
| `Pool3dKind.MAX_POOL3D`, `AVERAGE_POOL3D` | matching `MaxPool3dAttrs` or `AveragePool3dAttrs` | `1 -> 1` |
| `DropoutKind.DROPOUT` | `DropoutAttrs` | `2 -> 3` |
| `GraphRngKind.INITIAL_STATE` | `GraphRngStateAttrs` | `0 -> 1` |
| `RecurrentScanKind.RNN_TANH`, `GRU_RESET_AFTER` | `RecurrentDirection` | `5..6 -> 2` |
| `RecurrentScanKind.LSTM` | `RecurrentDirection` | `6..7 -> 3` |
| `AggregateReductionKind.SUM` | none, axis, multi-axis, masked, or sum-to-shape attributes | `1 -> 1`, except masked `2 -> 1` |
| `AggregateReductionKind.MEAN` | none, axis, multi-axis, or masked attributes | `1 -> 1`, except masked `2 -> 1` |
| `PROD`, `MIN`, `MAX`, `ALL`, `ANY` | none, `AxisReductionAttrs`, or `MultiAxisReductionAttrs` | `1 -> 1` |
| `ARG_MIN`, `ARG_MAX` | `ArgExtremaAttrs` | `1 -> 1` |
| `LOG_SUM_EXP`, `L1_NORM`, `L2_NORM` | `MultiAxisReductionAttrs` | `1 -> 1` |
| `VARIANCE`, `STANDARD_DEVIATION` | `StatisticalReductionAttrs` | `1 -> 1` |
| both `CumulativeScanKind` values | `CumulativeScanAttrs` | `1 -> 1` |

`Conv1dAttrs`, `MaxPool1dAttrs`, and `AveragePool1dAttrs` exist as Model values but are not named by
any current `OperationSignature`; they are therefore not schema-13 attribute tags. A later Model
kind that adopts one requires a new schema review rather than accidental acceptance.

## Fixed-record decision and one replacement

The fixed 160-byte record does **not** remain. It can inline only the current special cases: one
required input, one optional second input, one special third input, one output, and sixteen
attribute words. Keeping it would still require side tables for current variadic inputs, five-output
nodes, list-valued attributes, and nested shapes, while retaining 128 bytes of dead padding per
node. Pointer-bearing records or operation-specific auxiliary arrays would make validation and
lifetime ownership less uniform.

Schema 13 uses one canonical bounded `MetalProgramImage` instead:

- native ABI becomes version 5 because the existing executable-create function changes signature;
- all thirteen export **names** remain exact and no export is added or removed;
- `synaptik_metal_mpsgraph_executable_create` becomes
  `int32_t(void *context, const uint8_t *program, uint32_t program_bytes, void **out_executable)`;
- the image is little-endian and at most `Integer.MAX_VALUE` bytes; zero length and any unsigned
  count or size above `Integer.MAX_VALUE` are invalid on both sides;
- count fields use `uint32_t`, but the common legal count domain is `0..2_147_483_647`; therefore
  node inputs are `0..N`, outputs are `1..N`, and actual counts are additionally bounded by the
  enclosing image size; and
- ABI 4/schema 12 are removed, not decoded alongside ABI 5/schema 13. Loader mismatch is fatal and
  no fallback library, translation shim, compatibility alias, or dual workload codec is retained.

### Canonical image layout

The image contains exactly these sections, in this order:

1. one 64-byte header;
2. `valueCount` 16-byte value descriptors;
3. `nodeCount` 32-byte node descriptors;
4. `dimensionCount` unsigned 64-bit dimensions;
5. `referenceCount` unsigned 32-bit value indices;
6. zero four-byte alignment padding exactly when the preceding reference section is not
   eight-byte aligned; and
7. `attributeWordCount` unsigned 64-bit attribute words.

The 64-byte header is ten unsigned 32-bit fields followed by six required-zero unsigned 32-bit
words: magic bytes `SM13`, schema `13`, exact total byte count, value count, node count, feed count,
target count, dimension count, reference count, and attribute-word count. The create argument and
header byte count must match exactly. Derived section ends must equal the image end; gaps, trailing
bytes, nonzero padding, nonzero reserved words, overlap, misalignment, or arithmetic overflow are
invalid.

A value descriptor is `{type, rank, dimensionOffset, flags}`. `flags & 1` is the exact Model
`requiresGrad` bit and every other bit is zero. `dimensionOffset` is an index into the dimension
section. Descriptors occur in value-index order and their dimension ranges are tightly packed in
that same order. Rank is `0..16`; rank zero has no dimensions and element count one. Positive-rank
dimensions are positive and fully static. Checked element count and element-count-times-byte-width
must fit signed `long`, the prepared memory slot, and the native platform size before allocation.

A node descriptor is `{operation, attributeKind, inputOffset, inputCount, outputOffset,
outputCount, attributeOffset, attributeWordCount}`. Reference offsets index the one reference
section. Its first `feedCount` entries are the ordered feeds, its next `targetCount` entries are the
ordered targets, and remaining entries are each node's inputs followed by outputs in node order.
Those ranges must be tightly packed. Attribute ranges are likewise tightly packed in node order.
This canonical form permits no unused reference or attribute word.

`nodeCount` and `targetCount` are positive; `feedCount` may be zero. Feed and target indices are
unique. Every node output is unique and previously unavailable. Every node input is an earlier feed
or node output. Targets are produced values. Unknown and unused values, forward references,
duplicate producers, input/output aliasing that violates an operation contract, and non-canonical
range order fail before graph construction.

### Data-type wires

Existing private meanings remain stable and the three missing carriers append:

| Wire | Model data type | Bytes | Required representation |
|---:|---|---:|---|
| 1 | `FLOAT32` | 4 | raw IEEE binary32 bits |
| 2 | `INT32` | 4 | exact two's-complement bits |
| 3 | `BOOL` | 1 | exactly `0` or `1` |
| 4 | `FLOAT64` | 8 | raw IEEE binary64 bits |
| 5 | `BFLOAT16` | 2 | exact raw 16-bit pattern |
| 6 | `INT64` | 8 | exact two's-complement bits |

Wire zero and every value above six are invalid. The schema does not add FLOAT16 or reinterpret
BFLOAT16. Multi-byte CPU/Metal transfers preserve local represented bytes without conversion;
public canonical host publication remains big-endian as already required by `HostTensorValue`.

## Closed operation wires

Schema 13 retains operation wires `1..34` and appends every currently false Model kind exactly once.
Within each range below, consecutive wires map to the consecutive qualified names as listed.

| Wires | Exact ordered operation identities |
|---:|---|
| `1..19` | `UnaryElementwiseKind.NEG`; binary `ADD`; binary `SUB`; binary `MUL`; binary `DIV`; `RESHAPE`; `EXPAND`; `PERMUTE`; `EXPAND_DIMS`; `SQUEEZE`; `CONTIGUOUS`; unary `ABS`; reduction `SUM`; reduction `MEAN`; `MATMUL`; `GATHER`; `ONE_HOT`; `SCATTER_ELEMENTS`; `UNFOLD_AXIS` |
| `20..34` | `GREATER_THAN`; `GREATER_OR_EQUAL`; `LESS_THAN`; `LESS_OR_EQUAL`; `EQUAL`; `NOT_EQUAL`; binary `MIN`; binary `MAX`; scalar `MIN`; scalar `MAX`; scalar `CLAMP`; reduction `MIN`; reduction `MAX`; `CUM_SUM`; `CUM_PROD` |
| `35..42` | `SCALED_DOT_PRODUCT_ATTENTION`; `CONV2D`; `CONV3D`; binary `POW`; `CAST`; `IS_FINITE`; `IS_NAN`; `IS_INF` |
| `43..51` | logical `AND`; logical `OR`; logical `NOT`; scalar `ADD`; scalar `SUB`; scalar `MUL`; scalar `DIV`; scalar `POW`; `WHERE` |
| `52..68` | unary `RECIPROCAL`; `LOG`; `LOG1P`; `EXP`; `EXPM1`; `ERF`; `SQRT`; `RSQRT`; `FLOOR`; `CEIL`; `SIGN`; `RELU`; `SIGMOID`; `TANH`; `GELU`; `GELU_TANH_APPROXIMATION`; `SILU` |
| `69..84` | `GATHER_ELEMENTS`; `SCATTER_ADD`; `GATHER_ND`; `SCATTER_ND`; `SELECT`; `PAD`; `SLICE`; `SLICE_UPDATE`; `CONCAT`; `STACK`; `TILE`; `FOLD_AXIS`; `UNFOLD2D`; `FOLD2D`; `UNFOLD3D`; `FOLD3D` |
| `85..96` | `MEAN_SQUARED_ERROR`; `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`; `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`; `BATCH_NORM_INFERENCE`; `BATCH_NORM_TRAINING`; `LAYER_NORM`; `RMS_NORM`; `SOFTMAX`; `LOG_SOFTMAX`; `SORT`; `ARGSORT`; `TOP_K` |
| `97..105` | `MAX_POOL2D`; `AVERAGE_POOL2D`; `MAX_POOL3D`; `AVERAGE_POOL3D`; `DROPOUT`; `INITIAL_STATE`; `RNN_TANH`; `GRU_RESET_AFTER`; `LSTM` |
| `106..115` | reduction `PROD`; `ALL`; `ANY`; `ARG_MAX`; `ARG_MIN`; `LOG_SUM_EXP`; `VARIANCE`; `STANDARD_DEVIATION`; `L1_NORM`; `L2_NORM` |

The wire table is codec vocabulary, not a route table. Java capability rejects every currently
false occurrence before production lowering. Native schema validation recognizes its structural
identity but returns the new `UNSUPPORTED_OPERATION` status `13` after structural validation when
no production route owns that kind. Statuses `0..12` retain their current meanings.

## Closed attribute schema

Attribute payloads are unsigned 64-bit words. The following primitives are exact:

- `bool` is one word `0` or `1`;
- `i32` is its low 32 two's-complement bits with high bits zero; `u32` must fit 32 bits;
- `i64` and raw binary64 retain all 64 bits;
- `shape` is `[rank, d0, ...]`, rank `0..16`, with positive static dimensions;
- `list` is `[count, item0, ...]` with exact checked length;
- `scalar` is `[typeWire, rawBits]`; bits outside the declared 1/2/4-byte width are zero and BOOL
  bits are zero or one; and
- `optionalScalar` is `[present]` or `[1, typeWire, rawBits]`.

Existing attribute wires `0..9` retain their normalized meanings. Schema 13 appends only the
families needed by current signatures:

| Wire | Normalized attribute family | Exact payload |
|---:|---|---|
| 0 | `NONE` | empty |
| 1 | `TARGET_SHAPE` | `shape` |
| 2 | `PERMUTATION` | `list<u32 axis>` |
| 3 | `AXIS` | normalized `u32 axis` |
| 4 | `REDUCTION` | `[form, keepDimensions, count, items...]`; existing forms `FULL=1`, `SINGLE_AXIS=2`, `MULTI_AXIS=3`, `SUM_TO_SHAPE=4` |
| 5 | `DEPTH` | positive `u64 depth` |
| 6 | `WINDOW_AXIS` | `[axis, sizeOrOutputSize, step]`; operation disambiguates fold/unfold |
| 7 | `SCALAR_VALUE` | `scalar` |
| 8 | `CLAMP_RANGE` | `scalar min`, then `scalar max` |
| 9 | `SCAN` | `[axis, exclusive, reverse]` |
| 10 | `CAST_TARGET` | one type wire |
| 11 | `GATHER_ND` | `u32 batchDimensions` |
| 12 | `SCATTER_ELEMENTS` | `[axis, scatterReduction]` |
| 13 | `SCATTER_ND` | `[batchDimensions, scatterReduction]` |
| 14 | `SELECT` | `[axis, i64 index]` |
| 15 | `CROP_TO_SHAPE` | target `shape`, then prefix `shape` |
| 16 | `PAD` | `[rank, before..., after..., scalar constant]` |
| 17 | `SLICE` | `[count, starts..., lengths..., axes..., steps...]` |
| 18 | `TILE` | `list<u64 repeat>` |
| 19 | `WINDOW_2D` | kernel H/W, stride H/W, padding H/W, dilation H/W, `ceilMode` |
| 20 | `PADDED_WINDOW_2D` | `WINDOW_2D`, then padding `scalar` |
| 21 | `FOLD_WINDOW_2D` | output `shape`, then `WINDOW_2D` |
| 22 | `WINDOW_3D` | kernel D/H/W, stride D/H/W, padding D/H/W, dilation D/H/W, `ceilMode` |
| 23 | `PADDED_WINDOW_3D` | `WINDOW_3D`, then padding `scalar` |
| 24 | `FOLD_WINDOW_3D` | output `shape`, then `WINDOW_3D` |
| 25 | `MSE` | one loss-reduction wire |
| 26 | `DENSE_CROSS_ENTROPY` | `[axis, lossReduction]` |
| 27 | `INDEX_CROSS_ENTROPY` | `[axis, lossReduction, optionalScalar ignoreIndex]` |
| 28 | `NORMALIZED_SHAPE_EPSILON` | normalized `shape`, then epsilon `scalar`; operation and arity distinguish LayerNorm, affine LayerNorm, and RMSNorm |
| 29 | `BATCH_NORM_INFERENCE` | `[channelAxis, scalar epsilon]` |
| 30 | `BATCH_NORM_TRAINING` | `[channelAxis, scalar momentum, scalar epsilon]` |
| 31 | `SORT` | `[axis, descending]` |
| 32 | `TOP_K` | `[axis, k, largest, sorted]` |
| 33 | `DROPOUT` | exact raw binary64 probability bits |
| 34 | `GRAPH_RNG_STATE` | raw `key`, raw `counter` |
| 35 | `RECURRENT_DIRECTION` | one direction wire |
| 36 | `ARG_EXTREMA` | `[axis, keepDimensions, tiePolicy]` |
| 37 | `MASKED_REDUCTION` | normalized `axis` |
| 38 | `STATISTICAL_REDUCTION` | `[axisCount, axes..., keepDimensions, correction]` |
| 39 | `ATTENTION` | `[optionalScalar scale, causal]` |
| 40 | `CONV_2D` | stride H/W, padding H/W, dilation H/W, groups |
| 41 | `CONV_3D` | stride D/H/W, padding D/H/W, dilation D/H/W, groups |

`AXIS` normalizes `IndexAxisAttrs`, `AxisTransformAttrs`, `CompositionAxisAttrs`, and `SoftmaxAttrs`;
the operation wire supplies their semantic role. `WINDOW_2D` also normalizes the matching max- and
average-pool records, and `WINDOW_3D` does the same for 3D pools. `REDUCTION` normalizes none/full,
axis, multi-axis, and sum-to-shape variants; masked, arg-extrema, and statistical forms remain
separate because they carry different inputs or semantics.

Nested enum wires are append-only and explicit: scatter `NONE=1`, `ADD=2`, `MUL=3`, `MAX=4`,
`MIN=5`; loss `NONE=1`, `SUM=2`, `MEAN=3`; arg-extrema tie `FIRST_INDEX=1`, `LAST_INDEX=2`; and
recurrent direction `FORWARD=1`, `REVERSE=2`. Zero and unknown enum wires are invalid.

The Java encoder maps only the exact attribute class accepted by the selected Model signature.
Operation/attribute/cardinality mismatches, malformed list lengths, non-normalized or duplicate
axes, invalid enum/boolean/scalar words, illegal optional payloads, nonzero unused words, and
constructor-incompatible ranges fail in Java and independently in native code.

## Java and native validation boundary

The two validators must make the same structural decision without trusting each other.

Java validates before allocating the image or entering FFM:

1. exact operation wire, accepted Model signature, attribute family, and input/output cardinality;
2. explicit value type, fully static rank-`0..16` shape, legal rank zero for the occurrence, resolved
   canonical or authenticated local affine layout, gradient bit, checked element and byte count;
3. topological availability, unique production, ordered feed/target/reference closure, and exact
   prepared memory extents;
4. every attribute primitive and descriptor-relative bound that is decidable before execution;
5. all multiply/add/align conversions with `Math.*Exact`, no lossy `long`/`int` narrowing, and final
   image size in `1..Integer.MAX_VALUE`; and
6. capability and fixed-route ownership. A recognized wire without a current route never reaches
   production native creation.

Native validates before creating an MPSGraph, pipeline, buffer, executable, or Objective-C owner:

1. non-null arguments and null output on entry;
2. magic/schema/total size, section arithmetic, alignment, canonical packing, reserved zeros, and
   exact end-of-image;
3. all value/type/rank/dimension/element/byte bounds;
4. all feed/target/node reference ranges, topology, producer uniqueness, and operation cardinality;
5. exact attribute wire, word count, primitive ranges, nested enum values, axes, shapes, and
   operation/attribute pairing; and
6. current route support only after structural validity. Unsupported but well-formed future wires
   return status 13 with no published handle.

Both sides must reject before indexing any malformed range. No validation failure may leak a native
resource, mutate caller buffers, partially publish an executable, or fall through to another route.

## Typed ingress, consumers, publication, and cross-owner transfer

The foundation removes FLOAT32-named transfer APIs and migrates all callers to one typed canonical
contract. It does not change shared Runtime types, ownership, validity transitions, scheduling, or
synchronous execution policy.

- `MetalBackendRuntime` and `MetalBackendIntegration` accept canonical, fully static, zero-offset,
  non-view descriptors of every six current types, rank `0..16`, with exact byte widths.
- `CpuBackendRuntime`/`CpuBackendIntegration` expose the same typed predicate and binding semantics;
  the old `*ContiguousFloat32*` methods are removed rather than retained as aliases.
- `CpuMetalPreparedBufferTransfer` keeps its class role but becomes type-generic in both directions.
  It still binds exact source/destination representations cold and performs one direct transfer on
  the hot path without route lookup, allocation, conversion, staging, or validity mutation.
- BOOL ingress and CPU-to-Metal transfer validate every byte as zero or one before upload. Metal-
  produced BOOL consumers and publications retain canonical bytes. Integral and floating payloads
  preserve exact represented bits; no CAST, canonicalization, widening, or narrowing is implied.
- Rank-zero transfer has element count one. Positive ranks require positive extents. Dynamic/zero
  extents, unresolved layout, nonzero offsets, general striding, and unauthenticated views remain
  unsupported.
- `copyToCanonicalHostBytes` publishes all six canonical types at rank `0..16`, converting local
  multi-byte representation to the existing big-endian public canonical form. The existing
  authenticated positive-rank affine FLOAT32 publication remains exact and is not generalized by
  this task.
- Program values use their declared type rather than inferred first-use type. BOOL and INT32/INT64
  results may be consumed by later nodes and may cross CPU/Metal ownership boundaries; per-kind
  capability and semantic validators still decide which such graphs are executable.

Borrowed host storage remains caller-owned. Prepared and run-owned Metal buffers keep their current
owners and close order. The image lives only through synchronous executable creation; native code
must not retain its pointer. A successful executable owns every deep-copied or compiled object it
needs, and every failure path rolls back in reverse acquisition order.

## Tuning and workload identity

This cutover advances the Metal local workload, exact-policy, candidate-batch, compatibility,
route-choice, and tuning codec identities together from version 13 to version 14. Candidate wire
identities `CUSTOM_SINGLE_NEG=1`, `MPSGRAPH=2`, and `CUSTOM_TASK0052=3` remain stable; complete-plan
wrapper version 1 remains stable.

Version 14 hashes the complete logical schema-13 program without whole-image duplication:
operation and attribute wires, exact ordered cardinalities and references, every per-value type,
rank, dimension and gradient bit, feed/target order, exact attribute words, numerical profile,
route-relevant affine authentication, and current target/budget facts. It must distinguish all six
types, rank zero from rank one, every output slot, optional values, enum modes, and exact scalar raw
bits including floating signed zero and NaN payloads.

Codec 13 bytes are incompatible and rejected; no migration or dual reader is added. Cache misses
remain safe. The existing route heuristic and selected routes are unchanged, and no benchmark,
runtime timing, retry, or input-dependent choice is introduced.

## Native package and exports

The implementation updates the native source, Java FFM descriptor, local package producer,
verifier, package README, and exact manifest expectations atomically:

- `nativeAbiVersion` becomes `5`;
- `nodeSchemaVersion` becomes `13`;
- manifest schema version `1`, platform, architecture, minimum macOS, install name, frameworks,
  signature identity, checksums, and local-only distribution remain unchanged;
- the existing thirteen symbols remain the exact export set; and
- verifier logic checks ABI 5, the new executable-create signature through real loading/tests, and
  schema 13 metadata. It must reject an ABI-4/schema-12 dylib rather than silently accepting it.

Changing one existing symbol's function signature is the evidence that makes ABI 5 necessary; no
fourteenth export is needed. The three custom-NEG exports remain because removing them is unrelated
to the schema prerequisite and would change the current candidate surface.

## Dependency and implementation order

Implementation is serial and atomic at the repository boundary:

1. **Freeze source-derived tables.** Add behavioral tests that enumerate the current 115 operation
   identities, exact Model signatures, six data types, normalized attribute mappings, and the
   unchanged `34+2+79` capability partition.
2. **Implement the Java schema-13 model and encoder.** Replace `MetalMpsGraphProgram`'s fixed record
   with the bounded image model, checked size calculator, wire tables, canonical encoder, and Java
   validator. Migrate lowering for all current 34 routes; do not add false-kind lowering to the
   production preparation path.
3. **Cut native ABI 5/schema 13.** Replace the old create signature and decoder, independently
   validate the entire image, map explicit value descriptors for current routes, add status 13,
   preserve the other twelve function signatures and all thirteen names, and delete every ABI-4/
   schema-12 struct, constant, parser, and fixture.
4. **Generalize typed buffer plumbing.** Migrate Metal and CPU integration predicates/binders,
   `CpuMetalPreparedBufferTransfer`, ingress, publication, BOOL checking, rank-zero handling, and
   all call sites for the six exact byte widths without touching `modules/runtime` policy.
5. **Advance identity once.** Move all local identity/codec components to 14 and make the workload
   digest consume the same schema facts. Preserve candidate and complete-plan wrapper identities.
6. **Update package tooling and documentation.** Set ABI 5/schema 13 in producer/verifier metadata,
   document the changed create signature and typed transfer boundary, and remove stale schema-12,
   160-byte, FLOAT32-only, and terminal-BOOL claims.
7. **Validate once after the cutover.** Run focused pure-Java schema/transfer/identity tests first,
   then native/package/real-device tests for existing routes, then the repository validation owned
   by the implementation brief. Do not split a compatible Java half from an incompatible native or
   package half.

Tasks 0053 and the 79-kind workstreams consume this foundation only after Task 0055 is Complete.
They retain their independent proof, oracle, measured-cost, route, gradient, and capability gates.
No operation task may reinterpret a schema wire as authorization.

## Expected implementation scope

The implementation is expected to modify existing files rather than create a parallel framework:

- Metal program model/lowering/preparation/native binding, typed buffer integration, workload/tuning
  identity, tests, package Javadocs, and package documentation;
- Objective-C native schema/parser/create path and native tests;
- CPU typed transfer predicates/binders and focused tests;
- Engine CPU/Metal prepared transfer and registry/session tests;
- native package producer/verifier/README and manifest expectations; and
- Metal master/roadmap/task result documentation after completion.

`modules/model`, `modules/runtime`, Planning ownership, Compiler generation, public configuration,
Trace DTOs, Config policy, and unrelated backend operation implementations are not implementation
scope unless a concrete compile failure proves a caller migration is unavoidable. Such a finding
must preserve the public contract rather than widen this task.

## Required tests and proof of the future implementation

### Pure Java schema behavior

- all 115 operation wires are unique and every current kind maps exactly once;
- every signature row above accepts its boundary cardinalities and rejects adjacent invalid counts;
- representative cardinality proofs cover zero-input RNG, 17-input composition, attention `4 -> 2`,
  BatchNorm training `5 -> 5`, Top-K `1 -> 2`, dropout `2 -> 3`, and LSTM `7 -> 3`;
- all six value types and typed scalars preserve exact raw bits, including FLOAT64/FLOAT32 signed
  zero and NaN payloads, all BFLOAT16 patterns, INT minima/maxima, and BOOL zero/one;
- rank zero, rank sixteen, checked element/byte overflow, `Integer.MAX_VALUE` count/size boundaries,
  and exact image alignment/size behavior are covered;
- every normalized attribute family and nested enum has a round-trip/golden behavior test; and
- malformed topology, mismatched operation/attribute/cardinality, duplicate producer, forward
  reference, bad feed/target, unknown wire, bad reserved/padding word, truncated image, and range
  overflow fail deterministically.

### Native and ABI behavior

- ABI reports 5 and the package exports exactly the same thirteen names;
- Java and native accept the same valid schema-13 images for current production routes;
- native independently rejects each malformed-image class before creating resources;
- a structurally valid currently unsupported operation returns status 13 and no handle;
- existing 34-kind routes still execute their exact bounded behavior through schema 13; and
- create/release/run failure injection proves no image pointer retention, double release, partial
  publication, or resource leak.

### Typed transfer behavior

- CPU-to-Metal and Metal-to-CPU direct transfer preserves representative raw bytes for all six
  types at rank zero and positive rank;
- BOOL zero and one transfer, while any other BOOL byte is rejected before upload;
- exact byte extent, prepared-slot mismatch, overflow, closed/wrong-context representation,
  unresolved/dynamic/zero extent, noncanonical layout, offset, and unauthenticated view all reject;
- canonical publication yields the existing big-endian host bytes for every type; and
- cold binding retains exact handles/type/extent while the hot action performs no allocation,
  lookup, route selection, conversion, or validity mutation.

### Capability, identity, package, and integration regressions

- capability remains exactly 34 current true kinds and 81 false kinds under the existing bounded
  rows; no new kind, dtype row, layout row, gradient row, or numerical route becomes true;
- identity version 14 distinguishes all new schema facts and rejects version 13 while retaining
  candidate wires `1..3` and complete-plan wrapper 1;
- package producer and verifier agree on ABI 5/schema 13, exact exports, checksum, install name,
  frameworks, signature, and local archive behavior; and
- existing public Engine single-owner and mixed CPU/Metal scenarios execute against the newly built
  verified dylib with zero skips. Final validation must exercise the real surface, not only mocks.

## Explicit non-goals

Task 0055 does not:

- make any of the 81 currently false kinds capability-true;
- implement, probe, prove, benchmark, or choose a numerical operation route;
- rerun Task 0051, Task 0052, or Task 0053 oracles or alter their evidence;
- add FLOAT16, dynamic or zero-extent tensors, rank above 16, general strided/offset transfer, or
  cross-device transfer;
- change public Model operations, attributes, data types, Tensor descriptors, numerical profiles,
  gradient semantics, Compiler generation, Planning ownership, Runtime validity/lifecycle policy,
  asynchronous execution, default-device selection, or fallback behavior;
- add an export, generic attribute map, reflection/string dispatch, pointer-bearing node records,
  operation-specific side-channel arrays, compatibility shim, dual decoder, or cache migration;
- generalize authenticated affine publication beyond its existing positive-rank FLOAT32 contract;
  or
- remove the three custom-NEG exports, rename public configuration, publish a native artifact, add
  discovery, Developer ID signing, notarization, or release automation.

## Acceptance criteria

1. The detailed task records the exact current signature inventory and all 115 operation wires.
2. The 160-byte record decision is explicit and replaced by exactly one bounded schema-13 program
   image with exact layouts, counts, ordering, rank-zero, type, and attribute rules.
3. All six current Model carriers, BOOL/INT consumers, exact typed scalars, and both-direction typed
   CPU/Metal transfer are specified without changing Model or Runtime policy.
4. Java/native structural validation, checked arithmetic, lifecycle/rollback, unsupported-operation
   behavior, and no-dual-decoder migration are complete and symmetric.
5. ABI 5 retains exactly thirteen export names; manifest node schema 13 and native ABI 5 are exact.
6. Version-14 tuning/workload identities cover the new facts while candidate wires and wrapper 1
   remain stable.
7. Capability remains exactly `34+2+79=115`; no route, proof, oracle, benchmark, or operation
   capability is added by the foundation.
8. Future implementation files, tests, dependency order, cleanup, package changes, and explicit
   non-goals are sufficient for one atomic cutover with every caller migrated.
9. This planning commit changes exactly three Markdown planning files, passes documentation and
   diff validation, runs no build/test/native/device command, is committed, and leaves a clean
   worktree.

## Planning validation

For this planning commit only:

```bash
git diff --check
```

Also validate local Markdown links and anchors, unique effective headings, balanced fences, final
newlines, trailing whitespace, exact Task-0055 `Ready` status, Metal/roadmap sole-frontier
agreement, operation and attribute wire ranges, the `34+2+79=115` equation, and exactly three
changed planning paths. Do not build, test, compile native code, package a dylib, run a device,
measure performance, invoke a proof tool, or edit production/evidence/generated files.
