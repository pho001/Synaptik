# Task 0055: Private [Metal](../../../../glossary.md#term-metal) [Schema](../../../../glossary.md#term-schema), Type, and [Cardinality](../../../../glossary.md#term-cardinality) Foundation

## Reader context

The program image is a description crossing Java's Foreign Function and Memory (FFM) boundary, not a Tensor's numerical payload. Metal's Java encoder writes typed descriptors, references and attributes; the Objective-C decoder independently validates them before executable creation. Native creation must copy or compile what it needs because the temporary image pointer is not retained.

Typed CPU/Metal transfer moves represented bytes in separately owned buffers. Carrier and multi-output vocabulary supply representation, not permission to execute every operation using it. ABI-5/schema-13 below are historical; current image fields are documented separately.

- [Current program-image and failure boundary](../../../../backend-guide/metal-backend.md#abi-and-failures) — Shows the evolved wire format without rewriting the cutover.
- [Current binding and transfer ownership](../../../../backend-guide/metal-backend.md#cold-run-setup-and-binding) — Separates descriptions from borrowed inputs and run-owned outputs.

## Status

Complete

Implemented atomically at `ddeff1b2` from the approved [planning](../../../../glossary.md#term-planning) base
`b7b9bab8d099539977c7fefc4c69b9f53db7592b`. The production, native, package, [identity](../../../../glossary.md#term-identity-meanings),
typed-transfer, test, and documentation callers now agree on the foundation described here.

Implementation is one [atomic](../../../../glossary.md#resource-transaction-and-rollback), zero-new-kind foundation cutover. It replaces the current private
fixed-[node](../../../../glossary.md#node) transport with one bounded program image, represents every current [Model](../../../../glossary.md#term-model) [signature](../../../../glossary.md#term-signature) and
attribute family, carries every current Model [data type](../../../../glossary.md#data-type--datatype) explicitly, admits legal [scalar](../../../../glossary.md#term-scalar) values,
generalizes [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) typed transfer, and migrates every caller without retaining a [schema](../../../../glossary.md#term-schema)-12 or
[ABI](../../../../glossary.md#term-application-binary-interface-abi)-4 decoder. The public Model [API](../../../../glossary.md#term-application-programming-interface-api), shared [Runtime](../../../../glossary.md#term-runtime) policy, and exact thirteen native export names
remain unchanged.

## Change class

Class C — the implementation changes the private Java/native program [schema](../../../../glossary.md#term-schema), one existing native
function [signature](../../../../glossary.md#term-signature), native [ABI](../../../../glossary.md#term-application-binary-interface-abi) and package [metadata](../../../../glossary.md#term-metadata), workload [identities](../../../../glossary.md#term-identity-meanings), typed transfer plumbing,
and [backend](../../../../glossary.md#backend)-private [validation](../../../../glossary.md#term-validation). It adds no [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation) capability and selects no new numerical
or performance route.

The earlier [planning](../../../../glossary.md#term-planning) [commit](../../../../glossary.md#term-commit) changed only this task, the [Metal](../../../../glossary.md#term-metal) master plan, and the repository
roadmap. The completed implementation landed as one [atomic](../../../../glossary.md#resource-transaction-and-rollback) cutover whose Java, native, package,
tuning, transfer, test, and documentation callers agree; it retained no schema-12 or ABI-4
compatibility path.

## Goal and [invariant](../../../../glossary.md#term-invariant)

Create the last shared structural prerequisite needed before the remaining current 115-kind [Metal](../../../../glossary.md#term-metal)
workstreams can proceed independently:

- ordered [node](../../../../glossary.md#node) inputs in `0..N` and outputs in `1..N`, including zero-input `INITIAL_STATE`,
  variadic <span data-glossary-navigation="true"> ([variadic glossary](../../../../glossary.md#term-variadic))</span>`CONCAT`/`STACK`, and all current multi-output [signatures](../../../../glossary.md#term-signature);
- explicit per-value [data type](../../../../glossary.md#data-type--datatype), [rank](../../../../glossary.md#term-rank), [dimensions](../../../../glossary.md#dimension), and [gradient](../../../../glossary.md#term-gradient) flag;
- all six existing [Model](../../../../glossary.md#term-model) carriers: [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype), and [`BOOL`](../../../../glossary.md#data-type--datatype);
- [graph](../../../../glossary.md#graph)-local BOOL and integral consumers rather than terminal-only special cases;
- [canonical](../../../../glossary.md#term-canonical-form) typed [CPU](../../../../glossary.md#term-cpu)/Metal transfer in both directions for all six carriers;
- rank zero wherever the [operation](../../../../glossary.md#operation) and [descriptor](../../../../glossary.md#term-descriptor) contracts permit it;
- one closed, typed attribute vocabulary covering every current `OperationSignature`; and
- independent Java and native rejection of malformed, overflowing, non-canonical, or unsupported
  inputs before resource publication.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>

The capability invariant is unchanged: **[schema](../../../../glossary.md#term-schema) vocabulary is not execution support**. The current
34 bounded capability-true kinds remain the only true kinds after this foundation; [`EXP`](../../../../glossary.md#term-expx) and
[`SIGMOID`](../../../../glossary.md#term-sigmoid) remain Task-0053 false, and the other 79 kinds remain false. Thus the current accounting
is `34 + 2 + 79 = 115`, and Task 0055 contributes zero new kinds.

## Exact current [Model](../../../../glossary.md#term-model) [signature](../../../../glossary.md#term-signature) inventory

The source of truth is the 40 current [`OperationKind`](../../../../glossary.md#operationkind) enums and their `OperationSignature` values.
`a..b -> c..d` means inclusive input and output counts. `Integer.MAX_VALUE` is the Model's [variadic](../../../../glossary.md#term-variadic)
upper bound, not permission to omit the program-image byte bound defined later.

| [Operation](../../../../glossary.md#operation) kind family / exact kinds | Exact attribute variants | Inputs -> outputs |
|---|---|---:|
| `ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION` | `ScaledDotProductAttentionAttrs` | `3..4 -> 1..2` |
| `Conv2dKind.CONV2D` | `Conv2dAttrs` | `2..3 -> 1` |
| `Conv3dKind.CONV3D` | `Conv3dAttrs` | `2..3 -> 1` |
| all seven `BinaryArithmeticKind` values | [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) | `2 -> 1` |
| `CastKind.CAST` | `CastAttrs` | `1 -> 1` |
| all three `FloatingClassificationKind` values | `NoOperationAttrs` | `1 -> 1` |
| all six `BinaryComparisonKind` values | `NoOperationAttrs` | `2 -> 1` |
| `BooleanLogicalKind.AND`, `OR` | `NoOperationAttrs` | `2 -> 1` |
| `BooleanLogicalKind.NOT` | `NoOperationAttrs` | `1 -> 1` |
| [scalar](../../../../glossary.md#term-scalar) `ADD`, `SUB`, `MUL`, `DIV`, `POW`, `MIN`, `MAX` | `ScalarValueAttrs` | `1 -> 1` |
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
| dense [categorical cross entropy](../../../../glossary.md#categorical-cross-entropy) | `DenseCategoricalCrossEntropyWithLogitsAttrs` | `2 -> 1` |
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
| `AggregateReductionKind.SUM` | none, [axis](../../../../glossary.md#term-axis), multi-axis, masked, or [sum-to-shape](../../../../glossary.md#sum-to-shape) attributes | `1 -> 1`, except masked `2 -> 1` |
| `AggregateReductionKind.MEAN` | none, axis, multi-axis, or masked attributes | `1 -> 1`, except masked `2 -> 1` |
| `PROD`, `MIN`, `MAX`, `ALL`, `ANY` | none, `AxisReductionAttrs`, or `MultiAxisReductionAttrs` | `1 -> 1` |
| `ARG_MIN`, `ARG_MAX` | `ArgExtremaAttrs` | `1 -> 1` |
| `LOG_SUM_EXP`, `L1_NORM`, `L2_NORM` | `MultiAxisReductionAttrs` | `1 -> 1` |
| [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION` | `StatisticalReductionAttrs` | `1 -> 1` |
| both `CumulativeScanKind` values | `CumulativeScanAttrs` | `1 -> 1` |

`Conv1dAttrs`, `MaxPool1dAttrs`, and `AveragePool1dAttrs` exist as Model values but are not named by
any current `OperationSignature`; they are therefore not [schema](../../../../glossary.md#term-schema)-13 attribute tags. A later Model
kind that adopts one requires a new schema review rather than accidental acceptance.

## Fixed-record decision and one replacement

The fixed 160-byte record does **not** remain. It can inline only the current special cases: one
required input, one optional second input, one special third input, one output, and sixteen
attribute words. Keeping it would still require side tables for current [variadic](../../../../glossary.md#term-variadic) inputs, five-output
[nodes](../../../../glossary.md#node), list-valued attributes, and nested [shapes](../../../../glossary.md#shape), while retaining 128 bytes of dead [padding](../../../../glossary.md#padding) per
node. Pointer-bearing records or [operation](../../../../glossary.md#operation)-specific auxiliary arrays would make [validation](../../../../glossary.md#term-validation) and
[lifetime](../../../../glossary.md#term-lifetime) [ownership](../../../../glossary.md#term-ownership-meanings) less uniform.

[Schema](../../../../glossary.md#term-schema) 13 uses one [canonical](../../../../glossary.md#term-canonical-form) bounded `MetalProgramImage` instead:

- native [ABI](../../../../glossary.md#term-application-binary-interface-abi) becomes version 5 because the existing executable-create function changes [signature](../../../../glossary.md#term-signature);
- all thirteen export **names** remain exact and no export is added or removed;
- `synaptik_metal_mpsgraph_executable_create` becomes
  `int32_t(void *context, const uint8_t *program, uint32_t program_bytes, void **out_executable)`;
- the image is [little-endian](../../../../glossary.md#text-encoding-and-file-boundaries) and at most `Integer.MAX_VALUE` bytes; zero length and any unsigned
  count or size above `Integer.MAX_VALUE` are invalid on both sides;
- count fields use [`uint32_t`](../../../../glossary.md#index-mappings-and-cost-dominance), but the common legal count domain is `0..2_147_483_647`; therefore
  node inputs are `0..N`, outputs are `1..N`, and actual counts are additionally bounded by the
  enclosing image size; and
- ABI 4/schema 12 are removed, not decoded alongside ABI 5/schema 13. Loader mismatch is fatal and
  no fallback library, translation shim, compatibility alias, or dual workload codec is retained.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [codec glossary](../../../../glossary.md#term-codec))</span>

### [Canonical](../../../../glossary.md#term-canonical-form) image [layout](../../../../glossary.md#layout)

The image contains exactly these sections, in this order:

1. one 64-byte header;
2. `valueCount` 16-byte value [descriptors](../../../../glossary.md#term-descriptor);
3. `nodeCount` 32-byte [node](../../../../glossary.md#node) descriptors;
4. `dimensionCount` unsigned 64-bit [dimensions](../../../../glossary.md#dimension);
5. `referenceCount` unsigned 32-bit value indices;
6. zero four-byte alignment padding exactly when the preceding reference section is not
   eight-byte aligned; and<span data-glossary-navigation="true"> ([byte alignment glossary](../../../../glossary.md#term-byte-alignment))</span>
7. `attributeWordCount` unsigned 64-bit attribute words.

The 64-byte header is ten unsigned 32-bit fields followed by six required-zero unsigned 32-bit
words: [magic bytes](../../../../glossary.md#term-magic-bytes) `SM13`, [schema](../../../../glossary.md#term-schema) `13`, exact total byte count, value count, node count, feed count,
[target](../../../../glossary.md#term-target) count, dimension count, reference count, and attribute-word count. The create [argument](../../../../glossary.md#term-argument) and
header byte count must match exactly. Derived section ends must equal the image end; gaps, trailing
bytes, nonzero [padding](../../../../glossary.md#padding), nonzero reserved words, overlap, misalignment, or arithmetic [overflow](../../../../glossary.md#term-overflow) are
invalid.

A value descriptor is `{type, rank, dimensionOffset, flags}`. `flags & 1` is the exact [Model](../../../../glossary.md#term-model)
`requiresGrad` bit and every other bit is zero. `dimensionOffset` is an index into the dimension
section. Descriptors occur in value-index order and their dimension ranges are tightly packed in
that same order. [Rank](../../../../glossary.md#term-rank) is `0..16`; rank zero has no dimensions and [element](../../../../glossary.md#term-element) count one. Positive-rank
dimensions are positive and fully static. Checked element count and element-count-times-byte-width
must fit signed `long`, the prepared [memory slot](../../../../glossary.md#memory-slot), and the native platform size before allocation.

A node descriptor is `{operation, attributeKind, inputOffset, inputCount, outputOffset,
outputCount, attributeOffset, attributeWordCount}`. Reference offsets index the one reference
section. Its first `feedCount` entries are the ordered feeds, its next `targetCount` entries are the
ordered targets, and remaining entries are each node's inputs followed by outputs in node order.
Those ranges must be tightly packed. Attribute ranges are likewise tightly packed in node order.
This [canonical form](../../../../glossary.md#term-canonical-form) permits no unused reference or attribute word.

`nodeCount` and `targetCount` are positive; `feedCount` may be zero. Feed and target indices are
unique. Every node output is unique and previously unavailable. Every node input is an earlier feed
or node output. Targets are produced values. Unknown and unused values, forward references,
duplicate [producers](../../../../glossary.md#term-producer), input/output [aliasing](../../../../glossary.md#term-aliasing) that violates an [operation](../../../../glossary.md#operation) contract, and non-canonical
range order fail before [graph](../../../../glossary.md#graph) construction.

### Data-type wires

Existing private meanings remain stable and the three missing carriers append:

| Wire | [Model](../../../../glossary.md#term-model) [data type](../../../../glossary.md#data-type--datatype) | Bytes | Required representation |
|---:|---|---:|---|
| 1 | [`FLOAT32`](../../../../glossary.md#term-float32) | 4 | raw [IEEE](../../../../glossary.md#term-ieee) [binary32](../../../../glossary.md#floating-point-precision-and-rounding) bits |
| 2 | [`INT32`](../../../../glossary.md#data-type--datatype) | 4 | exact [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) bits |
| 3 | [`BOOL`](../../../../glossary.md#data-type--datatype) | 1 | exactly `0` or `1` |
| 4 | [`FLOAT64`](../../../../glossary.md#term-float64) | 8 | raw IEEE [binary64](../../../../glossary.md#floating-point-precision-and-rounding) bits |
| 5 | [`BFLOAT16`](../../../../glossary.md#term-bfloat16) | 2 | exact raw 16-bit pattern |
| 6 | [`INT64`](../../../../glossary.md#data-type--datatype) | 8 | exact two's-complement bits |

Wire zero and every value above six are invalid. The [schema](../../../../glossary.md#term-schema) does not add [FLOAT16](../../../../glossary.md#term-float16) or reinterpret
BFLOAT16. Multi-byte [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) transfers preserve local represented bytes without conversion;
public [canonical](../../../../glossary.md#term-canonical-form) [host](../../../../glossary.md#term-host) [publication](../../../../glossary.md#term-publication) remains [big-endian](../../../../glossary.md#text-encoding-and-file-boundaries) as already required by [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue).

## Closed [operation](../../../../glossary.md#operation) wires

[Schema](../../../../glossary.md#term-schema) 13 retains operation wires `1..34` and appends every currently false [Model](../../../../glossary.md#term-model) kind exactly once.
Within each range below, consecutive wires map to the consecutive qualified names as listed.

| Wires | Exact ordered operation [identities](../../../../glossary.md#term-identity-meanings) |
|---:|---|
| `1..19` | `UnaryElementwiseKind.NEG`; [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `ADD`; binary `SUB`; binary `MUL`; binary `DIV`; `RESHAPE`; `EXPAND`; `PERMUTE`; `EXPAND_DIMS`; `SQUEEZE`; [`CONTIGUOUS`](../../../../glossary.md#term-contiguous); [unary](../../../../glossary.md#term-unary) `ABS`; reduction `SUM`; reduction [`MEAN`](../../../../glossary.md#term-mean); [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul); [`GATHER`](../../../../glossary.md#gather); `ONE_HOT`; `SCATTER_ELEMENTS`; `UNFOLD_AXIS` |
| `20..34` | `GREATER_THAN`; `GREATER_OR_EQUAL`; `LESS_THAN`; `LESS_OR_EQUAL`; `EQUAL`; `NOT_EQUAL`; binary `MIN`; binary `MAX`; [scalar](../../../../glossary.md#term-scalar) `MIN`; scalar `MAX`; scalar `CLAMP`; reduction `MIN`; reduction `MAX`; `CUM_SUM`; `CUM_PROD` |
| `35..42` | `SCALED_DOT_PRODUCT_ATTENTION`; [`CONV2D`](../../../../glossary.md#convolution--conv2d); [`CONV3D`](../../../../glossary.md#conv3d--ncdhw); binary `POW`; `CAST`; `IS_FINITE`; `IS_NAN`; `IS_INF` |
| `43..51` | logical `AND`; logical `OR`; logical `NOT`; scalar `ADD`; scalar `SUB`; scalar `MUL`; scalar `DIV`; scalar `POW`; `WHERE` |
| `52..68` | unary `RECIPROCAL`; [`LOG`](../../../../glossary.md#term-log); `LOG1P`; [`EXP`](../../../../glossary.md#term-expx); `EXPM1`; [`ERF`](../../../../glossary.md#term-erf); `SQRT`; `RSQRT`; `FLOOR`; `CEIL`; `SIGN`; `RELU`; [`SIGMOID`](../../../../glossary.md#term-sigmoid); [`TANH`](../../../../glossary.md#term-tanh); [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu); `GELU_TANH_APPROXIMATION`; `SILU` |
| `69..84` | `GATHER_ELEMENTS`; `SCATTER_ADD`; `GATHER_ND`; `SCATTER_ND`; `SELECT`; [`PAD`](../../../../glossary.md#padding); [`SLICE`](../../../../glossary.md#slice); `SLICE_UPDATE`; `CONCAT`; `STACK`; `TILE`; `FOLD_AXIS`; `UNFOLD2D`; `FOLD2D`; `UNFOLD3D`; `FOLD3D` |
| `85..96` | `MEAN_SQUARED_ERROR`; `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`; `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`; `BATCH_NORM_INFERENCE`; `BATCH_NORM_TRAINING`; `LAYER_NORM`; `RMS_NORM`; [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax); `LOG_SOFTMAX`; `SORT`; `ARGSORT`; `TOP_K` |
| `97..105` | `MAX_POOL2D`; `AVERAGE_POOL2D`; `MAX_POOL3D`; `AVERAGE_POOL3D`; [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout); `INITIAL_STATE`; `RNN_TANH`; `GRU_RESET_AFTER`; [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm) |
| `106..115` | reduction `PROD`; `ALL`; `ANY`; `ARG_MAX`; `ARG_MIN`; `LOG_SUM_EXP`; [`VARIANCE`](../../../../glossary.md#term-variance); `STANDARD_DEVIATION`; `L1_NORM`; `L2_NORM` |

The wire table is [codec](../../../../glossary.md#term-codec) vocabulary, not a route table. Java capability rejects every currently
false [occurrence](../../../../glossary.md#term-occurrence) before production [lowering](../../../../glossary.md#lowering). Native schema [validation](../../../../glossary.md#term-validation) recognizes its structural
[identity](../../../../glossary.md#term-identity-meanings) but returns the new `UNSUPPORTED_OPERATION` status `13` after structural validation when
no production route owns that kind. Statuses `0..12` retain their current meanings.

## Closed attribute [schema](../../../../glossary.md#term-schema)

Attribute payloads are unsigned 64-bit words. The following primitives are exact:

- `bool` is one word `0` or `1`;
- `i32` is its low 32 [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) bits with high bits zero; `u32` must fit 32 bits;
- `i64` and raw [binary64](../../../../glossary.md#floating-point-precision-and-rounding) retain all 64 bits;
- [`shape`](../../../../glossary.md#shape) is `[rank, d0, ...]`, [rank](../../../../glossary.md#term-rank) `0..16`, with positive static [dimensions](../../../../glossary.md#dimension);
- `list` is `[count, item0, ...]` with exact checked length;
- [`scalar`](../../../../glossary.md#term-scalar) is `[typeWire, rawBits]`; bits outside the declared 1/2/4-byte width are zero and BOOL
  bits are zero or one; and<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- `optionalScalar` is `[present]` or `[1, typeWire, rawBits]`.

Existing attribute wires `0..9` retain their normalized meanings. Schema 13 appends only the
families needed by current [signatures](../../../../glossary.md#term-signature):

| Wire | Normalized attribute family | Exact [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) |
|---:|---|---|
| 0 | `NONE` | empty |
| 1 | `TARGET_SHAPE` | `shape` |
| 2 | [`PERMUTATION`](../../../../glossary.md#permutation) | `list<u32 axis>` |
| 3 | [`AXIS`](../../../../glossary.md#term-axis) | normalized `u32 axis` |
| 4 | `REDUCTION` | `[form, keepDimensions, count, items...]`; existing forms `FULL=1`, `SINGLE_AXIS=2`, `MULTI_AXIS=3`, `SUM_TO_SHAPE=4` |
| 5 | `DEPTH` | positive `u64 depth` |
| 6 | `WINDOW_AXIS` | `[axis, sizeOrOutputSize, step]`; [operation](../../../../glossary.md#operation) disambiguates fold/unfold |
| 7 | `SCALAR_VALUE` | `scalar` |
| 8 | `CLAMP_RANGE` | `scalar min`, then `scalar max` |
| 9 | [`SCAN`](../../../../glossary.md#term-cumulative-scan) | `[axis, exclusive, reverse]` |
| 10 | `CAST_TARGET` | one type wire |
| 11 | `GATHER_ND` | `u32 batchDimensions` |
| 12 | `SCATTER_ELEMENTS` | `[axis, scatterReduction]` |
| 13 | `SCATTER_ND` | `[batchDimensions, scatterReduction]` |
| 14 | `SELECT` | `[axis, i64 index]` |
| 15 | `CROP_TO_SHAPE` | [target](../../../../glossary.md#term-target) `shape`, then prefix `shape` |
| 16 | [`PAD`](../../../../glossary.md#padding) | `[rank, before..., after..., scalar constant]` |
| 17 | [`SLICE`](../../../../glossary.md#slice) | `[count, starts..., lengths..., axes..., steps...]` |
| 18 | `TILE` | `list<u64 repeat>` |
| 19 | `WINDOW_2D` | [kernel](../../../../glossary.md#kernel) H/W, [stride](../../../../glossary.md#term-stride) H/W, [padding](../../../../glossary.md#padding) H/W, [dilation](../../../../glossary.md#term-dilation) H/W, `ceilMode` |
| 20 | `PADDED_WINDOW_2D` | `WINDOW_2D`, then padding `scalar` |
| 21 | `FOLD_WINDOW_2D` | output `shape`, then `WINDOW_2D` |
| 22 | `WINDOW_3D` | kernel D/H/W, stride D/H/W, padding D/H/W, dilation D/H/W, `ceilMode` |
| 23 | `PADDED_WINDOW_3D` | `WINDOW_3D`, then padding `scalar` |
| 24 | `FOLD_WINDOW_3D` | output `shape`, then `WINDOW_3D` |
| 25 | [`MSE`](../../../../glossary.md#mean-squared-error--mse) | one [loss](../../../../glossary.md#loss)-reduction wire |
| 26 | `DENSE_CROSS_ENTROPY` | `[axis, lossReduction]` |
| 27 | `INDEX_CROSS_ENTROPY` | `[axis, lossReduction, optionalScalar ignoreIndex]` |
| 28 | `NORMALIZED_SHAPE_EPSILON` | normalized `shape`, then [epsilon](../../../../glossary.md#epsilon) `scalar`; operation and [arity](../../../../glossary.md#term-arity) distinguish LayerNorm, affine LayerNorm, and RMSNorm |
| 29 | `BATCH_NORM_INFERENCE` | `[channelAxis, scalar epsilon]` |
| 30 | `BATCH_NORM_TRAINING` | `[channelAxis, scalar momentum, scalar epsilon]` |
| 31 | `SORT` | `[axis, descending]` |
| 32 | `TOP_K` | `[axis, k, largest, sorted]` |
| 33 | [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) | exact raw binary64 probability bits |
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
axis, multi-axis, and [sum-to-shape](../../../../glossary.md#sum-to-shape) variants; masked, arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum), and statistical forms remain
separate because they [carry](../../../../glossary.md#term-carry) different inputs or semantics.

Nested enum wires are append-only and explicit: scatter `NONE=1`, `ADD=2`, `MUL=3`, `MAX=4`,
`MIN=5`; loss `NONE=1`, `SUM=2`, `MEAN=3`; arg-extrema tie `FIRST_INDEX=1`, `LAST_INDEX=2`; and
recurrent direction `FORWARD=1`, `REVERSE=2`. Zero and unknown enum wires are invalid.

The Java encoder maps only the exact attribute class accepted by the selected [Model](../../../../glossary.md#term-model) signature.
Operation/attribute/[cardinality](../../../../glossary.md#term-cardinality) mismatches, malformed list lengths, non-normalized or duplicate
axes, invalid enum/[boolean](../../../../glossary.md#data-type--datatype)/scalar words, illegal optional payloads, nonzero unused words, and
constructor-incompatible ranges fail in Java and independently in [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture).

## Java and native [validation](../../../../glossary.md#term-validation) boundary

The two validators must make the same structural decision without trusting each other.

Java validates before allocating the image or entering [FFM](../../../../glossary.md#term-ffm):

1. exact [operation](../../../../glossary.md#operation) wire, accepted [Model](../../../../glossary.md#term-model) [signature](../../../../glossary.md#term-signature), attribute family, and input/output [cardinality](../../../../glossary.md#term-cardinality);
2. explicit value type, fully static [rank](../../../../glossary.md#term-rank)-`0..16` shape, legal rank zero for the occurrence, resolved
   canonical or authenticated local affine layout, gradient bit, checked element and byte count;<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [occurrence glossary](../../../../glossary.md#term-occurrence); [canonical glossary](../../../../glossary.md#term-canonical-form); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
3. topological availability, unique production, ordered feed/target/reference closure, and exact
   prepared memory extents;<span data-glossary-navigation="true"> ([topological glossary](../../../../glossary.md#term-topological-order))</span>
4. every attribute primitive and [descriptor](../../../../glossary.md#term-descriptor)-relative bound that is decidable before execution;
5. all multiply/add/align conversions with `Math.*Exact`, no lossy `long`/`int` narrowing, and final
   image size in <span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing))</span>`1..Integer.MAX_VALUE`; and
6. capability and fixed-route ownership. A recognized wire without a current route never reaches
   production native creation.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

Native validates before creating an [MPSGraph](../../../../glossary.md#term-mpsgraph), pipeline, [buffer](../../../../glossary.md#term-buffer), executable, or [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) [owner](../../../../glossary.md#term-owner):

1. non-null [arguments](../../../../glossary.md#term-argument) and null output on entry;
2. magic/schema/total size, section arithmetic, alignment, canonical packing, reserved zeros, and
   exact end-of-image;<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [alignment glossary](../../../../glossary.md#term-alignment))</span>
3. all value/type/rank/[dimension](../../../../glossary.md#dimension)/[element](../../../../glossary.md#term-element)/byte bounds;
4. all feed/[target](../../../../glossary.md#term-target)/[node](../../../../glossary.md#node) reference ranges, [topology](../../../../glossary.md#term-topology), [producer](../../../../glossary.md#term-producer) uniqueness, and operation cardinality;
5. exact attribute wire, word count, primitive ranges, nested enum values, axes, shapes, and
   operation/attribute pairing; and<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis))</span>
6. current route support only after structural validity. Unsupported but well-formed future wires
   return status 13 with no published handle.

Both sides must reject before indexing any malformed range. No validation failure may leak a native
resource, mutate caller buffers, partially publish an executable, or fall through to another route.

## Typed ingress, consumers, [publication](../../../../glossary.md#term-publication), and cross-[owner](../../../../glossary.md#term-owner) transfer

The foundation removes [FLOAT32](../../../../glossary.md#term-float32)-named transfer APIs and migrates all callers to one typed [canonical](../../../../glossary.md#term-canonical-form)
contract. It does not change shared [Runtime](../../../../glossary.md#term-runtime) types, [ownership](../../../../glossary.md#term-ownership-meanings), validity transitions, scheduling, or
[synchronous](../../../../glossary.md#term-synchronous) execution policy.

- `MetalBackendRuntime` and `MetalBackendIntegration` accept canonical, fully static, zero-offset,
  non-view descriptors of every six current types, rank <span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>`0..16`, with exact byte widths.
- `CpuBackendRuntime`/`CpuBackendIntegration` expose the same typed predicate and binding semantics;
  the old <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`*ContiguousFloat32*` methods are removed rather than retained as [aliases](../../../../glossary.md#aliasing-and-arena).
- `CpuMetalPreparedBufferTransfer` keeps its class role but becomes type-generic in both directions.
  It still binds exact source/destination representations cold and performs one direct transfer on
  the hot path without route lookup, allocation, conversion, staging, or validity mutation.<span data-glossary-navigation="true"> ([hot path glossary](../../../../glossary.md#term-hot-path); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- BOOL ingress and CPU-to-Metal transfer validate every byte as zero or one before upload. Metal-
  produced BOOL consumers and publications retain canonical bytes. Integral and floating payloads
  preserve exact represented bits; no CAST, canonicalization, widening, or narrowing is implied.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [CPU glossary](../../../../glossary.md#term-cpu); [represented bits glossary](../../../../glossary.md#term-represented-bits); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [widening glossary](../../../../glossary.md#term-widening); [narrowing glossary](../../../../glossary.md#term-narrowing))</span>
- Rank-zero transfer has element count one. Positive ranks require positive extents. Dynamic/zero
  extents, unresolved layout, nonzero offsets, general striding, and unauthenticated views remain
  unsupported.<span data-glossary-navigation="true"> ([Rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [element glossary](../../../../glossary.md#term-element); [layout glossary](../../../../glossary.md#layout))</span>
- `copyToCanonicalHostBytes` publishes all six canonical types at [rank](../../../../glossary.md#term-rank) `0..16`, converting local
  multi-byte representation to the existing big-endian public canonical form. The existing
  authenticated positive-rank affine FLOAT32 publication remains exact and is not generalized by
  this task.<span data-glossary-navigation="true"> ([canonical form glossary](../../../../glossary.md#term-canonical-form))</span><span data-glossary-navigation="true"> ([big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries))</span>
- Program values use their declared type rather than inferred first-use type. BOOL and INT32/INT64
  results may be consumed by later nodes and may cross CPU/Metal ownership boundaries; per-kind
  capability and semantic validators still decide which such graphs are executable.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [nodes glossary](../../../../glossary.md#node); [graphs glossary](../../../../glossary.md#graph))</span>

[Borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [host storage](../../../../glossary.md#host-storage) remains caller-owned. Prepared and [run](../../../../glossary.md#term-run)-owned [Metal](../../../../glossary.md#term-metal) [buffers](../../../../glossary.md#term-buffer) keep their current
owners and close order. The image lives only through synchronous executable creation; [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture)
must not retain its pointer. A successful executable owns every deep-copied or compiled object it
needs, and every failure path rolls back in reverse acquisition order.

## Tuning and workload [identity](../../../../glossary.md#term-identity-meanings)

This cutover advances the [Metal](../../../../glossary.md#term-metal) local workload, exact-policy, candidate-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), compatibility,
route-choice, and tuning [codec](../../../../glossary.md#term-codec) identities together from version 13 to version 14. Candidate wire
identities `CUSTOM_SINGLE_NEG=1`, `MPSGRAPH=2`, and `CUSTOM_TASK0052=3` remain stable; complete-plan
wrapper version 1 remains stable.

Version 14 [hashes](../../../../glossary.md#term-hash) the complete logical [schema](../../../../glossary.md#term-schema)-13 program without whole-image duplication:
[operation](../../../../glossary.md#operation) and attribute wires, exact ordered cardinalities and references, every per-value type,
[rank](../../../../glossary.md#term-rank), [dimension](../../../../glossary.md#dimension) and [gradient](../../../../glossary.md#term-gradient) bit, feed/[target](../../../../glossary.md#term-target) order, exact attribute words, numerical profile,
route-relevant affine [authentication](../../../../glossary.md#term-authentication), and current target/budget facts. It must distinguish all six
types, rank zero from rank one, every output slot, optional values, enum modes, and exact [scalar](../../../../glossary.md#term-scalar) [raw
bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) including floating [signed zero](../../../../glossary.md#term-signed-zero) and [NaN](../../../../glossary.md#term-nan) payloads.

Codec 13 bytes are incompatible and rejected; no migration or dual reader is added. [Cache](../../../../glossary.md#term-cache) misses
remain safe. The existing route [heuristic](../../../../glossary.md#term-heuristic) and selected routes are unchanged, and no benchmark,
[runtime](../../../../glossary.md#term-runtime) timing, retry, or input-dependent choice is introduced.

## Native package and exports

The implementation updates the native source, Java [FFM](../../../../glossary.md#term-ffm) [descriptor](../../../../glossary.md#term-descriptor), local package [producer](../../../../glossary.md#term-producer),
verifier, package README, and exact [manifest](../../../../glossary.md#term-manifest) expectations atomically:

- `nativeAbiVersion` becomes `5`;
- `nodeSchemaVersion` becomes `13`;
- manifest [schema](../../../../glossary.md#term-schema) version `1`, platform, architecture, minimum macOS, install name, frameworks,
  signature identity, checksums, and local-only distribution remain unchanged;<span data-glossary-navigation="true"> ([install name glossary](../../../../glossary.md#term-install-name); [identity glossary](../../../../glossary.md#term-identity-meanings); [distribution glossary](../../../../glossary.md#term-distribution))</span>
- the existing thirteen symbols remain the exact export set; and
- verifier logic checks ABI 5, the new executable-create signature through real loading/tests, and
  schema 13 metadata. It must reject an ABI-4/schema-12 dylib rather than silently accepting it.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [dylib glossary](../../../../glossary.md#term-dylib))</span>

Changing one existing [symbol](../../../../glossary.md#term-symbol)'s function [signature](../../../../glossary.md#term-signature) is the evidence that makes [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5 necessary; no
fourteenth export is needed. The three custom-NEG exports remain because removing them is unrelated
to the schema prerequisite and would change the current candidate surface.

## [Dependency](../../../../glossary.md#term-dependency) and implementation order

Implementation is serial and [atomic](../../../../glossary.md#resource-transaction-and-rollback) at the repository boundary:

1. **Freeze source-derived tables.** Add behavioral tests that enumerate the current 115 operation
   identities, exact Model signatures, six data types, normalized attribute mappings, and the
   unchanged <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>`34+2+79` capability [partition](../../../../glossary.md#partition).
2. **Implement the Java [schema](../../../../glossary.md#term-schema)-13 [model](../../../../glossary.md#term-model) and encoder.** Replace `MetalMpsGraphProgram`'s fixed record
   with the bounded image model, checked size calculator, wire tables, canonical encoder, and Java
   validator. Migrate lowering for all current 34 routes; do not add false-kind lowering to the
   production preparation path.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [lowering glossary](../../../../glossary.md#lowering))</span>
3. **Cut native [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5/schema 13.** Replace the old create signature and decoder, independently
   validate the entire image, map explicit value descriptors for current routes, add status 13,
   preserve the other twelve function signatures and all thirteen names, and delete every ABI-4/
   schema-12 struct, constant, parser, and fixture.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [constant glossary](../../../../glossary.md#term-constant); [fixture glossary](../../../../glossary.md#term-fixture))</span>
4. **Generalize typed [buffer](../../../../glossary.md#term-buffer) plumbing.** Migrate [Metal](../../../../glossary.md#term-metal) and [CPU](../../../../glossary.md#term-cpu) integration predicates/binders,
   `CpuMetalPreparedBufferTransfer`, ingress, publication, BOOL checking, rank-zero handling, and
   all call sites for the six exact byte widths without touching <span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [BOOL glossary](../../../../glossary.md#data-type--datatype); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`modules/runtime` policy.
5. **Advance [identity](../../../../glossary.md#term-identity-meanings) once.** Move all local identity/codec components to 14 and make the workload
   digest consume the same schema facts. Preserve candidate and complete-plan wrapper identities.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [digest glossary](../../../../glossary.md#term-digest))</span>
6. **Update package tooling and documentation.** Set ABI 5/schema 13 in producer/verifier metadata,
   document the changed create signature and typed transfer boundary, and remove stale schema-12,
   160-byte, FLOAT32-only, and terminal-BOOL claims.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [metadata glossary](../../../../glossary.md#term-metadata); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
7. **Validate once after the cutover.** Run focused pure-Java schema/transfer/identity tests first,
   then native/package/real-device tests for existing routes, then the repository validation owned
   by the implementation brief. Do not split a compatible Java half from an incompatible native or
   package half.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [device glossary](../../../../glossary.md#term-device); [validation glossary](../../../../glossary.md#term-validation))</span>

Tasks 0053 and the 79-kind workstreams consume this foundation only after Task 0055 is Complete.
They retain their independent proof, [oracle](../../../../glossary.md#term-test-oracle), measured-cost, route, [gradient](../../../../glossary.md#term-gradient), and capability gates.
No [operation](../../../../glossary.md#operation) task may reinterpret a schema wire as authorization.

## Expected implementation [scope](../../../../glossary.md#term-scope)

The implementation is expected to modify existing files rather than create a parallel framework:

- Metal program model/lowering/preparation/native binding, typed buffer integration, workload/tuning
  identity, tests, package Javadocs, and package documentation;<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding); [buffer glossary](../../../../glossary.md#term-buffer); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) native [schema](../../../../glossary.md#term-schema)/parser/create path and native tests;
- [CPU](../../../../glossary.md#term-cpu) typed transfer predicates/binders and focused tests;
- [Engine](../../../../glossary.md#term-engine) CPU/[Metal](../../../../glossary.md#term-metal) prepared transfer and [registry](../../../../glossary.md#term-registry)/session tests;
- native package [producer](../../../../glossary.md#term-producer)/verifier/README and [manifest](../../../../glossary.md#term-manifest) expectations; and
- Metal master/roadmap/task result documentation after completion.

`modules/model`, `modules/runtime`, [Planning](../../../../glossary.md#term-planning) [ownership](../../../../glossary.md#term-ownership-meanings), [Compiler](../../../../glossary.md#term-compiler) generation, public configuration,
[Trace](../../../../glossary.md#trace) DTOs, Config policy, and unrelated [backend](../../../../glossary.md#backend) [operation](../../../../glossary.md#operation) implementations are not implementation
scope unless a concrete [compile](../../../../glossary.md#compile) failure proves a caller migration is unavoidable. Such a finding
must preserve the public contract rather than widen this task.

## Required tests and proof of the future implementation

### Pure Java [schema](../../../../glossary.md#term-schema) behavior

- all 115 [operation](../../../../glossary.md#operation) wires are unique and every current kind maps exactly once;
- every [signature](../../../../glossary.md#term-signature) row above accepts its boundary cardinalities and rejects adjacent invalid counts;
- representative [cardinality](../../../../glossary.md#term-cardinality) proofs cover zero-input [RNG](../../../../glossary.md#graph-rng-state--graphrngstate), 17-input composition, attention `4 -> 2`,
  BatchNorm training `5 -> 5`, Top-K `1 -> 2`, [dropout](../../../../glossary.md#dropout--inverted-dropout) `2 -> 3`, and [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) `7 -> 3`;
- all six value types and typed scalars preserve exact raw bits, including FLOAT64/FLOAT32 signed
  zero and NaN payloads, all BFLOAT16 patterns, INT minima/maxima, and BOOL zero/one;<span data-glossary-navigation="true"> ([scalars glossary](../../../../glossary.md#term-scalar); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- [rank](../../../../glossary.md#term-rank) zero, rank sixteen, checked [element](../../../../glossary.md#term-element)/byte [overflow](../../../../glossary.md#term-overflow), `Integer.MAX_VALUE` count/size boundaries,
  and exact image alignment/size behavior are covered;<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- every normalized attribute family and nested enum has a [round-trip](../../../../glossary.md#schema-serialization-and-checkpoint)/golden behavior test; and
- malformed topology, mismatched operation/attribute/cardinality, duplicate producer, forward
  reference, bad feed/target, unknown wire, bad reserved/padding word, truncated image, and range
  overflow fail deterministically.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [producer glossary](../../../../glossary.md#term-producer); [target glossary](../../../../glossary.md#term-target); [padding glossary](../../../../glossary.md#padding))</span>

### Native and [ABI](../../../../glossary.md#term-application-binary-interface-abi) behavior

- ABI reports 5 and the package exports exactly the same thirteen names;
- Java and native accept the same valid [schema](../../../../glossary.md#term-schema)-13 images for current production routes;
- native independently rejects each malformed-image class before creating resources;
- a structurally valid currently unsupported [operation](../../../../glossary.md#operation) returns status 13 and no handle;
- existing 34-kind routes still execute their exact bounded behavior through schema 13; and
- create/release/run failure injection proves no image pointer retention, double release, partial
  publication, or resource leak.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release); [run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication))</span>

### Typed transfer behavior

- CPU-to-Metal and Metal-to-CPU direct transfer preserves representative raw bytes for all six
  types at rank zero and positive rank;<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [rank glossary](../../../../glossary.md#term-rank))</span>
- [BOOL](../../../../glossary.md#data-type--datatype) zero and one transfer, while any other BOOL byte is rejected before upload;
- exact byte extent, prepared-slot mismatch, overflow, closed/wrong-context representation,
  unresolved/dynamic/zero extent, noncanonical layout, offset, and unauthenticated view all reject;<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [overflow glossary](../../../../glossary.md#term-overflow); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- [canonical](../../../../glossary.md#term-canonical-form) [publication](../../../../glossary.md#term-publication) yields the existing [big-endian](../../../../glossary.md#text-encoding-and-file-boundaries) [host](../../../../glossary.md#term-host) bytes for every type; and
- cold binding retains exact handles/type/extent while the hot action performs no allocation,
  lookup, route selection, conversion, or validity mutation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [lookup glossary](../../../../glossary.md#term-lookup))</span>

### Capability, [identity](../../../../glossary.md#term-identity-meanings), package, and integration regressions

- capability remains exactly 34 current true kinds and 81 false kinds under the existing bounded
  rows; no new kind, dtype row, layout row, gradient row, or numerical route becomes true;<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- identity version 14 distinguishes all new schema facts and rejects version 13 while retaining
  candidate wires <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`1..3` and complete-plan wrapper 1;
- package producer and verifier agree on ABI 5/schema 13, exact exports, checksum, install name,
  frameworks, signature, and local archive behavior; and<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [checksum glossary](../../../../glossary.md#term-checksum); [install name glossary](../../../../glossary.md#term-install-name); [signature glossary](../../../../glossary.md#term-signature))</span>
- existing public Engine single-owner and mixed CPU/Metal scenarios execute against the newly built
  verified dylib with zero skips. Final validation must exercise the real surface, not only mocks.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [dylib glossary](../../../../glossary.md#term-dylib); [validation glossary](../../../../glossary.md#term-validation))</span>

## Explicit non-goals

Task 0055 does not:

- make any of the 81 currently false kinds capability-true;
- implement, probe, prove, benchmark, or choose a numerical [operation](../../../../glossary.md#operation) route;
- rerun Task 0051, Task 0052, or Task 0053 oracles or alter their evidence;
- add FLOAT16, dynamic or zero-extent tensors, rank above 16, general strided/offset transfer, or
  cross-device transfer;<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [extent glossary](../../../../glossary.md#term-extent); [tensors glossary](../../../../glossary.md#tensor); [rank glossary](../../../../glossary.md#term-rank); [device glossary](../../../../glossary.md#term-device))</span>
- change public Model operations, attributes, data types, Tensor descriptors, numerical profiles,
  gradient semantics, Compiler generation, Planning ownership, Runtime validity/lifecycle policy,
  asynchronous execution, default-device selection, or fallback behavior;<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [descriptors glossary](../../../../glossary.md#term-descriptor); [gradient glossary](../../../../glossary.md#term-gradient); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Runtime glossary](../../../../glossary.md#term-runtime); [lifecycle glossary](../../../../glossary.md#lifecycle); [asynchronous glossary](../../../../glossary.md#term-asynchronous))</span>
- add an export, generic attribute map, reflection/string dispatch, pointer-bearing node records,
  operation-specific side-channel arrays, compatibility shim, dual decoder, or cache migration;<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [node glossary](../../../../glossary.md#node); [channel glossary](../../../../glossary.md#term-channel); [cache glossary](../../../../glossary.md#term-cache))</span>
- generalize authenticated affine publication beyond its existing positive-rank FLOAT32 contract;
  or<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- remove the three custom-NEG exports, rename public configuration, publish a native artifact, add
  discovery, Developer ID signing, notarization, or release automation.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [notarization glossary](../../../../glossary.md#term-notarization); [release glossary](../../../../glossary.md#term-release))</span>

## Acceptance criteria

1. The detailed task records the exact current [signature](../../../../glossary.md#term-signature) inventory and all 115 [operation](../../../../glossary.md#operation) wires.
2. The 160-byte record decision is explicit and replaced by exactly one bounded schema-13 program
   image with exact layouts, counts, ordering, rank-zero, type, and attribute rules.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
3. All six current Model carriers, BOOL/INT consumers, exact typed scalars, and both-direction typed
   CPU/Metal transfer are specified without changing Model or Runtime policy.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [BOOL glossary](../../../../glossary.md#data-type--datatype); [scalars glossary](../../../../glossary.md#term-scalar); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
4. Java/native structural validation, checked arithmetic, lifecycle/rollback, unsupported-operation
   behavior, and no-dual-decoder migration are complete and symmetric.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle); [rollback glossary](../../../../glossary.md#term-rollback))</span>
5. [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5 retains exactly thirteen export names; [manifest](../../../../glossary.md#term-manifest) [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) 13 and native ABI 5 are exact.
6. Version-14 tuning/workload identities cover the new facts while candidate wires and wrapper 1
   remain stable.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
7. Capability remains exactly `34+2+79=115`; no route, proof, oracle, benchmark, or operation
   capability is added by the foundation.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>
8. Implementation files, tests, dependency order, cleanup, package changes, and explicit non-goals
   landed as one atomic cutover with every caller migrated.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
9. Focused Java, native, packaged-dylib, typed-transfer, and public Engine checks pass without a
   timing run or a second repository-wide build.<span data-glossary-navigation="true"> ([dylib glossary](../../../../glossary.md#term-dylib); [Engine glossary](../../../../glossary.md#term-engine); [run glossary](../../../../glossary.md#term-run))</span>

## Implementation [validation](../../../../glossary.md#term-validation)

The completed cutover was verified with:

```text
native/metal-macos-arm64/build.sh
native/metal-macos-arm64/package-local.sh build/libsynaptik_metal_foundation.dylib
native/metal-macos-arm64/verify-package.sh build/package-v1/macos-arm64
./gradlew :backends:metal:test
./gradlew :backends:cpu:test --tests \
  io.github.pho001.synaptik.backend.cpu.spi.CpuBackendIntegrationAndCpuPreparedScheduleAssemblerPublicTest
./gradlew :testing:integration-tests:test --tests \
  io.github.pho001.synaptik.testing.integration.EngineExplicitCompositionMetalIntegrationTest
git diff --check
```

The packaged-[dylib](../../../../glossary.md#term-dylib) focus re-exercised [schema](../../../../glossary.md#term-schema)-image/native-[ABI](../../../../glossary.md#term-application-binary-interface-abi) tests, all-six typed transfers, and
the public [Engine](../../../../glossary.md#term-engine) [Metal](../../../../glossary.md#term-metal) integration surface. Capability remains exactly 34 executable [registry](../../../../glossary.md#term-registry)
kinds and 81 structurally registered unsupported kinds. No performance benchmark or timing
measurement was [run](../../../../glossary.md#term-run), and the requested focused validation deliberately omitted another full
repository build.
