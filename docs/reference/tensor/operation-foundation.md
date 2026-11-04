# Operation signatures and semantic foundation

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

<a id="operation-semantic-foundation"></a>
## Operation semantic foundation

The public operation-foundation contracts live in `io.github.pho001.synaptik.model.operation`.
They separate semantic identity, typed parameters, explicit parameter absence, and immutable
composition while the table also lists the current production families:

| Concept | Current role |
|---|---|
| `OperationKind` | An implemented open interface that identifies which backend-independent computation an operation describes. |
| `OperationAttrs` | An implemented marker interface for the immutable, typed semantic parameters of that computation. |
| `NoOperationAttrs.INSTANCE` | The implemented canonical attribute value for a kind that has no semantic parameters. |
| `OperationSignature` | An implemented immutable structural contract that pairs one exact attributes class with inclusive input/output-count bounds. |
| `Operation` | An implemented immutable descriptor that stores one kind and one compatible `OperationAttrs` value. |
| `BinaryArithmeticKind` | The implemented production enum for seven parameterless tensor-to-tensor arithmetic meanings. |
| `BinaryComparisonKind` | The implemented production enum for six parameterless ordered tensor-to-tensor comparison meanings. |
| `BooleanLogicalKind` | The implemented production enum for parameterless elementwise boolean conjunction, disjunction, and negation meanings. |
| `WhereSelectionKind` | The implemented production enum for one parameterless ternary conditional-selection meaning. |
| `CastKind` | The implemented production enum for one parameterized elementwise data-type conversion meaning. |
| `CastAttrs` | The implemented immutable value for one exact non-null target `DataType`. |
| `AggregateReductionKind` | The implemented production enum for seven ordinary aggregates, five floating advanced/statistical reductions, and axis-only arg-min/arg-max. |
| `AxisReductionAttrs` | The implemented immutable normalized-axis and retained-dimension value for ordinary single-axis reductions. |
| `MultiAxisReductionAttrs` | The implemented immutable caller-ordered distinct normalized axes and retained-dimension value for ordinary, log-sum-exp, and norm reductions. |
| `StatisticalReductionAttrs` | The implemented immutable ordered axes, retained-dimension choice, and non-negative correction for variance/standard deviation. |
| `ArgExtremaAttrs` | The implemented immutable normalized-axis, retained-dimension, and explicit tie-policy value shared by arg-min and arg-max. |
| `ArgExtremaTiePolicy` | The implemented `FIRST_INDEX` or `LAST_INDEX` choice for equal extrema. |
| `MaskedReductionAttrs` | The implemented immutable normalized reduction axis for first-class, two-input masked sum and mean semantics. |
| `CumulativeScanKind` | The implemented production enum whose `CUM_SUM` and `CUM_PROD` values identify cumulative addition and multiplication along one axis. |
| `CumulativeScanAttrs` | The implemented immutable normalized axis, inclusive/exclusive choice, and forward/reverse traversal choice shared by both cumulative-scan kinds. |
| `SoftmaxKind` | The implemented production enum for one-axis probability and log-probability normalization meanings. |
| `SoftmaxAttrs` | The implemented immutable normalized-axis value shared by softmax and log-softmax. |
| `BatchNormKind` | The implemented production enum for explicit five-input batch-normalization inference and pure five-output training/statistic-transition meanings. |
| `BatchNormInferenceAttrs` | The implemented immutable normalized channel-axis and exact typed epsilon value. |
| `BatchNormTrainingAttrs` | The implemented immutable normalized channel axis, exact typed new-batch-weight momentum, and exact typed epsilon. |
| `BatchNormTrainingResult` | The implemented public carrier for training output, next running mean, and next running variance; saved producer outputs remain intentionally omitted. |
| `ScaledDotProductAttentionKind` | The implemented production enum whose sole value identifies scaled dot-product attention with an exact three-to-four-input and one-through-two-output occurrence signature. |
| `ScaledDotProductAttentionAttrs` | The implemented immutable optional exact scale and top-left causal-eligibility value. |
| `ScaledDotProductAttentionResult` | The implemented public carrier for same-occurrence output slot zero and normalized-weights slot one. |
| `ContiguousKind` | The implemented production enum whose sole `CONTIGUOUS` value requests logically equivalent canonical dense row-major, zero-offset result geometry. |
| `ShapeTransformKind` | The implemented production enum for ordered-element-preserving `RESHAPE` and singleton/leading-axis-repeating `EXPAND` meanings. |
| `TargetShapeAttrs` | The implemented immutable normalized target-`Shape` value shared by reshape and expand. |
| `AxisTransformKind` | The implemented production enum for complete axis permutation, singleton-axis insertion, and selected singleton-axis removal meanings. |
| `PermutationAttrs` | The implemented immutable complete normalized output-to-input axis permutation. |
| `AxisTransformAttrs` | The implemented immutable normalized non-negative position shared by singleton-axis insertion and removal. |
| `SliceKind` | The implemented production enum for signed extraction and functional slice replacement. |
| `SliceAttrs` | The implemented immutable normalized starts, selected lengths, distinct axes, and signed non-zero steps shared by extraction and replacement. |
| `CropToShapeAttrs` | The implemented immutable exact target and per-axis prefix Shapes for target-relative `SLICE` extraction or `SLICE_UPDATE` placement. |
| `PadKind` | The implemented production enum whose sole `PAD` value identifies constant padding around every input axis. |
| `PadAttrs` | The implemented immutable ordered before/after widths and exact typed padding constant. |
| `TileKind` | The implemented production enum whose sole `TILE` value identifies complete-pattern per-axis tiling. |
| `TileAttrs` | The implemented immutable ordered positive complete-pattern repeat counts. |
| `TensorCompositionKind` | The implemented production enum for ordered `CONCAT` and inserted-axis `STACK` meanings. |
| `CompositionAxisAttrs` | The implemented immutable normalized non-negative axis shared by concat and stack. |
| `SelectKind` | The implemented production enum whose sole `SELECT` value fixes one scalar coordinate on one source axis and removes that axis. |
| `SelectAttrs` | The implemented immutable normalized non-negative source axis and scalar coordinate for `SELECT`. |
| `AxisGatherKind` | The implemented production enum for canonical Gather and aligned same-rank Gather Elements meanings. |
| `IndexAxisAttrs` | The implemented immutable normalized non-negative data-axis value shared by Gather, Gather Elements, and Gather-compatible Scatter Add. |
| `OneHotKind` | The implemented production enum whose sole `ONE_HOT` value identifies dense trailing-axis BOOL index encoding. |
| `OneHotAttrs` | The implemented immutable positive static depth of one-hot's appended class axis. |
| `AxisScatterKind` | The implemented production enum for configurable same-rank Scatter Elements and fixed-add Gather-compatible Scatter Add. |
| `ScatterReduction` | The implemented reusable replacement, addition, multiplication, maximum, or minimum meaning for configurable functional scatter. |
| `ScatterElementsAttrs` | The implemented immutable normalized axis and explicit non-null reduction for `SCATTER_ELEMENTS`. |
| `UnaryElementwiseKind` | The implemented production enum for nineteen parameterless unary numeric and activation meanings. |
| `FloatingClassificationKind` | The implemented production enum for three parameterless floating value classifications with fixed BOOL results. |
| `ScalarElementwiseKind` | The implemented production enum for eight parameterized one-input scalar elementwise meanings. |
| `ScalarValueAttrs` | The implemented immutable holder for one exact typed scalar parameter. |
| `ClampRangeAttrs` | The implemented immutable value for exact same-type numeric inclusive clamp bounds. |

`OperationKind` declares `name()` and `signatures()`. The name is stable diagnostic text, not a
serialization token or global lookup key. Each concrete family owns a stable immutable non-empty
list of signatures, so adding a family does not require a central registry or a root switch over
all kinds. `signatureFor(attrs)` matches the exact runtime attributes class and rejects missing,
duplicate, null, or incompatible declarations. Two values from different kind families remain
different typed values even if both names contain the same text.

An `OperationSignature` describes one structurally valid occurrence variant. Its inclusive
minimum and maximum input/output counts describe logical positions, not Tensor rank or element
count. Equal minimum and maximum values form a **fixed** cardinality. Different finite bounds form
a **bounded** cardinality. `Integer.MAX_VALUE` is the inclusive upper bound used for an
effectively **variadic** input list. Signatures deliberately do not validate operand data types,
Shapes, layouts, numerical policy, backend support, or executability.

`OperationAttrs` deliberately declares no methods. A family-specific implementation defines its
own typed fields and must be immutable, defensively isolate mutable constructor inputs, and provide
structural equality and hashing. `NoOperationAttrs.INSTANCE` makes the absence of parameters an
explicit non-null value rather than `null` or an empty map. The marker cannot enforce immutability,
but `Operation` now validates that the exact concrete attributes class is explicitly accepted by
the selected kind before retaining both references.


---

[Previous topic](identifiers.md) · [Next topic](operation-arithmetic.md)
