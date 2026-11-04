# [Operation](../../glossary.md#operation) [signatures](../../glossary.md#term-signature) and semantic foundation

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Three constructors answer three different questions

The Tensor expression helper normally calls `new Operation(kind, attrs)` with immutable meaning
and configuration. `Operation` retains those exact objects and checks their pairing, but receives
no operand Tensor or data array. A producer or `CompiledNode` then supplies ordered input/output
positions and checks their counts. Input-aware expression or Compiler checks separately validate
types and shapes. No numeric values are copied or evaluated by these description constructors.
Read [producer provenance](identity.md#expression-producers-and-indexed-provenance) for the source
occurrence and [compiled nodes](compiled-graph.md) for graph-local connections.

**[Reader model](../../glossary.md#term-reader-model).** An operation description separates **kind** (“what calculation?”) from
**attributes** (“which fixed [parameters](../../glossary.md#term-parameter)?”). Its **signature** checks which attribute class and
how many input/output positions are allowed. **[Arity](../../glossary.md#term-arity)** or **[cardinality](../../glossary.md#term-cardinality)** here means that position
count, not the number of array [axes](../../glossary.md#term-axis) or [elements](../../glossary.md#term-element). One input can itself be a million-element [tensor](../../glossary.md#tensor).
**Semantic** means the calculation's meaning, independent of the algorithm that will implement it.
An operation value is reusable description; a [producer](../../glossary.md#term-producer) or [graph](../../glossary.md#graph) [node](../../glossary.md#node) identifies one particular
[occurrence](../../glossary.md#term-occurrence) with particular inputs. Validating a signature does not establish numerical validity,
available [derivatives](../../glossary.md#term-derivative), or [backend](../../glossary.md#backend) support.

<a id="operation-semantic-foundation"></a>
## [Operation](../../glossary.md#operation) semantic foundation

The public operation-foundation contracts live in `io.github.pho001.synaptik.model.operation`.
They separate semantic [identity](../../glossary.md#term-identity-meanings), typed [parameters](../../glossary.md#term-parameter), explicit parameter absence, and [immutable](../../glossary.md#term-immutable)
composition while the table also lists the current production families:

| Concept | Current role |
|---|---|
| [`OperationKind`](../../glossary.md#operationkind) | An implemented open interface that identifies which [backend](../../glossary.md#backend)-independent computation an operation describes. |
| [`OperationAttrs`](../../glossary.md#operationattrs) | An implemented [marker interface](../../glossary.md#term-marker-interface) for the immutable, typed semantic parameters of that computation. |
| `NoOperationAttrs.INSTANCE` | The implemented [canonical](../../glossary.md#term-canonical-form) attribute value for a kind that has no semantic parameters. |
| `OperationSignature` | An implemented immutable structural contract that pairs one exact attributes class with inclusive input/output-count bounds. |
| `Operation` | An implemented immutable [descriptor](../../glossary.md#term-descriptor) that stores one kind and one compatible `OperationAttrs` value. |
| `BinaryArithmeticKind` | The implemented production enum for seven parameterless [tensor](../../glossary.md#tensor)-to-tensor arithmetic meanings. |
| `BinaryComparisonKind` | The implemented production enum for six parameterless ordered tensor-to-tensor comparison meanings. |
| `BooleanLogicalKind` | The implemented production enum for parameterless [elementwise](../../glossary.md#term-elementwise) [boolean](../../glossary.md#data-type--datatype) conjunction, disjunction, and negation meanings. |
| `WhereSelectionKind` | The implemented production enum for one parameterless ternary conditional-selection meaning. |
| `CastKind` | The implemented production enum for one parameterized elementwise data-type conversion meaning. |
| `CastAttrs` | The implemented immutable value for one exact non-null [target](../../glossary.md#term-target) [`DataType`](../../glossary.md#data-type--datatype). |
| `AggregateReductionKind` | The implemented production enum for seven ordinary aggregates, five floating advanced/statistical reductions, and [axis](../../glossary.md#term-axis)-only arg-min/arg-max. |
| `AxisReductionAttrs` | The implemented immutable normalized-axis and retained-[dimension](../../glossary.md#dimension) value for ordinary single-axis reductions. |
| `MultiAxisReductionAttrs` | The implemented immutable caller-ordered distinct normalized axes and retained-dimension value for ordinary, [log-sum-exp](../../glossary.md#log-sum-exp), and [norm](../../glossary.md#term-norm) reductions. |
| `StatisticalReductionAttrs` | The implemented immutable ordered axes, retained-dimension choice, and non-negative correction for [variance](../../glossary.md#term-variance)/[standard deviation](../../glossary.md#term-standard-deviation). |
| `ArgExtremaAttrs` | The implemented immutable normalized-axis, retained-dimension, and explicit tie-policy value shared by arg-min and arg-max. |
| `ArgExtremaTiePolicy` | The implemented `FIRST_INDEX` or `LAST_INDEX` choice for equal [extrema](../../glossary.md#saturation-pole-and-extremum). |
| `MaskedReductionAttrs` | The implemented immutable normalized reduction axis for first-class, two-input masked sum and [mean](../../glossary.md#term-mean) semantics. |
| `CumulativeScanKind` | The implemented production enum whose `CUM_SUM` and `CUM_PROD` values identify cumulative addition and multiplication along one axis. |
| `CumulativeScanAttrs` | The implemented immutable [normalized axis](../../glossary.md#normalized-axis), inclusive/exclusive choice, and forward/reverse traversal choice shared by both cumulative-[scan](../../glossary.md#term-cumulative-scan) kinds. |
| `SoftmaxKind` | The implemented production enum for one-axis probability and [log](../../glossary.md#term-log)-probability [normalization](../../glossary.md#term-normalization) meanings. |
| `SoftmaxAttrs` | The implemented immutable normalized-axis value shared by [softmax](../../glossary.md#softmax--log-softmax) and [log-softmax](../../glossary.md#softmax--log-softmax). |
| `BatchNormKind` | The implemented production enum for explicit five-input [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-normalization [inference](../../glossary.md#term-inference) and pure five-output training/[statistic](../../glossary.md#term-statistic)-transition meanings. |
| `BatchNormInferenceAttrs` | The implemented immutable normalized [channel](../../glossary.md#term-channel)-axis and exact typed [epsilon](../../glossary.md#epsilon) value. |
| `BatchNormTrainingAttrs` | The implemented immutable normalized channel axis, exact typed new-batch-[weight](../../glossary.md#term-weight) [momentum](../../glossary.md#term-momentum), and exact typed epsilon. |
| `BatchNormTrainingResult` | The implemented public [carrier](../../glossary.md#term-storage-carrier) for training output, next running mean, and next running variance; saved [producer](../../glossary.md#term-producer) outputs remain intentionally omitted. |
| `ScaledDotProductAttentionKind` | The implemented production enum whose sole value identifies [scaled dot-product attention](../../glossary.md#scaled-dot-product-attention) with an exact three-to-four-input and one-through-two-output [occurrence](../../glossary.md#term-occurrence) [signature](../../glossary.md#term-signature). |
| `ScaledDotProductAttentionAttrs` | The implemented immutable optional exact scale and top-left causal-eligibility value. |
| `ScaledDotProductAttentionResult` | The implemented public carrier for same-occurrence output slot zero and normalized-weights slot one. |
| `ContiguousKind` | The implemented production enum whose sole [`CONTIGUOUS`](../../glossary.md#term-contiguous) value requests logically equivalent canonical dense [row-major](../../glossary.md#term-row-major), zero-offset result geometry. |
| `ShapeTransformKind` | The implemented production enum for ordered-[element](../../glossary.md#term-element)-preserving `RESHAPE` and [singleton](../../glossary.md#term-singleton)/leading-axis-repeating `EXPAND` meanings. |
| `TargetShapeAttrs` | The implemented immutable normalized target-[`Shape`](../../glossary.md#shape) value shared by reshape and expand. |
| `AxisTransformKind` | The implemented production enum for complete axis [permutation](../../glossary.md#permutation), singleton-axis insertion, and selected singleton-axis removal meanings. |
| `PermutationAttrs` | The implemented immutable complete normalized output-to-input axis permutation. |
| `AxisTransformAttrs` | The implemented immutable normalized non-negative position shared by singleton-axis insertion and removal. |
| `SliceKind` | The implemented production enum for signed extraction and functional [slice](../../glossary.md#slice) replacement. |
| `SliceAttrs` | The implemented immutable normalized starts, selected lengths, distinct axes, and signed non-zero steps shared by extraction and replacement. |
| `CropToShapeAttrs` | The implemented immutable exact target and per-axis prefix Shapes for target-relative `SLICE` extraction or `SLICE_UPDATE` placement. |
| `PadKind` | The implemented production enum whose sole [`PAD`](../../glossary.md#padding) value identifies [constant](../../glossary.md#term-constant) [padding](../../glossary.md#padding) around every input axis. |
| `PadAttrs` | The implemented immutable ordered before/after widths and exact typed padding constant. |
| `TileKind` | The implemented production enum whose sole `TILE` value identifies complete-pattern per-axis [tiling](../../glossary.md#tiling). |
| `TileAttrs` | The implemented immutable ordered positive complete-pattern repeat counts. |
| `TensorCompositionKind` | The implemented production enum for ordered `CONCAT` and inserted-axis `STACK` meanings. |
| `CompositionAxisAttrs` | The implemented immutable normalized non-negative axis shared by concat and stack. |
| `SelectKind` | The implemented production enum whose sole `SELECT` value fixes one [scalar](../../glossary.md#term-scalar) [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) on one source axis and removes that axis. |
| `SelectAttrs` | The implemented immutable normalized non-negative source axis and scalar coordinate for `SELECT`. |
| `AxisGatherKind` | The implemented production enum for canonical [Gather](../../glossary.md#gather) and aligned same-[rank](../../glossary.md#term-rank) [Gather Elements](../../glossary.md#gather-elements) meanings. |
| `IndexAxisAttrs` | The implemented immutable normalized non-negative data-axis value shared by Gather, Gather Elements, and Gather-compatible [Scatter Add](../../glossary.md#scatter-add). |
| `OneHotKind` | The implemented production enum whose sole `ONE_HOT` value identifies dense trailing-axis [BOOL](../../glossary.md#data-type--datatype) index encoding. |
| `OneHotAttrs` | The implemented immutable positive static depth of one-hot's appended [class axis](../../glossary.md#class-axis). |
| `AxisScatterKind` | The implemented production enum for configurable same-rank [Scatter Elements](../../glossary.md#scatter-elements) and fixed-add Gather-compatible Scatter Add. |
| `ScatterReduction` | The implemented reusable replacement, addition, multiplication, maximum, or minimum meaning for configurable functional scatter. |
| `ScatterElementsAttrs` | The implemented immutable normalized axis and explicit non-null reduction for `SCATTER_ELEMENTS`. |
| `UnaryElementwiseKind` | The implemented production enum for nineteen parameterless [unary](../../glossary.md#term-unary) numeric and [activation](../../glossary.md#term-activation) meanings. |
| `FloatingClassificationKind` | The implemented production enum for three parameterless floating value classifications with fixed BOOL results. |
| `ScalarElementwiseKind` | The implemented production enum for eight parameterized one-input scalar elementwise meanings. |
| `ScalarValueAttrs` | The implemented immutable holder for one exact typed scalar parameter. |
| `ClampRangeAttrs` | The implemented immutable value for exact same-type numeric inclusive clamp bounds. |

`OperationKind` declares `name()` and `signatures()`. The name is stable diagnostic text, not a
[serialization](../../glossary.md#term-serialization) [token](../../glossary.md#term-token) or global [lookup](../../glossary.md#term-lookup) key. Each concrete family owns a stable immutable non-empty
list of signatures, so adding a family does not require a central [registry](../../glossary.md#term-registry) or a root switch over
all kinds. `signatureFor(attrs)` matches the exact [runtime](../../glossary.md#term-runtime) attributes class and rejects missing,
duplicate, null, or incompatible declarations. Two values from different kind families remain
different typed values even if both names contain the same text.

An `OperationSignature` describes one structurally valid occurrence variant. Its inclusive
minimum and maximum input/output counts describe logical positions, not Tensor rank or element
count. Equal minimum and maximum values form a **fixed** [cardinality](../../glossary.md#term-cardinality). Different [finite](../../glossary.md#term-finite) bounds form
a **bounded** cardinality. `Integer.MAX_VALUE` is the inclusive upper bound used for an
effectively **[variadic](../../glossary.md#term-variadic)** input list. Signatures deliberately do not validate operand data types,
Shapes, [layouts](../../glossary.md#layout), numerical policy, backend support, or executability.

`OperationAttrs` deliberately declares no methods. A family-specific implementation defines its
own typed fields and must be immutable, defensively isolate [mutable](../../glossary.md#term-mutable) constructor inputs, and provide
[structural equality](../../glossary.md#term-structural-equality) and hashing. `NoOperationAttrs.INSTANCE` makes the absence of parameters an
explicit non-null value rather than `null` or an empty map. The marker cannot enforce immutability,
but `Operation` now validates that the exact concrete attributes class is explicitly accepted by
the selected kind before retaining both references.


---

[Previous topic](identifiers.md) · [Next topic](operation-arithmetic.md)
