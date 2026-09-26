# Task 0056 route-evidence audit

Status: **Normative planning evidence for Task 0056**

This companion ledger closes the selector/source-evidence requirement for the
[Task 0056 route catalog](0056-deterministic-dual-route-catalog-and-fixed-route-identity.md).
It is descriptive planning evidence only: it authorizes no capability, route change, build, device
probe, oracle run, benchmark, or performance claim.

Every schema-13 operation wire occurs exactly once below. Each row gives:

1. the closed MPSGraph catalog state from Task 0056;
2. one exact installed-SDK selector or a finite ordered composition of exact selectors, including
   the header key that resolves to the installed absolute path; and
3. the exact Model enum member plus its bounded `OperationSignature` shape. The Model key resolves
   to the repository source that owns both the member's semantic Javadoc and its signatures.

Selector spelling omits Objective-C parameter types but retains every selector component. A `+`
means all named primitives participate in the finite composition; `then` states result flow. Literal
parameter tensors use `MEM::constantWithScalar:dataType:`,
`MEM::constantWithScalar:shape:dataType:`, or exact encoded bytes with
`MEM::constantWithData:shape:dataType:`. Descriptors, normalized axes, shapes, and literal values
come only from already-validated Model attributes; this ledger does not weaken those validations.

## Installed SDK header keys

The audited SDK root is exactly:

```text
/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Headers/
```

| Key | Header below the audited root |
|---|---|
| `ACT` | `MPSGraphActivationOps.h` |
| `AR` | `MPSGraphArithmeticOps.h` |
| `CONV` | `MPSGraphConvolutionOps.h` |
| `CUM` | `MPSGraphCumulativeOps.h` |
| `GATHER` | `MPSGraphGatherOps.h` |
| `IMCOL` | `MPSGraphImToColOps.h` |
| `LOSS` | `MPSGraphLossOps.h` |
| `MAT` | `MPSGraphMatrixMultiplicationOps.h` |
| `MEM` | `MPSGraphMemoryOps.h` |
| `NORM` | `MPSGraphNormalizationOps.h` |
| `ONEHOT` | `MPSGraphOneHotOps.h` |
| `POOL` | `MPSGraphPoolingOps.h` |
| `RANDOM` | `MPSGraphRandomOps.h` |
| `RED` | `MPSGraphReductionOps.h` |
| `RNN` | `MPSGraphRNNOps.h` |
| `SCATTER` | `MPSGraphScatterNDOps.h` |
| `SHAPE` | `MPSGraphTensorShapeOps.h` |
| `SORT` | `MPSGraphSortOps.h` |
| `TOPK` | `MPSGraphTopKOps.h` |

The evidence is intentionally tied to these installed headers. A future SDK replacement must rerun
this audit rather than silently treating selector names from another SDK as current evidence.

## Model semantic/signature source keys

Each link is the exact current source file whose enum Javadoc defines the semantics and whose
`signatures()` implementation owns the bounded input/output/attribute combinations cited per row.

| Key | Current Model source |
|---|---|
| `M-AGG` | [`AggregateReductionKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/AggregateReductionKind.java) |
| `M-ATTN` | [`ScaledDotProductAttentionKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionKind.java) |
| `M-AXIS-XFORM` | [`AxisTransformKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/AxisTransformKind.java) |
| `M-BINARY` | [`BinaryArithmeticKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/binary/BinaryArithmeticKind.java) |
| `M-BN` | [`BatchNormKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/BatchNormKind.java) |
| `M-BOOL` | [`BooleanLogicalKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/logical/BooleanLogicalKind.java) |
| `M-CAST` | [`CastKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastKind.java) |
| `M-CLASSIFY` | [`FloatingClassificationKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/classification/FloatingClassificationKind.java) |
| `M-COMPARE` | [`BinaryComparisonKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/comparison/BinaryComparisonKind.java) |
| `M-COMPOSE` | [`TensorCompositionKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/TensorCompositionKind.java) |
| `M-CONTIG` | [`ContiguousKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/ContiguousKind.java) |
| `M-CONV2` | [`Conv2dKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv2dKind.java) |
| `M-CONV3` | [`Conv3dKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv3dKind.java) |
| `M-DROPOUT` | [`DropoutKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/DropoutKind.java) |
| `M-GATHER` | [`AxisGatherKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisGatherKind.java) |
| `M-GATHER-ND` | [`GatherNdKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/GatherNdKind.java) |
| `M-LAYER` | [`LayerNormKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/LayerNormKind.java) |
| `M-LOSS` | [`LossKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/LossKind.java) |
| `M-MATMUL` | [`MatmulKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/linalg/MatmulKind.java) |
| `M-ONEHOT` | [`OneHotKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/OneHotKind.java) |
| `M-ORDER` | [`OrderingKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/ordering/OrderingKind.java) |
| `M-PAD` | [`PadKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/PadKind.java) |
| `M-POOL2` | [`Pool2dKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/Pool2dKind.java) |
| `M-POOL3` | [`Pool3dKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/Pool3dKind.java) |
| `M-RECURRENT` | [`RecurrentScanKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/recurrent/RecurrentScanKind.java) |
| `M-RMS` | [`RmsNormKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/RmsNormKind.java) |
| `M-RNG` | [`GraphRngKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/GraphRngKind.java) |
| `M-SCALAR` | [`ScalarElementwiseKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarElementwiseKind.java) |
| `M-SCAN` | [`CumulativeScanKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/CumulativeScanKind.java) |
| `M-SCATTER` | [`AxisScatterKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisScatterKind.java) |
| `M-SCATTER-ND` | [`ScatterNdKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdKind.java) |
| `M-SELECT` | [`SelectKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/SelectKind.java) |
| `M-SHAPE` | [`ShapeTransformKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/ShapeTransformKind.java) |
| `M-SLICE` | [`SliceKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/SliceKind.java) |
| `M-SOFTMAX` | [`SoftmaxKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/SoftmaxKind.java) |
| `M-TILE` | [`TileKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/TileKind.java) |
| `M-TOPK` | [`TopKKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/ordering/TopKKind.java) |
| `M-UNARY` | [`UnaryElementwiseKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKind.java) |
| `M-WHERE` | [`WhereSelectionKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/selection/WhereSelectionKind.java) |
| `M-WINDOW` | [`WindowTransformKind.java`](../../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformKind.java) |

Signature notation is `attrs; input-min..input-max -> output-min..output-max`; a single number is
exact. Slash-separated attribute types are the source-owned alternative signatures for that kind.

## Exhaustive per-wire selector/composition and Model evidence

| Wire | Exact Model kind and state | Installed SDK selector or exact finite composition | Bounded Model source/signature evidence |
|---:|---|---|---|
| 1 | `UnaryElementwiseKind.NEG`; `DIRECT` | `AR::negativeWithTensor:name:` | `M-UNARY::NEG`; `NoOperationAttrs; 1 -> 1` |
| 2 | `BinaryArithmeticKind.ADD`; `DIRECT` | `AR::additionWithPrimaryTensor:secondaryTensor:name:` | `M-BINARY::ADD`; `NoOperationAttrs; 2 -> 1` |
| 3 | `BinaryArithmeticKind.SUB`; `DIRECT` | `AR::subtractionWithPrimaryTensor:secondaryTensor:name:` | `M-BINARY::SUB`; `NoOperationAttrs; 2 -> 1` |
| 4 | `BinaryArithmeticKind.MUL`; `DIRECT` | `AR::multiplicationWithPrimaryTensor:secondaryTensor:name:` | `M-BINARY::MUL`; `NoOperationAttrs; 2 -> 1` |
| 5 | `BinaryArithmeticKind.DIV`; `DIRECT` | `AR::divisionWithPrimaryTensor:secondaryTensor:name:` | `M-BINARY::DIV`; `NoOperationAttrs; 2 -> 1` |
| 6 | `ShapeTransformKind.RESHAPE`; `DIRECT` | `SHAPE::reshapeTensor:withShape:name:` | `M-SHAPE::RESHAPE`; `TargetShapeAttrs; 1 -> 1` |
| 7 | `ShapeTransformKind.EXPAND`; `DIRECT` | `SHAPE::broadcastTensor:toShape:name:` | `M-SHAPE::EXPAND`; `TargetShapeAttrs; 1 -> 1` |
| 8 | `AxisTransformKind.PERMUTE`; `DIRECT` | `SHAPE::transposeTensor:permutation:name:` | `M-AXIS-XFORM::PERMUTE`; `PermutationAttrs; 1 -> 1` |
| 9 | `AxisTransformKind.EXPAND_DIMS`; `DIRECT` | `SHAPE::expandDimsOfTensor:axes:name:` | `M-AXIS-XFORM::EXPAND_DIMS`; `AxisTransformAttrs; 1 -> 1` |
| 10 | `AxisTransformKind.SQUEEZE`; `DIRECT` | `SHAPE::squeezeTensor:axes:name:` | `M-AXIS-XFORM::SQUEEZE`; `AxisTransformAttrs; 1 -> 1` |
| 11 | `ContiguousKind.CONTIGUOUS`; `DIRECT` | `SHAPE::reshapeTensor:withShape:name:` with the already-validated identical logical shape | `M-CONTIG::CONTIGUOUS`; `NoOperationAttrs; 1 -> 1` |
| 12 | `UnaryElementwiseKind.ABS`; `DIRECT` | `AR::absoluteWithTensor:name:` | `M-UNARY::ABS`; `NoOperationAttrs; 1 -> 1` |
| 13 | `AggregateReductionKind.SUM`; `DIRECT` | `RED::reductionSumWithTensor:axes:name:` | `M-AGG::SUM`; `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs/SumToShapeAttrs; 1 -> 1 / MaskedReductionAttrs; 2 -> 1` |
| 14 | `AggregateReductionKind.MEAN`; `DIRECT` | `NORM::meanOfTensor:axes:name:` | `M-AGG::MEAN`; `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs; 1 -> 1 / MaskedReductionAttrs; 2 -> 1` |
| 15 | `MatmulKind.MATMUL`; `DIRECT` | `MAT::matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:` | `M-MATMUL::MATMUL`; `NoOperationAttrs; 2 -> 1` |
| 16 | `AxisGatherKind.GATHER`; `DIRECT` | `GATHER::gatherWithUpdatesTensor:indicesTensor:axis:batchDimensions:name:` with batch dimensions zero | `M-GATHER::GATHER`; `IndexAxisAttrs; 2 -> 1` |
| 17 | `OneHotKind.ONE_HOT`; `DIRECT` | `ONEHOT::oneHotWithIndicesTensor:depth:axis:dataType:onValue:offValue:name:` | `M-ONEHOT::ONE_HOT`; `OneHotAttrs; 1 -> 1` |
| 18 | `AxisScatterKind.SCATTER_ELEMENTS`; `DIRECT` | `SCATTER::scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:` | `M-SCATTER::SCATTER_ELEMENTS`; `ScatterElementsAttrs; 3 -> 1` |
| 19 | `WindowTransformKind.UNFOLD_AXIS`; `COMPOSED` | repeat `SHAPE::sliceTensor:starts:ends:strides:name:` for each validated window, `SHAPE::reshapeTensor:withShape:name:`, then `SHAPE::concatTensors:dimension:name:` in canonical position order | `M-WINDOW::UNFOLD_AXIS`; `UnfoldAxisAttrs; 1 -> 1` |
| 20 | `BinaryComparisonKind.GREATER_THAN`; `DIRECT` | `AR::greaterThanWithPrimaryTensor:secondaryTensor:name:` | `M-COMPARE::GREATER_THAN`; `NoOperationAttrs; 2 -> 1` |
| 21 | `BinaryComparisonKind.GREATER_OR_EQUAL`; `DIRECT` | `AR::greaterThanOrEqualToWithPrimaryTensor:secondaryTensor:name:` | `M-COMPARE::GREATER_OR_EQUAL`; `NoOperationAttrs; 2 -> 1` |
| 22 | `BinaryComparisonKind.LESS_THAN`; `DIRECT` | `AR::lessThanWithPrimaryTensor:secondaryTensor:name:` | `M-COMPARE::LESS_THAN`; `NoOperationAttrs; 2 -> 1` |
| 23 | `BinaryComparisonKind.LESS_OR_EQUAL`; `DIRECT` | `AR::lessThanOrEqualToWithPrimaryTensor:secondaryTensor:name:` | `M-COMPARE::LESS_OR_EQUAL`; `NoOperationAttrs; 2 -> 1` |
| 24 | `BinaryComparisonKind.EQUAL`; `DIRECT` | `AR::equalWithPrimaryTensor:secondaryTensor:name:` | `M-COMPARE::EQUAL`; `NoOperationAttrs; 2 -> 1` |
| 25 | `BinaryComparisonKind.NOT_EQUAL`; `DIRECT` | `AR::notEqualWithPrimaryTensor:secondaryTensor:name:` | `M-COMPARE::NOT_EQUAL`; `NoOperationAttrs; 2 -> 1` |
| 26 | `BinaryArithmeticKind.MIN`; `DIRECT` | `AR::minimumWithPrimaryTensor:secondaryTensor:name:` | `M-BINARY::MIN`; `NoOperationAttrs; 2 -> 1` |
| 27 | `BinaryArithmeticKind.MAX`; `DIRECT` | `AR::maximumWithPrimaryTensor:secondaryTensor:name:` | `M-BINARY::MAX`; `NoOperationAttrs; 2 -> 1` |
| 28 | `ScalarElementwiseKind.MIN`; `COMPOSED` | `MEM::constantWithScalar:dataType:` + `AR::minimumWithPrimaryTensor:secondaryTensor:name:` | `M-SCALAR::MIN`; `ScalarValueAttrs; 1 -> 1` |
| 29 | `ScalarElementwiseKind.MAX`; `COMPOSED` | `MEM::constantWithScalar:dataType:` + `AR::maximumWithPrimaryTensor:secondaryTensor:name:` | `M-SCALAR::MAX`; `ScalarValueAttrs; 1 -> 1` |
| 30 | `ScalarElementwiseKind.CLAMP`; `DIRECT` | materialize validated literal bounds with `MEM::constantWithScalar:dataType:`, then one `AR::clampWithTensor:minValueTensor:maxValueTensor:name:` operation selector | `M-SCALAR::CLAMP`; `ClampRangeAttrs; 1 -> 1` |
| 31 | `AggregateReductionKind.MIN`; `DIRECT` | `RED::reductionMinimumWithTensor:axes:name:` | `M-AGG::MIN`; `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs; 1 -> 1` |
| 32 | `AggregateReductionKind.MAX`; `DIRECT` | `RED::reductionMaximumWithTensor:axes:name:` | `M-AGG::MAX`; `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs; 1 -> 1` |
| 33 | `CumulativeScanKind.CUM_SUM`; `DIRECT` | `CUM::cumulativeSumWithTensor:axis:exclusive:reverse:name:` | `M-SCAN::CUM_SUM`; `CumulativeScanAttrs; 1 -> 1` |
| 34 | `CumulativeScanKind.CUM_PROD`; `DIRECT` | `CUM::cumulativeProductWithTensor:axis:exclusive:reverse:name:` | `M-SCAN::CUM_PROD`; `CumulativeScanAttrs; 1 -> 1` |
| 35 | `ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION`; `DIRECT` | bounded base form: `MAT::scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:scale:name:`; mask, causal, and weights-output forms remain excluded | `M-ATTN::SCALED_DOT_PRODUCT_ATTENTION`; `ScaledDotProductAttentionAttrs; 3..4 -> 1..2` |
| 36 | `Conv2dKind.CONV2D`; `DIRECT` | bounded unbiased form: `CONV::convolution2DWithSourceTensor:weightsTensor:descriptor:name:`; optional bias remains excluded | `M-CONV2::CONV2D`; `Conv2dAttrs; 2..3 -> 1` |
| 37 | `Conv3dKind.CONV3D`; `DIRECT` | bounded unbiased form: `CONV::convolution3DWithSourceTensor:weightsTensor:descriptor:name:`; optional bias remains excluded | `M-CONV3::CONV3D`; `Conv3dAttrs; 2..3 -> 1` |
| 38 | `BinaryArithmeticKind.POW`; `DIRECT` | `AR::powerWithPrimaryTensor:secondaryTensor:name:` | `M-BINARY::POW`; `NoOperationAttrs; 2 -> 1` |
| 39 | `CastKind.CAST`; `DIRECT` | `SHAPE::castTensor:toType:name:` | `M-CAST::CAST`; `CastAttrs; 1 -> 1` |
| 40 | `FloatingClassificationKind.IS_FINITE`; `DIRECT` | `AR::isFiniteWithTensor:name:` | `M-CLASSIFY::IS_FINITE`; `NoOperationAttrs; 1 -> 1` |
| 41 | `FloatingClassificationKind.IS_NAN`; `DIRECT` | `AR::isNaNWithTensor:name:` | `M-CLASSIFY::IS_NAN`; `NoOperationAttrs; 1 -> 1` |
| 42 | `FloatingClassificationKind.IS_INF`; `DIRECT` | `AR::isInfiniteWithTensor:name:` | `M-CLASSIFY::IS_INF`; `NoOperationAttrs; 1 -> 1` |
| 43 | `BooleanLogicalKind.AND`; `DIRECT` | `AR::logicalANDWithPrimaryTensor:secondaryTensor:name:` | `M-BOOL::AND`; `NoOperationAttrs; 2 -> 1` |
| 44 | `BooleanLogicalKind.OR`; `DIRECT` | `AR::logicalORWithPrimaryTensor:secondaryTensor:name:` | `M-BOOL::OR`; `NoOperationAttrs; 2 -> 1` |
| 45 | `BooleanLogicalKind.NOT`; `DIRECT` | `AR::notWithTensor:name:` | `M-BOOL::NOT`; `NoOperationAttrs; 1 -> 1` |
| 46 | `ScalarElementwiseKind.ADD`; `COMPOSED` | exact four-byte raw scalar in rank-one `[1]` via `MEM::constantWithData:shape:dataType:` + `AR::additionWithPrimaryTensor:secondaryTensor:name:` with input primary and scalar secondary | `M-SCALAR::ADD`; `ScalarValueAttrs; 1 -> 1` |
| 47 | `ScalarElementwiseKind.SUB`; `COMPOSED` | exact four-byte raw scalar in rank-one `[1]` via `MEM::constantWithData:shape:dataType:` + `AR::subtractionWithPrimaryTensor:secondaryTensor:name:` with input primary and scalar secondary | `M-SCALAR::SUB`; `ScalarValueAttrs; 1 -> 1` |
| 48 | `ScalarElementwiseKind.MUL`; `COMPOSED` | exact four-byte raw scalar in rank-one `[1]` via `MEM::constantWithData:shape:dataType:` + `AR::multiplicationWithPrimaryTensor:secondaryTensor:name:` with input primary and scalar secondary | `M-SCALAR::MUL`; `ScalarValueAttrs; 1 -> 1` |
| 49 | `ScalarElementwiseKind.DIV`; `COMPOSED` | exact four-byte raw scalar in rank-one `[1]` via `MEM::constantWithData:shape:dataType:` + `AR::divisionWithPrimaryTensor:secondaryTensor:name:` with input primary and scalar secondary | `M-SCALAR::DIV`; `ScalarValueAttrs; 1 -> 1` |
| 50 | `ScalarElementwiseKind.POW`; `COMPOSED` | exact four-byte raw scalar in rank-one `[1]` via `MEM::constantWithData:shape:dataType:` + `AR::powerWithPrimaryTensor:secondaryTensor:name:` with input primary and scalar secondary | `M-SCALAR::POW`; `ScalarValueAttrs; 1 -> 1` |
| 51 | `WhereSelectionKind.WHERE`; `DIRECT` | `AR::selectWithPredicateTensor:truePredicateTensor:falsePredicateTensor:name:` | `M-WHERE::WHERE`; `NoOperationAttrs; 3 -> 1` |
| 52 | `UnaryElementwiseKind.RECIPROCAL`; `COMPOSED` | exact four-byte raw `+1.0f` in rank-one `[1]` via `MEM::constantWithData:shape:dataType:` + `AR::divisionWithPrimaryTensor:secondaryTensor:name:` with one primary and input secondary | `M-UNARY::RECIPROCAL`; `NoOperationAttrs; 1 -> 1` |
| 53 | `UnaryElementwiseKind.LOG`; `DIRECT` | `AR::logarithmWithTensor:name:` | `M-UNARY::LOG`; `NoOperationAttrs; 1 -> 1` |
| 54 | `UnaryElementwiseKind.LOG1P`; `COMPOSED` | `MEM::constantWithScalar:dataType:` (`1`) + `AR::additionWithPrimaryTensor:secondaryTensor:name:` then `AR::logarithmWithTensor:name:` | `M-UNARY::LOG1P`; `NoOperationAttrs; 1 -> 1` |
| 55 | `UnaryElementwiseKind.EXP`; `DIRECT` | `AR::exponentWithTensor:name:` | `M-UNARY::EXP`; `NoOperationAttrs; 1 -> 1` |
| 56 | `UnaryElementwiseKind.EXPM1`; `COMPOSED` | `AR::exponentWithTensor:name:` then `MEM::constantWithScalar:dataType:` (`1`) + `AR::subtractionWithPrimaryTensor:secondaryTensor:name:` | `M-UNARY::EXPM1`; `NoOperationAttrs; 1 -> 1` |
| 57 | `UnaryElementwiseKind.ERF`; `DIRECT` | `AR::erfWithTensor:name:` | `M-UNARY::ERF`; `NoOperationAttrs; 1 -> 1` |
| 58 | `UnaryElementwiseKind.SQRT`; `DIRECT` | `AR::squareRootWithTensor:name:` | `M-UNARY::SQRT`; `NoOperationAttrs; 1 -> 1` |
| 59 | `UnaryElementwiseKind.RSQRT`; `DIRECT` | `AR::reciprocalSquareRootWithTensor:name:` | `M-UNARY::RSQRT`; `NoOperationAttrs; 1 -> 1` |
| 60 | `UnaryElementwiseKind.FLOOR`; `DIRECT` | `AR::floorWithTensor:name:` | `M-UNARY::FLOOR`; `NoOperationAttrs; 1 -> 1` |
| 61 | `UnaryElementwiseKind.CEIL`; `DIRECT` | `AR::ceilWithTensor:name:` | `M-UNARY::CEIL`; `NoOperationAttrs; 1 -> 1` |
| 62 | `UnaryElementwiseKind.SIGN`; `DIRECT` | `AR::signWithTensor:name:` | `M-UNARY::SIGN`; `NoOperationAttrs; 1 -> 1` |
| 63 | `UnaryElementwiseKind.RELU`; `DIRECT` | `ACT::reLUWithTensor:name:` | `M-UNARY::RELU`; `NoOperationAttrs; 1 -> 1` |
| 64 | `UnaryElementwiseKind.SIGMOID`; `DIRECT` | `ACT::sigmoidWithTensor:name:` | `M-UNARY::SIGMOID`; `NoOperationAttrs; 1 -> 1` |
| 65 | `UnaryElementwiseKind.TANH`; `DIRECT` | `AR::tanhWithTensor:name:` | `M-UNARY::TANH`; `NoOperationAttrs; 1 -> 1` |
| 66 | `UnaryElementwiseKind.GELU`; `COMPOSED` | exact `0.5*x*(1+erf(x/sqrt(2)))` with `MEM::constantWithScalar:dataType:`, `AR::divisionWithPrimaryTensor:secondaryTensor:name:`, `AR::erfWithTensor:name:`, `AR::additionWithPrimaryTensor:secondaryTensor:name:`, and `AR::multiplicationWithPrimaryTensor:secondaryTensor:name:` | `M-UNARY::GELU`; `NoOperationAttrs; 1 -> 1` |
| 67 | `UnaryElementwiseKind.GELU_TANH_APPROXIMATION`; `COMPOSED` | Model formula `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))` with `MEM::constantWithScalar:dataType:` plus `AR::powerWithPrimaryTensor:secondaryTensor:name:`, `AR::multiplicationWithPrimaryTensor:secondaryTensor:name:`, `AR::additionWithPrimaryTensor:secondaryTensor:name:`, and `AR::tanhWithTensor:name:` | `M-UNARY::GELU_TANH_APPROXIMATION`; `NoOperationAttrs; 1 -> 1` |
| 68 | `UnaryElementwiseKind.SILU`; `COMPOSED` | `ACT::sigmoidWithTensor:name:` + `AR::multiplicationWithPrimaryTensor:secondaryTensor:name:` (`x * sigmoid(x)`) | `M-UNARY::SILU`; `NoOperationAttrs; 1 -> 1` |
| 69 | `AxisGatherKind.GATHER_ELEMENTS`; `DIRECT` | `GATHER::gatherAlongAxis:withUpdatesTensor:indicesTensor:name:` | `M-GATHER::GATHER_ELEMENTS`; `IndexAxisAttrs; 2 -> 1` |
| 70 | `AxisScatterKind.SCATTER_ADD`; `DIRECT` | `SCATTER::scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:` with add mode | `M-SCATTER::SCATTER_ADD`; `IndexAxisAttrs; 3 -> 1` |
| 71 | `GatherNdKind.GATHER_ND`; `DIRECT` | `GATHER::gatherNDWithUpdatesTensor:indicesTensor:batchDimensions:name:` | `M-GATHER-ND::GATHER_ND`; `GatherNdAttrs; 2 -> 1` |
| 72 | `ScatterNdKind.SCATTER_ND`; `DIRECT` | `SCATTER::scatterNDWithDataTensor:updatesTensor:indicesTensor:batchDimensions:mode:name:` | `M-SCATTER-ND::SCATTER_ND`; `ScatterNdAttrs; 3 -> 1` |
| 73 | `SelectKind.SELECT`; `COMPOSED` | `SHAPE::sliceTensor:starts:ends:strides:name:` with unit selected extent, then `SHAPE::squeezeTensor:axis:name:` | `M-SELECT::SELECT`; `SelectAttrs; 1 -> 1` |
| 74 | `PadKind.PAD`; `DIRECT` | `SHAPE::padTensor:withPaddingMode:leftPadding:rightPadding:constantValue:name:` | `M-PAD::PAD`; `PadAttrs; 1 -> 1` |
| 75 | `SliceKind.SLICE`; `DIRECT` | `SHAPE::sliceTensor:starts:ends:strides:name:` | `M-SLICE::SLICE`; `SliceAttrs/CropToShapeAttrs; 1 -> 1` |
| 76 | `SliceKind.SLICE_UPDATE`; `DIRECT` | `SHAPE::sliceUpdateDataTensor:updateTensor:starts:ends:strides:name:` | `M-SLICE::SLICE_UPDATE`; `SliceAttrs/CropToShapeAttrs; 2 -> 1` |
| 77 | `TensorCompositionKind.CONCAT`; `DIRECT` | `SHAPE::concatTensors:dimension:name:` | `M-COMPOSE::CONCAT`; `CompositionAxisAttrs; 1..Integer.MAX_VALUE -> 1` |
| 78 | `TensorCompositionKind.STACK`; `COMPOSED` | for every same-shaped input `SHAPE::expandDimsOfTensor:axis:name:`, then `SHAPE::concatTensors:dimension:name:`; the broader broadcast-compatible `SHAPE::stackTensors:axis:name:` selector is deliberately not the exact Model contract | `M-COMPOSE::STACK`; `CompositionAxisAttrs; 1..Integer.MAX_VALUE -> 1` |
| 79 | `TileKind.TILE`; `DIRECT` | `SHAPE::tileTensor:withMultiplier:name:` | `M-TILE::TILE`; `TileAttrs; 1 -> 1` |
| 80 | `WindowTransformKind.FOLD_AXIS`; `COMPOSED` | validated static geometry creates the complete destination map with `MEM::constantWithData:shape:dataType:`; `SHAPE::reshapeTensor:withShape:name:` aligns window updates, then `SCATTER::scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:` accumulates every contributor in add mode | `M-WINDOW::FOLD_AXIS`; `FoldAxisAttrs; 1 -> 1` |
| 81 | `WindowTransformKind.UNFOLD2D`; `DIRECT` | `IMCOL::imToColWithSourceTensor:descriptor:name:` | `M-WINDOW::UNFOLD2D`; `Window2dAttrs/Unfold2dAttrs; 1 -> 1` |
| 82 | `WindowTransformKind.FOLD2D`; `DIRECT` | `IMCOL::colToImWithSourceTensor:outputShape:descriptor:name:` | `M-WINDOW::FOLD2D`; `Fold2dAttrs; 1 -> 1` |
| 83 | `WindowTransformKind.UNFOLD3D`; `COMPOSED` | nested canonical NCDHW windows via repeated `SHAPE::sliceTensor:starts:ends:strides:name:`, `SHAPE::reshapeTensor:withShape:name:`, then `SHAPE::concatTensors:dimension:name:` | `M-WINDOW::UNFOLD3D`; `Window3dAttrs/Unfold3dAttrs; 1 -> 1` |
| 84 | `WindowTransformKind.FOLD3D`; `COMPOSED` | validated static NCDHW geometry creates canonical destination tuples with `MEM::constantWithData:shape:dataType:`; `SHAPE::reshapeTensor:withShape:name:` aligns updates, then `SCATTER::scatterNDWithDataTensor:updatesTensor:indicesTensor:batchDimensions:mode:name:` accumulates every contributor in add mode | `M-WINDOW::FOLD3D`; `Fold3dAttrs; 1 -> 1` |
| 85 | `LossKind.MEAN_SQUARED_ERROR`; `COMPOSED` | `AR::subtractionWithPrimaryTensor:secondaryTensor:name:` + `AR::multiplicationWithPrimaryTensor:secondaryTensor:name:`; reduction `NONE` is identity, `SUM` uses `RED::reductionSumWithTensor:axes:name:`, and `MEAN` additionally uses `MEM::constantWithScalar:dataType:` plus `AR::divisionWithPrimaryTensor:secondaryTensor:name:` for the exact contributor count | `M-LOSS::MEAN_SQUARED_ERROR`; `MeanSquaredErrorAttrs; 2 -> 1` |
| 86 | `LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`; `DIRECT` | `LOSS::softMaxCrossEntropyWithSourceTensor:labelsTensor:axis:reductionType:name:` | `M-LOSS::DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`; `DenseCategoricalCrossEntropyWithLogitsAttrs; 2 -> 1` |
| 87 | `LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`; `COMPOSED` | `ONEHOT::oneHotWithIndicesTensor:depth:axis:dataType:onValue:offValue:name:` + stable log-softmax using `RED::reductionMaximumWithTensor:axes:name:`, `SHAPE::expandDimsOfTensor:axes:name:`, `SHAPE::broadcastTensor:toShape:name:`, `AR::subtractionWithPrimaryTensor:secondaryTensor:name:`, `AR::exponentWithTensor:name:`, `RED::reductionSumWithTensor:axes:name:`, and `AR::logarithmWithTensor:name:`; then `AR::multiplicationWithPrimaryTensor:secondaryTensor:name:`, `RED::reductionSumWithTensor:axes:name:`, ignore masking via `AR::notEqualWithPrimaryTensor:secondaryTensor:name:` and `AR::selectWithPredicateTensor:truePredicateTensor:falsePredicateTensor:name:`, and final `RED::reductionSumWithTensor:axes:name:` or `MEM::constantWithScalar:dataType:` plus `AR::divisionWithPrimaryTensor:secondaryTensor:name:` as requested | `M-LOSS::INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`; `IndexCategoricalCrossEntropyWithLogitsAttrs; 2 -> 1` |
| 88 | `BatchNormKind.BATCH_NORM_INFERENCE`; `DIRECT` | `NORM::normalizationWithTensor:meanTensor:varianceTensor:gammaTensor:betaTensor:epsilon:name:` | `M-BN::BATCH_NORM_INFERENCE`; `BatchNormInferenceAttrs; 5 -> 1` |
| 89 | `BatchNormKind.BATCH_NORM_TRAINING`; `COMPOSED` | `NORM::meanOfTensor:axes:name:` + `NORM::varianceOfTensor:meanTensor:axes:name:` + `NORM::normalizationWithTensor:meanTensor:varianceTensor:gammaTensor:betaTensor:epsilon:name:`; saved inverse deviation and both unbiased running-statistic updates use typed `MEM::constantWithScalar:dataType:` values with `AR::additionWithPrimaryTensor:secondaryTensor:name:`, `AR::subtractionWithPrimaryTensor:secondaryTensor:name:`, `AR::multiplicationWithPrimaryTensor:secondaryTensor:name:`, `AR::divisionWithPrimaryTensor:secondaryTensor:name:`, and `AR::reciprocalSquareRootWithTensor:name:`, producing all five outputs | `M-BN::BATCH_NORM_TRAINING`; `BatchNormTrainingAttrs; 5 -> 5` |
| 90 | `LayerNormKind.LAYER_NORM`; `COMPOSED` | trailing axes `NORM::meanOfTensor:axes:name:` + `NORM::varianceOfTensor:meanTensor:axes:name:`; `SHAPE::expandDimsOfTensor:axes:name:` and `SHAPE::broadcastTensor:toShape:name:` return statistics to input shape; then `MEM::constantWithScalar:dataType:` epsilon, `AR::additionWithPrimaryTensor:secondaryTensor:name:`, `AR::reciprocalSquareRootWithTensor:name:`, `AR::subtractionWithPrimaryTensor:secondaryTensor:name:`, and `AR::multiplicationWithPrimaryTensor:secondaryTensor:name:` normalize; the 3-input form adds affine `AR` multiplication and addition through those same exact selectors | `M-LAYER::LAYER_NORM`; `LayerNormAttrs; 1 -> 1 / AffineLayerNormAttrs; 3 -> 1` |
| 91 | `RmsNormKind.RMS_NORM`; `COMPOSED` | `AR::multiplicationWithPrimaryTensor:secondaryTensor:name:` (`x*x`) + trailing-axis `NORM::meanOfTensor:axes:name:`; `SHAPE::expandDimsOfTensor:axes:name:` and `SHAPE::broadcastTensor:toShape:name:` return mean square to input shape; then `MEM::constantWithScalar:dataType:` epsilon, `AR::additionWithPrimaryTensor:secondaryTensor:name:`, `AR::reciprocalSquareRootWithTensor:name:`, and `AR::multiplicationWithPrimaryTensor:secondaryTensor:name:` by `x` normalize; the optional scale input adds another multiplication through that exact selector | `M-RMS::RMS_NORM`; `RmsNormAttrs; 1..2 -> 1` |
| 92 | `SoftmaxKind.SOFTMAX`; `DIRECT` | `ACT::softMaxWithTensor:axis:name:` | `M-SOFTMAX::SOFTMAX`; `SoftmaxAttrs; 1 -> 1` |
| 93 | `SoftmaxKind.LOG_SOFTMAX`; `COMPOSED` | stable `RED::reductionMaximumWithTensor:axes:name:`, `SHAPE::expandDimsOfTensor:axes:name:`, and `SHAPE::broadcastTensor:toShape:name:` feed `AR::subtractionWithPrimaryTensor:secondaryTensor:name:`; then `AR::exponentWithTensor:name:`, `RED::reductionSumWithTensor:axes:name:`, another expand/broadcast through the same exact shape selectors, `AR::logarithmWithTensor:name:`, and final subtraction produce the original shape | `M-SOFTMAX::LOG_SOFTMAX`; `SoftmaxAttrs; 1 -> 1` |
| 94 | `OrderingKind.SORT`; `DIRECT` | `SORT::sortWithTensor:axis:descending:name:`; stability/NaN/signed-zero parity remains a later proof obligation | `M-ORDER::SORT`; `SortAttrs; 1 -> 1` |
| 95 | `OrderingKind.ARGSORT`; `COMPOSED` | `SORT::argSortWithTensor:axis:descending:name:` returns INT32, then `SHAPE::castTensor:toType:name:` produces required INT64 | `M-ORDER::ARGSORT`; `SortAttrs; 1 -> 1` |
| 96 | `TopKKind.TOP_K`; `COMPOSED` | `TOPK::topKWithSourceTensor:axis:k:name:` (or `TOPK::bottomKWithSourceTensor:axis:k:name:` for the bounded direction), retain values, then `SHAPE::castTensor:toType:name:` converts index output from INT32 to required INT64 | `M-TOPK::TOP_K`; `TopKAttrs; 1 -> 2` |
| 97 | `Pool2dKind.MAX_POOL2D`; `DIRECT` | `POOL::maxPooling2DWithSourceTensor:descriptor:name:` with explicit padding/data layout and validated ceil mode | `M-POOL2::MAX_POOL2D`; `MaxPool2dAttrs; 1 -> 1` |
| 98 | `Pool2dKind.AVERAGE_POOL2D`; `DIRECT` | `POOL::avgPooling2DWithSourceTensor:descriptor:name:` with explicit padding, ceil mode, and `includeZeroPadToAverage` mapped from Model divisor semantics | `M-POOL2::AVERAGE_POOL2D`; `AveragePool2dAttrs; 1 -> 1` |
| 99 | `Pool3dKind.MAX_POOL3D`; `COMPOSED` | canonical NCDHW windows via repeated `SHAPE::sliceTensor:starts:ends:strides:name:`, `SHAPE::reshapeTensor:withShape:name:`, and `SHAPE::concatTensors:dimension:name:`, then `RED::reductionMaximumWithTensor:axes:name:` | `M-POOL3::MAX_POOL3D`; `MaxPool3dAttrs; 1 -> 1` |
| 100 | `Pool3dKind.AVERAGE_POOL3D`; `COMPOSED` | canonical NCDHW windows via repeated `SHAPE::sliceTensor:starts:ends:strides:name:`, `SHAPE::reshapeTensor:withShape:name:`, and `SHAPE::concatTensors:dimension:name:`, then `RED::reductionSumWithTensor:axes:name:` and exact validated divisor through `MEM::constantWithScalar:dataType:` + `AR::divisionWithPrimaryTensor:secondaryTensor:name:` | `M-POOL3::AVERAGE_POOL3D`; `AveragePool3dAttrs; 1 -> 1` |
| 101 | `DropoutKind.DROPOUT`; `UNAVAILABLE` | blocker evidence: `RANDOM::dropoutTensor:rate:name:` returns only one tensor; stateful `RANDOM::randomTensorWithShape:descriptor:stateTensor:name:` returns random values plus opaque updated state, not Model output/mask/raw-state triplet | `M-DROPOUT::DROPOUT`; `DropoutAttrs; 2 -> 3` |
| 102 | `GraphRngKind.INITIAL_STATE`; `UNAVAILABLE` | blocker evidence: `RANDOM::randomPhiloxStateTensorWithSeed:name:` and `RANDOM::randomPhiloxStateTensorWithCounterLow:counterHigh:key:name:` return opaque MPSGraph state, not Model raw INT64 `[key,counter]` | `M-RNG::INITIAL_STATE`; `GraphRngStateAttrs; 0 -> 1` |
| 103 | `RecurrentScanKind.RNN_TANH`; `UNAVAILABLE` | blocker evidence: `RNN::singleGateRNNWithSourceTensor:recurrentWeight:inputWeight:bias:initState:descriptor:name:` has no runtime valid-length tensor or Model atomic skip/positive-zero-padding contract | `M-RECURRENT::RNN_TANH`; `RecurrentDirection; 5..6 -> 2` |
| 104 | `RecurrentScanKind.GRU_RESET_AFTER`; `UNAVAILABLE` | blocker evidence: `RNN::GRUWithSourceTensor:recurrentWeight:inputWeight:bias:initState:mask:secondaryBias:descriptor:name:` has no runtime valid-length tensor or Model atomic skip/positive-zero-padding contract | `M-RECURRENT::GRU_RESET_AFTER`; `RecurrentDirection; 5..6 -> 2` |
| 105 | `RecurrentScanKind.LSTM`; `UNAVAILABLE` | blocker evidence: `RNN::LSTMWithSourceTensor:recurrentWeight:inputWeight:bias:initState:initCell:descriptor:name:` has no runtime valid-length tensor or Model atomic skip/positive-zero-padding contract | `M-RECURRENT::LSTM`; `RecurrentDirection; 6..7 -> 3` |
| 106 | `AggregateReductionKind.PROD`; `DIRECT` | `RED::reductionProductWithTensor:axes:name:` | `M-AGG::PROD`; `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs; 1 -> 1` |
| 107 | `AggregateReductionKind.ALL`; `DIRECT` | `RED::reductionAndWithTensor:axes:name:` | `M-AGG::ALL`; `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs; 1 -> 1` |
| 108 | `AggregateReductionKind.ANY`; `DIRECT` | `RED::reductionOrWithTensor:axes:name:` | `M-AGG::ANY`; `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs; 1 -> 1` |
| 109 | `AggregateReductionKind.ARG_MAX`; `COMPOSED` | `RED::reductionArgMaximumWithTensor:axis:name:` returns INT32, then `SHAPE::castTensor:toType:name:` produces required INT64 | `M-AGG::ARG_MAX`; `ArgExtremaAttrs; 1 -> 1` |
| 110 | `AggregateReductionKind.ARG_MIN`; `COMPOSED` | `RED::reductionArgMinimumWithTensor:axis:name:` returns INT32, then `SHAPE::castTensor:toType:name:` produces required INT64 | `M-AGG::ARG_MIN`; `ArgExtremaAttrs; 1 -> 1` |
| 111 | `AggregateReductionKind.LOG_SUM_EXP`; `COMPOSED` | stable `RED::reductionMaximumWithTensor:axes:name:`; `SHAPE::expandDimsOfTensor:axes:name:` and `SHAPE::broadcastTensor:toShape:name:` feed `AR::subtractionWithPrimaryTensor:secondaryTensor:name:`; then `AR::exponentWithTensor:name:`, `RED::reductionSumWithTensor:axes:name:`, `AR::logarithmWithTensor:name:`, and `AR::additionWithPrimaryTensor:secondaryTensor:name:` produce the reduced result | `M-AGG::LOG_SUM_EXP`; `MultiAxisReductionAttrs; 1 -> 1` |
| 112 | `AggregateReductionKind.VARIANCE`; `DIRECT` | `NORM::varianceOfTensor:axes:name:` | `M-AGG::VARIANCE`; `StatisticalReductionAttrs; 1 -> 1` |
| 113 | `AggregateReductionKind.STANDARD_DEVIATION`; `COMPOSED` | `NORM::varianceOfTensor:axes:name:` then `AR::squareRootWithTensor:name:` | `M-AGG::STANDARD_DEVIATION`; `StatisticalReductionAttrs; 1 -> 1` |
| 114 | `AggregateReductionKind.L1_NORM`; `COMPOSED` | `AR::absoluteWithTensor:name:` then `RED::reductionSumWithTensor:axes:name:` | `M-AGG::L1_NORM`; `MultiAxisReductionAttrs; 1 -> 1` |
| 115 | `AggregateReductionKind.L2_NORM`; `COMPOSED` | `AR::multiplicationWithPrimaryTensor:secondaryTensor:name:` (`x*x`) then `RED::reductionSumWithTensor:axes:name:` and `AR::squareRootWithTensor:name:` | `M-AGG::L2_NORM`; `MultiAxisReductionAttrs; 1 -> 1` |

## Audit invariants and implementation use

- The wire and exact-kind columns must remain byte-for-byte aligned with the primary Task 0056
  catalog. A documentation check must parse both tables and reject a missing, duplicate, reordered,
  or mismatched row.
- The MPSGraph state totals are exactly `75 DIRECT + 35 COMPOSED + 5 UNAVAILABLE = 115`.
- Every `DIRECT` row names the operation selector that structurally serves one bounded occurrence;
  every `COMPOSED` row names every primitive family needed by a finite construction; every
  `UNAVAILABLE` row names the closest current SDK selector and the mandatory Model observable it
  cannot expose.
- Header evidence is structural only. It does not assert numerical parity, deterministic winner
  behavior, dtype/rank breadth, capability, route approval, or production selection.
- Model source is authoritative over shorthand in this ledger. If a linked enum member, Javadoc, or
  `OperationSignature` changes, Task 0056 must be reviewed again before implementation.
- The implementation catalog stores closed states/reasons only; it must not embed SDK paths,
  selector strings, source links, or prose on a Runtime path.
