package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormTrainingAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaAttrs;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Cold inverse of the ordinary Metal operation/attribute transport used by Scan bodies.
 *
 * <p>This adapter restores only Model's existing typed operation descriptors. It neither proves
 * row identity nor evaluates an operation: the caller supplies already-proved operands to
 * {@code Scan.ordinarySemantics} and checks every canonical output independently. Wire shapes
 * retain their actual zero extents, and canonical sibling cardinalities are never collapsed.
 * No graph, tensor contents, executable state or numerical recipe is retained.</p>
 */
final class MetalScanOrdinaryOperation {
    private MetalScanOrdinaryOperation() {}

    /**
     * Restores the exact ordinary semantic signature selected by the raw Java Metal transport.
     *
     * <p>Attribute words are those produced by ordinary lowering, before native normalization.
     * The local descriptor table is consulted only for FoldAxis's restored output extent, which
     * ordinary transport stores in the output descriptor rather than its window words. Equal
     * extents never select an axis or infer a row identity.</p>
     *
     * @param node non-null canonical ordinary node, including all canonical output ports
     * @param localValues non-null local descriptor table indexed by the node's value ordinals;
     *     descriptors and shapes are not changed or retained
     * @return non-null existing Model operation with its original typed signature and attributes
     * @throws NullPointerException if either argument or a referenced descriptor is null
     * @throws IllegalArgumentException if the node is effectful, fixed recurrent, nested Scan,
     *     has missing or malformed attribute words, an unsupported transport kind, or cannot
     *     represent its exact Model signature
     * @throws ArithmeticException if a wire integer does not fit its Model field
     * @throws IndexOutOfBoundsException if a referenced local descriptor or axis is absent
     */
    static Operation decode(MetalMpsGraphProgram.Node node, List<TensorDescriptor> localValues) {
        Objects.requireNonNull(node, "node");
        Objects.requireNonNull(localValues, "localValues");
        long[] w = node.attributeWords();
        int inputs = node.inputs().length;
        int[] outputs = node.outputs();
        Operation operation = switch (node.kind()) {
            case NEG, ABS, RECIPROCAL, LOG, LOG1P, EXP, EXPM1, ERF, SQRT, RSQRT,
                    FLOOR, CEIL, SIGN, RELU, SIGMOID, TANH, GELU, GELU_TANH_APPROXIMATION, SILU -> plain(UnaryElementwiseKind.valueOf(node.kind().name()));
            case ADD, SUB, MUL, DIV -> plain(BinaryArithmeticKind.valueOf(node.kind().name()));
            case TENSOR_MIN -> plain(BinaryArithmeticKind.MIN);
            case TENSOR_MAX -> plain(BinaryArithmeticKind.MAX);
            case TENSOR_POW -> plain(BinaryArithmeticKind.POW);
            case GT -> plain(BinaryComparisonKind.GREATER_THAN);
            case GE -> plain(BinaryComparisonKind.GREATER_OR_EQUAL);
            case LT -> plain(BinaryComparisonKind.LESS_THAN);
            case LE -> plain(BinaryComparisonKind.LESS_OR_EQUAL);
            case EQ -> plain(BinaryComparisonKind.EQUAL);
            case NE -> plain(BinaryComparisonKind.NOT_EQUAL);
            case IS_FINITE, IS_NAN, IS_INF -> plain(FloatingClassificationKind.valueOf(node.kind().name()));
            case LOGICAL_AND -> plain(BooleanLogicalKind.AND);
            case LOGICAL_OR -> plain(BooleanLogicalKind.OR);
            case LOGICAL_NOT -> plain(BooleanLogicalKind.NOT);
            case WHERE -> plain(WhereSelectionKind.WHERE);
            case CAST -> new Operation(CastKind.CAST, new CastAttrs(type(word(w, 0))));
            case SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV, SCALAR_POW, SCALAR_MIN, SCALAR_MAX -> new Operation(
                    ScalarElementwiseKind.valueOf(node.kind().name().substring("SCALAR_".length())),
                    new ScalarValueAttrs(scalar(w, 0)));
            case CLAMP -> new Operation(ScalarElementwiseKind.CLAMP,
                    new ClampRangeAttrs(scalar(w, 0), scalar(w, 2)));
            case MATMUL -> plain(MatmulKind.MATMUL);
            case CONTIGUOUS -> plain(ContiguousKind.CONTIGUOUS);
            case RESHAPE, EXPAND -> new Operation(ShapeTransformKind.valueOf(node.kind().name()),
                    new TargetShapeAttrs(shape(w, 1, integer(word(w, 0)))));
            case PERMUTE -> new Operation(AxisTransformKind.PERMUTE,
                    new PermutationAttrs(integers(w, 1, integer(word(w, 0)))));
            case EXPAND_DIMS, SQUEEZE -> new Operation(AxisTransformKind.valueOf(node.kind().name()),
                    new AxisTransformAttrs(integer(word(w, 0))));
            case CONCAT, STACK -> new Operation(TensorCompositionKind.valueOf(node.kind().name()),
                    new CompositionAxisAttrs(integer(word(w, 0))));
            case SLICE, SLICE_UPDATE -> new Operation(SliceKind.valueOf(node.kind().name()), slice(node.attributeKind(), w));
            case SELECT -> new Operation(SelectKind.SELECT, new SelectAttrs(integer(word(w, 0)), word(w, 1)));
            case PAD -> new Operation(PadKind.PAD, pad(w));
            case TILE -> new Operation(TileKind.TILE, new TileAttrs(longs(w, 1, integer(word(w, 0)))));
            case GATHER, GATHER_ELEMENTS -> new Operation(AxisGatherKind.valueOf(node.kind().name()),
                    new IndexAxisAttrs(integer(word(w, 0))));
            case GATHER_ND -> new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(integer(word(w, 0))));
            case ONE_HOT -> new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(word(w, 0)));
            case SCATTER_ADD -> new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(integer(word(w, 0))));
            case SCATTER_ELEMENTS -> new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                    new ScatterElementsAttrs(integer(word(w, 0)), scatterReduction(word(w, 1))));
            case SCATTER_ND -> new Operation(ScatterNdKind.SCATTER_ND,
                    new ScatterNdAttrs(integer(word(w, 0)), scatterReduction(word(w, 1))));
            case CUM_SUM, CUM_PROD -> new Operation(CumulativeScanKind.valueOf(node.kind().name()),
                    new CumulativeScanAttrs(integer(word(w, 0)), bool(word(w, 1)), bool(word(w, 2))));
            case SUM, MEAN, PROD, ALL, ANY, LOG_SUM_EXP, VARIANCE, STANDARD_DEVIATION, L1_NORM, L2_NORM -> new Operation(
                    AggregateReductionKind.valueOf(node.kind().name()), reduction(node.attributeKind(), w, localValues.get(node.firstInputIndex()).shape().rank()));
            case REDUCTION_MIN -> new Operation(AggregateReductionKind.MIN, reduction(node.attributeKind(), w, localValues.get(node.firstInputIndex()).shape().rank()));
            case REDUCTION_MAX -> new Operation(AggregateReductionKind.MAX, reduction(node.attributeKind(), w, localValues.get(node.firstInputIndex()).shape().rank()));
            case ARG_MAX, ARG_MIN -> new Operation(AggregateReductionKind.valueOf(node.kind().name()),
                    new ArgExtremaAttrs(axis(word(w, 0), localValues.get(node.firstInputIndex()).shape().rank()), bool(word(w, 1)), tiePolicy(word(w, 2))));
            case SOFTMAX, LOG_SOFTMAX -> new Operation(SoftmaxKind.valueOf(node.kind().name()),
                    new SoftmaxAttrs(integer(word(w, 0))));
            case LAYER_NORM -> new Operation(LayerNormKind.LAYER_NORM, inputs == 1
                    ? new LayerNormAttrs(shape(w, 1, integer(word(w, 0))), scalar(w, integer(word(w, 0)) + 1))
                    : new AffineLayerNormAttrs(shape(w, 1, integer(word(w, 0))), scalar(w, integer(word(w, 0)) + 1)));
            case RMS_NORM -> new Operation(RmsNormKind.RMS_NORM,
                    new RmsNormAttrs(shape(w, 1, integer(word(w, 0))), scalar(w, integer(word(w, 0)) + 1)));
            case BATCH_NORM_INFERENCE -> new Operation(BatchNormKind.BATCH_NORM_INFERENCE,
                    new BatchNormInferenceAttrs(integer(word(w, 0)), scalar(w, 1)));
            case BATCH_NORM_TRAINING -> new Operation(BatchNormKind.BATCH_NORM_TRAINING,
                    new BatchNormTrainingAttrs(integer(word(w, 0)), scalar(w, 1), scalar(w, 3)));
            case CONV2D -> new Operation(Conv2dKind.CONV2D,
                    new Conv2dAttrs(word(w, 0), word(w, 1), word(w, 2), word(w, 3), word(w, 4), word(w, 5), word(w, 6)));
            case CONV3D -> new Operation(Conv3dKind.CONV3D,
                    new Conv3dAttrs(word(w, 0), word(w, 1), word(w, 2), word(w, 3), word(w, 4), word(w, 5), word(w, 6), word(w, 7), word(w, 8), word(w, 9)));
            case UNFOLD_AXIS -> new Operation(WindowTransformKind.UNFOLD_AXIS,
                    new UnfoldAxisAttrs(integer(word(w, 0)), word(w, 1), word(w, 2)));
            case FOLD_AXIS -> {
                Shape outputShape = localValues.get(outputs[0]).shape();
                int foldAxis = axis(word(w, 0), outputShape.rank());
                yield new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(foldAxis, outputShape.toLongArray()[foldAxis], word(w, 2)));
            }
            case UNFOLD2D -> new Operation(WindowTransformKind.UNFOLD2D,
                    node.attributeKind() == MetalMpsGraphProgram.AttributeKind.PADDED_WINDOW_2D
                            ? new Unfold2dAttrs(window2d(w, 0), scalar(w, 9)) : window2d(w, 0));
            case UNFOLD3D -> new Operation(WindowTransformKind.UNFOLD3D,
                    node.attributeKind() == MetalMpsGraphProgram.AttributeKind.PADDED_WINDOW_3D
                            ? new Unfold3dAttrs(window3d(w, 0), scalar(w, 13)) : window3d(w, 0));
            case FOLD2D -> new Operation(WindowTransformKind.FOLD2D,
                    new Fold2dAttrs(shape(w, 1, integer(word(w, 0))), window2d(w, integer(word(w, 0)) + 1)));
            case FOLD3D -> new Operation(WindowTransformKind.FOLD3D,
                    new Fold3dAttrs(shape(w, 1, integer(word(w, 0))), window3d(w, integer(word(w, 0)) + 1)));
            case MAX_POOL2D -> new Operation(Pool2dKind.MAX_POOL2D,
                    new MaxPool2dAttrs(word(w, 0), word(w, 1), word(w, 2), word(w, 3), word(w, 4), word(w, 5), word(w, 6), word(w, 7), bool(word(w, 8))));
            case AVERAGE_POOL2D -> new Operation(Pool2dKind.AVERAGE_POOL2D,
                    new AveragePool2dAttrs(word(w, 0), word(w, 1), word(w, 2), word(w, 3), word(w, 4), word(w, 5), word(w, 6), word(w, 7), bool(word(w, 8))));
            case MAX_POOL3D -> new Operation(Pool3dKind.MAX_POOL3D,
                    new MaxPool3dAttrs(word(w, 0), word(w, 1), word(w, 2), word(w, 3), word(w, 4), word(w, 5), word(w, 6), word(w, 7), word(w, 8), word(w, 9), word(w, 10), word(w, 11), bool(word(w, 12))));
            case AVERAGE_POOL3D -> new Operation(Pool3dKind.AVERAGE_POOL3D,
                    new AveragePool3dAttrs(word(w, 0), word(w, 1), word(w, 2), word(w, 3), word(w, 4), word(w, 5), word(w, 6), word(w, 7), word(w, 8), word(w, 9), word(w, 10), word(w, 11), bool(word(w, 12))));
            case MEAN_SQUARED_ERROR -> new Operation(LossKind.MEAN_SQUARED_ERROR,
                    new MeanSquaredErrorAttrs(lossReduction(word(w, 0))));
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> new Operation(
                    LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                    new DenseCategoricalCrossEntropyWithLogitsAttrs(integer(word(w, 0)), lossReduction(word(w, 1))));
            case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> new Operation(
                    LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                    new IndexCategoricalCrossEntropyWithLogitsAttrs(integer(word(w, 0)), lossReduction(word(w, 1)),
                            bool(word(w, 2)) ? Optional.of(scalar(w, 3)) : Optional.empty()));
            case SCALED_DOT_PRODUCT_ATTENTION -> new Operation(
                    ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                    new ScaledDotProductAttentionAttrs(bool(word(w, 0)) ? Optional.of(scalar(w, 1))
                            : Optional.empty(), bool(word(w, bool(word(w, 0)) ? 3 : 1))));
            case SORT, ARGSORT -> new Operation(OrderingKind.valueOf(node.kind().name()),
                    new SortAttrs(axis(word(w, 0), localValues.get(node.firstInputIndex()).shape().rank()), bool(word(w, 1))));
            case TOP_K -> new Operation(TopKKind.TOP_K,
                    new TopKAttrs(axis(word(w, 0), localValues.get(node.firstInputIndex()).shape().rank()), word(w, 1), bool(word(w, 2)), bool(word(w, 3))));
            default -> throw malformed();
        };
        operation.signature().validateOccurrence(inputs, outputs.length);
        return operation;
    }

    /**
     * @param kind parameterless Model identity
     * @return its existing no-attributes signature
     */
    private static Operation plain(OperationKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }

    /**
     * Restores the attribute class, not merely an equivalent set of reduced axes.
     * @param kind raw reduction attribute grammar
     * @param w validated ordinary words
     * @param rank source rank bounding every transported reduction axis
     * @return exact full, axis, multi-axis, masked, statistical or sum-to-shape attributes
     * @throws IllegalArgumentException if the grammar or form is not an ordinary reduction
     */
    private static OperationAttrs reduction(MetalMpsGraphProgram.AttributeKind kind, long[] w, int rank) {
        return switch (kind) {
            case MASKED_REDUCTION -> new MaskedReductionAttrs(integer(word(w, 0)));
            case STATISTICAL_REDUCTION -> {
                int count = integer(word(w, 0));
                yield new StatisticalReductionAttrs(axes(w, 1, count, rank), bool(word(w, count + 1)), word(w, count + 2));
            }
            case REDUCTION -> switch (integer(word(w, 0))) {
                case 1 -> NoOperationAttrs.INSTANCE;
                case 2 -> new AxisReductionAttrs(axis(word(w, 3), rank), bool(word(w, 1)));
                case 3 -> new MultiAxisReductionAttrs(axes(w, 3, integer(word(w, 2)), rank), bool(word(w, 1)));
                case 4 -> new SumToShapeAttrs(shape(w, 3, integer(word(w, 2))));
                default -> throw malformed();
            };
            default -> throw malformed();
        };
    }

    /**
     * @param kind slice or crop grammar selected by lowering
     * @param w validated ordinary words with paired list counts
     * @return exact slice or crop attributes, preserving signed steps and zero lengths
     * @throws IllegalArgumentException if the grammar is neither slice nor crop
     */
    private static OperationAttrs slice(MetalMpsGraphProgram.AttributeKind kind, long[] w) {
        int count = integer(word(w, 0));
        return switch (kind) {
            case CROP_TO_SHAPE -> new CropToShapeAttrs(shape(w, 1, count),
                    shape(w, count + 2, integer(word(w, count + 1))));
            case SLICE -> new SliceAttrs(longs(w, 1, count), longs(w, 1 + count, count),
                    integers(w, 1 + count * 2, count), longs(w, 1 + count * 3, count));
            default -> throw malformed();
        };
    }

    /**
     * @param w validated rank-prefixed padding words
     * @return exact padding and scalar metadata
     */
    private static PadAttrs pad(long[] w) {
        int rank = integer(word(w, 0));
        return new PadAttrs(longs(w, 1, rank), longs(w, 1 + rank, rank), scalar(w, 1 + rank * 2));
    }

    /**
     * @param w validated window words
     * @param offset first window field
     * @return exact 2D window
     */
    private static Window2dAttrs window2d(long[] w, int offset) {
        return new Window2dAttrs(word(w, offset), word(w, offset + 1), word(w, offset + 2), word(w, offset + 3),
                word(w, offset + 4), word(w, offset + 5), word(w, offset + 6), word(w, offset + 7), bool(word(w, offset + 8)));
    }

    /**
     * @param w validated window words
     * @param offset first window field
     * @return exact 3D window
     */
    private static Window3dAttrs window3d(long[] w, int offset) {
        return new Window3dAttrs(word(w, offset), word(w, offset + 1), word(w, offset + 2), word(w, offset + 3),
                word(w, offset + 4), word(w, offset + 5), word(w, offset + 6), word(w, offset + 7), word(w, offset + 8),
                word(w, offset + 9), word(w, offset + 10), word(w, offset + 11), bool(word(w, offset + 12)));
    }

    /**
     * Uses Model scalar factories with checked raw storage widths; no numeric type coercion occurs.
     * @param w ordinary scalar words
     * @param offset type wire followed by raw bits
     * @return exact typed scalar, including low raw bits, signed zero and NaN payload
     * @throws IllegalArgumentException if a bit pattern exceeds its declared storage width
     */
    private static ScalarValue scalar(long[] w, int offset) {
        long bits = word(w, offset + 1);
        return switch (type(word(w, offset))) {
            case FLOAT64 -> ScalarValue.float64(Double.longBitsToDouble(bits));
            case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(word32(bits)));
            case BFLOAT16 -> ScalarValue.bfloat16Bits(word16(bits));
            case FLOAT16 -> ScalarValue.float16Bits(word16(bits));
            case INT32 -> ScalarValue.int32(word32(bits));
            case INT64 -> ScalarValue.int64(bits);
            case BOOL -> ScalarValue.bool(bool(bits));
        };
    }

    /**
     * @param wire existing Metal data-type identity
     * @return corresponding Model type, never an ordinal-based coercion
     * @throws IllegalArgumentException if the wire identity is unknown
     */
    private static DataType type(long wire) {
        return switch (integer(wire)) {
            case 1 -> DataType.FLOAT32;
            case 2 -> DataType.INT32;
            case 3 -> DataType.BOOL;
            case 4 -> DataType.FLOAT64;
            case 5 -> DataType.BFLOAT16;
            case 6 -> DataType.INT64;
            case 7 -> DataType.FLOAT16;
            default -> throw malformed();
        };
    }

    /**
     * @param wire loss reduction wire
     * @return existing Model reduction
     * @throws IllegalArgumentException for an unknown wire
     */
    private static LossReduction lossReduction(long wire) {
        return switch (integer(wire)) {
            case 1 -> LossReduction.NONE;
            case 2 -> LossReduction.SUM;
            case 3 -> LossReduction.MEAN;
            default -> throw malformed();
        };
    }

    /**
     * @param wire scatter reduction wire
     * @return existing Model reduction
     * @throws IllegalArgumentException for an unknown wire
     */
    private static ScatterReduction scatterReduction(long wire) {
        return switch (integer(wire)) {
            case 1 -> ScatterReduction.NONE;
            case 2 -> ScatterReduction.ADD;
            case 3 -> ScatterReduction.MUL;
            case 4 -> ScatterReduction.MAX;
            case 5 -> ScatterReduction.MIN;
            default -> throw malformed();
        };
    }

    /**
     * @param wire extrema tie wire
     * @return exact Model tie policy
     * @throws IllegalArgumentException for an unknown wire
     */
    private static ArgExtremaTiePolicy tiePolicy(long wire) {
        return switch (integer(wire)) {
            case 1 -> ArgExtremaTiePolicy.FIRST_INDEX;
            case 2 -> ArgExtremaTiePolicy.LAST_INDEX;
            default -> throw malformed();
        };
    }

    /**
     * @param word integer wire field
     * @return exact signed Java integer
     * @throws ArithmeticException on overflow
     */
    private static int integer(long word) {
        return Math.toIntExact(word);
    }

    /**
     * @param bits unsigned 32-bit storage word
     * @return unchanged signed bit carrier
     * @throws IllegalArgumentException on excess bits
     */
    private static int word32(long bits) {
        if ((bits & ~0xffff_ffffL) != 0) throw malformed();
        return (int) bits;
    }

    /**
     * @param bits unsigned 16-bit storage word
     * @return unchanged signed bit carrier
     * @throws IllegalArgumentException on excess bits
     */
    private static short word16(long bits) {
        if ((bits & ~0xffffL) != 0) throw malformed();
        return (short) bits;
    }

    /**
     * @param word boolean wire field
     * @return exact boolean
     * @throws IllegalArgumentException unless zero or one
     */
    private static boolean bool(long word) {
        if (word != 0 && word != 1) throw malformed();
        return word == 1;
    }

    /**
     * @param w ordinary words
     * @param offset first extent
     * @param count rank, including zero
     * @return existing Model shape preserving all actual zero extents
     * @throws IllegalArgumentException if the extent span is absent
     */
    private static Shape shape(long[] w, int offset, int count) {
        requireSpan(w, offset, count);
        return Shape.of(Arrays.copyOfRange(w, offset, offset + count));
    }

    /**
     * @param w ordinary words
     * @param offset first field
     * @param count number of fields
     * @return ordered integer fields without axis sorting or deduplication
     * @throws ArithmeticException if a field exceeds the Model integer width
     * @throws IllegalArgumentException if the field span is absent
     */
    private static List<Integer> integers(long[] w, int offset, int count) {
        requireSpan(w, offset, count);
        List<Integer> result = new ArrayList<>(count);
        for (int i = 0; i < count; i++) result.add(integer(word(w, offset + i)));
        return result;
    }

    /**
     * @param w ordinary words
     * @param offset first field
     * @param count number of fields
     * @return ordered exact long fields, including signed slice steps
     * @throws IllegalArgumentException if the field span is absent
     */
    private static List<Long> longs(long[] w, int offset, int count) {
        requireSpan(w, offset, count);
        List<Long> result = new ArrayList<>(count);
        for (int i = 0; i < count; i++) result.add(word(w, offset + i));
        return result;
    }

    /**
     * @param word canonical nonnegative axis wire
     * @param rank rank of the source descriptor
     * @return bounded Model axis
     * @throws IllegalArgumentException if the wire does not name a source axis
     */
    private static int axis(long word, int rank) {
        if (word < 0 || word >= rank) throw malformed();
        return (int) word;
    }

    /**
     * @param w ordinary words
     * @param offset first axis word
     * @param count number of axes
     * @param rank source rank bounding each axis
     * @return ordered axes without sorting or deduplication
     * @throws IllegalArgumentException if the span or any axis is absent
     */
    private static List<Integer> axes(long[] w, int offset, int count, int rank) {
        requireSpan(w, offset, count);
        List<Integer> result = new ArrayList<>(count);
        for (int i = 0; i < count; i++) result.add(axis(w[offset + i], rank));
        return result;
    }

    /**
     * Reads one transport word only after checking its presence.
     * @param w ordinary attribute words
     * @param offset requested word position
     * @return the unchanged transport word
     * @throws IllegalArgumentException if the word is absent
     */
    private static long word(long[] w, int offset) {
        requireSpan(w, offset, 1);
        return w[offset];
    }

    /**
     * Checks a transport span without overflowing its endpoint or allocating from an untrusted count.
     * @param w ordinary attribute words
     * @param offset first requested word
     * @param count number of requested words
     * @throws IllegalArgumentException if any requested word is absent or the count is negative
     */
    private static void requireSpan(long[] w, int offset, int count) {
        if (offset < 0 || count < 0 || offset > w.length || count > w.length - offset) {
            throw malformed();
        }
    }

    /** @return rejection of a transport that cannot restore an ordinary PURE Model operation */
    private static IllegalArgumentException malformed() {
        return new IllegalArgumentException("Metal Scan requires an exact ordinary PURE Model operation");
    }
}
