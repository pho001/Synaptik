package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
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
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaAttrs;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Exhaustive current-wire audit; each row is a constructible Model occurrence, not count arithmetic. */
class MetalOperationCompletenessAuditTest {
    private final MetalCapabilityProvider provider = new MetalCapabilityProvider();

    @Test
    void everyOrdinaryWireHasADistinctSupportedModelRepresentative() {
        Set<String> modelKinds = new HashSet<>();
        for (var kind : MetalMpsGraphProgram.NodeKind.values()) {
            // Scan carries a body and activity protocol, exercised by MetalScanPlanTest.
            if (kind == MetalMpsGraphProgram.NodeKind.GENERAL_SCAN) continue;
            var occurrence = representative(kind);
            assertTrue(modelKinds.add(modelKind(occurrence.operation())), kind.name());
            assertTrue(provider.supports(occurrence), kind + " representative");
            assertTrue(kind.executable(), kind.name());
        }
    }

    private static String modelKind(Operation operation) {
        Enum<?> kind = (Enum<?>) operation.kind();
        return kind.getClass().getSimpleName() + "." + kind.name();
    }

    private static OperationCapabilityQuery representative(MetalMpsGraphProgram.NodeKind kind) {
        return switch (kind) {
            case NEG, ABS, RECIPROCAL, LOG, LOG1P, EXP, EXPM1, ERF, SQRT, RSQRT, FLOOR, CEIL,
                    SIGN, RELU, SIGMOID, TANH, GELU, GELU_TANH_APPROXIMATION, SILU -> unary(kind);
            case ADD, SUB, MUL, DIV, TENSOR_MIN, TENSOR_MAX, TENSOR_POW -> binary(kind);
            case SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV, SCALAR_POW, SCALAR_MIN,
                    SCALAR_MAX, CLAMP -> scalar(kind);
            case SUM, MEAN, REDUCTION_MIN, REDUCTION_MAX, PROD, ALL, ANY, ARG_MAX, ARG_MIN,
                    LOG_SUM_EXP, VARIANCE, STANDARD_DEVIATION, L1_NORM, L2_NORM -> reduction(kind);
            case GT, GE, LT, LE, EQ, NE -> comparison(kind);
            case IS_FINITE, IS_NAN, IS_INF -> classification(kind);
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> logical(kind);
            case RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS -> affine(kind);
            case GATHER, ONE_HOT, SCATTER_ELEMENTS, GATHER_ELEMENTS, SCATTER_ADD, GATHER_ND,
                    SCATTER_ND -> indexing(kind);
            case UNFOLD_AXIS, FOLD_AXIS, UNFOLD2D, FOLD2D, UNFOLD3D, FOLD3D -> window(kind);
            case SELECT, PAD, SLICE, SLICE_UPDATE, CONCAT, STACK, TILE -> movement(kind);
            case MATMUL -> query(new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                    List.of(f32(2, 3), f32(3, 4)), List.of(f32(2, 4)));
            case WHERE -> query(new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                    List.of(bool(3), f32(2, 1), f32(1, 3)), List.of(f32(2, 3)));
            case CAST -> query(new Operation(CastKind.CAST, new CastAttrs(DataType.INT64)),
                    List.of(f32(2, 3)), List.of(descriptor(DataType.INT64, Shape.of(2, 3), false)));
            case CUM_SUM, CUM_PROD -> query(new Operation(kind == MetalMpsGraphProgram.NodeKind.CUM_SUM
                            ? CumulativeScanKind.CUM_SUM : CumulativeScanKind.CUM_PROD,
                            new CumulativeScanAttrs(1, false, false)),
                    List.of(f32(2, 3)), List.of(f32(2, 3)));
            case SCALED_DOT_PRODUCT_ATTENTION -> attention();
            case CONV2D, CONV3D -> convolution(kind);
            case MEAN_SQUARED_ERROR, DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                    INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> loss(kind);
            case BATCH_NORM_INFERENCE, BATCH_NORM_TRAINING, LAYER_NORM, RMS_NORM, SOFTMAX,
                    LOG_SOFTMAX -> normalization(kind);
            case SORT, ARGSORT, TOP_K -> ordering(kind);
            case MAX_POOL2D, AVERAGE_POOL2D, MAX_POOL3D, AVERAGE_POOL3D -> pooling(kind);
            case DROPOUT, INITIAL_STATE -> random(kind);
            case RNN_TANH, GRU_RESET_AFTER, LSTM -> recurrent(kind);
            case GENERAL_SCAN -> throw new IllegalArgumentException("Scan requires a body/activity fixture");
        };
    }

    private static OperationCapabilityQuery unary(MetalMpsGraphProgram.NodeKind node) {
        UnaryElementwiseKind kind = UnaryElementwiseKind.valueOf(node.name());
        return query(new Operation(kind, NoOperationAttrs.INSTANCE), List.of(f32(2, 3)), List.of(f32(2, 3)));
    }

    private static OperationCapabilityQuery binary(MetalMpsGraphProgram.NodeKind node) {
        BinaryArithmeticKind kind = switch (node) {
            case TENSOR_MIN -> BinaryArithmeticKind.MIN;
            case TENSOR_MAX -> BinaryArithmeticKind.MAX;
            case TENSOR_POW -> BinaryArithmeticKind.POW;
            default -> BinaryArithmeticKind.valueOf(node.name());
        };
        return query(new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(f32(2, 1, 3), f32(1, 4, 3)), List.of(f32(2, 4, 3)));
    }

    private static OperationCapabilityQuery scalar(MetalMpsGraphProgram.NodeKind node) {
        ScalarElementwiseKind kind = node == MetalMpsGraphProgram.NodeKind.CLAMP
                ? ScalarElementwiseKind.CLAMP
                : ScalarElementwiseKind.valueOf(node.name().replace("SCALAR_", ""));
        Object attrs = kind == ScalarElementwiseKind.CLAMP
                ? new ClampRangeAttrs(ScalarValue.float32(-1.0f), ScalarValue.float32(1.0f))
                : new ScalarValueAttrs(ScalarValue.float32(2.0f));
        return query(new Operation(kind,
                (io.github.pho001.synaptik.model.operation.OperationAttrs) attrs),
                List.of(f32(2, 3)), List.of(f32(2, 3)));
    }

    private static OperationCapabilityQuery reduction(MetalMpsGraphProgram.NodeKind node) {
        AggregateReductionKind kind = switch (node) {
            case REDUCTION_MIN -> AggregateReductionKind.MIN;
            case REDUCTION_MAX -> AggregateReductionKind.MAX;
            default -> AggregateReductionKind.valueOf(node.name());
        };
        Object attrs = switch (kind) {
            case ARG_MAX, ARG_MIN -> new ArgExtremaAttrs(
                    1, false, ArgExtremaTiePolicy.FIRST_INDEX);
            case L1_NORM -> new MultiAxisReductionAttrs(List.of(0), false);
            case VARIANCE -> new StatisticalReductionAttrs(List.of(0), false, 0);
            case LOG_SUM_EXP, L2_NORM -> new MultiAxisReductionAttrs(List.of(1), false);
            case STANDARD_DEVIATION ->
                    new StatisticalReductionAttrs(List.of(1), false, 0);
            default -> NoOperationAttrs.INSTANCE;
        };
        TensorDescriptor input;
        TensorDescriptor output;
        if (kind == AggregateReductionKind.ALL || kind == AggregateReductionKind.ANY) {
            input = bool(2, 3);
            output = boolScalar();
        } else if (kind == AggregateReductionKind.PROD) {
            input = descriptor(DataType.INT32, Shape.of(2, 3), false);
            output = descriptor(DataType.INT32, Shape.scalar(), false);
        } else if (kind == AggregateReductionKind.ARG_MAX
                || kind == AggregateReductionKind.ARG_MIN) {
            input = f32(2, 3);
            output = descriptor(DataType.INT64, Shape.of(2), false);
        } else if (kind == AggregateReductionKind.L1_NORM) {
            input = f32(3);
            output = f32Scalar();
        } else if (kind == AggregateReductionKind.VARIANCE) {
            input = f32(1);
            output = f32Scalar();
        } else if (attrs instanceof MultiAxisReductionAttrs
                || attrs instanceof StatisticalReductionAttrs) {
            input = f32(2, 3);
            output = f32(2);
        } else {
            input = f32(2, 3);
            output = f32Scalar();
        }
        boolean accelerator = kind == AggregateReductionKind.SUM
                || kind == AggregateReductionKind.MEAN
                || kind == AggregateReductionKind.MIN
                || kind == AggregateReductionKind.MAX
                || kind == AggregateReductionKind.LOG_SUM_EXP
                || kind == AggregateReductionKind.VARIANCE
                || kind == AggregateReductionKind.STANDARD_DEVIATION
                || kind == AggregateReductionKind.L1_NORM
                || kind == AggregateReductionKind.L2_NORM;
        return query(new Operation(kind, (io.github.pho001.synaptik.model.operation.OperationAttrs) attrs),
                List.of(input), List.of(output));
    }

    private static OperationCapabilityQuery comparison(MetalMpsGraphProgram.NodeKind node) {
        BinaryComparisonKind kind = switch (node) {
            case GT -> BinaryComparisonKind.GREATER_THAN;
            case GE -> BinaryComparisonKind.GREATER_OR_EQUAL;
            case LT -> BinaryComparisonKind.LESS_THAN;
            case LE -> BinaryComparisonKind.LESS_OR_EQUAL;
            case EQ -> BinaryComparisonKind.EQUAL;
            case NE -> BinaryComparisonKind.NOT_EQUAL;
            default -> throw new AssertionError(node);
        };
        return query(new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(f32(2, 1, 3), f32(1, 4, 3)), List.of(bool(2, 4, 3)));
    }

    private static OperationCapabilityQuery classification(MetalMpsGraphProgram.NodeKind node) {
        FloatingClassificationKind kind = FloatingClassificationKind.valueOf(node.name());
        return query(new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(f32(2, 3)), List.of(bool(2, 3)));
    }

    private static OperationCapabilityQuery logical(MetalMpsGraphProgram.NodeKind node) {
        BooleanLogicalKind kind = BooleanLogicalKind.valueOf(node.name().replace("LOGICAL_", ""));
        List<TensorDescriptor> inputs = kind == BooleanLogicalKind.NOT
                ? List.of(bool(2, 3)) : List.of(bool(2, 1), bool(1, 3));
        return query(new Operation(kind, NoOperationAttrs.INSTANCE),
                inputs, List.of(bool(2, 3)));
    }

    private static OperationCapabilityQuery affine(MetalMpsGraphProgram.NodeKind node) {
        return switch (node) {
            case RESHAPE -> query(new Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(Shape.of(2, 3))),
                    List.of(f32(6)), List.of(view(DataType.FLOAT32, Shape.of(2, 3), 0, 3, 1)));
            case EXPAND -> query(new Operation(ShapeTransformKind.EXPAND, new TargetShapeAttrs(Shape.of(2, 3))),
                    List.of(f32(1, 3)), List.of(view(DataType.FLOAT32, Shape.of(2, 3), 0, 0, 1)));
            case PERMUTE -> query(new Operation(AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1, 0))),
                    List.of(f32(2, 3)), List.of(view(DataType.FLOAT32, Shape.of(3, 2), 0, 1, 3)));
            case EXPAND_DIMS -> query(new Operation(AxisTransformKind.EXPAND_DIMS, new AxisTransformAttrs(1)),
                    List.of(f32(2, 3)), List.of(view(DataType.FLOAT32, Shape.of(2, 1, 3), 0, 3, 3, 1)));
            case SQUEEZE -> query(new Operation(AxisTransformKind.SQUEEZE, new AxisTransformAttrs(1)),
                    List.of(f32(2, 1, 3)), List.of(view(DataType.FLOAT32, Shape.of(2, 3), 0, 3, 1)));
            case CONTIGUOUS -> query(new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE),
                    List.of(view(DataType.FLOAT32, Shape.of(2, 3), 0, 0, 1)), List.of(f32(2, 3)));
            default -> throw new AssertionError(node);
        };
    }

    private static OperationCapabilityQuery indexing(MetalMpsGraphProgram.NodeKind node) {
        return switch (node) {
            case GATHER -> query(new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                    List.of(f32(2, 3, 4), i32(5, 6)), List.of(f32(2, 5, 6, 4)));
            case ONE_HOT -> query(new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(4)),
                    List.of(i32(2, 3)), List.of(bool(2, 3, 4)));
            case SCATTER_ELEMENTS -> query(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                            new ScatterElementsAttrs(1, ScatterReduction.NONE)),
                    List.of(f32(2, 3), i32(2, 2), f32(2, 2)), List.of(f32(2, 3)));
            case GATHER_ELEMENTS -> query(new Operation(AxisGatherKind.GATHER_ELEMENTS, new IndexAxisAttrs(1)),
                    List.of(f32(2, 3), i32(2, 2)), List.of(f32(2, 2)));
            case SCATTER_ADD -> query(new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(0)),
                    List.of(f32(3), i32(2), f32(2)), List.of(f32(3)));
            case GATHER_ND -> query(new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(0)),
                    List.of(f32(2, 3), i32(2, 1)), List.of(f32(2, 3)));
            case SCATTER_ND -> query(new Operation(ScatterNdKind.SCATTER_ND,
                            new ScatterNdAttrs(0, ScatterReduction.NONE)),
                    List.of(f32(3), i32(2, 1), f32(2)), List.of(f32(3)));
            default -> throw new AssertionError(node);
        };
    }

    private static OperationCapabilityQuery window(MetalMpsGraphProgram.NodeKind node) {
        Window2dAttrs unfold2d =
                new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false);
        Window3dAttrs unfold3d =
                new Window3dAttrs(2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        Window2dAttrs fold2d = new Window2dAttrs(2, 2, 2, 2, 1, 1, 1, 1, true);
        Window3dAttrs fold3d =
                new Window3dAttrs(2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, true);
        return switch (node) {
            case UNFOLD_AXIS -> query(new Operation(WindowTransformKind.UNFOLD_AXIS, new UnfoldAxisAttrs(1, 3, 2)),
                    List.of(f32(2, 6)), List.of(f32(2, 2, 3)));
            case FOLD_AXIS -> query(new Operation(WindowTransformKind.FOLD_AXIS, new FoldAxisAttrs(1, 4, 2)),
                    List.of(f32(2, 2, 2)), List.of(f32(2, 4)));
            case UNFOLD2D -> query(new Operation(WindowTransformKind.UNFOLD2D, unfold2d),
                    List.of(f32(1, 1, 2, 2)), List.of(f32(1, 4, 1)));
            case FOLD2D -> query(new Operation(WindowTransformKind.FOLD2D,
                            new Fold2dAttrs(Shape.of(1, 1, 3, 3), fold2d)),
                    List.of(f32(1, 4, 9)), List.of(f32(1, 1, 3, 3)));
            case UNFOLD3D -> query(new Operation(WindowTransformKind.UNFOLD3D, unfold3d),
                    List.of(f32(1, 1, 2, 2, 2)), List.of(f32(1, 8, 1)));
            case FOLD3D -> query(new Operation(WindowTransformKind.FOLD3D,
                            new Fold3dAttrs(Shape.of(1, 1, 3, 3, 3), fold3d)),
                    List.of(f32(1, 8, 27)), List.of(f32(1, 1, 3, 3, 3)));
            default -> throw new AssertionError(node);
        };
    }

    private static OperationCapabilityQuery movement(MetalMpsGraphProgram.NodeKind node) {
        return switch (node) {
            case SELECT -> query(new Operation(SelectKind.SELECT, new SelectAttrs(0, 1)),
                    List.of(i32(2, 3)),
                    List.of(view(DataType.INT32, Shape.of(3), 3, 1)));
            case PAD -> query(new Operation(PadKind.PAD,
                            new PadAttrs(List.of(1L), List.of(2L), ScalarValue.int32(-1))),
                    List.of(i32(2)), List.of(i32(5)));
            case SLICE -> query(new Operation(SliceKind.SLICE,
                            new SliceAttrs(List.of(1L), List.of(2L), List.of(1), List.of(1L))),
                    List.of(i32(2, 3)),
                    List.of(view(DataType.INT32, Shape.of(2, 2), 1, 3, 1)));
            case SLICE_UPDATE -> query(new Operation(SliceKind.SLICE_UPDATE,
                            new SliceAttrs(List.of(3L), List.of(2L), List.of(0), List.of(-2L))),
                    List.of(i32(4), i32(2)), List.of(i32(4)));
            case CONCAT -> query(new Operation(TensorCompositionKind.CONCAT, new CompositionAxisAttrs(0)),
                    List.of(i32(2), i32(1), i32(2)), List.of(i32(5)));
            case STACK -> query(new Operation(TensorCompositionKind.STACK, new CompositionAxisAttrs(1)),
                    List.of(i32(2), i32(2)), List.of(i32(2, 2)));
            case TILE -> query(new Operation(TileKind.TILE, new TileAttrs(List.of(3L))),
                    List.of(i32(2)), List.of(i32(6)));
            default -> throw new AssertionError(node);
        };
    }

    private static OperationCapabilityQuery attention() {
        TensorDescriptor value = f32(1, 1, 1);
        return query(new Operation(ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                        new ScaledDotProductAttentionAttrs(Optional.empty(), false)),
                List.of(value, value, value), List.of(value));
    }

    private static OperationCapabilityQuery convolution(MetalMpsGraphProgram.NodeKind node) {
        if (node == MetalMpsGraphProgram.NodeKind.CONV2D) {
            return query(new Operation(Conv2dKind.CONV2D, Conv2dAttrs.defaults()),
                    List.of(f32(1, 2, 3, 4), f32(4, 2, 1, 1), f32(4)),
                    List.of(f32(1, 4, 3, 4)));
        }
        return query(new Operation(Conv3dKind.CONV3D,
                        new Conv3dAttrs(1, 2, 1, 0, 1, 0, 1, 1, 2, 1)),
                List.of(f32(1, 2, 4, 5, 6), f32(3, 2, 2, 2, 2)),
                List.of(f32(1, 3, 3, 3, 4)));
    }

    private static OperationCapabilityQuery loss(MetalMpsGraphProgram.NodeKind node) {
        TensorDescriptor value = f32(2, 3);
        return switch (node) {
            case MEAN_SQUARED_ERROR -> query(new Operation(LossKind.MEAN_SQUARED_ERROR,
                            new MeanSquaredErrorAttrs(LossReduction.NONE)),
                    List.of(value, value), List.of(value));
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> query(
                    new Operation(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                            new DenseCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE)),
                    List.of(value, value), List.of(f32(2)));
            case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> query(
                    new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                            new IndexCategoricalCrossEntropyWithLogitsAttrs(
                                    1, LossReduction.NONE, Optional.empty())),
                    List.of(value, i32(2)), List.of(f32(2)));
            default -> throw new AssertionError(node);
        };
    }

    private static OperationCapabilityQuery normalization(MetalMpsGraphProgram.NodeKind node) {
        TensorDescriptor value = f32(2, 3);
        TensorDescriptor channel = f32(3);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        return switch (node) {
            case BATCH_NORM_INFERENCE -> query(new Operation(BatchNormKind.BATCH_NORM_INFERENCE,
                            new BatchNormInferenceAttrs(1, epsilon)),
                    List.of(value, channel, channel, channel, channel), List.of(value));
            case BATCH_NORM_TRAINING -> query(new Operation(BatchNormKind.BATCH_NORM_TRAINING,
                            new BatchNormTrainingAttrs(1, ScalarValue.float32(0.5f), epsilon)),
                    List.of(value, channel, channel, channel, channel),
                    List.of(value, channel, channel, channel, channel));
            case LAYER_NORM -> query(new Operation(LayerNormKind.LAYER_NORM,
                            new LayerNormAttrs(Shape.of(3), epsilon)),
                    List.of(value), List.of(value));
            case RMS_NORM -> query(new Operation(RmsNormKind.RMS_NORM,
                            new RmsNormAttrs(Shape.of(3), epsilon)),
                    List.of(value), List.of(value));
            case SOFTMAX, LOG_SOFTMAX -> query(new Operation(node == MetalMpsGraphProgram.NodeKind.SOFTMAX
                            ? SoftmaxKind.SOFTMAX : SoftmaxKind.LOG_SOFTMAX, new SoftmaxAttrs(1)),
                    List.of(value), List.of(value));
            default -> throw new AssertionError(node);
        };
    }

    private static OperationCapabilityQuery ordering(MetalMpsGraphProgram.NodeKind node) {
        TensorDescriptor input = bool(2, 4, 3);
        return switch (node) {
            case SORT -> query(new Operation(OrderingKind.SORT, new SortAttrs(1, true)),
                    List.of(input), List.of(input));
            case ARGSORT -> query(new Operation(OrderingKind.ARGSORT, new SortAttrs(1, false)),
                    List.of(input), List.of(descriptor(DataType.INT64, input.shape(), false)));
            case TOP_K -> query(new Operation(TopKKind.TOP_K, new TopKAttrs(1, 2, true, false)),
                    List.of(input), List.of(bool(2, 2, 3),
                            descriptor(DataType.INT64, Shape.of(2, 2, 3), false)));
            default -> throw new AssertionError(node);
        };
    }

    private static OperationCapabilityQuery pooling(MetalMpsGraphProgram.NodeKind node) {
        return switch (node) {
            case MAX_POOL2D -> query(new Operation(Pool2dKind.MAX_POOL2D,
                            new MaxPool2dAttrs(2, 3, 2, 2, 1, 1, 1, 1, true)),
                    List.of(f32(1, 2, 3, 4)), List.of(f32(1, 2, 3, 3)));
            case AVERAGE_POOL2D -> query(new Operation(Pool2dKind.AVERAGE_POOL2D,
                            new AveragePool2dAttrs(2, 3, 2, 2, 1, 1, 1, 1, true)),
                    List.of(f32(1, 2, 3, 4)), List.of(f32(1, 2, 3, 3)));
            case MAX_POOL3D -> query(new Operation(Pool3dKind.MAX_POOL3D,
                            new MaxPool3dAttrs(2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, false)),
                    List.of(f32(1, 1, 3, 3, 3)), List.of(f32(1, 1, 2, 2, 2)));
            case AVERAGE_POOL3D -> query(new Operation(Pool3dKind.AVERAGE_POOL3D,
                            new AveragePool3dAttrs(1, 1, 2, 1, 1, 2, 0, 0, 2, 1, 1, 1, true)),
                    List.of(f32(1, 1, 1, 1, 1)), List.of(f32(1, 1, 1, 1, 3)));
            default -> throw new AssertionError(node);
        };
    }

    private static OperationCapabilityQuery random(MetalMpsGraphProgram.NodeKind node) {
        TensorDescriptor state = descriptor(DataType.INT64, Shape.of(2), false);
        if (node == MetalMpsGraphProgram.NodeKind.INITIAL_STATE) {
            return query(new Operation(GraphRngKind.INITIAL_STATE,
                            new GraphRngStateAttrs(0x1234L, 0x5678L)),
                    List.of(), List.of(state));
        }
        return query(new Operation(DropoutKind.DROPOUT, new DropoutAttrs(0.25d)),
                List.of(f32(2, 3), state),
                List.of(f32(2, 3), bool(2, 3), state));
    }

    private static OperationCapabilityQuery recurrent(MetalMpsGraphProgram.NodeKind node) {
        RecurrentScanKind kind = RecurrentScanKind.valueOf(node.name());
        int gates = switch (kind) {
            case RNN_TANH -> 1;
            case GRU_RESET_AFTER -> 3;
            case LSTM -> 4;
        };
        java.util.ArrayList<TensorDescriptor> inputs = new java.util.ArrayList<>();
        inputs.add(f32(2, 2, 3));
        inputs.add(descriptor(DataType.INT64, Shape.of(2), false));
        inputs.add(f32(2, 4));
        if (kind == RecurrentScanKind.LSTM) inputs.add(f32(2, 4));
        inputs.add(f32(gates * 4L, 3));
        inputs.add(f32(gates * 4L, 4));
        java.util.ArrayList<TensorDescriptor> outputs =
                new java.util.ArrayList<>(List.of(f32(2, 2, 4), f32(2, 4)));
        if (kind == RecurrentScanKind.LSTM) outputs.add(f32(2, 4));
        return query(new Operation(kind, RecurrentDirection.FORWARD), inputs, outputs);
    }

    private static OperationCapabilityQuery query(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        return new OperationCapabilityQuery(operation, inputs, outputs);
    }

    private static TensorDescriptor f32(long... dimensions) {
        return descriptor(DataType.FLOAT32, Shape.of(dimensions), false);
    }

    private static TensorDescriptor i32(long... dimensions) {
        return descriptor(DataType.INT32, Shape.of(dimensions), false);
    }

    private static TensorDescriptor bool(long... dimensions) {
        return descriptor(DataType.BOOL, Shape.of(dimensions), false);
    }

    private static TensorDescriptor f32Scalar() {
        return descriptor(DataType.FLOAT32, Shape.scalar(), false);
    }

    private static TensorDescriptor boolScalar() {
        return descriptor(DataType.BOOL, Shape.scalar(), false);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(
                type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
    }

    private static TensorDescriptor view(
            DataType type, Shape shape, long offset, long... strides) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.of(shape, strides, offset, true)), false);
    }

}
