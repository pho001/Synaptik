package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
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
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.loss.*;
import io.github.pho001.synaptik.model.operation.normalization.*;
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
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Defines the closed current first-order disposition of every Model operation role.
 *
 * <p>The checker is the package-private compile-time seam shared by preflight role selection and
 * formula-family dispatch. A role is conditionally differentiable ({@link Disposition#D}),
 * intentionally non-differentiable ({@link Disposition#ND}), or fail-closed
 * ({@link Disposition#FC}). Conditional support still requires
 * {@link AutogradPreflight}'s occurrence-local shape, type, cardinality, auxiliary, and policy
 * proof before a formula may be constructed.</p>
 *
 * <p>The immutable signature inventory is deliberately exact: 38 current operation-kind enum
 * families, 111 constants, and 133 complete kind/attributes/cardinality fingerprints. It names
 * every accepted current variant instead of deriving the inventory from Model declarations, so
 * a newly added or changed production signature fails closed until its first-order disposition
 * is reviewed. This type constructs no Tensor or graph state, scans no class path, uses no
 * reflection, retains no request state, and is neither a public registry nor an extension
 * surface.</p>
 */
final class FirstOrderGradientCoverage {
    private static final List<SignatureFingerprint> SIGNATURES = buildSignatures();

    private FirstOrderGradientCoverage() {}

    /**
     * Classifies one exact output/input edge of a producer occurrence without constructing model
     * or graph state.
     *
     * @param producer non-null exact producer occurrence
     * @param outputIndex selected zero-based output slot
     * @param inputIndex selected zero-based input position, or {@code -1} to classify the output
     *     role itself
     * @return the non-null closed disposition, formula owner when differentiable, and deterministic
     *     reason when non-differentiable or fail-closed
     * @throws NullPointerException if {@code producer} is {@code null}
     */
    static Decision classify(TensorProducer producer, int outputIndex, int inputIndex) {
        Objects.requireNonNull(producer, "producer");
        DataType inputType = inputIndex >= 0 && inputIndex < producer.inputs().size()
                ? producer.inputs().get(inputIndex).descriptor().dataType()
                : null;
        DataType outputType = outputIndex >= 0 && outputIndex < producer.outputCount()
                ? producer.output(outputIndex).descriptor().dataType()
                : null;
        return classify(
                producer.operation().kind(),
                producer.operation().attrs().getClass(),
                producer.inputs().size(),
                producer.outputCount(),
                outputIndex,
                inputIndex,
                inputType,
                outputType);
    }

    /**
     * Classifies explicit immutable occurrence facts. This overload is the Tensor- and
     * graph-construction-free test seam for complete signature and legal-cardinality scans. It
     * may create the returned {@link Decision}.
     *
     * @param kind non-null exact operation kind
     * @param attributesType non-null exact attributes implementation
     * @param inputCount logical input count; a value outside the exact signature range fails
     *     closed
     * @param outputCount logical output count; a value outside the exact signature range fails
     *     closed
     * @param outputIndex selected zero-based output slot; an illegal slot fails closed
     * @param inputIndex selected zero-based input position, or {@code -1} for the output role; an
     *     illegal position fails closed
     * @param inputType exact input type for an input role, or {@code null} when unavailable;
     *     ignored for an output role
     * @param outputType exact output type when available, otherwise {@code null}
     * @return the non-null closed disposition, with exactly one family owner for {@code D} or one
     *     nonblank reason for {@code ND}/{@code FC}
     * @throws NullPointerException if {@code kind} or {@code attributesType} is {@code null}
     */
    static Decision classify(
            OperationKind kind,
            Class<? extends OperationAttrs> attributesType,
            int inputCount,
            int outputCount,
            int outputIndex,
            int inputIndex,
            DataType inputType,
            DataType outputType) {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(attributesType, "attributesType");

        SignatureFingerprint signature = find(kind, attributesType);
        if (signature == null) {
            return fc("unknown or unclassified operation kind/attributes pairing");
        }
        if (inputCount < signature.minimumInputs() || inputCount > signature.maximumInputs()
                || outputCount < signature.minimumOutputs()
                || outputCount > signature.maximumOutputs()) {
            return fc("operation cardinality is outside the classified signature");
        }
        if (outputIndex < 0 || outputIndex >= outputCount) {
            return fc("selected output slot is outside the operation occurrence");
        }
        if (inputIndex < -1 || inputIndex >= inputCount) {
            return fc("selected input position is outside the operation occurrence");
        }

        if (inputIndex == -1) {
            return classifyOutput(kind, outputIndex, outputType);
        }
        return classifyInput(kind, outputCount, outputIndex, inputIndex, inputType, outputType);
    }

    /**
     * Returns the exact immutable Compiler checkpoint inventory.
     *
     * @return the non-null immutable 133-row signature fingerprint list
     */
    static List<SignatureFingerprint> signatures() {
        return SIGNATURES;
    }

    private static Decision classifyOutput(
            OperationKind kind, int outputIndex, DataType outputType) {
        if (kind instanceof BinaryComparisonKind
                || kind instanceof BooleanLogicalKind
                || kind instanceof FloatingClassificationKind
                || kind instanceof OneHotKind
                || kind == OrderingKind.ARGSORT
                || kind instanceof GraphRngKind) {
            return nd("output has no Tensor cotangent route");
        }
        if (kind == BatchNormKind.BATCH_NORM_TRAINING && outputIndex >= 3) {
            return fc(
                    "saved auxiliary output is non-differentiable as an independent cotangent root");
        }
        if (kind instanceof TopKKind && outputIndex == 1) {
            return nd("TOP_K indices output has no Tensor cotangent route");
        }
        if (kind instanceof DropoutKind && outputIndex != 0) {
            return nd("dropout mask and next-state outputs have no Tensor cotangent route");
        }
        if (outputType != null && !outputType.isFloating()) {
            return nd("non-floating output has no Tensor cotangent route");
        }
        FamilyOwner owner = owner(kind);
        return owner == null
                ? nd("operation output is intentionally non-differentiable")
                : d(owner);
    }

    private static Decision classifyInput(
            OperationKind kind,
            int outputCount,
            int outputIndex,
            int inputIndex,
            DataType inputType,
            DataType outputType) {
        if (kind instanceof BinaryComparisonKind
                || kind instanceof BooleanLogicalKind
                || kind instanceof FloatingClassificationKind
                || kind instanceof OneHotKind
                || kind == OrderingKind.ARGSORT
                || kind instanceof GraphRngKind) {
            return nd("input role has no Tensor cotangent route");
        }
        if (kind == WhereSelectionKind.WHERE && inputIndex == 0) {
            return nd("WHERE condition role is non-differentiable");
        }
        if (kind instanceof AggregateReductionKind reduction) {
            if (reduction == AggregateReductionKind.ALL
                    || reduction == AggregateReductionKind.ANY
                    || reduction == AggregateReductionKind.ARG_MIN
                    || reduction == AggregateReductionKind.ARG_MAX) {
                return nd("logical and arg-extrema reductions have no Tensor cotangent route");
            }
            if (inputIndex != 0) {
                return nd("reduction mask role is non-differentiable");
            }
        }
        if (kind instanceof AxisGatherKind || kind instanceof GatherNdKind) {
            if (inputIndex != 0) {
                return nd("gather index role is non-differentiable");
            }
        }
        if (kind instanceof AxisScatterKind || kind instanceof ScatterNdKind) {
            if (inputIndex == 1) {
                return nd("scatter index role is non-differentiable");
            }
        }
        if (kind == BatchNormKind.BATCH_NORM_TRAINING) {
            boolean differentiable = switch (outputIndex) {
                case 0 -> inputIndex <= 2;
                case 1 -> inputIndex == 0 || inputIndex == 3;
                case 2 -> inputIndex == 0 || inputIndex == 4;
                case 3, 4 -> false;
                default -> false;
            };
            if (!differentiable) {
                return outputIndex >= 3
                        ? fc(
                                "saved auxiliary output is non-differentiable as an independent cotangent root")
                        : nd("batch-normalization role has no cotangent route for this output");
            }
        }
        if (kind == ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION) {
            if (inputIndex == 3 || (outputIndex == 1 && inputIndex >= 2)) {
                return nd("attention mask/value role has no cotangent route for this output");
            }
            if (outputCount == 1) {
                return fc("attention gradients require output and canonical weights slots");
            }
        }
        if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                && inputIndex == 1) {
            return nd("index categorical target role is non-differentiable");
        }
        if (kind instanceof TopKKind && outputIndex == 1) {
            return nd("TOP_K indices output has no Tensor cotangent route");
        }
        if (kind instanceof DropoutKind) {
            if (outputIndex != 0 || inputIndex != 0) {
                return nd("dropout state and mask roles are non-differentiable");
            }
        }
        if (kind == CastKind.CAST
                && (inputType == null || outputType == null
                        || !inputType.isFloating() || !outputType.isFloating())) {
            return nd("CAST is differentiable only between floating types");
        }
        if (kind == OrderingKind.SORT && (inputType == null || !inputType.isFloating())) {
            return nd("non-floating SORT input has no Tensor cotangent route");
        }
        if (kind instanceof CumulativeScanKind
                && (inputType == null || !inputType.isFloating())) {
            return nd("integral cumulative-scan input has no Tensor cotangent route");
        }
        if (inputType == null || !inputType.isFloating()) {
            return nd("non-floating data role has no Tensor cotangent route");
        }

        FamilyOwner owner = owner(kind);
        return owner == null
                ? fc("differentiable role has no classified formula-family owner")
                : d(owner);
    }

    private static FamilyOwner owner(OperationKind kind) {
        if (kind instanceof BinaryArithmeticKind
                || kind instanceof ScalarElementwiseKind
                || kind instanceof UnaryElementwiseKind
                || kind == WhereSelectionKind.WHERE
                || kind == CastKind.CAST) {
            return FamilyOwner.ELEMENTWISE;
        }
        if (kind instanceof AggregateReductionKind
                || kind instanceof CumulativeScanKind
                || kind instanceof SoftmaxKind) {
            return FamilyOwner.REDUCTION;
        }
        if (kind instanceof LayerNormKind
                || kind instanceof RmsNormKind
                || kind instanceof BatchNormKind) {
            return FamilyOwner.NORMALIZATION;
        }
        if (kind == MatmulKind.MATMUL) {
            return FamilyOwner.LINEAR_ALGEBRA;
        }
        if (kind == ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION) {
            return FamilyOwner.ATTENTION;
        }
        if (kind == Conv2dKind.CONV2D) {
            return FamilyOwner.CONVOLUTION;
        }
        if (kind instanceof Pool2dKind || kind instanceof Pool3dKind) {
            return FamilyOwner.POOLING;
        }
        if (kind instanceof LossKind) {
            return FamilyOwner.LOSS;
        }
        if (kind == ContiguousKind.CONTIGUOUS
                || kind instanceof ShapeTransformKind
                || kind instanceof AxisTransformKind
                || kind instanceof SliceKind
                || kind == SelectKind.SELECT
                || kind == PadKind.PAD
                || kind == TileKind.TILE
                || kind instanceof TensorCompositionKind
                || kind instanceof WindowTransformKind) {
            return FamilyOwner.LAYOUT;
        }
        if (kind instanceof AxisGatherKind
                || kind instanceof GatherNdKind
                || kind instanceof AxisScatterKind
                || kind instanceof ScatterNdKind) {
            return FamilyOwner.INDEXING;
        }
        if (kind == OrderingKind.SORT || kind instanceof TopKKind) {
            return FamilyOwner.ORDERING;
        }
        if (kind instanceof DropoutKind) {
            return FamilyOwner.STOCHASTIC;
        }
        return null;
    }

    private static SignatureFingerprint find(
            OperationKind kind, Class<? extends OperationAttrs> attributesType) {
        for (SignatureFingerprint fingerprint : SIGNATURES) {
            if (fingerprint.kind() == kind && fingerprint.attributesType() == attributesType) {
                return fingerprint;
            }
        }
        return null;
    }

    private static Decision d(FamilyOwner owner) {
        return new Decision(Disposition.D, owner, "");
    }

    private static Decision nd(String reason) {
        return new Decision(Disposition.ND, null, reason);
    }

    private static Decision fc(String reason) {
        return new Decision(Disposition.FC, null, reason);
    }

    private static List<SignatureFingerprint> buildSignatures() {
        List<SignatureFingerprint> rows = new ArrayList<>(133);
        rows.add(range(
                ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                ScaledDotProductAttentionAttrs.class, 3, 4, 1, 2));
        rows.add(range(Conv2dKind.CONV2D, Conv2dAttrs.class, 2, 3, 1, 1));
        addFixed(rows, List.of(
                BinaryArithmeticKind.ADD, BinaryArithmeticKind.SUB, BinaryArithmeticKind.MUL,
                BinaryArithmeticKind.DIV, BinaryArithmeticKind.MIN, BinaryArithmeticKind.MAX,
                BinaryArithmeticKind.POW), NoOperationAttrs.class, 2, 1);
        rows.add(fixed(CastKind.CAST, CastAttrs.class, 1, 1));
        addFixed(rows, List.of(
                FloatingClassificationKind.IS_FINITE, FloatingClassificationKind.IS_NAN,
                FloatingClassificationKind.IS_INF), NoOperationAttrs.class, 1, 1);
        addFixed(rows, List.of(
                BinaryComparisonKind.GREATER_THAN, BinaryComparisonKind.GREATER_OR_EQUAL,
                BinaryComparisonKind.LESS_THAN, BinaryComparisonKind.LESS_OR_EQUAL,
                BinaryComparisonKind.EQUAL, BinaryComparisonKind.NOT_EQUAL),
                NoOperationAttrs.class, 2, 1);
        addFixed(rows, List.of(BooleanLogicalKind.AND, BooleanLogicalKind.OR),
                NoOperationAttrs.class, 2, 1);
        rows.add(fixed(BooleanLogicalKind.NOT, NoOperationAttrs.class, 1, 1));
        addFixed(rows, List.of(
                ScalarElementwiseKind.ADD, ScalarElementwiseKind.SUB, ScalarElementwiseKind.MUL,
                ScalarElementwiseKind.DIV, ScalarElementwiseKind.MIN, ScalarElementwiseKind.MAX,
                ScalarElementwiseKind.POW), ScalarValueAttrs.class, 1, 1);
        rows.add(fixed(ScalarElementwiseKind.CLAMP, ClampRangeAttrs.class, 1, 1));
        rows.add(fixed(WhereSelectionKind.WHERE, NoOperationAttrs.class, 3, 1));
        addFixed(rows, List.of(
                UnaryElementwiseKind.ABS, UnaryElementwiseKind.NEG,
                UnaryElementwiseKind.RECIPROCAL, UnaryElementwiseKind.LOG,
                UnaryElementwiseKind.LOG1P, UnaryElementwiseKind.EXP,
                UnaryElementwiseKind.EXPM1, UnaryElementwiseKind.ERF,
                UnaryElementwiseKind.SQRT, UnaryElementwiseKind.RSQRT,
                UnaryElementwiseKind.FLOOR, UnaryElementwiseKind.CEIL,
                UnaryElementwiseKind.SIGN, UnaryElementwiseKind.RELU,
                UnaryElementwiseKind.SIGMOID, UnaryElementwiseKind.TANH,
                UnaryElementwiseKind.GELU, UnaryElementwiseKind.GELU_TANH_APPROXIMATION,
                UnaryElementwiseKind.SILU), NoOperationAttrs.class, 1, 1);
        addFixed(rows, List.of(AxisGatherKind.GATHER, AxisGatherKind.GATHER_ELEMENTS),
                IndexAxisAttrs.class, 2, 1);
        rows.add(fixed(
                AxisScatterKind.SCATTER_ELEMENTS, ScatterElementsAttrs.class, 3, 1));
        rows.add(fixed(AxisScatterKind.SCATTER_ADD, IndexAxisAttrs.class, 3, 1));
        rows.add(fixed(GatherNdKind.GATHER_ND, GatherNdAttrs.class, 2, 1));
        rows.add(fixed(OneHotKind.ONE_HOT, OneHotAttrs.class, 1, 1));
        rows.add(fixed(ScatterNdKind.SCATTER_ND, ScatterNdAttrs.class, 3, 1));
        rows.add(fixed(SelectKind.SELECT, SelectAttrs.class, 1, 1));
        rows.add(fixed(AxisTransformKind.PERMUTE, PermutationAttrs.class, 1, 1));
        addFixed(rows, List.of(AxisTransformKind.EXPAND_DIMS, AxisTransformKind.SQUEEZE),
                AxisTransformAttrs.class, 1, 1);
        rows.add(fixed(ContiguousKind.CONTIGUOUS, NoOperationAttrs.class, 1, 1));
        rows.add(fixed(PadKind.PAD, PadAttrs.class, 1, 1));
        addFixed(rows, List.of(ShapeTransformKind.RESHAPE, ShapeTransformKind.EXPAND),
                TargetShapeAttrs.class, 1, 1);
        rows.add(fixed(SliceKind.SLICE, SliceAttrs.class, 1, 1));
        rows.add(fixed(SliceKind.SLICE, CropToShapeAttrs.class, 1, 1));
        rows.add(fixed(SliceKind.SLICE_UPDATE, SliceAttrs.class, 2, 1));
        rows.add(fixed(SliceKind.SLICE_UPDATE, CropToShapeAttrs.class, 2, 1));
        addRange(rows, List.of(TensorCompositionKind.CONCAT, TensorCompositionKind.STACK),
                CompositionAxisAttrs.class, 1, Integer.MAX_VALUE, 1, 1);
        rows.add(fixed(TileKind.TILE, TileAttrs.class, 1, 1));
        rows.add(fixed(WindowTransformKind.UNFOLD_AXIS, UnfoldAxisAttrs.class, 1, 1));
        rows.add(fixed(WindowTransformKind.FOLD_AXIS, FoldAxisAttrs.class, 1, 1));
        rows.add(fixed(WindowTransformKind.UNFOLD2D, Window2dAttrs.class, 1, 1));
        rows.add(fixed(WindowTransformKind.UNFOLD2D, Unfold2dAttrs.class, 1, 1));
        rows.add(fixed(WindowTransformKind.FOLD2D, Fold2dAttrs.class, 1, 1));
        rows.add(fixed(WindowTransformKind.UNFOLD3D, Window3dAttrs.class, 1, 1));
        rows.add(fixed(WindowTransformKind.UNFOLD3D, Unfold3dAttrs.class, 1, 1));
        rows.add(fixed(WindowTransformKind.FOLD3D, Fold3dAttrs.class, 1, 1));
        rows.add(fixed(MatmulKind.MATMUL, NoOperationAttrs.class, 2, 1));
        rows.add(fixed(
                LossKind.MEAN_SQUARED_ERROR, MeanSquaredErrorAttrs.class, 2, 1));
        rows.add(fixed(
                LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                DenseCategoricalCrossEntropyWithLogitsAttrs.class, 2, 1));
        rows.add(fixed(
                LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                IndexCategoricalCrossEntropyWithLogitsAttrs.class, 2, 1));
        rows.add(fixed(
                BatchNormKind.BATCH_NORM_INFERENCE, BatchNormInferenceAttrs.class, 5, 1));
        rows.add(fixed(
                BatchNormKind.BATCH_NORM_TRAINING, BatchNormTrainingAttrs.class, 5, 5));
        rows.add(fixed(LayerNormKind.LAYER_NORM, LayerNormAttrs.class, 1, 1));
        rows.add(fixed(LayerNormKind.LAYER_NORM, AffineLayerNormAttrs.class, 3, 1));
        rows.add(range(RmsNormKind.RMS_NORM, RmsNormAttrs.class, 1, 2, 1, 1));
        addFixed(rows, List.of(SoftmaxKind.SOFTMAX, SoftmaxKind.LOG_SOFTMAX),
                SoftmaxAttrs.class, 1, 1);
        addFixed(rows, List.of(OrderingKind.SORT, OrderingKind.ARGSORT),
                SortAttrs.class, 1, 1);
        rows.add(fixed(TopKKind.TOP_K, TopKAttrs.class, 1, 2));
        rows.add(fixed(Pool2dKind.MAX_POOL2D, MaxPool2dAttrs.class, 1, 1));
        rows.add(fixed(Pool2dKind.AVERAGE_POOL2D, AveragePool2dAttrs.class, 1, 1));
        rows.add(fixed(Pool3dKind.MAX_POOL3D, MaxPool3dAttrs.class, 1, 1));
        rows.add(fixed(Pool3dKind.AVERAGE_POOL3D, AveragePool3dAttrs.class, 1, 1));
        rows.add(fixed(DropoutKind.DROPOUT, DropoutAttrs.class, 2, 3));
        rows.add(fixed(GraphRngKind.INITIAL_STATE, GraphRngStateAttrs.class, 0, 1));

        addFixed(rows, List.of(AggregateReductionKind.SUM),
                NoOperationAttrs.class, 1, 1);
        rows.add(fixed(AggregateReductionKind.SUM, AxisReductionAttrs.class, 1, 1));
        rows.add(fixed(AggregateReductionKind.SUM, MultiAxisReductionAttrs.class, 1, 1));
        rows.add(fixed(AggregateReductionKind.SUM, MaskedReductionAttrs.class, 2, 1));
        rows.add(fixed(AggregateReductionKind.SUM, SumToShapeAttrs.class, 1, 1));
        rows.add(fixed(AggregateReductionKind.MEAN, NoOperationAttrs.class, 1, 1));
        rows.add(fixed(AggregateReductionKind.MEAN, AxisReductionAttrs.class, 1, 1));
        rows.add(fixed(AggregateReductionKind.MEAN, MultiAxisReductionAttrs.class, 1, 1));
        rows.add(fixed(AggregateReductionKind.MEAN, MaskedReductionAttrs.class, 2, 1));
        addOrdinaryReductions(rows, List.of(
                AggregateReductionKind.PROD, AggregateReductionKind.MIN,
                AggregateReductionKind.MAX, AggregateReductionKind.ALL,
                AggregateReductionKind.ANY));
        rows.add(fixed(AggregateReductionKind.ARG_MIN, ArgExtremaAttrs.class, 1, 1));
        rows.add(fixed(AggregateReductionKind.ARG_MAX, ArgExtremaAttrs.class, 1, 1));
        addFixed(rows, List.of(
                AggregateReductionKind.LOG_SUM_EXP, AggregateReductionKind.L1_NORM,
                AggregateReductionKind.L2_NORM), MultiAxisReductionAttrs.class, 1, 1);
        addFixed(rows, List.of(
                AggregateReductionKind.VARIANCE,
                AggregateReductionKind.STANDARD_DEVIATION),
                StatisticalReductionAttrs.class, 1, 1);
        addFixed(rows, List.of(CumulativeScanKind.CUM_SUM, CumulativeScanKind.CUM_PROD),
                CumulativeScanAttrs.class, 1, 1);
        return List.copyOf(rows);
    }

    private static void addOrdinaryReductions(
            List<SignatureFingerprint> rows, List<? extends OperationKind> kinds) {
        for (OperationKind kind : kinds) {
            rows.add(fixed(kind, NoOperationAttrs.class, 1, 1));
            rows.add(fixed(kind, AxisReductionAttrs.class, 1, 1));
            rows.add(fixed(kind, MultiAxisReductionAttrs.class, 1, 1));
        }
    }

    private static void addFixed(
            List<SignatureFingerprint> rows,
            List<? extends OperationKind> kinds,
            Class<? extends OperationAttrs> attributesType,
            int inputCount,
            int outputCount) {
        addRange(
                rows,
                kinds,
                attributesType,
                inputCount,
                inputCount,
                outputCount,
                outputCount);
    }

    private static void addRange(
            List<SignatureFingerprint> rows,
            List<? extends OperationKind> kinds,
            Class<? extends OperationAttrs> attributesType,
            int minimumInputs,
            int maximumInputs,
            int minimumOutputs,
            int maximumOutputs) {
        for (OperationKind kind : kinds) {
            rows.add(range(
                    kind,
                    attributesType,
                    minimumInputs,
                    maximumInputs,
                    minimumOutputs,
                    maximumOutputs));
        }
    }

    private static SignatureFingerprint fixed(
            OperationKind kind,
            Class<? extends OperationAttrs> attributesType,
            int inputCount,
            int outputCount) {
        return range(
                kind,
                attributesType,
                inputCount,
                inputCount,
                outputCount,
                outputCount);
    }

    private static SignatureFingerprint range(
            OperationKind kind,
            Class<? extends OperationAttrs> attributesType,
            int minimumInputs,
            int maximumInputs,
            int minimumOutputs,
            int maximumOutputs) {
        return new SignatureFingerprint(
                kind,
                attributesType,
                minimumInputs,
                maximumInputs,
                minimumOutputs,
                maximumOutputs);
    }

    /**
     * Closed current outcome for one exact operation output/input role.
     */
    enum Disposition {
        /** Conditionally differentiable after occurrence-local preflight succeeds. */
        D,

        /** Intentionally non-differentiable and therefore has no formula owner. */
        ND,

        /** Rejected fail-closed because a required structural or semantic proof is unavailable. */
        FC
    }

    /**
     * Existing compiler component that owns the selected first-order formula family.
     *
     * <p>These values route only package-private dispatch. They are not public capability names,
     * registry keys, extension identifiers, or serialized state.</p>
     */
    enum FamilyOwner {
        /** Binary, scalar, unary, selection, and cast formulas. */
        ELEMENTWISE,

        /** Aggregate-reduction, cumulative-scan, and softmax formulas. */
        REDUCTION,

        /** Layer, root-mean-square, and batch-normalization formulas. */
        NORMALIZATION,

        /** Matrix-multiplication formulas. */
        LINEAR_ALGEBRA,

        /** Scaled-dot-product-attention formulas. */
        ATTENTION,

        /** Two-dimensional convolution formulas. */
        CONVOLUTION,

        /** Two- and three-dimensional pooling formulas. */
        POOLING,

        /** Current loss formulas. */
        LOSS,

        /** Layout, shape, slice, composition, and window-transform formulas. */
        LAYOUT,

        /** Gather and functional-scatter formulas. */
        INDEXING,

        /** Sort and top-K formulas. */
        ORDERING,

        /** Explicit-state dropout formulas. */
        STOCHASTIC
    }

    /**
     * Immutable result of classifying one exact output/input role.
     *
     * @param disposition non-null closed outcome
     * @param owner exact formula family for {@link Disposition#D}, otherwise {@code null}
     * @param reason empty for {@link Disposition#D}; nonblank deterministic explanation for
     *     {@link Disposition#ND} and {@link Disposition#FC}
     */
    record Decision(Disposition disposition, FamilyOwner owner, String reason) {
        /**
         * Validates the closed decision invariant.
         *
         * @param disposition non-null closed outcome
         * @param owner exact formula family for {@code D}, otherwise {@code null}
         * @param reason empty for {@code D}; nonblank for {@code ND} and {@code FC}
         * @throws NullPointerException if {@code disposition} or {@code reason} is {@code null}
         * @throws IllegalArgumentException if owner and reason do not match the disposition
         */
        Decision {
            Objects.requireNonNull(disposition, "disposition");
            Objects.requireNonNull(reason, "reason");
            if (disposition == Disposition.D) {
                if (owner == null || !reason.isEmpty()) {
                    throw new IllegalArgumentException(
                            "D requires one family owner and no rejection reason");
                }
            } else if (owner != null || reason.isBlank()) {
                throw new IllegalArgumentException(
                        "ND/FC require no family owner and one nonblank reason");
            }
        }
    }

    /**
     * Exact current operation-signature identity used by the closed coverage inventory.
     *
     * @param kind non-null exact operation-kind constant
     * @param attributesType non-null exact attributes implementation
     * @param minimumInputs inclusive minimum logical input count
     * @param maximumInputs inclusive maximum logical input count
     * @param minimumOutputs inclusive minimum logical output count
     * @param maximumOutputs inclusive maximum logical output count
     */
    record SignatureFingerprint(
            OperationKind kind,
            Class<? extends OperationAttrs> attributesType,
            int minimumInputs,
            int maximumInputs,
            int minimumOutputs,
            int maximumOutputs) {
        /**
         * Validates the non-null identity components of one fingerprint.
         *
         * @param kind non-null exact operation-kind constant
         * @param attributesType non-null exact attributes implementation
         * @param minimumInputs inclusive minimum logical input count
         * @param maximumInputs inclusive maximum logical input count
         * @param minimumOutputs inclusive minimum logical output count
         * @param maximumOutputs inclusive maximum logical output count
         * @throws NullPointerException if {@code kind} or {@code attributesType} is {@code null}
         */
        SignatureFingerprint {
            Objects.requireNonNull(kind, "kind");
            Objects.requireNonNull(attributesType, "attributesType");
        }
    }
}
