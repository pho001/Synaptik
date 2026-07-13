package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free loss expressions.
 *
 * <p>Mean-squared error records ordered {@code [prediction, target]} inputs with exact positional
 * shape compatibility and no broadcasting. It describes squared differences followed by explicit
 * none, sum, or complete-domain mean reduction, but reads no values and selects no executable
 * algorithm, gradient, compiler transformation, backend route, runtime behavior, or training
 * coordination.</p>
 *
 * <p>Dense categorical cross-entropy records ordered {@code [logits, target]} inputs with an
 * exact-shape dense target and one normalized class axis. Its meaning is target-weighted negative
 * log-softmax computed stably from logits, followed by explicit none, sum, or sample-count mean
 * reduction. Target values remain a caller obligation: construction neither reads nor normalizes
 * them. Class-extent and unresolved-shape obligations may remain for later compiler binding.</p>
 *
 * <p>Index categorical cross-entropy uses exact INT32 or INT64 targets whose Shape is the logits
 * Shape with the normalized class axis removed. It selects one stable negative-log-softmax value
 * per target, optionally ignores one exact typed target value before bounds or logits evaluation,
 * and divides mean reduction by the non-ignored count. Construction reads no target or logits
 * values and records the ignore value only in operation attributes.</p>
 */
final class TensorLossExpressions {
    /** Prevents instantiation because loss expression construction owns no state. */
    private TensorLossExpressions() {
    }

    /**
     * Creates one fresh mean-squared-error expression.
     *
     * <p>Validation checks prediction, target, and reduction nulls in that order, then floating
     * types in input order, rank equality, and dimensions in increasing axis order. Unequal static
     * dimensions fail; unequal pairs involving an unresolved dimension are deferred for later
     * compiler proof. Promotion occurs only after those validations.</p>
     *
     * <p>The selected meaning is {@code (prediction - target)^2}. None retains every coordinate,
     * sum reduces the complete logical domain, and mean divides that sum by the complete logical
     * element count. A scalar count is one; a zero-extent domain has empty none, positive-zero sum,
     * and NaN mean. BFLOAT16/FLOAT32 use FLOAT32 computation and FLOAT64 uses FLOAT64. NaN,
     * infinity, signed-zero, overflow, reassociation, and determinism follow the public
     * {@link Tensor#meanSquaredError(Tensor, LossReduction)} contract.</p>
     *
     * @param prediction non-null floating prediction retained at producer input position zero
     * @param target non-null floating exact-shape target retained at producer input position one
     * @param reduction non-null explicit {@link LossReduction#NONE}, {@link LossReduction#SUM}, or
     *     {@link LossReduction#MEAN} selection
     * @return a fresh unlabeled, storage-free, unresolved-layout tensor whose type is the promoted
     *     floating type, whose gradient eligibility is the input logical OR, and whose shape is
     *     the exact prediction shape for {@code NONE} or shared scalar shape otherwise
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if either input is not floating, ranks differ, or unequal
     *     static dimensions occur at any corresponding axis
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    static Tensor meanSquaredError(
            Tensor prediction, Tensor target, LossReduction reduction) {
        Objects.requireNonNull(prediction, "prediction");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(reduction, "reduction");

        DataType predictionType = requireFloating(prediction, "meanSquaredError", "prediction");
        DataType targetType = requireFloating(target, "meanSquaredError", "target");
        Shape predictionShape = prediction.descriptor().shape();
        Shape targetShape = target.descriptor().shape();
        validateExactShape(
                predictionShape, targetShape, "meanSquaredError", "prediction");
        DataType resultType = DataTypePromotion.promoteFloating(predictionType, targetType);
        MeanSquaredErrorAttrs attrs = new MeanSquaredErrorAttrs(reduction);
        Shape resultShape = reduction == LossReduction.NONE ? predictionShape : Shape.scalar();
        boolean requiresGrad = prediction.descriptor().requiresGrad()
                || target.descriptor().requiresGrad();
        TensorDescriptor descriptor = new TensorDescriptor(
                resultType, resultShape, Optional.empty(), requiresGrad);
        Operation operation = new Operation(LossKind.MEAN_SQUARED_ERROR, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(prediction, target));
    }

    /**
     * Dispatches one fresh categorical-cross-entropy occurrence directly from logits.
     *
     * <p>A floating target selects the existing dense target-weighted branch. Exact INT32 or INT64
     * selects the index branch without an ignore value. Any other target follows the dense
     * unsupported-target failure. Construction validates metadata only; it does not inspect target
     * values, decompose either stable meaning into public primitives, or define gradients,
     * compiler proof, lowering, execution, or training behavior.</p>
     *
     * <p>{@link LossReduction#NONE} removes the normalized class axis, while sum and mean return
     * scalar Shape; mean divides by the number of non-class groups. Rank-one logits therefore
     * have sample count one. A zero non-class extent gives empty none, positive-zero sum, and NaN
     * mean without evaluating a class slice. A statically zero class extent is valid for that
     * definitely empty sample domain, fails for a definitely non-empty domain, and otherwise
     * retains the later obligation {@code sampleCount == 0 || classExtent > 0}.</p>
     *
     * <p>On the dense branch, logits and target participate in floating promotion and gradient
     * eligibility. On the index branch, result type and gradient eligibility come only from
     * logits; the integral target is neither promoted nor cast. BFLOAT16 and FLOAT32 index results
     * use at least FLOAT32 log-sum-exp, subtraction, accumulation, and division, while FLOAT64
     * uses FLOAT64. These are semantic requirements, not eager evaluation or backend-algorithm
     * selection.</p>
     *
     * @param logits non-null floating logits retained at producer input position zero
     * @param target non-null floating exact-shape dense probability target or exact INT32/INT64
     *     class-index target retained at input one; construction reads none of its values
     * @param classAxis positive or negative logits class axis normalized exactly once
     * @param reduction non-null explicit none, sum, or sample-domain mean reduction
     * @return fresh unlabeled, storage-free loss tensor with the selected dense or index metadata,
     *     unresolved layout, and output-index-zero provenance
     * @throws NullPointerException if {@code logits}, {@code target}, or {@code reduction} is null,
     *     checked in that order
     * @throws IndexOutOfBoundsException if {@code classAxis} is outside the logits rank
     * @throws IllegalArgumentException if logits is not floating; target is neither floating nor
     *     exact INT32/INT64; the selected target Shape rule fails; or the no-ignore branch has a
     *     statically empty class axis and a definitely non-empty sample domain
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor categoricalCrossEntropyWithLogits(
            Tensor logits, Tensor target, int classAxis, LossReduction reduction) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(reduction, "reduction");

        DataType logitsType = requireFloating(
                logits, "categoricalCrossEntropyWithLogits", "logits");
        DataType targetType = target.descriptor().dataType();
        if (targetType == DataType.INT32 || targetType == DataType.INT64) {
            return indexCategoricalCrossEntropyWithLogits(
                    logits, target, classAxis, reduction, Optional.empty(), logitsType);
        }
        requireFloating(target, "categoricalCrossEntropyWithLogits", "target");
        return denseCategoricalCrossEntropyWithLogits(
                logits, target, classAxis, reduction, logitsType, targetType);
    }

    /**
     * Creates one fresh ignore-aware index-target categorical-cross-entropy occurrence.
     *
     * <p>The target must have exact INT32 or INT64 type and the Shape obtained by removing the
     * normalized class axis from logits. The ignore value must have the exact target type. It is
     * retained in attributes, not provenance; ordered producer inputs remain
     * {@code [logits, target]}. {@link LossReduction#NONE} retains the exact target Shape, while
     * sum and non-ignored-count mean use scalar Shape. Result type and gradient eligibility come
     * only from logits.</p>
     *
     * <p>At execution, matching the ignore value precedes bounds checking and logits evaluation.
     * Every non-ignored target must satisfy {@code 0 <= target < classExtent} and selects
     * {@code lse - selectedLogit} from its stable log-softmax slice. Empty or all-ignored domains
     * have positive-zero sum and NaN mean. Because construction reads no values, a zero class
     * extent retains the later obligation that the sample domain is empty or every target is
     * ignored; dynamic Shape equality and bounds also remain deferred.</p>
     *
     * <p>BFLOAT16 and FLOAT32 use at least FLOAT32 computation; FLOAT64 uses FLOAT64. Ignored
     * positions stay positive zero without propagating their logits' NaN or infinity. For a
     * non-ignored slice, NaN or positive infinity and an all-negative-infinity slice produce NaN;
     * selecting negative infinity when another class is finite produces positive infinity.</p>
     *
     * @param logits non-null BFLOAT16, FLOAT32, or FLOAT64 logits retained at input zero
     * @param target non-null exact INT32 or INT64 class-index target retained at input one
     * @param classAxis positive or negative logits class axis normalized exactly once
     * @param reduction non-null explicit none, sum, or non-ignored-count mean reduction
     * @param ignoreIndex non-null exact INT32 or INT64 scalar whose type equals target type
     * @return fresh unlabeled, storage-free loss tensor with exact logits type, exact target Shape
     *     for none or scalar Shape otherwise, logits-only gradient eligibility, unresolved layout,
     *     and output-index-zero provenance
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IndexOutOfBoundsException if {@code classAxis} is outside the logits rank
     * @throws IllegalArgumentException if logits, target, or ignore type is invalid; ignore and
     *     target types differ; or target rank or a mapped static dimension mismatches logits
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor categoricalCrossEntropyWithLogits(
            Tensor logits,
            Tensor target,
            int classAxis,
            LossReduction reduction,
            ScalarValue ignoreIndex) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(reduction, "reduction");
        Objects.requireNonNull(ignoreIndex, "ignoreIndex");

        DataType logitsType = requireFloating(
                logits, "categoricalCrossEntropyWithLogits", "logits");
        DataType targetType = target.descriptor().dataType();
        if (targetType != DataType.INT32 && targetType != DataType.INT64) {
            throw new IllegalArgumentException(
                    "categoricalCrossEntropyWithLogits target must have data type INT32 or INT64 "
                            + "when ignoreIndex is present, but was " + targetType);
        }
        DataType ignoreType = ignoreIndex.dataType();
        if (ignoreType != DataType.INT32 && ignoreType != DataType.INT64) {
            throw new IllegalArgumentException(
                    "ignoreIndex must have data type INT32 or INT64, but was " + ignoreType);
        }
        if (targetType != ignoreType) {
            throw new IllegalArgumentException(
                    "categoricalCrossEntropyWithLogits ignoreIndex data type must equal target "
                            + "data type: target=" + targetType + ", ignoreIndex=" + ignoreType);
        }
        return indexCategoricalCrossEntropyWithLogits(
                logits, target, classAxis, reduction, Optional.of(ignoreIndex), logitsType);
    }

    private static Tensor denseCategoricalCrossEntropyWithLogits(
            Tensor logits,
            Tensor target,
            int classAxis,
            LossReduction reduction,
            DataType logitsType,
            DataType targetType) {
        Shape logitsShape = logits.descriptor().shape();
        Shape targetShape = target.descriptor().shape();
        int normalizedAxis = logitsShape.normalizeAxis(classAxis);
        validateExactShape(
                logitsShape,
                targetShape,
                "categoricalCrossEntropyWithLogits",
                "logits");
        validateClassExtent(logitsShape, normalizedAxis);
        DataType resultType = DataTypePromotion.promoteFloating(logitsType, targetType);
        DenseCategoricalCrossEntropyWithLogitsAttrs attrs =
                new DenseCategoricalCrossEntropyWithLogitsAttrs(normalizedAxis, reduction);
        Shape resultShape = reduction == LossReduction.NONE
                ? removeAxis(logitsShape, normalizedAxis)
                : Shape.scalar();
        boolean requiresGrad = logits.descriptor().requiresGrad()
                || target.descriptor().requiresGrad();
        TensorDescriptor descriptor = new TensorDescriptor(
                resultType, resultShape, Optional.empty(), requiresGrad);
        Operation operation = new Operation(
                LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(logits, target));
    }

    private static Tensor indexCategoricalCrossEntropyWithLogits(
            Tensor logits,
            Tensor target,
            int classAxis,
            LossReduction reduction,
            Optional<ScalarValue> ignoreIndex,
            DataType logitsType) {
        Shape logitsShape = logits.descriptor().shape();
        Shape targetShape = target.descriptor().shape();
        int normalizedAxis = logitsShape.normalizeAxis(classAxis);
        validateIndexTargetShape(logitsShape, targetShape, normalizedAxis);
        if (ignoreIndex.isEmpty()) {
            validateClassExtent(logitsShape, normalizedAxis);
        }
        IndexCategoricalCrossEntropyWithLogitsAttrs attrs =
                new IndexCategoricalCrossEntropyWithLogitsAttrs(
                        normalizedAxis, reduction, ignoreIndex);
        Shape resultShape = reduction == LossReduction.NONE ? targetShape : Shape.scalar();
        TensorDescriptor descriptor = new TensorDescriptor(
                logitsType,
                resultShape,
                Optional.empty(),
                logits.descriptor().requiresGrad());
        Operation operation = new Operation(
                LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(logits, target));
    }

    private static DataType requireFloating(
            Tensor tensor, String operationName, String role) {
        DataType dataType = tensor.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    operationName + " " + role
                            + " must have a floating data type, but was " + dataType);
        }
        return dataType;
    }

    private static void validateExactShape(
            Shape inputShape,
            Shape targetShape,
            String operationName,
            String inputRole) {
        int inputRank = inputShape.rank();
        int targetRank = targetShape.rank();
        if (inputRank != targetRank) {
            throw new IllegalArgumentException(
                    operationName + " target rank must equal " + inputRole + " rank: "
                            + inputRole + "=" + inputRank + ", target=" + targetRank);
        }
        for (int axis = 0; axis < inputRank; axis++) {
            Dimension inputDimension = inputShape.dimension(axis);
            Dimension targetDimension = targetShape.dimension(axis);
            if (!inputDimension.equals(targetDimension)
                    && inputDimension instanceof StaticDimension
                    && targetDimension instanceof StaticDimension) {
                throw new IllegalArgumentException(
                        operationName + " target dimension mismatch at axis " + axis
                                + ": " + inputRole + "=" + inputDimension
                                + ", target=" + targetDimension);
            }
        }
    }

    private static void validateClassExtent(Shape logitsShape, int classAxis) {
        Dimension classDimension = logitsShape.dimension(classAxis);
        if (!(classDimension instanceof StaticDimension staticClass)
                || staticClass.size() != 0) {
            return;
        }

        boolean sampleDomainDefinitelyEmpty = false;
        boolean sampleDomainFullyKnown = true;
        for (int axis = 0; axis < logitsShape.rank(); axis++) {
            if (axis == classAxis) {
                continue;
            }
            Dimension dimension = logitsShape.dimension(axis);
            if (dimension instanceof StaticDimension staticDimension) {
                if (staticDimension.size() == 0) {
                    sampleDomainDefinitelyEmpty = true;
                }
            } else {
                sampleDomainFullyKnown = false;
            }
        }
        if (!sampleDomainDefinitelyEmpty && sampleDomainFullyKnown) {
            throw new IllegalArgumentException(
                    "categoricalCrossEntropyWithLogits class dimension must be positive when "
                            + "sample domain is non-empty: axis=" + classAxis
                            + ", dimension=" + classDimension);
        }
    }

    private static void validateIndexTargetShape(
            Shape logitsShape, Shape targetShape, int classAxis) {
        int logitsRank = logitsShape.rank();
        int targetRank = targetShape.rank();
        if (targetRank != logitsRank - 1) {
            throw new IllegalArgumentException(
                    "categoricalCrossEntropyWithLogits index target rank must equal logits rank "
                            + "minus one: logits=" + logitsRank + ", target=" + targetRank);
        }
        for (int targetAxis = 0; targetAxis < targetRank; targetAxis++) {
            int logitsAxis = targetAxis < classAxis ? targetAxis : targetAxis + 1;
            Dimension logitsDimension = logitsShape.dimension(logitsAxis);
            Dimension targetDimension = targetShape.dimension(targetAxis);
            if (!logitsDimension.equals(targetDimension)
                    && logitsDimension instanceof StaticDimension
                    && targetDimension instanceof StaticDimension) {
                throw new IllegalArgumentException(
                        "categoricalCrossEntropyWithLogits index target dimension mismatch at "
                                + "target axis " + targetAxis + " (logits axis " + logitsAxis
                                + "): logits=" + logitsDimension + ", target=" + targetDimension);
            }
        }
    }

    private static Shape removeAxis(Shape shape, int removedAxis) {
        Dimension[] dimensions = new Dimension[shape.rank() - 1];
        int resultIndex = 0;
        for (int axis = 0; axis < shape.rank(); axis++) {
            if (axis != removedAxis) {
                dimensions[resultIndex++] = shape.dimension(axis);
            }
        }
        return Shape.ofDimensions(dimensions);
    }
}
