package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
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
     * Creates one fresh dense-target categorical-cross-entropy occurrence directly from logits.
     *
     * <p>Construction validates metadata only. It does not inspect target values, decompose the
     * stable target-weighted log-softmax meaning into public primitives, or define gradients,
     * compiler proof, lowering, execution, or training behavior.</p>
     *
     * <p>{@link LossReduction#NONE} removes the normalized class axis, while sum and mean return
     * scalar Shape; mean divides by the number of non-class groups. Rank-one logits therefore
     * have sample count one. A zero non-class extent gives empty none, positive-zero sum, and NaN
     * mean without evaluating a class slice. A statically zero class extent is valid for that
     * definitely empty sample domain, fails for a definitely non-empty domain, and otherwise
     * retains the later obligation {@code sampleCount == 0 || classExtent > 0}.</p>
     *
     * <p>Logits and target participate in current floating promotion. BFLOAT16 and FLOAT32 results
     * use FLOAT32 log-sum-exp, weighting, accumulation, and mean-division meaning; FLOAT64 results
     * use FLOAT64. Final values have the promoted result type. These are semantic computation
     * requirements, not eager evaluation or selection of a backend algorithm.</p>
     *
     * @param logits non-null floating logits retained at producer input position zero
     * @param target non-null floating exact-shape dense probability target retained at input one;
     *     its values remain a caller/execution obligation rather than construction validation
     * @param classAxis positive or negative logits class axis normalized exactly once
     * @param reduction non-null explicit none, sum, or sample-domain mean reduction
     * @return fresh unlabeled, storage-free loss tensor with promoted floating type, unresolved
     *     layout, combined gradient eligibility, selected Shape, and output-index-zero provenance
     * @throws NullPointerException if {@code logits}, {@code target}, or {@code reduction} is null,
     *     checked in that order
     * @throws IndexOutOfBoundsException if {@code classAxis} is outside the logits rank
     * @throws IllegalArgumentException if an input is not floating, target rank or static
     *     dimensions mismatch logits, or a statically empty class axis has a definitely non-empty
     *     sample domain
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor categoricalCrossEntropyWithLogits(
            Tensor logits, Tensor target, int classAxis, LossReduction reduction) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(reduction, "reduction");

        DataType logitsType = requireFloating(
                logits, "categoricalCrossEntropyWithLogits", "logits");
        DataType targetType = requireFloating(
                target, "categoricalCrossEntropyWithLogits", "target");
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
