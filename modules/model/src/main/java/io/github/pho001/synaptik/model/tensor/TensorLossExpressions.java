package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.Operation;
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

        DataType predictionType = requireFloating(prediction, "prediction");
        DataType targetType = requireFloating(target, "target");
        Shape predictionShape = prediction.descriptor().shape();
        Shape targetShape = target.descriptor().shape();
        validateExactShape(predictionShape, targetShape);
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

    private static DataType requireFloating(Tensor tensor, String role) {
        DataType dataType = tensor.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "meanSquaredError " + role
                            + " must have a floating data type, but was " + dataType);
        }
        return dataType;
    }

    private static void validateExactShape(Shape predictionShape, Shape targetShape) {
        int predictionRank = predictionShape.rank();
        int targetRank = targetShape.rank();
        if (predictionRank != targetRank) {
            throw new IllegalArgumentException(
                    "meanSquaredError target rank must equal prediction rank: prediction="
                            + predictionRank + ", target=" + targetRank);
        }
        for (int axis = 0; axis < predictionRank; axis++) {
            Dimension predictionDimension = predictionShape.dimension(axis);
            Dimension targetDimension = targetShape.dimension(axis);
            if (!predictionDimension.equals(targetDimension)
                    && predictionDimension instanceof StaticDimension
                    && targetDimension instanceof StaticDimension) {
                throw new IllegalArgumentException(
                        "meanSquaredError target dimension mismatch at axis " + axis
                                + ": prediction=" + predictionDimension
                                + ", target=" + targetDimension);
            }
        }
    }
}
