package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.ArrayList;
import java.util.List;

/**
 * Builds mean-squared-error and categorical-cross-entropy first-order formulas.
 *
 * <p>Reduction cotangents are restored by expanding logical typed ones and reducing them rather
 * than converting a host element count. Dense targets retain their exact mass. Index targets are
 * non-differentiable, are sanitized only to make one-hot construction total for ignored rows, and
 * use an exact integral ignore predicate before exceptional logits can enter arithmetic.</p>
 */
final class LossGradientRules {
    /**
     * Prevents construction of this stateless formula owner.
     */
    private LossGradientRules() {}

    /**
     * Builds selected loss operand cotangents.
     *
     * @param producer non-null exact original loss occurrence
     * @param gradient non-null accumulated cotangent for its sole output
     * @param selectedInputs non-null input-position-aligned selected-route flags; observed but
     *     not mutated
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new input-position-aligned array of cotangents, with {@code null} for unselected
     *     roles
     */
    static Tensor[] apply(
            TensorProducer producer,
            Tensor gradient,
            boolean[] selectedInputs,
            FirstOrderAutograd.DerivativeConstants constants) {
        return switch ((LossKind) producer.operation().kind()) {
            case MEAN_SQUARED_ERROR ->
                    meanSquaredError(producer, gradient, selectedInputs, constants);
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ->
                    denseCategorical(producer, gradient, selectedInputs, constants);
            case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ->
                    indexCategorical(producer, gradient, selectedInputs, constants);
        };
    }

    /**
     * Builds prediction and selected target cotangents for mean-squared error.
     *
     * @param producer non-null exact original mean-squared-error occurrence
     * @param gradient non-null accumulated output cotangent
     * @param selectedInputs non-null input-position-aligned selected-route flags
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new two-position array of selected normalized cotangents
     */
    private static Tensor[] meanSquaredError(
            TensorProducer producer,
            Tensor gradient,
            boolean[] selectedInputs,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor predictionInput = producer.inputs().get(0);
        Tensor targetInput = producer.inputs().get(1);
        DataType resultType = producer.output(0).descriptor().dataType();
        Tensor difference = cast(predictionInput, resultType)
                .sub(cast(targetInput, resultType));
        LossReduction reduction =
                ((MeanSquaredErrorAttrs) producer.operation().attrs()).reduction();
        Tensor restored = restore(
                gradient, predictionInput.descriptor().shape(), reduction, constants);
        Tensor scaled = difference.mul(constants.two(resultType)).mul(restored);
        Tensor[] result = new Tensor[2];
        if (selectedInputs[0]) {
            result[0] = normalize(scaled, predictionInput);
        }
        if (selectedInputs[1]) {
            result[1] = normalize(
                    difference.mul(constants.negativeTwo(resultType)).mul(restored),
                    targetInput);
        }
        return result;
    }

    /**
     * Builds logits and selected dense-target cotangents while retaining the target's exact mass.
     *
     * @param producer non-null exact original dense categorical-loss occurrence
     * @param gradient non-null accumulated output cotangent
     * @param selectedInputs non-null input-position-aligned selected-route flags
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new two-position array of selected normalized cotangents
     */
    private static Tensor[] denseCategorical(
            TensorProducer producer,
            Tensor gradient,
            boolean[] selectedInputs,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor logitsInput = producer.inputs().get(0);
        Tensor targetInput = producer.inputs().get(1);
        DataType resultType = producer.output(0).descriptor().dataType();
        Tensor logits = cast(logitsInput, resultType);
        Tensor target = cast(targetInput, resultType);
        DenseCategoricalCrossEntropyWithLogitsAttrs attrs =
                (DenseCategoricalCrossEntropyWithLogitsAttrs) producer.operation().attrs();
        Shape sampleShape = removeAxis(logits.descriptor().shape(), attrs.axis());
        Tensor restored = restore(
                        gradient, sampleShape, attrs.reduction(), constants)
                .expandDims(attrs.axis());
        Tensor[] result = new Tensor[2];
        if (selectedInputs[0]) {
            Tensor targetMass = target.sum(attrs.axis(), true);
            result[0] = normalize(
                    logits.softmax(attrs.axis()).mul(targetMass).sub(target).mul(restored),
                    logitsInput);
        }
        if (selectedInputs[1]) {
            Tensor logProbabilities = logits.logSoftmax(attrs.axis());
            Tensor raw = logProbabilities.neg().mul(restored);
            Tensor excluded = target.equalTo(constants.zeroLike(target))
                    .logicalAnd(logProbabilities.isFinite().logicalNot());
            result[1] = normalize(
                    Tensor.where(excluded, constants.zeroLike(raw), raw),
                    targetInput);
        }
        return result;
    }

    /**
     * Builds the logits cotangent for an index-target categorical loss.
     *
     * @param producer non-null exact original index categorical-loss occurrence
     * @param gradient non-null accumulated output cotangent
     * @param selectedInputs non-null input-position-aligned selected-route flags
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new two-position array containing the selected logits cotangent and a
     *     non-differentiable {@code null} target role
     */
    private static Tensor[] indexCategorical(
            TensorProducer producer,
            Tensor gradient,
            boolean[] selectedInputs,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor logits = producer.inputs().get(0);
        Tensor target = producer.inputs().get(1);
        IndexCategoricalCrossEntropyWithLogitsAttrs attrs =
                (IndexCategoricalCrossEntropyWithLogitsAttrs) producer.operation().attrs();
        long depth = ((StaticDimension)
                logits.descriptor().shape().dimension(attrs.axis())).size();
        ScalarValue zeroIndex = integral(target.descriptor().dataType(), 0);
        long upperValue = target.descriptor().dataType() == DataType.INT32
                ? Math.min(depth - 1, Integer.MAX_VALUE)
                : depth - 1;
        ScalarValue upper = integral(target.descriptor().dataType(), upperValue);
        Tensor safeTarget = target.maximum(zeroIndex).minimum(upper);
        Tensor hotBoolean = insertClassAxis(safeTarget.oneHot(depth), attrs.axis());
        Tensor hot = Tensor.where(
                hotBoolean, constants.oneLike(logits), constants.zeroLike(logits));

        Shape sampleShape = target.descriptor().shape();
        Tensor scale;
        Tensor ignored = null;
        if (attrs.ignoreIndex().isPresent()) {
            ScalarValue ignore = attrs.ignoreIndex().orElseThrow();
            Tensor delta = target.sub(ignore);
            ignored = delta.equalTo(delta.sub(delta));
            Tensor active = Tensor.where(
                    ignored,
                    constants.zeroBase(logits.descriptor().dataType()).expand(sampleShape),
                    constants.oneBase(logits.descriptor().dataType()).expand(sampleShape));
            scale = gradient.descriptor().shape().equals(sampleShape)
                    ? gradient
                    : gradient.expand(sampleShape);
            if (attrs.reduction() == LossReduction.MEAN) {
                scale = scale.div(active.sum());
            }
        } else {
            scale = restore(gradient, sampleShape, attrs.reduction(), constants);
        }
        scale = scale.expandDims(attrs.axis());
        Tensor raw = logits.softmax(attrs.axis()).sub(hot).mul(scale);
        Tensor contribution = ignored == null
                ? raw
                : Tensor.where(
                        ignored.expandDims(attrs.axis()),
                        constants.zeroLike(logits),
                        raw);
        Tensor[] result = new Tensor[2];
        if (selectedInputs[0]) {
            result[0] = normalize(contribution, logits);
        }
        return result;
    }

    /**
     * Restores one reduced loss cotangent to its logical unreduced domain.
     *
     * @param gradient non-null accumulated reduced cotangent
     * @param domain non-null exact unreduced logical Shape
     * @param reduction non-null loss reduction policy
     * @param constants non-null request-local exact floating logical-splat owner
     * @return {@code gradient} for {@code NONE}; otherwise a new expanded expression, divided by
     *     a logical typed element count for {@code MEAN}
     */
    private static Tensor restore(
            Tensor gradient,
            Shape domain,
            LossReduction reduction,
            FirstOrderAutograd.DerivativeConstants constants) {
        if (reduction == LossReduction.NONE) {
            return gradient;
        }
        Tensor restored = gradient.expand(domain);
        if (reduction == LossReduction.MEAN) {
            Tensor count = constants.oneBase(gradient.descriptor().dataType())
                    .expand(domain)
                    .sum();
            restored = restored.div(count);
        }
        return restored;
    }

    /**
     * Moves the final one-hot depth axis to the configured class-axis position.
     *
     * @param oneHot non-null one-hot Tensor whose final axis is class depth
     * @param classAxis normalized target class-axis position
     * @return {@code oneHot} when the class axis is already final; otherwise a new permutation
     */
    private static Tensor insertClassAxis(Tensor oneHot, int classAxis) {
        int rank = oneHot.descriptor().shape().rank();
        if (classAxis == rank - 1) {
            return oneHot;
        }
        int[] axes = new int[rank];
        int sourceSampleAxis = 0;
        for (int outputAxis = 0; outputAxis < rank; outputAxis++) {
            axes[outputAxis] = outputAxis == classAxis
                    ? rank - 1
                    : sourceSampleAxis++;
        }
        return oneHot.permute(axes);
    }

    /**
     * Removes one normalized axis while preserving all remaining Dimension references.
     *
     * @param shape non-null source Shape
     * @param axis normalized axis to remove
     * @return a new Shape containing the remaining exact Dimension references
     */
    private static Shape removeAxis(Shape shape, int axis) {
        List<Dimension> dimensions = new ArrayList<>(shape.dimensions());
        dimensions.remove(axis);
        return Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
    }

    /**
     * Returns exact signed-integral scalar metadata for an index-target operation.
     *
     * @param dataType non-null INT32 or INT64 type
     * @param value signed value representable by the selected type
     * @return exact integral scalar metadata
     * @throws IllegalArgumentException if {@code dataType} is not signed integral
     */
    private static ScalarValue integral(DataType dataType, long value) {
        return switch (dataType) {
            case INT32 -> ScalarValue.int32((int) value);
            case INT64 -> ScalarValue.int64(value);
            case BFLOAT16, FLOAT32, FLOAT64, BOOL ->
                    throw new IllegalArgumentException("index target must be signed integral");
        };
    }

    /**
     * Converts an operand to the requested promoted result type only when required.
     *
     * @param input non-null floating operand
     * @param dataType non-null floating result type
     * @return {@code input} when its type already matches; otherwise a new cast expression
     */
    private static Tensor cast(Tensor input, DataType dataType) {
        return input.descriptor().dataType() == dataType ? input : input.cast(dataType);
    }

    /**
     * Restores one selected loss cotangent to an original input's exact Shape and data type.
     *
     * @param gradient non-null selected cotangent
     * @param input non-null original input whose descriptor is restored
     * @return a non-null ordinary Tensor expression with the input's exact descriptor
     */
    private static Tensor normalize(Tensor gradient, Tensor input) {
        Tensor result = gradient.descriptor().shape().equals(input.descriptor().shape())
                ? gradient
                : gradient.sumToShape(input.descriptor().shape());
        return result.descriptor().dataType() == input.descriptor().dataType()
                ? result
                : result.cast(input.descriptor().dataType());
    }
}
