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
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Builds mean-squared-error and categorical-cross-entropy first-order formulas.
 *
 * <p>Reduction cotangents are restored by expanding logical typed ones and reducing them rather
 * than converting a host element count. Categorical formulas use their loss-specific max-shifted
 * exponentials, not first-class softmax's distinct mixed-infinity semantics. Their ordinary
 * arithmetic works in FLOAT32 unless the loss result is FLOAT64, with final input normalization.
 * Dense targets retain their exact mass; all-zero rows exclude logits before probability
 * arithmetic. Index targets are non-differentiable, are sanitized only to make one-hot
 * construction total for ignored rows, and use an exact integral ignore predicate before
 * exceptional logits can enter arithmetic.</p>
 *
 * <p>Descriptor access preserves original source roles without resolving primal values.
 * Real construction uses Tensor references directly; checked metadata construction retains
 * the same eager operation sites, including temporaries not used by a returned cotangent.</p>
 *
 * @param <V> direct Tensor or checked metadata operand representation
 */
final class LossGradientRules<V> {
    private final OrdinaryConstruction<V> ops;

    /**
     * Binds this formula owner to one compile-local construction route.
     * @param ops non-null direct Tensor or checked metadata construction
     */
    LossGradientRules(OrdinaryConstruction<V> ops) {
        this.ops = java.util.Objects.requireNonNull(ops);
    }

    /**
     * Builds selected loss operand cotangents.
     *
     * @param producer non-null exact original loss occurrence
     * @param gradient non-null accumulated cotangent for its sole output
     * @param selectedInputs non-null input-position-aligned selected-route flags; observed but
     *     not mutated
     * @param primal total lexical binding of original value operands
     * @return a new input-position-aligned array of cotangents, with {@code null} for unselected
     *     roles
     */
    V[] apply(OrdinaryConstruction.Source<V> producer, V gradient, boolean[] selectedInputs, Function<V, V> primal) {
            return switch ((LossKind) producer.operation().kind()) {
                case MEAN_SQUARED_ERROR ->
                        meanSquaredError(producer, gradient, selectedInputs, primal);
                case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ->
                        denseCategorical(producer, gradient, selectedInputs, primal);
                case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ->
                        indexCategorical(producer, gradient, selectedInputs, primal);
            };
    }

    /**
     * Builds prediction and selected target cotangents for mean-squared error.
     *
     * @param producer non-null exact original mean-squared-error occurrence
     * @param gradient non-null accumulated output cotangent
     * @param selectedInputs non-null input-position-aligned selected-route flags
     * @param primal total lexical binding of original value operands
     * @return a new two-position array of selected normalized cotangents
     */
    private V[] meanSquaredError(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        V predictionInput = producer.inputs().get(0);
        V targetInput = producer.inputs().get(1);
        DataType resultType = ops.descriptor(producer.output(0)).dataType();
        V difference = ops.sub(cast(primal.apply(predictionInput), resultType), cast(primal.apply(targetInput), resultType));
        LossReduction reduction =
                ((MeanSquaredErrorAttrs) producer.operation().attrs()).reduction();
        V restored = restore(
                gradient, ops.descriptor(predictionInput).shape(), reduction);
        V scaled = ops.mul(ops.mul(difference, ops.two(resultType)), restored);
        V[] result = ops.array(2);
        if (selectedInputs[0]) {
            result[0] = normalize(scaled, predictionInput);
        }
        if (selectedInputs[1]) {
            result[1] = normalize(
                    ops.mul(ops.mul(difference, ops.negativeTwo(resultType)), restored),
                    targetInput);
        }
        return result;
    }

    /**
     * Builds logits and selected dense-target cotangents while retaining the target's exact mass.
     * The target derivative retains the loss's maximum-addition rounding site and excludes only
     * exact zero targets with nonfinite log-probabilities; finite zero-target derivatives remain.
     *
     * @param producer non-null exact original dense categorical-loss occurrence
     * @param gradient non-null accumulated output cotangent
     * @param selectedInputs non-null input-position-aligned selected-route flags
     * @param primal total lexical binding of original value operands
     * @return a new two-position array of selected normalized cotangents
     */
    private V[] denseCategorical(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        V logitsInput = producer.inputs().get(0);
        V targetInput = producer.inputs().get(1);
        DataType workingType = ops.descriptor(producer.output(0)).dataType() == DataType.FLOAT64
                ? DataType.FLOAT64 : DataType.FLOAT32;
        V logits = cast(primal.apply(logitsInput), workingType);
        V target = cast(primal.apply(targetInput), workingType);
        DenseCategoricalCrossEntropyWithLogitsAttrs attrs =
                (DenseCategoricalCrossEntropyWithLogitsAttrs) producer.operation().attrs();
        Shape sampleShape = removeAxis(ops.descriptor(logits).shape(), attrs.axis());
        V restored = ops.expandDims(restore(
                        cast(gradient, workingType), sampleShape, attrs.reduction()), attrs.axis());
        V[] result = ops.array(2);
        if (selectedInputs[0]) {
            V zeroTarget = ops.equalTo(target, ops.zeroLike(target));
            V membership = ops.sum(
                    ops.where(zeroTarget, ops.zeroLike(target), ops.oneLike(target)),
                    attrs.axis(), true);
            V emptyRow = ops.equalTo(membership, ops.zeroLike(membership));
            V safeLogits = ops.where(emptyRow, ops.zeroLike(logits), logits);
            V probabilities = categoricalProbabilities(safeLogits, attrs.axis());
            V targetMass = ops.sum(target, attrs.axis(), true);
            V safeScale = ops.where(emptyRow, ops.zeroLike(restored), restored);
            V raw = ops.mul(ops.sub(ops.mul(probabilities, targetMass), target), safeScale);
            result[0] = normalize(
                    ops.where(emptyRow, ops.zeroLike(raw), raw), logitsInput);
        }
        if (selectedInputs[1]) {
            V maximum = ops.max(logits, attrs.axis(), true);
            V exponentials = ops.exp(ops.sub(logits, maximum));
            V logNormalizer = ops.add(maximum, ops.log(ops.sum(exponentials, attrs.axis(), true)));
            V logProbabilities = ops.sub(logits, logNormalizer);
            V excluded = ops.logicalAnd(ops.equalTo(target, ops.zeroLike(target)),
                    ops.logicalNot(ops.isFinite(logProbabilities)));
            V safeLogProbabilities = ops.where(excluded, ops.zeroLike(logProbabilities), logProbabilities);
            V safeScale = ops.where(excluded, ops.zeroLike(restored), restored);
            V raw = ops.mul(ops.neg(safeLogProbabilities), safeScale);
            result[1] = normalize(
                    ops.where(excluded, ops.zeroLike(raw), raw),
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
     * @param primal total lexical binding of original value operands
     * @return a new two-position array containing the selected logits cotangent and a
     *     non-differentiable {@code null} target role
     */
    private V[] indexCategorical(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        DataType workingType = ops.descriptor(producer.output(0)).dataType() == DataType.FLOAT64
                ? DataType.FLOAT64 : DataType.FLOAT32;
        V logits = cast(primal.apply(producer.inputs().get(0)), workingType);
        V target = primal.apply(producer.inputs().get(1));
        IndexCategoricalCrossEntropyWithLogitsAttrs attrs =
                (IndexCategoricalCrossEntropyWithLogitsAttrs) producer.operation().attrs();
        long depth = ((StaticDimension)
                ops.descriptor(logits).shape().dimension(attrs.axis())).size();
        ScalarValue zeroIndex = integral(ops.descriptor(target).dataType(), 0);
        long upperValue = ops.descriptor(target).dataType() == DataType.INT32
                ? Math.min(depth - 1, Integer.MAX_VALUE)
                : depth - 1;
        ScalarValue upper = integral(ops.descriptor(target).dataType(), upperValue);
        V safeTarget = ops.minimum(ops.maximum(target, zeroIndex), upper);
        V hotBoolean = insertClassAxis(ops.oneHot(safeTarget, depth), attrs.axis());
        V hot = ops.where(hotBoolean, ops.oneLike(logits), ops.zeroLike(logits));

        Shape sampleShape = ops.descriptor(target).shape();
        V scale;
        V ignored = null;
        if (attrs.ignoreIndex().isPresent()) {
            ScalarValue ignore = attrs.ignoreIndex().orElseThrow();
            V delta = ops.sub(target, ignore);
            ignored = ops.equalTo(delta, ops.sub(delta, delta));
            V active = ops.where(ignored, ops.expand(ops.zeroBase(ops.descriptor(logits).dataType()), sampleShape), ops.expand(ops.oneBase(ops.descriptor(logits).dataType()), sampleShape));
            V workingGradient = cast(gradient, workingType);
            scale = ops.descriptor(workingGradient).shape().equals(sampleShape)
                    ? workingGradient
                    : ops.expand(workingGradient, sampleShape);
            if (attrs.reduction() == LossReduction.MEAN) {
                scale = ops.div(scale, ops.sum(active));
            }
        } else {
            scale = restore(cast(gradient, workingType), sampleShape, attrs.reduction());
        }
        scale = ops.expandDims(scale, attrs.axis());
        V ignoredRow = ignored == null ? null : ops.expandDims(ignored, attrs.axis());
        V safeLogits = ignoredRow == null
                ? logits : ops.where(ignoredRow, ops.zeroLike(logits), logits);
        if (ignoredRow != null) {
            scale = ops.where(ignoredRow, ops.zeroLike(scale), scale);
        }
        V raw = ops.mul(ops.sub(categoricalProbabilities(safeLogits, attrs.axis()), hot), scale);
        V contribution = ignoredRow == null
                ? raw
                : ops.where(ignoredRow, ops.zeroLike(logits), raw);
        V[] result = ops.array(2);
        if (selectedInputs[0]) {
            result[0] = normalize(contribution, producer.inputs().get(0));
        }
        return result;
    }

    /**
     * Builds probabilities from the categorical loss's max-shifted exponential slice.
     * Callers apply exact target/ignore eligibility to logits before invoking this formula.
     * Unlike ordinary softmax, mixed-infinity slices always retain this shifted formula.
     *
     * @param logits non-null logits in the loss's FLOAT32 or FLOAT64 working type
     * @param axis normalized class axis
     * @return shifted exponentials divided by their all-contributors-once class sum
     */
    private V categoricalProbabilities(V logits, int axis) {
        V maximum = ops.max(logits, axis, true);
        V exponentials = ops.exp(ops.sub(logits, maximum));
        return ops.div(exponentials, ops.sum(exponentials, axis, true));
    }

    /**
     * Restores one reduced loss cotangent to its logical unreduced domain.
     *
     * @param gradient non-null accumulated reduced cotangent
     * @param domain non-null exact unreduced logical Shape
     * @param reduction non-null loss reduction policy
     * @return {@code gradient} for {@code NONE}; otherwise a new expanded expression, divided by
     *     a logical typed element count for {@code MEAN}
     */
    private V restore(
            V gradient,
            Shape domain,
            LossReduction reduction) {
        if (reduction == LossReduction.NONE) {
            return gradient;
        }
        V restored = ops.expand(gradient, domain);
        if (reduction == LossReduction.MEAN) {
            V count = ops.sum(ops.contiguous(ops.expand(ops.oneBase(ops.descriptor(gradient).dataType()), domain)));
            restored = ops.div(restored, count);
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
    private V insertClassAxis(V oneHot, int classAxis) {
        int rank = ops.descriptor(oneHot).shape().rank();
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
        return ops.permute(oneHot, axes);
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
            case BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2, FLOAT32, FLOAT64, BOOL ->
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
    private V cast(V input, DataType dataType) {
        return ops.descriptor(input).dataType() == dataType ? input : ops.cast(input, dataType);
    }

    /**
     * Restores one selected loss cotangent to an original input's exact Shape and data type.
     *
     * @param gradient non-null selected cotangent
     * @param input non-null original input whose descriptor is restored
     * @return a non-null ordinary Tensor expression with the input's exact descriptor
     */
    private V normalize(V gradient, V input) {
        V result = ops.descriptor(gradient).shape().equals(ops.descriptor(input).shape())
                ? gradient
                : ops.sumToShape(gradient, ops.descriptor(input).shape());
        return ops.descriptor(result).dataType() == ops.descriptor(input).dataType()
                ? result
                : ops.cast(result, ops.descriptor(input).dataType());
    }
}
