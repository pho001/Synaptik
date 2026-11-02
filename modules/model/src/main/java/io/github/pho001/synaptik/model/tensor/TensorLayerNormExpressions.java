package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free trailing-Shape layer-normalization expressions.
 *
 * <p>The helper validates caller inputs before allocating result identity, derives an unresolved
 * one-output descriptor, and delegates exactly once to the derived-Tensor factory. It records
 * population-variance semantics only: BFLOAT16, FLOAT16, and FLOAT32 results use FLOAT32
 * accumulation, FLOAT64 results use FLOAT64 accumulation, and affine operands promote in input,
 * scale, bias order. FLOAT16 result metadata uses an exact FLOAT32 epsilon. Empty results contain
 * no normalized values. NaN, infinity, signed-zero, overflow, reassociation, and rounding behavior
 * are semantic constraints retained for later conforming
 * execution; this helper reads no values and selects no algorithm.</p>
 *
 * <p>Every successful request is fresh, unlabeled, storage-free, and layout-unresolved. It retains
 * the exact input Shape, records output index zero and exact ordered provenance, and creates no
 * saved mean, variance, inverse standard deviation, hidden output, gradient, or runtime state.</p>
 */
final class TensorLayerNormExpressions {
    private TensorLayerNormExpressions() {
    }

    /**
     * Creates one no-affine layer-normalization expression.
     *
     * @param input non-null floating input retained as sole producer input
     * @param normalizedShape non-null positive-rank Shape matched to trailing input axes
     * @param epsilon non-null finite positive scalar matching the input type, except FLOAT16 uses FLOAT32
     * @return fresh unlabeled, storage-free, layout-unresolved result retaining the exact input
     *     Shape, type, and gradient eligibility, with sole-input provenance at output index zero;
     *     never {@code null}
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if input is non-floating, normalized Shape is empty or is
     *     statically incompatible with the trailing input axes, or epsilon does not match the
     *     result arithmetic type
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor apply(Tensor input, Shape normalizedShape, ScalarValue epsilon) {
        return applyChecked(input, normalizedShape, epsilon, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null floating input retained as sole producer input
     * @param normalizedShape non-null positive-rank Shape matched to trailing input axes
     * @param epsilon non-null finite positive scalar matching the input type, except FLOAT16 uses FLOAT32
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if input is non-floating, normalized Shape is empty or is
     *     statically incompatible with the trailing input axes, or epsilon does not match the
     *     result arithmetic type
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor applyChecked(Tensor input, Shape normalizedShape, ScalarValue epsilon,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(normalizedShape, "normalizedShape");
        Objects.requireNonNull(epsilon, "epsilon");
        return applyCore(input.descriptor(), normalizedShape, epsilon, input, descriptorCheck);
    }

    private static Tensor applyCore(
            TensorDescriptor inputDescriptor, Shape normalizedShape, ScalarValue epsilon,
            Tensor input,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        DataType resultType = requireFloating(inputDescriptor, "input");
        Shape inputShape = inputDescriptor.shape();
        validateNormalizedShape(inputShape, normalizedShape);
        validateEpsilonType(epsilon, resultType);

        LayerNormAttrs attrs = new LayerNormAttrs(normalizedShape, epsilon);
        TensorDescriptor descriptor = new TensorDescriptor(
                resultType, inputShape, Optional.empty(), inputDescriptor.requiresGrad());
        Operation operation = new Operation(LayerNormKind.LAYER_NORM, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(input));
    }

    /**
     * Creates one explicit affine layer-normalization expression.
     *
     * @param input non-null floating input retained as first producer input
     * @param normalizedShape non-null positive-rank Shape matched to trailing input axes
     * @param scale non-null floating scale with Shape exactly equal to normalized Shape
     * @param bias non-null floating bias with Shape exactly equal to normalized Shape
     * @param epsilon non-null finite positive scalar matching the result type, except FLOAT16 uses FLOAT32
     * @return fresh unlabeled, storage-free, layout-unresolved result retaining the exact input
     *     Shape, promoted type, combined gradient eligibility, and ordered
     *     {@code [input, scale, bias]} provenance at output index zero; never {@code null}
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if an operand is non-floating, normalized Shape is empty or
     *     statically incompatible with trailing input axes, scale or bias Shape is not exactly the
     *     normalized Shape, or epsilon does not match the promoted result arithmetic type
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor apply(
            Tensor input,
            Shape normalizedShape,
            Tensor scale,
            Tensor bias,
            ScalarValue epsilon) {
        return applyChecked(input, normalizedShape, scale, bias, epsilon, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null floating input retained as first producer input
     * @param normalizedShape non-null positive-rank Shape matched to trailing input axes
     * @param scale non-null floating scale with Shape exactly equal to normalized Shape
     * @param bias non-null floating bias with Shape exactly equal to normalized Shape
     * @param epsilon non-null finite positive scalar matching the result type, except FLOAT16 uses FLOAT32
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if an operand is non-floating, normalized Shape is empty or
     *     statically incompatible with trailing input axes, scale or bias Shape is not exactly the
     *     normalized Shape, or epsilon does not match the promoted result arithmetic type
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor applyChecked(
            Tensor input,
            Shape normalizedShape,
            Tensor scale,
            Tensor bias,
            ScalarValue epsilon,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(normalizedShape, "normalizedShape");
        Objects.requireNonNull(scale, "scale");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(epsilon, "epsilon");
        return applyCore(input.descriptor(), normalizedShape, scale.descriptor(), bias.descriptor(),
                epsilon, input, scale, bias, descriptorCheck);
    }

    private static Tensor applyCore(
            TensorDescriptor inputDescriptor, Shape normalizedShape,
            TensorDescriptor scaleDescriptor, TensorDescriptor biasDescriptor, ScalarValue epsilon,
            Tensor input, Tensor scale, Tensor bias,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {

        DataType inputType = requireFloating(inputDescriptor, "input");
        DataType scaleType = requireFloating(scaleDescriptor, "scale");
        DataType biasType = requireFloating(biasDescriptor, "bias");
        Shape inputShape = inputDescriptor.shape();
        validateNormalizedShape(inputShape, normalizedShape);
        validateAffineShape(scaleDescriptor.shape(), normalizedShape, "scale");
        validateAffineShape(biasDescriptor.shape(), normalizedShape, "bias");
        DataType resultType = DataTypePromotion.promoteFloating(
                DataTypePromotion.promoteFloating(inputType, scaleType), biasType);
        validateEpsilonType(epsilon, resultType);

        AffineLayerNormAttrs attrs = new AffineLayerNormAttrs(normalizedShape, epsilon);
        boolean requiresGrad = inputDescriptor.requiresGrad()
                || scaleDescriptor.requiresGrad()
                || biasDescriptor.requiresGrad();
        TensorDescriptor descriptor = new TensorDescriptor(
                resultType, inputShape, Optional.empty(), requiresGrad);
        Operation operation = new Operation(LayerNormKind.LAYER_NORM, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(input, scale, bias));
    }

    /**
     * Describes canonical layer-normalization metadata through the same guards as construction.
     * The sink is invoked synchronously exactly once without creating Tensor identities.
     */
    static void describeChecked(
            Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        if (operation.attrs() instanceof LayerNormAttrs attrs) {
            applyCore(inputs.get(0), attrs.normalizedShape(), attrs.epsilon(), null, sink);
        } else {
            AffineLayerNormAttrs attrs = (AffineLayerNormAttrs) operation.attrs();
            applyCore(inputs.get(0), attrs.normalizedShape(), inputs.get(1), inputs.get(2),
                    attrs.epsilon(), null, null, null, sink);
        }
    }

    private static DataType requireFloating(TensorDescriptor descriptor, String role) {
        DataType dataType = descriptor.dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "layerNorm " + role + " must have a floating data type, but was " + dataType);
        }
        return dataType;
    }

    private static void validateNormalizedShape(Shape inputShape, Shape normalizedShape) {
        int normalizedRank = normalizedShape.rank();
        int inputRank = inputShape.rank();
        if (normalizedRank == 0) {
            throw new IllegalArgumentException("normalizedShape rank must be positive");
        }
        if (normalizedRank > inputRank) {
            throw new IllegalArgumentException(
                    "layerNorm normalized rank must not exceed input rank: normalizedRank="
                            + normalizedRank + ", inputRank=" + inputRank);
        }
        int firstInputAxis = inputRank - normalizedRank;
        for (int normalizedAxis = 0; normalizedAxis < normalizedRank; normalizedAxis++) {
            Dimension inputDimension = inputShape.dimension(firstInputAxis + normalizedAxis);
            Dimension normalizedDimension = normalizedShape.dimension(normalizedAxis);
            if (!inputDimension.equals(normalizedDimension)
                    && inputDimension instanceof StaticDimension
                    && normalizedDimension instanceof StaticDimension) {
                throw new IllegalArgumentException(
                        "layerNorm normalized dimension mismatch at normalized axis "
                                + normalizedAxis + ": input=" + inputDimension
                                + ", normalized=" + normalizedDimension);
            }
        }
    }

    private static void validateAffineShape(
            Shape operandShape, Shape normalizedShape, String role) {
        if (!operandShape.equals(normalizedShape)) {
            throw new IllegalArgumentException(
                    "layerNorm " + role + " Shape must equal normalizedShape: " + role + "="
                            + operandShape + ", normalizedShape=" + normalizedShape);
        }
    }

    private static void validateEpsilonType(ScalarValue epsilon, DataType resultType) {
        if (epsilon.dataType() != resultType
                && (resultType != DataType.FLOAT16
                        || epsilon.dataType() != DataType.FLOAT32)) {
            throw new IllegalArgumentException(
                    "layerNorm epsilon data type must match result data type: epsilon="
                            + epsilon.dataType() + ", result=" + resultType);
        }
    }
}
