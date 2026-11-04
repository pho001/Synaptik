package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free trailing-Shape RMS-normalization expressions.
 *
 * <p>Each non-empty slice has the semantic formula
 * {@code x / sqrt(sum(x * x) / N + epsilon)}, optionally followed by elementwise scale. There is
 * no centering, correction, bias, saved output, hidden state, or eager evaluation. BFLOAT16,
 * FLOAT16, OFP8, and FLOAT32 results accumulate squares and sums in FLOAT32; FLOAT64 results use
 * FLOAT64. FLOAT16 and OFP8 result metadata uses an exact FLOAT32 epsilon. Empty, NaN, infinity,
 * signed-zero, overflow, reassociation, and rounding policies remain semantic constraints for
 * later conforming execution rather than algorithms selected here.</p>
 *
 * <p>Every successful request delegates exactly once to the derived-Tensor factory and produces
 * one fresh unlabeled, storage-free, layout-unresolved output with the exact input Shape, ordered
 * input provenance, and output index zero. This helper creates no compiler, backend, runtime,
 * execution, gradient, or parameter state.</p>
 */
final class TensorRmsNormExpressions {
    private TensorRmsNormExpressions() {
    }

    /**
     * Creates one RMS-normalization expression without scale.
     *
     * @param input non-null floating input retained as sole producer input
     * @param normalizedShape non-null positive-rank Shape matched to trailing input axes
     * @param epsilon non-null finite positive scalar matching the input type, except FLOAT16 and OFP8 use FLOAT32
     * @return fresh unlabeled, storage-free, layout-unresolved result retaining exact input Shape,
     *     type, and gradient eligibility, with sole-input provenance at output index zero; never
     *     {@code null}
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if input is non-floating, normalized Shape is empty or is
     *     statically incompatible with trailing input axes, or epsilon does not match the input
     *     arithmetic type
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
     * @param epsilon non-null finite positive scalar matching the input type, except FLOAT16 and OFP8 use FLOAT32
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if input is non-floating, normalized Shape is empty or is
     *     statically incompatible with trailing input axes, or epsilon does not match the input
     *     arithmetic type
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

        RmsNormAttrs attrs = new RmsNormAttrs(normalizedShape, epsilon);
        TensorDescriptor descriptor = new TensorDescriptor(
                resultType, inputShape, Optional.empty(), inputDescriptor.requiresGrad());
        Operation operation = new Operation(RmsNormKind.RMS_NORM, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(input));
    }

    /**
     * Creates one RMS-normalization expression followed by explicit elementwise scale.
     *
     * @param input non-null floating input retained as first producer input
     * @param normalizedShape non-null positive-rank Shape matched to trailing input axes
     * @param scale non-null floating scale with Shape exactly equal to normalized Shape
     * @param epsilon non-null finite positive scalar matching the result type, except FLOAT16 and OFP8 use FLOAT32
     * @return fresh unlabeled, storage-free, layout-unresolved result retaining exact input Shape,
     *     promoted type, combined gradient eligibility, and ordered {@code [input, scale]}
     *     provenance at output index zero; never {@code null}
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if an operand is non-floating, normalized Shape is empty or
     *     statically incompatible with trailing input axes, scale Shape is not exactly normalized
     *     Shape, or epsilon does not match the promoted result arithmetic type
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor apply(
            Tensor input, Shape normalizedShape, Tensor scale, ScalarValue epsilon) {
        return applyChecked(input, normalizedShape, scale, epsilon, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null floating input retained as first producer input
     * @param normalizedShape non-null positive-rank Shape matched to trailing input axes
     * @param scale non-null floating scale with Shape exactly equal to normalized Shape
     * @param epsilon non-null finite positive scalar matching the result type, except FLOAT16 and OFP8 use FLOAT32
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if an operand is non-floating, normalized Shape is empty or
     *     statically incompatible with trailing input axes, scale Shape is not exactly normalized
     *     Shape, or epsilon does not match the promoted result arithmetic type
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor applyChecked(
            Tensor input, Shape normalizedShape, Tensor scale, ScalarValue epsilon,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(normalizedShape, "normalizedShape");
        Objects.requireNonNull(scale, "scale");
        Objects.requireNonNull(epsilon, "epsilon");
        return applyCore(input.descriptor(), normalizedShape, scale.descriptor(), epsilon,
                input, scale, descriptorCheck);
    }

    private static Tensor applyCore(
            TensorDescriptor inputDescriptor, Shape normalizedShape,
            TensorDescriptor scaleDescriptor, ScalarValue epsilon, Tensor input, Tensor scale,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {

        DataType inputType = requireFloating(inputDescriptor, "input");
        DataType scaleType = requireFloating(scaleDescriptor, "scale");
        Shape inputShape = inputDescriptor.shape();
        validateNormalizedShape(inputShape, normalizedShape);
        validateScaleShape(scaleDescriptor.shape(), normalizedShape);
        DataType resultType = DataTypePromotion.promoteFloating(inputType, scaleType);
        validateEpsilonType(epsilon, resultType);

        RmsNormAttrs attrs = new RmsNormAttrs(normalizedShape, epsilon);
        boolean requiresGrad = inputDescriptor.requiresGrad()
                || scaleDescriptor.requiresGrad();
        TensorDescriptor descriptor = new TensorDescriptor(
                resultType, inputShape, Optional.empty(), requiresGrad);
        Operation operation = new Operation(RmsNormKind.RMS_NORM, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(input, scale));
    }

    /**
     * Describes canonical RMS-normalization metadata through the construction guards.
     * The sink is invoked synchronously exactly once without creating Tensor identities.
     */
    static void describeChecked(
            Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        RmsNormAttrs attrs = (RmsNormAttrs) operation.attrs();
        if (inputs.size() == 1) {
            applyCore(inputs.get(0), attrs.normalizedShape(), attrs.epsilon(), null, sink);
        } else {
            applyCore(inputs.get(0), attrs.normalizedShape(), inputs.get(1), attrs.epsilon(),
                    null, null, sink);
        }
    }

    private static DataType requireFloating(TensorDescriptor descriptor, String role) {
        DataType dataType = descriptor.dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "rmsNorm " + role + " must have a floating data type, but was " + dataType);
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
                    "rmsNorm normalized rank must not exceed input rank: normalizedRank="
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
                        "rmsNorm normalized dimension mismatch at normalized axis "
                                + normalizedAxis + ": input=" + inputDimension
                                + ", normalized=" + normalizedDimension);
            }
        }
    }

    private static void validateScaleShape(Shape scaleShape, Shape normalizedShape) {
        if (!scaleShape.equals(normalizedShape)) {
            throw new IllegalArgumentException(
                    "rmsNorm scale Shape must equal normalizedShape: scale=" + scaleShape
                            + ", normalizedShape=" + normalizedShape);
        }
    }

    private static void validateEpsilonType(ScalarValue epsilon, DataType resultType) {
        if (epsilon.dataType() != resultType
                && ((resultType != DataType.FLOAT16 && resultType != DataType.FLOAT8_E4M3
                                && resultType != DataType.FLOAT8_E5M2)
                        || epsilon.dataType() != DataType.FLOAT32)) {
            throw new IllegalArgumentException(
                    "rmsNorm epsilon data type must match result data type: epsilon="
                            + epsilon.dataType() + ", result=" + resultType);
        }
    }
}
