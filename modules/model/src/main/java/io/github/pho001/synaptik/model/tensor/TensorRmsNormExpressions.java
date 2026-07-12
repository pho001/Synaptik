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
 * no centering, correction, bias, saved output, hidden state, or eager evaluation. BFLOAT16 and
 * FLOAT32 results accumulate squares and sums in FLOAT32; FLOAT64 results use FLOAT64. Empty,
 * NaN, infinity, signed-zero, overflow, reassociation, and rounding policies remain semantic
 * constraints for later conforming execution rather than algorithms selected here.</p>
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
     * @param epsilon non-null finite positive floating scalar with the exact input data type
     * @return fresh unlabeled, storage-free, layout-unresolved result retaining exact input Shape,
     *     type, and gradient eligibility, with sole-input provenance at output index zero; never
     *     {@code null}
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if input is non-floating, normalized Shape is empty or is
     *     statically incompatible with trailing input axes, or epsilon is invalid or not exactly
     *     input-typed
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor apply(Tensor input, Shape normalizedShape, ScalarValue epsilon) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(normalizedShape, "normalizedShape");
        Objects.requireNonNull(epsilon, "epsilon");
        DataType resultType = requireFloating(input, "input");
        Shape inputShape = input.descriptor().shape();
        validateNormalizedShape(inputShape, normalizedShape);
        validateEpsilonType(epsilon, resultType);

        RmsNormAttrs attrs = new RmsNormAttrs(normalizedShape, epsilon);
        TensorDescriptor descriptor = new TensorDescriptor(
                resultType, inputShape, Optional.empty(), input.descriptor().requiresGrad());
        Operation operation = new Operation(RmsNormKind.RMS_NORM, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(input));
    }

    /**
     * Creates one RMS-normalization expression followed by explicit elementwise scale.
     *
     * @param input non-null floating input retained as first producer input
     * @param normalizedShape non-null positive-rank Shape matched to trailing input axes
     * @param scale non-null floating scale with Shape exactly equal to normalized Shape
     * @param epsilon non-null finite positive floating scalar with exact promoted result type
     * @return fresh unlabeled, storage-free, layout-unresolved result retaining exact input Shape,
     *     promoted type, combined gradient eligibility, and ordered {@code [input, scale]}
     *     provenance at output index zero; never {@code null}
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if an operand is non-floating, normalized Shape is empty or
     *     statically incompatible with trailing input axes, scale Shape is not exactly normalized
     *     Shape, or epsilon is invalid or differs from the promoted result type
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    static Tensor apply(
            Tensor input, Shape normalizedShape, Tensor scale, ScalarValue epsilon) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(normalizedShape, "normalizedShape");
        Objects.requireNonNull(scale, "scale");
        Objects.requireNonNull(epsilon, "epsilon");

        DataType inputType = requireFloating(input, "input");
        DataType scaleType = requireFloating(scale, "scale");
        Shape inputShape = input.descriptor().shape();
        validateNormalizedShape(inputShape, normalizedShape);
        validateScaleShape(scale.descriptor().shape(), normalizedShape);
        DataType resultType = DataTypePromotion.promoteFloating(inputType, scaleType);
        validateEpsilonType(epsilon, resultType);

        RmsNormAttrs attrs = new RmsNormAttrs(normalizedShape, epsilon);
        boolean requiresGrad = input.descriptor().requiresGrad()
                || scale.descriptor().requiresGrad();
        TensorDescriptor descriptor = new TensorDescriptor(
                resultType, inputShape, Optional.empty(), requiresGrad);
        Operation operation = new Operation(RmsNormKind.RMS_NORM, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(input, scale));
    }

    private static DataType requireFloating(Tensor tensor, String role) {
        DataType dataType = tensor.descriptor().dataType();
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
        if (epsilon.dataType() != resultType) {
            throw new IllegalArgumentException(
                    "rmsNorm epsilon data type must match result data type: epsilon="
                            + epsilon.dataType() + ", result=" + resultType);
        }
    }
}
