package io.github.pho001.synaptik.model.operation.normalization;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries the normalized channel axis and exact transition scalars for batch-normalization
 * training.
 *
 * <p>Momentum is the new-batch weight in the explicit running-statistic transition. Epsilon is
 * added only to biased batch variance inside the saved inverse-standard-deviation square root.
 * Both exact immutable scalar references are retained. This record owns no Tensor, running state,
 * training session, compiler saved-value lifetime, backend, runtime, or execution behavior.</p>
 *
 * @param channelAxis normalized non-negative logical channel axis
 * @param momentum non-null exact finite floating new-batch weight in {@code [0, 1]}
 * @param epsilon non-null exact finite strictly positive floating stabilizer
 */
public record BatchNormTrainingAttrs(
        int channelAxis, ScalarValue momentum, ScalarValue epsilon) implements OperationAttrs {
    /**
     * Creates validated batch-normalization training parameters.
     *
     * @param channelAxis normalized non-negative logical channel axis
     * @param momentum non-null exact finite floating new-batch weight in {@code [0, 1]}
     * @param epsilon non-null exact finite strictly positive floating stabilizer
     * @throws IllegalArgumentException if the channel axis is negative, momentum is non-floating,
     *     non-finite, or outside {@code [0, 1]}, or epsilon is non-floating, non-finite,
     *     non-positive, or either signed zero
     * @throws NullPointerException if momentum or epsilon is null, checked in that order
     */
    public BatchNormTrainingAttrs {
        if (channelAxis < 0) {
            throw new IllegalArgumentException(
                    "channelAxis must be non-negative: " + channelAxis);
        }
        momentum = Objects.requireNonNull(momentum, "momentum");
        requireFloating(momentum, "momentum");
        if (!isFiniteInClosedUnitInterval(momentum)) {
            throw new IllegalArgumentException(
                    "momentum must be finite and in [0, 1]: " + momentum);
        }
        epsilon = Objects.requireNonNull(epsilon, "epsilon");
        requireFloating(epsilon, "epsilon");
        if (!isFinitePositive(epsilon)) {
            throw new IllegalArgumentException(
                    "epsilon must be finite and positive: " + epsilon);
        }
    }

    private static void requireFloating(ScalarValue value, String role) {
        if (!value.dataType().isFloating()) {
            throw new IllegalArgumentException(
                    role + " must have a floating data type, but was " + value.dataType());
        }
    }

    private static boolean isFiniteInClosedUnitInterval(ScalarValue value) {
        return switch (value.dataType()) {
            case BFLOAT16 -> {
                float number = BFloat16Bits.toFloat(value.bfloat16Bits());
                yield Float.isFinite(number) && number >= 0.0f && number <= 1.0f;
            }
            case FLOAT32 -> {
                float number = value.float32Value();
                yield Float.isFinite(number) && number >= 0.0f && number <= 1.0f;
            }
            case FLOAT64 -> {
                double number = value.float64Value();
                yield Double.isFinite(number) && number >= 0.0d && number <= 1.0d;
            }
            default -> false;
        };
    }

    private static boolean isFinitePositive(ScalarValue value) {
        return switch (value.dataType()) {
            case BFLOAT16 -> {
                float number = BFloat16Bits.toFloat(value.bfloat16Bits());
                yield Float.isFinite(number) && number > 0.0f;
            }
            case FLOAT32 -> {
                float number = value.float32Value();
                yield Float.isFinite(number) && number > 0.0f;
            }
            case FLOAT64 -> {
                double number = value.float64Value();
                yield Double.isFinite(number) && number > 0.0d;
            }
            default -> false;
        };
    }
}
