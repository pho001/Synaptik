package io.github.pho001.synaptik.model.operation.normalization;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries the normalized channel axis and exact typed epsilon for batch-normalization inference.
 *
 * <p>The channel axis is already normalized to a non-negative logical position. Epsilon is a
 * finite, strictly positive BFLOAT16, FLOAT32, or FLOAT64 value added to the explicit running
 * variance inside the square root. Both values are immutable semantic metadata; this record owns
 * no Tensor, raw axis, statistic, training state, compiler, backend, runtime, or result.</p>
 *
 * @param channelAxis normalized non-negative logical channel axis
 * @param epsilon non-null exact finite positive floating value retained unchanged
 */
public record BatchNormInferenceAttrs(int channelAxis, ScalarValue epsilon)
        implements OperationAttrs {
    /**
     * Creates validated batch-normalization inference parameters.
     *
     * @param channelAxis normalized non-negative logical channel axis
     * @param epsilon non-null exact finite positive floating value retained unchanged
     * @throws IllegalArgumentException if {@code channelAxis} is negative, or epsilon is
     *     non-floating, non-finite, negative, or either signed zero
     * @throws NullPointerException if {@code epsilon} is null
     */
    public BatchNormInferenceAttrs {
        if (channelAxis < 0) {
            throw new IllegalArgumentException(
                    "channelAxis must be non-negative: " + channelAxis);
        }
        epsilon = Objects.requireNonNull(epsilon, "epsilon");
        DataType dataType = epsilon.dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "epsilon must have a floating data type, but was " + dataType);
        }
        boolean valid = switch (dataType) {
            case BFLOAT16 -> {
                float value = BFloat16Bits.toFloat(epsilon.bfloat16Bits());
                yield Float.isFinite(value) && value > 0.0f;
            }
            case FLOAT32 -> {
                float value = epsilon.float32Value();
                yield Float.isFinite(value) && value > 0.0f;
            }
            case FLOAT64 -> {
                double value = epsilon.float64Value();
                yield Double.isFinite(value) && value > 0.0d;
            }
            default -> false;
        };
        if (!valid) {
            throw new IllegalArgumentException("epsilon must be finite and positive: " + epsilon);
        }
    }
}
