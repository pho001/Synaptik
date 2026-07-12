package io.github.pho001.synaptik.model.operation.normalization;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Objects;

/**
 * Carries the exact trailing normalized Shape and positive typed epsilon for RMS normalization.
 *
 * <p>The normalized Shape has positive rank and identifies exact trailing input axes. Epsilon is
 * a finite, strictly positive BFLOAT16, FLOAT32, or FLOAT64 value added to the uncentered mean
 * square inside the square root. Input cardinality identifies whether explicit scale is present.
 * This record retains both immutable references unchanged and owns no operand, statistic,
 * gradient, compiler, backend, runtime, or execution state.</p>
 *
 * @param normalizedShape non-null positive-rank Shape corresponding to exact trailing input axes;
 *     retained unchanged
 * @param epsilon non-null exact finite floating value strictly greater than zero;
 *     retained unchanged
 */
public record RmsNormAttrs(Shape normalizedShape, ScalarValue epsilon) implements OperationAttrs {
    /**
     * Creates validated RMS-normalization parameters shared by the one- and two-input forms.
     *
     * @param normalizedShape non-null Shape with positive rank; retained unchanged
     * @param epsilon non-null floating, finite, strictly positive typed value; retained unchanged
     * @throws NullPointerException if {@code normalizedShape} or {@code epsilon} is null, checked
     *     in declaration order
     * @throws IllegalArgumentException if {@code normalizedShape} has rank zero, or if
     *     {@code epsilon} is non-floating, non-finite, negative, or either signed zero
     */
    public RmsNormAttrs {
        normalizedShape = Objects.requireNonNull(normalizedShape, "normalizedShape");
        epsilon = Objects.requireNonNull(epsilon, "epsilon");
        if (normalizedShape.rank() == 0) {
            throw new IllegalArgumentException("normalizedShape rank must be positive");
        }
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
