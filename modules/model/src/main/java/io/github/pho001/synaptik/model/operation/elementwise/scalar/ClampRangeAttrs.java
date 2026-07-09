package io.github.pho001.synaptik.model.operation.elementwise.scalar;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries the exact ordered inclusive bounds for a scalar range-clamp operation.
 *
 * <p>The lower bound precedes the upper bound. Both values must have the same numeric data type
 * and are retained by exact reference. Construction compares them with that type's primitive
 * {@code >} operation, converting only raw BFLOAT16 bits to binary32 for comparison. Equal
 * bounds, either ordering of signed zeros, ordered infinities, and one or two floating NaN
 * endpoints are accepted. INT64 comparison never passes through a floating representation.</p>
 *
 * <p>The record is immutable and owns no mutable input. Record-generated equality and hashing use
 * exact typed-bit value semantics, so signed floating zeros and distinct NaN payloads remain
 * distinct. The record does not match the bounds to an input Tensor, convert values, define clamp
 * result behavior, or provide a backend format.</p>
 *
 * @param minValue the non-null exact numeric inclusive lower bound, retained by reference
 * @param maxValue the non-null exact numeric inclusive upper bound of the same data type, retained
 *     by reference
 */
public record ClampRangeAttrs(ScalarValue minValue, ScalarValue maxValue) implements OperationAttrs {
    /**
     * Creates inclusive clamp-range attributes from exact lower and upper bounds.
     *
     * <p>Validation null-checks the lower then upper bound, requires equal data types, rejects
     * BOOL, and finally compares the exact represented numeric type. Equal values, either
     * signed-zero ordering, ordered infinities, and floating NaN endpoints are accepted.</p>
     *
     * @param minValue non-null exact numeric inclusive lower bound retained by reference
     * @param maxValue non-null exact numeric inclusive upper bound retained by reference
     * @throws NullPointerException if a bound is {@code null}, checked in parameter order with its
     *     name as the message
     * @throws IllegalArgumentException if the data types differ, either type is BOOL, or
     *     {@code minValue} is strictly greater than {@code maxValue} in their represented type
     */
    public ClampRangeAttrs {
        Objects.requireNonNull(minValue, "minValue");
        Objects.requireNonNull(maxValue, "maxValue");
        DataType dataType = minValue.dataType();
        if (dataType != maxValue.dataType()) {
            throw new IllegalArgumentException(
                    "minValue and maxValue must have the same data type: "
                            + dataType + " != " + maxValue.dataType());
        }
        if (dataType == DataType.BOOL) {
            throw new IllegalArgumentException("clamp bounds must be numeric, but were BOOL");
        }
        boolean inverted = switch (dataType) {
            case FLOAT64 -> minValue.float64Value() > maxValue.float64Value();
            case FLOAT32 -> minValue.float32Value() > maxValue.float32Value();
            case BFLOAT16 -> BFloat16Bits.toFloat(minValue.bfloat16Bits())
                    > BFloat16Bits.toFloat(maxValue.bfloat16Bits());
            case INT32 -> minValue.int32Value() > maxValue.int32Value();
            case INT64 -> minValue.int64Value() > maxValue.int64Value();
            case BOOL -> throw new AssertionError("BOOL handled above");
        };
        if (inverted) {
            throw new IllegalArgumentException(
                    "minValue must be less than or equal to maxValue");
        }
    }

    /**
     * Returns the exact inclusive lower bound supplied at construction.
     *
     * @return the non-null exact stored lower bound by its original reference
     */
    @Override
    public ScalarValue minValue() {
        return minValue;
    }

    /**
     * Returns the exact inclusive upper bound supplied at construction.
     *
     * @return the non-null exact stored upper bound by its original reference
     */
    @Override
    public ScalarValue maxValue() {
        return maxValue;
    }
}
