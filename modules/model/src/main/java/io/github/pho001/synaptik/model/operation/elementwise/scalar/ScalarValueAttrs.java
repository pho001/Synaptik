package io.github.pho001.synaptik.model.operation.elementwise.scalar;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries one exact scalar parameter for a scalar elementwise operation.
 *
 * <p>The value is the multiplier for {@link ScalarElementwiseKind#MUL}, exponent for {@link
 * ScalarElementwiseKind#POW}, minimum for {@link ScalarElementwiseKind#CLAMP_MIN}, or maximum for
 * {@link ScalarElementwiseKind#CLAMP_MAX}. The owning kind determines that role. Construction
 * retains the supplied {@link ScalarValue} by exact reference without conversion, normalization,
 * alternate-precision caching, or defaulting. Receiver compatibility is deliberately not checked
 * here because attributes contain no input Tensor descriptor.</p>
 *
 * <p>The record is immutable and owns no mutable input. Record-generated equality and hashing use
 * the value's exact data type and bit equality. Signed floating zeros and distinct NaN payloads
 * therefore remain distinct. Record-generated text is diagnostic only and is not a
 * serialization, parsing, scalar-conversion, or backend format.</p>
 *
 * @param value the exact scalar multiplier, exponent, minimum, or maximum selected by the owning
 *     kind; must be non-null and is retained by exact immutable reference
 */
public record ScalarValueAttrs(ScalarValue value) implements OperationAttrs {
    /**
     * Creates attributes retaining one exact typed scalar value.
     *
     * @param value non-null exact scalar value retained by reference
     * @throws NullPointerException if {@code value} is {@code null}, with message {@code value}
     */
    public ScalarValueAttrs {
        Objects.requireNonNull(value, "value");
    }
    /**
     * Returns the exact scalar parameter supplied at construction.
     *
     * <p>The owning {@link ScalarElementwiseKind} determines whether this value is a multiplier,
     * exponent, minimum, or maximum. The value is returned without validation, conversion,
     * normalization, or replacement. No input compatibility is implied.</p>
     *
     * @return the non-null exact stored scalar parameter by the same reference supplied at
     *     construction
     */
    @Override
    public ScalarValue value() {
        return value;
    }
}
