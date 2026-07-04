package io.github.pho001.synaptik.model.operation.elementwise.scalar;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries one exact scalar parameter for a scalar elementwise operation.
 *
 * <p>The value is the multiplier for {@link ScalarElementwiseKind#MUL}, exponent for {@link
 * ScalarElementwiseKind#POW}, minimum for {@link ScalarElementwiseKind#CLAMP_MIN}, or maximum for
 * {@link ScalarElementwiseKind#CLAMP_MAX}. The owning kind determines that role. Construction
 * retains the supplied Java binary64 primitive unchanged, without validation, conversion,
 * normalization, alternate-precision caching, or defaulting. Finite values, either infinity,
 * either signed zero, and every NaN payload are representable.</p>
 *
 * <p>The record is immutable and owns no mutable input. Record-generated equality and hashing use
 * Java's standard {@code double} component semantics: positive and negative zero compare as
 * different components, while all NaN values compare as equal components even when their retained
 * raw payload bits differ. Record-generated text is diagnostic only and is not a serialization,
 * parsing, scalar-conversion, or backend format.</p>
 *
 * @param value the exact scalar multiplier, exponent, minimum, or maximum selected by the owning
 *     kind; every Java {@code double} value is accepted and retained unchanged
 */
public record ScalarValueAttrs(double value) implements OperationAttrs {
    /**
     * Returns the exact scalar parameter supplied at construction.
     *
     * <p>The owning {@link ScalarElementwiseKind} determines whether this value is a multiplier,
     * exponent, minimum, or maximum. The value is returned without validation, conversion,
     * normalization, or replacement.</p>
     *
     * @return the exact stored scalar parameter
     */
    @Override
    public double value() {
        return value;
    }
}
