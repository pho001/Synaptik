package io.github.pho001.synaptik.model.operation.elementwise.scalar;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the exact ordered inclusive bounds for a scalar range-clamp operation.
 *
 * <p>The lower bound precedes the upper bound and both Java binary64 primitives are retained
 * unchanged. Construction rejects only a strictly inverted range under the primitive {@code >}
 * comparison. Equal bounds, either ordering of signed zeros, ordered infinities, and one or two
 * NaN endpoints are accepted. No finite check, NaN canonicalization, infinity replacement,
 * signed-zero normalization, scalar conversion, or numerical execution policy is applied.</p>
 *
 * <p>The record is immutable and owns no mutable input. Record-generated equality and hashing use
 * Java's standard {@code double} component semantics: positive and negative zero compare as
 * different components, while all NaN values compare as equal components even when their retained
 * raw payload bits differ. Record-generated text is diagnostic only and is not a serialization,
 * parsing, scalar-conversion, or backend format.</p>
 *
 * @param minValue the exact inclusive lower bound, retained unchanged
 * @param maxValue the exact inclusive upper bound, retained unchanged
 */
public record ClampRangeAttrs(double minValue, double maxValue) implements OperationAttrs {
    /**
     * Creates inclusive clamp-range attributes from exact lower and upper bounds.
     *
     * <p>The primitive values are compared once in lower-then-upper parameter order. Only
     * {@code minValue > maxValue} fails. Equal values, either signed-zero ordering, ordered
     * infinities, and NaN endpoints are accepted and stored without conversion or normalization.</p>
     *
     * @param minValue the exact inclusive lower bound to retain
     * @param maxValue the exact inclusive upper bound to retain
     * @throws IllegalArgumentException if {@code minValue} is strictly greater than
     *     {@code maxValue}
     */
    public ClampRangeAttrs {
        if (minValue > maxValue) {
            throw new IllegalArgumentException(
                    "minValue must be less than or equal to maxValue");
        }
    }

    /**
     * Returns the exact inclusive lower bound supplied at construction.
     *
     * @return the exact stored lower bound, without conversion or normalization
     */
    @Override
    public double minValue() {
        return minValue;
    }

    /**
     * Returns the exact inclusive upper bound supplied at construction.
     *
     * @return the exact stored upper bound, without conversion or normalization
     */
    @Override
    public double maxValue() {
        return maxValue;
    }
}
