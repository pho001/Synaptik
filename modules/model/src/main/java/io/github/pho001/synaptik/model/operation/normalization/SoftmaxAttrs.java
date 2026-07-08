package io.github.pho001.synaptik.model.operation.normalization;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the normalized axis for softmax or log-softmax normalization.
 *
 * <p>A normalization slice contains positions that differ only along {@link #axis()} while every
 * other logical coordinate remains fixed. Both {@link SoftmaxKind#SOFTMAX} and
 * {@link SoftmaxKind#LOG_SOFTMAX} have one logical input and produce one result at every input
 * position, so their ideal semantics preserve shape and axis order rather than reducing the
 * selected axis. Neither an input nor its shape is stored here.</p>
 *
 * <p>The axis is already normalized to a non-negative index by a later expression contract that
 * has access to the input shape. This record validates only that the supplied index is
 * non-negative; it cannot determine whether that index exists for a particular input rank. Zero,
 * positive values, and {@link Integer#MAX_VALUE} are structurally valid.</p>
 *
 * <p>For the slice {@code [1, 2, 3]}, ideal softmax probabilities are approximately
 * {@code [0.09003057, 0.24472847, 0.66524096]} and sum to one. Ideal log-softmax values are
 * approximately {@code [-2.40760596, -1.40760596, -0.40760596]}; exponentiating each one yields
 * the corresponding softmax probability. These examples define mathematical meaning and select
 * no finite-precision algorithm.</p>
 *
 * <p>The immutable record stores only the axis. Record-generated equality and hashing use that
 * component, and generated text is diagnostic only rather than a serialization, parser,
 * operation-dispatch, or backend contract. Valid operations explicitly pair this value with
 * either {@link SoftmaxKind#SOFTMAX} or {@link SoftmaxKind#LOG_SOFTMAX}; their family-owned
 * signatures enforce those pairings. Data-type eligibility, descriptor and provenance
 * construction, numerical policy, gradients, compiler behavior, storage, backend support, and
 * execution are deliberately deferred.</p>
 *
 * @param axis the already normalized, non-negative input-axis index
 */
public record SoftmaxAttrs(int axis) implements OperationAttrs {
    /**
     * Creates immutable parameters for one-axis softmax or log-softmax normalization.
     *
     * <p>The axis is retained unchanged after the non-negative check. Construction does not
     * normalize the axis, inspect an input rank or data type, derive a result descriptor, read
     * values, or execute normalization.</p>
     *
     * @param axis the already normalized input-axis index; must be non-negative
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     */
    public SoftmaxAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
    }

    /**
     * Returns the already normalized input-axis index.
     *
     * <p>The result is structurally non-negative but is not validated against an input rank by
     * this attributes value.</p>
     *
     * @return the exact non-negative axis supplied at construction
     */
    @Override
    public int axis() {
        return axis;
    }
}
