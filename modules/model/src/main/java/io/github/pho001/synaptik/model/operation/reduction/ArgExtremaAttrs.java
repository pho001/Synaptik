package io.github.pho001.synaptik.model.operation.reduction;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries the normalized axis, dimension-retention choice, and tie policy for arg extrema.
 *
 * <p>The record is shared by {@link AggregateReductionKind#ARG_MIN} and {@link
 * AggregateReductionKind#ARG_MAX}. Tensor expression construction normalizes the caller axis
 * against its input Shape before creating this value. This record can therefore reject a negative
 * normalized axis, but it cannot validate a particular input rank or determine whether the
 * selected extent is empty. When {@link #keepDimensions()} is false, the result removes the
 * selected axis; when true, the result retains that position with extent one. {@link #tiePolicy()}
 * chooses the smallest or largest logical coordinate among equal extrema.</p>
 *
 * <p>The immutable record owns no Tensor, Shape, storage, or mutable input. It retains primitive
 * components and the exact enum reference. Record-generated equality and hashing use all three
 * components, and generated text is diagnostic only rather than a serialization, parser,
 * operation-dispatch, or backend contract.</p>
 *
 * @param axis the already normalized, non-negative input-axis index
 * @param keepDimensions whether the result retains the selected axis with extent one
 * @param tiePolicy the non-null first- or last-logical-index policy
 */
public record ArgExtremaAttrs(
        int axis,
        boolean keepDimensions,
        ArgExtremaTiePolicy tiePolicy) implements OperationAttrs {
    /**
     * Creates complete attributes for a single-axis arg-extrema occurrence.
     *
     * <p>Validation checks the axis before the policy. Construction does not normalize an axis,
     * inspect an input rank or extent, derive a result Shape or descriptor, select a value, or
     * execute a reduction.</p>
     *
     * @param axis the normalized input axis; must be non-negative
     * @param keepDimensions whether to retain the selected result axis with extent one
     * @param tiePolicy the non-null explicit tie-selection policy retained by exact enum reference
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}; this check precedes policy validation
     * @throws NullPointerException if {@code tiePolicy} is null, with message {@code tiePolicy}
     */
    public ArgExtremaAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
        tiePolicy = Objects.requireNonNull(tiePolicy, "tiePolicy");
    }

    /**
     * Returns the already normalized input-axis index.
     *
     * <p>The result has not been validated against an input rank by this attributes value.</p>
     *
     * @return the exact non-negative axis supplied at construction
     */
    @Override
    public int axis() {
        return axis;
    }

    /**
     * Reports whether the selected result axis is retained with extent one.
     *
     * @return the exact retention choice supplied at construction
     */
    @Override
    public boolean keepDimensions() {
        return keepDimensions;
    }

    /**
     * Returns the explicit first- or last-index tie policy.
     *
     * @return the exact non-null immutable enum reference supplied at construction
     */
    @Override
    public ArgExtremaTiePolicy tiePolicy() {
        return tiePolicy;
    }
}
