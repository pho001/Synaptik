package io.github.pho001.synaptik.model.operation.reduction;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries a normalized axis and retained-dimension choice for one ordinary aggregate reduction.
 *
 * <p>The {@link #axis()} is already normalized to a non-negative index by the later expression
 * contract that has access to the input shape. This value therefore validates only that the
 * supplied index is non-negative; it cannot determine whether that index exists for a particular
 * input rank. When {@link #keepDimensions()} is {@code false}, the selected axis is removed from
 * the eventual result. When it is {@code true}, that axis remains with extent one. This record
 * stores no input or result shape.</p>
 *
 * <p>The record is immutable and owns no mutable input. Record-generated equality and hashing use
 * both components. Generated text is diagnostic only and is not a serialization, parser,
 * operation-dispatch, or backend contract. Full reductions use
 * {@code NoOperationAttrs.INSTANCE} instead of this record or a negative axis sentinel.</p>
 *
 * @param axis the already normalized, non-negative input-axis index
 * @param keepDimensions whether the eventual result retains the selected axis with extent one
 */
public record AxisReductionAttrs(int axis, boolean keepDimensions) implements OperationAttrs {
    /**
     * Creates attributes for an ordinary reduction along one normalized input axis.
     *
     * <p>The values are retained unchanged after the non-negative-axis check. Construction does
     * not normalize the axis, inspect an input rank, or construct an output shape.</p>
     *
     * @param axis the already normalized input-axis index; must be non-negative
     * @param keepDimensions {@code true} to retain the selected result axis with extent one, or
     *     {@code false} to remove it
     * @throws IllegalArgumentException if {@code axis} is negative, with a message containing the
     *     rejected value
     */
    public AxisReductionAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
    }

    /**
     * Returns the already normalized input-axis index.
     *
     * @return the exact non-negative axis supplied at construction
     */
    @Override
    public int axis() {
        return axis;
    }

    /**
     * Reports whether the selected axis remains in the eventual result with extent one.
     *
     * @return the exact retained-dimension choice supplied at construction
     */
    @Override
    public boolean keepDimensions() {
        return keepDimensions;
    }
}
