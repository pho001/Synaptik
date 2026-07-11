package io.github.pho001.synaptik.model.operation.ordering;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the normalized axis and direction for a stable full-ordering request.
 *
 * <p>Tensor construction normalizes the caller axis against an input Shape before creating this
 * value. Stability and NaN-last placement are fixed family semantics rather than configurable
 * attributes. This record does not inspect a Tensor, rank, extent, values, layout, or storage and
 * supplies no comparator or executable algorithm.</p>
 *
 * @param axis the already normalized, non-negative logical input-axis index
 * @param descending whether non-NaN values are requested in descending rather than ascending order
 */
public record SortAttrs(int axis, boolean descending) implements OperationAttrs {
    /**
     * Creates immutable full-ordering attributes.
     *
     * @param axis the already normalized input axis; must be non-negative
     * @param descending the exact requested direction flag
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     */
    public SortAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
    }

    /**
     * Returns the already normalized logical input-axis index.
     *
     * @return the exact non-negative axis supplied at construction
     */
    @Override
    public int axis() {
        return axis;
    }

    /**
     * Reports the requested ordering direction.
     *
     * @return {@code true} for descending non-NaN order, or {@code false} for ascending order
     */
    @Override
    public boolean descending() {
        return descending;
    }
}
