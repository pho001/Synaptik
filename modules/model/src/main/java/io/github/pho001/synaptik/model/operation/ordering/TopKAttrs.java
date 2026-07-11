package io.github.pho001.synaptik.model.operation.ordering;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the normalized axis, static selection count, direction, and output order for top-K.
 *
 * <p>This value retains only normalized semantic parameters. It does not validate Tensor rank or
 * the selected extent, bind a dynamic dimension, compare values, choose an algorithm, or contain
 * compiler, backend, gradient, runtime, or execution state. Ordinary record equality and hashing
 * include all four components.</p>
 *
 * @param axis already normalized non-negative logical axis
 * @param k non-negative number of selected pairs
 * @param largest whether selection starts from descending rather than ascending numerical order
 * @param sorted whether output pairs retain selection order rather than logical-index order
 */
public record TopKAttrs(int axis, long k, boolean largest, boolean sorted)
        implements OperationAttrs {
    /**
     * Creates immutable top-K attributes, validating axis before K.
     *
     * @param axis already normalized axis; must be non-negative
     * @param k selection count; must be non-negative
     * @param largest exact requested selection direction
     * @param sorted exact requested output-order flag
     * @throws IllegalArgumentException if {@code axis} or {@code k} is negative
     */
    public TopKAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
        if (k < 0) {
            throw new IllegalArgumentException("k must be non-negative: " + k);
        }
    }

    /**
     * Returns the selected logical axis after Tensor construction has normalized it.
     *
     * @return the exact already normalized non-negative logical axis
     */
    @Override
    public int axis() {
        return axis;
    }

    /**
     * Returns the number of value/index pairs requested along each selected-axis slice.
     *
     * @return the exact non-negative selection count
     */
    @Override
    public long k() {
        return k;
    }

    /**
     * Returns the direction used to determine the selected set.
     *
     * @return whether largest rather than smallest values are selected
     */
    @Override
    public boolean largest() {
        return largest;
    }

    /**
     * Returns the requested order of the already selected value/index pairs.
     *
     * @return {@code true} when selected pairs retain stable value-selection order; {@code false}
     *     when they use increasing original logical-axis index
     */
    @Override
    public boolean sorted() {
        return sorted;
    }
}
