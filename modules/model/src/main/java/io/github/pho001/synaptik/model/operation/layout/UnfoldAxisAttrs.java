package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries normalized parameters for materializing sliding windows along one general axis.
 *
 * <p>The axis is already normalized to a non-negative value. A future public expression may
 * accept negative syntax, but only after input-rank context normalizes it. This record has no
 * rank or input Shape, so it cannot prove that the axis exists, that {@code size} fits the selected
 * extent, or that window-count arithmetic is representable.</p>
 *
 * <p>For a static selected extent {@code D}, later result construction uses
 * {@code floor((D - size) / step) + 1} after proving {@code size <= D}. The selected extent is
 * replaced by that position count and the window size is appended as the final axis. This record
 * stores the three intrinsic values but performs none of that arithmetic or Shape construction.</p>
 *
 * <p>These attributes pair only with {@link WindowTransformKind#UNFOLD_AXIS}. They define no
 * Tensor, result, values, layout, storage, provenance, gradient, compiler, backend, or execution
 * behavior. Record-generated equality and hashing use all components in declaration order.</p>
 *
 * @param axis the already normalized, non-negative source-axis position
 * @param size the positive number of elements in each window, in logical axis positions
 * @param step the positive distance between consecutive window starts, in logical axis positions
 */
public record UnfoldAxisAttrs(int axis, long size, long step) implements OperationAttrs {
    /**
     * Creates immutable normalized single-axis unfold parameters.
     *
     * <p>Validation checks axis, size, and step in component order and retains valid values
     * unchanged. Construction does not inspect a rank or extent, normalize raw negative syntax,
     * derive a window count, or perform arithmetic.</p>
     *
     * @param axis the already normalized source axis; must be non-negative
     * @param size the window extent; must be positive
     * @param step the distance between window starts; must be positive
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     * @throws IllegalArgumentException if {@code size} is zero or negative after axis validation,
     *     with message {@code size must be positive: <size>}
     * @throws IllegalArgumentException if {@code step} is zero or negative after axis and size
     *     validation, with message {@code step must be positive: <step>}
     */
    public UnfoldAxisAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
        if (size <= 0) {
            throw new IllegalArgumentException("size must be positive: " + size);
        }
        if (step <= 0) {
            throw new IllegalArgumentException("step must be positive: " + step);
        }
    }

    /**
     * Returns the already normalized source-axis position.
     *
     * @return the exact non-negative axis supplied at construction
     */
    @Override
    public int axis() {
        return axis;
    }

    /**
     * Returns the logical extent of every materialized window.
     *
     * @return the exact positive window size supplied at construction
     */
    @Override
    public long size() {
        return size;
    }

    /**
     * Returns the distance between consecutive logical window starts.
     *
     * @return the exact positive step supplied at construction
     */
    @Override
    public long step() {
        return step;
    }
}
