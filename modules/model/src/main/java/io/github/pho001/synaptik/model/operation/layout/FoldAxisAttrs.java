package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries normalized target geometry for scatter-adding general-axis windows.
 *
 * <p>These attributes pair only with {@link WindowTransformKind#FOLD_AXIS}. Later construction
 * interprets the eventual input's final dimension as window size, removes that dimension, and
 * restores {@link #axis()} to {@link #outputSize()}. Window element {@code offset} from window
 * {@code windowIndex} targets {@code windowIndex * step + offset}; overlaps are added and valid
 * target positions receiving no contribution remain zero. This record performs none of that
 * Shape or value behavior.</p>
 *
 * <p>The explicit output size is necessary because window count, window size, and step cannot
 * always recover trailing source positions that no window covered. Conceptual windows of Shape
 * {@code [3, 3]} with values {@code [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, axis zero, output size
 * five, and step one fold to conceptual Shape {@code [5]} with values
 * {@code [1, 6, 15, 14, 9]} under the later scatter-add meaning.</p>
 *
 * <p>The axis is already normalized in the target rank. This record has no input Shape, rank,
 * window size, or window count and therefore cannot validate bounds or compatibility. It names
 * the same semantic operation that task 0017N will expose publicly and task 0023 may later
 * generate during autograd, without defining Tensor, gradient, compiler, backend, or execution
 * behavior.</p>
 *
 * @param axis the already normalized, non-negative restored target-axis position
 * @param outputSize the non-negative target extent restored at {@code axis}, in logical positions
 * @param step the positive distance between consecutive window starts, in logical positions
 */
public record FoldAxisAttrs(int axis, long outputSize, long step) implements OperationAttrs {
    /**
     * Creates immutable normalized single-axis fold parameters.
     *
     * <p>Validation checks axis, output size, and step in component order and retains every valid
     * value unchanged. Zero output size is structurally valid. Construction does not inspect the
     * eventual input's final window dimension, validate compatibility, or perform arithmetic.</p>
     *
     * @param axis the already normalized target axis; must be non-negative
     * @param outputSize the restored target extent; must be non-negative
     * @param step the distance between window starts; must be positive
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     * @throws IllegalArgumentException if {@code outputSize} is negative after axis validation,
     *     with message {@code outputSize must be non-negative: <outputSize>}
     * @throws IllegalArgumentException if {@code step} is zero or negative after axis and output
     *     size validation, with message {@code step must be positive: <step>}
     */
    public FoldAxisAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
        if (outputSize < 0) {
            throw new IllegalArgumentException(
                    "outputSize must be non-negative: " + outputSize);
        }
        if (step <= 0) {
            throw new IllegalArgumentException("step must be positive: " + step);
        }
    }

    /**
     * Returns the already normalized restored target-axis position.
     *
     * @return the exact non-negative axis supplied at construction
     */
    @Override
    public int axis() {
        return axis;
    }

    /**
     * Returns the explicit logical extent to restore at the target axis.
     *
     * @return the exact non-negative output size supplied at construction, including zero
     */
    @Override
    public long outputSize() {
        return outputSize;
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
