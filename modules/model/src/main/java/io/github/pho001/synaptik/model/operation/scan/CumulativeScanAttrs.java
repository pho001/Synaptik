package io.github.pho001.synaptik.model.operation.scan;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the normalized axis, inclusion mode, and traversal direction for a cumulative scan.
 *
 * <p>The {@link #axis()} is already normalized to a non-negative index by an expression contract
 * that has access to the input Shape. This record validates only that the supplied index is
 * non-negative; it cannot determine whether that index exists for a particular input rank.</p>
 *
 * <p>When {@link #exclusive()} is {@code false}, the value at the current logical position is
 * included in that position's accumulated result. When it is {@code true}, only values visited
 * before the current position are included, so the first traversed position receives the
 * selected kind's identity: additive zero for {@link CumulativeScanKind#CUM_SUM} or
 * multiplicative positive one for {@link CumulativeScanKind#CUM_PROD}. When {@link #reverse()}
 * is {@code false}, traversal proceeds from lower to higher indices; when it is {@code true},
 * traversal proceeds from higher to lower indices. Reverse traversal does not reverse output
 * indexing.</p>
 *
 * <p>The immutable record stores only the three components. Record-generated equality and hashing
 * use all components, and generated text is diagnostic rather than a serialization, parser,
 * dispatch, or backend contract.</p>
 *
 * @param axis the already normalized, non-negative input-axis index
 * @param exclusive whether each output excludes the value at its own logical position
 * @param reverse whether accumulation traverses from higher to lower logical indices
 */
public record CumulativeScanAttrs(
        int axis,
        boolean exclusive,
        boolean reverse) implements OperationAttrs {
    /**
     * Creates complete immutable parameters for one cumulative scan.
     *
     * <p>The values are retained unchanged after the non-negative-axis check. Construction does
     * not normalize the axis, inspect input metadata, derive a result descriptor, materialize an
     * identity value, or execute accumulation.</p>
     *
     * @param axis the already normalized input-axis index; must be non-negative
     * @param exclusive {@code true} to exclude the current value and emit the selected kind's
     *     identity at the first traversed position, or {@code false} to include the current value
     * @param reverse {@code true} to traverse from higher to lower logical indices, or
     *     {@code false} to traverse from lower to higher indices
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     */
    public CumulativeScanAttrs {
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
     * Reports whether each output excludes the value at its own logical position.
     *
     * @return {@code true} for an exclusive scan, or {@code false} for an inclusive scan
     */
    @Override
    public boolean exclusive() {
        return exclusive;
    }

    /**
     * Reports the requested logical traversal direction.
     *
     * @return {@code true} for traversal from higher to lower logical indices, or {@code false}
     *     for traversal from lower to higher indices
     */
    @Override
    public boolean reverse() {
        return reverse;
    }
}
