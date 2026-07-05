package io.github.pho001.synaptik.model.operation.scan;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the normalized axis, inclusion mode, and traversal direction for a cumulative sum.
 *
 * <p>The {@link #axis()} is already normalized to a non-negative index by a later expression
 * contract that has access to the input shape. This record validates only that the supplied index
 * is non-negative; it cannot determine whether that index exists for a particular input rank. A
 * cumulative sum has one logical input and preserves its logical shape, but neither the input nor
 * the shape is stored here.</p>
 *
 * <p>When {@link #exclusive()} is {@code false}, the value at the current logical position is
 * included in that position's accumulated result. When it is {@code true}, only values visited
 * before the current position are included, so the first position visited in the selected
 * direction receives additive zero. When {@link #reverse()} is {@code false}, traversal proceeds
 * from lower to higher indices; when it is {@code true}, traversal proceeds from higher to lower
 * indices. Reverse traversal does not reverse output indexing.</p>
 *
 * <p>For logical input {@code [1, 2, 3]}, the four combinations mean:</p>
 *
 * <ul>
 *   <li>inclusive forward ({@code false, false}) produces {@code [1, 3, 6]};</li>
 *   <li>exclusive forward ({@code true, false}) produces {@code [0, 1, 3]}, with zero at the
 *       lowest index because no value precedes it;</li>
 *   <li>inclusive reverse ({@code false, true}) produces {@code [6, 5, 3]}; and</li>
 *   <li>exclusive reverse ({@code true, true}) produces {@code [5, 3, 0]}, with zero at the
 *       highest index because reverse traversal visits it first.</li>
 * </ul>
 *
 * <p>The examples define semantic meaning and do not execute a scan. The immutable record stores
 * only the three components. Record-generated equality and hashing use all components, and
 * generated text is diagnostic only rather than a serialization, parser, operation-dispatch, or
 * backend contract. {@link CumulativeSumKind#CUM_SUM} is the intended paired kind.</p>
 *
 * @param axis the already normalized, non-negative input-axis index
 * @param exclusive whether each output excludes the value at its own logical position
 * @param reverse whether accumulation traverses from higher to lower logical indices
 */
public record CumulativeSumAttrs(
        int axis,
        boolean exclusive,
        boolean reverse) implements OperationAttrs {
    /**
     * Creates complete immutable parameters for one cumulative-sum scan.
     *
     * <p>The values are retained unchanged after the non-negative-axis check. Construction does
     * not normalize the axis, inspect an input rank or data type, derive a result descriptor,
     * allocate an additive-zero value, or execute accumulation.</p>
     *
     * @param axis the already normalized input-axis index; must be non-negative
     * @param exclusive {@code true} to exclude the current value and emit additive zero at the
     *     first traversed position, or {@code false} to include the current value
     * @param reverse {@code true} to traverse from higher to lower logical indices, or
     *     {@code false} to traverse from lower to higher indices
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     */
    public CumulativeSumAttrs {
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

    /**
     * Reports whether each output excludes the value at its own logical position.
     *
     * @return {@code true} for an exclusive scan whose first traversed output is additive zero,
     *     or {@code false} for an inclusive scan
     */
    @Override
    public boolean exclusive() {
        return exclusive;
    }

    /**
     * Reports the requested logical traversal direction.
     *
     * <p>A reverse scan still exposes output positions in input order; only the direction in which
     * values enter the accumulated prefix changes.</p>
     *
     * @return {@code true} for traversal from higher to lower logical indices, or {@code false}
     *     for traversal from lower to higher indices
     */
    @Override
    public boolean reverse() {
        return reverse;
    }
}
