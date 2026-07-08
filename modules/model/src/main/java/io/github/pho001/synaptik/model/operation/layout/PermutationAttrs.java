package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.List;
import java.util.Objects;

/**
 * Carries a complete normalized output-to-input axis permutation.
 *
 * <p>Element {@code axes[i]} identifies the input axis placed at output axis {@code i}. For
 * example, {@code [1, 0, 2]} exchanges the first two axes and leaves the third axis in place.
 * Every value is in {@code [0, axes.size())}, and every axis occurs exactly once. An empty list is
 * the valid identity permutation for a rank-zero scalar.</p>
 *
 * <p>The caller-owned list is validated in ascending element-index order and copied exactly once
 * after validation. The stored list is an immutable snapshot: later caller mutation cannot alter
 * it, and callers cannot mutate it through {@link #axes()}. List contents and order define record
 * equality and hashing; no list-object identity is promised. Generated text is diagnostic only
 * and is not serialization, parser input, public request syntax, or backend dispatch.</p>
 *
 * <p>This value pairs with {@link AxisTransformKind#PERMUTE}. It validates completeness against
 * its own list size but does not know or validate an eventual input rank. The current rank-two
 * {@code transpose()} convenience uses {@code [1, 0]}. The attributes contain no Tensor, Shape,
 * layout, storage-view, provenance, gradient, compiler, backend, or execution behavior.</p>
 *
 * @param axes the non-null complete normalized permutation in output-to-input order; an empty
 *     list is valid, and the stored value is an immutable snapshot
 */
public record PermutationAttrs(List<Integer> axes) implements OperationAttrs {
    /**
     * Creates immutable complete-permutation attributes.
     *
     * <p>Validation first checks the list reference, creates rank-sized duplicate-tracking state,
     * and then checks each element in ascending index order for null, non-negativity, the upper
     * rank bound, and duplication. Only after every element passes is the list copied once. Range
     * and uniqueness together prove that all axes in {@code [0, rank)} occur exactly once.</p>
     *
     * @param axes the complete output-to-input permutation; must be non-null, contain no null
     *     elements, and contain every integer in {@code [0, axes.size())} exactly once
     * @throws NullPointerException if {@code axes} is {@code null}, with message {@code axes}, or
     *     if element {@code i} is {@code null}, with message {@code axes[<i>]}
     * @throws IllegalArgumentException if element {@code i} is negative, with message
     *     {@code axes[<i>] must be non-negative: <value>}; if it is at least the permutation rank,
     *     with message
     *     {@code axes[<i>] must be less than permutation rank <rank>: <value>}; or if it is the
     *     first duplicate, with message
     *     {@code axes contains duplicate axis <value> at index <i>}
     */
    public PermutationAttrs {
        Objects.requireNonNull(axes, "axes");
        int rank = axes.size();
        boolean[] seen = new boolean[rank];
        for (int index = 0; index < rank; index++) {
            Integer axis = Objects.requireNonNull(axes.get(index), "axes[" + index + "]");
            if (axis < 0) {
                throw new IllegalArgumentException(
                        "axes[" + index + "] must be non-negative: " + axis);
            }
            if (axis >= rank) {
                throw new IllegalArgumentException(
                        "axes["
                                + index
                                + "] must be less than permutation rank "
                                + rank
                                + ": "
                                + axis);
            }
            if (seen[axis]) {
                throw new IllegalArgumentException(
                        "axes contains duplicate axis " + axis + " at index " + index);
            }
            seen[axis] = true;
        }
        axes = List.copyOf(axes);
    }

    /**
     * Returns the immutable complete permutation in output-to-input order.
     *
     * <p>Element {@code i} is the normalized input axis occupying output position {@code i}. The
     * returned list is the stored immutable snapshot; no identity relationship with the caller's
     * original list is promised.</p>
     *
     * @return the non-null immutable complete output-to-input permutation; an empty list denotes
     *     the rank-zero scalar identity
     */
    @Override
    public List<Integer> axes() {
        return axes;
    }
}
