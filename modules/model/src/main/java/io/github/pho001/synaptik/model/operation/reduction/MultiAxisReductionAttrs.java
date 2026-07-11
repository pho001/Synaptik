package io.github.pho001.synaptik.model.operation.reduction;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Carries an ordered set of normalized axes and a dimension-retention choice for one multi-axis
 * reduction occurrence.
 *
 * <p>Tensor construction normalizes each positive or negative caller axis against the input Shape
 * before creating this value. Axis order is retained for semantic equality, diagnostics,
 * transformation, and interchange; it does not prescribe a sequential fold or physical traversal
 * order. An empty list means reduction over no axes, not full reduction. When
 * {@link #keepDimensions()} is false, selected axes are removed from the eventual result; when it
 * is true, each selected position remains with extent one.</p>
 *
 * <p>The record snapshots the supplied list and exposes the immutable snapshot. It stores no
 * Tensor, Shape, rank, descriptor, storage, graph, algorithm, or backend state, so a non-negative
 * axis is not checked against an input rank here. Record equality and hashing include the ordered
 * axes and retention flag; generated text is diagnostic only.</p>
 *
 * @param axes the non-null ordered normalized axes; elements must be non-null, non-negative, and
 *     distinct, and the list may be empty
 * @param keepDimensions whether selected axes remain as extent-one result dimensions
 */
public record MultiAxisReductionAttrs(
        List<Integer> axes,
        boolean keepDimensions) implements OperationAttrs {
    /**
     * Validates the ordered normalized axes and snapshots the caller's list.
     *
     * <p>Validation checks the list reference, then each element in index order for null,
     * non-negative value, and duplication. The caller retains ownership of the original list;
     * later mutation cannot change this value.</p>
     *
     * @param axes the ordered normalized axes to snapshot; may be empty
     * @param keepDimensions whether selected axes remain with extent one
     * @throws NullPointerException if {@code axes} is null, with message {@code axes}, or if an
     *     element is null, with message {@code axes[<index>]}
     * @throws IllegalArgumentException if an element is negative, with message
     *     {@code axes[<index>] must be non-negative: <axis>}, or duplicates an earlier axis, with
     *     message {@code axes contains duplicate axis <axis> at index <index>}
     */
    public MultiAxisReductionAttrs {
        Objects.requireNonNull(axes, "axes");
        Set<Integer> seen = new HashSet<>();
        for (int index = 0; index < axes.size(); index++) {
            Integer axis = Objects.requireNonNull(axes.get(index), "axes[" + index + "]");
            if (axis < 0) {
                throw new IllegalArgumentException(
                        "axes[" + index + "] must be non-negative: " + axis);
            }
            if (!seen.add(axis)) {
                throw new IllegalArgumentException(
                        "axes contains duplicate axis " + axis + " at index " + index);
            }
        }
        axes = List.copyOf(axes);
    }

    /**
     * Returns the immutable ordered normalized-axis snapshot.
     *
     * @return the exact non-null immutable snapshot created at construction; possibly empty
     */
    @Override
    public List<Integer> axes() {
        return axes;
    }
}
