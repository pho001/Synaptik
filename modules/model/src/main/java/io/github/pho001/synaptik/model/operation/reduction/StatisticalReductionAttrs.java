package io.github.pho001.synaptik.model.operation.reduction;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Carries ordered normalized axes, dimension retention, and correction for variance or standard
 * deviation.
 *
 * <p>The axes have the same ordered-set and empty-list meaning as
 * {@link MultiAxisReductionAttrs}. {@code correction} is the non-negative integer subtracted from
 * selected-domain count {@code N}; zero requests population statistics and one represents the
 * usual sample estimator when {@code N > 1}. This attributes value validates only local parameter
 * form. Tensor construction owns Shape-aware proof that {@code N > correction} when the count is
 * statically known, while later owning layers validate dynamic counts.</p>
 *
 * <p>The record snapshots the list and owns no Tensor, Shape, rank, descriptor, storage, graph,
 * execution, or backend state. Record equality and hashing include ordered axes, retention, and
 * correction; generated text is diagnostic only.</p>
 *
 * @param axes the non-null ordered normalized axes; elements must be non-null, non-negative, and
 *     distinct, and the list may be empty
 * @param keepDimensions whether selected axes remain as extent-one result dimensions
 * @param correction the non-negative value subtracted from selected-domain count
 */
public record StatisticalReductionAttrs(
        List<Integer> axes,
        boolean keepDimensions,
        long correction) implements OperationAttrs {
    /**
     * Validates axes before correction and snapshots the caller's list.
     *
     * @param axes the ordered normalized axes to snapshot; may be empty
     * @param keepDimensions whether selected axes remain with extent one
     * @param correction the non-negative denominator correction
     * @throws NullPointerException if {@code axes} is null, with message {@code axes}, or if an
     *     element is null, with message {@code axes[<index>]}
     * @throws IllegalArgumentException if an axis is negative or duplicated, using the indexed
     *     axis messages documented by {@link MultiAxisReductionAttrs}; or if {@code correction}
     *     is negative, with message {@code correction must be non-negative: <correction>}; axis
     *     validation and snapshotting precede correction validation
     */
    public StatisticalReductionAttrs {
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
        if (correction < 0) {
            throw new IllegalArgumentException("correction must be non-negative: " + correction);
        }
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
