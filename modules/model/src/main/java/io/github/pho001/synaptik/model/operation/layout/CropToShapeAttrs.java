package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Objects;

/**
 * Carries the exact target and non-negative prefix Shapes for a target-relative slice region.
 *
 * <p>On each axis, the region begins after {@code prefixShape}'s extent and has exactly
 * {@code targetShape}'s extent. For {@link SliceKind#SLICE}, the target is the exact extraction
 * result Shape. For {@link SliceKind#SLICE_UPDATE}, the target is the exact update-region Shape;
 * the operation result instead retains the base Shape. In both forms, the prefix identifies the
 * exact non-negative logical extent preceding the region on each axis.</p>
 *
 * <p>Either Shape may retain symbolic Dimensions. For target {@code [N]} and prefix {@code [1]},
 * the region begins after one logical position and has symbolic length {@code N}. The prefix is
 * Shape metadata, not a Tensor, bound coordinate, storage offset, or execution value.</p>
 *
 * <p>This value retains both exact immutable Shape references. It performs no rank, input,
 * data-type, bounds, layout, binding, compiler, or execution validation.</p>
 *
 * @param targetShape the non-null exact extraction-result or update-region Shape retained
 *     unchanged
 * @param prefixShape the non-null exact per-axis prefix-extent Shape retained unchanged
 */
public record CropToShapeAttrs(
        Shape targetShape,
        Shape prefixShape) implements OperationAttrs {
    /**
     * Creates target-relative region attributes from exact immutable Shape values.
     *
     * @param targetShape the non-null exact extraction-result or update-region Shape retained
     *     unchanged
     * @param prefixShape the non-null exact per-axis prefix-extent Shape retained unchanged
     * @throws NullPointerException if {@code targetShape} or {@code prefixShape} is null, checked
     *     in that order with the component name as the message
     */
    public CropToShapeAttrs {
        Objects.requireNonNull(targetShape, "targetShape");
        Objects.requireNonNull(prefixShape, "prefixShape");
    }

    /**
     * Returns the exact extraction-result or update-region Shape.
     *
     * @return the non-null immutable Shape reference supplied at construction
     */
    @Override
    public Shape targetShape() {
        return targetShape;
    }

    /**
     * Returns the exact per-axis non-negative prefix extents preceding the slice region.
     *
     * @return the non-null immutable Shape reference supplied at construction
     */
    @Override
    public Shape prefixShape() {
        return prefixShape;
    }
}
