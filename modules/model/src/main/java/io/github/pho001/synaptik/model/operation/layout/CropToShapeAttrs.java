package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Objects;

/**
 * Carries the exact target and non-negative prefix Shapes for target-relative slice extraction.
 *
 * <p>On each axis, the logical crop begins after {@code prefixShape}'s extent and selects exactly
 * {@code targetShape}'s extent. Either Shape may retain symbolic Dimensions; for input
 * {@code [N + 3]}, target {@code [N]}, and prefix {@code [1]}, the result begins after one logical
 * position and retains symbolic length {@code N}. The prefix is Shape metadata, not a Tensor,
 * bound coordinate, storage offset, or execution value.</p>
 *
 * <p>This value retains both exact immutable Shape references. It performs no rank, input,
 * bounds, data-type, layout, binding, compiler, or execution validation.</p>
 *
 * @param targetShape the non-null exact logical result Shape retained unchanged
 * @param prefixShape the non-null exact per-axis prefix-extent Shape retained unchanged
 */
public record CropToShapeAttrs(
        Shape targetShape,
        Shape prefixShape) implements OperationAttrs {
    /**
     * Creates target-relative crop attributes from exact immutable Shape values.
     *
     * @param targetShape the non-null exact logical result Shape retained unchanged
     * @param prefixShape the non-null exact per-axis prefix-extent Shape retained unchanged
     * @throws NullPointerException if {@code targetShape} or {@code prefixShape} is null, checked
     *     in that order with the component name as the message
     */
    public CropToShapeAttrs {
        Objects.requireNonNull(targetShape, "targetShape");
        Objects.requireNonNull(prefixShape, "prefixShape");
    }

    /**
     * Returns the exact logical result Shape.
     *
     * @return the non-null immutable Shape reference supplied at construction
     */
    @Override
    public Shape targetShape() {
        return targetShape;
    }

    /**
     * Returns the exact per-axis non-negative prefix extents preceding the crop region.
     *
     * @return the non-null immutable Shape reference supplied at construction
     */
    @Override
    public Shape prefixShape() {
        return prefixShape;
    }
}
