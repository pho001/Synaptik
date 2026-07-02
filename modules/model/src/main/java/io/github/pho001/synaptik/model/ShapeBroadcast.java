package io.github.pho001.synaptik.model;

import java.util.Objects;

/**
 * Computes backend-independent result shapes for right-aligned broadcasting.
 *
 * <p>This utility performs deterministic local shape algebra only. It preserves equal symbolic
 * dimensions and singleton expansion, but it does not create symbolic constraints, calculate
 * strides, or infer shapes across a graph.</p>
 */
public final class ShapeBroadcast {
    private static final StaticDimension SINGLETON = new StaticDimension(1);

    /** Prevents instantiation of this stateless broadcasting utility. */
    private ShapeBroadcast() {
    }

    /**
     * Computes the immutable shape produced by right-aligned broadcasting of two input shapes.
     *
     * <p>Equal dimensions remain unchanged and a static singleton expands to the opposing
     * dimension. Different symbolic dimensions, incompatible static sizes, and symbolic dimensions
     * paired with non-singleton static sizes are rejected because their compatibility cannot be
     * proven locally.</p>
     *
     * @param left non-null left input shape
     * @param right non-null right input shape
     * @return non-null immutable broadcast result with rank equal to the larger input rank
     * @throws NullPointerException if either input is {@code null}
     * @throws IllegalArgumentException if any aligned dimension pair is incompatible or cannot be
     *     proven compatible
     */
    public static Shape broadcast(Shape left, Shape right) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");

        int resultRank = Math.max(left.rank(), right.rank());
        int leftOffset = resultRank - left.rank();
        int rightOffset = resultRank - right.rank();
        Dimension[] result = new Dimension[resultRank];

        for (int axis = 0; axis < resultRank; axis++) {
            Dimension leftDimension = axis < leftOffset
                    ? SINGLETON
                    : left.dimensions().get(axis - leftOffset);
            Dimension rightDimension = axis < rightOffset
                    ? SINGLETON
                    : right.dimensions().get(axis - rightOffset);
            result[axis] = broadcastDimension(leftDimension, rightDimension, axis, left, right);
        }
        return Shape.ofDimensions(result);
    }

    /**
     * Resolves one aligned dimension pair under the local broadcasting contract.
     *
     * @param left non-null aligned left dimension
     * @param right non-null aligned right dimension
     * @param axis non-negative axis in the result shape, used only for diagnostics
     * @param leftShape non-null complete left shape, used only for diagnostics
     * @param rightShape non-null complete right shape, used only for diagnostics
     * @return one of the immutable input dimensions representing the broadcast result
     * @throws IllegalArgumentException if compatibility cannot be proven locally
     */
    private static Dimension broadcastDimension(
            Dimension left,
            Dimension right,
            int axis,
            Shape leftShape,
            Shape rightShape) {
        if (left.equals(right)) {
            return left;
        }
        if (isSingleton(left)) {
            return right;
        }
        if (isSingleton(right)) {
            return left;
        }
        throw new IllegalArgumentException(
                "Cannot broadcast dimensions " + left + " and " + right + " at result axis "
                        + axis + " for " + leftShape + " and " + rightShape);
    }

    /**
     * Reports whether one dimension is the statically known singleton extent.
     *
     * @param dimension non-null dimension to inspect
     * @return {@code true} only for a static dimension of size one
     */
    private static boolean isSingleton(Dimension dimension) {
        return dimension instanceof StaticDimension staticDimension && staticDimension.size() == 1;
    }
}
