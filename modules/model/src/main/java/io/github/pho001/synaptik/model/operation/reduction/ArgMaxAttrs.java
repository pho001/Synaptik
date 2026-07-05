package io.github.pho001.synaptik.model.operation.reduction;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries the normalized axis, retained-dimension choice, and explicit tie policy for arg-max.
 *
 * <p>The {@link #axis()} is already normalized to a non-negative index by the later expression
 * contract that knows the input shape. This record cannot determine whether that index exists for
 * a particular rank. {@link #keepDimensions()} controls whether the selected axis is removed or
 * retained with extent one in the eventual result. {@link #tiePolicy()} explicitly selects the
 * smallest or largest logical index among equal maxima; construction never supplies an implicit
 * default.</p>
 *
 * <p>Validation follows component order: a negative axis fails before a null tie policy is
 * inspected. The immutable record owns no mutable input, and record-generated equality and hashing
 * use all three components. Generated text is diagnostic only and is not a serialization, parser,
 * operation-dispatch, or backend contract.</p>
 *
 * @param axis the already normalized, non-negative input-axis index
 * @param keepDimensions whether the eventual result retains the selected axis with extent one
 * @param tiePolicy the non-null explicit selection policy for equal maximum values
 */
public record ArgMaxAttrs(
        int axis,
        boolean keepDimensions,
        ArgMaxTiePolicy tiePolicy) implements OperationAttrs {
    /**
     * Creates complete attributes for an arg-max reduction along one normalized input axis.
     *
     * <p>The primitive values and exact enum reference are retained unchanged after validation.
     * Construction does not normalize the axis, inspect an input rank, choose a default policy,
     * compare values, or construct an output descriptor.</p>
     *
     * @param axis the already normalized input-axis index; must be non-negative
     * @param keepDimensions {@code true} to retain the selected result axis with extent one, or
     *     {@code false} to remove it
     * @param tiePolicy the non-null explicit tie-selection policy retained unchanged
     * @throws IllegalArgumentException if {@code axis} is negative, with a message containing the
     *     rejected value; this check occurs before tie-policy validation
     * @throws NullPointerException if {@code tiePolicy} is {@code null}, with message
     *     {@code tiePolicy}
     */
    public ArgMaxAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
        tiePolicy = Objects.requireNonNull(tiePolicy, "tiePolicy");
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
     * Reports whether the selected axis remains in the eventual result with extent one.
     *
     * @return the exact retained-dimension choice supplied at construction
     */
    @Override
    public boolean keepDimensions() {
        return keepDimensions;
    }

    /**
     * Returns the explicit policy for choosing among equal maximum values.
     *
     * @return the exact non-null tie-policy enum reference supplied at construction
     */
    @Override
    public ArgMaxTiePolicy tiePolicy() {
        return tiePolicy;
    }
}
