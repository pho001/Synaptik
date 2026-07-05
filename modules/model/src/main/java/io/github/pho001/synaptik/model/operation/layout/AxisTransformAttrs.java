package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries one normalized axis position for singleton-axis insertion or removal.
 *
 * <p>{@link AxisTransformKind#EXPAND_DIMS} interprets {@link #axis()} as the normalized output
 * position at which one extent-one axis is inserted. {@link AxisTransformKind#SQUEEZE} interprets
 * it as the normalized input position of the selected extent-one axis to remove. The different
 * meanings come from the paired kind rather than additional record state.</p>
 *
 * <p>The axis is already normalized to a non-negative value by a later expression contract with
 * input-rank context. This record has no rank, so it cannot decide whether the value is a valid
 * insertion position, names an existing input axis, or selects a dimension of extent one. Zero,
 * positive values, and {@link Integer#MAX_VALUE} are structurally valid.</p>
 *
 * <p>Record-generated equality and hashing use the axis, and generated text is diagnostic only.
 * This value contains no Tensor, Shape, layout, result descriptor, storage view, provenance,
 * gradient, compiler, backend, or execution behavior.</p>
 *
 * @param axis the already normalized, non-negative insertion or removal axis position
 */
public record AxisTransformAttrs(int axis) implements OperationAttrs {
    /**
     * Creates immutable parameters for one singleton-axis insertion or removal.
     *
     * <p>The axis is retained unchanged after the non-negative check. Construction does not
     * normalize a raw axis, inspect an input rank or dimension, or derive a result Shape or
     * layout.</p>
     *
     * @param axis the already normalized insertion or removal axis position; must be non-negative
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     */
    public AxisTransformAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
    }

    /**
     * Returns the already normalized singleton-axis insertion or removal position.
     *
     * <p>The paired kind determines whether the result denotes an output insertion position or
     * an input removal position. The value is structurally non-negative but is not validated
     * against any rank by this attributes record.</p>
     *
     * @return the exact non-negative axis supplied at construction
     */
    @Override
    public int axis() {
        return axis;
    }
}
