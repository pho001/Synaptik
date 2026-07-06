package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries one normalized axis position for concatenation or stacking.
 *
 * <p>{@link TensorCompositionKind#CONCAT} interprets {@link #axis()} as an existing input-axis
 * position along which ordered inputs are joined. {@link TensorCompositionKind#STACK} interprets
 * it as the newly inserted result-axis position. For conceptual Shape {@code [2, 3]}, CONCAT axis
 * one names the existing extent-three axis, while STACK axis one names the gap between the two
 * existing axes where a new axis will appear.</p>
 *
 * <p>The axis is already normalized to a non-negative {@code int} by a later expression contract
 * with rank context. This record has no rank and cannot decide whether the value names an
 * existing CONCAT axis or a valid STACK insertion position. Zero, positive values, and
 * {@link Integer#MAX_VALUE} are structurally valid and retained unchanged.</p>
 *
 * <p>Record-generated equality and hashing use the axis, and generated text is diagnostic only.
 * This value contains no Tensor, input list or count, Shape, result descriptor, layout,
 * provenance, graph grouping, gradient, compiler, backend, ONNX, or execution behavior.</p>
 *
 * @param axis the already normalized, non-negative existing or inserted axis position
 */
public record CompositionAxisAttrs(int axis) implements OperationAttrs {
    /**
     * Creates immutable normalized axis parameters for CONCAT or STACK.
     *
     * <p>The value is retained unchanged after the non-negative check. Construction does not
     * normalize a raw axis, inspect a rank or input sequence, validate Shapes, or derive a result
     * Shape or layout.</p>
     *
     * @param axis the already normalized existing or inserted axis position; must be non-negative
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     */
    public CompositionAxisAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
    }

    /**
     * Returns the already normalized composition-axis position.
     *
     * <p>The paired kind determines whether this is an existing CONCAT input axis or a newly
     * inserted STACK result-axis position. The value is structurally non-negative but is not
     * checked against any rank by this attributes record.</p>
     *
     * @return the exact non-negative axis supplied at construction
     */
    @Override
    public int axis() {
        return axis;
    }
}
