package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the normalized source axis and scalar coordinate for one {@link SelectKind#SELECT}
 * operation.
 *
 * <p>The {@link #axis()} and {@link #index()} are already normalized, zero-based, and
 * non-negative. They mean that the coordinate at {@code index} is fixed on the source
 * {@code axis}, which conceptually removes that axis from the logical result. For a conceptual
 * source shape {@code [2, 3, 4]}, axis {@code 1} and index {@code 2} therefore describe a
 * conceptual result shape {@code [2, 4]}.</p>
 *
 * <p>This value contains no input shape, rank, or selected-axis extent. It consequently cannot
 * prove that the axis exists, that the index is in bounds, or that the conceptual result shape is
 * valid. A later input-aware expression boundary owns caller-facing negative normalization and
 * those checks. Zero, {@link Integer#MAX_VALUE}, and {@link Long#MAX_VALUE} remain structurally
 * valid here.</p>
 *
 * <p>The exact semantic composition is an {@code Operation} whose kind is
 * {@link SelectKind#SELECT} and whose attributes are this value. The kind's family-owned
 * signature enforces that exact pairing and declares one input and one output. The immutable
 * record retains the two primitive components unchanged. Record-generated equality and hashing
 * use both components, and generated text is diagnostic rather than a serialization,
 * request-syntax, compiler, dispatch, or backend contract.</p>
 *
 * <p>These attributes define no Tensor construction, result descriptor, layout or view geometry,
 * provenance, storage, materialization, value access, gradient, graph or compiler behavior,
 * backend support, or execution. Scalar selection remains distinct from conditional
 * {@code WHERE}, general {@code SLICE}, and tensor-index gather.</p>
 *
 * @param axis the already normalized, zero-based, non-negative source-axis index
 * @param index the already normalized, zero-based, non-negative scalar coordinate on that axis
 */
public record SelectAttrs(int axis, long index) implements OperationAttrs {
    /**
     * Creates immutable normalized parameters for one scalar-axis selection.
     *
     * <p>The primitive values are retained unchanged after the ordered non-negative checks.
     * Construction does not normalize raw caller values, inspect an input rank or selected-axis
     * extent, validate bounds, or derive a result shape or layout.</p>
     *
     * @param axis the already normalized source-axis index; must be non-negative
     * @param index the already normalized scalar coordinate on that axis; must be non-negative
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     * @throws IllegalArgumentException if {@code axis} is non-negative and {@code index} is
     *     negative, with message {@code index must be non-negative: <index>}
     */
    public SelectAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
        if (index < 0) {
            throw new IllegalArgumentException("index must be non-negative: " + index);
        }
    }

    /**
     * Returns the already normalized, zero-based source-axis index.
     *
     * <p>The result is structurally non-negative but has not been validated against an input rank
     * by this attributes value.</p>
     *
     * @return the exact non-negative axis supplied at construction
     */
    @Override
    public int axis() {
        return axis;
    }

    /**
     * Returns the already normalized, zero-based scalar coordinate on the selected axis.
     *
     * <p>The result is structurally non-negative but has not been validated against a selected
     * input-axis extent by this attributes value.</p>
     *
     * @return the exact non-negative index supplied at construction
     */
    @Override
    public long index() {
        return index;
    }
}
