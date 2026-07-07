package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the normalized data-axis position shared by axis-gather and fixed-add axis-scatter
 * operations.
 *
 * <p>The {@link #axis()} value is already normalized, zero-based, and non-negative. Valid
 * axis-gather operations pair this value with {@link AxisGatherKind#GATHER},
 * {@link AxisGatherKind#GATHER_AXIS}, or {@link AxisGatherKind#TAKE_ALONG_AXIS}; fixed-add
 * axis-scatter operations pair it with {@link AxisScatterKind#SCATTER_ADD} or
 * {@link AxisScatterKind#SCATTER_AXIS_ADD}. Gather has ordered logical inputs
 * {@code [data, indices]}, while scatter has {@code [data, indices, updates]}. Each kind determines
 * its own index alignment and result-shape relationship. Addition is intrinsic to the two scatter
 * kinds rather than stored as configurable attribute state. Generic operation composition retains
 * the exact kind and attributes references without enforcing those family pairings.</p>
 *
 * <p>This value contains no input rank, data shape, selected-axis extent, indices shape, or result
 * shape. It therefore cannot prove that the axis exists or validate the family-specific shape
 * rules. Zero, positive values, and {@link Integer#MAX_VALUE} are structurally valid and retained
 * unchanged. The current public axis-gather Tensor-expression contract owns caller-facing
 * negative-axis normalization, rank and shape checks, and the requirement that index tensors use
 * {@code INT32} or {@code INT64}. Task 0018H owns the corresponding public fixed-add axis-scatter
 * checks. This attributes value reads no index values and therefore performs no index-value bounds
 * check.</p>
 *
 * <p>The immutable record stores only the primitive axis. Record-generated equality and hashing
 * use that component, and generated text is diagnostic rather than a serialization, request,
 * parsing, compiler-dispatch, or backend contract. These attributes define no Tensor
 * construction, descriptor, provenance, storage, materialization, gradient, graph or compiler
 * behavior, backend support, or execution.</p>
 *
 * @param axis the already normalized, zero-based, non-negative data-axis index
 */
public record IndexAxisAttrs(int axis) implements OperationAttrs {
    /**
     * Creates immutable normalized axis parameters for an axis-gather or fixed-add axis-scatter
     * operation.
     *
     * <p>The primitive value is retained unchanged after the non-negative check. Construction does
     * not normalize a raw caller axis, inspect data or indices, validate rank, shape, data type, or
     * bounds, or derive a result.</p>
     *
     * @param axis the already normalized data-axis index; must be non-negative
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     */
    public IndexAxisAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
    }

    /**
     * Returns the already normalized, zero-based data-axis index.
     *
     * <p>The result is structurally non-negative but has not been validated against a data rank by
     * this attributes value.</p>
     *
     * @return the exact non-negative axis supplied at construction
     */
    @Override
    public int axis() {
        return axis;
    }
}
