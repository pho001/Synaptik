package io.github.pho001.synaptik.model.operation.reduction;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the normalized reduction axis for one first-class masked sum or mean occurrence.
 *
 * <p>{@link AggregateReductionKind#SUM} and {@link AggregateReductionKind#MEAN} pair with this
 * value for their two-input, axis-removing variants. The ordered inputs are {@code [input, mask]},
 * and the mask must use ordinary right-aligned broadcasting to produce exactly the input Shape before these
 * attributes are constructed. Mask alignment is therefore caller-visible Shape provenance, not
 * stored attribute state.</p>
 *
 * <p>This immutable semantic value stores neither input, mask, Shape, broadcast plan, nor
 * selected-count policy. A false broadcast mask position excludes its corresponding input value
 * before aggregation, including NaN and infinity. An empty selected set produces floating zero
 * for masked sum and NaN for masked mean; no NaN payload or bit pattern is specified. This value
 * does not inspect data, derive a descriptor, construct provenance, define gradients, or execute
 * a reduction.</p>
 *
 * @param axis the already normalized, non-negative reduction-axis index removed from the result
 */
public record MaskedReductionAttrs(int axis) implements OperationAttrs {
    /**
     * Creates attributes for one masked, axis-removing sum or mean.
     *
     * <p>The constructor validates only that the supplied normalized axis is non-negative. Axis
     * normalization and rank validation belong to Tensor expression construction.</p>
     *
     * @param axis the already normalized reduction-axis index; must be non-negative
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     */
    public MaskedReductionAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
    }

    /**
     * Returns the already normalized reduction-axis index.
     *
     * @return the exact non-negative axis supplied at construction; the masked reduction removes
     *     this axis from its result
     */
    @Override
    public int axis() {
        return axis;
    }
}
