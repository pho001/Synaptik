package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the positive static trailing-axis extent for one {@link OneHotKind#ONE_HOT} operation.
 *
 * <p>The retained {@code long} is the exact extent of the new final result Dimension. Every
 * positive value, including {@link Long#MAX_VALUE}, is structurally valid; materializability and
 * total element-count representability belong to later storage and execution layers. The value is
 * structural model metadata. It does not contain indices, an axis, on/off values, an output type,
 * execution state, or a bounds-enforcement policy.</p>
 *
 * @param depth the positive extent of the appended one-hot axis
 */
public record OneHotAttrs(long depth) implements OperationAttrs {
    /**
     * Creates immutable one-hot attributes, retaining the supplied positive value unchanged.
     *
     * @param depth the positive extent of the appended one-hot axis
     * @throws IllegalArgumentException if {@code depth} is not positive, with message
     *     {@code depth must be positive: <depth>}
     */
    public OneHotAttrs {
        if (depth <= 0) {
            throw new IllegalArgumentException("depth must be positive: " + depth);
        }
    }

    /**
     * Returns the positive extent of the appended one-hot axis.
     *
     * @return the exact positive value supplied at construction, including
     *     {@link Long#MAX_VALUE}
     */
    @Override
    public long depth() {
        return depth;
    }
}
