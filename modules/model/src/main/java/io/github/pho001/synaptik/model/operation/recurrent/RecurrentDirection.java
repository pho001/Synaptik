package io.github.pho001.synaptik.model.operation.recurrent;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Selects traversal order for one fixed recurrent-scan occurrence.
 *
 * <p>{@link #FORWARD} consumes each row's valid prefix from its first coordinate to its last.
 * {@link #REVERSE} consumes that same valid prefix from its last coordinate to its first; it does
 * not traverse the padded suffix or reverse the dense output coordinates. This immutable enum is
 * the complete operation-attributes value. It carries no callback, body, state, runtime length,
 * execution route, or backend behavior.</p>
 */
public enum RecurrentDirection implements OperationAttrs {
    /** Consumes valid coordinates in increasing original-time order. */
    FORWARD,

    /** Consumes valid coordinates in decreasing original-time order. */
    REVERSE
}
