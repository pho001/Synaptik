package io.github.pho001.synaptik.model.operation.index;

/**
 * Selects the replacement or mathematical reduction used to combine functional-scatter updates
 * at one target coordinate.
 *
 * <p>The base value is the value initially present in the scatter's {@code data} input. An update
 * is a value from the ordered {@code updates} input, and a target coordinate is the result
 * coordinate selected through {@code indices}. Multiple updates may address one target. A
 * reduction defines the mathematical combination of that base value and all addressed updates;
 * it does not define an execution algorithm, traversal or accumulation order, numeric edge
 * behavior, supported data types, atomicity, determinism, or backend capability.</p>
 *
 * <p>This vocabulary is explicit: {@code null} never means {@link #NONE}. Value-aware validation
 * occurs after model metadata construction. In particular, {@code NONE} requires target
 * coordinates to be unique within one operation; duplicate targets are invalid rather than
 * resolved according to an unspecified update order.</p>
 */
public enum ScatterReduction {
    /**
     * Replaces each addressed base value with its single update value.
     *
     * <p>Target coordinates within one operation must be unique. Multiple updates addressing the
     * same target are invalid and are not resolved by first-write, last-write, or any other update
     * order. Detection requires index values and therefore occurs after semantic metadata
     * construction.</p>
     */
    NONE,

    /** Combines the base value and all updates addressed to a target by addition. */
    ADD,

    /** Combines the base value and all updates addressed to a target by multiplication. */
    MUL,

    /** Combines the base value and all updates addressed to a target by maximum. */
    MAX,

    /** Combines the base value and all updates addressed to a target by minimum. */
    MIN
}
