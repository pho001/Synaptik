package operations.index;

/**
 * Reduction policy for functional scatter writes.
 */
public enum ScatterReduction {
    NONE,
    ADD,
    MUL,
    MAX,
    MIN
}
