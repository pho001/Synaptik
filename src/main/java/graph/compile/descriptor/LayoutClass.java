package graph.compile.descriptor;

/**
 * Backend-neutral compile-time layout class derived from a compiled tensor snapshot.
 */
public enum LayoutClass {
    DENSE_CONTIGUOUS,
    DENSE_WITH_OFFSET,
    STRIDED_VIEW,
    BROADCAST_ZERO_STRIDE,
    UNKNOWN_OR_COMPLEX
}
