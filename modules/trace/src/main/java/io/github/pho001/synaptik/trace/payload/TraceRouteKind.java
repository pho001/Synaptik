package io.github.pho001.synaptik.trace.payload;

/**
 * Classifies the execution mechanism selected for a traced prepared unit.
 *
 * <p>This closed vocabulary describes only the mechanisms required by the current outcome
 * payloads. It is not a universal registry to which every backend route must conform.</p>
 */
public enum TraceRouteKind {
    /** A backend-owned custom compute kernel implements the prepared unit. */
    CUSTOM_KERNEL,
    /** A backend-owned compiled graph executable implements the prepared unit. */
    GRAPH_EXECUTABLE
}
