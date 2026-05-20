package graph.compile.planning.region;

/**
 * How a region value is transported across unit or region boundaries.
 */
public enum ValueTransportKind {
    MATERIALIZED,
    VIRTUAL,
    CONTINUATION
}
