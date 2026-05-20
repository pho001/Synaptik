package graph.compile.planning.region;

/**
 * Memory planning decision for a region value.
 */
public enum MaterializationDecision {
    MATERIALIZE,
    VIRTUALIZE,
    CONTINUE
}
