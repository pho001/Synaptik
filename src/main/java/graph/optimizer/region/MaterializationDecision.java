package graph.optimizer.region;

/**
 * Memory planning decision for a region value.
 */
public enum MaterializationDecision {
    MATERIALIZE,
    VIRTUALIZE,
    CONTINUE
}
