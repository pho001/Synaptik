package planning.partition.execution;

/**
 * Memory planning decision for a partition value.
 */
public enum MaterializationDecision {
    MATERIALIZE,
    VIRTUALIZE,
    CONTINUE
}
