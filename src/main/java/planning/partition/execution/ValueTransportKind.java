package planning.partition.execution;

/**
 * How a partition value is transported across unit or partition boundaries.
 */
public enum ValueTransportKind {
    MATERIALIZED,
    VIRTUAL,
    CONTINUATION
}
