package graph.compile.planning.partition;

/**
 * Search strategy used by partition planning.
 */
public enum PartitionPlannerStrategy {
    ANCHOR_MAX_REGION,
    GREEDY_MAX_REGION,
    SCORED_CANDIDATE_SEARCH,
    CPU_NATURAL_EXECUTION_REGION
}
