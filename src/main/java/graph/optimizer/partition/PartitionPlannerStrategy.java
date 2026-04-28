package graph.optimizer.partition;

/**
 * Search strategy used by partition planning.
 */
public enum PartitionPlannerStrategy {
    GREEDY_MAX_REGION,
    SCORED_CANDIDATE_SEARCH,
    CPU_NATURAL_EXECUTION_REGION
}
