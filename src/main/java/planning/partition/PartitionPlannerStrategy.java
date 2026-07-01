package planning.partition;

/**
 * Search strategy used by partition planning.
 */
public enum PartitionPlannerStrategy {
    ANCHOR_MAX_PARTITION,
    GREEDY_MAX_PARTITION,
    SCORED_CANDIDATE_SEARCH,
    CPU_NATURAL_EXECUTION_PARTITION
}
