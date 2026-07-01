package config.compile;

import planning.partition.PartitionPlannerStrategy;

/**
 * Planner strategy for compile-time backend ownership partitions.
 */
public enum PartitionOwnershipPlannerStrategy {
    ANCHOR,
    NODE_ORDER_GREEDY,
    SCORED;

    public PartitionPlannerStrategy toPartitionPlannerStrategy() {
        return switch (this) {
            case ANCHOR -> PartitionPlannerStrategy.ANCHOR_MAX_PARTITION;
            case NODE_ORDER_GREEDY -> PartitionPlannerStrategy.GREEDY_MAX_PARTITION;
            case SCORED -> PartitionPlannerStrategy.SCORED_CANDIDATE_SEARCH;
        };
    }
}
