package config.compile;

import planning.partition.PartitionPlannerStrategy;

/**
 * Planner strategy for compile-time backend ownership regions.
 */
public enum RegionOwnershipPlannerStrategy {
    ANCHOR,
    NODE_ORDER_GREEDY,
    SCORED;

    public PartitionPlannerStrategy toPartitionPlannerStrategy() {
        return switch (this) {
            case ANCHOR -> PartitionPlannerStrategy.ANCHOR_MAX_REGION;
            case NODE_ORDER_GREEDY -> PartitionPlannerStrategy.GREEDY_MAX_REGION;
            case SCORED -> PartitionPlannerStrategy.SCORED_CANDIDATE_SEARCH;
        };
    }
}
