package config.optimizer;

import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionPlannerStrategy;

public record PartitionConfig(
        int maxSearchNodes,
        int maxVisitedCandidates,
        double nodeWeight,
        double internalEdgeWeight,
        double mergeNodeBonus,
        double tailDepthWeight,
        double externalInputPenalty,
        double workWeight,
        PartitionPlannerStrategy plannerStrategy,
        PartitionTarget target
) {
    public PartitionConfig {
        maxSearchNodes = Math.max(1, maxSearchNodes);
        maxVisitedCandidates = Math.max(1, maxVisitedCandidates);
        plannerStrategy = plannerStrategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : plannerStrategy;
        target = target == null ? PartitionTarget.AUTO : target;
    }

    public PartitionConfig(
            int maxSearchNodes,
            int maxVisitedCandidates,
            double nodeWeight,
            double internalEdgeWeight,
            double mergeNodeBonus,
            double tailDepthWeight,
            double externalInputPenalty,
            double workWeight
    ) {
        this(
                maxSearchNodes,
                maxVisitedCandidates,
                nodeWeight,
                internalEdgeWeight,
                mergeNodeBonus,
                tailDepthWeight,
                externalInputPenalty,
                workWeight,
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                PartitionTarget.AUTO
        );
    }

    public static PartitionConfig defaults() {
        return new PartitionConfig(
                16,
                512,
                1000.0,
                120.0,
                450.0,
                80.0,
                60.0,
                1.0,
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                PartitionTarget.AUTO
        );
    }

    public PartitionConfig withTarget(PartitionTarget newTarget) {
        return new PartitionConfig(
                maxSearchNodes,
                maxVisitedCandidates,
                nodeWeight,
                internalEdgeWeight,
                mergeNodeBonus,
                tailDepthWeight,
                externalInputPenalty,
                workWeight,
                plannerStrategy,
                newTarget
        );
    }
}
