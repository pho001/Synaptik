package config.optimizer;

import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionPlannerStrategy;

/**
 * Search and scoring configuration for backend partition planning.
 *
 * <p>Partition planning chooses candidate regions for CPU or accelerator execution. The weights shape
 * the score model used by planner strategies. Search limits are normalized to at least one.</p>
 *
 * @param maxSearchNodes maximum nodes considered in one search expansion
 * @param maxVisitedCandidates maximum candidate partitions visited
 * @param nodeWeight per-node score weight
 * @param internalEdgeWeight score weight for edges inside a partition
 * @param mergeNodeBonus score bonus for merging nodes into a region
 * @param tailDepthWeight score weight for region tail depth
 * @param externalInputPenalty score penalty for external inputs
 * @param workWeight score weight for estimated work
 * @param plannerStrategy partition planner strategy; {@code null} uses greedy max-region
 * @param target preferred partition target; {@code null} uses auto
 * @param metalTransferModel transfer-cost preset used by scored Metal planning
 */
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
        PartitionTarget target,
        MetalTransferModel metalTransferModel
) {
    public PartitionConfig {
        maxSearchNodes = Math.max(1, maxSearchNodes);
        maxVisitedCandidates = Math.max(1, maxVisitedCandidates);
        plannerStrategy = plannerStrategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : plannerStrategy;
        target = target == null ? PartitionTarget.AUTO : target;
        metalTransferModel = metalTransferModel == null ? MetalTransferModel.CONSERVATIVE : metalTransferModel;
    }

    public PartitionConfig(
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
        this(
                maxSearchNodes,
                maxVisitedCandidates,
                nodeWeight,
                internalEdgeWeight,
                mergeNodeBonus,
                tailDepthWeight,
                externalInputPenalty,
                workWeight,
                plannerStrategy,
                target,
                MetalTransferModel.CONSERVATIVE
        );
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
                PartitionTarget.AUTO,
                MetalTransferModel.CONSERVATIVE
        );
    }

    /**
     * @return default partition planner configuration
     */
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
                PartitionTarget.AUTO,
                MetalTransferModel.CONSERVATIVE
        );
    }

    /**
     * Returns a copy targeting a specific backend family.
     *
     * @param newTarget replacement target; {@code null} uses auto
     * @return updated partition config
     */
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
                newTarget,
                metalTransferModel
        );
    }

    /**
     * Returns a copy with a different Metal transfer-cost model.
     *
     * <p>This affects scored Metal profitability only. It does not make unsupported operations,
     * dtypes, layouts, masks, or native bridge paths legal.</p>
     *
     * @param newMetalTransferModel replacement transfer model; {@code null} uses conservative
     * @return updated partition config
     */
    public PartitionConfig withMetalTransferModel(MetalTransferModel newMetalTransferModel) {
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
                target,
                newMetalTransferModel
        );
    }
}
