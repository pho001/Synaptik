package graph.optimizer.partition.cost;

import config.optimizer.PartitionConfig;

public final class AcceleratorPartitionScoreModel {
    private AcceleratorPartitionScoreModel() {
    }

    public static double structuralScore(CandidateMetrics metrics, PlannerPolicy policy) {
        if (metrics == null || policy == null) {
            return Double.NEGATIVE_INFINITY;
        }
        return metrics.nodeCount() * policy.nodeWeight()
                + metrics.internalEdgeCount() * policy.internalEdgeWeight()
                + metrics.mergeNodeCount() * policy.mergeNodeBonus()
                + metrics.tailDepth() * policy.tailDepthWeight()
                - metrics.externalInputCount() * policy.externalInputPenalty();
    }

    public static double acceptedScore(CandidateMetrics metrics, long estimatedWork, PlannerPolicy policy) {
        if (metrics == null || policy == null || estimatedWork <= 0L) {
            return Double.NEGATIVE_INFINITY;
        }
        return structuralScore(metrics, policy) + estimatedWork * policy.workWeight();
    }

    public record CandidateMetrics(
            int nodeCount,
            int internalEdgeCount,
            int externalInputCount,
            int mergeNodeCount,
            int tailDepth
    ) {
        public CandidateMetrics {
            nodeCount = Math.max(0, nodeCount);
            internalEdgeCount = Math.max(0, internalEdgeCount);
            externalInputCount = Math.max(0, externalInputCount);
            mergeNodeCount = Math.max(0, mergeNodeCount);
            tailDepth = Math.max(0, tailDepth);
        }
    }

    public record PlannerPolicy(
            int maxSearchNodes,
            int maxVisitedCandidates,
            double nodeWeight,
            double internalEdgeWeight,
            double mergeNodeBonus,
            double tailDepthWeight,
            double externalInputPenalty,
            double workWeight
    ) {
        public PlannerPolicy {
            maxSearchNodes = Math.max(1, maxSearchNodes);
            maxVisitedCandidates = Math.max(1, maxVisitedCandidates);
        }

        public static PlannerPolicy defaults() {
            return new PlannerPolicy(
                    16,
                    512,
                    1000.0,
                    120.0,
                    450.0,
                    80.0,
                    60.0,
                    1.0
            );
        }

        public static PlannerPolicy fromConfig(PartitionConfig config) {
            PartitionConfig resolved = config == null ? PartitionConfig.defaults() : config;
            return new PlannerPolicy(
                    resolved.maxSearchNodes(),
                    resolved.maxVisitedCandidates(),
                    resolved.nodeWeight(),
                    resolved.internalEdgeWeight(),
                    resolved.mergeNodeBonus(),
                    resolved.tailDepthWeight(),
                    resolved.externalInputPenalty(),
                    resolved.workWeight()
            );
        }
    }
}
