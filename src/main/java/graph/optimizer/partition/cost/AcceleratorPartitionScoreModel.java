package graph.optimizer.partition.cost;

import config.optimizer.PartitionConfig;

/**
 * Scoring helpers for accelerator-oriented partition search.
 *
 * <p>Structural score rewards larger, internally connected regions and penalizes boundary inputs. Accepted score adds
 * backend work estimates so planners can prefer candidates that amortize launch or transfer costs.
 */
public final class AcceleratorPartitionScoreModel {
    private AcceleratorPartitionScoreModel() {
    }

    /**
     * Scores candidate structure without backend work estimates.
     *
     * @param metrics candidate structural metrics
     * @param policy score weights and search limits
     * @return score, or negative infinity when inputs are missing
     */
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

    /**
     * Scores an accepted lowered candidate with backend work estimates.
     *
     * @param metrics candidate structural metrics
     * @param estimatedWork backend work estimate
     * @param policy score weights and search limits
     * @return score, or negative infinity when candidate data is not usable
     */
    public static double acceptedScore(CandidateMetrics metrics, long estimatedWork, PlannerPolicy policy) {
        if (metrics == null || policy == null || estimatedWork <= 0L) {
            return Double.NEGATIVE_INFINITY;
        }
        return structuralScore(metrics, policy) + estimatedWork * policy.workWeight();
    }

    /**
     * Structural metrics used by partition scoring.
     *
     * @param nodeCount selected node count
     * @param internalEdgeCount edges entirely inside the candidate
     * @param externalInputCount inputs crossing into the candidate
     * @param mergeNodeCount selected nodes with multiple selected inputs
     * @param tailDepth length of the candidate tail
     */
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

    /**
     * Search limits and score weights for partition planning.
     *
     * @param maxSearchNodes maximum nodes to include while expanding a candidate
     * @param maxVisitedCandidates maximum candidates to visit during scored search
     * @param nodeWeight score weight per selected node
     * @param internalEdgeWeight score weight per internal edge
     * @param mergeNodeBonus score bonus for merge-heavy regions
     * @param tailDepthWeight score weight for tail depth
     * @param externalInputPenalty score penalty per external input
     * @param workWeight score weight for estimated backend work
     */
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

        /**
         * Returns default scorer policy.
         *
         * @return default policy
         */
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

        /**
         * Builds a scorer policy from partition configuration.
         *
         * @param config partition configuration, or {@code null} for defaults
         * @return scorer policy
         */
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
