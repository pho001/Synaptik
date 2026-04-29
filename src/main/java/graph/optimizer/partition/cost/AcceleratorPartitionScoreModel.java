package graph.optimizer.partition.cost;

import config.optimizer.MetalTransferModel;
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
     * Scores an accepted candidate while accounting for accelerator transfer pressure.
     *
     * <p>This is intended for device ownership regions such as Metal. Larger
     * device regions are useful when they reduce intermediate materialization,
     * but expensive when they require large CPU/device boundary transfers. The
     * transfer model is deliberately explicit so graph autotune can later vary
     * the policy without changing legality rules.</p>
     *
     * @param metrics candidate structural metrics
     * @param estimatedWork backend work estimate
     * @param transfers estimated transfer/materialization metrics
     * @param policy structural/work score policy
     * @param transferPolicy transfer score policy
     * @return transfer-aware score, or negative infinity when candidate data is not usable
     */
    public static double acceptedScore(
            CandidateMetrics metrics,
            long estimatedWork,
            TransferMetrics transfers,
            PlannerPolicy policy,
            TransferPolicy transferPolicy
    ) {
        double base = acceptedScore(metrics, estimatedWork, policy);
        if (!Double.isFinite(base)) {
            return base;
        }
        TransferMetrics resolvedTransfers = transfers == null ? TransferMetrics.none() : transfers;
        TransferPolicy resolvedPolicy = transferPolicy == null ? TransferPolicy.defaults() : transferPolicy;
        return base
                - resolvedTransfers.inputBytes() * resolvedPolicy.inputBytePenalty()
                - resolvedTransfers.outputBytes() * resolvedPolicy.outputBytePenalty()
                + resolvedTransfers.avoidedIntermediateBytes() * resolvedPolicy.avoidedIntermediateByteCredit();
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
     * Estimated accelerator boundary transfer metrics for one candidate.
     *
     * @param inputBytes bytes that must cross into the accelerator region
     * @param outputBytes bytes that must leave the accelerator region
     * @param avoidedIntermediateBytes bytes of selected internal intermediates kept inside the region
     */
    public record TransferMetrics(
            long inputBytes,
            long outputBytes,
            long avoidedIntermediateBytes
    ) {
        public TransferMetrics {
            inputBytes = Math.max(0L, inputBytes);
            outputBytes = Math.max(0L, outputBytes);
            avoidedIntermediateBytes = Math.max(0L, avoidedIntermediateBytes);
        }

        /**
         * Returns an empty transfer metric value.
         *
         * @return zeroed transfer metrics
         */
        public static TransferMetrics none() {
            return new TransferMetrics(0L, 0L, 0L);
        }
    }

    /**
     * Weights used by transfer-aware accelerator scoring.
     *
     * @param inputBytePenalty score penalty per input byte copied into a device region
     * @param outputBytePenalty score penalty per output byte copied out of a device region
     * @param avoidedIntermediateByteCredit score credit per intermediate byte kept inside a device region
     */
    public record TransferPolicy(
            double inputBytePenalty,
            double outputBytePenalty,
            double avoidedIntermediateByteCredit
    ) {
        public TransferPolicy {
            inputBytePenalty = Math.max(0.0d, inputBytePenalty);
            outputBytePenalty = Math.max(0.0d, outputBytePenalty);
            avoidedIntermediateByteCredit = Math.max(0.0d, avoidedIntermediateByteCredit);
        }

        /**
         * Returns the conservative default transfer policy for current Metal execution.
         *
         * <p>Outputs are penalized more than inputs because the current bridge
         * synchronously copies native results back into Java arrays. Future
         * shared-buffer/device-resident execution can tune this policy down.</p>
         *
         * @return default transfer score policy
         */
        public static TransferPolicy defaults() {
            return new TransferPolicy(0.05d, 0.10d, 0.025d);
        }

        /**
         * Builds a transfer policy from a graph-level Metal transfer model.
         *
         * @param model transfer model, or {@code null} for conservative defaults
         * @return transfer score policy
         */
        public static TransferPolicy fromMetalTransferModel(MetalTransferModel model) {
            MetalTransferModel resolved = model == null ? MetalTransferModel.CONSERVATIVE : model;
            return new TransferPolicy(
                    resolved.inputBytePenalty(),
                    resolved.outputBytePenalty(),
                    resolved.avoidedIntermediateByteCredit()
            );
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
