package graph.compile.planning.partition.cost;

import config.compile.PartitionSearchConfig;
import graph.optimizer.cost.CostComponent;
import graph.optimizer.cost.CostScore;

import java.util.List;

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
        TransferMetrics resolvedTransfers = transfers == null ? TransferMetrics.none() : transfers;
        TransferPolicy resolvedPolicy = transferPolicy == null ? TransferPolicy.defaults() : transferPolicy;
        return scoreMaterializationAware(
                metrics,
                estimatedWork,
                new MaterializationSignals(
                        0,
                        resolvedTransfers.inputBytes(),
                        resolvedTransfers.outputBytes(),
                        0L,
                        0L,
                        resolvedTransfers.avoidedIntermediateBytes(),
                        "TRANSFER_ONLY",
                        ""
                ),
                policy,
                StaticCostPreset.fromTransferPolicy("TRANSFER_POLICY", resolvedPolicy)
        ).finalScore();
    }

    /**
     * Scores a lowered accelerator candidate with static materialization, fallback, and dispatch signals.
     *
     * @param metrics candidate structural metrics
     * @param estimatedWork backend work estimate
     * @param signals static materialization and fallback signals
     * @param plannerPolicy structural/work score policy
     * @param preset named static cost preset
     * @return explainable score summary
     */
    public static MaterializationCostSummary scoreMaterializationAware(
            CandidateMetrics metrics,
            long estimatedWork,
            MaterializationSignals signals,
            PlannerPolicy plannerPolicy,
            StaticCostPreset preset
    ) {
        MaterializationSignals resolvedSignals = signals == null ? MaterializationSignals.none() : signals;
        StaticCostPreset resolvedPreset = preset == null ? StaticCostPreset.conservative() : preset;
        if (metrics == null || plannerPolicy == null || estimatedWork <= 0L) {
            return new MaterializationCostSummary(
                    resolvedPreset.name(),
                    resolvedSignals.boundaryCount(),
                    resolvedSignals.estimatedTransferBytes(),
                    resolvedSignals.layoutFallbackBytes(),
                    Math.max(0L, estimatedWork),
                    resolvedSignals.avoidedIntermediateBytes(),
                    resolvedPreset.dispatchOverhead(),
                    Double.NEGATIVE_INFINITY,
                    "rejected-non-positive-work",
                    resolvedSignals.fallbackMode(),
                    resolvedSignals.layoutClass()
            );
        }
        double structural = structuralScore(metrics, plannerPolicy);
        double finalScore = structural
                + estimatedWork * plannerPolicy.workWeight() * resolvedPreset.computeWorkCredit()
                + resolvedSignals.avoidedIntermediateBytes() * resolvedPreset.avoidedIntermediateByteCredit()
                - resolvedSignals.boundaryCount() * resolvedPreset.boundaryPenalty()
                - resolvedSignals.uploadBytes() * resolvedPreset.uploadBytePenalty()
                - resolvedSignals.downloadBytes() * resolvedPreset.downloadBytePenalty()
                - resolvedSignals.tensorArrayFallbackBytes() * resolvedPreset.tensorArrayFallbackBytePenalty()
                - resolvedSignals.layoutFallbackBytes() * resolvedPreset.layoutFallbackBytePenalty()
                - resolvedPreset.dispatchOverhead();
        String reason;
        if (!Double.isFinite(finalScore)) {
            reason = "rejected-non-finite-score";
            finalScore = Double.NEGATIVE_INFINITY;
        } else if (finalScore <= 0.0d) {
            reason = "rejected-materialization-cost";
        } else {
            reason = "accepted-static-profitable";
        }
        return new MaterializationCostSummary(
                resolvedPreset.name(),
                resolvedSignals.boundaryCount(),
                resolvedSignals.estimatedTransferBytes(),
                resolvedSignals.layoutFallbackBytes(),
                estimatedWork,
                resolvedSignals.avoidedIntermediateBytes(),
                resolvedPreset.dispatchOverhead(),
                finalScore,
                reason,
                resolvedSignals.fallbackMode(),
                resolvedSignals.layoutClass()
        );
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

    }

    /**
     * Static materialization and fallback signals used by the accelerator score model.
     *
     * @param boundaryCount number of CPU/accelerator boundaries
     * @param uploadBytes estimated bytes copied into the accelerator region
     * @param downloadBytes estimated bytes copied out of the accelerator region
     * @param tensorArrayFallbackBytes bytes exposed to tensor-array fallback paths
     * @param layoutFallbackBytes bytes affected by layout fallback
     * @param avoidedIntermediateBytes bytes kept inside the accelerator region
     * @param fallbackMode fallback mode name
     * @param layoutClass layout class name
     */
    public record MaterializationSignals(
            int boundaryCount,
            long uploadBytes,
            long downloadBytes,
            long tensorArrayFallbackBytes,
            long layoutFallbackBytes,
            long avoidedIntermediateBytes,
            String fallbackMode,
            String layoutClass
    ) {
        public MaterializationSignals {
            boundaryCount = Math.max(0, boundaryCount);
            uploadBytes = Math.max(0L, uploadBytes);
            downloadBytes = Math.max(0L, downloadBytes);
            tensorArrayFallbackBytes = Math.max(0L, tensorArrayFallbackBytes);
            layoutFallbackBytes = Math.max(0L, layoutFallbackBytes);
            avoidedIntermediateBytes = Math.max(0L, avoidedIntermediateBytes);
            fallbackMode = fallbackMode == null ? "" : fallbackMode;
            layoutClass = layoutClass == null ? "" : layoutClass;
        }

        /**
         * Returns estimated transfer bytes crossing the accelerator boundary.
         *
         * @return upload plus download bytes
         */
        public long estimatedTransferBytes() {
            return uploadBytes + downloadBytes;
        }

        /**
         * Returns zeroed materialization signals.
         *
         * @return empty signal value
         */
        public static MaterializationSignals none() {
            return new MaterializationSignals(0, 0L, 0L, 0L, 0L, 0L, "", "");
        }
    }

    /**
     * Named static cost preset. Values are internal constants, not profile-derived costs.
     *
     * @param name preset name surfaced in traces
     * @param boundaryPenalty score penalty per CPU/accelerator boundary
     * @param uploadBytePenalty score penalty per uploaded byte
     * @param downloadBytePenalty score penalty per downloaded byte
     * @param tensorArrayFallbackBytePenalty score penalty per tensor-array fallback byte
     * @param layoutFallbackBytePenalty score penalty per layout fallback byte
     * @param avoidedIntermediateByteCredit score credit per avoided intermediate byte
     * @param dispatchOverhead fixed accelerator dispatch cost
     * @param computeWorkCredit multiplier for estimated compute work
     */
    public record StaticCostPreset(
            String name,
            double boundaryPenalty,
            double uploadBytePenalty,
            double downloadBytePenalty,
            double tensorArrayFallbackBytePenalty,
            double layoutFallbackBytePenalty,
            double avoidedIntermediateByteCredit,
            double dispatchOverhead,
            double computeWorkCredit
    ) {
        public StaticCostPreset {
            name = name == null || name.isBlank() ? "CONSERVATIVE" : name;
            boundaryPenalty = Math.max(0.0d, boundaryPenalty);
            uploadBytePenalty = Math.max(0.0d, uploadBytePenalty);
            downloadBytePenalty = Math.max(0.0d, downloadBytePenalty);
            tensorArrayFallbackBytePenalty = Math.max(0.0d, tensorArrayFallbackBytePenalty);
            layoutFallbackBytePenalty = Math.max(0.0d, layoutFallbackBytePenalty);
            avoidedIntermediateByteCredit = Math.max(0.0d, avoidedIntermediateByteCredit);
            dispatchOverhead = Math.max(0.0d, dispatchOverhead);
            computeWorkCredit = Math.max(0.0d, computeWorkCredit);
        }

        /**
         * Returns the conservative static preset for current copy-heavy accelerator execution.
         *
         * @return conservative preset
         */
        public static StaticCostPreset conservative() {
            return new StaticCostPreset("CONSERVATIVE", 125.0d, 0.05d, 0.10d, 0.10d, 0.08d, 0.025d, 250.0d, 1.0d);
        }

        /**
         * Returns the measured static preset for lower assumed boundary pressure.
         *
         * @return measured preset
         */
        public static StaticCostPreset measured() {
            return new StaticCostPreset("MEASURED", 75.0d, 0.025d, 0.05d, 0.05d, 0.04d, 0.05d, 125.0d, 1.0d);
        }

        /**
         * Returns the aggressive static preset for exploring longer accelerator regions.
         *
         * @return aggressive preset
         */
        public static StaticCostPreset aggressive() {
            return new StaticCostPreset("AGGRESSIVE", 40.0d, 0.01d, 0.02d, 0.02d, 0.02d, 0.10d, 60.0d, 1.0d);
        }

        private static StaticCostPreset fromTransferPolicy(String name, TransferPolicy transferPolicy) {
            TransferPolicy resolved = transferPolicy == null ? TransferPolicy.defaults() : transferPolicy;
            return new StaticCostPreset(
                    name,
                    0.0d,
                    resolved.inputBytePenalty(),
                    resolved.outputBytePenalty(),
                    0.0d,
                    0.0d,
                    resolved.avoidedIntermediateByteCredit(),
                    0.0d,
                    1.0d
            );
        }
    }

    /**
     * Explainable static score summary for traces and reports.
     *
     * @param preset selected static preset name
     * @param boundaryCount CPU/accelerator boundary count
     * @param estimatedTransferBytes upload plus download bytes
     * @param layoutFallbackBytes bytes affected by layout fallback or GPU-side dense materialization
     * @param estimatedComputeWork backend work estimate
     * @param avoidedIntermediateBytes bytes kept inside the accelerator region
     * @param dispatchCost fixed dispatch cost applied
     * @param finalScore final static score
     * @param reasonCode stable accepted/rejected reason code
     * @param fallbackMode fallback mode name
     * @param layoutClass layout class name
     */
    public record MaterializationCostSummary(
            String preset,
            int boundaryCount,
            long estimatedTransferBytes,
            long layoutFallbackBytes,
            long estimatedComputeWork,
            long avoidedIntermediateBytes,
            double dispatchCost,
            double finalScore,
            String reasonCode,
            String fallbackMode,
            String layoutClass
    ) {
        public MaterializationCostSummary {
            preset = preset == null ? "" : preset;
            boundaryCount = Math.max(0, boundaryCount);
            estimatedTransferBytes = Math.max(0L, estimatedTransferBytes);
            layoutFallbackBytes = Math.max(0L, layoutFallbackBytes);
            estimatedComputeWork = Math.max(0L, estimatedComputeWork);
            avoidedIntermediateBytes = Math.max(0L, avoidedIntermediateBytes);
            dispatchCost = Math.max(0.0d, dispatchCost);
            reasonCode = reasonCode == null ? "" : reasonCode;
            fallbackMode = fallbackMode == null ? "" : fallbackMode;
            layoutClass = layoutClass == null ? "" : layoutClass;
        }

        /**
         * Exports this accelerator partition summary through the shared cost vocabulary.
         *
         * <p>This is report-only. It does not change {@link AcceleratorPartitionScoreModel}
         * formulas or accelerator acceptance decisions.</p>
         *
         * @return shared cost score explanation input
         */
        public CostScore toCostScore() {
            return CostScore.of(
                    "AcceleratorPartitionCostModel",
                    "accelerator-partition-materialization",
                    List.of(
                            CostComponent.higherIsBetter(
                                    "finalScore",
                                    finalScore,
                                    "materialization-aware accelerator partition score"
                            ),
                            CostComponent.higherIsBetter(
                                    "estimatedComputeWork",
                                    estimatedComputeWork,
                                    "larger accelerator work can amortize dispatch and transfer cost"
                            ),
                            CostComponent.higherIsBetter(
                                    "avoidedIntermediateBytes",
                                    avoidedIntermediateBytes,
                                    "intermediate bytes retained inside the accelerator region"
                            ),
                            CostComponent.lowerIsBetter(
                                    "boundaryCount",
                                    boundaryCount,
                                    "CPU/accelerator boundaries introduce handoff cost"
                            ),
                            CostComponent.lowerIsBetter(
                                    "estimatedTransferBytes",
                                    estimatedTransferBytes,
                                    "estimated bytes copied across accelerator boundaries"
                            ),
                            CostComponent.lowerIsBetter(
                                    "layoutFallbackBytes",
                                    layoutFallbackBytes,
                                    "bytes affected by layout fallback or dense materialization"
                            ),
                            CostComponent.lowerIsBetter(
                                    "dispatchCost",
                                    dispatchCost,
                                    "fixed accelerator dispatch cost applied by the preset"
                            ),
                            CostComponent.informational(
                                    "preset",
                                    0.0d,
                                    preset
                            ),
                            CostComponent.informational(
                                    "fallbackMode",
                                    0.0d,
                                    fallbackMode
                            ),
                            CostComponent.informational(
                                    "layoutClass",
                                    0.0d,
                                    layoutClass
                            )
                    )
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
                    64,
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
         * Builds a scorer policy from backend planning search configuration.
         *
         * @param config search configuration, or {@code null} for defaults
         * @return scorer policy
         */
        public static PlannerPolicy fromConfig(PartitionSearchConfig config) {
            PartitionSearchConfig resolved = config == null ? PartitionSearchConfig.defaults() : config;
            var weights = resolved.scoreWeights();
            return new PlannerPolicy(
                    resolved.maxSearchNodes(),
                    resolved.maxVisitedCandidates(),
                    weights.nodeWeight(),
                    weights.internalEdgeWeight(),
                    weights.mergeNodeBonus(),
                    weights.tailDepthWeight(),
                    weights.externalInputPenalty(),
                    weights.workWeight()
            );
        }
    }
}
