package backend.accelerator.select;

import config.runtime.RuntimeConfig;
import planning.partition.cost.AcceleratorPartitionScoreModel;
import planning.partition.PartitionPlan;

/**
 * Cost gate used before accepting an accelerator partition plan.
 */
public final class AcceleratorPlanCostModel {
    private static final String PROFILE_DERIVED_PRESET = "PROFILE_DERIVED";

    private AcceleratorPlanCostModel() {
    }

    /**
     * Accepts or rejects a plan using backend-specific runtime thresholds.
     */
    public static Decision decide(PartitionPlan plan, RuntimeConfig runtimeConfig) {
        if (plan == null) {
            return Decision.reject("missing-plan");
        }
        AcceleratorPartitionScoreModel.MaterializationCostSummary costSummary = summarize(plan, runtimeConfig);
        if (plan.estimatedWork() <= 0L) {
            return Decision.reject("non-positive-estimated-work", costSummary);
        }
        long minimumEstimatedWork = runtimeConfig == null
                ? 0L
                : runtimeConfig.accelerator().forBackend(plan.backend()).minimumEstimatedWork();
        if (minimumEstimatedWork > 0L && plan.estimatedWork() < minimumEstimatedWork) {
            return Decision.reject("estimated-work-below-minimum", costSummary);
        }
        if ("rejected-materialization-cost".equals(costSummary.reasonCode())) {
            return Decision.reject("rejected-materialization-cost", costSummary);
        }
        return Decision.accept("accepted", costSummary);
    }

    /**
     * Builds the static prepare-time cost summary for a partition plan.
     *
     * @param plan partition plan
     * @return static cost summary, or {@code null} for missing plans
     */
    public static AcceleratorPartitionScoreModel.MaterializationCostSummary summarize(PartitionPlan plan) {
        return summarize(plan, AcceleratorPartitionScoreModel.StaticCostPreset.conservative());
    }

    /**
     * Builds a runtime-aware prepare-time cost summary for a partition plan using the PROFILE_DERIVED preset.
     *
     * @param plan partition plan
     * @param runtimeConfig audited runtime config assembled before preparation
     * @return runtime-derived cost summary, or {@code null} for missing plans
     */
    public static AcceleratorPartitionScoreModel.MaterializationCostSummary summarize(PartitionPlan plan, RuntimeConfig runtimeConfig) {
        if (plan == null) {
            return null;
        }
        return summarize(
                plan,
                ProfileDerivedAcceleratorCostFactors.fromRuntimeConfig(runtimeConfig, plan.backend()).toStaticCostPreset()
        );
    }

    private static AcceleratorPartitionScoreModel.MaterializationCostSummary summarize(
            PartitionPlan plan,
            AcceleratorPartitionScoreModel.StaticCostPreset preset
    ) {
        if (plan == null) {
            return null;
        }
        int nodeCount = plan.nodeIds().size();
        var metrics = new AcceleratorPartitionScoreModel.CandidateMetrics(
                nodeCount,
                Math.max(0, nodeCount - 1),
                plan.externalInputNodeIds().size(),
                0,
                Math.max(0, nodeCount - 1)
        );
        var signals = new AcceleratorPartitionScoreModel.MaterializationSignals(
                plan.externalInputNodeIds().size() + plan.producedOutputNodeIds().size(),
                0L,
                0L,
                0L,
                0L,
                0L,
                "BUFFER_BINDING",
                "UNKNOWN"
        );
        return AcceleratorPartitionScoreModel.scoreMaterializationAware(
                metrics,
                plan.estimatedWork(),
                signals,
                AcceleratorPartitionScoreModel.PlannerPolicy.defaults(),
                preset
        );
    }

    /**
     * Accelerator plan cost decision.
     *
     * @param accepted whether the partition should use the accelerator backend
     * @param reason stable diagnostic reason for the decision
     * @param costSummary static cost summary, if a plan was available
     */
    public record Decision(
            boolean accepted,
            String reason,
            AcceleratorPartitionScoreModel.MaterializationCostSummary costSummary
    ) {
        public Decision {
            reason = reason == null ? "" : reason;
        }

        public Decision(boolean accepted, String reason) {
            this(accepted, reason, null);
        }

        /**
         * Creates an accepted decision with a diagnostic reason.
         */
        public static Decision accept(String reason) {
            return new Decision(true, reason, null);
        }

        /**
         * Creates an accepted decision with a diagnostic reason and summary.
         */
        public static Decision accept(
                String reason,
                AcceleratorPartitionScoreModel.MaterializationCostSummary costSummary
        ) {
            return new Decision(true, reason, costSummary);
        }

        /**
         * Creates a rejected decision with a diagnostic reason.
         */
        public static Decision reject(String reason) {
            return new Decision(false, reason, null);
        }

        /**
         * Creates a rejected decision with a diagnostic reason and summary.
         */
        public static Decision reject(
                String reason,
                AcceleratorPartitionScoreModel.MaterializationCostSummary costSummary
        ) {
            return new Decision(false, reason, costSummary);
        }
    }
}
