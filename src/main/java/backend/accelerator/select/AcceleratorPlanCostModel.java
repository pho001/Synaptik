package backend.accelerator.select;

import config.runtime.RuntimeConfig;
import graph.optimizer.partition.PartitionPlan;

/**
 * Cost gate used before accepting an accelerator partition plan.
 */
public final class AcceleratorPlanCostModel {
    private AcceleratorPlanCostModel() {
    }

    /**
     * Accepts or rejects a plan using backend-specific runtime thresholds.
     */
    public static Decision decide(PartitionPlan plan, RuntimeConfig runtimeConfig) {
        if (plan == null) {
            return Decision.reject("missing-plan");
        }
        if (plan.estimatedWork() <= 0L) {
            return Decision.reject("non-positive-estimated-work");
        }
        long minimumEstimatedWork = runtimeConfig == null
                ? 0L
                : runtimeConfig.accelerator().forBackend(plan.backend()).minimumEstimatedWork();
        if (minimumEstimatedWork > 0L && plan.estimatedWork() < minimumEstimatedWork) {
            return Decision.reject("estimated-work-below-minimum");
        }
        return Decision.accept("accepted");
    }

    /**
     * Accelerator plan cost decision.
     *
     * @param accepted whether the partition should use the accelerator backend
     * @param reason stable diagnostic reason for the decision
     */
    public record Decision(
            boolean accepted,
            String reason
    ) {
        public Decision {
            reason = reason == null ? "" : reason;
        }

        /**
         * Creates an accepted decision with a diagnostic reason.
         */
        public static Decision accept(String reason) {
            return new Decision(true, reason);
        }

        /**
         * Creates a rejected decision with a diagnostic reason.
         */
        public static Decision reject(String reason) {
            return new Decision(false, reason);
        }
    }
}
