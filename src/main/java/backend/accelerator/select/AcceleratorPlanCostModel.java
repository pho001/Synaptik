package backend.accelerator.select;

import config.runtime.RuntimeConfig;
import graph.optimizer.partition.PartitionPlan;

public final class AcceleratorPlanCostModel {
    private AcceleratorPlanCostModel() {
    }

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

    public record Decision(
            boolean accepted,
            String reason
    ) {
        public Decision {
            reason = reason == null ? "" : reason;
        }

        public static Decision accept(String reason) {
            return new Decision(true, reason);
        }

        public static Decision reject(String reason) {
            return new Decision(false, reason);
        }
    }
}
