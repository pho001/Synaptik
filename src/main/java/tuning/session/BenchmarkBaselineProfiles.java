package tuning.session;

import backend.ApproxMode;
import config.profile.ExecutionProfile;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;

final class BenchmarkBaselineProfiles {
    private BenchmarkBaselineProfiles() {
    }

    static ExecutionProfile noOptimization(ExecutionProfile reference) {
        return new ExecutionProfile(
                reference.profileName() + "_baseline_no_opt",
                "BASELINE_NO_OPT",
                reference.dataType(),
                reference.mode(),
                config.optimizer.OptimizerConfig.noOptimization(),
                reference.runtime(),
                reference.workload()
        );
    }

    static ExecutionProfile noOptimizationConservativeRuntime(ExecutionProfile reference) {
        return new ExecutionProfile(
                reference.profileName() + "_baseline_no_opt_conservative_runtime",
                "BASELINE_NO_OPT_CONSERVATIVE_RUNTIME",
                reference.dataType(),
                reference.mode(),
                config.optimizer.OptimizerConfig.noOptimization(),
                new RuntimeConfig(
                        reference.runtime().kernel(),
                        new ApproximationConfig(ApproxMode.OFF, true),
                        BlasConfig.disabled()
                ),
                reference.workload()
        );
    }
}
