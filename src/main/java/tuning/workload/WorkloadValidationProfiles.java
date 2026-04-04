package tuning.workload;

import backend.ApproxMode;
import config.profile.ExecutionProfile;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;

final class WorkloadValidationProfiles {
    private WorkloadValidationProfiles() {
    }

    static ExecutionProfile baselineFor(ExecutionProfile candidateProfile) {
        if (candidateProfile == null) {
            throw new IllegalArgumentException("candidateProfile cannot be null");
        }
        return new ExecutionProfile(
                candidateProfile.profileName() + "_baseline",
                candidateProfile.candidateName() + "_baseline",
                candidateProfile.dataType(),
                candidateProfile.mode(),
                config.optimizer.OptimizerConfig.noOptimization(),
                new RuntimeConfig(
                        candidateProfile.runtime().kernel(),
                        new ApproximationConfig(ApproxMode.OFF, true),
                        BlasConfig.disabled()
                ),
                candidateProfile.workload()
        );
    }
}
