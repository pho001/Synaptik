package tuning.workload;

import config.profile.ExecutionProfile;

import java.util.Objects;

public record WorkloadEnvironment(
        ExecutionProfile profile
) {
    public WorkloadEnvironment {
        Objects.requireNonNull(profile, "profile cannot be null");
    }
}
