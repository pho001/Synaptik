package tuning.workload;

import config.profile.ExecutionProfile;

import java.util.Objects;

/**
 * Candidate execution environment passed to workload factories.
 *
 * @param profile execution profile currently being validated or measured
 */
public record WorkloadEnvironment(
        ExecutionProfile profile
) {
    public WorkloadEnvironment {
        Objects.requireNonNull(profile, "profile cannot be null");
    }
}
