package tuning.candidate;

import config.profile.ExecutionProfile;

/**
 * Legacy executable-profile fingerprint facade.
 * New code should use {@link ExecutableProfileFingerprint} or {@link CandidateIdentityFingerprint}
 * depending on whether it needs runnable-profile identity or structured candidate identity.
 */
public final class CandidateFingerprint {
    private CandidateFingerprint() {
    }

    public static String of(Candidate candidate) {
        return ExecutableProfileFingerprint.of(candidate);
    }

    public static String of(ExecutionProfile profile) {
        return ExecutableProfileFingerprint.of(profile);
    }
}
