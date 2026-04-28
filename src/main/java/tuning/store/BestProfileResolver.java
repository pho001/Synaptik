package tuning.store;

import config.profile.ExecutionProfile;

import java.nio.file.Path;
import java.util.Optional;

/**
 * Resolves a persisted best execution profile for a hardware/workload pair.
 */
public interface BestProfileResolver {
    /**
     * Looks up a profile matching the supplied fingerprints.
     *
     * @param path store path
     * @param hardware target hardware fingerprint
     * @param workload target workload fingerprint
     * @return matching execution profile, if any
     */
    Optional<ExecutionProfile> resolve(Path path, HardwareFingerprint hardware, WorkloadFingerprint workload);
}
