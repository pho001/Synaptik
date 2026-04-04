package tuning.store;

import config.profile.ExecutionProfile;

import java.nio.file.Path;
import java.util.Optional;

public interface BestProfileResolver {
    Optional<ExecutionProfile> resolve(Path path, HardwareFingerprint hardware, WorkloadFingerprint workload);
}
