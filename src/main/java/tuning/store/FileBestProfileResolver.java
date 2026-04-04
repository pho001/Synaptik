package tuning.store;

import config.profile.ExecutionProfile;

import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

public final class FileBestProfileResolver implements BestProfileResolver {
    private final BestProfileStore store;

    public FileBestProfileResolver(BestProfileStore store) {
        this.store = Objects.requireNonNull(store, "store cannot be null");
    }

    @Override
    public Optional<ExecutionProfile> resolve(Path path, HardwareFingerprint hardware, WorkloadFingerprint workload) {
        return store.load(path)
                .filter(record -> record.hardware().key().equals(hardware.key()))
                .filter(record -> record.workload().key().equals(workload.key()))
                .map(BestProfileRecord::profile);
    }
}
