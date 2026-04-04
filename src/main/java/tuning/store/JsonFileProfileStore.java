package tuning.store;

import benchmark.OptimizerProfileIO;
import config.profile.ExecutionProfile;

import java.nio.file.Path;
import java.util.Optional;

public final class JsonFileProfileStore implements ProfileStore {
    private final ExecutionProfile fallbackProfile;

    public JsonFileProfileStore(ExecutionProfile fallbackProfile) {
        this.fallbackProfile = fallbackProfile;
    }

    @Override
    public void save(Path path, ExecutionProfile profile) {
        if (path == null) {
            throw new IllegalArgumentException("path cannot be null");
        }
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        OptimizerProfileIO.saveExecutionProfile(path, profile);
    }

    @Override
    public Optional<ExecutionProfile> load(Path path) {
        if (path == null) {
            throw new IllegalArgumentException("path cannot be null");
        }
        if (fallbackProfile == null) {
            return Optional.empty();
        }
        return Optional.of(OptimizerProfileIO.loadExecutionProfileOrDefault(path, fallbackProfile));
    }
}
