package tuning.store;

import config.profile.ExecutionProfile;

import java.nio.file.Path;
import java.util.Optional;

public interface ProfileStore {
    void save(Path path, ExecutionProfile profile);

    Optional<ExecutionProfile> load(Path path);
}
