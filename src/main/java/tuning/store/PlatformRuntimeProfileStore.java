package tuning.store;

import config.profile.PlatformRuntimeProfile;

import java.nio.file.Path;
import java.util.Optional;

public interface PlatformRuntimeProfileStore {
    void save(Path path, PlatformRuntimeProfile profile);

    Optional<PlatformRuntimeProfile> load(Path path);
}
