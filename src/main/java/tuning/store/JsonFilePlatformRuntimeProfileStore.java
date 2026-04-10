package tuning.store;

import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;

import java.nio.file.Path;
import java.util.Optional;

public final class JsonFilePlatformRuntimeProfileStore implements PlatformRuntimeProfileStore {
    private final PlatformRuntimeProfile fallbackProfile;

    public JsonFilePlatformRuntimeProfileStore(PlatformRuntimeProfile fallbackProfile) {
        this.fallbackProfile = fallbackProfile;
    }

    @Override
    public void save(Path path, PlatformRuntimeProfile profile) {
        PlatformRuntimeProfileIO.save(path, profile);
    }

    @Override
    public Optional<PlatformRuntimeProfile> load(Path path) {
        if (fallbackProfile == null) {
            return Optional.empty();
        }
        return Optional.of(PlatformRuntimeProfileIO.loadOrDefault(path, fallbackProfile));
    }
}
