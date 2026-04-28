package tuning.store;

import config.profile.PlatformRuntimeProfile;

import java.nio.file.Path;
import java.util.Optional;

/**
 * Persistence abstraction for platform runtime profiles.
 *
 * <p>Platform calibration uses this store to save the final calibrated runtime
 * profile when its request supplies an output path.</p>
 */
public interface PlatformRuntimeProfileStore {
    /**
     * Saves a runtime profile.
     *
     * @param path destination path
     * @param profile runtime profile to save
     */
    void save(Path path, PlatformRuntimeProfile profile);

    /**
     * Loads a runtime profile.
     *
     * @param path source path
     * @return profile when present and readable
     */
    Optional<PlatformRuntimeProfile> load(Path path);
}
