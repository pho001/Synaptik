package tuning.store;

import config.profile.ExecutionProfile;

import java.nio.file.Path;
import java.util.Optional;

/**
 * Persistence abstraction for complete execution profiles.
 */
public interface ProfileStore {
    /**
     * Saves an execution profile.
     *
     * @param path destination path
     * @param profile profile to save
     */
    void save(Path path, ExecutionProfile profile);

    /**
     * Loads an execution profile.
     *
     * @param path source path
     * @return profile when present and readable
     */
    Optional<ExecutionProfile> load(Path path);
}
