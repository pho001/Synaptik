package tuning.store;

import java.nio.file.Path;
import java.util.Optional;

/**
 * Persistence abstraction for the selected best profile from an autotune run.
 *
 * <p>Implementations may write files or external stores. They should document
 * their own concurrency guarantees; file-backed stores in this package are
 * intended for simple process-local use.</p>
 */
public interface BestProfileStore {
    /**
     * Saves or replaces a best-profile record.
     *
     * @param path destination path
     * @param record record to save
     */
    void save(Path path, BestProfileRecord record);

    /**
     * Loads a best-profile record.
     *
     * @param path source path
     * @return record when present and readable
     */
    Optional<BestProfileRecord> load(Path path);
}
