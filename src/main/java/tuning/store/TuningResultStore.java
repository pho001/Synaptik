package tuning.store;

import tuning.autotune.TuningResult;

import java.nio.file.Path;

/**
 * Persistence abstraction for complete autotune results.
 */
public interface TuningResultStore {
    /**
     * Saves an autotune result.
     *
     * @param path destination path
     * @param result result to save
     */
    void save(Path path, TuningResult result);
}
