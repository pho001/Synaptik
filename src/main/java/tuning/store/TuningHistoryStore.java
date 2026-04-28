package tuning.store;

import java.nio.file.Path;
import java.util.List;

/**
 * Persistence abstraction for per-candidate autotune history.
 */
public interface TuningHistoryStore {
    /**
     * Appends one history entry.
     *
     * @param path history destination
     * @param entry entry to append
     */
    void append(Path path, TuningHistoryEntry entry);

    /**
     * Loads all history entries from a store.
     *
     * @param path history source
     * @return entries in store order
     */
    List<TuningHistoryEntry> loadAll(Path path);
}
