package tuning.store;

import java.nio.file.Path;
import java.util.List;

public interface TuningHistoryStore {
    void append(Path path, TuningHistoryEntry entry);

    List<TuningHistoryEntry> loadAll(Path path);
}
