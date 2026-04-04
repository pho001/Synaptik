package tuning.store;

import java.nio.file.Path;
import java.util.Optional;

public interface BestProfileStore {
    void save(Path path, BestProfileRecord record);

    Optional<BestProfileRecord> load(Path path);
}
