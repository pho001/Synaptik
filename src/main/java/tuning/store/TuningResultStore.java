package tuning.store;

import tuning.autotune.TuningResult;

import java.nio.file.Path;

public interface TuningResultStore {
    void save(Path path, TuningResult result);
}
