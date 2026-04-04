package tuning.store;

import tuning.session.TuningResult;

import java.nio.file.Path;

public interface TuningResultStore {
    void save(Path path, TuningResult result);
}
