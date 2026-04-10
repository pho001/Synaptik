package tuning.store;

import tuning.session.PlatformCalibrationResult;

import java.nio.file.Path;

public interface PlatformCalibrationResultStore {
    void save(Path path, PlatformCalibrationResult result);
}
