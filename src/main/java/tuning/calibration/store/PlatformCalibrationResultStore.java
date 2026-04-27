package tuning.calibration.store;

import tuning.calibration.PlatformCalibrationResult;

import java.nio.file.Path;

public interface PlatformCalibrationResultStore {
    void save(Path path, PlatformCalibrationResult result);
}
