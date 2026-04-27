package tuning.calibration.store;

public final class CalibrationSaveHelper {
    private CalibrationSaveHelper() {
    }

    public static CalibrationRunStore runStore(CalibrationArtifactLayout layout) {
        return new CalibrationRunStore(layout);
    }
}
