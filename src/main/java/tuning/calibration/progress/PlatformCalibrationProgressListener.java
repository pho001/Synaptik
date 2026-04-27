package tuning.calibration.progress;

public interface PlatformCalibrationProgressListener {
    void onEvent(PlatformCalibrationProgressEvent event);

    static PlatformCalibrationProgressListener noop() {
        return event -> { };
    }
}
