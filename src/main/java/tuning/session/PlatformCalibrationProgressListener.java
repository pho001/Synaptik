package tuning.session;

public interface PlatformCalibrationProgressListener {
    void onEvent(PlatformCalibrationProgressEvent event);

    static PlatformCalibrationProgressListener noop() {
        return event -> { };
    }
}
