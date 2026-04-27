package tuning.calibration.store;

import tuning.store.HardwareFingerprint;

import java.nio.file.Path;
import java.util.Objects;

public record PlatformCalibrationLayout(
        String platformId,
        HardwareFingerprint hardware,
        Path profilePath,
        Path jsonReportPath,
        Path textReportPath
) {
    public PlatformCalibrationLayout {
        platformId = (platformId == null || platformId.isBlank()) ? "platform" : platformId;
        hardware = hardware == null ? HardwareFingerprint.capture() : hardware;
        Objects.requireNonNull(profilePath, "profilePath cannot be null");
        Objects.requireNonNull(jsonReportPath, "jsonReportPath cannot be null");
        Objects.requireNonNull(textReportPath, "textReportPath cannot be null");
    }
}
