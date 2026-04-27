package tuning.calibration.store;

import config.profile.ExecutionProfile;
import tuning.store.HardwareFingerprint;

import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;

public final class PlatformCalibrationPaths {
    private PlatformCalibrationPaths() {
    }

    public static PlatformCalibrationLayout defaultLayout(Path rootDir, ExecutionProfile seedProfile) {
        return defaultLayout(rootDir, seedProfile, HardwareFingerprint.capture());
    }

    public static PlatformCalibrationLayout defaultLayout(
            Path rootDir,
            ExecutionProfile seedProfile,
            HardwareFingerprint hardware
    ) {
        Objects.requireNonNull(rootDir, "rootDir cannot be null");
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        HardwareFingerprint safeHardware = hardware == null ? HardwareFingerprint.capture() : hardware;

        String platformId = platformId(safeHardware);
        String variant = variantId(seedProfile);

        Path profilePath = rootDir
                .resolve("profiles")
                .resolve("platform")
                .resolve(platformId)
                .resolve(variant + ".json");

        Path jsonReportPath = rootDir
                .resolve("reports")
                .resolve("platform")
                .resolve(platformId)
                .resolve("calibration-" + variant + ".json");

        Path textReportPath = rootDir
                .resolve("reports")
                .resolve("platform")
                .resolve(platformId)
                .resolve("calibration-" + variant + ".txt");

        return new PlatformCalibrationLayout(
                platformId,
                safeHardware,
                profilePath,
                jsonReportPath,
                textReportPath
        );
    }

    public static String platformId(HardwareFingerprint hardware) {
        HardwareFingerprint safeHardware = hardware == null ? HardwareFingerprint.capture() : hardware;
        return safeHardware.os() + "-"
                + safeHardware.arch() + "-"
                + safeHardware.vendor() + "-"
                + safeHardware.cores() + "c";
    }

    public static String variantId(ExecutionProfile profile) {
        Objects.requireNonNull(profile, "profile cannot be null");
        return dataTypeId(profile) + "-" + modeId(profile);
    }

    private static String dataTypeId(ExecutionProfile profile) {
        return switch (profile.dataType()) {
            case FLOAT64 -> "f64";
            case FLOAT32 -> "f32";
            case BFLOAT16 -> "bf16";
            case INT32 -> "i32";
            case BOOL -> "bool";
        };
    }

    private static String modeId(ExecutionProfile profile) {
        return profile.mode().name().toLowerCase(Locale.ROOT).replace('_', '-');
    }
}
