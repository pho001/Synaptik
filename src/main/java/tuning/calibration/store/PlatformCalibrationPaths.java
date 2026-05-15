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
        return canonicalOs(safeHardware.os()) + "-" + canonicalArch(safeHardware.arch());
    }

    /**
     * Returns the pre-canonicalization id used by older local profile artifacts.
     *
     * <p>New calibration and tuning writes must use {@link #platformId(HardwareFingerprint)}. This helper exists only
     * so runtime/profile resolvers can read already-created local profiles while users regenerate them under the new
     * shorter platform id.</p>
     */
    public static String legacyPlatformId(HardwareFingerprint hardware) {
        HardwareFingerprint safeHardware = hardware == null ? HardwareFingerprint.capture() : hardware;
        return safeHardware.os() + "-"
                + safeHardware.arch() + "-"
                + safeHardware.vendor() + "-"
                + safeHardware.cores() + "c";
    }

    public static String canonicalOs(String os) {
        String normalized = os == null ? "" : os.toLowerCase(Locale.ROOT).replace(' ', '_').replace('\t', '_');
        if (normalized.contains("mac")) {
            return "macos";
        }
        if (normalized.contains("windows")) {
            return "windows";
        }
        if (normalized.contains("linux")) {
            return "linux";
        }
        return normalized.isBlank() ? "unknown" : normalized.replaceAll("[^a-z0-9_]+", "_");
    }

    public static String canonicalArch(String arch) {
        String normalized = arch == null ? "" : arch.toLowerCase(Locale.ROOT).replace(' ', '_').replace('\t', '_');
        if (normalized.equals("aarch64") || normalized.equals("arm64")) {
            return "arm64";
        }
        if (normalized.equals("x86_64") || normalized.equals("amd64") || normalized.equals("x64")) {
            return "x64";
        }
        return normalized.isBlank() ? "unknown" : normalized.replaceAll("[^a-z0-9_]+", "_");
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
            case INT64 -> "i64";
            case BOOL -> "bool";
        };
    }

    private static String modeId(ExecutionProfile profile) {
        return profile.mode().name().toLowerCase(Locale.ROOT).replace('_', '-');
    }
}
