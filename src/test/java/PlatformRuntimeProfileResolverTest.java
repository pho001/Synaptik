import config.runtime.BlasProvider;
import runtime.contract.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.PlatformRuntimeProfileResolver;
import config.profile.WorkloadProfile;
import config.runtime.BlasConfig;
import config.runtime.CpuExecutionPolicy;
import config.runtime.FusedExecutionPolicy;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tensor.DataType;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.store.HardwareFingerprint;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PlatformRuntimeProfileResolverTest {
    @TempDir
    Path tempDir;

    @Test
    void trainingDefaultsLoadLatestProfileForCurrentCanonicalPlatform() throws Exception {
        String previous = System.getProperty(PlatformRuntimeProfileResolver.PROFILES_ROOT_PROPERTY);
        try {
            System.setProperty(PlatformRuntimeProfileResolver.PROFILES_ROOT_PROPERTY, tempDir.toString());
            HardwareFingerprint hardware = HardwareFingerprint.capture();
            String platformId = PlatformCalibrationPaths.platformId(hardware);
            Path profilePath = tempDir
                    .resolve("platform")
                    .resolve(platformId)
                    .resolve("calibration")
                    .resolve("schema-v2")
                    .resolve("latest")
                    .resolve("f64")
                    .resolve("forward-backward")
                    .resolve("profile.json");
            PlatformRuntimeProfileIO.save(profilePath, calibratedProfile(platformId, hardware.key(), DataType.FLOAT64, ExecutionMode.FORWARD_BACKWARD));

            RuntimeConfig runtime = RuntimeConfig.trainingDefaults(DataType.FLOAT64);

            assertEquals(BlasProvider.OPENBLAS_FFM, runtime.blas().provider());
            assertEquals(123_456L, runtime.blas().matmulMinWork());
            assertEquals(2, runtime.blas().threads());
            assertEquals(new CpuExecutionPolicy(true, false), runtime.cpuExecutionPolicy());
            assertEquals(new FusedExecutionPolicy(false, true), runtime.fused());
        } finally {
            restoreProperty(previous);
        }
    }

    @Test
    void resolverCanReadLegacyPlatformDirectoryWhileNewWritesUseCanonicalId() throws Exception {
        String previous = System.getProperty(PlatformRuntimeProfileResolver.PROFILES_ROOT_PROPERTY);
        try {
            System.setProperty(PlatformRuntimeProfileResolver.PROFILES_ROOT_PROPERTY, tempDir.toString());
            HardwareFingerprint hardware = HardwareFingerprint.capture();
            String canonicalId = PlatformCalibrationPaths.platformId(hardware);
            String legacyId = PlatformCalibrationPaths.legacyPlatformId(hardware);
            Path profilePath = tempDir
                    .resolve("platform")
                    .resolve(legacyId)
                    .resolve("calibration")
                    .resolve("f32-forward-backward.json");
            PlatformRuntimeProfileIO.save(profilePath, calibratedProfile(legacyId, hardware.key(), DataType.FLOAT32, ExecutionMode.FORWARD_BACKWARD));

            var resolution = PlatformRuntimeProfileResolver.resolve(
                    DataType.FLOAT32,
                    ExecutionMode.FORWARD_BACKWARD,
                    RuntimeConfig.trainingDefaults()
            );

            assertTrue(resolution.isPresent());
            assertEquals(legacyId, resolution.get().platformId());
            assertEquals("macos-arm64", PlatformCalibrationPaths.platformId(
                    new HardwareFingerprint("Mac OS X", "aarch64", "vm", "vendor", 16)
            ));
            assertEquals(canonicalId, PlatformRuntimeProfileResolver.currentPlatformId());
        } finally {
            restoreProperty(previous);
        }
    }

    @Test
    void inferenceDefaultsCanLoadFlatCanonicalProfile() throws Exception {
        String previous = System.getProperty(PlatformRuntimeProfileResolver.PROFILES_ROOT_PROPERTY);
        try {
            System.setProperty(PlatformRuntimeProfileResolver.PROFILES_ROOT_PROPERTY, tempDir.toString());
            HardwareFingerprint hardware = HardwareFingerprint.capture();
            String platformId = PlatformCalibrationPaths.platformId(hardware);
            Path profilePath = tempDir
                    .resolve("platform")
                    .resolve(platformId)
                    .resolve("calibration")
                    .resolve("f32-forward.json");
            PlatformRuntimeProfileIO.save(profilePath, calibratedProfile(platformId, hardware.key(), DataType.FLOAT32, ExecutionMode.FORWARD));

            RuntimeConfig runtime = RuntimeConfig.inferenceDefaults(DataType.FLOAT32);

            assertEquals(BlasProvider.OPENBLAS_FFM, runtime.blas().provider());
            assertEquals(123_456L, runtime.blas().matmulMinWork());
            assertEquals(new CpuExecutionPolicy(true, false), runtime.cpuExecutionPolicy());
            assertEquals(new FusedExecutionPolicy(false, true), runtime.fused());
        } finally {
            restoreProperty(previous);
        }
    }

    @Test
    void packagedJarContainsCalibrationProfileResources() {
        assertTrue(
                PlatformRuntimeProfileResolver.class.getClassLoader()
                        .getResource("profiles/platform/macos-arm64/calibration/schema-v2/latest/f32/forward-backward/profile.json") != null
                        || Files.exists(Path.of("profiles/platform/macos-arm64/calibration/schema-v2/latest/f32/forward-backward/profile.json")),
                "expected canonical calibration profile to be available as a resource after processResources or as a source-tree fixture"
        );
    }

    @Test
    void canonicalCalibrationProfilesEnableCpu1DefaultRoutePolicy() throws Exception {
        for (DataType dataType : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16)) {
            Path profilePath = canonicalProfilePath(dataType);
            assertTrue(Files.exists(profilePath), () -> "missing canonical profile fixture: " + profilePath);

            PlatformRuntimeProfile profile = PlatformRuntimeProfileIO.fromJsonStrict(
                    Files.readString(profilePath),
                    fallbackProfileWithHardcodedDefaults(dataType)
            );
            RuntimeConfig runtime = profile.toRuntimeConfig();

            assertEquals(new CpuExecutionPolicy(true, true), runtime.cpuExecutionPolicy(), dataType.name());
            assertEquals(new FusedExecutionPolicy(true, true), runtime.fused(), dataType.name());
        }
    }

    private static PlatformRuntimeProfile calibratedProfile(
            String platformId,
            String hardwareKey,
            DataType dataType,
            ExecutionMode mode
    ) {
        RuntimeConfig runtime = new RuntimeConfig(
                config.backend.KernelTuningConfig.defaultsTraining(),
                config.runtime.ApproximationConfig.defaults(),
                new BlasConfig(BlasProvider.OPENBLAS_FFM, 123_456L, false, 1.0d, false, 2),
                config.runtime.Conv2dConfig.fromBlasConfig(new BlasConfig(BlasProvider.OPENBLAS_FFM, 123_456L, false, 1.0d, false, 2)),
                new FusedExecutionPolicy(false, true),
                new CpuExecutionPolicy(true, false),
                config.runtime.AcceleratorConfig.defaultsTraining(),
                config.runtime.CpuStorageProfile.CPU_ARRAY,
                config.runtime.NativeCpuFailurePolicy.FALLBACK_TO_ARRAY,
                config.runtime.DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE,
                config.runtime.NativeCpuMemoryConfig.disabled(),
                config.runtime.BFloat16TrainingPolicy.ACTIVATIONS_ONLY
        );
        return PlatformRuntimeProfile.fromExecutionProfile(
                platformId,
                hardwareKey,
                "TEST",
                new ExecutionProfile(
                        "calibrated",
                        "calibrated",
                        dataType,
                        mode,
                        mode == ExecutionMode.FORWARD_BACKWARD
                                ? config.compile.CompileConfig.training()
                                : config.compile.CompileConfig.inference(),
                        runtime,
                        WorkloadProfile.none()
                )
        );
    }

    private static PlatformRuntimeProfile fallbackProfileWithHardcodedDefaults(DataType dataType) {
        return PlatformRuntimeProfile.fromExecutionProfile(
                "macos-arm64",
                "test-fallback",
                "TEST_DEFAULTS",
                new ExecutionProfile(
                        "fallback",
                        "fallback",
                        dataType,
                        ExecutionMode.FORWARD_BACKWARD,
                        config.compile.CompileConfig.training(),
                        RuntimeConfig.trainingDefaults(),
                        WorkloadProfile.none()
                )
        );
    }

    private static Path canonicalProfilePath(DataType dataType) {
        return Path.of(
                "profiles",
                "platform",
                "macos-arm64",
                "calibration",
                "schema-v2",
                "latest",
                dtypePath(dataType),
                "forward-backward",
                "profile.json"
        );
    }

    private static String dtypePath(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> "f32";
            case FLOAT64 -> "f64";
            case BFLOAT16 -> "bf16";
            default -> throw new IllegalArgumentException("No canonical cpu1 default-route profile for " + dataType);
        };
    }

    private static void restoreProperty(String previous) {
        if (previous == null) {
            System.clearProperty(PlatformRuntimeProfileResolver.PROFILES_ROOT_PROPERTY);
        } else {
            System.setProperty(PlatformRuntimeProfileResolver.PROFILES_ROOT_PROPERTY, previous);
        }
    }
}
