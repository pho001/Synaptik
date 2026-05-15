import backend.blas.BlasProvider;
import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.PlatformRuntimeProfileResolver;
import config.profile.WorkloadProfile;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tensor.DataType;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.store.HardwareFingerprint;

import java.nio.file.Files;
import java.nio.file.Path;

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
            assertEquals(config.runtime.BlasConfig.DEFAULT_THREADS, runtime.blas().threads());
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

    private static PlatformRuntimeProfile calibratedProfile(
            String platformId,
            String hardwareKey,
            DataType dataType,
            ExecutionMode mode
    ) {
        RuntimeConfig runtime = new RuntimeConfig(
                config.backend.KernelTuningConfig.defaultsTraining(),
                config.runtime.ApproximationConfig.defaults(),
                new BlasConfig(BlasProvider.OPENBLAS_FFM, 123_456L, false, 1.0d, false, 2)
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

    private static void restoreProperty(String previous) {
        if (previous == null) {
            System.clearProperty(PlatformRuntimeProfileResolver.PROFILES_ROOT_PROPERTY);
        } else {
            System.setProperty(PlatformRuntimeProfileResolver.PROFILES_ROOT_PROPERTY, previous);
        }
    }
}
