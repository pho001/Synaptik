package config.profile;

import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import tensor.DataType;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.store.HardwareFingerprint;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

/**
 * Resolves calibrated platform runtime profiles for the current JVM process.
 *
 * <p>The resolver is intentionally runtime-only. It reads {@link PlatformRuntimeProfile} artifacts produced by
 * platform calibration and converts them to {@link RuntimeConfig}; it does not load graph autotune winners and does not
 * change graph/backend ownership policy.</p>
 */
public final class PlatformRuntimeProfileResolver {
    public static final String PROFILES_ROOT_PROPERTY = "synaptik.profiles.root";
    public static final String PROFILES_ROOT_ENVIRONMENT = "SYNAPTIK_PROFILES_ROOT";

    private PlatformRuntimeProfileResolver() {
    }

    /**
     * Resolves a calibrated runtime config or returns the supplied fallback runtime.
     *
     * @param dataType dtype requested by the graph root
     * @param mode execution mode requested by prepare/compute
     * @param fallbackRuntime hardcoded runtime defaults used when no compatible profile is found
     * @return calibrated runtime when present, otherwise {@code fallbackRuntime}
     */
    public static RuntimeConfig resolveRuntimeConfig(
            DataType dataType,
            ExecutionMode mode,
            RuntimeConfig fallbackRuntime
    ) {
        RuntimeConfig safeFallback = fallbackRuntime == null ? defaultRuntimeFor(mode) : fallbackRuntime;
        return resolve(dataType, mode, safeFallback)
                .map(resolution -> resolution.profile().toRuntimeConfig())
                .orElse(safeFallback);
    }

    /**
     * Resolves a compatible calibrated profile for the current platform.
     *
     * @param dataType dtype requested by the graph root
     * @param mode execution mode requested by prepare/compute
     * @param fallbackRuntime fallback runtime used as a JSON default shape while loading older profile files
     * @return resolution metadata and loaded profile when present
     */
    public static Optional<Resolution> resolve(
            DataType dataType,
            ExecutionMode mode,
            RuntimeConfig fallbackRuntime
    ) {
        if (dataType == null || mode == null) {
            return Optional.empty();
        }
        RuntimeConfig safeFallback = fallbackRuntime == null ? defaultRuntimeFor(mode) : fallbackRuntime;
        HardwareFingerprint hardware = HardwareFingerprint.capture();
        PlatformRuntimeProfile fallback = fallbackProfile(dataType, mode, safeFallback, hardware);
        List<String> platformIds = platformIds(hardware);
        List<String> relativePaths = relativeProfilePaths(platformIds, dataType, mode);

        for (Path root : profileRoots()) {
            for (String relativePath : relativePaths) {
                Path path = root.resolve(relativePath);
                if (!Files.exists(path)) {
                    continue;
                }
                Optional<PlatformRuntimeProfile> loaded = loadFile(path, fallback, dataType, mode);
                if (loaded.isPresent()) {
                    String platformId = platformIdFromRelativePath(relativePath);
                    return Optional.of(new Resolution(ResolutionSource.FILESYSTEM, path, "", platformId, loaded.get()));
                }
            }
        }

        for (String relativePath : relativePaths) {
            String resourceName = "profiles/" + relativePath.replace('\\', '/');
            Optional<PlatformRuntimeProfile> loaded = loadResource(resourceName, fallback, dataType, mode);
            if (loaded.isPresent()) {
                String platformId = platformIdFromRelativePath(relativePath);
                return Optional.of(new Resolution(ResolutionSource.CLASSPATH_RESOURCE, null, resourceName, platformId, loaded.get()));
            }
        }

        return Optional.empty();
    }

    /**
     * Returns the canonical short platform id used for new profile artifacts.
     */
    public static String currentPlatformId() {
        return PlatformCalibrationPaths.platformId(HardwareFingerprint.capture());
    }

    /**
     * Returns filesystem roots searched before classpath resources.
     */
    public static List<Path> profileRoots() {
        LinkedHashSet<Path> roots = new LinkedHashSet<>();
        addConfiguredRoots(roots, System.getProperty(PROFILES_ROOT_PROPERTY));
        addConfiguredRoots(roots, System.getenv(PROFILES_ROOT_ENVIRONMENT));
        roots.add(Path.of("profiles"));
        roots.add(Path.of(System.getProperty("user.home"), ".synaptik", "profiles"));
        return List.copyOf(roots);
    }

    private static List<String> platformIds(HardwareFingerprint hardware) {
        LinkedHashSet<String> ids = new LinkedHashSet<>();
        ids.add(PlatformCalibrationPaths.platformId(hardware));
        ids.add(PlatformCalibrationPaths.legacyPlatformId(hardware));
        return List.copyOf(ids);
    }

    private static List<String> relativeProfilePaths(List<String> platformIds, DataType dataType, ExecutionMode mode) {
        String dtype = dtypeId(dataType);
        String modeId = modeId(mode);
        ArrayList<String> paths = new ArrayList<>();
        for (String platformId : platformIds) {
            paths.add(Path.of("platform", platformId, "calibration", "schema-v2", "latest", dtype, modeId, "profile.json").toString());
            paths.add(Path.of("platform", platformId, "calibration", dtype + "-" + modeId + ".json").toString());
        }
        return List.copyOf(paths);
    }

    private static Optional<PlatformRuntimeProfile> loadFile(
            Path path,
            PlatformRuntimeProfile fallback,
            DataType dataType,
            ExecutionMode mode
    ) {
        try {
            return loadJson(Files.readString(path, StandardCharsets.UTF_8), fallback, dataType, mode);
        } catch (IOException | RuntimeException ignored) {
            return Optional.empty();
        }
    }

    private static Optional<PlatformRuntimeProfile> loadResource(
            String resourceName,
            PlatformRuntimeProfile fallback,
            DataType dataType,
            ExecutionMode mode
    ) {
        ClassLoader loader = PlatformRuntimeProfileResolver.class.getClassLoader();
        try (InputStream in = loader == null
                ? ClassLoader.getSystemResourceAsStream(resourceName)
                : loader.getResourceAsStream(resourceName)) {
            if (in == null) {
                return Optional.empty();
            }
            return loadJson(new String(in.readAllBytes(), StandardCharsets.UTF_8), fallback, dataType, mode);
        } catch (IOException | RuntimeException ignored) {
            return Optional.empty();
        }
    }

    private static Optional<PlatformRuntimeProfile> loadJson(
            String json,
            PlatformRuntimeProfile fallback,
            DataType dataType,
            ExecutionMode mode
    ) {
        PlatformRuntimeProfile profile = PlatformRuntimeProfileIO.fromJsonStrict(json, fallback);
        if (profile.dataType() != dataType || profile.metadata().executionMode() != mode) {
            return Optional.empty();
        }
        return Optional.of(profile);
    }

    private static PlatformRuntimeProfile fallbackProfile(
            DataType dataType,
            ExecutionMode mode,
            RuntimeConfig fallbackRuntime,
            HardwareFingerprint hardware
    ) {
        String platformId = PlatformCalibrationPaths.platformId(hardware);
        return PlatformRuntimeProfile.fromExecutionProfile(
                platformId,
                hardware.key(),
                "DEFAULT_RUNTIME_FALLBACK",
                new ExecutionProfile(
                        "platform-runtime-fallback-" + dtypeId(dataType) + "-" + modeId(mode),
                        "platform-runtime-fallback",
                        dataType,
                        mode,
                        mode == ExecutionMode.FORWARD_BACKWARD ? CompileConfig.training() : CompileConfig.inference(),
                        fallbackRuntime,
                        WorkloadProfile.none()
                )
        );
    }

    private static RuntimeConfig defaultRuntimeFor(ExecutionMode mode) {
        return mode == ExecutionMode.FORWARD_BACKWARD
                ? RuntimeConfig.trainingDefaults()
                : RuntimeConfig.inferenceDefaults();
    }

    private static void addConfiguredRoots(LinkedHashSet<Path> roots, String configured) {
        if (configured == null || configured.isBlank()) {
            return;
        }
        for (String token : configured.split(java.io.File.pathSeparator)) {
            if (token != null && !token.isBlank()) {
                roots.add(Path.of(token.trim()));
            }
        }
    }

    private static String platformIdFromRelativePath(String relativePath) {
        String normalized = relativePath == null ? "" : relativePath.replace('\\', '/');
        String prefix = "platform/";
        if (!normalized.startsWith(prefix)) {
            return "";
        }
        int end = normalized.indexOf('/', prefix.length());
        return end < 0 ? "" : normalized.substring(prefix.length(), end);
    }

    private static String dtypeId(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> "f64";
            case FLOAT32 -> "f32";
            case BFLOAT16 -> "bf16";
            case INT32 -> "i32";
            case INT64 -> "i64";
            case BOOL -> "bool";
        };
    }

    private static String modeId(ExecutionMode mode) {
        return mode.name().toLowerCase(Locale.ROOT).replace('_', '-');
    }

    public enum ResolutionSource {
        FILESYSTEM,
        CLASSPATH_RESOURCE
    }

    public record Resolution(
            ResolutionSource source,
            Path path,
            String resourceName,
            String platformId,
            PlatformRuntimeProfile profile
    ) {
        public Resolution {
            source = source == null ? ResolutionSource.FILESYSTEM : source;
            resourceName = resourceName == null ? "" : resourceName;
            platformId = platformId == null ? "" : platformId;
            if (profile == null) {
                throw new IllegalArgumentException("profile cannot be null");
            }
        }
    }
}
