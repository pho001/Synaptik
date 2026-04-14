package config.profile;

import backend.ApproxMode;
import backend.blas.BlasProvider;
import backend.runtime.ExecutionMode;
import config.backend.SumAccuracyMode;
import tensor.DataType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class PlatformRuntimeProfileIO {
    private PlatformRuntimeProfileIO() {
    }

    public static void save(Path path, PlatformRuntimeProfile profile) {
        if (path == null) {
            throw new IllegalArgumentException("path cannot be null");
        }
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            Files.writeString(path, toJson(profile), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to save platform runtime profile to " + path, e);
        }
    }

    public static String toJson(PlatformRuntimeProfile profile) {
        PlatformProfileMetadata m = profile.metadata();
        return "{\n" +
                "  \"metadata\": {\n" +
                "    \"platformProfileId\": \"" + escape(m.platformProfileId()) + "\",\n" +
                "    \"hardwareKey\": \"" + escape(m.hardwareKey()) + "\",\n" +
                "    \"frameworkVersion\": \"" + escape(m.frameworkVersion()) + "\",\n" +
                "    \"plannerSchemaVersion\": \"" + escape(m.plannerSchemaVersion()) + "\",\n" +
                "    \"persistenceSchemaVersion\": \"" + escape(m.persistenceSchemaVersion()) + "\",\n" +
                "    \"createdAtIso\": \"" + escape(m.createdAtIso()) + "\",\n" +
                "    \"calibrationPreset\": \"" + escape(m.calibrationPreset()) + "\",\n" +
                "    \"dataType\": \"" + m.dataType().name() + "\",\n" +
                "    \"executionMode\": \"" + m.executionMode().name() + "\"\n" +
                "  },\n" +
                "  \"matmul\": {\n" +
                "    \"blasProvider\": \"" + profile.matmul().blasProvider().name() + "\",\n" +
                "    \"blasMatmulMinWork\": " + profile.matmul().blasMatmulMinWork() + ",\n" +
                "    \"blasThreads\": " + profile.matmul().blasThreads() + ",\n" +
                "    \"f32RequireMgeK\": " + profile.matmul().f32RequireMgeK() + ",\n" +
                "    \"f32MaxNOverK\": " + profile.matmul().f32MaxNOverK() + ",\n" +
                "    \"loopUnrollFactor\": " + profile.matmul().loopUnrollFactor() + ",\n" +
                "    \"matMulTileM\": " + profile.matmul().matMulTileM() + ",\n" +
                "    \"matMulTileN\": " + profile.matmul().matMulTileN() + ",\n" +
                "    \"matMulTileK\": " + profile.matmul().matMulTileK() + ",\n" +
                "    \"matMulParallelMinSize\": " + profile.matmul().matMulParallelMinSize() + "\n" +
                "  },\n" +
                "  \"fused\": {\n" +
                "    \"fusedCheapVectorMinSize\": " + profile.fused().fusedCheapVectorMinSize() + ",\n" +
                "    \"fusedTranscendentalVectorMinSize\": " + profile.fused().fusedTranscendentalVectorMinSize() + ",\n" +
                "    \"fusedCheapParallelMinSize\": " + profile.fused().fusedCheapParallelMinSize() + ",\n" +
                "    \"fusedTranscendentalParallelMinSize\": " + profile.fused().fusedTranscendentalParallelMinSize() + ",\n" +
                "    \"fusedAsmVectorWidth\": " + profile.fused().fusedAsmVectorWidth() + "\n" +
                "  },\n" +
                "  \"elementwiseDispatch\": {\n" +
                "    \"cheapVectorMinSize\": " + profile.elementwiseDispatch().cheapVectorMinSize() + ",\n" +
                "    \"transcendentalVectorMinSize\": " + profile.elementwiseDispatch().transcendentalVectorMinSize() + ",\n" +
                "    \"cheapParallelMinSize\": " + profile.elementwiseDispatch().cheapParallelMinSize() + ",\n" +
                "    \"transcendentalParallelMinSize\": " + profile.elementwiseDispatch().transcendentalParallelMinSize() + "\n" +
                "  },\n" +
                "  \"reduction\": {\n" +
                "    \"reductionVectorMinSize\": " + profile.reduction().reductionVectorMinSize() + ",\n" +
                "    \"reductionParallelMinSize\": " + profile.reduction().reductionParallelMinSize() + ",\n" +
                "    \"sumAccuracyMode\": \"" + profile.reduction().sumAccuracyMode().name() + "\"\n" +
                "  },\n" +
                "  \"scheduler\": {\n" +
                "    \"lowCostTargetChunksPerWorker\": " + profile.scheduler().lowCostTargetChunksPerWorker() + ",\n" +
                "    \"mediumCostTargetChunksPerWorker\": " + profile.scheduler().mediumCostTargetChunksPerWorker() + ",\n" +
                "    \"highCostTargetChunksPerWorker\": " + profile.scheduler().highCostTargetChunksPerWorker() + ",\n" +
                "    \"minScalarChunkSize\": " + profile.scheduler().minScalarChunkSize() + ",\n" +
                "    \"minVectorChunkSize\": " + profile.scheduler().minVectorChunkSize() + ",\n" +
                "    \"minReductionChunkSize\": " + profile.scheduler().minReductionChunkSize() + ",\n" +
                "    \"commonPoolLowCostMaxWorkPerWorker\": " + profile.scheduler().commonPoolLowCostMaxWorkPerWorker() + "\n" +
                "  },\n" +
                "  \"materialization\": {\n" +
                "    \"contiguousMaterializeThreshold\": " + profile.materialization().contiguousMaterializeThreshold() + "\n" +
                "  },\n" +
                "  \"numerics\": {\n" +
                "    \"approxMode\": \"" + profile.numerics().approxMode().name() + "\",\n" +
                "    \"forceExactTranscendentals\": " + profile.numerics().forceExactTranscendentals() + "\n" +
                "  }\n" +
                "}\n";
    }

    public static PlatformRuntimeProfile loadOrDefault(Path path, PlatformRuntimeProfile fallback) {
        if (path == null || !Files.exists(path)) {
            return fallback;
        }
        try {
            String json = Files.readString(path, StandardCharsets.UTF_8);
            return fromJsonOrDefault(json, fallback);
        } catch (IOException e) {
            return fallback;
        }
    }

    public static PlatformRuntimeProfile fromJsonOrDefault(String json, PlatformRuntimeProfile fallback) {
        if (json == null || json.isBlank() || fallback == null) {
            return fallback;
        }
        try {
            PlatformProfileMetadata m = fallback.metadata();
            return new PlatformRuntimeProfile(
                    new PlatformProfileMetadata(
                            findString(json, "platformProfileId", m.platformProfileId()),
                            findString(json, "hardwareKey", m.hardwareKey()),
                            findString(json, "frameworkVersion", m.frameworkVersion()),
                            findString(json, "plannerSchemaVersion", m.plannerSchemaVersion()),
                            findString(json, "persistenceSchemaVersion", m.persistenceSchemaVersion()),
                            findString(json, "createdAtIso", m.createdAtIso()),
                            findString(json, "calibrationPreset", m.calibrationPreset()),
                            findEnum(json, "dataType", m.dataType(), DataType.class),
                            findEnum(json, "executionMode", m.executionMode(), ExecutionMode.class)
                    ),
                    new MatmulPlatformProfile(
                            findEnum(json, "blasProvider", fallback.matmul().blasProvider(), BlasProvider.class),
                            findLong(json, "blasMatmulMinWork", fallback.matmul().blasMatmulMinWork()),
                            findInt(json, "blasThreads", fallback.matmul().blasThreads()),
                            findBoolean(json, "f32RequireMgeK", fallback.matmul().f32RequireMgeK()),
                            findDouble(json, "f32MaxNOverK", fallback.matmul().f32MaxNOverK()),
                            findInt(json, "loopUnrollFactor", fallback.matmul().loopUnrollFactor()),
                            findInt(json, "matMulTileM", fallback.matmul().matMulTileM()),
                            findInt(json, "matMulTileN", fallback.matmul().matMulTileN()),
                            findInt(json, "matMulTileK", fallback.matmul().matMulTileK()),
                            findInt(json, "matMulParallelMinSize", fallback.matmul().matMulParallelMinSize())
                    ),
                    new FusedPlatformProfile(
                            findInt(json, "fusedCheapVectorMinSize", fallback.fused().fusedCheapVectorMinSize()),
                            findInt(json, "fusedTranscendentalVectorMinSize", fallback.fused().fusedTranscendentalVectorMinSize()),
                            findInt(json, "fusedCheapParallelMinSize", fallback.fused().fusedCheapParallelMinSize()),
                            findInt(json, "fusedTranscendentalParallelMinSize", fallback.fused().fusedTranscendentalParallelMinSize()),
                            findInt(json, "fusedAsmVectorWidth", fallback.fused().fusedAsmVectorWidth())
                    ),
                    new ElementwiseDispatchPlatformProfile(
                            findInt(json, "cheapVectorMinSize", fallback.elementwiseDispatch().cheapVectorMinSize()),
                            findInt(json, "transcendentalVectorMinSize", fallback.elementwiseDispatch().transcendentalVectorMinSize()),
                            findInt(json, "cheapParallelMinSize", fallback.elementwiseDispatch().cheapParallelMinSize()),
                            findInt(json, "transcendentalParallelMinSize", fallback.elementwiseDispatch().transcendentalParallelMinSize())
                    ),
                    new ReductionPlatformProfile(
                            findInt(json, "reductionVectorMinSize", fallback.reduction().reductionVectorMinSize()),
                            findInt(json, "reductionParallelMinSize", fallback.reduction().reductionParallelMinSize()),
                            findEnum(json, "sumAccuracyMode", fallback.reduction().sumAccuracyMode(), SumAccuracyMode.class)
                    ),
                    new SchedulerPlatformProfile(
                            findInt(json, "lowCostTargetChunksPerWorker", fallback.scheduler().lowCostTargetChunksPerWorker()),
                            findInt(json, "mediumCostTargetChunksPerWorker", fallback.scheduler().mediumCostTargetChunksPerWorker()),
                            findInt(json, "highCostTargetChunksPerWorker", fallback.scheduler().highCostTargetChunksPerWorker()),
                            findInt(json, "minScalarChunkSize", fallback.scheduler().minScalarChunkSize()),
                            findInt(json, "minVectorChunkSize", fallback.scheduler().minVectorChunkSize()),
                            findInt(json, "minReductionChunkSize", fallback.scheduler().minReductionChunkSize()),
                            findInt(json, "commonPoolLowCostMaxWorkPerWorker", fallback.scheduler().commonPoolLowCostMaxWorkPerWorker())
                    ),
                    new MaterializationPlatformProfile(
                            findInt(json, "contiguousMaterializeThreshold", fallback.materialization().contiguousMaterializeThreshold())
                    ),
                    new NumericsPlatformProfile(
                            findEnum(json, "approxMode", fallback.numerics().approxMode(), ApproxMode.class),
                            findBoolean(json, "forceExactTranscendentals", fallback.numerics().forceExactTranscendentals())
                    )
            );
        } catch (Exception e) {
            return fallback;
        }
    }

    private static String findString(String json, String key, String fallback) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) {
            return fallback;
        }
        int firstQuote = json.indexOf('"', json.indexOf(':', idx) + 1);
        int secondQuote = json.indexOf('"', firstQuote + 1);
        return firstQuote < 0 || secondQuote < 0 ? fallback : json.substring(firstQuote + 1, secondQuote);
    }

    private static int findInt(String json, String key, int fallback) {
        try {
            return (int) Math.round(findDouble(json, key, fallback));
        } catch (Exception e) {
            return fallback;
        }
    }

    private static long findLong(String json, String key, long fallback) {
        try {
            return Math.round(findDouble(json, key, fallback));
        } catch (Exception e) {
            return fallback;
        }
    }

    private static double findDouble(String json, String key, double fallback) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) {
            return fallback;
        }
        int colon = json.indexOf(':', idx);
        if (colon < 0) {
            return fallback;
        }
        int end = colon + 1;
        while (end < json.length() && " \n\r\t".indexOf(json.charAt(end)) >= 0) {
            end++;
        }
        int stop = end;
        while (stop < json.length() && ",}\n\r\t ".indexOf(json.charAt(stop)) < 0) {
            stop++;
        }
        return Double.parseDouble(json.substring(end, stop));
    }

    private static boolean findBoolean(String json, String key, boolean fallback) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) {
            return fallback;
        }
        int colon = json.indexOf(':', idx);
        if (colon < 0) {
            return fallback;
        }
        String tail = json.substring(colon + 1).trim();
        if (tail.startsWith("true")) {
            return true;
        }
        if (tail.startsWith("false")) {
            return false;
        }
        return fallback;
    }

    private static <E extends Enum<E>> E findEnum(String json, String key, E fallback, Class<E> type) {
        try {
            return Enum.valueOf(type, findString(json, key, fallback.name()));
        } catch (Exception e) {
            return fallback;
        }
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
