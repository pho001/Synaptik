package Benchmark;

import Config.backend.CpuKernelConfig;
import Config.backend.CudaKernelConfig;
import Config.backend.KernelTuningConfig;
import Config.backend.OpenClKernelConfig;
import Config.optimizer.FuseConfig;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class OptimizerProfileIO {
    private OptimizerProfileIO() {}

    public static TuningKnobs loadKnobsOrDefault(Path path, TuningKnobs defaultKnobs) {
        if (!Files.exists(path)) return defaultKnobs;
        try {
            String json = Files.readString(path, StandardCharsets.UTF_8);
            return fromJsonOrDefault(json, defaultKnobs);
        } catch (IOException e) {
            return defaultKnobs;
        }
    }

    public static OptimizerCandidate loadRecommendedOverrideOrDefault(Path path, OptimizerCandidate defaultRecommended) {
        if (!Files.exists(path)) return defaultRecommended;
        try {
            String json = Files.readString(path, StandardCharsets.UTF_8);
            TuningKnobs knobs = fromJsonOrDefault(json, defaultRecommended.knobs());
            List<OptimizationStage> stages = parseStageOrderOrDefault(json, defaultRecommended.stageOrder());
            return new OptimizerCandidate(defaultRecommended.name(), stages, knobs);
        } catch (Exception e) {
            return defaultRecommended;
        }
    }

    public static double loadScoreOrInfinity(Path path) {
        if (!Files.exists(path)) return Double.POSITIVE_INFINITY;
        try {
            String json = Files.readString(path, StandardCharsets.UTF_8);
            Matcher metricsMatcher = Pattern.compile("\"metrics\"\\s*:\\s*\\{([\\s\\S]*?)\\}").matcher(json);
            if (!metricsMatcher.find()) return Double.POSITIVE_INFINITY;
            String metricsBody = metricsMatcher.group(1);
            return findDouble(metricsBody, "score", Double.POSITIVE_INFINITY);
        } catch (Exception e) {
            return Double.POSITIVE_INFINITY;
        }
    }

    public static void saveKnobs(Path path, TuningKnobs knobs, String candidateName) {
        String json = toJson(knobs, candidateName);
        try {
            Files.createDirectories(path.getParent());
            Files.writeString(path, json, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to save optimizer profile to " + path, e);
        }
    }

    public static String toJson(TuningKnobs knobs, String candidateName) {
        FuseConfig f = knobs.fuseConfig();
        KernelTuningConfig k = knobs.kernelConfig();
        return "{\n" +
                "  \"candidateName\": \"" + candidateName + "\",\n" +
                "  \"strictCseSafety\": " + knobs.strictCseSafety() + ",\n" +
                "  \"kernel\": {\n" +
                "    \"cpu\": {\n" +
                "      \"cpuLoopUnrollFactor\": " + k.cpu().loopUnrollFactor() + ",\n" +
                "      \"cpuMatMulTileM\": " + k.cpu().matMulTileM() + ",\n" +
                "      \"cpuMatMulTileN\": " + k.cpu().matMulTileN() + ",\n" +
                "      \"cpuMatMulTileK\": " + k.cpu().matMulTileK() + ",\n" +
                "      \"cpuVectorMinSize\": " + k.cpu().vectorMinSize() + ",\n" +
                "      \"cpuParallelMinSize\": " + k.cpu().parallelMinSize() + ",\n" +
                "      \"cpuParallelism\": " + k.cpu().parallelism() + ",\n" +
                "      \"cpuChunksPerWorker\": " + k.cpu().chunksPerWorker() + ",\n" +
                "      \"cpuMinChunkSize\": " + k.cpu().minChunkSize() + ",\n" +
                "      \"cpuContiguousMaterializeThreshold\": " + k.cpu().contiguousMaterializeThreshold() + "\n" +
                "    },\n" +
                "    \"cuda\": {\n" +
                "      \"cudaLoopUnrollFactor\": " + k.cuda().loopUnrollFactor() + ",\n" +
                "      \"cudaMatMulTileM\": " + k.cuda().matMulTileM() + ",\n" +
                "      \"cudaMatMulTileN\": " + k.cuda().matMulTileN() + ",\n" +
                "      \"cudaMatMulTileK\": " + k.cuda().matMulTileK() + "\n" +
                "    },\n" +
                "    \"opencl\": {\n" +
                "      \"openclLoopUnrollFactor\": " + k.opencl().loopUnrollFactor() + ",\n" +
                "      \"openclMatMulTileM\": " + k.opencl().matMulTileM() + ",\n" +
                "      \"openclMatMulTileN\": " + k.opencl().matMulTileN() + ",\n" +
                "      \"openclMatMulTileK\": " + k.opencl().matMulTileK() + "\n" +
                "    }\n" +
                "  },\n" +
                "  \"fuse\": {\n" +
                "    \"maxClusterNodes\": " + f.maxClusterNodes() + ",\n" +
                "    \"scoreThreshold\": " + f.scoreThreshold() + ",\n" +
                "    \"internalEdgeBonus\": " + f.internalEdgeBonus() + ",\n" +
                "    \"externalInputPenalty\": " + f.externalInputPenalty() + ",\n" +
                "    \"sharedExpensivePenalty\": " + f.sharedExpensivePenalty() + ",\n" +
                "    \"nonCheapBonus\": " + f.nonCheapBonus() + ",\n" +
                "    \"preserveSharedExpensiveNodes\": " + f.preserveSharedExpensiveNodes() + "\n" +
                "  }\n" +
                "}\n";
    }

    private static TuningKnobs fromJsonOrDefault(String json, TuningKnobs d) {
        try {
            boolean strictCse = findBoolean(json, "strictCseSafety", d.strictCseSafety());

            // Backward-compatible fallback: if old global keys exist, use them as CPU defaults.
            int legacyUnroll = findInt(json, "loopUnrollFactor", d.kernelConfig().cpu().loopUnrollFactor());
            int legacyTileM = findInt(json, "tileM", d.kernelConfig().cpu().matMulTileM());
            int legacyTileN = findInt(json, "tileN", d.kernelConfig().cpu().matMulTileN());
            int legacyTileK = findInt(json, "tileK", d.kernelConfig().cpu().matMulTileK());

            CpuKernelConfig cpu = new CpuKernelConfig(
                    findInt(json, "cpuLoopUnrollFactor", findInt(json, "loopUnrollFactor", legacyUnroll)),
                    findInt(json, "cpuMatMulTileM", legacyTileM),
                    findInt(json, "cpuMatMulTileN", legacyTileN),
                    findInt(json, "cpuMatMulTileK", legacyTileK),
                    findInt(json, "cpuVectorMinSize", d.kernelConfig().cpu().vectorMinSize()),
                    findInt(json, "cpuParallelMinSize", d.kernelConfig().cpu().parallelMinSize()),
                    findInt(json, "cpuParallelism", d.kernelConfig().cpu().parallelism()),
                    findInt(json, "cpuChunksPerWorker", d.kernelConfig().cpu().chunksPerWorker()),
                    findInt(json, "cpuMinChunkSize", d.kernelConfig().cpu().minChunkSize()),
                    findInt(json, "cpuContiguousMaterializeThreshold", d.kernelConfig().cpu().contiguousMaterializeThreshold())
            );
            CudaKernelConfig cuda = new CudaKernelConfig(
                    findInt(json, "cudaLoopUnrollFactor", d.kernelConfig().cuda().loopUnrollFactor()),
                    findInt(json, "cudaMatMulTileM", d.kernelConfig().cuda().matMulTileM()),
                    findInt(json, "cudaMatMulTileN", d.kernelConfig().cuda().matMulTileN()),
                    findInt(json, "cudaMatMulTileK", d.kernelConfig().cuda().matMulTileK())
            );
            OpenClKernelConfig opencl = new OpenClKernelConfig(
                    findInt(json, "openclLoopUnrollFactor", d.kernelConfig().opencl().loopUnrollFactor()),
                    findInt(json, "openclMatMulTileM", d.kernelConfig().opencl().matMulTileM()),
                    findInt(json, "openclMatMulTileN", d.kernelConfig().opencl().matMulTileN()),
                    findInt(json, "openclMatMulTileK", d.kernelConfig().opencl().matMulTileK())
            );
            KernelTuningConfig kernel = new KernelTuningConfig(cpu, cuda, opencl);

            FuseConfig df = d.fuseConfig();
            FuseConfig fuse = new FuseConfig(
                    findInt(json, "maxClusterNodes", df.maxClusterNodes()),
                    findDouble(json, "scoreThreshold", df.scoreThreshold()),
                    findDouble(json, "internalEdgeBonus", df.internalEdgeBonus()),
                    findDouble(json, "externalInputPenalty", df.externalInputPenalty()),
                    findDouble(json, "sharedExpensivePenalty", df.sharedExpensivePenalty()),
                    findDouble(json, "nonCheapBonus", df.nonCheapBonus()),
                    findBoolean(json, "preserveSharedExpensiveNodes", df.preserveSharedExpensiveNodes())
            );

            return new TuningKnobs(strictCse, fuse, kernel);
        } catch (Exception e) {
            return d;
        }
    }

    private static List<OptimizationStage> parseStageOrderOrDefault(String json, List<OptimizationStage> defaultStages) {
        Matcher m = Pattern.compile("\"stageOrder\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL).matcher(json);
        if (!m.find()) return defaultStages;

        String body = m.group(1);
        Matcher tokens = Pattern.compile("\"([A-Z_]+)\"").matcher(body);
        List<OptimizationStage> out = new ArrayList<>();
        while (tokens.find()) {
            try {
                out.add(OptimizationStage.valueOf(tokens.group(1)));
            } catch (IllegalArgumentException ignored) {
                return defaultStages;
            }
        }
        return out.isEmpty() ? defaultStages : List.copyOf(out);
    }

    private static boolean findBoolean(String json, String key, boolean defaultValue) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(true|false)").matcher(json);
        if (!m.find()) return defaultValue;
        return Boolean.parseBoolean(m.group(1));
    }

    private static int findInt(String json, String key, int defaultValue) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?\\d+)").matcher(json);
        if (!m.find()) return defaultValue;
        return Integer.parseInt(m.group(1));
    }

    private static double findDouble(String json, String key, double defaultValue) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)").matcher(json);
        if (!m.find()) return defaultValue;
        return Double.parseDouble(m.group(1));
    }
}
