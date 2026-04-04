package config.profile;

import backend.ApproxMode;
import backend.blas.BlasProvider;
import backend.blas.BlasThreadPolicy;
import backend.runtime.ExecutionMode;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuKernelConfig;
import config.backend.CudaKernelConfig;
import config.backend.KernelTuningConfig;
import config.backend.OpenClKernelConfig;
import config.backend.SumAccuracyMode;
import config.backend.VectorPolicy;
import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import config.optimizer.CseConfig;
import config.optimizer.FuseConfig;
import config.optimizer.MemoryConfig;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.optimizer.RewriteConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;
import tensor.DataType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class ExecutionProfileIO {
    private ExecutionProfileIO() {
    }

    public static ExecutionProfile loadExecutionProfileOrDefault(Path path, ExecutionProfile defaultProfile) {
        if (path == null || !Files.exists(path)) {
            return defaultProfile;
        }
        try {
            return fromJsonOrDefault(Files.readString(path, StandardCharsets.UTF_8), defaultProfile);
        } catch (IOException e) {
            return defaultProfile;
        }
    }

    public static ExecutionProfile fromJsonOrDefault(String json, ExecutionProfile defaultProfile) {
        if (json == null || json.isBlank()) {
            return defaultProfile;
        }
        try {
            DataType dataType = findEnum(json, "dataType", defaultProfile.dataType(), DataType.class);
            ExecutionMode mode = findEnum(json, "mode", defaultProfile.mode(), ExecutionMode.class);
            String profileName = findString(json, "profileName", defaultProfile.profileName());
            String candidateName = findString(json, "candidateName", defaultProfile.candidateName());

            List<OptimizerStage> stageOrder = parseStageOrderOrDefault(json, defaultProfile.optimizer().stageOrder());
            RewriteConfig defaultRewrite = defaultProfile.optimizer().rewrite();
            RewriteConfig rewrite = new RewriteConfig(
                    new Conv2dLoweringConfig(
                            findEnum(
                                    json,
                                    "conv2dLoweringMode",
                                    defaultRewrite.conv2dLowering().mode(),
                                    Conv2dLoweringMode.class
                            )
                    )
            );

            boolean strictSafety = findBoolean(
                    json,
                    "strictSafety",
                    defaultProfile.optimizer().cse().strictSafety()
            );
            FuseConfig defaultFuse = defaultProfile.optimizer().fuse();
            FuseConfig fuse = new FuseConfig(
                    findInt(json, "maxClusterNodes", defaultFuse.maxClusterNodes()),
                    findDouble(json, "scoreThreshold", defaultFuse.scoreThreshold()),
                    findDouble(json, "internalEdgeBonus", defaultFuse.internalEdgeBonus()),
                    findDouble(json, "externalInputPenalty", defaultFuse.externalInputPenalty()),
                    findDouble(json, "sharedExpensivePenalty", defaultFuse.sharedExpensivePenalty()),
                    findDouble(json, "nonCheapBonus", defaultFuse.nonCheapBonus()),
                    findBoolean(json, "preserveSharedExpensiveNodes", defaultFuse.preserveSharedExpensiveNodes())
            );
            MemoryConfig defaultMemory = defaultProfile.optimizer().memory();
            MemoryConfig memory = new MemoryConfig(
                    findBoolean(json, "separateForwardBackwardPools", defaultMemory.separateForwardBackwardPools()),
                    findBoolean(json, "allowCrossPhaseReuse", defaultMemory.allowCrossPhaseReuse()),
                    findBoolean(json, "allowLargerBufferReuse", defaultMemory.allowLargerBufferReuse()),
                    findInt(json, "minReusableBufferSize", defaultMemory.minReusableBufferSize())
            );
            OptimizerConfig optimizer = new OptimizerConfig(
                    stageOrder,
                    rewrite,
                    strictSafety ? CseConfig.strictDefaults() : CseConfig.aggressiveDefaults(),
                    fuse,
                    memory
            );

            KernelTuningConfig defaultKernel = defaultProfile.runtime().kernel();
            CpuKernelConfig cpu = new CpuKernelConfig(
                    findInt(json, "cpuLoopUnrollFactor", defaultKernel.cpu().loopUnrollFactor()),
                    findInt(json, "cpuMatMulTileM", defaultKernel.cpu().matMulTileM()),
                    findInt(json, "cpuMatMulTileN", defaultKernel.cpu().matMulTileN()),
                    findInt(json, "cpuMatMulTileK", defaultKernel.cpu().matMulTileK()),
                    findInt(json, "cpuVectorMinSize", defaultKernel.cpu().vectorMinSize()),
                    findInt(json, "cpuParallelMinSize", defaultKernel.cpu().parallelMinSize()),
                    findInt(json, "cpuParallelism", defaultKernel.cpu().parallelism()),
                    findInt(json, "cpuChunksPerWorker", defaultKernel.cpu().chunksPerWorker()),
                    findInt(json, "cpuMinChunkSize", defaultKernel.cpu().minChunkSize()),
                    findInt(json, "cpuContiguousMaterializeThreshold", defaultKernel.cpu().contiguousMaterializeThreshold()),
                    findEnum(json, "cpuSumAccuracyMode", defaultKernel.cpu().sumAccuracyMode(), SumAccuracyMode.class),
                    findDouble(json, "cpuLowCostNsPerElementThreshold", defaultKernel.cpu().lowCostNsPerElementThreshold()),
                    findEnum(json, "cpuVectorPolicyCheap", defaultKernel.cpu().vectorPolicyCheap(), VectorPolicy.class),
                    findEnum(json, "cpuVectorPolicyTranscendental", defaultKernel.cpu().vectorPolicyTranscendental(), VectorPolicy.class),
                    findEnum(json, "cpuVectorPolicyReduction", defaultKernel.cpu().vectorPolicyReduction(), VectorPolicy.class),
                    findInt(json, "cpuMatMulParallelMinSize", defaultKernel.cpu().matMulParallelMinSize()),
                    findEnum(json, "cpuAttentionMatMulPolicy", defaultKernel.cpu().attentionMatMulPolicy(), AttentionMatMulPolicy.class)
            );
            CudaKernelConfig cuda = new CudaKernelConfig(
                    findInt(json, "cudaLoopUnrollFactor", defaultKernel.cuda().loopUnrollFactor()),
                    findInt(json, "cudaMatMulTileM", defaultKernel.cuda().matMulTileM()),
                    findInt(json, "cudaMatMulTileN", defaultKernel.cuda().matMulTileN()),
                    findInt(json, "cudaMatMulTileK", defaultKernel.cuda().matMulTileK())
            );
            OpenClKernelConfig opencl = new OpenClKernelConfig(
                    findInt(json, "openclLoopUnrollFactor", defaultKernel.opencl().loopUnrollFactor()),
                    findInt(json, "openclMatMulTileM", defaultKernel.opencl().matMulTileM()),
                    findInt(json, "openclMatMulTileN", defaultKernel.opencl().matMulTileN()),
                    findInt(json, "openclMatMulTileK", defaultKernel.opencl().matMulTileK())
            );

            ApproximationConfig approximation = new ApproximationConfig(
                    findEnum(json, "approxMode", defaultProfile.runtime().approximation().approxMode(), ApproxMode.class),
                    findBoolean(
                            json,
                            "forceExactTranscendentals",
                            defaultProfile.runtime().approximation().forceExactTranscendentals()
                    )
            );
            BlasConfig blas = new BlasConfig(
                    BlasProvider.fromProperty(findString(json, "provider", defaultProfile.runtime().blas().provider().name())),
                    Math.max(1L, Math.round(findDouble(json, "matmulMinWork", defaultProfile.runtime().blas().matmulMinWork()))),
                    findBoolean(json, "f32RequireMgeK", defaultProfile.runtime().blas().f32RequireMgeK()),
                    findDouble(json, "f32MaxNOverK", defaultProfile.runtime().blas().f32MaxNOverK()),
                    findBoolean(json, "debug", defaultProfile.runtime().blas().debug()),
                    findEnum(json, "threadPolicy", defaultProfile.runtime().blas().threadPolicy(), BlasThreadPolicy.class),
                    findInt(json, "threads", defaultProfile.runtime().blas().threads())
            );
            RuntimeConfig runtime = new RuntimeConfig(new KernelTuningConfig(cpu, cuda, opencl), approximation, blas);

            WorkloadProfile defaultWorkload = defaultProfile.workload();
            WorkloadKind workloadKind = findEnum(json, "kind", defaultWorkload.kind(), WorkloadKind.class);
            WorkloadProfile workload = workloadKind == WorkloadKind.NONE
                    ? WorkloadProfile.none()
                    : new WorkloadProfile(
                    workloadKind,
                    findInt(json, "batch", defaultWorkload.batch()),
                    findInt(json, "heads", defaultWorkload.heads()),
                    findInt(json, "seqLen", defaultWorkload.seqLen()),
                    findInt(json, "headDim", defaultWorkload.headDim()),
                    findInt(json, "valueDim", defaultWorkload.valueDim()),
                    findInt(json, "ffHiddenDim", defaultWorkload.ffHiddenDim()),
                    findBoolean(json, "causal", defaultWorkload.causal())
            );

            return new ExecutionProfile(profileName, candidateName, dataType, mode, optimizer, runtime, workload);
        } catch (Exception e) {
            return defaultProfile;
        }
    }

    public static void saveExecutionProfile(Path path, ExecutionProfile profile) {
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
            throw new IllegalStateException("Failed to save execution profile to " + path, e);
        }
    }

    public static String toJson(ExecutionProfile profile) {
        var optimizer = profile.optimizer();
        var runtime = profile.runtime();
        var kernel = runtime.kernel();
        var cpu = kernel.cpu();
        var cuda = kernel.cuda();
        var opencl = kernel.opencl();
        var blas = runtime.blas();
        var approximation = runtime.approximation();
        var workload = profile.workload();

        return "{\n" +
                "  \"profileName\": \"" + escapeJson(profile.profileName()) + "\",\n" +
                "  \"candidateName\": \"" + escapeJson(profile.candidateName()) + "\",\n" +
                "  \"dataType\": \"" + profile.dataType().name() + "\",\n" +
                "  \"mode\": \"" + profile.mode().name() + "\",\n" +
                "  \"optimizer\": {\n" +
                "    \"stageOrder\": " + jsonStageArray(optimizer.stageOrder()) + ",\n" +
                "    \"rewrite\": {\n" +
                "      \"conv2dLoweringMode\": \"" + optimizer.rewrite().conv2dLowering().mode().name() + "\"\n" +
                "    },\n" +
                "    \"cse\": {\n" +
                "      \"strictSafety\": " + optimizer.cse().strictSafety() + "\n" +
                "    },\n" +
                "    \"fuse\": {\n" +
                "      \"maxClusterNodes\": " + optimizer.fuse().maxClusterNodes() + ",\n" +
                "      \"scoreThreshold\": " + optimizer.fuse().scoreThreshold() + ",\n" +
                "      \"internalEdgeBonus\": " + optimizer.fuse().internalEdgeBonus() + ",\n" +
                "      \"externalInputPenalty\": " + optimizer.fuse().externalInputPenalty() + ",\n" +
                "      \"sharedExpensivePenalty\": " + optimizer.fuse().sharedExpensivePenalty() + ",\n" +
                "      \"nonCheapBonus\": " + optimizer.fuse().nonCheapBonus() + ",\n" +
                "      \"preserveSharedExpensiveNodes\": " + optimizer.fuse().preserveSharedExpensiveNodes() + "\n" +
                "    },\n" +
                "    \"memory\": {\n" +
                "      \"separateForwardBackwardPools\": " + optimizer.memory().separateForwardBackwardPools() + ",\n" +
                "      \"allowCrossPhaseReuse\": " + optimizer.memory().allowCrossPhaseReuse() + ",\n" +
                "      \"allowLargerBufferReuse\": " + optimizer.memory().allowLargerBufferReuse() + ",\n" +
                "      \"minReusableBufferSize\": " + optimizer.memory().minReusableBufferSize() + "\n" +
                "    }\n" +
                "  },\n" +
                "  \"runtime\": {\n" +
                "    \"approximation\": {\n" +
                "      \"approxMode\": \"" + approximation.approxMode().name() + "\",\n" +
                "      \"forceExactTranscendentals\": " + approximation.forceExactTranscendentals() + "\n" +
                "    },\n" +
                "    \"kernel\": {\n" +
                "      \"cpu\": {\n" +
                "        \"cpuLoopUnrollFactor\": " + cpu.loopUnrollFactor() + ",\n" +
                "        \"cpuMatMulTileM\": " + cpu.matMulTileM() + ",\n" +
                "        \"cpuMatMulTileN\": " + cpu.matMulTileN() + ",\n" +
                "        \"cpuMatMulTileK\": " + cpu.matMulTileK() + ",\n" +
                "        \"cpuVectorMinSize\": " + cpu.vectorMinSize() + ",\n" +
                "        \"cpuParallelMinSize\": " + cpu.parallelMinSize() + ",\n" +
                "        \"cpuMatMulParallelMinSize\": " + cpu.matMulParallelMinSize() + ",\n" +
                "        \"cpuParallelism\": " + cpu.parallelism() + ",\n" +
                "        \"cpuChunksPerWorker\": " + cpu.chunksPerWorker() + ",\n" +
                "        \"cpuMinChunkSize\": " + cpu.minChunkSize() + ",\n" +
                "        \"cpuContiguousMaterializeThreshold\": " + cpu.contiguousMaterializeThreshold() + ",\n" +
                "        \"cpuSumAccuracyMode\": \"" + cpu.sumAccuracyMode().name() + "\",\n" +
                "        \"cpuLowCostNsPerElementThreshold\": " + cpu.lowCostNsPerElementThreshold() + ",\n" +
                "        \"cpuVectorPolicyCheap\": \"" + cpu.vectorPolicyCheap().name() + "\",\n" +
                "        \"cpuVectorPolicyTranscendental\": \"" + cpu.vectorPolicyTranscendental().name() + "\",\n" +
                "        \"cpuVectorPolicyReduction\": \"" + cpu.vectorPolicyReduction().name() + "\",\n" +
                "        \"cpuAttentionMatMulPolicy\": \"" + cpu.attentionMatMulPolicy().name() + "\"\n" +
                "      },\n" +
                "      \"cuda\": {\n" +
                "        \"cudaLoopUnrollFactor\": " + cuda.loopUnrollFactor() + ",\n" +
                "        \"cudaMatMulTileM\": " + cuda.matMulTileM() + ",\n" +
                "        \"cudaMatMulTileN\": " + cuda.matMulTileN() + ",\n" +
                "        \"cudaMatMulTileK\": " + cuda.matMulTileK() + "\n" +
                "      },\n" +
                "      \"opencl\": {\n" +
                "        \"openclLoopUnrollFactor\": " + opencl.loopUnrollFactor() + ",\n" +
                "        \"openclMatMulTileM\": " + opencl.matMulTileM() + ",\n" +
                "        \"openclMatMulTileN\": " + opencl.matMulTileN() + ",\n" +
                "        \"openclMatMulTileK\": " + opencl.matMulTileK() + "\n" +
                "      }\n" +
                "    },\n" +
                "    \"blas\": {\n" +
                "      \"provider\": \"" + blas.provider().name() + "\",\n" +
                "      \"matmulMinWork\": " + blas.matmulMinWork() + ",\n" +
                "      \"f32RequireMgeK\": " + blas.f32RequireMgeK() + ",\n" +
                "      \"f32MaxNOverK\": " + blas.f32MaxNOverK() + ",\n" +
                "      \"debug\": " + blas.debug() + ",\n" +
                "      \"threadPolicy\": \"" + blas.threadPolicy().name() + "\",\n" +
                "      \"threads\": " + blas.threads() + "\n" +
                "    },\n" +
                "    \"workload\": {\n" +
                "      \"kind\": \"" + workload.kind().name() + "\",\n" +
                "      \"batch\": " + workload.batch() + ",\n" +
                "      \"heads\": " + workload.heads() + ",\n" +
                "      \"seqLen\": " + workload.seqLen() + ",\n" +
                "      \"headDim\": " + workload.headDim() + ",\n" +
                "      \"valueDim\": " + workload.valueDim() + ",\n" +
                "      \"ffHiddenDim\": " + workload.ffHiddenDim() + ",\n" +
                "      \"causal\": " + workload.causal() + "\n" +
                "    }\n" +
                "  }\n" +
                "}\n";
    }

    private static List<OptimizerStage> parseStageOrderOrDefault(String json, List<OptimizerStage> defaultStages) {
        Matcher matcher = Pattern.compile("\"stageOrder\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL).matcher(json);
        if (!matcher.find()) {
            return defaultStages;
        }
        String body = matcher.group(1);
        Matcher tokens = Pattern.compile("\"([A-Z_]+)\"").matcher(body);
        List<OptimizerStage> out = new ArrayList<>();
        while (tokens.find()) {
            try {
                out.add(OptimizerStage.valueOf(tokens.group(1)));
            } catch (IllegalArgumentException ignored) {
                return defaultStages;
            }
        }
        return out.isEmpty() ? defaultStages : List.copyOf(out);
    }

    private static String jsonStageArray(List<OptimizerStage> stages) {
        if (stages == null || stages.isEmpty()) {
            return "[]";
        }
        return "[" + String.join(", ", stages.stream().map(stage -> "\"" + stage.name() + "\"").toList()) + "]";
    }

    private static String escapeJson(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static boolean findBoolean(String json, String key, boolean defaultValue) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(true|false)").matcher(json);
        return matcher.find() ? Boolean.parseBoolean(matcher.group(1)) : defaultValue;
    }

    private static int findInt(String json, String key, int defaultValue) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?\\d+)").matcher(json);
        return matcher.find() ? Integer.parseInt(matcher.group(1)) : defaultValue;
    }

    private static String findString(String json, String key, String defaultValue) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"([^\"]*)\"").matcher(json);
        if (!matcher.find()) {
            return defaultValue;
        }
        String value = matcher.group(1);
        return value == null || value.isBlank() ? defaultValue : value;
    }

    private static double findDouble(String json, String key, double defaultValue) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)").matcher(json);
        return matcher.find() ? Double.parseDouble(matcher.group(1)) : defaultValue;
    }

    private static <E extends Enum<E>> E findEnum(String json, String key, E defaultValue, Class<E> enumClass) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"([A-Z0-9_]+)\"").matcher(json);
        if (!matcher.find()) {
            return defaultValue;
        }
        try {
            return Enum.valueOf(enumClass, matcher.group(1));
        } catch (IllegalArgumentException ignored) {
            return defaultValue;
        }
    }
}
