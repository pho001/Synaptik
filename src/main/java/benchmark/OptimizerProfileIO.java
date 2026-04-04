package benchmark;

import backend.ApproxMode;
import backend.blas.BlasProvider;
import backend.blas.BlasThreadPolicy;
import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.backend.CudaKernelConfig;
import config.backend.KernelTuningConfig;
import config.backend.OpenClKernelConfig;
import config.backend.AttentionMatMulPolicy;
import config.backend.SumAccuracyMode;
import config.backend.VectorPolicy;
import config.profile.ExecutionProfile;
import config.profile.WorkloadKind;
import config.profile.WorkloadProfile;
import tensor.DataType;
import config.optimizer.FuseConfig;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class OptimizerProfileIO {
    private OptimizerProfileIO() {}

    private static final String HW_PROFILE_HEADER =
            "# bucket\tmode\tscore\tupdatedAt\tcandidateName\tstageOrder\tprofileJsonBase64";

    public static TuningKnobs loadKnobsOrDefault(Path path, TuningKnobs defaultKnobs) {
        if (!Files.exists(path)) return defaultKnobs;
        try {
            String json = Files.readString(path, StandardCharsets.UTF_8);
            return fromJsonOrDefault(json, defaultKnobs);
        } catch (IOException e) {
            return defaultKnobs;
        }
    }

    public static ExecutionProfile loadExecutionProfileOrDefault(Path path, ExecutionProfile defaultProfile) {
        if (!Files.exists(path)) return defaultProfile;
        try {
            String json = Files.readString(path, StandardCharsets.UTF_8);
            return executionProfileFromJsonOrDefault(json, defaultProfile);
        } catch (IOException e) {
            return defaultProfile;
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

    public static String hardwareBucketKey() {
        String os = normalizeValue(System.getProperty("os.name", "unknown"));
        String arch = normalizeValue(System.getProperty("os.arch", "unknown"));
        String vm = normalizeValue(System.getProperty("java.vm.name", "unknown"));
        String vendor = normalizeValue(System.getProperty("java.vendor", "unknown"));
        int cores = Runtime.getRuntime().availableProcessors();
        return "os=" + os + "|arch=" + arch + "|vm=" + vm + "|vendor=" + vendor + "|cores=" + cores;
    }

    public static OptimizerCandidate loadHardwareOverrideOrDefault(
            Path path,
            String bucket,
            String mode,
            OptimizerCandidate defaultCandidate
    ) {
        if (!Files.exists(path)) return defaultCandidate;
        HwProfileEntry best = null;
        for (HwProfileEntry entry : readHardwareEntries(path)) {
            if (!entry.bucket.equals(bucket) || !entry.mode.equals(mode)) {
                continue;
            }
            if (best == null || entry.score < best.score) {
                best = entry;
            }
        }
        if (best == null) {
            return defaultCandidate;
        }
        try {
            String json = new String(Base64.getDecoder().decode(best.profileJsonBase64), StandardCharsets.UTF_8);
            TuningKnobs knobs = fromJsonOrDefault(json, defaultCandidate.knobs());
            List<OptimizationStage> stages = parseStageOrderCsvOrDefault(best.stageOrderCsv, defaultCandidate.stageOrder());
            String name = best.candidateName == null || best.candidateName.isBlank()
                    ? defaultCandidate.name()
                    : best.candidateName;
            return new OptimizerCandidate(name, stages, knobs);
        } catch (IllegalArgumentException ignored) {
            return defaultCandidate;
        }
    }

    public static boolean saveHardwareProfileIfImproved(
            Path path,
            String bucket,
            String mode,
            OptimizerCandidate candidate,
            double score,
            int maxBuckets
    ) {
        List<HwProfileEntry> entries = readHardwareEntries(path);
        int existingIndex = -1;
        for (int i = 0; i < entries.size(); i++) {
            HwProfileEntry e = entries.get(i);
            if (e.bucket.equals(bucket) && e.mode.equals(mode)) {
                existingIndex = i;
                break;
            }
        }
        if (existingIndex >= 0) {
            HwProfileEntry existing = entries.get(existingIndex);
            if (score + 1e-12 >= existing.score) {
                return false;
            }
            entries.set(existingIndex, HwProfileEntry.fromCandidate(bucket, mode, candidate, score));
        } else {
            entries.add(HwProfileEntry.fromCandidate(bucket, mode, candidate, score));
        }
        trimToMaxBuckets(entries, Math.max(1, maxBuckets));
        writeHardwareEntries(path, entries);
        return true;
    }

    public static OptimizerCandidate loadArchitectureDefaultOverrideOrDefault(
            String mode,
            OptimizerCandidate defaultCandidate
    ) {
        if (defaultCandidate == null) {
            return null;
        }
        String arch = normalizeValue(System.getProperty("os.arch", "unknown")).toLowerCase(Locale.ROOT);
        if (isArmArch(arch)) {
            return applyArmPreset(mode, defaultCandidate);
        }
        if (isX86Arch(arch)) {
            return applyX86Preset(mode, defaultCandidate);
        }
        return defaultCandidate;
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

    public static void saveExecutionProfile(Path path, ExecutionProfile profile) {
        String json = toJson(profile);
        try {
            Files.createDirectories(path.getParent());
            Files.writeString(path, json, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to save execution profile to " + path, e);
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
                "      \"cpuMatMulParallelMinSize\": " + k.cpu().matMulParallelMinSize() + ",\n" +
                "      \"cpuParallelism\": " + k.cpu().parallelism() + ",\n" +
                "      \"cpuChunksPerWorker\": " + k.cpu().chunksPerWorker() + ",\n" +
                "      \"cpuMinChunkSize\": " + k.cpu().minChunkSize() + ",\n" +
                "      \"cpuContiguousMaterializeThreshold\": " + k.cpu().contiguousMaterializeThreshold() + ",\n" +
                "      \"cpuSumAccuracyMode\": \"" + k.cpu().sumAccuracyMode().name() + "\",\n" +
                "      \"cpuLowCostNsPerElementThreshold\": " + k.cpu().lowCostNsPerElementThreshold() + ",\n" +
                "      \"cpuVectorPolicyCheap\": \"" + k.cpu().vectorPolicyCheap().name() + "\",\n" +
                "      \"cpuVectorPolicyTranscendental\": \"" + k.cpu().vectorPolicyTranscendental().name() + "\",\n" +
                "      \"cpuVectorPolicyReduction\": \"" + k.cpu().vectorPolicyReduction().name() + "\"\n" +
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
                "  \"blas\": {\n" +
                "    \"provider\": \"" + knobs.blasProvider() + "\",\n" +
                "    \"matmulMinWork\": " + knobs.blasMatMulMinWork() + ",\n" +
                "    \"f32RequireMgeK\": " + knobs.blasF32RequireMgeK() + ",\n" +
                "    \"f32MaxNOverK\": " + knobs.blasF32MaxNOverK() + ",\n" +
                "    \"threadPolicy\": \"" + knobs.blasThreadPolicy().name() + "\",\n" +
                "    \"threads\": " + knobs.blasThreads() + "\n" +
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
                "    \"stageOrder\": " + jsonStageArray(profile.optimizer().stageOrder()) + ",\n" +
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

    private static TuningKnobs fromJsonOrDefault(String json, TuningKnobs d) {
        try {
            boolean strictCse = findBoolean(
                    json,
                    "strictCseSafety",
                    findBoolean(json, "strictSafety", d.strictCseSafety())
            );

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
                    findInt(json, "cpuContiguousMaterializeThreshold", d.kernelConfig().cpu().contiguousMaterializeThreshold()),
                    findEnum(json, "cpuSumAccuracyMode", d.kernelConfig().cpu().sumAccuracyMode(), SumAccuracyMode.class),
                    findDouble(
                            json,
                            "cpuLowCostNsPerElementThreshold",
                            d.kernelConfig().cpu().lowCostNsPerElementThreshold()
                    ),
                    findEnum(json, "cpuVectorPolicyCheap", d.kernelConfig().cpu().vectorPolicyCheap(), VectorPolicy.class),
                    findEnum(
                            json,
                            "cpuVectorPolicyTranscendental",
                            d.kernelConfig().cpu().vectorPolicyTranscendental(),
                            VectorPolicy.class
                    ),
                    findEnum(json, "cpuVectorPolicyReduction", d.kernelConfig().cpu().vectorPolicyReduction(), VectorPolicy.class),
                    findInt(
                            json,
                            "cpuMatMulParallelMinSize",
                            d.kernelConfig().cpu().matMulParallelMinSize()
                    ),
                    findEnum(json, "cpuAttentionMatMulPolicy", d.kernelConfig().cpu().attentionMatMulPolicy(), AttentionMatMulPolicy.class)
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

            String blasProvider = findString(json, "provider", "NONE");
            long blasMatMulMinWork = Math.max(
                    1L,
                    Math.round(findDouble(json, "matmulMinWork", 2_000_000d))
            );
            boolean blasF32RequireMgeK = findBoolean(json, "f32RequireMgeK", true);
            double blasF32MaxNOverK = findDouble(json, "f32MaxNOverK", 3.0d);
            BlasThreadPolicy blasThreadPolicy = findEnum(json, "threadPolicy", BlasThreadPolicy.AUTO, BlasThreadPolicy.class);
            int blasThreads = findInt(json, "threads", 0);

            return new TuningKnobs(
                    strictCse,
                    fuse,
                    kernel,
                    blasProvider,
                    blasMatMulMinWork,
                    blasF32RequireMgeK,
                    blasF32MaxNOverK,
                    blasThreadPolicy,
                    blasThreads
            );
        } catch (Exception e) {
            return d;
        }
    }

    private static ExecutionProfile executionProfileFromJsonOrDefault(String json, ExecutionProfile d) {
        try {
            DataType dataType = findEnum(json, "dataType", d.dataType(), DataType.class);
            ExecutionMode mode = findEnum(json, "mode", d.mode(), ExecutionMode.class);
            String profileName = findString(json, "profileName", d.profileName());
            String candidateName = findString(json, "candidateName", d.candidateName());

            List<config.optimizer.OptimizerStage> stageOrder = parseConfigStageOrderOrDefault(
                    json,
                    d.optimizer().stageOrder()
            );
            boolean strictSafety = findBoolean(
                    json,
                    "strictSafety",
                    findBoolean(json, "strictCseSafety", d.optimizer().cse().strictSafety())
            );
            FuseConfig df = d.optimizer().fuse();
            FuseConfig fuse = new FuseConfig(
                    findInt(json, "maxClusterNodes", df.maxClusterNodes()),
                    findDouble(json, "scoreThreshold", df.scoreThreshold()),
                    findDouble(json, "internalEdgeBonus", df.internalEdgeBonus()),
                    findDouble(json, "externalInputPenalty", df.externalInputPenalty()),
                    findDouble(json, "sharedExpensivePenalty", df.sharedExpensivePenalty()),
                    findDouble(json, "nonCheapBonus", df.nonCheapBonus()),
                    findBoolean(json, "preserveSharedExpensiveNodes", df.preserveSharedExpensiveNodes())
            );
            config.optimizer.OptimizerConfig optimizer = new config.optimizer.OptimizerConfig(
                    stageOrder,
                    strictSafety ? config.optimizer.CseConfig.strictDefaults() : config.optimizer.CseConfig.aggressiveDefaults(),
                    fuse
            );

            KernelTuningConfig dk = d.runtime().kernel();
            CpuKernelConfig cpu = new CpuKernelConfig(
                    findInt(json, "cpuLoopUnrollFactor", dk.cpu().loopUnrollFactor()),
                    findInt(json, "cpuMatMulTileM", dk.cpu().matMulTileM()),
                    findInt(json, "cpuMatMulTileN", dk.cpu().matMulTileN()),
                    findInt(json, "cpuMatMulTileK", dk.cpu().matMulTileK()),
                    findInt(json, "cpuVectorMinSize", dk.cpu().vectorMinSize()),
                    findInt(json, "cpuParallelMinSize", dk.cpu().parallelMinSize()),
                    findInt(json, "cpuParallelism", dk.cpu().parallelism()),
                    findInt(json, "cpuChunksPerWorker", dk.cpu().chunksPerWorker()),
                    findInt(json, "cpuMinChunkSize", dk.cpu().minChunkSize()),
                    findInt(json, "cpuContiguousMaterializeThreshold", dk.cpu().contiguousMaterializeThreshold()),
                    findEnum(json, "cpuSumAccuracyMode", dk.cpu().sumAccuracyMode(), SumAccuracyMode.class),
                    findDouble(json, "cpuLowCostNsPerElementThreshold", dk.cpu().lowCostNsPerElementThreshold()),
                    findEnum(json, "cpuVectorPolicyCheap", dk.cpu().vectorPolicyCheap(), VectorPolicy.class),
                    findEnum(json, "cpuVectorPolicyTranscendental", dk.cpu().vectorPolicyTranscendental(), VectorPolicy.class),
                    findEnum(json, "cpuVectorPolicyReduction", dk.cpu().vectorPolicyReduction(), VectorPolicy.class),
                    findInt(json, "cpuMatMulParallelMinSize", dk.cpu().matMulParallelMinSize()),
                    findEnum(json, "cpuAttentionMatMulPolicy", dk.cpu().attentionMatMulPolicy(), AttentionMatMulPolicy.class)
            );
            CudaKernelConfig cuda = new CudaKernelConfig(
                    findInt(json, "cudaLoopUnrollFactor", dk.cuda().loopUnrollFactor()),
                    findInt(json, "cudaMatMulTileM", dk.cuda().matMulTileM()),
                    findInt(json, "cudaMatMulTileN", dk.cuda().matMulTileN()),
                    findInt(json, "cudaMatMulTileK", dk.cuda().matMulTileK())
            );
            OpenClKernelConfig opencl = new OpenClKernelConfig(
                    findInt(json, "openclLoopUnrollFactor", dk.opencl().loopUnrollFactor()),
                    findInt(json, "openclMatMulTileM", dk.opencl().matMulTileM()),
                    findInt(json, "openclMatMulTileN", dk.opencl().matMulTileN()),
                    findInt(json, "openclMatMulTileK", dk.opencl().matMulTileK())
            );
            config.runtime.ApproximationConfig approximation = new config.runtime.ApproximationConfig(
                    findEnum(json, "approxMode", d.runtime().approximation().approxMode(), ApproxMode.class),
                    findBoolean(
                            json,
                            "forceExactTranscendentals",
                            d.runtime().approximation().forceExactTranscendentals()
                    )
            );
            config.runtime.BlasConfig blas = new config.runtime.BlasConfig(
                    BlasProvider.fromProperty(findString(json, "provider", d.runtime().blas().provider().name())),
                    Math.max(1L, Math.round(findDouble(json, "matmulMinWork", d.runtime().blas().matmulMinWork()))),
                    findBoolean(json, "f32RequireMgeK", d.runtime().blas().f32RequireMgeK()),
                    findDouble(json, "f32MaxNOverK", d.runtime().blas().f32MaxNOverK()),
                    findBoolean(json, "debug", d.runtime().blas().debug()),
                    findEnum(json, "threadPolicy", d.runtime().blas().threadPolicy(), BlasThreadPolicy.class),
                    findInt(json, "threads", d.runtime().blas().threads())
            );
            config.runtime.RuntimeConfig runtime = new config.runtime.RuntimeConfig(
                    new KernelTuningConfig(cpu, cuda, opencl),
                    approximation,
                    blas
            );
            WorkloadProfile defaultWorkload = d.workload();
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

    private static List<OptimizationStage> parseStageOrderCsvOrDefault(String csv, List<OptimizationStage> defaultStages) {
        if (csv == null || csv.isBlank()) return defaultStages;
        String[] parts = csv.split(",");
        List<OptimizationStage> out = new ArrayList<>();
        for (String part : parts) {
            String token = part.trim();
            if (token.isEmpty()) continue;
            try {
                out.add(OptimizationStage.valueOf(token));
            } catch (IllegalArgumentException ignored) {
                return defaultStages;
            }
        }
        return out.isEmpty() ? defaultStages : List.copyOf(out);
    }

    private static List<config.optimizer.OptimizerStage> parseConfigStageOrderOrDefault(
            String json,
            List<config.optimizer.OptimizerStage> defaultStages
    ) {
        Matcher m = Pattern.compile("\"stageOrder\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL).matcher(json);
        if (!m.find()) return defaultStages;

        String body = m.group(1);
        Matcher tokens = Pattern.compile("\"([A-Z_]+)\"").matcher(body);
        List<config.optimizer.OptimizerStage> out = new ArrayList<>();
        while (tokens.find()) {
            try {
                out.add(config.optimizer.OptimizerStage.valueOf(tokens.group(1)));
            } catch (IllegalArgumentException ignored) {
                return defaultStages;
            }
        }
        return out.isEmpty() ? defaultStages : List.copyOf(out);
    }

    private static List<HwProfileEntry> readHardwareEntries(Path path) {
        if (!Files.exists(path)) return new ArrayList<>();
        List<HwProfileEntry> out = new ArrayList<>();
        try {
            List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
            for (String line : lines) {
                if (line == null || line.isBlank() || line.startsWith("#")) {
                    continue;
                }
                String[] cols = line.split("\t", 7);
                if (cols.length < 7) {
                    continue;
                }
                double score;
                try {
                    score = Double.parseDouble(cols[2]);
                } catch (NumberFormatException ignored) {
                    continue;
                }
                out.add(new HwProfileEntry(
                        cols[0],
                        cols[1],
                        score,
                        cols[3],
                        cols[4],
                        cols[5],
                        cols[6]
                ));
            }
        } catch (IOException ignored) {
            return new ArrayList<>();
        }
        return out;
    }

    private static void writeHardwareEntries(Path path, List<HwProfileEntry> entries) {
        try {
            Files.createDirectories(path.getParent());
            List<String> lines = new ArrayList<>(entries.size() + 1);
            lines.add(HW_PROFILE_HEADER);
            for (HwProfileEntry e : entries) {
                lines.add(e.toLine());
            }
            Files.write(path, lines, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write HW optimizer profiles to " + path, e);
        }
    }

    private static void trimToMaxBuckets(List<HwProfileEntry> entries, int maxBuckets) {
        Map<String, String> latestByBucket = new HashMap<>();
        for (HwProfileEntry entry : entries) {
            String latest = latestByBucket.get(entry.bucket);
            if (latest == null || entry.updatedAt.compareTo(latest) > 0) {
                latestByBucket.put(entry.bucket, entry.updatedAt);
            }
        }
        while (latestByBucket.size() > maxBuckets) {
            String oldestBucket = null;
            String oldestTs = null;
            for (Map.Entry<String, String> kv : latestByBucket.entrySet()) {
                if (oldestTs == null || kv.getValue().compareTo(oldestTs) < 0) {
                    oldestTs = kv.getValue();
                    oldestBucket = kv.getKey();
                }
            }
            if (oldestBucket == null) {
                break;
            }
            String bucketToRemove = oldestBucket;
            entries.removeIf(e -> e.bucket.equals(bucketToRemove));
            latestByBucket.remove(bucketToRemove);
        }
    }

    private static String normalizeValue(String value) {
        if (value == null || value.isBlank()) {
            return "unknown";
        }
        return value
                .replace('|', '_')
                .replace('\t', '_')
                .replace('\n', '_')
                .replace('\r', '_')
                .trim();
    }

    private static String jsonStageArray(List<config.optimizer.OptimizerStage> stages) {
        if (stages == null || stages.isEmpty()) {
            return "[]";
        }
        return "[" + String.join(", ", stages.stream().map(stage -> "\"" + stage.name() + "\"").toList()) + "]";
    }

    private static String escapeJson(String value) {
        if (value == null) {
            return "";
        }
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static boolean isArmArch(String arch) {
        return "aarch64".equals(arch) || "arm64".equals(arch);
    }

    private static boolean isX86Arch(String arch) {
        return "x86_64".equals(arch) || "amd64".equals(arch);
    }

    private static OptimizerCandidate applyArmPreset(String mode, OptimizerCandidate base) {
        boolean inference = "INFERENCE".equalsIgnoreCase(mode);
        List<OptimizationStage> stageOrder = inference
                ? List.of(OptimizationStage.CSE, OptimizationStage.FUSE, OptimizationStage.MEM)
                : List.of(OptimizationStage.FUSE, OptimizationStage.CSE);
        var src = base.knobs();
        var ksrc = src.kernelConfig();
        CpuKernelConfig cpu = inference
                ? new CpuKernelConfig(
                        4, 32, 32, 32,
                        512, 100_000, 0, 4, 4_096, 65_536,
                        SumAccuracyMode.FAST, 4.0d,
                        VectorPolicy.FORCE_ON, VectorPolicy.FORCE_OFF, VectorPolicy.AUTO,
                        2_000_000
                )
                : new CpuKernelConfig(
                        4, 32, 32, 32,
                        256, 50_000, 0, 2, 2_048, 16_384,
                        SumAccuracyMode.FAST, 1.0d,
                        VectorPolicy.FORCE_ON, VectorPolicy.FORCE_OFF, VectorPolicy.AUTO,
                        8_000_000
                );
        KernelTuningConfig kernel = new KernelTuningConfig(cpu, ksrc.cuda(), ksrc.opencl());
        TuningKnobs knobs = new TuningKnobs(
                src.strictCseSafety(),
                src.fuseConfig(),
                kernel,
                src.blasProvider(),
                src.blasMatMulMinWork(),
                src.blasF32RequireMgeK(),
                src.blasF32MaxNOverK()
        );
        return new OptimizerCandidate(base.name(), stageOrder, knobs);
    }

    private static OptimizerCandidate applyX86Preset(String mode, OptimizerCandidate base) {
        boolean inference = "INFERENCE".equalsIgnoreCase(mode);
        List<OptimizationStage> stageOrder = inference
                ? List.of(OptimizationStage.AR, OptimizationStage.CSE, OptimizationStage.FUSE, OptimizationStage.MEM)
                : List.of(OptimizationStage.AR, OptimizationStage.CSE, OptimizationStage.MEM);
        var src = base.knobs();
        var ksrc = src.kernelConfig();
        CpuKernelConfig cpu = inference
                ? new CpuKernelConfig(
                        4, 32, 32, 32,
                        1_024, 100_000, 0, 4, 4_096, 1_000_000_000,
                        SumAccuracyMode.FAST, 2.0d,
                        VectorPolicy.AUTO, VectorPolicy.AUTO, VectorPolicy.AUTO,
                        2_000_000
                )
                : new CpuKernelConfig(
                        4, 32, 32, 32,
                        1_024, 100_000, 0, 4, 4_096, 65_536,
                        SumAccuracyMode.FAST, 2.0d,
                        VectorPolicy.AUTO, VectorPolicy.AUTO, VectorPolicy.AUTO,
                        2_000_000
                );
        KernelTuningConfig kernel = new KernelTuningConfig(cpu, ksrc.cuda(), ksrc.opencl());
        TuningKnobs knobs = new TuningKnobs(
                src.strictCseSafety(),
                src.fuseConfig(),
                kernel,
                src.blasProvider(),
                src.blasMatMulMinWork(),
                src.blasF32RequireMgeK(),
                src.blasF32MaxNOverK()
        );
        return new OptimizerCandidate(base.name(), stageOrder, knobs);
    }

    private static final class HwProfileEntry {
        private final String bucket;
        private final String mode;
        private final double score;
        private final String updatedAt;
        private final String candidateName;
        private final String stageOrderCsv;
        private final String profileJsonBase64;

        private HwProfileEntry(
                String bucket,
                String mode,
                double score,
                String updatedAt,
                String candidateName,
                String stageOrderCsv,
                String profileJsonBase64
        ) {
            this.bucket = bucket;
            this.mode = mode;
            this.score = score;
            this.updatedAt = updatedAt;
            this.candidateName = candidateName;
            this.stageOrderCsv = stageOrderCsv;
            this.profileJsonBase64 = profileJsonBase64;
        }

        private static HwProfileEntry fromCandidate(String bucket, String mode, OptimizerCandidate candidate, double score) {
            String profileJson = toJson(candidate.knobs(), candidate.name());
            String payload = Base64.getEncoder().encodeToString(profileJson.getBytes(StandardCharsets.UTF_8));
            String stageCsv = String.join(",", candidate.stageOrder().stream().map(Enum::name).toList());
            return new HwProfileEntry(
                    sanitize(bucket),
                    sanitize(mode),
                    score,
                    OffsetDateTime.now().toString(),
                    sanitize(candidate.name()),
                    sanitize(stageCsv),
                    payload
            );
        }

        private String toLine() {
            return bucket
                    + "\t" + mode
                    + "\t" + String.format(Locale.US, "%.12f", score)
                    + "\t" + updatedAt
                    + "\t" + candidateName
                    + "\t" + stageOrderCsv
                    + "\t" + profileJsonBase64;
        }

        private static String sanitize(String value) {
            if (value == null) {
                return "";
            }
            return value.replace('\t', ' ')
                    .replace('\n', ' ')
                    .replace('\r', ' ');
        }
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

    private static String findString(String json, String key, String defaultValue) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"([^\"]*)\"").matcher(json);
        if (!m.find()) return defaultValue;
        String v = m.group(1);
        return (v == null || v.isBlank()) ? defaultValue : v;
    }

    private static double findDouble(String json, String key, double defaultValue) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)").matcher(json);
        if (!m.find()) return defaultValue;
        return Double.parseDouble(m.group(1));
    }

    private static <E extends Enum<E>> E findEnum(String json, String key, E defaultValue, Class<E> enumClass) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"([A-Z0-9_]+)\"").matcher(json);
        if (!m.find()) return defaultValue;
        try {
            return Enum.valueOf(enumClass, m.group(1));
        } catch (IllegalArgumentException ignored) {
            return defaultValue;
        }
    }
}
