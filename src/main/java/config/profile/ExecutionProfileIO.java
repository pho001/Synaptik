package config.profile;

import backend.ApproxMode;
import backend.blas.BlasProvider;
import backend.runtime.ExecutionMode;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuMatMulMicroKernel;
import config.backend.CpuKernelConfig;
import config.backend.CudaKernelConfig;
import config.backend.KernelTuningConfig;
import config.backend.OpenClKernelConfig;
import config.backend.SumAccuracyMode;
import config.optimizer.AlgebraicRewriteConfig;
import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import config.optimizer.CseConfig;
import config.optimizer.FuseConfig;
import config.optimizer.LinearLoweringConfig;
import config.optimizer.MemoryConfig;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.optimizer.PartitionConfig;
import config.optimizer.PiecewiseLoweringConfig;
import config.optimizer.RewriteConfig;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.Conv2dConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.FusedPrimaryBackend;
import config.runtime.RuntimeConfig;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionPlannerStrategy;
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
                    new AlgebraicRewriteConfig(
                            findBoolean(json, "algebraicEnabled", defaultRewrite.algebraic().enabled())
                    ),
                    new LinearLoweringConfig(
                            findBoolean(json, "linearLoweringEnabled", defaultRewrite.linearLowering().enabled())
                    ),
                    new Conv2dLoweringConfig(
                            findEnum(
                                    json,
                                    "conv2dLoweringMode",
                                    defaultRewrite.conv2dLowering().mode(),
                                    Conv2dLoweringMode.class
                            )
                    ),
                    new PiecewiseLoweringConfig(
                            findBoolean(
                                    json,
                                    "canonicalSigmoid",
                                    defaultRewrite.piecewiseLowering().canonicalSigmoid()
                            ),
                            findBoolean(
                                    json,
                                    "reluLikeWhere",
                                    defaultRewrite.piecewiseLowering().reluLikeWhere()
                            ),
                            findBoolean(
                                    json,
                                    "clampLikeWhere",
                                    defaultRewrite.piecewiseLowering().clampLikeWhere()
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
            PartitionConfig defaultPartition = defaultProfile.optimizer().partition();
            PartitionConfig partition = new PartitionConfig(
                    findInt(json, "partitionMaxSearchNodes", defaultPartition.maxSearchNodes()),
                    findInt(json, "partitionMaxVisitedCandidates", defaultPartition.maxVisitedCandidates()),
                    findDouble(json, "partitionNodeWeight", defaultPartition.nodeWeight()),
                    findDouble(json, "partitionInternalEdgeWeight", defaultPartition.internalEdgeWeight()),
                    findDouble(json, "partitionMergeNodeBonus", defaultPartition.mergeNodeBonus()),
                    findDouble(json, "partitionTailDepthWeight", defaultPartition.tailDepthWeight()),
                    findDouble(json, "partitionExternalInputPenalty", defaultPartition.externalInputPenalty()),
                    findDouble(json, "partitionWorkWeight", defaultPartition.workWeight()),
                    findEnum(json, "partitionPlannerStrategy", defaultPartition.plannerStrategy(), PartitionPlannerStrategy.class),
                    findEnum(json, "partitionAcceleratorTarget", defaultPartition.target(), PartitionTarget.class)
            );
            OptimizerConfig optimizer = new OptimizerConfig(
                    stageOrder,
                    rewrite,
                    strictSafety ? CseConfig.strictDefaults() : CseConfig.aggressiveDefaults(),
                    fuse,
                    memory,
                    partition
            );

            KernelTuningConfig defaultKernel = defaultProfile.runtime().kernel();
            CpuMatMulMicroKernel loadedMatMulMicroKernel = findEnum(
                    json,
                    "cpuMatMulMicroKernel",
                    defaultKernel.cpu().matMulMicroKernel(),
                    CpuMatMulMicroKernel.class
            );
            int loadedMatMulTileM = findInt(json, "cpuMatMulTileM", defaultKernel.cpu().matMulTileM());
            int loadedMatMulTileN = findInt(json, "cpuMatMulTileN", defaultKernel.cpu().matMulTileN());
            int loadedMatMulTileK = findInt(json, "cpuMatMulTileK", defaultKernel.cpu().matMulTileK());
            CpuKernelConfig cpu = new CpuKernelConfig(
                    findInt(json, "cpuLoopUnrollFactor", defaultKernel.cpu().loopUnrollFactor()),
                    loadedMatMulTileM,
                    loadedMatMulTileN,
                    loadedMatMulTileK,
                    findInt(json, "cpuCheapVectorMinSize", defaultKernel.cpu().cheapVectorMinSize()),
                    findInt(json, "cpuTranscendentalVectorMinSize", defaultKernel.cpu().transcendentalVectorMinSize()),
                    findInt(json, "cpuFusedCheapVectorMinSize", defaultKernel.cpu().fusedCheapVectorMinSize()),
                    findInt(json, "cpuFusedTranscendentalVectorMinSize", defaultKernel.cpu().fusedTranscendentalVectorMinSize()),
                    findInt(json, "cpuReductionVectorMinSize", defaultKernel.cpu().reductionVectorMinSize()),
                    findInt(json, "cpuAttentionVectorMinSize", defaultKernel.cpu().attentionVectorMinSize()),
                    findInt(json, "cpuCheapParallelMinSize", defaultKernel.cpu().cheapParallelMinSize()),
                    findInt(json, "cpuTranscendentalParallelMinSize", defaultKernel.cpu().transcendentalParallelMinSize()),
                    findInt(json, "cpuFusedCheapParallelMinSize", defaultKernel.cpu().fusedCheapParallelMinSize()),
                    findInt(json, "cpuFusedTranscendentalParallelMinSize", defaultKernel.cpu().fusedTranscendentalParallelMinSize()),
                    findInt(json, "cpuReductionParallelMinSize", defaultKernel.cpu().reductionParallelMinSize()),
                    findInt(json, "cpuAttentionParallelMinSize", defaultKernel.cpu().attentionParallelMinSize()),
                    findInt(json, "cpuContiguousMaterializeThreshold", defaultKernel.cpu().contiguousMaterializeThreshold()),
                    findInt(json, "cpuCheapF64MaterializeThreshold", defaultKernel.cpu().cheapF64MaterializeThreshold()),
                    findInt(json, "cpuCheapF32MaterializeThreshold", defaultKernel.cpu().cheapF32MaterializeThreshold()),
                    findInt(json, "cpuCheapBF16MaterializeThreshold", defaultKernel.cpu().cheapBF16MaterializeThreshold()),
                    findInt(json, "cpuWhereMaterializeThreshold", defaultKernel.cpu().whereMaterializeThreshold()),
                    findInt(json, "cpuLowCostTargetChunksPerWorker", defaultKernel.cpu().lowCostTargetChunksPerWorker()),
                    findInt(json, "cpuMediumCostTargetChunksPerWorker", defaultKernel.cpu().mediumCostTargetChunksPerWorker()),
                    findInt(json, "cpuHighCostTargetChunksPerWorker", defaultKernel.cpu().highCostTargetChunksPerWorker()),
                    findInt(json, "cpuMinScalarChunkSize", defaultKernel.cpu().minScalarChunkSize()),
                    findInt(json, "cpuMinVectorChunkSize", defaultKernel.cpu().minVectorChunkSize()),
                    findInt(json, "cpuMinReductionChunkSize", defaultKernel.cpu().minReductionChunkSize()),
                    findInt(json, "cpuCommonPoolLowCostMaxWorkPerWorker", defaultKernel.cpu().commonPoolLowCostMaxWorkPerWorker()),
                    findInt(
                            json,
                            "cpuFusedCheapContiguousAsmVectorWidth",
                            findInt(json, "cpuFusedAsmVectorWidth", defaultKernel.cpu().fusedCheapContiguousAsmVectorWidth())
                    ),
                    findInt(
                            json,
                            "cpuFusedCheapStridedAsmVectorWidth",
                            findInt(json, "cpuFusedAsmVectorWidth", defaultKernel.cpu().fusedCheapStridedAsmVectorWidth())
                    ),
                    findInt(
                            json,
                            "cpuFusedNonCheapContiguousAsmVectorWidth",
                            findInt(json, "cpuFusedAsmVectorWidth", defaultKernel.cpu().fusedNonCheapContiguousAsmVectorWidth())
                    ),
                    findInt(
                            json,
                            "cpuFusedNonCheapStridedAsmVectorWidth",
                            findInt(json, "cpuFusedAsmVectorWidth", defaultKernel.cpu().fusedNonCheapStridedAsmVectorWidth())
                    ),
                    findEnum(json, "cpuSumAccuracyMode", defaultKernel.cpu().sumAccuracyMode(), SumAccuracyMode.class),
                    findInt(json, "cpuMatMulParallelMinSize", defaultKernel.cpu().matMulParallelMinSize()),
                    findEnum(json, "cpuAttentionMatMulPolicy", defaultKernel.cpu().attentionMatMulPolicy(), AttentionMatMulPolicy.class),
                    loadedMatMulMicroKernel,
                    findEnum(
                            json,
                            "cpuAttentionMatMulMicroKernel",
                            loadedMatMulMicroKernel,
                            CpuMatMulMicroKernel.class
                    ),
                    findInt(json, "cpuAttentionMatMulTileM", loadedMatMulTileM),
                    findInt(json, "cpuAttentionMatMulTileN", loadedMatMulTileN),
                    findInt(json, "cpuAttentionMatMulTileK", loadedMatMulTileK)
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
                    findBoolean(json, "f32WideRequireMgeK", defaultProfile.runtime().blas().f32WideRequireMgeK()),
                    findDouble(json, "f32WideMaxNOverK", defaultProfile.runtime().blas().f32WideMaxNOverK()),
                    findBoolean(json, "debug", defaultProfile.runtime().blas().debug()),
                    findInt(json, "threads", defaultProfile.runtime().blas().threads())
            );
            Conv2dConfig conv2d = new Conv2dConfig(
                    BlasProvider.fromProperty(findString(
                            json,
                            "conv2dProvider",
                            defaultProfile.runtime().conv2d().provider().name()
                    )),
                    Math.max(1L, Math.round(findDouble(
                            json,
                            "conv2dF64MinWork",
                            defaultProfile.runtime().conv2d().f64MinWork()
                    ))),
                    Math.max(1L, Math.round(findDouble(
                            json,
                            "conv2dF32MinWork",
                            defaultProfile.runtime().conv2d().f32MinWork()
                    ))),
                    findBoolean(
                            json,
                            "conv2dF32RequireMgeK",
                            defaultProfile.runtime().conv2d().f32RequireMgeK()
                    ),
                    findDouble(
                            json,
                            "conv2dF32MaxNOverK",
                            defaultProfile.runtime().conv2d().f32MaxNOverK()
                    ),
                    Math.max(1L, Math.round(findDouble(
                            json,
                            "conv2dBf16MinWork",
                            defaultProfile.runtime().conv2d().bf16MinWork()
                    ))),
                    findBoolean(
                            json,
                            "conv2dBf16RequireMgeK",
                            defaultProfile.runtime().conv2d().bf16RequireMgeK()
                    ),
                    findDouble(
                            json,
                            "conv2dBf16MaxNOverK",
                            defaultProfile.runtime().conv2d().bf16MaxNOverK()
                    )
            );
            FusedExecutionPolicy fused = new FusedExecutionPolicy(
                    findEnum(json, "fusedPrimaryBackend", defaultProfile.runtime().fused().primaryBackend(), FusedPrimaryBackend.class),
                    findBoolean(json, "fusedAllowBackendFallback", defaultProfile.runtime().fused().allowBackendFallback())
            );
            AcceleratorConfig accelerator = new AcceleratorConfig(
                    new AcceleratorBackendConfig(
                            findBoolean(json, "cudaEnabled", defaultProfile.runtime().accelerator().cuda().enabled()),
                            findBoolean(
                                    json,
                                    "cudaRequireRuntimeAvailability",
                                    defaultProfile.runtime().accelerator().cuda().requireRuntimeAvailability()
                            ),
                            findLong(
                                    json,
                                    "cudaMinimumEstimatedWork",
                                    defaultProfile.runtime().accelerator().cuda().minimumEstimatedWork()
                            )
                    ),
                    new AcceleratorBackendConfig(
                            findBoolean(json, "openclEnabled", defaultProfile.runtime().accelerator().opencl().enabled()),
                            findBoolean(
                                    json,
                                    "openclRequireRuntimeAvailability",
                                    defaultProfile.runtime().accelerator().opencl().requireRuntimeAvailability()
                            ),
                            findLong(
                                    json,
                                    "openclMinimumEstimatedWork",
                                    defaultProfile.runtime().accelerator().opencl().minimumEstimatedWork()
                            )
                    ),
                    new AcceleratorBackendConfig(
                            findBoolean(json, "metalEnabled", defaultProfile.runtime().accelerator().metal().enabled()),
                            findBoolean(
                                    json,
                                    "metalRequireRuntimeAvailability",
                                    defaultProfile.runtime().accelerator().metal().requireRuntimeAvailability()
                            ),
                            findLong(
                                    json,
                                    "metalMinimumEstimatedWork",
                                    defaultProfile.runtime().accelerator().metal().minimumEstimatedWork()
                            )
                    )
            );
            RuntimeConfig runtime = new RuntimeConfig(new KernelTuningConfig(cpu, cuda, opencl), approximation, blas, conv2d, fused, accelerator);

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
        var conv2d = runtime.conv2d();
        var approximation = runtime.approximation();
        var fused = runtime.fused();
        var accelerator = runtime.accelerator();
        var workload = profile.workload();

        return "{\n" +
                "  \"profileName\": \"" + escapeJson(profile.profileName()) + "\",\n" +
                "  \"candidateName\": \"" + escapeJson(profile.candidateName()) + "\",\n" +
                "  \"dataType\": \"" + profile.dataType().name() + "\",\n" +
                "  \"mode\": \"" + profile.mode().name() + "\",\n" +
                "  \"optimizer\": {\n" +
                "    \"stageOrder\": " + jsonStageArray(optimizer.stageOrder()) + ",\n" +
                "    \"rewrite\": {\n" +
                "      \"algebraicEnabled\": " + optimizer.rewrite().algebraic().enabled() + ",\n" +
                "      \"linearLoweringEnabled\": " + optimizer.rewrite().linearLowering().enabled() + ",\n" +
                "      \"conv2dLoweringMode\": \"" + optimizer.rewrite().conv2dLowering().mode().name() + "\",\n" +
                "      \"piecewiseLowering\": {\n" +
                "        \"canonicalSigmoid\": " + optimizer.rewrite().piecewiseLowering().canonicalSigmoid() + ",\n" +
                "        \"reluLikeWhere\": " + optimizer.rewrite().piecewiseLowering().reluLikeWhere() + ",\n" +
                "        \"clampLikeWhere\": " + optimizer.rewrite().piecewiseLowering().clampLikeWhere() + "\n" +
                "      }\n" +
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
                "    },\n" +
                "    \"partition\": {\n" +
                "      \"partitionMaxSearchNodes\": " + optimizer.partition().maxSearchNodes() + ",\n" +
                "      \"partitionMaxVisitedCandidates\": " + optimizer.partition().maxVisitedCandidates() + ",\n" +
                "      \"partitionNodeWeight\": " + optimizer.partition().nodeWeight() + ",\n" +
                "      \"partitionInternalEdgeWeight\": " + optimizer.partition().internalEdgeWeight() + ",\n" +
                "      \"partitionMergeNodeBonus\": " + optimizer.partition().mergeNodeBonus() + ",\n" +
                "      \"partitionTailDepthWeight\": " + optimizer.partition().tailDepthWeight() + ",\n" +
                "      \"partitionExternalInputPenalty\": " + optimizer.partition().externalInputPenalty() + ",\n" +
                "      \"partitionWorkWeight\": " + optimizer.partition().workWeight() + ",\n" +
                "      \"partitionPlannerStrategy\": \"" + optimizer.partition().plannerStrategy().name() + "\",\n" +
                "      \"partitionAcceleratorTarget\": \"" + optimizer.partition().target().name() + "\"\n" +
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
                "        \"cpuAttentionMatMulTileM\": " + cpu.attentionMatMulTileM() + ",\n" +
                "        \"cpuAttentionMatMulTileN\": " + cpu.attentionMatMulTileN() + ",\n" +
                "        \"cpuAttentionMatMulTileK\": " + cpu.attentionMatMulTileK() + ",\n" +
                "        \"cpuCheapVectorMinSize\": " + cpu.cheapVectorMinSize() + ",\n" +
                "        \"cpuTranscendentalVectorMinSize\": " + cpu.transcendentalVectorMinSize() + ",\n" +
                "        \"cpuFusedCheapVectorMinSize\": " + cpu.fusedCheapVectorMinSize() + ",\n" +
                "        \"cpuFusedTranscendentalVectorMinSize\": " + cpu.fusedTranscendentalVectorMinSize() + ",\n" +
                "        \"cpuReductionVectorMinSize\": " + cpu.reductionVectorMinSize() + ",\n" +
                "        \"cpuAttentionVectorMinSize\": " + cpu.attentionVectorMinSize() + ",\n" +
                "        \"cpuCheapParallelMinSize\": " + cpu.cheapParallelMinSize() + ",\n" +
                "        \"cpuTranscendentalParallelMinSize\": " + cpu.transcendentalParallelMinSize() + ",\n" +
                "        \"cpuFusedCheapParallelMinSize\": " + cpu.fusedCheapParallelMinSize() + ",\n" +
                "        \"cpuFusedTranscendentalParallelMinSize\": " + cpu.fusedTranscendentalParallelMinSize() + ",\n" +
                "        \"cpuReductionParallelMinSize\": " + cpu.reductionParallelMinSize() + ",\n" +
                "        \"cpuAttentionParallelMinSize\": " + cpu.attentionParallelMinSize() + ",\n" +
                "        \"cpuMatMulParallelMinSize\": " + cpu.matMulParallelMinSize() + ",\n" +
                "        \"cpuContiguousMaterializeThreshold\": " + cpu.contiguousMaterializeThreshold() + ",\n" +
                "        \"cpuCheapF64MaterializeThreshold\": " + cpu.cheapF64MaterializeThreshold() + ",\n" +
                "        \"cpuCheapF32MaterializeThreshold\": " + cpu.cheapF32MaterializeThreshold() + ",\n" +
                "        \"cpuCheapBF16MaterializeThreshold\": " + cpu.cheapBF16MaterializeThreshold() + ",\n" +
                "        \"cpuWhereMaterializeThreshold\": " + cpu.whereMaterializeThreshold() + ",\n" +
                "        \"cpuLowCostTargetChunksPerWorker\": " + cpu.lowCostTargetChunksPerWorker() + ",\n" +
                "        \"cpuMediumCostTargetChunksPerWorker\": " + cpu.mediumCostTargetChunksPerWorker() + ",\n" +
                "        \"cpuHighCostTargetChunksPerWorker\": " + cpu.highCostTargetChunksPerWorker() + ",\n" +
                "        \"cpuMinScalarChunkSize\": " + cpu.minScalarChunkSize() + ",\n" +
                "        \"cpuMinVectorChunkSize\": " + cpu.minVectorChunkSize() + ",\n" +
                "        \"cpuMinReductionChunkSize\": " + cpu.minReductionChunkSize() + ",\n" +
                "        \"cpuCommonPoolLowCostMaxWorkPerWorker\": " + cpu.commonPoolLowCostMaxWorkPerWorker() + ",\n" +
                "        \"cpuFusedCheapContiguousAsmVectorWidth\": " + cpu.fusedCheapContiguousAsmVectorWidth() + ",\n" +
                "        \"cpuFusedCheapStridedAsmVectorWidth\": " + cpu.fusedCheapStridedAsmVectorWidth() + ",\n" +
                "        \"cpuFusedNonCheapContiguousAsmVectorWidth\": " + cpu.fusedNonCheapContiguousAsmVectorWidth() + ",\n" +
                "        \"cpuFusedNonCheapStridedAsmVectorWidth\": " + cpu.fusedNonCheapStridedAsmVectorWidth() + ",\n" +
                "        \"cpuSumAccuracyMode\": \"" + cpu.sumAccuracyMode().name() + "\",\n" +
                "        \"cpuAttentionMatMulPolicy\": \"" + cpu.attentionMatMulPolicy().name() + "\",\n" +
                "        \"cpuMatMulMicroKernel\": \"" + cpu.matMulMicroKernel().name() + "\",\n" +
                "        \"cpuAttentionMatMulMicroKernel\": \"" + cpu.attentionMatMulMicroKernel().name() + "\"\n" +
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
                "      \"f32WideRequireMgeK\": " + blas.f32WideRequireMgeK() + ",\n" +
                "      \"f32WideMaxNOverK\": " + blas.f32WideMaxNOverK() + ",\n" +
                "      \"debug\": " + blas.debug() + ",\n" +
                "      \"threads\": " + blas.threads() + "\n" +
                "    },\n" +
                "    \"conv2d\": {\n" +
                "      \"conv2dProvider\": \"" + conv2d.provider().name() + "\",\n" +
                "      \"conv2dF64MinWork\": " + conv2d.f64MinWork() + ",\n" +
                "      \"conv2dF32MinWork\": " + conv2d.f32MinWork() + ",\n" +
                "      \"conv2dF32RequireMgeK\": " + conv2d.f32RequireMgeK() + ",\n" +
                "      \"conv2dF32MaxNOverK\": " + conv2d.f32MaxNOverK() + ",\n" +
                "      \"conv2dBf16MinWork\": " + conv2d.bf16MinWork() + ",\n" +
                "      \"conv2dBf16RequireMgeK\": " + conv2d.bf16RequireMgeK() + ",\n" +
                "      \"conv2dBf16MaxNOverK\": " + conv2d.bf16MaxNOverK() + "\n" +
                "    },\n" +
                "    \"fused\": {\n" +
                "      \"fusedPrimaryBackend\": \"" + fused.primaryBackend().name() + "\",\n" +
                "      \"fusedAllowBackendFallback\": " + fused.allowBackendFallback() + "\n" +
                "    },\n" +
                "    \"accelerator\": {\n" +
                "      \"cudaEnabled\": " + accelerator.cuda().enabled() + ",\n" +
                "      \"cudaRequireRuntimeAvailability\": " + accelerator.cuda().requireRuntimeAvailability() + ",\n" +
                "      \"cudaMinimumEstimatedWork\": " + accelerator.cuda().minimumEstimatedWork() + ",\n" +
                "      \"openclEnabled\": " + accelerator.opencl().enabled() + ",\n" +
                "      \"openclRequireRuntimeAvailability\": " + accelerator.opencl().requireRuntimeAvailability() + ",\n" +
                "      \"openclMinimumEstimatedWork\": " + accelerator.opencl().minimumEstimatedWork() + ",\n" +
                "      \"metalEnabled\": " + accelerator.metal().enabled() + ",\n" +
                "      \"metalRequireRuntimeAvailability\": " + accelerator.metal().requireRuntimeAvailability() + ",\n" +
                "      \"metalMinimumEstimatedWork\": " + accelerator.metal().minimumEstimatedWork() + "\n" +
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

    private static long findLong(String json, String key, long defaultValue) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?\\d+)").matcher(json);
        return matcher.find() ? Long.parseLong(matcher.group(1)) : defaultValue;
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
