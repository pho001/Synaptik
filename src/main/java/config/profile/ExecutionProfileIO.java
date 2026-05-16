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
import config.compile.CompileConfig;
import config.compile.BackendDiscoveryMode;
import config.compile.BackendPlanningConfig;
import config.compile.BackendPlanningCostConfig;
import config.compile.BackendPlanningFailurePolicy;
import config.compile.BackendPlanningRequirementScope;
import config.compile.BackendTarget;
import config.compile.GraphOptimizationConfig;
import config.compile.MemoryPlanningConfig;
import config.compile.PartitionScoreWeights;
import config.compile.PartitionSearchConfig;
import config.compile.PlanningCostProfile;
import config.compile.RegionOptimizationConfig;
import config.compile.RegionOwnershipPlannerStrategy;
import config.optimizer.AlgebraicRewriteConfig;
import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import config.optimizer.CpuFusionCheapProducerPolicy;
import config.optimizer.CpuFusionConfig;
import config.optimizer.CpuFusionFanoutPolicy;
import config.optimizer.CpuFusionLayoutPolicy;
import config.optimizer.CpuFusionMode;
import config.optimizer.CpuRegionBoundaryPolicy;
import config.optimizer.CpuRegionConfig;
import config.optimizer.CpuRegionFanoutPolicy;
import config.optimizer.CpuRegionPolicy;
import config.optimizer.CseConfig;
import config.optimizer.FuseConfig;
import config.optimizer.LinearLoweringConfig;
import config.optimizer.MemoryConfig;
import config.optimizer.MetalTransferModel;
import config.optimizer.PiecewiseLoweringConfig;
import config.optimizer.RewriteConfig;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.AcceleratorConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.BlasStorageMode;
import config.runtime.Conv2dConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.FusedPrimaryBackend;
import config.runtime.RuntimeConfig;
import tensor.DataType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
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

    public static ExecutionProfile loadExecutionProfileStrict(Path path, ExecutionProfile defaultProfile) {
        if (path == null) {
            throw new IllegalArgumentException("path cannot be null");
        }
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("Execution profile does not exist: " + path);
        }
        try {
            return fromJsonStrict(Files.readString(path, StandardCharsets.UTF_8), defaultProfile);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read execution profile: " + path, e);
        }
    }

    public static ExecutionProfile fromJsonOrDefault(String json, ExecutionProfile defaultProfile) {
        if (json == null || json.isBlank()) {
            return defaultProfile;
        }
        try {
            return fromJsonStrict(json, defaultProfile);
        } catch (Exception e) {
            return defaultProfile;
        }
    }

    public static ExecutionProfile fromJsonStrict(String json, ExecutionProfile defaultProfile) {
            if (json == null || json.isBlank()) {
                throw new IllegalArgumentException("Execution profile JSON cannot be blank");
            }
            if (defaultProfile == null) {
                throw new IllegalArgumentException("defaultProfile cannot be null");
            }
            if (json.contains("\"optimizer\"")) {
                throw new IllegalArgumentException("Execution profile schema v2 rejects legacy optimizer blocks");
            }
            if (!json.contains("\"compile\"")) {
                throw new IllegalArgumentException("Execution profile schema v2 requires a compile block");
            }
            DataType dataType = findEnum(json, "dataType", defaultProfile.dataType(), DataType.class);
            ExecutionMode mode = findEnum(json, "mode", defaultProfile.mode(), ExecutionMode.class);
            String profileName = findString(json, "profileName", defaultProfile.profileName());
            String candidateName = findString(json, "candidateName", defaultProfile.candidateName());

            var defaultCompile = defaultProfile.compile();
            RewriteConfig defaultRewrite = defaultCompile.graphOptimization().rewrite();
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
                    defaultCompile.graphOptimization().cse().strictSafety()
            );
            FuseConfig defaultFuse = defaultCompile.regionOptimization().fuse();
            FuseConfig fuse = new FuseConfig(
                    findInt(json, "maxClusterNodes", defaultFuse.maxClusterNodes()),
                    findDouble(json, "scoreThreshold", defaultFuse.scoreThreshold()),
                    findDouble(json, "internalEdgeBonus", defaultFuse.internalEdgeBonus()),
                    findDouble(json, "externalInputPenalty", defaultFuse.externalInputPenalty()),
                    findDouble(json, "sharedExpensivePenalty", defaultFuse.sharedExpensivePenalty()),
                    findDouble(json, "nonCheapBonus", defaultFuse.nonCheapBonus()),
                    findBoolean(json, "preserveSharedExpensiveNodes", defaultFuse.preserveSharedExpensiveNodes())
            );
            MemoryConfig defaultMemory = defaultCompile.memoryPlanning().memory();
            MemoryConfig memory = new MemoryConfig(
                    findBoolean(json, "separateForwardBackwardPools", defaultMemory.separateForwardBackwardPools()),
                    findBoolean(json, "allowCrossPhaseReuse", defaultMemory.allowCrossPhaseReuse()),
                    findBoolean(json, "allowLargerBufferReuse", defaultMemory.allowLargerBufferReuse()),
                    findInt(json, "minReusableBufferSize", defaultMemory.minReusableBufferSize())
            );
            PartitionSearchConfig defaultSearch = defaultCompile.backendPlanning().search();
            PartitionScoreWeights defaultWeights = defaultSearch.scoreWeights();
            PartitionSearchConfig search = new PartitionSearchConfig(
                    findInt(json, "partitionMaxSearchNodes", defaultSearch.maxSearchNodes()),
                    findInt(json, "partitionMaxVisitedCandidates", defaultSearch.maxVisitedCandidates()),
                    new PartitionScoreWeights(
                            findDouble(json, "partitionNodeWeight", defaultWeights.nodeWeight()),
                            findDouble(json, "partitionInternalEdgeWeight", defaultWeights.internalEdgeWeight()),
                            findDouble(json, "partitionMergeNodeBonus", defaultWeights.mergeNodeBonus()),
                            findDouble(json, "partitionTailDepthWeight", defaultWeights.tailDepthWeight()),
                            findDouble(json, "partitionExternalInputPenalty", defaultWeights.externalInputPenalty()),
                            findDouble(json, "partitionWorkWeight", defaultWeights.workWeight())
                    )
            );
            MetalTransferModel metalTransferModel = findEnum(
                    json,
                    "partitionMetalTransferModel",
                    defaultCompile.backendPlanning().cost().planningCostProfile().metalTransferModel(),
                    MetalTransferModel.class
            );
            CpuRegionConfig defaultCpuRegion = defaultCompile.backendPlanning().cpuRegions();
            CpuRegionConfig cpuRegion = new CpuRegionConfig(
                    findEnum(json, "cpuRegionPolicy", defaultCpuRegion.policy(), CpuRegionPolicy.class),
                    findInt(json, "cpuRegionMaxRegionNodes", defaultCpuRegion.maxRegionNodes()),
                    findEnum(json, "cpuRegionFanoutPolicy", defaultCpuRegion.fanoutPolicy(), CpuRegionFanoutPolicy.class),
                    findEnum(json, "cpuRegionBoundaryPolicy", defaultCpuRegion.boundaryPolicy(), CpuRegionBoundaryPolicy.class)
            );
            CpuFusionConfig defaultCpuFusion = defaultCompile.regionOptimization().cpuFusion();
            CpuFusionConfig cpuFusion = new CpuFusionConfig(
                    findEnum(json, "cpuFusionMode", defaultCpuFusion.mode(), CpuFusionMode.class),
                    findInt(json, "cpuFusionMaxChainNodes", defaultCpuFusion.maxChainNodes()),
                    findEnum(json, "cpuFusionFanoutPolicy", defaultCpuFusion.fanoutPolicy(), CpuFusionFanoutPolicy.class),
                    findEnum(json, "cpuFusionLayoutPolicy", defaultCpuFusion.layoutPolicy(), CpuFusionLayoutPolicy.class),
                    findEnum(
                            json,
                            "cpuFusionCheapProducerPolicy",
                            defaultCpuFusion.cheapProducerPolicy(),
                            CpuFusionCheapProducerPolicy.class
                    )
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
                    findEnum(json, "blasStorageMode", defaultProfile.runtime().blas().storageMode(), BlasStorageMode.class),
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
                            ),
                            acceleratorBufferConfig(json, "cuda", defaultProfile.runtime().accelerator().cuda().buffer())
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
                            ),
                            acceleratorBufferConfig(json, "opencl", defaultProfile.runtime().accelerator().opencl().buffer())
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
                            ),
                            acceleratorBufferConfig(json, "metal", defaultProfile.runtime().accelerator().metal().buffer())
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

            GraphOptimizationConfig graphOptimization = new GraphOptimizationConfig(
                    findBoolean(json, "algebraicRewrite", defaultCompile.graphOptimization().algebraicRewrite()),
                    findBoolean(json, "constantFolding", defaultCompile.graphOptimization().constantFolding()),
                    findBoolean(json, "commonSubexpressionElimination", defaultCompile.graphOptimization().commonSubexpressionElimination()),
                    findBoolean(json, "deadCodeElimination", defaultCompile.graphOptimization().deadCodeElimination()),
                    findBoolean(json, "optionalLowering", defaultCompile.graphOptimization().optionalLowering()),
                    rewrite,
                    strictSafety ? CseConfig.strictDefaults() : CseConfig.aggressiveDefaults()
            );
            BackendDiscoveryMode discoveryMode = findEnum(
                    json,
                    "backendDiscoveryMode",
                    defaultCompile.backendPlanning().discoveryMode(),
                    BackendDiscoveryMode.class
            );
            BackendPlanningConfig backendPlanning = switch (discoveryMode) {
                case CPU_ONLY -> BackendPlanningConfig.cpuOnly();
                case EXPLICIT -> BackendPlanningConfig.explicitOnly();
                case AUTO -> BackendPlanningConfig.autoAccelerator();
            };
            backendPlanning = backendPlanning
                    .withTargets(findEnumSet(
                            json,
                            "backendTargets",
                            defaultCompile.backendPlanning().targets(),
                            BackendTarget.class
                    ))
                    .withFailurePolicy(
                            findEnum(
                                    json,
                                    "backendFailurePolicy",
                                    defaultCompile.backendPlanning().failurePolicy(),
                                    BackendPlanningFailurePolicy.class
                            ),
                            findEnum(
                                    json,
                                    "backendRequirementScope",
                                    defaultCompile.backendPlanning().requirementScope(),
                                    BackendPlanningRequirementScope.class
                            )
                    )
                    .withOwnershipPlanner(findEnum(
                            json,
                            "ownershipPlanner",
                            defaultCompile.backendPlanning().ownershipPlanner(),
                            RegionOwnershipPlannerStrategy.class
                    ))
                    .withSearch(search)
                    .withCpuRegions(cpuRegion)
                    .withCost(new BackendPlanningCostConfig(new PlanningCostProfile(metalTransferModel)));
            CompileConfig compile = new CompileConfig(
                    defaultCompile.semanticCanonicalization(),
                    graphOptimization,
                    backendPlanning,
                    new RegionOptimizationConfig(
                            findBoolean(json, "regionOptimizationEnabled", defaultCompile.regionOptimization().enabled()),
                            fuse,
                            cpuFusion
                    ),
                    new MemoryPlanningConfig(
                            findBoolean(json, "memoryPlanningEnabled", defaultCompile.memoryPlanning().enabled()),
                            memory
                    )
            );
            return new ExecutionProfile(profileName, candidateName, dataType, mode, compile, runtime, workload);
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
        var compile = profile.compile();
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
                "  \"compile\": {\n" +
                "    \"semanticCanonicalization\": {\n" +
                "      \"enabled\": " + compile.semanticCanonicalization().enabled() + "\n" +
                "    },\n" +
                "    \"graphOptimization\": {\n" +
                "      \"algebraicRewrite\": " + compile.graphOptimization().algebraicRewrite() + ",\n" +
                "      \"constantFolding\": " + compile.graphOptimization().constantFolding() + ",\n" +
                "      \"commonSubexpressionElimination\": " + compile.graphOptimization().commonSubexpressionElimination() + ",\n" +
                "      \"deadCodeElimination\": " + compile.graphOptimization().deadCodeElimination() + ",\n" +
                "      \"optionalLowering\": " + compile.graphOptimization().optionalLowering() + ",\n" +
                "      \"algebraicEnabled\": " + compile.graphOptimization().rewrite().algebraic().enabled() + ",\n" +
                "      \"linearLoweringEnabled\": " + compile.graphOptimization().rewrite().linearLowering().enabled() + ",\n" +
                "      \"conv2dLoweringMode\": \"" + compile.graphOptimization().rewrite().conv2dLowering().mode().name() + "\",\n" +
                "      \"piecewiseLowering\": {\n" +
                "        \"canonicalSigmoid\": " + compile.graphOptimization().rewrite().piecewiseLowering().canonicalSigmoid() + ",\n" +
                "        \"reluLikeWhere\": " + compile.graphOptimization().rewrite().piecewiseLowering().reluLikeWhere() + ",\n" +
                "        \"clampLikeWhere\": " + compile.graphOptimization().rewrite().piecewiseLowering().clampLikeWhere() + "\n" +
                "      }\n" +
                "    },\n" +
                "    \"cse\": {\n" +
                "      \"strictSafety\": " + compile.graphOptimization().cse().strictSafety() + "\n" +
                "    },\n" +
                "    \"regionOptimization\": {\n" +
                "      \"regionOptimizationEnabled\": " + compile.regionOptimization().enabled() + ",\n" +
                "      \"maxClusterNodes\": " + compile.regionOptimization().fuse().maxClusterNodes() + ",\n" +
                "      \"scoreThreshold\": " + compile.regionOptimization().fuse().scoreThreshold() + ",\n" +
                "      \"internalEdgeBonus\": " + compile.regionOptimization().fuse().internalEdgeBonus() + ",\n" +
                "      \"externalInputPenalty\": " + compile.regionOptimization().fuse().externalInputPenalty() + ",\n" +
                "      \"sharedExpensivePenalty\": " + compile.regionOptimization().fuse().sharedExpensivePenalty() + ",\n" +
                "      \"nonCheapBonus\": " + compile.regionOptimization().fuse().nonCheapBonus() + ",\n" +
                "      \"preserveSharedExpensiveNodes\": " + compile.regionOptimization().fuse().preserveSharedExpensiveNodes() + "\n" +
                "    },\n" +
                "    \"memoryPlanning\": {\n" +
                "      \"memoryPlanningEnabled\": " + compile.memoryPlanning().enabled() + ",\n" +
                "      \"separateForwardBackwardPools\": " + compile.memoryPlanning().memory().separateForwardBackwardPools() + ",\n" +
                "      \"allowCrossPhaseReuse\": " + compile.memoryPlanning().memory().allowCrossPhaseReuse() + ",\n" +
                "      \"allowLargerBufferReuse\": " + compile.memoryPlanning().memory().allowLargerBufferReuse() + ",\n" +
                "      \"minReusableBufferSize\": " + compile.memoryPlanning().memory().minReusableBufferSize() + "\n" +
                "    },\n" +
                "    \"backendPlanning\": {\n" +
                "      \"backendDiscoveryMode\": \"" + compile.backendPlanning().discoveryMode().name() + "\",\n" +
                "      \"backendFailurePolicy\": \"" + compile.backendPlanning().failurePolicy().name() + "\",\n" +
                "      \"backendRequirementScope\": \"" + compile.backendPlanning().requirementScope().name() + "\",\n" +
                "      \"ownershipPlanner\": \"" + compile.backendPlanning().ownershipPlanner().name() + "\",\n" +
                "      \"backendTargets\": " + jsonStringArray(compile.backendPlanning().targets().stream().map(Enum::name).toList()) + ",\n" +
                "      \"partitionMaxSearchNodes\": " + compile.backendPlanning().search().maxSearchNodes() + ",\n" +
                "      \"partitionMaxVisitedCandidates\": " + compile.backendPlanning().search().maxVisitedCandidates() + ",\n" +
                "      \"partitionNodeWeight\": " + compile.backendPlanning().search().scoreWeights().nodeWeight() + ",\n" +
                "      \"partitionInternalEdgeWeight\": " + compile.backendPlanning().search().scoreWeights().internalEdgeWeight() + ",\n" +
                "      \"partitionMergeNodeBonus\": " + compile.backendPlanning().search().scoreWeights().mergeNodeBonus() + ",\n" +
                "      \"partitionTailDepthWeight\": " + compile.backendPlanning().search().scoreWeights().tailDepthWeight() + ",\n" +
                "      \"partitionExternalInputPenalty\": " + compile.backendPlanning().search().scoreWeights().externalInputPenalty() + ",\n" +
                "      \"partitionWorkWeight\": " + compile.backendPlanning().search().scoreWeights().workWeight() + ",\n" +
                "      \"partitionMetalTransferModel\": \"" + compile.backendPlanning().cost().planningCostProfile().metalTransferModel().name() + "\",\n" +
                "      \"cpuRegionPolicy\": \"" + compile.backendPlanning().cpuRegions().policy().name() + "\",\n" +
                "      \"cpuRegionMaxRegionNodes\": " + compile.backendPlanning().cpuRegions().maxRegionNodes() + ",\n" +
                "      \"cpuRegionFanoutPolicy\": \"" + compile.backendPlanning().cpuRegions().fanoutPolicy().name() + "\",\n" +
                "      \"cpuRegionBoundaryPolicy\": \"" + compile.backendPlanning().cpuRegions().boundaryPolicy().name() + "\"\n" +
                "    },\n" +
                "    \"cpuFusion\": {\n" +
                "      \"cpuFusionMode\": \"" + compile.regionOptimization().cpuFusion().mode().name() + "\",\n" +
                "      \"cpuFusionMaxChainNodes\": " + compile.regionOptimization().cpuFusion().maxChainNodes() + ",\n" +
                "      \"cpuFusionFanoutPolicy\": \"" + compile.regionOptimization().cpuFusion().fanoutPolicy().name() + "\",\n" +
                "      \"cpuFusionLayoutPolicy\": \"" + compile.regionOptimization().cpuFusion().layoutPolicy().name() + "\",\n" +
                "      \"cpuFusionCheapProducerPolicy\": \"" + compile.regionOptimization().cpuFusion().cheapProducerPolicy().name() + "\"\n" +
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
              "      \"blasStorageMode\": \"" + blas.storageMode().name() + "\",\n" +
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
                "      \"cudaBufferBindingMode\": \"" + accelerator.cuda().buffer().bindingMode().name() + "\",\n" +
                "      \"cudaAllowPreparedInputMaterialization\": " + accelerator.cuda().buffer().allowPreparedInputMaterialization() + ",\n" +
                "      \"cudaBufferMinimumEstimatedWork\": " + accelerator.cuda().buffer().minimumEstimatedWork() + ",\n" +
                "      \"openclEnabled\": " + accelerator.opencl().enabled() + ",\n" +
                "      \"openclRequireRuntimeAvailability\": " + accelerator.opencl().requireRuntimeAvailability() + ",\n" +
                "      \"openclMinimumEstimatedWork\": " + accelerator.opencl().minimumEstimatedWork() + ",\n" +
                "      \"openclBufferBindingMode\": \"" + accelerator.opencl().buffer().bindingMode().name() + "\",\n" +
                "      \"openclAllowPreparedInputMaterialization\": " + accelerator.opencl().buffer().allowPreparedInputMaterialization() + ",\n" +
                "      \"openclBufferMinimumEstimatedWork\": " + accelerator.opencl().buffer().minimumEstimatedWork() + ",\n" +
                "      \"metalEnabled\": " + accelerator.metal().enabled() + ",\n" +
                "      \"metalRequireRuntimeAvailability\": " + accelerator.metal().requireRuntimeAvailability() + ",\n" +
                "      \"metalMinimumEstimatedWork\": " + accelerator.metal().minimumEstimatedWork() + ",\n" +
                "      \"metalBufferBindingMode\": \"" + accelerator.metal().buffer().bindingMode().name() + "\",\n" +
                "      \"metalAllowPreparedInputMaterialization\": " + accelerator.metal().buffer().allowPreparedInputMaterialization() + ",\n" +
                "      \"metalBufferMinimumEstimatedWork\": " + accelerator.metal().buffer().minimumEstimatedWork() + "\n" +
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

    private static String jsonStringArray(List<String> values) {
        if (values == null || values.isEmpty()) {
            return "[]";
        }
        return "[" + String.join(", ", values.stream().map(value -> "\"" + escapeJson(value) + "\"").toList()) + "]";
    }

    private static String escapeJson(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static AcceleratorBufferConfig acceleratorBufferConfig(
            String json,
            String backendPrefix,
            AcceleratorBufferConfig defaultConfig
    ) {
        AcceleratorBufferConfig fallback = defaultConfig == null ? AcceleratorBufferConfig.defaults() : defaultConfig;
        String prefix = backendPrefix == null || backendPrefix.isBlank() ? "" : backendPrefix;
        String normalizedPrefix = prefix.isBlank()
                ? ""
                : prefix.substring(0, 1).toLowerCase(java.util.Locale.ROOT) + prefix.substring(1);
        String keyPrefix = normalizedPrefix;
        return new AcceleratorBufferConfig(
                findEnum(
                        json,
                        keyPrefix + "BufferBindingMode",
                        fallback.bindingMode(),
                        AcceleratorBufferBindingMode.class
                ),
                findBoolean(
                        json,
                        keyPrefix + "AllowPreparedInputMaterialization",
                        fallback.allowPreparedInputMaterialization()
                ),
                findLong(
                        json,
                        keyPrefix + "BufferMinimumEstimatedWork",
                        fallback.minimumEstimatedWork()
                )
        );
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

    private static <E extends Enum<E>> Set<E> findEnumSet(
            String json,
            String key,
            Set<E> defaultValue,
            Class<E> enumClass
    ) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\\[(.*?)]", Pattern.DOTALL)
                .matcher(json);
        if (!matcher.find()) {
            return defaultValue == null ? Set.of() : Set.copyOf(defaultValue);
        }
        EnumSet<E> values = EnumSet.noneOf(enumClass);
        Matcher valueMatcher = Pattern.compile("\"([A-Z0-9_]+)\"").matcher(matcher.group(1));
        while (valueMatcher.find()) {
            values.add(Enum.valueOf(enumClass, valueMatcher.group(1)));
        }
        return values.isEmpty() ? Set.of() : Set.copyOf(values);
    }
}
