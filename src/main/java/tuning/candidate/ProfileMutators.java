package tuning.candidate;

import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import config.optimizer.OptimizerStage;
import config.optimizer.RewriteConfig;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuKernelConfig;
import config.profile.ExecutionProfile;
import config.runtime.BlasConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.FusedPrimaryBackend;
import backend.blas.BlasProvider;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadSpec;

import java.util.ArrayList;
import java.util.List;

public final class ProfileMutators {
    private ProfileMutators() {
    }

    public static List<ExecutionProfileMutator> conv2dWorkloadMutators() {
        return List.of(
                conv2dLoweringModes(List.of(
                        Conv2dLoweringMode.HEURISTIC,
                        Conv2dLoweringMode.OFF,
                        Conv2dLoweringMode.ALWAYS
                )),
                matmulBlasProviders(List.of(BlasProvider.NONE, BlasProvider.OPENBLAS_FFM), List.of(1_000_000L, 2_000_000L)),
                blasThreads(List.of(0, 1, 2, 4)),
                fusedExecutionPolicies(List.of(FusedPrimaryBackend.ASM), List.of(true))
        );
    }

    public static List<ExecutionProfileMutator> matmulWorkloadMutators() {
        return List.of(
                matmulBlasProviders(List.of(BlasProvider.NONE, BlasProvider.OPENBLAS_FFM), List.of(1_000_000L, 2_000_000L, 4_000_000L)),
                blasThreads(List.of(0, 1, 2, 4)),
                matmulParallelThresholds(List.of(100_000, 500_000, 2_000_000)),
                fusedExecutionPolicies(List.of(FusedPrimaryBackend.ASM), List.of(true))
        );
    }

    public static ExecutionProfileMutator stageOrders(List<List<OptimizerStage>> stageOrders) {
        List<List<OptimizerStage>> safeStageOrders = stageOrders == null
                ? List.of()
                : stageOrders.stream().map(List::copyOf).toList();
        return (baseProfile, workload) -> {
            if (safeStageOrders.isEmpty()) {
                return List.of(new ExecutionProfileVariant(
                        "stageOrder=" + formatStageOrder(baseProfile.optimizer().stageOrder()),
                        baseProfile
                ));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>(safeStageOrders.size());
            for (List<OptimizerStage> stageOrder : safeStageOrders) {
                variants.add(new ExecutionProfileVariant(
                        "stageOrder=" + formatStageOrder(stageOrder),
                        new ExecutionProfile(
                                baseProfile.profileName(),
                                baseProfile.candidateName(),
                                baseProfile.dataType(),
                                baseProfile.mode(),
                                baseProfile.optimizer().withStageOrder(stageOrder),
                                baseProfile.runtime(),
                                baseProfile.workload()
                        )
                ));
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator fullStageOrderSpace() {
        return stageOrders(allStageOrders());
    }

    public static ExecutionProfileMutator constrainedStageOrderSpace() {
        return stageOrders(allConstrainedStageOrders());
    }

    public static List<List<OptimizerStage>> allStageOrders() {
        return allStageOrders(List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.FUSE, OptimizerStage.MEM), true);
    }

    public static List<List<OptimizerStage>> allNonEmptyStageOrders() {
        return allStageOrders(List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.FUSE, OptimizerStage.MEM), false);
    }

    public static List<List<OptimizerStage>> allConstrainedStageOrders() {
        List<List<OptimizerStage>> unconstrained = allStageOrders();
        List<List<OptimizerStage>> out = new ArrayList<>();
        for (List<OptimizerStage> stageOrder : unconstrained) {
            if (isConstrainedStageOrder(stageOrder)) {
                out.add(stageOrder);
            }
        }
        return List.copyOf(out);
    }

    public static List<List<OptimizerStage>> allStageOrders(List<OptimizerStage> stages, boolean includeEmpty) {
        List<OptimizerStage> safeStages = stages == null ? List.of() : List.copyOf(stages);
        List<List<OptimizerStage>> out = new ArrayList<>();
        if (includeEmpty) {
            out.add(List.of());
        }
        for (int length = 1; length <= safeStages.size(); length++) {
            enumerateStageOrders(safeStages, length, new boolean[safeStages.size()], new ArrayList<>(), out);
        }
        return List.copyOf(out);
    }

    public static List<ExecutionProfileMutator> transformerHotPathMutators() {
        return List.of(
                attentionMatMulPolicies(List.of(
                        AttentionMatMulPolicy.AUTO,
                        AttentionMatMulPolicy.FORCE_OFF,
                        AttentionMatMulPolicy.FORCE_ON
                )),
                matmulBlasProviders(List.of(BlasProvider.NONE, BlasProvider.OPENBLAS_FFM), List.of(1_000_000L, 2_000_000L)),
                blasThreads(List.of(0, 1, 2, 4)),
                vectorThresholds(
                        List.of(256, 1_024, 4_096),
                        List.of(512, 2_048, 8_192),
                        List.of(1_024, 4_096, 16_384)
                ),
                fusedExecutionPolicies(List.of(FusedPrimaryBackend.ASM), List.of(true))
        );
    }

    public static List<ExecutionProfileMutator> mlpWorkloadMutators() {
        return List.of(
                matmulBlasProviders(List.of(BlasProvider.NONE, BlasProvider.OPENBLAS_FFM), List.of(1_000_000L, 2_000_000L, 4_000_000L)),
                blasThreads(List.of(0, 1, 2, 4)),
                matmulParallelThresholds(List.of(100_000, 500_000, 2_000_000)),
                fusedExecutionPolicies(List.of(FusedPrimaryBackend.ASM), List.of(true))
        );
    }

    public static List<ExecutionProfileMutator> normalizationWorkloadMutators() {
        return List.of(
                vectorThresholds(
                        List.of(128, 512, 2_048),
                        List.of(256, 1_024, 4_096),
                        List.of(512, 2_048, 8_192)
                ),
                fusedExecutionPolicies(List.of(FusedPrimaryBackend.ASM), List.of(true))
        );
    }

    public static List<ExecutionProfileMutator> lossWorkloadMutators() {
        return List.of(
                vectorThresholds(
                        List.of(128, 512, 2_048),
                        List.of(256, 1_024, 4_096),
                        List.of(512, 2_048, 8_192)
                ),
                fusedExecutionPolicies(List.of(FusedPrimaryBackend.ASM), List.of(true))
        );
    }

    public static List<ExecutionProfileMutator> genericWorkloadMutators() {
        return List.of(
                vectorThresholds(
                        List.of(128, 512, 2_048),
                        List.of(256, 1_024, 4_096),
                        List.of(512, 2_048, 8_192)
                ),
                fusedExecutionPolicies(List.of(FusedPrimaryBackend.ASM), List.of(true))
        );
    }

    public static ExecutionProfileMutator advancedSchedulerPolicies(
            List<Integer> lowCostChunksPerWorker,
            List<Integer> mediumCostChunksPerWorker,
            List<Integer> highCostChunksPerWorker,
            List<Integer> minScalarChunkSizes,
            List<Integer> minVectorChunkSizes,
            List<Integer> minReductionChunkSizes,
            List<Integer> commonPoolLowCostMaxWorkPerWorkerValues
    ) {
        List<Integer> safeLow = lowCostChunksPerWorker == null ? List.of() : List.copyOf(lowCostChunksPerWorker);
        List<Integer> safeMedium = mediumCostChunksPerWorker == null ? List.of() : List.copyOf(mediumCostChunksPerWorker);
        List<Integer> safeHigh = highCostChunksPerWorker == null ? List.of() : List.copyOf(highCostChunksPerWorker);
        List<Integer> safeMinScalar = minScalarChunkSizes == null ? List.of() : List.copyOf(minScalarChunkSizes);
        List<Integer> safeMinVector = minVectorChunkSizes == null ? List.of() : List.copyOf(minVectorChunkSizes);
        List<Integer> safeMinReduction = minReductionChunkSizes == null ? List.of() : List.copyOf(minReductionChunkSizes);
        List<Integer> safeCommonPool = commonPoolLowCostMaxWorkPerWorkerValues == null ? List.of() : List.copyOf(commonPoolLowCostMaxWorkPerWorkerValues);
        return (baseProfile, workload) -> {
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (Integer low : safeLow) {
                for (Integer medium : safeMedium) {
                    for (Integer high : safeHigh) {
                        for (Integer minScalar : safeMinScalar) {
                            for (Integer minVector : safeMinVector) {
                                for (Integer minReduction : safeMinReduction) {
                                    for (Integer commonPool : safeCommonPool) {
                                        CpuKernelConfig baseCpu = baseProfile.runtime().kernel().cpu();
                                        CpuKernelConfig tunedCpu = new CpuKernelConfig(
                                                baseCpu.loopUnrollFactor(),
                                                baseCpu.matMulTileM(),
                                                baseCpu.matMulTileN(),
                                                baseCpu.matMulTileK(),
                                                baseCpu.cheapVectorMinSize(),
                                                baseCpu.transcendentalVectorMinSize(),
                                                baseCpu.reductionVectorMinSize(),
                                                baseCpu.cheapParallelMinSize(),
                                                baseCpu.transcendentalParallelMinSize(),
                                                baseCpu.reductionParallelMinSize(),
                                                baseCpu.contiguousMaterializeThreshold(),
                                                low,
                                                medium,
                                                high,
                                                minScalar,
                                                minVector,
                                                minReduction,
                                                commonPool,
                                                baseCpu.fusedAsmVectorWidth(),
                                                baseCpu.sumAccuracyMode(),
                                                baseCpu.matMulParallelMinSize(),
                                                baseCpu.attentionMatMulPolicy()
                                        );
                                        variants.add(new ExecutionProfileVariant(
                                                "scheduler="
                                                        + low + "/" + medium + "/" + high
                                                        + ":chunks=" + minScalar + "/" + minVector + "/" + minReduction
                                                        + ":commonPool=" + commonPool,
                                                withCpuKernelConfig(baseProfile, tunedCpu)
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator fusedExecutionPolicies(
            List<FusedPrimaryBackend> primaryBackends,
            List<Boolean> allowFallbackValues
    ) {
        List<FusedPrimaryBackend> safePrimary = primaryBackends == null ? List.of() : List.copyOf(primaryBackends);
        List<Boolean> safeFallback = allowFallbackValues == null ? List.of() : List.copyOf(allowFallbackValues);
        return (baseProfile, workload) -> {
            if (!usesFusedRuntimePolicies(workload.kind())) {
                return List.of(new ExecutionProfileVariant("fusedPolicy=current", baseProfile));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (FusedPrimaryBackend primary : safePrimary) {
                for (Boolean allowFallback : safeFallback) {
                    FusedExecutionPolicy policy = new FusedExecutionPolicy(
                            primary,
                            allowFallback
                    );
                    variants.add(new ExecutionProfileVariant(
                            "fused=" + primary.name()
                                    + ":fallback=" + allowFallback,
                            withRuntime(baseProfile, new config.runtime.RuntimeConfig(
                                    baseProfile.runtime().kernel(),
                                    baseProfile.runtime().approximation(),
                                    baseProfile.runtime().blas(),
                                    policy
                            ))
                    ));
                }
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator conv2dLoweringModes(List<Conv2dLoweringMode> modes) {
        List<Conv2dLoweringMode> safeModes = modes == null ? List.of() : List.copyOf(modes);
        return (baseProfile, workload) -> {
            if (workload.kind() != WorkloadKind.CONV2D) {
                return List.of(new ExecutionProfileVariant("conv2dLowering=" + baseProfile.optimizer().rewrite().conv2dLowering().mode().name(), baseProfile));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (Conv2dLoweringMode mode : safeModes) {
                ExecutionProfile profile = new ExecutionProfile(
                        baseProfile.profileName(),
                        baseProfile.candidateName(),
                        baseProfile.dataType(),
                        baseProfile.mode(),
                        baseProfile.optimizer().withRewrite(
                                new RewriteConfig(new Conv2dLoweringConfig(mode))
                        ),
                        baseProfile.runtime(),
                        baseProfile.workload()
                );
                variants.add(new ExecutionProfileVariant("conv2dLowering=" + mode.name(), profile));
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator blasThreads(List<Integer> threadCounts) {
        List<Integer> safeCounts = threadCounts == null ? List.of() : List.copyOf(threadCounts);
        return (baseProfile, workload) -> {
            if (!usesMatmulRuntimePolicies(workload.kind())) {
                return List.of(new ExecutionProfileVariant("blasThreads=" + formatBlasThreads(baseProfile.runtime().blas().threads()), baseProfile));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (Integer threads : safeCounts) {
                BlasConfig cfg = new BlasConfig(
                        baseProfile.runtime().blas().provider(),
                        baseProfile.runtime().blas().matmulMinWork(),
                        baseProfile.runtime().blas().f32RequireMgeK(),
                        baseProfile.runtime().blas().f32MaxNOverK(),
                        baseProfile.runtime().blas().debug(),
                        threads == null ? 0 : threads
                );
                variants.add(new ExecutionProfileVariant(
                        "blasThreads=" + formatBlasThreads(cfg.threads()),
                        new ExecutionProfile(
                                baseProfile.profileName(),
                                baseProfile.candidateName(),
                                baseProfile.dataType(),
                                baseProfile.mode(),
                                baseProfile.optimizer(),
                                new config.runtime.RuntimeConfig(
                                        baseProfile.runtime().kernel(),
                                        baseProfile.runtime().approximation(),
                                        cfg,
                                        baseProfile.runtime().fused()
                                ),
                                baseProfile.workload()
                        )
                ));
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator matmulBlasProviders(List<BlasProvider> providers, List<Long> minWorks) {
        List<BlasProvider> safeProviders = providers == null ? List.of() : List.copyOf(providers);
        List<Long> safeMinWorks = minWorks == null ? List.of() : List.copyOf(minWorks);
        return (baseProfile, workload) -> {
            if (!usesMatmulRuntimePolicies(workload.kind())) {
                return List.of(new ExecutionProfileVariant("blasProvider=" + baseProfile.runtime().blas().provider().name(), baseProfile));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (BlasProvider provider : safeProviders) {
                if (provider == BlasProvider.NONE) {
                    variants.add(new ExecutionProfileVariant(
                            "blasProvider=NONE",
                            withBlas(baseProfile, BlasConfig.disabled())
                    ));
                    continue;
                }
                for (Long minWork : safeMinWorks) {
                    BlasConfig cfg = new BlasConfig(
                            provider,
                            minWork == null ? baseProfile.runtime().blas().matmulMinWork() : minWork,
                            baseProfile.runtime().blas().f32RequireMgeK(),
                            baseProfile.runtime().blas().f32MaxNOverK(),
                            baseProfile.runtime().blas().debug(),
                            baseProfile.runtime().blas().threads()
                    );
                    variants.add(new ExecutionProfileVariant(
                            "blasProvider=" + provider.name() + ":minWork=" + cfg.matmulMinWork(),
                            withBlas(baseProfile, cfg)
                    ));
                }
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator matmulBlasShapeHeuristics(
            List<Boolean> requireMgeKValues,
            List<Double> maxNOverKValues
    ) {
        List<Boolean> safeRequire = requireMgeKValues == null ? List.of() : List.copyOf(requireMgeKValues);
        List<Double> safeRatios = maxNOverKValues == null ? List.of() : List.copyOf(maxNOverKValues);
        return (baseProfile, workload) -> {
            if (!usesMatmulRuntimePolicies(workload.kind())) {
                return List.of(new ExecutionProfileVariant(
                        "blasShape="
                                + baseProfile.runtime().blas().f32RequireMgeK()
                                + ":" + baseProfile.runtime().blas().f32MaxNOverK(),
                        baseProfile
                ));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (Boolean requireMgeK : safeRequire) {
                for (Double maxNOverK : safeRatios) {
                    BlasConfig cfg = new BlasConfig(
                            baseProfile.runtime().blas().provider(),
                            baseProfile.runtime().blas().matmulMinWork(),
                            requireMgeK == null ? baseProfile.runtime().blas().f32RequireMgeK() : requireMgeK,
                            maxNOverK == null ? baseProfile.runtime().blas().f32MaxNOverK() : maxNOverK,
                            baseProfile.runtime().blas().debug(),
                            baseProfile.runtime().blas().threads()
                    );
                    variants.add(new ExecutionProfileVariant(
                            "blasShape=" + cfg.f32RequireMgeK() + ":" + cfg.f32MaxNOverK(),
                            withBlas(baseProfile, cfg)
                    ));
                }
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator attentionMatMulPolicies(List<AttentionMatMulPolicy> policies) {
        List<AttentionMatMulPolicy> safePolicies = policies == null ? List.of() : List.copyOf(policies);
        return (baseProfile, workload) -> {
            if (workload.kind() != WorkloadKind.TRANSFORMER_HOT_PATH) {
                return List.of(new ExecutionProfileVariant(
                        "attentionMatMul=" + baseProfile.runtime().kernel().cpu().attentionMatMulPolicy().name(),
                        baseProfile
                ));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (AttentionMatMulPolicy policy : safePolicies) {
                variants.add(new ExecutionProfileVariant(
                        "attentionMatMul=" + policy.name(),
                        withCpuKernelConfig(baseProfile, copyCpuKernelConfig(
                                baseProfile.runtime().kernel().cpu(),
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                policy,
                                null
                        ))
                ));
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator vectorThresholds(
            List<Integer> cheapThresholds,
            List<Integer> transcendentalThresholds,
            List<Integer> reductionThresholds
    ) {
        List<Integer> safeCheap = cheapThresholds == null ? List.of() : List.copyOf(cheapThresholds);
        List<Integer> safeTrans = transcendentalThresholds == null ? List.of() : List.copyOf(transcendentalThresholds);
        List<Integer> safeRed = reductionThresholds == null ? List.of() : List.copyOf(reductionThresholds);
        return (baseProfile, workload) -> {
            if (!usesVectorRuntimePolicies(workload.kind())) {
                CpuKernelConfig cpu = baseProfile.runtime().kernel().cpu();
                return List.of(new ExecutionProfileVariant(
                        "vectorThresholds="
                                + cpu.cheapVectorMinSize() + "/"
                                + cpu.transcendentalVectorMinSize() + "/"
                                + cpu.reductionVectorMinSize(),
                        baseProfile
                ));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (Integer cheap : safeCheap) {
                for (Integer trans : safeTrans) {
                    for (Integer red : safeRed) {
                        variants.add(new ExecutionProfileVariant(
                                "vectorThresholds=" + cheap + "/" + trans + "/" + red,
                                withCpuKernelConfig(baseProfile, copyCpuKernelConfig(
                                        baseProfile.runtime().kernel().cpu(),
                                        cheap,
                                        trans,
                                        null,
                                        null,
                                        red,
                                        null,
                                        null,
                                        null,
                                        null,
                                        null,
                                        null,
                                        null
                                ))
                        ));
                    }
                }
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator fusedAsmVectorWidths(List<Integer> widths) {
        List<Integer> safeWidths = widths == null ? List.of() : List.copyOf(widths);
        return (baseProfile, workload) -> {
            if (!usesFusedRuntimePolicies(workload.kind())) {
                return List.of(new ExecutionProfileVariant(
                        "fusedAsmVectorWidth=" + baseProfile.runtime().kernel().cpu().fusedAsmVectorWidth(),
                        baseProfile
                ));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (Integer width : safeWidths) {
                CpuKernelConfig baseCpu = baseProfile.runtime().kernel().cpu();
                CpuKernelConfig tunedCpu = new CpuKernelConfig(
                        baseCpu.loopUnrollFactor(),
                        baseCpu.matMulTileM(),
                        baseCpu.matMulTileN(),
                        baseCpu.matMulTileK(),
                        baseCpu.cheapVectorMinSize(),
                        baseCpu.transcendentalVectorMinSize(),
                        baseCpu.fusedCheapVectorMinSize(),
                        baseCpu.fusedTranscendentalVectorMinSize(),
                        baseCpu.reductionVectorMinSize(),
                        baseCpu.cheapParallelMinSize(),
                        baseCpu.transcendentalParallelMinSize(),
                        baseCpu.fusedCheapParallelMinSize(),
                        baseCpu.fusedTranscendentalParallelMinSize(),
                        baseCpu.reductionParallelMinSize(),
                        baseCpu.contiguousMaterializeThreshold(),
                        baseCpu.lowCostTargetChunksPerWorker(),
                        baseCpu.mediumCostTargetChunksPerWorker(),
                        baseCpu.highCostTargetChunksPerWorker(),
                        baseCpu.minScalarChunkSize(),
                        baseCpu.minVectorChunkSize(),
                        baseCpu.minReductionChunkSize(),
                        baseCpu.commonPoolLowCostMaxWorkPerWorker(),
                        width == null ? baseCpu.fusedAsmVectorWidth() : width,
                        baseCpu.sumAccuracyMode(),
                        baseCpu.matMulParallelMinSize(),
                        baseCpu.attentionMatMulPolicy()
                );
                variants.add(new ExecutionProfileVariant(
                        "fusedAsmVectorWidth=" + tunedCpu.fusedAsmVectorWidth(),
                        withCpuKernelConfig(baseProfile, tunedCpu)
                ));
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator matmulParallelThresholds(List<Integer> thresholds) {
        List<Integer> safeThresholds = thresholds == null ? List.of() : List.copyOf(thresholds);
        return (baseProfile, workload) -> {
            if (!usesMatmulRuntimePolicies(workload.kind())) {
                return List.of(new ExecutionProfileVariant(
                        "matmulParallelMin=" + baseProfile.runtime().kernel().cpu().matMulParallelMinSize(),
                        baseProfile
                ));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (Integer threshold : safeThresholds) {
                variants.add(new ExecutionProfileVariant(
                        "matmulParallelMin=" + threshold,
                        withCpuKernelConfig(baseProfile, copyCpuKernelConfig(
                                baseProfile.runtime().kernel().cpu(),
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                threshold
                        ))
                ));
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator parallelThresholds(
            List<Integer> cheapThresholds,
            List<Integer> transcendentalThresholds,
            List<Integer> reductionThresholds
    ) {
        List<Integer> safeCheap = cheapThresholds == null ? List.of() : List.copyOf(cheapThresholds);
        List<Integer> safeTrans = transcendentalThresholds == null ? List.of() : List.copyOf(transcendentalThresholds);
        List<Integer> safeRed = reductionThresholds == null ? List.of() : List.copyOf(reductionThresholds);
        return (baseProfile, workload) -> {
            if (!usesVectorRuntimePolicies(workload.kind())) {
                CpuKernelConfig cpu = baseProfile.runtime().kernel().cpu();
                return List.of(new ExecutionProfileVariant(
                        "parallelThresholds="
                                + cpu.cheapParallelMinSize() + "/"
                                + cpu.transcendentalParallelMinSize() + "/"
                                + cpu.reductionParallelMinSize(),
                        baseProfile
                ));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (Integer cheap : safeCheap) {
                for (Integer trans : safeTrans) {
                    for (Integer red : safeRed) {
                        CpuKernelConfig baseCpu = baseProfile.runtime().kernel().cpu();
                        CpuKernelConfig tunedCpu = new CpuKernelConfig(
                                baseCpu.loopUnrollFactor(),
                                baseCpu.matMulTileM(),
                                baseCpu.matMulTileN(),
                                baseCpu.matMulTileK(),
                                baseCpu.cheapVectorMinSize(),
                                baseCpu.transcendentalVectorMinSize(),
                                baseCpu.fusedCheapVectorMinSize(),
                                baseCpu.fusedTranscendentalVectorMinSize(),
                                baseCpu.reductionVectorMinSize(),
                                cheap == null ? baseCpu.cheapParallelMinSize() : cheap,
                                trans == null ? baseCpu.transcendentalParallelMinSize() : trans,
                                baseCpu.fusedCheapParallelMinSize(),
                                baseCpu.fusedTranscendentalParallelMinSize(),
                                red == null ? baseCpu.reductionParallelMinSize() : red,
                                baseCpu.contiguousMaterializeThreshold(),
                                baseCpu.lowCostTargetChunksPerWorker(),
                                baseCpu.mediumCostTargetChunksPerWorker(),
                                baseCpu.highCostTargetChunksPerWorker(),
                                baseCpu.minScalarChunkSize(),
                                baseCpu.minVectorChunkSize(),
                                baseCpu.minReductionChunkSize(),
                                baseCpu.commonPoolLowCostMaxWorkPerWorker(),
                                baseCpu.fusedAsmVectorWidth(),
                                baseCpu.sumAccuracyMode(),
                                baseCpu.matMulParallelMinSize(),
                                baseCpu.attentionMatMulPolicy()
                        );
                        variants.add(new ExecutionProfileVariant(
                                "parallelThresholds=" + tunedCpu.cheapParallelMinSize()
                                        + "/" + tunedCpu.transcendentalParallelMinSize()
                                        + "/" + tunedCpu.reductionParallelMinSize(),
                                withCpuKernelConfig(baseProfile, tunedCpu)
                        ));
                    }
                }
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator fusedDispatchThresholds(
            List<Integer> cheapVectorThresholds,
            List<Integer> transcendentalVectorThresholds,
            List<Integer> cheapParallelThresholds,
            List<Integer> transcendentalParallelThresholds
    ) {
        List<Integer> safeCheapVec = cheapVectorThresholds == null ? List.of() : List.copyOf(cheapVectorThresholds);
        List<Integer> safeTransVec = transcendentalVectorThresholds == null ? List.of() : List.copyOf(transcendentalVectorThresholds);
        List<Integer> safeCheapPar = cheapParallelThresholds == null ? List.of() : List.copyOf(cheapParallelThresholds);
        List<Integer> safeTransPar = transcendentalParallelThresholds == null ? List.of() : List.copyOf(transcendentalParallelThresholds);
        return (baseProfile, workload) -> {
            if (!usesFusedRuntimePolicies(workload.kind())) {
                CpuKernelConfig cpu = baseProfile.runtime().kernel().cpu();
                return List.of(new ExecutionProfileVariant(
                        "fusedDispatchThresholds="
                                + cpu.fusedCheapVectorMinSize() + "/"
                                + cpu.fusedTranscendentalVectorMinSize() + "/"
                                + cpu.fusedCheapParallelMinSize() + "/"
                                + cpu.fusedTranscendentalParallelMinSize(),
                        baseProfile
                ));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (Integer cheapVec : safeCheapVec) {
                for (Integer transVec : safeTransVec) {
                    for (Integer cheapPar : safeCheapPar) {
                        for (Integer transPar : safeTransPar) {
                            variants.add(new ExecutionProfileVariant(
                                    "fusedDispatchThresholds=" + cheapVec + "/" + transVec + "/" + cheapPar + "/" + transPar,
                                    withCpuKernelConfig(baseProfile, copyCpuKernelConfig(
                                            baseProfile.runtime().kernel().cpu(),
                                            null,
                                            null,
                                            cheapVec,
                                            transVec,
                                            null,
                                            null,
                                            null,
                                            cheapPar,
                                            transPar,
                                            null,
                                            null,
                                            null
                                    ))
                            ));
                        }
                    }
                }
            }
            return variants;
        };
    }

    private static boolean usesMatmulRuntimePolicies(WorkloadKind kind) {
        return kind == WorkloadKind.MATMUL
                || kind == WorkloadKind.MLP_CLASSIFICATION
                || kind == WorkloadKind.ABC_SEQUENCE_MATMUL
                || kind == WorkloadKind.CONV2D
                || kind == WorkloadKind.TRANSFORMER_HOT_PATH;
    }

    private static boolean usesVectorRuntimePolicies(WorkloadKind kind) {
        return kind == WorkloadKind.NORMALIZATION
                || kind == WorkloadKind.TRANSFORMER_HOT_PATH
                || kind == WorkloadKind.GENERIC
                || kind == WorkloadKind.LOSS;
    }

    private static boolean usesFusedRuntimePolicies(WorkloadKind kind) {
        return kind == WorkloadKind.MATMUL
                || kind == WorkloadKind.TRANSFORMER_HOT_PATH
                || kind == WorkloadKind.MLP_CLASSIFICATION
                || kind == WorkloadKind.ABC_SEQUENCE_MATMUL
                || kind == WorkloadKind.GENERIC
                || kind == WorkloadKind.NORMALIZATION
                || kind == WorkloadKind.LOSS;
    }

    private static ExecutionProfile withBlas(ExecutionProfile baseProfile, BlasConfig cfg) {
        return withRuntime(baseProfile, new config.runtime.RuntimeConfig(
                baseProfile.runtime().kernel(),
                baseProfile.runtime().approximation(),
                cfg,
                baseProfile.runtime().fused()
        ));
    }

    private static ExecutionProfile withRuntime(ExecutionProfile baseProfile, config.runtime.RuntimeConfig runtime) {
        return new ExecutionProfile(
                baseProfile.profileName(),
                baseProfile.candidateName(),
                baseProfile.dataType(),
                baseProfile.mode(),
                baseProfile.optimizer(),
                runtime,
                baseProfile.workload()
        );
    }

    private static ExecutionProfile withCpuKernelConfig(ExecutionProfile baseProfile, CpuKernelConfig cpu) {
        return new ExecutionProfile(
                baseProfile.profileName(),
                baseProfile.candidateName(),
                baseProfile.dataType(),
                baseProfile.mode(),
                baseProfile.optimizer(),
                new config.runtime.RuntimeConfig(
                        new config.backend.KernelTuningConfig(
                                cpu,
                                baseProfile.runtime().kernel().cuda(),
                                baseProfile.runtime().kernel().opencl()
                        ),
                        baseProfile.runtime().approximation(),
                        baseProfile.runtime().blas(),
                        baseProfile.runtime().fused()
                ),
                baseProfile.workload()
        );
    }

    private static CpuKernelConfig copyCpuKernelConfig(
            CpuKernelConfig base,
            Integer cheapVectorMinSize,
            Integer transcendentalVectorMinSize,
            Integer fusedCheapVectorMinSize,
            Integer fusedTranscendentalVectorMinSize,
            Integer reductionVectorMinSize,
            Integer cheapParallelMinSize,
            Integer transcendentalParallelMinSize,
            Integer fusedCheapParallelMinSize,
            Integer fusedTranscendentalParallelMinSize,
            Integer reductionParallelMinSize,
            AttentionMatMulPolicy attention,
            Integer matmulParallelMin
    ) {
        return new CpuKernelConfig(
                base.loopUnrollFactor(),
                base.matMulTileM(),
                base.matMulTileN(),
                base.matMulTileK(),
                cheapVectorMinSize == null ? base.cheapVectorMinSize() : cheapVectorMinSize,
                transcendentalVectorMinSize == null ? base.transcendentalVectorMinSize() : transcendentalVectorMinSize,
                fusedCheapVectorMinSize == null ? base.fusedCheapVectorMinSize() : fusedCheapVectorMinSize,
                fusedTranscendentalVectorMinSize == null ? base.fusedTranscendentalVectorMinSize() : fusedTranscendentalVectorMinSize,
                reductionVectorMinSize == null ? base.reductionVectorMinSize() : reductionVectorMinSize,
                cheapParallelMinSize == null ? base.cheapParallelMinSize() : cheapParallelMinSize,
                transcendentalParallelMinSize == null ? base.transcendentalParallelMinSize() : transcendentalParallelMinSize,
                fusedCheapParallelMinSize == null ? base.fusedCheapParallelMinSize() : fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize == null ? base.fusedTranscendentalParallelMinSize() : fusedTranscendentalParallelMinSize,
                reductionParallelMinSize == null ? base.reductionParallelMinSize() : reductionParallelMinSize,
                base.contiguousMaterializeThreshold(),
                base.lowCostTargetChunksPerWorker(),
                base.mediumCostTargetChunksPerWorker(),
                base.highCostTargetChunksPerWorker(),
                base.minScalarChunkSize(),
                base.minVectorChunkSize(),
                base.minReductionChunkSize(),
                base.commonPoolLowCostMaxWorkPerWorker(),
                base.fusedAsmVectorWidth(),
                base.sumAccuracyMode(),
                matmulParallelMin == null ? base.matMulParallelMinSize() : matmulParallelMin,
                attention == null ? base.attentionMatMulPolicy() : attention
        );
    }

    private static void enumerateStageOrders(
            List<OptimizerStage> stages,
            int targetLength,
            boolean[] used,
            List<OptimizerStage> current,
            List<List<OptimizerStage>> out
    ) {
        if (current.size() == targetLength) {
            out.add(List.copyOf(current));
            return;
        }
        for (int i = 0; i < stages.size(); i++) {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            current.add(stages.get(i));
            enumerateStageOrders(stages, targetLength, used, current, out);
            current.removeLast();
            used[i] = false;
        }
    }

    private static String formatStageOrder(List<OptimizerStage> stageOrder) {
        if (stageOrder == null || stageOrder.isEmpty()) {
            return "NONE";
        }
        return String.join("-", stageOrder.stream().map(Enum::name).toList());
    }

    private static String formatBlasThreads(int threads) {
        return threads <= 0 ? "AUTO" : Integer.toString(threads);
    }

    private static boolean isConstrainedStageOrder(List<OptimizerStage> stageOrder) {
        if (stageOrder == null || stageOrder.isEmpty()) {
            return true;
        }
        int memIndex = stageOrder.indexOf(OptimizerStage.MEM);
        return memIndex < 0 || memIndex == stageOrder.size() - 1;
    }
}
