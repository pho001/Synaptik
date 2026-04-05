package tuning.candidate;

import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import config.optimizer.OptimizerStage;
import config.optimizer.RewriteConfig;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuKernelConfig;
import config.backend.VectorPolicy;
import config.profile.ExecutionProfile;
import config.runtime.BlasConfig;
import backend.blas.BlasProvider;
import backend.blas.BlasThreadPolicy;
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
                matmulBlasProviders(List.of(BlasProvider.NONE, BlasProvider.OPENBLAS_FFM), List.of(1_000_000L, 2_000_000L))
        );
    }

    public static List<ExecutionProfileMutator> matmulWorkloadMutators() {
        return List.of(
                matmulBlasProviders(List.of(BlasProvider.NONE, BlasProvider.OPENBLAS_FFM), List.of(1_000_000L, 2_000_000L, 4_000_000L)),
                blasThreadPolicies(List.of(BlasThreadPolicy.AUTO, BlasThreadPolicy.FIXED), List.of(1, 4)),
                matmulParallelThresholds(List.of(100_000, 500_000, 2_000_000))
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

    public static List<List<OptimizerStage>> allStageOrders() {
        return allStageOrders(List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.FUSE, OptimizerStage.MEM), true);
    }

    public static List<List<OptimizerStage>> allNonEmptyStageOrders() {
        return allStageOrders(List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.FUSE, OptimizerStage.MEM), false);
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
                blasThreadPolicies(List.of(BlasThreadPolicy.AUTO, BlasThreadPolicy.FIXED), List.of(1, 4)),
                vectorPolicies(
                        List.of(VectorPolicy.AUTO, VectorPolicy.FORCE_ON),
                        List.of(VectorPolicy.AUTO, VectorPolicy.FORCE_OFF),
                        List.of(VectorPolicy.AUTO)
                )
        );
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

    public static ExecutionProfileMutator blasThreadPolicies(List<BlasThreadPolicy> policies, List<Integer> fixedThreadCounts) {
        List<BlasThreadPolicy> safePolicies = policies == null ? List.of() : List.copyOf(policies);
        List<Integer> safeCounts = fixedThreadCounts == null ? List.of(1) : List.copyOf(fixedThreadCounts);
        return (baseProfile, workload) -> {
            if (!usesMatmulRuntimePolicies(workload.kind())) {
                return List.of(new ExecutionProfileVariant("blasThread=" + baseProfile.runtime().blas().threadPolicy().name(), baseProfile));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (BlasThreadPolicy policy : safePolicies) {
                if (policy == BlasThreadPolicy.FIXED) {
                    for (Integer threads : safeCounts) {
                        BlasConfig cfg = new BlasConfig(
                                baseProfile.runtime().blas().provider(),
                                baseProfile.runtime().blas().matmulMinWork(),
                                baseProfile.runtime().blas().f32RequireMgeK(),
                                baseProfile.runtime().blas().f32MaxNOverK(),
                                baseProfile.runtime().blas().debug(),
                                policy,
                                threads
                        );
                        variants.add(new ExecutionProfileVariant(
                                "blasThread=" + policy.name() + ":" + threads,
                                new ExecutionProfile(
                                        baseProfile.profileName(),
                                        baseProfile.candidateName(),
                                        baseProfile.dataType(),
                                        baseProfile.mode(),
                                        baseProfile.optimizer(),
                                        new config.runtime.RuntimeConfig(
                                                baseProfile.runtime().kernel(),
                                                baseProfile.runtime().approximation(),
                                                cfg
                                        ),
                                        baseProfile.workload()
                                )
                        ));
                    }
                } else {
                    BlasConfig cfg = new BlasConfig(
                            baseProfile.runtime().blas().provider(),
                            baseProfile.runtime().blas().matmulMinWork(),
                            baseProfile.runtime().blas().f32RequireMgeK(),
                            baseProfile.runtime().blas().f32MaxNOverK(),
                            baseProfile.runtime().blas().debug(),
                            policy,
                            0
                    );
                    variants.add(new ExecutionProfileVariant(
                            "blasThread=" + policy.name(),
                            new ExecutionProfile(
                                    baseProfile.profileName(),
                                    baseProfile.candidateName(),
                                    baseProfile.dataType(),
                                    baseProfile.mode(),
                                    baseProfile.optimizer(),
                                    new config.runtime.RuntimeConfig(
                                            baseProfile.runtime().kernel(),
                                            baseProfile.runtime().approximation(),
                                            cfg
                                    ),
                                    baseProfile.workload()
                            )
                    ));
                }
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
                            baseProfile.runtime().blas().threadPolicy(),
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
                                policy,
                                null
                        ))
                ));
            }
            return variants;
        };
    }

    public static ExecutionProfileMutator vectorPolicies(
            List<VectorPolicy> cheapPolicies,
            List<VectorPolicy> transcendentalPolicies,
            List<VectorPolicy> reductionPolicies
    ) {
        List<VectorPolicy> safeCheap = cheapPolicies == null ? List.of() : List.copyOf(cheapPolicies);
        List<VectorPolicy> safeTrans = transcendentalPolicies == null ? List.of() : List.copyOf(transcendentalPolicies);
        List<VectorPolicy> safeRed = reductionPolicies == null ? List.of() : List.copyOf(reductionPolicies);
        return (baseProfile, workload) -> {
            if (!usesVectorRuntimePolicies(workload.kind())) {
                return List.of(new ExecutionProfileVariant("vectorPolicies=current", baseProfile));
            }
            List<ExecutionProfileVariant> variants = new ArrayList<>();
            for (VectorPolicy cheap : safeCheap) {
                for (VectorPolicy trans : safeTrans) {
                    for (VectorPolicy red : safeRed) {
                        variants.add(new ExecutionProfileVariant(
                                "vectorPolicies=" + cheap.name() + "/" + trans.name() + "/" + red.name(),
                                withCpuKernelConfig(baseProfile, copyCpuKernelConfig(
                                        baseProfile.runtime().kernel().cpu(),
                                        cheap,
                                        trans,
                                        red,
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
                                threshold
                        ))
                ));
            }
            return variants;
        };
    }

    private static boolean usesMatmulRuntimePolicies(WorkloadKind kind) {
        return kind == WorkloadKind.MATMUL
                || kind == WorkloadKind.MLP_CLASSIFICATION
                || kind == WorkloadKind.CONV2D
                || kind == WorkloadKind.TRANSFORMER_HOT_PATH;
    }

    private static boolean usesVectorRuntimePolicies(WorkloadKind kind) {
        return kind == WorkloadKind.NORMALIZATION
                || kind == WorkloadKind.TRANSFORMER_HOT_PATH
                || kind == WorkloadKind.GENERIC
                || kind == WorkloadKind.LOSS;
    }

    private static ExecutionProfile withBlas(ExecutionProfile baseProfile, BlasConfig cfg) {
        return new ExecutionProfile(
                baseProfile.profileName(),
                baseProfile.candidateName(),
                baseProfile.dataType(),
                baseProfile.mode(),
                baseProfile.optimizer(),
                new config.runtime.RuntimeConfig(
                        baseProfile.runtime().kernel(),
                        baseProfile.runtime().approximation(),
                        cfg
                ),
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
                        baseProfile.runtime().blas()
                ),
                baseProfile.workload()
        );
    }

    private static CpuKernelConfig copyCpuKernelConfig(
            CpuKernelConfig base,
            VectorPolicy cheap,
            VectorPolicy transcendental,
            VectorPolicy reduction,
            AttentionMatMulPolicy attention,
            Integer matmulParallelMin
    ) {
        return new CpuKernelConfig(
                base.loopUnrollFactor(),
                base.matMulTileM(),
                base.matMulTileN(),
                base.matMulTileK(),
                base.vectorMinSize(),
                base.parallelMinSize(),
                base.parallelism(),
                base.chunksPerWorker(),
                base.minChunkSize(),
                base.contiguousMaterializeThreshold(),
                base.sumAccuracyMode(),
                base.lowCostNsPerElementThreshold(),
                cheap == null ? base.vectorPolicyCheap() : cheap,
                transcendental == null ? base.vectorPolicyTranscendental() : transcendental,
                reduction == null ? base.vectorPolicyReduction() : reduction,
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
}
