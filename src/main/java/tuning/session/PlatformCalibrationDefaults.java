package tuning.session;

import config.profile.ExecutionProfile;
import tuning.workload.CalibrationWorkloads;

import java.nio.file.Path;
import java.util.List;

public final class PlatformCalibrationDefaults {
    private PlatformCalibrationDefaults() {
    }

    public static PlatformCalibrationRequest balancedInference(
            String platformId,
            ExecutionProfile seedProfile,
            Path outputProfilePath
    ) {
        return PlatformCalibrationRequest.fromSeedExecutionProfile(
                platformId,
                seedProfile,
                List.of(
                        matmulStep("calib-matmul", TuningPreset.BALANCED),
                        fusedStep("calib-fused", TuningPreset.BALANCED),
                        elementwiseDispatchStep("calib-elementwise", TuningPreset.BALANCED),
                        reductionStep("calib-reduction", TuningPreset.BALANCED),
                        schedulerStep("calib-scheduler", TuningPreset.BALANCED),
                        materializationStep("calib-materialization", TuningPreset.BALANCED)
                ),
                outputProfilePath
        );
    }

    public static PlatformCalibrationRequest balancedInferenceFull(
            String platformId,
            ExecutionProfile seedProfile,
            Path outputProfilePath
    ) {
        return PlatformCalibrationRequest.fromSeedExecutionProfile(
                platformId,
                seedProfile,
                List.of(
                        matmulStep("calib-matmul", TuningPreset.BALANCED),
                        fusedStep("calib-fused", TuningPreset.BALANCED),
                        elementwiseDispatchStep("calib-elementwise", TuningPreset.BALANCED),
                        reductionStep("calib-reduction", TuningPreset.BALANCED),
                        schedulerStep("calib-scheduler", TuningPreset.BALANCED),
                        materializationStep("calib-materialization", TuningPreset.BALANCED),
                        numericsStep("calib-numerics", TuningPreset.BALANCED)
                ),
                outputProfilePath
        );
    }

    public static PlatformCalibrationRequest thoroughInference(
            String platformId,
            ExecutionProfile seedProfile,
            Path outputProfilePath
    ) {
        return PlatformCalibrationRequest.fromSeedExecutionProfile(
                platformId,
                seedProfile,
                List.of(
                        matmulStep("calib-matmul", TuningPreset.THOROUGH),
                        fusedStep("calib-fused", TuningPreset.THOROUGH),
                        elementwiseDispatchStep("calib-elementwise", TuningPreset.THOROUGH),
                        reductionStep("calib-reduction", TuningPreset.THOROUGH),
                        schedulerStep("calib-scheduler", TuningPreset.THOROUGH),
                        materializationStep("calib-materialization", TuningPreset.THOROUGH)
                ),
                outputProfilePath
        );
    }

    public static PlatformCalibrationRequest thoroughInferenceFull(
            String platformId,
            ExecutionProfile seedProfile,
            Path outputProfilePath
    ) {
        return PlatformCalibrationRequest.fromSeedExecutionProfile(
                platformId,
                seedProfile,
                List.of(
                        matmulStep("calib-matmul", TuningPreset.THOROUGH),
                        fusedStep("calib-fused", TuningPreset.THOROUGH),
                        elementwiseDispatchStep("calib-elementwise", TuningPreset.THOROUGH),
                        reductionStep("calib-reduction", TuningPreset.THOROUGH),
                        schedulerStep("calib-scheduler", TuningPreset.THOROUGH),
                        materializationStep("calib-materialization", TuningPreset.THOROUGH),
                        numericsStep("calib-numerics", TuningPreset.THOROUGH)
                ),
                outputProfilePath
        );
    }

    public static PlatformCalibrationRequest quickInference(
            String platformId,
            ExecutionProfile seedProfile,
            Path outputProfilePath
    ) {
        return PlatformCalibrationRequest.fromSeedExecutionProfile(
                platformId,
                seedProfile,
                List.of(
                        matmulStep("calib-matmul", TuningPreset.QUICK),
                        fusedStep("calib-fused", TuningPreset.QUICK),
                        elementwiseDispatchStep("calib-elementwise", TuningPreset.QUICK),
                        schedulerStep("calib-scheduler", TuningPreset.QUICK)
                ),
                outputProfilePath
        );
    }

    public static PlatformCalibrationRequest balancedTraining(
            String platformId,
            ExecutionProfile seedProfile,
            Path outputProfilePath
    ) {
        return PlatformCalibrationRequest.fromSeedExecutionProfile(
                platformId,
                seedProfile,
                List.of(
                        matmulStep("calib-matmul-train", TuningPreset.BALANCED),
                        fusedStep("calib-fused-train", TuningPreset.BALANCED),
                        elementwiseDispatchStep("calib-elementwise-train", TuningPreset.BALANCED),
                        reductionStep("calib-reduction-train", TuningPreset.BALANCED),
                        schedulerStep("calib-scheduler-train", TuningPreset.BALANCED),
                        materializationStep("calib-materialization-train", TuningPreset.BALANCED),
                        numericsStep("calib-numerics-train", TuningPreset.BALANCED)
                ),
                outputProfilePath
        );
    }

    public static PlatformCalibrationRequest balancedTrainingFull(
            String platformId,
            ExecutionProfile seedProfile,
            Path outputProfilePath
    ) {
        return balancedTraining(platformId, seedProfile, outputProfilePath);
    }

    public static PlatformCalibrationRequest thoroughTraining(
            String platformId,
            ExecutionProfile seedProfile,
            Path outputProfilePath
    ) {
        return PlatformCalibrationRequest.fromSeedExecutionProfile(
                platformId,
                seedProfile,
                List.of(
                        matmulStep("calib-matmul-train", TuningPreset.THOROUGH),
                        fusedStep("calib-fused-train", TuningPreset.THOROUGH),
                        elementwiseDispatchStep("calib-elementwise-train", TuningPreset.THOROUGH),
                        reductionStep("calib-reduction-train", TuningPreset.THOROUGH),
                        schedulerStep("calib-scheduler-train", TuningPreset.THOROUGH),
                        materializationStep("calib-materialization-train", TuningPreset.THOROUGH),
                        numericsStep("calib-numerics-train", TuningPreset.THOROUGH)
                ),
                outputProfilePath
        );
    }

    public static PlatformCalibrationRequest thoroughTrainingFull(
            String platformId,
            ExecutionProfile seedProfile,
            Path outputProfilePath
    ) {
        return thoroughTraining(platformId, seedProfile, outputProfilePath);
    }

    public static PlatformCalibrationRequest quickTraining(
            String platformId,
            ExecutionProfile seedProfile,
            Path outputProfilePath
    ) {
        return PlatformCalibrationRequest.fromSeedExecutionProfile(
                platformId,
                seedProfile,
                List.of(
                        matmulStep("calib-matmul-train", TuningPreset.QUICK),
                        fusedStep("calib-fused-train", TuningPreset.QUICK),
                        elementwiseDispatchStep("calib-elementwise-train", TuningPreset.QUICK),
                        reductionStep("calib-reduction-train", TuningPreset.QUICK)
                ),
                outputProfilePath
        );
    }

    public static PlatformCalibrationStep matmulStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                PlatformCalibrationFamily.MATMUL,
                List.of(
                        CalibrationWorkloads.matmulSquare(name + "_workload_small", 64),
                        CalibrationWorkloads.matmulSquare(name + "_workload_medium", 128),
                        CalibrationWorkloads.matmulTallSkinny(name + "_workload_tall_skinny", 256, 64, 64),
                        CalibrationWorkloads.matmulBatchedAttentionLike(name + "_workload_attention_like", 8, 128, 64, 64)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.matmulShapeHeuristics(
                                        List.of(true, false),
                                        List.of(1.5, 2.0, 3.0, 4.0, 6.0)
                                ),
                                PlatformRuntimeProfileMutators.blasThreads(List.of(0, 1, 2, 4)),
                                PlatformRuntimeProfileMutators.matmulParallelThresholds(List.of(100_000, 500_000, 2_000_000))
                        )
                ),
                PlatformCalibrationScorePolicy.averageMedianMs()
        );
    }

    public static PlatformCalibrationStep fusedStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                PlatformCalibrationFamily.FUSED_ARITHMETIC,
                List.of(
                        CalibrationWorkloads.fusedCheapElementwise(name + "_workload_cheap", 65_536),
                        CalibrationWorkloads.fusedTranscendental(name + "_workload_trans", 65_536)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.fusedDispatchThresholds(
                                        List.of(64, 128, 256, 512, 1_024),
                                        List.of(16, 32, 64, 128, 256),
                                        List.of(4_096, 8_192, 16_384, 32_768),
                                        List.of(1_024, 2_048, 4_096, 8_192)
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.averageMedianMs()
        );
    }

    public static PlatformCalibrationStep elementwiseDispatchStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                PlatformCalibrationFamily.ELEMENTWISE_DISPATCH,
                List.of(
                        CalibrationWorkloads.schedulerCheapParallel(name + "_workload_medium", 65_536),
                        CalibrationWorkloads.schedulerCheapParallel(name + "_workload_large", 262_144)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.elementwiseDispatchThresholds(
                                        List.of(128, 256, 512, 1_024, 2_048),
                                        List.of(32, 64, 128, 256, 512),
                                        List.of(8_192, 16_384, 32_768, 65_536),
                                        List.of(2_048, 4_096, 8_192, 16_384)
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.averageMedianMs()
        );
    }

    public static PlatformCalibrationStep reductionStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                PlatformCalibrationFamily.REDUCTION,
                List.of(
                        CalibrationWorkloads.reductionSum(name + "_workload_medium", 65_536),
                        CalibrationWorkloads.reductionSum(name + "_workload_large", 262_144)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.reductionThresholds(
                                        List.of(512, 2_048, 8_192, 16_384),
                                        List.of(8_192, 16_384, 32_768, 65_536)
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.averageMedianMs()
        );
    }

    public static PlatformCalibrationStep schedulerStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                PlatformCalibrationFamily.SCHEDULER,
                List.of(
                        CalibrationWorkloads.schedulerCheapParallel(name + "_workload_medium", 65_536),
                        CalibrationWorkloads.schedulerCheapParallel(name + "_workload_large", 262_144)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.advancedSchedulerPolicies(
                                        aroundInt(base.scheduler().lowCostTargetChunksPerWorker(), 1, 8),
                                        aroundInt(base.scheduler().mediumCostTargetChunksPerWorker(), 1, 4),
                                        aroundInt(base.scheduler().highCostTargetChunksPerWorker(), 1, 2),
                                        aroundScaled(base.scheduler().minScalarChunkSize(), 512, 8_192),
                                        aroundScaled(base.scheduler().minVectorChunkSize(), 1_024, 16_384),
                                        aroundScaled(base.scheduler().minReductionChunkSize(), 2_048, 32_768),
                                        aroundScaled(base.scheduler().commonPoolLowCostMaxWorkPerWorker(), 4_096, 65_536)
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.averageMedianMs()
        );
    }

    public static PlatformCalibrationStep materializationStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                PlatformCalibrationFamily.MATERIALIZATION,
                List.of(
                        CalibrationWorkloads.materializationStridedElementwise(name + "_workload_small", 128, 128),
                        CalibrationWorkloads.materializationStridedElementwise(name + "_workload_medium", 256, 256)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.materializationThresholds(
                                        aroundScaled(base.materialization().contiguousMaterializeThreshold(), 4_096, 1_048_576)
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.averageMedianMs()
        );
    }

    public static PlatformCalibrationStep numericsStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                PlatformCalibrationFamily.NUMERICS,
                List.of(
                        CalibrationWorkloads.fusedTranscendental(name + "_workload", 65_536)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.numericsPolicies(
                                        List.of(backend.ApproxMode.OFF, backend.ApproxMode.TRAINING_ONLY, backend.ApproxMode.ALWAYS),
                                        List.of(true, false)
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.averageMedianMs()
        );
    }

    private static List<Integer> aroundInt(int value, int min, int max) {
        java.util.LinkedHashSet<Integer> out = new java.util.LinkedHashSet<>();
        int base = Math.max(min, Math.min(max, value));
        out.add(base);
        out.add(Math.max(min, base - 1));
        out.add(Math.min(max, base + 1));
        return List.copyOf(out);
    }

    private static List<Integer> aroundScaled(int value, int min, int max) {
        java.util.LinkedHashSet<Integer> out = new java.util.LinkedHashSet<>();
        int base = clamp(value, min, max);
        out.add(base);
        out.add(clamp(base / 2, min, max));
        out.add(clamp(base * 2, min, max));
        return List.copyOf(out);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }
}
