package tuning.calibration;

import backend.blas.BlasProvider;
import backend.cpu.kernels.plan.CpuExecutionPlanner;
import config.backend.CpuMatMulMicroKernel;
import config.profile.ExecutionProfile;
import backend.cpu.fused.optimize.FusedDispatchFamily;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.runtime.PlatformRuntimeProfileGridCandidateSpace;
import tuning.calibration.runtime.PlatformRuntimeProfileMutator;
import tuning.calibration.runtime.PlatformRuntimeProfileMutators;
import tuning.preset.TuningPreset;
import tuning.workload.CalibrationWorkloads;
import tensor.DataType;

import java.util.ArrayList;

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
                standardInferenceSteps("calib", TuningPreset.BALANCED, seedProfile.dataType(), true, true, true),
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
                standardInferenceSteps("calib", TuningPreset.BALANCED, seedProfile.dataType(), true, true, true),
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
                standardInferenceSteps("calib", TuningPreset.THOROUGH, seedProfile.dataType(), true, true, true),
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
                standardInferenceSteps("calib", TuningPreset.THOROUGH, seedProfile.dataType(), true, true, true),
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
                standardInferenceSteps("calib", TuningPreset.QUICK, seedProfile.dataType(), false, false, true),
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
                standardTrainingSteps("calib", TuningPreset.BALANCED, seedProfile.dataType(), true, true, true),
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
                standardTrainingSteps("calib", TuningPreset.THOROUGH, seedProfile.dataType(), true, true, true),
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
                standardTrainingSteps("calib", TuningPreset.QUICK, seedProfile.dataType(), true, false, false),
                outputProfilePath
        );
    }

    public static PlatformCalibrationStep matmulJavaStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                CalibrationFamilyId.MATMUL,
                List.of(
                        CalibrationWorkloads.matmulSquare(name + "_workload_medium", 128),
                        CalibrationWorkloads.matmulSquare(name + "_workload_large", 256),
                        CalibrationWorkloads.matmulWide(name + "_workload_projection_wide", 256, 256, 2_048),
                        CalibrationWorkloads.matmulTallSkinny(name + "_workload_projection_tall", 2_048, 256, 256),
                        CalibrationWorkloads.matmulBatchedAttentionLike(name + "_workload_attention_like", 8, 128, 64, 64)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.matmulMicroKernels(supportedMatMulMicroKernels(seedProfileDataType(base))),
                                PlatformRuntimeProfileMutators.matmulTiles(supportedMatMulTiles(seedProfileDataType(base))),
                                PlatformRuntimeProfileMutators.matmulParallelThresholds(List.of(100_000, 500_000, 2_000_000))
                        )
                ),
                PlatformCalibrationScorePolicy.averageMedianMs()
        );
    }

    public static PlatformCalibrationStep matmulBlasDispatchStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                CalibrationFamilyId.MATMUL,
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
                                PlatformRuntimeProfileMutators.matmulBlasProviders(
                                        List.of(BlasProvider.NONE, BlasProvider.OPENBLAS_FFM),
                                        List.of(1_000_000L, 2_000_000L, 4_000_000L)
                                ),
                                PlatformRuntimeProfileMutators.matmulShapeHeuristics(
                                        List.of(true, false),
                                        supportedMatMulBlasShapeRatios(seedProfileDataType(base))
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.averageMedianMs()
        );
    }

    public static PlatformCalibrationStep matmulBlasWideDispatchStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                CalibrationFamilyId.MATMUL,
                List.of(
                        CalibrationWorkloads.matmulWide(name + "_workload_ratio6_medium_wide", 128, 128, 768),
                        CalibrationWorkloads.matmulWide(name + "_workload_ratio8_medium_wide", 128, 128, 1024),
                        CalibrationWorkloads.matmulWide(name + "_workload_ratio8_large_wide", 256, 256, 2048),
                        CalibrationWorkloads.matmulWide(name + "_workload_ratio12_medium_wide", 128, 128, 1536)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.matmulWideShapeHeuristics(
                                        List.of(true, false),
                                        supportedMatMulWideBlasShapeRatios(seedProfileDataType(base))
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.weightedGeometricMeanWithWorstBucketPenalty(0.25d)
        );
    }

    public static PlatformCalibrationStep attentionMatmulStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                CalibrationFamilyId.ATTENTION_MATMUL,
                List.of(
                        CalibrationWorkloads.maskedAttention(name + "_workload_medium", 4, 8, 64, 32, 32),
                        CalibrationWorkloads.maskedAttention(name + "_workload_large", 4, 8, 128, 32, 32)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.attentionMatmulMicroKernels(
                                        supportedMatMulMicroKernels(seedProfileDataType(base))
                                ),
                                PlatformRuntimeProfileMutators.attentionMatmulTiles(
                                        supportedAttentionMatMulTiles(seedProfileDataType(base))
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.weightedGeometricMeanWithWorstBucketPenalty(0.25d)
        );
    }

    public static PlatformCalibrationStep acceleratorMetalSelectionStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                CalibrationFamilyId.METAL_SELECTION,
                List.of(
                        CalibrationWorkloads.appleMetalMatmulAddTanh(name + "_workload_medium", 128, 256, 256),
                        CalibrationWorkloads.appleMetalMatmulAddTanh(name + "_workload_large", 512, 1024, 1024)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.metalSelectionPolicies(
                                        List.of(true, false),
                                        List.of(false, true),
                                        List.of(0L, 8_000_000L, 64_000_000L, 256_000_000L)
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.weightedGeometricMeanWithWorstBucketPenalty(0.25d)
        );
    }

    public static PlatformCalibrationStep conv2dGemmDispatchStep(String name, TuningPreset preset, DataType dataType) {
        CalibrationFamilyId family = switch (dataType) {
            case FLOAT64 -> CalibrationFamilyId.CONV2D_GEMM_DISPATCH;
            case FLOAT32 -> CalibrationFamilyId.CONV2D_GEMM_DISPATCH;
            case BFLOAT16 -> CalibrationFamilyId.CONV2D_GEMM_DISPATCH;
            default -> throw new IllegalArgumentException("Unsupported conv2d calibration dtype: " + dataType);
        };
        return new PlatformCalibrationStep(
                name,
                family,
                List.of(
                        CalibrationWorkloads.conv2dPointwiseProjection(name + "_workload_pointwise_8_low", 4, 128, 64, 8, 8),
                        CalibrationWorkloads.conv2dPointwiseProjection(name + "_workload_pointwise_8_edge_1m", 4, 128, 128, 8, 8),
                        CalibrationWorkloads.conv2dPointwiseProjection(name + "_workload_pointwise_8_edge_2m", 4, 128, 256, 8, 8),
                        CalibrationWorkloads.conv2dPointwiseProjection(name + "_workload_pointwise_16_edge_4m", 1, 128, 128, 16, 16),
                        CalibrationWorkloads.conv2dResnet3x3(name + "_workload_resnet_8_mid", 8, 64, 64, 8, 8),
                        CalibrationWorkloads.conv2dResnet3x3(name + "_workload_resnet_8_high", 8, 64, 128, 8, 8),
                        CalibrationWorkloads.conv2dResnet3x3(name + "_workload_resnet_28", 2, 64, 128, 28, 28),
                        CalibrationWorkloads.conv2dPointwiseProjection(name + "_workload_pointwise_56", 2, 128, 256, 56, 56)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        supportedConv2dDispatchMutators(dataType)
                ),
                PlatformCalibrationScorePolicy.weightedGeometricMeanWithWorstBucketPenalty(0.25d)
        );
    }

    private static DataType seedProfileDataType(config.profile.PlatformRuntimeProfile base) {
        return base.metadata().dataType();
    }

    private static List<CpuMatMulMicroKernel> supportedMatMulMicroKernels(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> List.of(
                    CpuMatMulMicroKernel.F64_2X1,
                    CpuMatMulMicroKernel.F64_4X1,
                    CpuMatMulMicroKernel.F64_2X2
            );
            case BFLOAT16 -> List.of(
                    CpuMatMulMicroKernel.BF16_2X4,
                    CpuMatMulMicroKernel.BF16_4X2,
                    CpuMatMulMicroKernel.BF16_4X4
            );
            case FLOAT32 -> List.of(
                    CpuMatMulMicroKernel.F32_2X4,
                    CpuMatMulMicroKernel.F32_2X8,
                    CpuMatMulMicroKernel.F32_4X2,
                    CpuMatMulMicroKernel.F32_4X4
            );
            default -> List.of(CpuMatMulMicroKernel.AUTO);
        };
    }

    private static List<PlatformRuntimeProfileMutators.MatmulTiles> supportedMatMulTiles(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> List.of(
                    new PlatformRuntimeProfileMutators.MatmulTiles(16, 64, 32),
                    new PlatformRuntimeProfileMutators.MatmulTiles(32, 64, 32),
                    new PlatformRuntimeProfileMutators.MatmulTiles(32, 64, 64),
                    new PlatformRuntimeProfileMutators.MatmulTiles(32, 128, 64)
            );
            case BFLOAT16 -> List.of(
                    new PlatformRuntimeProfileMutators.MatmulTiles(16, 64, 64),
                    new PlatformRuntimeProfileMutators.MatmulTiles(16, 128, 64),
                    new PlatformRuntimeProfileMutators.MatmulTiles(32, 64, 64),
                    new PlatformRuntimeProfileMutators.MatmulTiles(32, 128, 64),
                    new PlatformRuntimeProfileMutators.MatmulTiles(64, 128, 64)
            );
            case FLOAT32 -> List.of(
                    new PlatformRuntimeProfileMutators.MatmulTiles(32, 64, 64),
                    new PlatformRuntimeProfileMutators.MatmulTiles(32, 128, 64),
                    new PlatformRuntimeProfileMutators.MatmulTiles(64, 128, 64),
                    new PlatformRuntimeProfileMutators.MatmulTiles(64, 128, 128),
                    new PlatformRuntimeProfileMutators.MatmulTiles(64, 256, 128)
            );
            default -> List.of(
                    new PlatformRuntimeProfileMutators.MatmulTiles(
                            CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M,
                            CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N,
                            CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K
                    )
            );
        };
    }

    private static List<PlatformRuntimeProfileMutators.MatmulTiles> supportedAttentionMatMulTiles(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> List.of(
                    new PlatformRuntimeProfileMutators.MatmulTiles(16, 64, 32),
                    new PlatformRuntimeProfileMutators.MatmulTiles(32, 64, 32),
                    new PlatformRuntimeProfileMutators.MatmulTiles(32, 128, 64)
            );
            case FLOAT32 -> List.of(
                    new PlatformRuntimeProfileMutators.MatmulTiles(32, 64, 64),
                    new PlatformRuntimeProfileMutators.MatmulTiles(32, 128, 64),
                    new PlatformRuntimeProfileMutators.MatmulTiles(64, 128, 64),
                    new PlatformRuntimeProfileMutators.MatmulTiles(64, 128, 128),
                    new PlatformRuntimeProfileMutators.MatmulTiles(64, 256, 128)
            );
            default -> List.of(
                    new PlatformRuntimeProfileMutators.MatmulTiles(
                            CpuExecutionPlanner.DEFAULT_MATMUL_TILE_M,
                            CpuExecutionPlanner.DEFAULT_MATMUL_TILE_N,
                            CpuExecutionPlanner.DEFAULT_MATMUL_TILE_K
                    )
            );
        };
    }

    private static List<Double> supportedMatMulBlasShapeRatios(DataType dataType) {
        return switch (dataType) {
            case BFLOAT16 -> List.of(1.5, 2.0, 3.0, 4.0, 6.0);
            case FLOAT32 -> List.of(1.5, 2.0, 3.0, 4.0, 6.0);
            default -> List.of(1.5, 2.0, 3.0, 4.0, 6.0);
        };
    }

    private static List<Double> supportedMatMulWideBlasShapeRatios(DataType dataType) {
        return switch (dataType) {
            case BFLOAT16 -> List.of(4.0, 6.0, 8.0, 12.0);
            case FLOAT32 -> List.of(4.0, 6.0, 8.0);
            default -> List.of(4.0, 6.0, 8.0);
        };
    }

    public static PlatformCalibrationStep fusedDispatchStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                CalibrationFamilyId.FUSED_DISPATCH,
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

    public static PlatformCalibrationStep fusedCheapContiguousStep(String name, TuningPreset preset, DataType dataType) {
        return fusedAsmWidthStep(
                name,
                CalibrationFamilyId.FUSED_CHEAP_CONTIGUOUS_WIDTH,
                FusedDispatchFamily.CHEAP_CONTIGUOUS,
                List.of(CalibrationWorkloads.fusedCheapElementwise(name + "_workload", 65_536)),
                preset,
                dataType
        );
    }

    public static PlatformCalibrationStep fusedCheapStridedStep(String name, TuningPreset preset, DataType dataType) {
        return fusedAsmWidthStep(
                name,
                CalibrationFamilyId.FUSED_CHEAP_STRIDED_WIDTH,
                FusedDispatchFamily.CHEAP_STRIDED,
                List.of(CalibrationWorkloads.fusedCheapStridedElementwise(name + "_workload", 256, 256)),
                preset,
                dataType
        );
    }

    public static PlatformCalibrationStep fusedNonCheapContiguousStep(String name, TuningPreset preset, DataType dataType) {
        return fusedAsmWidthStep(
                name,
                CalibrationFamilyId.FUSED_NON_CHEAP_CONTIGUOUS_WIDTH,
                FusedDispatchFamily.NON_CHEAP_CONTIGUOUS,
                List.of(CalibrationWorkloads.fusedTranscendental(name + "_workload", 65_536)),
                preset,
                dataType
        );
    }

    public static PlatformCalibrationStep fusedNonCheapStridedStep(String name, TuningPreset preset, DataType dataType) {
        return fusedAsmWidthStep(
                name,
                CalibrationFamilyId.FUSED_NON_CHEAP_STRIDED_WIDTH,
                FusedDispatchFamily.NON_CHEAP_STRIDED,
                List.of(
                        CalibrationWorkloads.fusedTranscendentalStrided(name + "_transcendental_workload", 256, 256),
                        CalibrationWorkloads.fusedAffineRationalStrided(name + "_affine_rational_workload", 256, 2048)
                ),
                preset,
                dataType
        );
    }

    private static List<Integer> supportedFusedAsmVectorWidths(FusedDispatchFamily family, DataType dataType) {
        int maxWidth = switch (dataType) {
            case FLOAT64 -> jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length();
            case FLOAT32, BFLOAT16 -> jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length();
            default -> 1;
        };
        List<Integer> widths = new ArrayList<>();
        widths.add(1);
        if (maxWidth >= 2) {
            widths.add(2);
        }
        if (maxWidth >= 4) {
            widths.add(4);
        }
        if (maxWidth >= 8
                || (family == FusedDispatchFamily.CHEAP_CONTIGUOUS
                && (dataType == DataType.FLOAT32 || dataType == DataType.BFLOAT16))) {
            widths.add(8);
        }
        return List.copyOf(widths);
    }

    private static PlatformCalibrationStep fusedAsmWidthStep(
            String name,
            CalibrationFamilyId family,
            FusedDispatchFamily dispatchFamily,
            List<tuning.workload.WorkloadSpec> workloads,
            TuningPreset preset,
            DataType dataType
    ) {
        return new PlatformCalibrationStep(
                name,
                family,
                workloads,
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.fusedAsmVectorWidths(
                                        dispatchFamily,
                                        supportedFusedAsmVectorWidths(dispatchFamily, dataType)
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.averageMedianMs()
        );
    }

    private static List<PlatformCalibrationStep> standardInferenceSteps(
            String prefix,
            TuningPreset preset,
            DataType dataType,
            boolean includeReduction,
            boolean includeMaterialization,
            boolean includeScheduler
    ) {
        List<PlatformCalibrationStep> steps = new ArrayList<>();
        addMatmulSteps(steps, prefix + "-matmul", preset);
        steps.add(conv2dGemmDispatchStep(prefix + "-conv2d", preset, dataType));
        addFusedSteps(steps, prefix + "-fused", preset, dataType);
        steps.add(elementwiseDispatchStep(prefix + "-elementwise", preset));
        if (includeReduction) {
            steps.add(reductionStep(prefix + "-reduction", preset));
            steps.add(attentionStep(prefix + "-attention", preset));
            steps.add(attentionMatmulStep(prefix + "-attention-matmul", preset));
        }
        if (includeScheduler) {
            steps.add(schedulerStep(prefix + "-scheduler", preset));
        }
        if (includeMaterialization) {
            steps.add(materializationStep(prefix + "-materialization", preset));
            steps.add(whereMaterializationStep(prefix + "-materialization-where", preset));
        }
        return List.copyOf(steps);
    }

    private static List<PlatformCalibrationStep> standardTrainingSteps(
            String prefix,
            TuningPreset preset,
            DataType dataType,
            boolean includeReduction,
            boolean includeMaterialization,
            boolean includeScheduler
    ) {
        List<PlatformCalibrationStep> steps = new ArrayList<>();
        addMatmulSteps(steps, prefix + "-matmul-train", preset);
        steps.add(conv2dGemmDispatchStep(prefix + "-conv2d-train", preset, dataType));
        addFusedSteps(steps, prefix + "-fused-train", preset, dataType);
        steps.add(elementwiseDispatchStep(prefix + "-elementwise-train", preset));
        if (includeReduction) {
            steps.add(reductionStep(prefix + "-reduction-train", preset));
            steps.add(attentionStep(prefix + "-attention-train", preset));
            steps.add(attentionMatmulStep(prefix + "-attention-matmul-train", preset));
        }
        if (includeScheduler) {
            steps.add(schedulerStep(prefix + "-scheduler-train", preset));
        }
        if (includeMaterialization) {
            steps.add(materializationStep(prefix + "-materialization-train", preset));
            steps.add(whereMaterializationStep(prefix + "-materialization-where-train", preset));
        }
        return List.copyOf(steps);
    }

    private static List<PlatformRuntimeProfileMutator> supportedConv2dDispatchMutators(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> List.of(
                    PlatformRuntimeProfileMutators.conv2dBlasProviders(
                            List.of(BlasProvider.NONE, BlasProvider.OPENBLAS_FFM),
                            List.of(50_000L, 100_000L, 250_000L, 1_000_000L, 4_000_000L),
                            List.of(),
                            List.of()
                    )
            );
            case FLOAT32 -> List.of(
                    PlatformRuntimeProfileMutators.conv2dBlasProviders(
                            List.of(BlasProvider.NONE, BlasProvider.OPENBLAS_FFM),
                            List.of(),
                            List.of(50_000L, 100_000L, 250_000L, 1_000_000L, 4_000_000L),
                            List.of()
                    ),
                    PlatformRuntimeProfileMutators.conv2dShapeHeuristics(
                            List.of(true, false),
                            List.of(1.5, 2.0, 3.0, 4.0, 6.0, 100.0)
                    )
            );
            case BFLOAT16 -> List.of(
                    PlatformRuntimeProfileMutators.conv2dBlasProviders(
                            List.of(BlasProvider.NONE, BlasProvider.OPENBLAS_FFM),
                            List.of(),
                            List.of(),
                            List.of(50_000L, 100_000L, 250_000L, 1_000_000L, 4_000_000L)
                    ),
                    PlatformRuntimeProfileMutators.conv2dShapeHeuristics(
                            List.of(true, false),
                            List.of(1.5, 2.0, 3.0, 4.0, 6.0, 100.0)
                    )
            );
            default -> List.of();
        };
    }

    private static void addFusedSteps(
            List<PlatformCalibrationStep> steps,
            String prefix,
            TuningPreset preset,
            DataType dataType
    ) {
        steps.add(fusedDispatchStep(prefix + "-thresholds", preset));
        steps.add(fusedCheapContiguousStep(prefix + "-cheap-contig", preset, dataType));
        steps.add(fusedCheapStridedStep(prefix + "-cheap-strided", preset, dataType));
        steps.add(fusedNonCheapContiguousStep(prefix + "-noncheap-contig", preset, dataType));
        steps.add(fusedNonCheapStridedStep(prefix + "-noncheap-strided", preset, dataType));
    }

    private static void addMatmulSteps(
            List<PlatformCalibrationStep> steps,
            String prefix,
            TuningPreset preset
    ) {
        steps.add(matmulJavaStep(prefix + "-java", preset));
        steps.add(matmulBlasDispatchStep(prefix + "-blas", preset));
        steps.add(matmulBlasWideDispatchStep(prefix + "-blas-wide", preset));
    }

    public static PlatformCalibrationStep elementwiseDispatchStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                CalibrationFamilyId.ELEMENTWISE_DISPATCH,
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
                CalibrationFamilyId.REDUCTION,
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

    public static PlatformCalibrationStep attentionStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                CalibrationFamilyId.ATTENTION_THRESHOLDS,
                List.of(
                        CalibrationWorkloads.maskedAttention(name + "_workload_medium", 4, 8, 64, 32, 32),
                        CalibrationWorkloads.maskedAttention(name + "_workload_large", 4, 8, 128, 32, 32)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.attentionThresholds(
                                        List.of(512, 2_048, 8_192, 16_384),
                                        List.of(2_048, 8_192, 16_384, 32_768)
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.weightedGeometricMeanWithWorstBucketPenalty(0.25d)
        );
    }

    public static PlatformCalibrationStep schedulerStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                CalibrationFamilyId.SCHEDULER,
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
                CalibrationFamilyId.MATERIALIZATION,
                List.of(
                        CalibrationWorkloads.materializationStridedElementwise(name + "_workload_small", 128, 128),
                        CalibrationWorkloads.materializationStridedElementwise(name + "_workload_medium", 256, 256),
                        CalibrationWorkloads.materializationStridedElementwise(name + "_workload_threshold_524k", 512, 1024),
                        CalibrationWorkloads.materializationStridedElementwise(name + "_workload_threshold_1m", 1024, 1024)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.contiguousMaterializationThresholds(
                                        materializationThresholdCandidates(base.materialization().contiguousMaterializeThreshold())
                                ),
                                PlatformRuntimeProfileMutators.materializationThresholds(
                                        materializationThresholdCandidates(switch (base.dataType()) {
                                            case FLOAT64 -> base.materialization().cheapF64MaterializeThreshold();
                                            case FLOAT32 -> base.materialization().cheapF32MaterializeThreshold();
                                            case BFLOAT16 -> base.materialization().cheapBF16MaterializeThreshold();
                                            default -> base.materialization().contiguousMaterializeThreshold();
                                        })
                                )
                        )
                ),
                PlatformCalibrationScorePolicy.averageMedianMs()
        );
    }

    public static PlatformCalibrationStep whereMaterializationStep(String name, TuningPreset preset) {
        return new PlatformCalibrationStep(
                name,
                CalibrationFamilyId.MATERIALIZATION,
                List.of(
                        CalibrationWorkloads.materializationStridedWhere(name + "_workload_small", 128, 128),
                        CalibrationWorkloads.materializationStridedWhere(name + "_workload_medium", 256, 256),
                        CalibrationWorkloads.materializationStridedWhere(name + "_workload_threshold_524k", 512, 1024),
                        CalibrationWorkloads.materializationStridedWhere(name + "_workload_threshold_1m", 1024, 1024)
                ),
                preset,
                base -> new PlatformRuntimeProfileGridCandidateSpace(
                        base,
                        List.of(
                                PlatformRuntimeProfileMutators.whereMaterializationThresholds(
                                        materializationThresholdCandidates(base.materialization().whereMaterializeThreshold())
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

    private static List<Integer> materializationThresholdCandidates(int baseValue) {
        java.util.LinkedHashSet<Integer> out = new java.util.LinkedHashSet<>();
        int base = clamp(baseValue, 4_096, 1_048_576);
        out.add(base);
        out.add(clamp(base / 2, 4_096, 1_048_576));
        out.add(clamp(base * 2, 4_096, 1_048_576));
        out.add(262_144);
        out.add(524_288);
        out.add(1_048_576);
        return List.copyOf(out);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }
}
