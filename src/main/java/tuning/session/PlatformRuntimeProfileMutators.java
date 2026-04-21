package tuning.session;

import backend.ApproxMode;
import backend.blas.BlasProvider;
import config.backend.CpuMatMulMicroKernel;
import config.profile.Conv2dPlatformProfile;
import config.profile.ElementwiseDispatchPlatformProfile;
import config.profile.FusedPlatformProfile;
import config.profile.MaterializationPlatformProfile;
import config.profile.NumericsPlatformProfile;
import config.profile.MatmulPlatformProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.ReductionPlatformProfile;
import config.profile.SchedulerPlatformProfile;
import graph.optimizer.fusion.FusedDispatchFamily;
import tuning.workload.WorkloadKind;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class PlatformRuntimeProfileMutators {
    public record MatmulTiles(int tileM, int tileN, int tileK) {
        public MatmulTiles {
            tileM = Math.max(1, tileM);
            tileN = Math.max(1, tileN);
            tileK = Math.max(1, tileK);
        }
    }

    private PlatformRuntimeProfileMutators() {
    }

    public static PlatformRuntimeProfileMutator conv2dBlasProviders(
            List<BlasProvider> providers,
            List<Long> f64MinWorks,
            List<Long> f32MinWorks,
            List<Long> bf16MinWorks
    ) {
        List<BlasProvider> safeProviders = providers == null ? List.of() : List.copyOf(providers);
        List<Long> safeF64 = f64MinWorks == null ? List.of() : List.copyOf(f64MinWorks);
        List<Long> safeF32 = f32MinWorks == null ? List.of() : List.copyOf(f32MinWorks);
        List<Long> safeBf16 = bf16MinWorks == null ? List.of() : List.copyOf(bf16MinWorks);
        return (baseProfile, workload) -> {
            if (!usesConv2dFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("conv2dBlas=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (BlasProvider provider : safeProviders) {
                if (provider == null) {
                    continue;
                }
                if (provider == BlasProvider.NONE) {
                    Conv2dPlatformProfile conv2d = new Conv2dPlatformProfile(
                            BlasProvider.NONE,
                            baseProfile.conv2d().f64BlasMinWork(),
                            baseProfile.conv2d().f32BlasMinWork(),
                            baseProfile.conv2d().f32RequireMgeK(),
                            baseProfile.conv2d().f32MaxNOverK(),
                            baseProfile.conv2d().bf16BlasMinWork(),
                            baseProfile.conv2d().bf16RequireMgeK(),
                            baseProfile.conv2d().bf16MaxNOverK()
                    );
                    out.add(new RuntimeProfileCandidate(
                            "conv2dBlasProvider=NONE",
                            withConv2d(baseProfile, conv2d),
                            Map.of("runtime.conv2d.blasProvider", conv2d.blasProvider().name())
                    ));
                    continue;
                }
                List<Long> mins = switch (baseProfile.dataType()) {
                    case FLOAT64 -> safeF64;
                    case FLOAT32 -> safeF32;
                    case BFLOAT16 -> safeBf16;
                    default -> List.of(baseProfile.conv2d().f64BlasMinWork());
                };
                for (Long minWork : mins) {
                    Conv2dPlatformProfile conv2d = switch (baseProfile.dataType()) {
                        case FLOAT64 -> new Conv2dPlatformProfile(
                                provider,
                                minWork == null ? baseProfile.conv2d().f64BlasMinWork() : minWork,
                                baseProfile.conv2d().f32BlasMinWork(),
                                baseProfile.conv2d().f32RequireMgeK(),
                                baseProfile.conv2d().f32MaxNOverK(),
                                baseProfile.conv2d().bf16BlasMinWork(),
                                baseProfile.conv2d().bf16RequireMgeK(),
                                baseProfile.conv2d().bf16MaxNOverK()
                        );
                        case FLOAT32 -> new Conv2dPlatformProfile(
                                provider,
                                baseProfile.conv2d().f64BlasMinWork(),
                                minWork == null ? baseProfile.conv2d().f32BlasMinWork() : minWork,
                                baseProfile.conv2d().f32RequireMgeK(),
                                baseProfile.conv2d().f32MaxNOverK(),
                                baseProfile.conv2d().bf16BlasMinWork(),
                                baseProfile.conv2d().bf16RequireMgeK(),
                                baseProfile.conv2d().bf16MaxNOverK()
                        );
                        case BFLOAT16 -> new Conv2dPlatformProfile(
                                provider,
                                baseProfile.conv2d().f64BlasMinWork(),
                                baseProfile.conv2d().f32BlasMinWork(),
                                baseProfile.conv2d().f32RequireMgeK(),
                                baseProfile.conv2d().f32MaxNOverK(),
                                minWork == null ? baseProfile.conv2d().bf16BlasMinWork() : minWork,
                                baseProfile.conv2d().bf16RequireMgeK(),
                                baseProfile.conv2d().bf16MaxNOverK()
                        );
                        default -> baseProfile.conv2d();
                    };
                    String key = switch (baseProfile.dataType()) {
                        case FLOAT64 -> "runtime.conv2d.f64MinWork";
                        case FLOAT32 -> "runtime.conv2d.f32MinWork";
                        case BFLOAT16 -> "runtime.conv2d.bf16MinWork";
                        default -> "runtime.conv2d.minWork";
                    };
                    long selectedMinWork = switch (baseProfile.dataType()) {
                        case FLOAT64 -> conv2d.f64BlasMinWork();
                        case FLOAT32 -> conv2d.f32BlasMinWork();
                        case BFLOAT16 -> conv2d.bf16BlasMinWork();
                        default -> 0L;
                    };
                    out.add(new RuntimeProfileCandidate(
                            "conv2dBlasProvider=" + conv2d.blasProvider().name() + ":minWork=" + selectedMinWork,
                            withConv2d(baseProfile, conv2d),
                            Map.of(
                                    "runtime.conv2d.blasProvider", conv2d.blasProvider().name(),
                                    key, String.valueOf(selectedMinWork)
                            )
                    ));
                }
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator conv2dShapeHeuristics(
            List<Boolean> requireMgeK,
            List<Double> maxNOverK
    ) {
        List<Boolean> safeRequire = requireMgeK == null ? List.of() : List.copyOf(requireMgeK);
        List<Double> safeRatios = maxNOverK == null ? List.of() : List.copyOf(maxNOverK);
        return (baseProfile, workload) -> {
            if (!usesConv2dFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("conv2dShape=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Boolean require : safeRequire) {
                for (Double ratio : safeRatios) {
                    Conv2dPlatformProfile conv2d = switch (baseProfile.dataType()) {
                        case FLOAT32 -> new Conv2dPlatformProfile(
                                baseProfile.conv2d().blasProvider(),
                                baseProfile.conv2d().f64BlasMinWork(),
                                baseProfile.conv2d().f32BlasMinWork(),
                                require == null ? baseProfile.conv2d().f32RequireMgeK() : require,
                                ratio == null ? baseProfile.conv2d().f32MaxNOverK() : ratio,
                                baseProfile.conv2d().bf16BlasMinWork(),
                                baseProfile.conv2d().bf16RequireMgeK(),
                                baseProfile.conv2d().bf16MaxNOverK()
                        );
                        case BFLOAT16 -> new Conv2dPlatformProfile(
                                baseProfile.conv2d().blasProvider(),
                                baseProfile.conv2d().f64BlasMinWork(),
                                baseProfile.conv2d().f32BlasMinWork(),
                                baseProfile.conv2d().f32RequireMgeK(),
                                baseProfile.conv2d().f32MaxNOverK(),
                                baseProfile.conv2d().bf16BlasMinWork(),
                                require == null ? baseProfile.conv2d().bf16RequireMgeK() : require,
                                ratio == null ? baseProfile.conv2d().bf16MaxNOverK() : ratio
                        );
                        default -> baseProfile.conv2d();
                    };
                    String requireKey = baseProfile.dataType() == tensor.DataType.BFLOAT16
                            ? "runtime.conv2d.bf16RequireMgeK"
                            : "runtime.conv2d.f32RequireMgeK";
                    String ratioKey = baseProfile.dataType() == tensor.DataType.BFLOAT16
                            ? "runtime.conv2d.bf16MaxNOverK"
                            : "runtime.conv2d.f32MaxNOverK";
                    boolean selectedRequire = baseProfile.dataType() == tensor.DataType.BFLOAT16
                            ? conv2d.bf16RequireMgeK()
                            : conv2d.f32RequireMgeK();
                    double selectedRatio = baseProfile.dataType() == tensor.DataType.BFLOAT16
                            ? conv2d.bf16MaxNOverK()
                            : conv2d.f32MaxNOverK();
                    out.add(new RuntimeProfileCandidate(
                            "conv2dShape=" + selectedRequire + "/" + selectedRatio,
                            withConv2d(baseProfile, conv2d),
                            Map.of(
                                    requireKey, String.valueOf(selectedRequire),
                                    ratioKey, String.valueOf(selectedRatio)
                            )
                    ));
                }
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator matmulBlasProviders(
            List<BlasProvider> providers,
            List<Long> minWorks
    ) {
        List<BlasProvider> safeProviders = providers == null ? List.of() : List.copyOf(providers);
        List<Long> safeMinWorks = minWorks == null ? List.of() : List.copyOf(minWorks);
        return (baseProfile, workload) -> {
            if (!usesMatmulFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("blasProvider=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (BlasProvider provider : safeProviders) {
                if (provider == null) {
                    continue;
                }
                if (provider == BlasProvider.NONE) {
                    MatmulPlatformProfile matmul = new MatmulPlatformProfile(
                            BlasProvider.NONE,
                            baseProfile.matmul().blasMatmulMinWork(),
                            baseProfile.matmul().blasThreads(),
                            baseProfile.matmul().f32RequireMgeK(),
                            baseProfile.matmul().f32MaxNOverK(),
                            baseProfile.matmul().loopUnrollFactor(),
                            baseProfile.matmul().matMulTileM(),
                            baseProfile.matmul().matMulTileN(),
                            baseProfile.matmul().matMulTileK(),
                            baseProfile.matmul().attentionMatMulTileM(),
                            baseProfile.matmul().attentionMatMulTileN(),
                            baseProfile.matmul().attentionMatMulTileK(),
                            baseProfile.matmul().matMulParallelMinSize(),
                            baseProfile.matmul().matMulMicroKernel(),
                            baseProfile.matmul().attentionMatMulMicroKernel()
                    );
                    out.add(new RuntimeProfileCandidate(
                            "blasProvider=NONE",
                            new PlatformRuntimeProfile(
                                    baseProfile.metadata(),
                                    matmul,
                                    baseProfile.fused(),
                                    baseProfile.elementwiseDispatch(),
                                    baseProfile.reduction(),
                                    baseProfile.scheduler(),
                                    baseProfile.materialization(),
                                    baseProfile.numerics()
                            ),
                            Map.of("runtime.blas.provider", matmul.blasProvider().name())
                    ));
                    continue;
                }
                for (Long minWork : safeMinWorks) {
                    MatmulPlatformProfile matmul = new MatmulPlatformProfile(
                            provider,
                            minWork == null ? baseProfile.matmul().blasMatmulMinWork() : minWork,
                            baseProfile.matmul().blasThreads(),
                            baseProfile.matmul().f32RequireMgeK(),
                            baseProfile.matmul().f32MaxNOverK(),
                            baseProfile.matmul().loopUnrollFactor(),
                            baseProfile.matmul().matMulTileM(),
                            baseProfile.matmul().matMulTileN(),
                            baseProfile.matmul().matMulTileK(),
                            baseProfile.matmul().attentionMatMulTileM(),
                            baseProfile.matmul().attentionMatMulTileN(),
                            baseProfile.matmul().attentionMatMulTileK(),
                            baseProfile.matmul().matMulParallelMinSize(),
                            baseProfile.matmul().matMulMicroKernel(),
                            baseProfile.matmul().attentionMatMulMicroKernel()
                    );
                    out.add(new RuntimeProfileCandidate(
                            "blasProvider=" + matmul.blasProvider().name() + ":minWork=" + matmul.blasMatmulMinWork(),
                            new PlatformRuntimeProfile(
                                    baseProfile.metadata(),
                                    matmul,
                                    baseProfile.fused(),
                                    baseProfile.elementwiseDispatch(),
                                    baseProfile.reduction(),
                                    baseProfile.scheduler(),
                                    baseProfile.materialization(),
                                    baseProfile.numerics()
                            ),
                            Map.of(
                                    "runtime.blas.provider", matmul.blasProvider().name(),
                                    "runtime.blas.minWork", String.valueOf(matmul.blasMatmulMinWork())
                            )
                    ));
                }
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator matmulShapeHeuristics(
            List<Boolean> requireMgeK,
            List<Double> maxNOverK
    ) {
        List<Boolean> safeRequire = requireMgeK == null ? List.of() : List.copyOf(requireMgeK);
        List<Double> safeRatios = maxNOverK == null ? List.of() : List.copyOf(maxNOverK);
        return (baseProfile, workload) -> {
            if (!usesMatmulFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("matmulShape=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Boolean require : safeRequire) {
                for (Double ratio : safeRatios) {
                    MatmulPlatformProfile matmul = new MatmulPlatformProfile(
                            baseProfile.matmul().blasProvider(),
                            baseProfile.matmul().blasMatmulMinWork(),
                            baseProfile.matmul().blasThreads(),
                            require == null ? baseProfile.matmul().f32RequireMgeK() : require,
                            ratio == null ? baseProfile.matmul().f32MaxNOverK() : ratio,
                            baseProfile.matmul().loopUnrollFactor(),
                            baseProfile.matmul().matMulTileM(),
                            baseProfile.matmul().matMulTileN(),
                            baseProfile.matmul().matMulTileK(),
                            baseProfile.matmul().attentionMatMulTileM(),
                            baseProfile.matmul().attentionMatMulTileN(),
                            baseProfile.matmul().attentionMatMulTileK(),
                            baseProfile.matmul().matMulParallelMinSize(),
                            baseProfile.matmul().matMulMicroKernel(),
                            baseProfile.matmul().attentionMatMulMicroKernel()
                    );
                    out.add(new RuntimeProfileCandidate(
                            "matmulShape=" + matmul.f32RequireMgeK() + "/" + matmul.f32MaxNOverK(),
                            new PlatformRuntimeProfile(
                                    baseProfile.metadata(),
                                    matmul,
                                    baseProfile.fused(),
                                    baseProfile.elementwiseDispatch(),
                                    baseProfile.reduction(),
                                    baseProfile.scheduler(),
                                    baseProfile.materialization(),
                                    baseProfile.numerics()
                            ),
                            Map.of(
                                    "runtime.blas.f32RequireMgeK", String.valueOf(matmul.f32RequireMgeK()),
                                    "runtime.blas.f32MaxNOverK", String.valueOf(matmul.f32MaxNOverK())
                            )
                    ));
                }
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator matmulWideShapeHeuristics(
            List<Boolean> requireMgeK,
            List<Double> maxNOverK
    ) {
        List<Boolean> safeRequire = requireMgeK == null ? List.of() : List.copyOf(requireMgeK);
        List<Double> safeRatios = maxNOverK == null ? List.of() : List.copyOf(maxNOverK);
        return (baseProfile, workload) -> {
            if (!usesMatmulFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("matmulWideShape=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Boolean require : safeRequire) {
                for (Double ratio : safeRatios) {
                    MatmulPlatformProfile matmul = new MatmulPlatformProfile(
                            baseProfile.matmul().blasProvider(),
                            baseProfile.matmul().blasMatmulMinWork(),
                            baseProfile.matmul().blasThreads(),
                            baseProfile.matmul().f32RequireMgeK(),
                            baseProfile.matmul().f32MaxNOverK(),
                            require == null ? baseProfile.matmul().f32WideRequireMgeK() : require,
                            ratio == null ? baseProfile.matmul().f32WideMaxNOverK() : ratio,
                            baseProfile.matmul().loopUnrollFactor(),
                            baseProfile.matmul().matMulTileM(),
                            baseProfile.matmul().matMulTileN(),
                            baseProfile.matmul().matMulTileK(),
                            baseProfile.matmul().attentionMatMulTileM(),
                            baseProfile.matmul().attentionMatMulTileN(),
                            baseProfile.matmul().attentionMatMulTileK(),
                            baseProfile.matmul().matMulParallelMinSize(),
                            baseProfile.matmul().matMulMicroKernel(),
                            baseProfile.matmul().attentionMatMulMicroKernel()
                    );
                    out.add(new RuntimeProfileCandidate(
                            "matmulWideShape=" + matmul.f32WideRequireMgeK() + "/" + matmul.f32WideMaxNOverK(),
                            new PlatformRuntimeProfile(
                                    baseProfile.metadata(),
                                    matmul,
                                    baseProfile.fused(),
                                    baseProfile.elementwiseDispatch(),
                                    baseProfile.reduction(),
                                    baseProfile.scheduler(),
                                    baseProfile.materialization(),
                                    baseProfile.numerics()
                            ),
                            Map.of(
                                    "runtime.blas.f32WideRequireMgeK", String.valueOf(matmul.f32WideRequireMgeK()),
                                    "runtime.blas.f32WideMaxNOverK", String.valueOf(matmul.f32WideMaxNOverK())
                            )
                    ));
                }
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator blasThreads(List<Integer> threadCounts) {
        return (baseProfile, workload) -> {
            String candidateName = "blasThreads=AUTO";
            if (!usesMatmulFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate(candidateName, baseProfile, Map.of("runtime.blas.threads", "0")));
            }
            MatmulPlatformProfile matmul = new MatmulPlatformProfile(
                    baseProfile.matmul().blasProvider(),
                    baseProfile.matmul().blasMatmulMinWork(),
                    0,
                    baseProfile.matmul().f32RequireMgeK(),
                    baseProfile.matmul().f32MaxNOverK(),
                    baseProfile.matmul().loopUnrollFactor(),
                    baseProfile.matmul().matMulTileM(),
                    baseProfile.matmul().matMulTileN(),
                    baseProfile.matmul().matMulTileK(),
                    baseProfile.matmul().attentionMatMulTileM(),
                    baseProfile.matmul().attentionMatMulTileN(),
                    baseProfile.matmul().attentionMatMulTileK(),
                    baseProfile.matmul().matMulParallelMinSize(),
                    baseProfile.matmul().matMulMicroKernel(),
                    baseProfile.matmul().attentionMatMulMicroKernel()
            );
            return List.of(new RuntimeProfileCandidate(
                    candidateName,
                    new PlatformRuntimeProfile(
                            baseProfile.metadata(),
                            matmul,
                            baseProfile.fused(),
                            baseProfile.elementwiseDispatch(),
                            baseProfile.reduction(),
                            baseProfile.scheduler(),
                            baseProfile.materialization(),
                            baseProfile.numerics()
                    ),
                    Map.of("runtime.blas.threads", "0")
            ));
        };
    }

    public static PlatformRuntimeProfileMutator matmulParallelThresholds(List<Integer> thresholds) {
        List<Integer> safe = thresholds == null ? List.of() : List.copyOf(thresholds);
        return (baseProfile, workload) -> {
            if (!usesMatmulFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("matmulParallel=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer threshold : safe) {
                MatmulPlatformProfile matmul = new MatmulPlatformProfile(
                        baseProfile.matmul().blasProvider(),
                        baseProfile.matmul().blasMatmulMinWork(),
                        baseProfile.matmul().blasThreads(),
                        baseProfile.matmul().f32RequireMgeK(),
                        baseProfile.matmul().f32MaxNOverK(),
                        baseProfile.matmul().loopUnrollFactor(),
                        baseProfile.matmul().matMulTileM(),
                        baseProfile.matmul().matMulTileN(),
                        baseProfile.matmul().matMulTileK(),
                        baseProfile.matmul().attentionMatMulTileM(),
                        baseProfile.matmul().attentionMatMulTileN(),
                        baseProfile.matmul().attentionMatMulTileK(),
                        threshold == null ? baseProfile.matmul().matMulParallelMinSize() : threshold,
                        baseProfile.matmul().matMulMicroKernel(),
                        baseProfile.matmul().attentionMatMulMicroKernel()
                );
                out.add(new RuntimeProfileCandidate(
                        "matmulParallel=" + matmul.matMulParallelMinSize(),
                        new PlatformRuntimeProfile(
                                baseProfile.metadata(),
                                matmul,
                                baseProfile.fused(),
                                baseProfile.elementwiseDispatch(),
                                baseProfile.reduction(),
                                baseProfile.scheduler(),
                                baseProfile.materialization(),
                                baseProfile.numerics()
                        ),
                        Map.of("cpu.matMulParallelMinSize", String.valueOf(matmul.matMulParallelMinSize()))
                ));
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator matmulTiles(List<MatmulTiles> tiles) {
        List<MatmulTiles> safe = tiles == null ? List.of() : List.copyOf(tiles);
        return (baseProfile, workload) -> {
            if (!usesMatmulFamily(workload.kind()) || safe.isEmpty()) {
                return List.of(new RuntimeProfileCandidate("matmulTiles=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (MatmulTiles tile : safe) {
                MatmulTiles resolved = tile == null
                        ? new MatmulTiles(
                        baseProfile.matmul().matMulTileM(),
                        baseProfile.matmul().matMulTileN(),
                        baseProfile.matmul().matMulTileK()
                )
                        : tile;
                MatmulPlatformProfile matmul = new MatmulPlatformProfile(
                        baseProfile.matmul().blasProvider(),
                        baseProfile.matmul().blasMatmulMinWork(),
                        baseProfile.matmul().blasThreads(),
                        baseProfile.matmul().f32RequireMgeK(),
                        baseProfile.matmul().f32MaxNOverK(),
                        baseProfile.matmul().loopUnrollFactor(),
                        resolved.tileM(),
                        resolved.tileN(),
                        resolved.tileK(),
                        baseProfile.matmul().attentionMatMulTileM(),
                        baseProfile.matmul().attentionMatMulTileN(),
                        baseProfile.matmul().attentionMatMulTileK(),
                        baseProfile.matmul().matMulParallelMinSize(),
                        baseProfile.matmul().matMulMicroKernel(),
                        baseProfile.matmul().attentionMatMulMicroKernel()
                );
                out.add(new RuntimeProfileCandidate(
                        "matmulTiles=" + matmul.matMulTileM() + "x" + matmul.matMulTileN() + "x" + matmul.matMulTileK(),
                        new PlatformRuntimeProfile(
                                baseProfile.metadata(),
                                matmul,
                                baseProfile.fused(),
                                baseProfile.elementwiseDispatch(),
                                baseProfile.reduction(),
                                baseProfile.scheduler(),
                                baseProfile.materialization(),
                                baseProfile.numerics()
                        ),
                        Map.of(
                                "cpu.matMulTileM", String.valueOf(matmul.matMulTileM()),
                                "cpu.matMulTileN", String.valueOf(matmul.matMulTileN()),
                                "cpu.matMulTileK", String.valueOf(matmul.matMulTileK())
                        )
                ));
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator fusedDispatchThresholds(
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
            if (!usesGenericRuntimeFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("fusedDispatch=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer cheapVec : safeCheapVec) {
                for (Integer transVec : safeTransVec) {
                    for (Integer cheapPar : safeCheapPar) {
                        for (Integer transPar : safeTransPar) {
                            FusedPlatformProfile fused = new FusedPlatformProfile(
                                    cheapVec == null ? baseProfile.fused().fusedCheapVectorMinSize() : cheapVec,
                                    transVec == null ? baseProfile.fused().fusedTranscendentalVectorMinSize() : transVec,
                                    cheapPar == null ? baseProfile.fused().fusedCheapParallelMinSize() : cheapPar,
                                    transPar == null ? baseProfile.fused().fusedTranscendentalParallelMinSize() : transPar,
                                    baseProfile.fused().fusedCheapContiguousAsmVectorWidth(),
                                    baseProfile.fused().fusedCheapStridedAsmVectorWidth(),
                                    baseProfile.fused().fusedNonCheapContiguousAsmVectorWidth(),
                                    baseProfile.fused().fusedNonCheapStridedAsmVectorWidth()
                            );
                            out.add(new RuntimeProfileCandidate(
                                    "fusedDispatch=" + fused.fusedCheapVectorMinSize() + "/" + fused.fusedTranscendentalVectorMinSize()
                                            + "/" + fused.fusedCheapParallelMinSize() + "/" + fused.fusedTranscendentalParallelMinSize(),
                                    new PlatformRuntimeProfile(
                                            baseProfile.metadata(),
                                            baseProfile.matmul(),
                                            fused,
                                            baseProfile.elementwiseDispatch(),
                                            baseProfile.reduction(),
                                            baseProfile.scheduler(),
                                            baseProfile.materialization(),
                                            baseProfile.numerics()
                                    ),
                                    Map.of(
                                            "cpu.fusedCheapVectorMinSize", String.valueOf(fused.fusedCheapVectorMinSize()),
                                            "cpu.fusedTranscendentalVectorMinSize", String.valueOf(fused.fusedTranscendentalVectorMinSize()),
                                            "cpu.fusedCheapParallelMinSize", String.valueOf(fused.fusedCheapParallelMinSize()),
                                            "cpu.fusedTranscendentalParallelMinSize", String.valueOf(fused.fusedTranscendentalParallelMinSize())
                                    )
                            ));
                        }
                    }
                }
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator matmulMicroKernels(List<CpuMatMulMicroKernel> kernels) {
        List<CpuMatMulMicroKernel> safe = kernels == null ? List.of() : List.copyOf(kernels);
        return (baseProfile, workload) -> {
            if (!usesMatmulFamily(workload.kind()) || safe.isEmpty()) {
                return List.of(new RuntimeProfileCandidate("matmulMicroKernel=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (CpuMatMulMicroKernel kernel : safe) {
                CpuMatMulMicroKernel resolved = kernel == null ? baseProfile.matmul().matMulMicroKernel() : kernel;
                MatmulPlatformProfile matmul = new MatmulPlatformProfile(
                        baseProfile.matmul().blasProvider(),
                        baseProfile.matmul().blasMatmulMinWork(),
                        baseProfile.matmul().blasThreads(),
                        baseProfile.matmul().f32RequireMgeK(),
                        baseProfile.matmul().f32MaxNOverK(),
                        baseProfile.matmul().loopUnrollFactor(),
                        baseProfile.matmul().matMulTileM(),
                        baseProfile.matmul().matMulTileN(),
                        baseProfile.matmul().matMulTileK(),
                        baseProfile.matmul().attentionMatMulTileM(),
                        baseProfile.matmul().attentionMatMulTileN(),
                        baseProfile.matmul().attentionMatMulTileK(),
                        baseProfile.matmul().matMulParallelMinSize(),
                        resolved,
                        baseProfile.matmul().attentionMatMulMicroKernel()
                );
                out.add(new RuntimeProfileCandidate(
                        "matmulMicroKernel=" + matmul.matMulMicroKernel().name(),
                        new PlatformRuntimeProfile(
                                baseProfile.metadata(),
                                matmul,
                                baseProfile.fused(),
                                baseProfile.elementwiseDispatch(),
                                baseProfile.reduction(),
                                baseProfile.scheduler(),
                                baseProfile.materialization(),
                                baseProfile.numerics()
                        ),
                        Map.of("cpu.matMulMicroKernel", matmul.matMulMicroKernel().name())
                ));
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator attentionMatmulMicroKernels(List<CpuMatMulMicroKernel> kernels) {
        List<CpuMatMulMicroKernel> safe = kernels == null ? List.of() : List.copyOf(kernels);
        return (baseProfile, workload) -> {
            if (safe.isEmpty()) {
                return List.of(new RuntimeProfileCandidate("attentionMatmulMicroKernel=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (CpuMatMulMicroKernel kernel : safe) {
                CpuMatMulMicroKernel resolved = kernel == null
                        ? baseProfile.matmul().attentionMatMulMicroKernel()
                        : kernel;
                MatmulPlatformProfile matmul = new MatmulPlatformProfile(
                        baseProfile.matmul().blasProvider(),
                        baseProfile.matmul().blasMatmulMinWork(),
                        baseProfile.matmul().blasThreads(),
                        baseProfile.matmul().f32RequireMgeK(),
                        baseProfile.matmul().f32MaxNOverK(),
                        baseProfile.matmul().loopUnrollFactor(),
                        baseProfile.matmul().matMulTileM(),
                        baseProfile.matmul().matMulTileN(),
                        baseProfile.matmul().matMulTileK(),
                        baseProfile.matmul().attentionMatMulTileM(),
                        baseProfile.matmul().attentionMatMulTileN(),
                        baseProfile.matmul().attentionMatMulTileK(),
                        baseProfile.matmul().matMulParallelMinSize(),
                        baseProfile.matmul().matMulMicroKernel(),
                        resolved
                );
                out.add(new RuntimeProfileCandidate(
                        "attentionMatmulMicroKernel=" + matmul.attentionMatMulMicroKernel().name(),
                        new PlatformRuntimeProfile(
                                baseProfile.metadata(),
                                matmul,
                                baseProfile.fused(),
                                baseProfile.elementwiseDispatch(),
                                baseProfile.reduction(),
                                baseProfile.scheduler(),
                                baseProfile.materialization(),
                                baseProfile.numerics()
                        ),
                        Map.of("cpu.attentionMatMulMicroKernel", matmul.attentionMatMulMicroKernel().name())
                ));
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator attentionMatmulTiles(List<MatmulTiles> tiles) {
        List<MatmulTiles> safe = tiles == null ? List.of() : List.copyOf(tiles);
        return (baseProfile, workload) -> {
            if (safe.isEmpty()) {
                return List.of(new RuntimeProfileCandidate("attentionMatmulTiles=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (MatmulTiles tile : safe) {
                MatmulTiles resolved = tile == null
                        ? new MatmulTiles(
                        baseProfile.matmul().attentionMatMulTileM(),
                        baseProfile.matmul().attentionMatMulTileN(),
                        baseProfile.matmul().attentionMatMulTileK()
                )
                        : tile;
                MatmulPlatformProfile matmul = new MatmulPlatformProfile(
                        baseProfile.matmul().blasProvider(),
                        baseProfile.matmul().blasMatmulMinWork(),
                        baseProfile.matmul().blasThreads(),
                        baseProfile.matmul().f32RequireMgeK(),
                        baseProfile.matmul().f32MaxNOverK(),
                        baseProfile.matmul().loopUnrollFactor(),
                        baseProfile.matmul().matMulTileM(),
                        baseProfile.matmul().matMulTileN(),
                        baseProfile.matmul().matMulTileK(),
                        resolved.tileM(),
                        resolved.tileN(),
                        resolved.tileK(),
                        baseProfile.matmul().matMulParallelMinSize(),
                        baseProfile.matmul().matMulMicroKernel(),
                        baseProfile.matmul().attentionMatMulMicroKernel()
                );
                out.add(new RuntimeProfileCandidate(
                        "attentionMatmulTiles=" + matmul.attentionMatMulTileM()
                                + "x" + matmul.attentionMatMulTileN()
                                + "x" + matmul.attentionMatMulTileK(),
                        new PlatformRuntimeProfile(
                                baseProfile.metadata(),
                                matmul,
                                baseProfile.fused(),
                                baseProfile.elementwiseDispatch(),
                                baseProfile.reduction(),
                                baseProfile.scheduler(),
                                baseProfile.materialization(),
                                baseProfile.numerics()
                        ),
                        Map.of(
                                "cpu.attentionMatMulTileM", String.valueOf(matmul.attentionMatMulTileM()),
                                "cpu.attentionMatMulTileN", String.valueOf(matmul.attentionMatMulTileN()),
                                "cpu.attentionMatMulTileK", String.valueOf(matmul.attentionMatMulTileK())
                        )
                ));
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator fusedAsmVectorWidths(List<Integer> widths) {
        List<Integer> safeWidths = widths == null ? List.of() : List.copyOf(widths);
        return (baseProfile, workload) -> {
            if (!usesGenericRuntimeFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("fusedAsmVectorWidth=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer width : safeWidths) {
                FusedPlatformProfile fused = new FusedPlatformProfile(
                        baseProfile.fused().fusedCheapVectorMinSize(),
                        baseProfile.fused().fusedTranscendentalVectorMinSize(),
                        baseProfile.fused().fusedCheapParallelMinSize(),
                        baseProfile.fused().fusedTranscendentalParallelMinSize(),
                        width == null ? baseProfile.fused().fusedCheapContiguousAsmVectorWidth() : width,
                        width == null ? baseProfile.fused().fusedCheapStridedAsmVectorWidth() : width,
                        width == null ? baseProfile.fused().fusedNonCheapContiguousAsmVectorWidth() : width,
                        width == null ? baseProfile.fused().fusedNonCheapStridedAsmVectorWidth() : width
                );
                out.add(new RuntimeProfileCandidate(
                        "fusedAsmVectorWidth=" + fused.fusedAsmVectorWidth(),
                        new PlatformRuntimeProfile(
                                baseProfile.metadata(),
                                baseProfile.matmul(),
                                fused,
                                baseProfile.elementwiseDispatch(),
                                baseProfile.reduction(),
                                baseProfile.scheduler(),
                                baseProfile.materialization(),
                                baseProfile.numerics()
                        ),
                        Map.of("cpu.fusedAsmVectorWidth", String.valueOf(fused.fusedAsmVectorWidth()))
                ));
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator fusedAsmVectorWidths(
            FusedDispatchFamily family,
            List<Integer> widths
    ) {
        List<Integer> safeWidths = widths == null ? List.of() : List.copyOf(widths);
        return (baseProfile, workload) -> {
            if (!usesGenericRuntimeFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("fusedAsmVectorWidth=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer width : safeWidths) {
                int resolvedWidth = width == null ? currentFusedAsmVectorWidth(baseProfile.fused(), family) : width;
                FusedPlatformProfile fused = withFusedAsmVectorWidth(baseProfile.fused(), family, resolvedWidth);
                out.add(new RuntimeProfileCandidate(
                        "fusedAsmVectorWidth[" + family.id() + "]=" + currentFusedAsmVectorWidth(fused, family),
                        new PlatformRuntimeProfile(
                                baseProfile.metadata(),
                                baseProfile.matmul(),
                                fused,
                                baseProfile.elementwiseDispatch(),
                                baseProfile.reduction(),
                                baseProfile.scheduler(),
                                baseProfile.materialization(),
                                baseProfile.numerics()
                        ),
                        Map.of(fusedAsmVectorWidthKey(family), String.valueOf(currentFusedAsmVectorWidth(fused, family)))
                ));
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator reductionVectorThresholds(List<Integer> thresholds) {
        List<Integer> safe = thresholds == null ? List.of() : List.copyOf(thresholds);
        return (baseProfile, workload) -> {
            if (!usesGenericRuntimeFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("reductionVector=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer threshold : safe) {
                ReductionPlatformProfile reduction = new ReductionPlatformProfile(
                        threshold == null ? baseProfile.reduction().reductionVectorMinSize() : threshold,
                        baseProfile.reduction().reductionParallelMinSize(),
                        baseProfile.reduction().attentionVectorMinSize(),
                        baseProfile.reduction().attentionParallelMinSize(),
                        baseProfile.reduction().sumAccuracyMode()
                );
                out.add(new RuntimeProfileCandidate(
                        "reductionVector=" + reduction.reductionVectorMinSize(),
                        new PlatformRuntimeProfile(
                                baseProfile.metadata(),
                                baseProfile.matmul(),
                                baseProfile.fused(),
                                baseProfile.elementwiseDispatch(),
                                reduction,
                                baseProfile.scheduler(),
                                baseProfile.materialization(),
                                baseProfile.numerics()
                        ),
                        Map.of("cpu.reductionVectorMinSize", String.valueOf(reduction.reductionVectorMinSize()))
                ));
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator parallelThresholds(
            List<Integer> cheapThresholds,
            List<Integer> transcendentalThresholds,
            List<Integer> reductionThresholds
    ) {
        List<Integer> safeCheap = cheapThresholds == null ? List.of() : List.copyOf(cheapThresholds);
        List<Integer> safeTrans = transcendentalThresholds == null ? List.of() : List.copyOf(transcendentalThresholds);
        List<Integer> safeRed = reductionThresholds == null ? List.of() : List.copyOf(reductionThresholds);
        return (baseProfile, workload) -> {
            if (!usesGenericRuntimeFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("parallelThresholds=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer cheap : safeCheap) {
                for (Integer trans : safeTrans) {
                    for (Integer red : safeRed) {
                        ElementwiseDispatchPlatformProfile elementwise = new ElementwiseDispatchPlatformProfile(
                                baseProfile.elementwiseDispatch().cheapVectorMinSize(),
                                baseProfile.elementwiseDispatch().transcendentalVectorMinSize(),
                                cheap == null ? baseProfile.elementwiseDispatch().cheapParallelMinSize() : cheap,
                                trans == null ? baseProfile.elementwiseDispatch().transcendentalParallelMinSize() : trans
                        );
                        ReductionPlatformProfile reduction = new ReductionPlatformProfile(
                                baseProfile.reduction().reductionVectorMinSize(),
                                red == null ? baseProfile.reduction().reductionParallelMinSize() : red,
                                baseProfile.reduction().attentionVectorMinSize(),
                                baseProfile.reduction().attentionParallelMinSize(),
                                baseProfile.reduction().sumAccuracyMode()
                        );
                        out.add(new RuntimeProfileCandidate(
                                "parallelThresholds=" + elementwise.cheapParallelMinSize() + "/" + elementwise.transcendentalParallelMinSize() + "/" + reduction.reductionParallelMinSize(),
                                new PlatformRuntimeProfile(
                                        baseProfile.metadata(),
                                        baseProfile.matmul(),
                                        baseProfile.fused(),
                                        elementwise,
                                        reduction,
                                        baseProfile.scheduler(),
                                        baseProfile.materialization(),
                                        baseProfile.numerics()
                                ),
                                Map.of(
                                        "cpu.cheapParallelMinSize", String.valueOf(elementwise.cheapParallelMinSize()),
                                        "cpu.transcendentalParallelMinSize", String.valueOf(elementwise.transcendentalParallelMinSize()),
                                        "cpu.reductionParallelMinSize", String.valueOf(reduction.reductionParallelMinSize())
                                )
                        ));
                    }
                }
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator elementwiseDispatchThresholds(
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
            if (!usesGenericRuntimeFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("elementwiseDispatch=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer cheapVec : safeCheapVec) {
                for (Integer transVec : safeTransVec) {
                    for (Integer cheapPar : safeCheapPar) {
                        for (Integer transPar : safeTransPar) {
                            ElementwiseDispatchPlatformProfile elementwise = new ElementwiseDispatchPlatformProfile(
                                    cheapVec == null ? baseProfile.elementwiseDispatch().cheapVectorMinSize() : cheapVec,
                                    transVec == null ? baseProfile.elementwiseDispatch().transcendentalVectorMinSize() : transVec,
                                    cheapPar == null ? baseProfile.elementwiseDispatch().cheapParallelMinSize() : cheapPar,
                                    transPar == null ? baseProfile.elementwiseDispatch().transcendentalParallelMinSize() : transPar
                            );
                            out.add(new RuntimeProfileCandidate(
                                    "elementwiseDispatch="
                                            + elementwise.cheapVectorMinSize() + "/"
                                            + elementwise.transcendentalVectorMinSize() + "/"
                                            + elementwise.cheapParallelMinSize() + "/"
                                            + elementwise.transcendentalParallelMinSize(),
                                    new PlatformRuntimeProfile(
                                            baseProfile.metadata(),
                                            baseProfile.matmul(),
                                            baseProfile.fused(),
                                            elementwise,
                                            baseProfile.reduction(),
                                            baseProfile.scheduler(),
                                            baseProfile.materialization(),
                                            baseProfile.numerics()
                                    ),
                                    Map.of(
                                            "cpu.cheapVectorMinSize", String.valueOf(elementwise.cheapVectorMinSize()),
                                            "cpu.transcendentalVectorMinSize", String.valueOf(elementwise.transcendentalVectorMinSize()),
                                            "cpu.cheapParallelMinSize", String.valueOf(elementwise.cheapParallelMinSize()),
                                            "cpu.transcendentalParallelMinSize", String.valueOf(elementwise.transcendentalParallelMinSize())
                                    )
                            ));
                        }
                    }
                }
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator reductionThresholds(
            List<Integer> vectorThresholds,
            List<Integer> parallelThresholds
    ) {
        List<Integer> safeVec = vectorThresholds == null ? List.of() : List.copyOf(vectorThresholds);
        List<Integer> safePar = parallelThresholds == null ? List.of() : List.copyOf(parallelThresholds);
        return (baseProfile, workload) -> {
            if (!usesGenericRuntimeFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("reductionThresholds=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer vec : safeVec) {
                for (Integer par : safePar) {
                    ReductionPlatformProfile reduction = new ReductionPlatformProfile(
                            vec == null ? baseProfile.reduction().reductionVectorMinSize() : vec,
                            par == null ? baseProfile.reduction().reductionParallelMinSize() : par,
                            baseProfile.reduction().attentionVectorMinSize(),
                            baseProfile.reduction().attentionParallelMinSize(),
                            baseProfile.reduction().sumAccuracyMode()
                    );
                    out.add(new RuntimeProfileCandidate(
                            "reductionThresholds=" + reduction.reductionVectorMinSize() + "/" + reduction.reductionParallelMinSize(),
                            new PlatformRuntimeProfile(
                                    baseProfile.metadata(),
                                    baseProfile.matmul(),
                                    baseProfile.fused(),
                                    baseProfile.elementwiseDispatch(),
                                    reduction,
                                    baseProfile.scheduler(),
                                    baseProfile.materialization(),
                                    baseProfile.numerics()
                            ),
                            Map.of(
                                    "cpu.reductionVectorMinSize", String.valueOf(reduction.reductionVectorMinSize()),
                                    "cpu.reductionParallelMinSize", String.valueOf(reduction.reductionParallelMinSize())
                            )
                    ));
                }
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator attentionThresholds(
            List<Integer> vectorThresholds,
            List<Integer> parallelThresholds
    ) {
        List<Integer> safeVec = vectorThresholds == null ? List.of() : List.copyOf(vectorThresholds);
        List<Integer> safePar = parallelThresholds == null ? List.of() : List.copyOf(parallelThresholds);
        return (baseProfile, workload) -> {
            if (!usesGenericRuntimeFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("attentionThresholds=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer vec : safeVec) {
                for (Integer par : safePar) {
                    ReductionPlatformProfile reduction = new ReductionPlatformProfile(
                            baseProfile.reduction().reductionVectorMinSize(),
                            baseProfile.reduction().reductionParallelMinSize(),
                            vec == null ? baseProfile.reduction().attentionVectorMinSize() : vec,
                            par == null ? baseProfile.reduction().attentionParallelMinSize() : par,
                            baseProfile.reduction().sumAccuracyMode()
                    );
                    out.add(new RuntimeProfileCandidate(
                            "attentionThresholds=" + reduction.attentionVectorMinSize() + "/" + reduction.attentionParallelMinSize(),
                            new PlatformRuntimeProfile(
                                    baseProfile.metadata(),
                                    baseProfile.matmul(),
                                    baseProfile.fused(),
                                    baseProfile.elementwiseDispatch(),
                                    reduction,
                                    baseProfile.scheduler(),
                                    baseProfile.materialization(),
                                    baseProfile.numerics()
                            ),
                            Map.of(
                                    "cpu.attentionVectorMinSize", String.valueOf(reduction.attentionVectorMinSize()),
                                    "cpu.attentionParallelMinSize", String.valueOf(reduction.attentionParallelMinSize())
                            )
                    ));
                }
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator advancedSchedulerPolicies(
            List<Integer> lowTargets,
            List<Integer> mediumTargets,
            List<Integer> highTargets,
            List<Integer> minScalarChunks,
            List<Integer> minVectorChunks,
            List<Integer> minReductionChunks,
            List<Integer> commonPoolLimits
    ) {
        List<Integer> safeLow = lowTargets == null ? List.of() : List.copyOf(lowTargets);
        List<Integer> safeMed = mediumTargets == null ? List.of() : List.copyOf(mediumTargets);
        List<Integer> safeHigh = highTargets == null ? List.of() : List.copyOf(highTargets);
        List<Integer> safeScalar = minScalarChunks == null ? List.of() : List.copyOf(minScalarChunks);
        List<Integer> safeVector = minVectorChunks == null ? List.of() : List.copyOf(minVectorChunks);
        List<Integer> safeReduction = minReductionChunks == null ? List.of() : List.copyOf(minReductionChunks);
        List<Integer> safeCommon = commonPoolLimits == null ? List.of() : List.copyOf(commonPoolLimits);
        return (baseProfile, workload) -> {
            if (!usesGenericRuntimeFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("scheduler=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer low : safeLow) {
                for (Integer med : safeMed) {
                    for (Integer high : safeHigh) {
                        for (Integer scalar : safeScalar) {
                            for (Integer vector : safeVector) {
                                for (Integer reduction : safeReduction) {
                                    for (Integer common : safeCommon) {
                                        SchedulerPlatformProfile scheduler = new SchedulerPlatformProfile(
                                                low == null ? baseProfile.scheduler().lowCostTargetChunksPerWorker() : low,
                                                med == null ? baseProfile.scheduler().mediumCostTargetChunksPerWorker() : med,
                                                high == null ? baseProfile.scheduler().highCostTargetChunksPerWorker() : high,
                                                scalar == null ? baseProfile.scheduler().minScalarChunkSize() : scalar,
                                                vector == null ? baseProfile.scheduler().minVectorChunkSize() : vector,
                                                reduction == null ? baseProfile.scheduler().minReductionChunkSize() : reduction,
                                                common == null ? baseProfile.scheduler().commonPoolLowCostMaxWorkPerWorker() : common
                                        );
                                        out.add(new RuntimeProfileCandidate(
                                                "scheduler=" + scheduler.lowCostTargetChunksPerWorker() + "/" + scheduler.mediumCostTargetChunksPerWorker()
                                                        + "/" + scheduler.highCostTargetChunksPerWorker(),
                                                new PlatformRuntimeProfile(
                                                        baseProfile.metadata(),
                                                        baseProfile.matmul(),
                                                        baseProfile.fused(),
                                                        baseProfile.elementwiseDispatch(),
                                                        baseProfile.reduction(),
                                                        scheduler,
                                                        baseProfile.materialization(),
                                                        baseProfile.numerics()
                                                ),
                                                Map.of(
                                                        "cpu.lowCostTargetChunksPerWorker", String.valueOf(scheduler.lowCostTargetChunksPerWorker()),
                                                        "cpu.mediumCostTargetChunksPerWorker", String.valueOf(scheduler.mediumCostTargetChunksPerWorker()),
                                                        "cpu.highCostTargetChunksPerWorker", String.valueOf(scheduler.highCostTargetChunksPerWorker())
                                                )
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator numericsPolicies(
            List<ApproxMode> approxModes,
            List<Boolean> exactTranscendentals
    ) {
        List<ApproxMode> safeModes = approxModes == null ? List.of() : List.copyOf(approxModes);
        List<Boolean> safeExact = exactTranscendentals == null ? List.of() : List.copyOf(exactTranscendentals);
        return (baseProfile, workload) -> {
            if (!usesGenericRuntimeFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("numerics=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (ApproxMode mode : safeModes) {
                for (Boolean exact : safeExact) {
                    NumericsPlatformProfile numerics = new NumericsPlatformProfile(
                            mode == null ? baseProfile.numerics().approxMode() : mode,
                            exact == null ? baseProfile.numerics().forceExactTranscendentals() : exact
                    );
                    out.add(new RuntimeProfileCandidate(
                            "numerics=" + numerics.approxMode() + "/" + numerics.forceExactTranscendentals(),
                            new PlatformRuntimeProfile(
                                    baseProfile.metadata(),
                                    baseProfile.matmul(),
                                    baseProfile.fused(),
                                    baseProfile.elementwiseDispatch(),
                                    baseProfile.reduction(),
                                    baseProfile.scheduler(),
                                    baseProfile.materialization(),
                                    numerics
                            ),
                            Map.of(
                                    "runtime.approximation.approxMode", numerics.approxMode().name(),
                                    "runtime.approximation.forceExactTranscendentals", String.valueOf(numerics.forceExactTranscendentals())
                            )
                    ));
                }
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator materializationThresholds(List<Integer> thresholds) {
        List<Integer> safe = thresholds == null ? List.of() : List.copyOf(thresholds);
        return (baseProfile, workload) -> {
            if (!usesGenericRuntimeFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("materialization=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer threshold : safe) {
                MaterializationPlatformProfile base = baseProfile.materialization();
                MaterializationPlatformProfile materialization = switch (baseProfile.dataType()) {
                    case FLOAT64 -> new MaterializationPlatformProfile(
                            base.contiguousMaterializeThreshold(),
                            threshold == null ? base.cheapF64MaterializeThreshold() : threshold,
                            base.cheapF32MaterializeThreshold(),
                            base.cheapBF16MaterializeThreshold(),
                            base.whereMaterializeThreshold()
                    );
                    case FLOAT32 -> new MaterializationPlatformProfile(
                            base.contiguousMaterializeThreshold(),
                            base.cheapF64MaterializeThreshold(),
                            threshold == null ? base.cheapF32MaterializeThreshold() : threshold,
                            base.cheapBF16MaterializeThreshold(),
                            base.whereMaterializeThreshold()
                    );
                    case BFLOAT16 -> new MaterializationPlatformProfile(
                            base.contiguousMaterializeThreshold(),
                            base.cheapF64MaterializeThreshold(),
                            base.cheapF32MaterializeThreshold(),
                            threshold == null ? base.cheapBF16MaterializeThreshold() : threshold,
                            base.whereMaterializeThreshold()
                    );
                    default -> base;
                };
                String key = switch (baseProfile.dataType()) {
                    case FLOAT64 -> "cpu.materialization.cheapF64Threshold";
                    case FLOAT32 -> "cpu.materialization.cheapF32Threshold";
                    case BFLOAT16 -> "cpu.materialization.cheapBF16Threshold";
                    default -> "cpu.materialization.threshold";
                };
                int selectedThreshold = switch (baseProfile.dataType()) {
                    case FLOAT64 -> materialization.cheapF64MaterializeThreshold();
                    case FLOAT32 -> materialization.cheapF32MaterializeThreshold();
                    case BFLOAT16 -> materialization.cheapBF16MaterializeThreshold();
                    default -> materialization.contiguousMaterializeThreshold();
                };
                out.add(new RuntimeProfileCandidate(
                        "materialization=" + selectedThreshold,
                        withMaterialization(baseProfile, materialization),
                        Map.of(key, String.valueOf(selectedThreshold))
                ));
            }
            return out;
        };
    }

    public static PlatformRuntimeProfileMutator whereMaterializationThresholds(List<Integer> thresholds) {
        List<Integer> safe = thresholds == null ? List.of() : List.copyOf(thresholds);
        return (baseProfile, workload) -> {
            if (!usesGenericRuntimeFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("where-materialization=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer threshold : safe) {
                MaterializationPlatformProfile base = baseProfile.materialization();
                MaterializationPlatformProfile materialization = new MaterializationPlatformProfile(
                        base.contiguousMaterializeThreshold(),
                        base.cheapF64MaterializeThreshold(),
                        base.cheapF32MaterializeThreshold(),
                        base.cheapBF16MaterializeThreshold(),
                        threshold == null ? base.whereMaterializeThreshold() : threshold
                );
                out.add(new RuntimeProfileCandidate(
                        "where-materialization=" + materialization.whereMaterializeThreshold(),
                        withMaterialization(baseProfile, materialization),
                        Map.of("cpu.materialization.whereThreshold", String.valueOf(materialization.whereMaterializeThreshold()))
                ));
            }
            return out;
        };
    }

    private static boolean usesMatmulFamily(WorkloadKind kind) {
        return kind == WorkloadKind.MATMUL
                || kind == WorkloadKind.MLP_CLASSIFICATION
                || kind == WorkloadKind.ABC_SEQUENCE_MATMUL
                || kind == WorkloadKind.TRANSFORMER_HOT_PATH;
    }

    private static boolean usesConv2dFamily(WorkloadKind kind) {
        return kind == WorkloadKind.CONV2D;
    }

    private static boolean usesGenericRuntimeFamily(WorkloadKind kind) {
        return kind == WorkloadKind.NORMALIZATION
                || kind == WorkloadKind.TRANSFORMER_HOT_PATH
                || kind == WorkloadKind.GENERIC
                || kind == WorkloadKind.LOSS
                || kind == WorkloadKind.MATMUL
                || kind == WorkloadKind.MLP_CLASSIFICATION
                || kind == WorkloadKind.ABC_SEQUENCE_MATMUL;
    }

    private static FusedPlatformProfile withFusedAsmVectorWidth(
            FusedPlatformProfile base,
            FusedDispatchFamily family,
            int width
    ) {
        return switch (family) {
            case CHEAP_CONTIGUOUS -> new FusedPlatformProfile(
                    base.fusedCheapVectorMinSize(),
                    base.fusedTranscendentalVectorMinSize(),
                    base.fusedCheapParallelMinSize(),
                    base.fusedTranscendentalParallelMinSize(),
                    width,
                    base.fusedCheapStridedAsmVectorWidth(),
                    base.fusedNonCheapContiguousAsmVectorWidth(),
                    base.fusedNonCheapStridedAsmVectorWidth()
            );
            case CHEAP_STRIDED -> new FusedPlatformProfile(
                    base.fusedCheapVectorMinSize(),
                    base.fusedTranscendentalVectorMinSize(),
                    base.fusedCheapParallelMinSize(),
                    base.fusedTranscendentalParallelMinSize(),
                    base.fusedCheapContiguousAsmVectorWidth(),
                    width,
                    base.fusedNonCheapContiguousAsmVectorWidth(),
                    base.fusedNonCheapStridedAsmVectorWidth()
            );
            case NON_CHEAP_CONTIGUOUS -> new FusedPlatformProfile(
                    base.fusedCheapVectorMinSize(),
                    base.fusedTranscendentalVectorMinSize(),
                    base.fusedCheapParallelMinSize(),
                    base.fusedTranscendentalParallelMinSize(),
                    base.fusedCheapContiguousAsmVectorWidth(),
                    base.fusedCheapStridedAsmVectorWidth(),
                    width,
                    base.fusedNonCheapStridedAsmVectorWidth()
            );
            case NON_CHEAP_STRIDED -> new FusedPlatformProfile(
                    base.fusedCheapVectorMinSize(),
                    base.fusedTranscendentalVectorMinSize(),
                    base.fusedCheapParallelMinSize(),
                    base.fusedTranscendentalParallelMinSize(),
                    base.fusedCheapContiguousAsmVectorWidth(),
                    base.fusedCheapStridedAsmVectorWidth(),
                    base.fusedNonCheapContiguousAsmVectorWidth(),
                    width
            );
        };
    }

    private static int currentFusedAsmVectorWidth(FusedPlatformProfile profile, FusedDispatchFamily family) {
        return switch (family) {
            case CHEAP_CONTIGUOUS -> profile.fusedCheapContiguousAsmVectorWidth();
            case CHEAP_STRIDED -> profile.fusedCheapStridedAsmVectorWidth();
            case NON_CHEAP_CONTIGUOUS -> profile.fusedNonCheapContiguousAsmVectorWidth();
            case NON_CHEAP_STRIDED -> profile.fusedNonCheapStridedAsmVectorWidth();
        };
    }

    private static String fusedAsmVectorWidthKey(FusedDispatchFamily family) {
        return switch (family) {
            case CHEAP_CONTIGUOUS -> "cpu.fusedCheapContiguousAsmVectorWidth";
            case CHEAP_STRIDED -> "cpu.fusedCheapStridedAsmVectorWidth";
            case NON_CHEAP_CONTIGUOUS -> "cpu.fusedNonCheapContiguousAsmVectorWidth";
            case NON_CHEAP_STRIDED -> "cpu.fusedNonCheapStridedAsmVectorWidth";
        };
    }

    private static PlatformRuntimeProfile withConv2d(PlatformRuntimeProfile baseProfile, Conv2dPlatformProfile conv2d) {
        return new PlatformRuntimeProfile(
                baseProfile.metadata(),
                baseProfile.matmul(),
                conv2d,
                baseProfile.fused(),
                baseProfile.elementwiseDispatch(),
                baseProfile.reduction(),
                baseProfile.scheduler(),
                baseProfile.materialization(),
                baseProfile.numerics()
        );
    }

    private static PlatformRuntimeProfile withMaterialization(PlatformRuntimeProfile baseProfile, MaterializationPlatformProfile materialization) {
        return new PlatformRuntimeProfile(
                baseProfile.metadata(),
                baseProfile.matmul(),
                baseProfile.conv2d(),
                baseProfile.fused(),
                baseProfile.elementwiseDispatch(),
                baseProfile.reduction(),
                baseProfile.scheduler(),
                materialization,
                baseProfile.numerics()
        );
    }
}
