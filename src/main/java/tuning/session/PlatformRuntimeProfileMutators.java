package tuning.session;

import backend.ApproxMode;
import config.profile.ElementwiseDispatchPlatformProfile;
import config.profile.FusedPlatformProfile;
import config.profile.MaterializationPlatformProfile;
import config.profile.NumericsPlatformProfile;
import config.profile.MatmulPlatformProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.ReductionPlatformProfile;
import config.profile.SchedulerPlatformProfile;
import tuning.workload.WorkloadKind;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class PlatformRuntimeProfileMutators {
    private PlatformRuntimeProfileMutators() {
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
                            baseProfile.matmul().matMulParallelMinSize()
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

    public static PlatformRuntimeProfileMutator blasThreads(List<Integer> threadCounts) {
        List<Integer> safe = threadCounts == null ? List.of() : List.copyOf(threadCounts);
        return (baseProfile, workload) -> {
            if (!usesMatmulFamily(workload.kind())) {
                return List.of(new RuntimeProfileCandidate("blasThreads=current", baseProfile, Map.of()));
            }
            List<RuntimeProfileCandidate> out = new ArrayList<>();
            for (Integer threads : safe) {
                MatmulPlatformProfile matmul = new MatmulPlatformProfile(
                        baseProfile.matmul().blasProvider(),
                        baseProfile.matmul().blasMatmulMinWork(),
                        threads == null ? baseProfile.matmul().blasThreads() : threads,
                        baseProfile.matmul().f32RequireMgeK(),
                        baseProfile.matmul().f32MaxNOverK(),
                        baseProfile.matmul().loopUnrollFactor(),
                        baseProfile.matmul().matMulTileM(),
                        baseProfile.matmul().matMulTileN(),
                        baseProfile.matmul().matMulTileK(),
                        baseProfile.matmul().matMulParallelMinSize()
                );
                out.add(new RuntimeProfileCandidate(
                        "blasThreads=" + matmul.blasThreads(),
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
                        Map.of("runtime.blas.threads", String.valueOf(matmul.blasThreads()))
                ));
            }
            return out;
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
                        threshold == null ? baseProfile.matmul().matMulParallelMinSize() : threshold
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
                                    baseProfile.fused().fusedAsmVectorWidth()
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
                        width == null ? baseProfile.fused().fusedAsmVectorWidth() : width
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
                MaterializationPlatformProfile materialization = new MaterializationPlatformProfile(
                        threshold == null ? baseProfile.materialization().contiguousMaterializeThreshold() : threshold
                );
                out.add(new RuntimeProfileCandidate(
                        "materialization=" + materialization.contiguousMaterializeThreshold(),
                        new PlatformRuntimeProfile(
                                baseProfile.metadata(),
                                baseProfile.matmul(),
                                baseProfile.fused(),
                                baseProfile.elementwiseDispatch(),
                                baseProfile.reduction(),
                                baseProfile.scheduler(),
                                materialization,
                                baseProfile.numerics()
                        ),
                        Map.of("cpu.contiguousMaterializeThreshold", String.valueOf(materialization.contiguousMaterializeThreshold()))
                ));
            }
            return out;
        };
    }

    private static boolean usesMatmulFamily(WorkloadKind kind) {
        return kind == WorkloadKind.MATMUL
                || kind == WorkloadKind.MLP_CLASSIFICATION
                || kind == WorkloadKind.ABC_SEQUENCE_MATMUL
                || kind == WorkloadKind.CONV2D
                || kind == WorkloadKind.TRANSFORMER_HOT_PATH;
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
}
