package benchmark;

import config.backend.CpuKernelConfig;
import config.backend.CudaKernelConfig;
import config.backend.KernelTuningConfig;
import config.backend.OpenClKernelConfig;
import config.backend.AttentionMatMulPolicy;
import config.backend.VectorPolicy;
import config.optimizer.FuseConfig;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;

public final class OptimizerCandidateFactory {
    private static final boolean AUTOTUNE_INCLUDE_BLAS_POLICIES =
            Boolean.parseBoolean(System.getProperty("benchmark.autotuneIncludeBlasPolicies", "false"));

    private OptimizerCandidateFactory() {}

    public static List<OptimizerCandidate> defaultCandidates() {
        List<OptimizerCandidate> out = new ArrayList<>();
        out.add(new OptimizerCandidate("NO_OPT", List.of(), TuningKnobs.trainingDefaults()));
        out.add(new OptimizerCandidate("MEM_ONLY", List.of(OptimizationStage.MEM), TuningKnobs.trainingDefaults()));
        out.add(new OptimizerCandidate("AR", List.of(OptimizationStage.AR), TuningKnobs.trainingDefaults()));
        out.add(new OptimizerCandidate("AR+CSE", List.of(OptimizationStage.AR, OptimizationStage.CSE), TuningKnobs.trainingDefaults()));
        out.add(new OptimizerCandidate("AR+CSE+MEM", List.of(OptimizationStage.AR, OptimizationStage.CSE, OptimizationStage.MEM), TuningKnobs.trainingDefaults()));
        out.add(new OptimizerCandidate("AR+CSE+FUSE", List.of(OptimizationStage.AR, OptimizationStage.CSE, OptimizationStage.FUSE), TuningKnobs.trainingDefaults()));
        out.add(new OptimizerCandidate("AR+CSE+FUSE+MEM", List.of(OptimizationStage.AR, OptimizationStage.CSE, OptimizationStage.FUSE, OptimizationStage.MEM), TuningKnobs.trainingDefaults()));
        out.add(new OptimizerCandidate("CSE+FUSE+MEM", List.of(OptimizationStage.CSE, OptimizationStage.FUSE, OptimizationStage.MEM), TuningKnobs.trainingDefaults()));
        out.add(new OptimizerCandidate("FUSE", List.of(OptimizationStage.FUSE), TuningKnobs.trainingDefaults()));
        out.add(new OptimizerCandidate("RECOMMENDED", List.of(OptimizationStage.AR, OptimizationStage.CSE, OptimizationStage.MEM), TuningKnobs.trainingDefaults()));
        out.add(new OptimizerCandidate("INFERENCE_PERF", List.of(OptimizationStage.AR, OptimizationStage.CSE, OptimizationStage.FUSE, OptimizationStage.MEM), TuningKnobs.inferencePerfDefaults()));
        return out;
    }

    /**
     * Generates search space for stage combinations and stage orders.
     * Useful for future auto-tuning beyond fixed presets.
     */
    public static List<OptimizerCandidate> generateCombinationsAndOrders(TuningKnobs knobs) {
        return generateCombinationsMemLast(knobs);
    }

    private static List<OptimizerCandidate> generateCombinationsMemLast(TuningKnobs knobs) {
        List<OptimizerCandidate> out = new ArrayList<>();
        List<OptimizationStage> nonMem = new ArrayList<>(EnumSet.allOf(OptimizationStage.class));
        nonMem.remove(OptimizationStage.MEM);

        int nameId = 0;
        for (int mask = 0; mask < (1 << nonMem.size()); mask++) {
            List<OptimizationStage> subset = new ArrayList<>();
            for (int i = 0; i < nonMem.size(); i++) {
                if ((mask & (1 << i)) != 0) {
                    subset.add(nonMem.get(i));
                }
            }

            List<List<OptimizationStage>> permutations = new ArrayList<>();
            if (subset.isEmpty()) {
                permutations.add(List.of());
            } else {
                permute(subset, 0, permutations);
            }

            for (List<OptimizationStage> p : permutations) {
                out.add(new OptimizerCandidate("SEARCH_" + (nameId++), p, knobs));

                List<OptimizationStage> withMemLast = new ArrayList<>(p.size() + 1);
                withMemLast.addAll(p);
                withMemLast.add(OptimizationStage.MEM);
                out.add(new OptimizerCandidate("SEARCH_" + (nameId++), withMemLast, knobs));
            }
        }
        return out;
    }

    public static List<OptimizerCandidate> autotuneCandidates() {
        List<OptimizerCandidate> out = new ArrayList<>();
        int nameId = 0;

        List<TuningKnobs> knobGrid = new ArrayList<>(List.of(
                new TuningKnobs(true, FuseConfig.trainingDefaults(), KernelTuningConfig.defaultsTraining()),
                new TuningKnobs(true, FuseConfig.trainingDefaults().withPreserveSharedExpensiveNodes(false), KernelTuningConfig.defaultsTraining()),
                new TuningKnobs(
                        true,
                        new FuseConfig(64, 0.0, 0.25, 0.20, 1.00, 0.30, true),
                        KernelTuningConfig.defaultsTraining()
                ),
                new TuningKnobs(
                        true,
                        new FuseConfig(32, 0.55, 0.25, 0.20, 1.00, 0.30, true),
                        KernelTuningConfig.defaultsTraining()
                ),
                new TuningKnobs(false, FuseConfig.inferencePerfDefaults().withPreserveSharedExpensiveNodes(true), KernelTuningConfig.defaultsInference()),
                new TuningKnobs(false, FuseConfig.inferencePerfDefaults(), KernelTuningConfig.defaultsInference()),
                new TuningKnobs(
                        false,
                        new FuseConfig(96, 0.0, 0.50, 0.10, 0.50, 0.35, false),
                        KernelTuningConfig.defaultsInference()
                ),
                new TuningKnobs(
                        false,
                        new FuseConfig(32, 0.60, 0.50, 0.10, 0.50, 0.35, false),
                        KernelTuningConfig.defaultsInference()
                ),
                new TuningKnobs(
                        true,
                        new FuseConfig(80, 0.85, 0.50, 0.20, 1.00, 0.35, true),
                        new KernelTuningConfig(
                                new CpuKernelConfig(4, 16, 16, 16),
                                new CudaKernelConfig(4, 32, 32, 16),
                                new OpenClKernelConfig(2, 16, 16, 16)
                        )
                ),
                new TuningKnobs(
                        false,
                        new FuseConfig(96, 0.55, 0.50, 0.10, 0.50, 0.40, false),
                        new KernelTuningConfig(
                                new CpuKernelConfig(4, 32, 32, 32, 512, 100_000),
                                new CudaKernelConfig(8, 32, 32, 32),
                                new OpenClKernelConfig(4, 32, 32, 16)
                        )
                )
        ));

        // CPU dispatch tuning grid (threshold-based mode selection only).
        List<CpuKernelConfig> cpuDispatchProfiles = new ArrayList<>();
        List<CpuKernelConfig> cpuDispatchBaseProfiles = List.of(
                new CpuKernelConfig(4, 32, 32, 32, 256, 50_000, 0, 2, 2_048),
                new CpuKernelConfig(4, 32, 32, 32, 512, 100_000, 0, 4, 4_096),
                new CpuKernelConfig(4, 32, 32, 32, 2_048, 250_000, 0, 8, 8_192),
                new CpuKernelConfig(4, 64, 32, 32, 256, 1_000_000, 0, 4, 4_096),
                new CpuKernelConfig(4, 32, 32, 32, 1_000_000_000, 2_000_000, 0, 4, 4_096),
                new CpuKernelConfig(4, 32, 32, 32, 1_000_000_000, 1_000_000_000, 0, 4, 4_096)
        );
        int[] contiguousMaterializeThresholds = new int[]{0, 4_096, 16_384, 65_536, 262_144, 1_000_000_000};
        double[] lowCostNsPerElemThresholds = new double[]{0.5, 1.0, 2.0, 4.0};
        int[] matMulParallelMinSizes = new int[]{100_000, 500_000, 2_000_000, 8_000_000};
        VectorPolicy[][] vectorPolicyProfiles = new VectorPolicy[][]{
                {VectorPolicy.AUTO, VectorPolicy.AUTO, VectorPolicy.AUTO},
                {VectorPolicy.AUTO, VectorPolicy.FORCE_OFF, VectorPolicy.AUTO},
                {VectorPolicy.FORCE_ON, VectorPolicy.FORCE_OFF, VectorPolicy.AUTO}
        };
        AttentionMatMulPolicy[] attentionPolicies = new AttentionMatMulPolicy[]{
                AttentionMatMulPolicy.AUTO,
                AttentionMatMulPolicy.FORCE_OFF,
                AttentionMatMulPolicy.FORCE_ON
        };
        for (CpuKernelConfig base : cpuDispatchBaseProfiles) {
            for (int threshold : contiguousMaterializeThresholds) {
                for (double lowCostThreshold : lowCostNsPerElemThresholds) {
                    for (int matMulParMin : matMulParallelMinSizes) {
                        for (VectorPolicy[] vp : vectorPolicyProfiles) {
                            for (AttentionMatMulPolicy attentionPolicy : attentionPolicies) {
                            cpuDispatchProfiles.add(new CpuKernelConfig(
                                    base.loopUnrollFactor(),
                                    base.matMulTileM(),
                                    base.matMulTileN(),
                                    base.matMulTileK(),
                                    base.vectorMinSize(),
                                    base.parallelMinSize(),
                                    base.parallelism(),
                                    base.chunksPerWorker(),
                                    base.minChunkSize(),
                                    threshold,
                                    base.sumAccuracyMode(),
                                    lowCostThreshold,
                                    vp[0],
                                    vp[1],
                                    vp[2],
                                    matMulParMin,
                                    attentionPolicy
                            ));
                            }
                        }
                    }
                }
            }
        }

        for (CpuKernelConfig cpu : cpuDispatchProfiles) {
            knobGrid.add(new TuningKnobs(
                    true,
                    FuseConfig.trainingDefaults(),
                    new KernelTuningConfig(cpu, CudaKernelConfig.defaultsTraining(), OpenClKernelConfig.defaultsTraining())
            ));
            knobGrid.add(new TuningKnobs(
                    false,
                    FuseConfig.inferencePerfDefaults(),
                    new KernelTuningConfig(cpu, CudaKernelConfig.defaultsInference(), OpenClKernelConfig.defaultsInference())
            ));
        }

        if (AUTOTUNE_INCLUDE_BLAS_POLICIES) {
            knobGrid = expandWithBlasPolicies(knobGrid);
        }

        for (TuningKnobs knobs : knobGrid) {
            for (OptimizerCandidate base : generateCombinationsAndOrders(knobs)) {
                out.add(new OptimizerCandidate("AUTO_" + (nameId++), base.stageOrder(), knobs));
            }
        }

        return out;
    }

    private static void permute(List<OptimizationStage> arr, int idx, List<List<OptimizationStage>> out) {
        if (idx == arr.size()) {
            out.add(List.copyOf(arr));
            return;
        }
        for (int i = idx; i < arr.size(); i++) {
            Collections.swap(arr, idx, i);
            permute(arr, idx + 1, out);
            Collections.swap(arr, idx, i);
        }
    }

    private static List<TuningKnobs> expandWithBlasPolicies(List<TuningKnobs> base) {
        List<TuningKnobs> out = new ArrayList<>(base.size() * 3);
        for (TuningKnobs knobs : base) {
            out.add(knobs);
            out.add(knobs.withBlasPolicy("OPENBLAS_FFM", 2_000_000L, true, 3.0d));
            out.add(knobs.withBlasPolicy("OPENBLAS_FFM", 4_000_000L, true, 2.0d));
        }
        return out;
    }
}
