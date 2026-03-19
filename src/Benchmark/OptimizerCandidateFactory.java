package Benchmark;

import Config.backend.CpuKernelConfig;
import Config.backend.CudaKernelConfig;
import Config.backend.KernelTuningConfig;
import Config.backend.OpenClKernelConfig;
import Config.optimizer.FuseConfig;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;

public final class OptimizerCandidateFactory {
    private OptimizerCandidateFactory() {}

    public static List<OptimizerCandidate> defaultCandidates() {
        List<OptimizerCandidate> out = new ArrayList<>();
        out.add(new OptimizerCandidate("NO_OPT", List.of(), TuningKnobs.trainingDefaults()));
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
        List<OptimizerCandidate> out = new ArrayList<>();
        List<OptimizationStage> all = new ArrayList<>(EnumSet.allOf(OptimizationStage.class));

        int nameId = 0;
        for (int mask = 0; mask < (1 << all.size()); mask++) {
            List<OptimizationStage> subset = new ArrayList<>();
            for (int i = 0; i < all.size(); i++) {
                if ((mask & (1 << i)) != 0) subset.add(all.get(i));
            }
            if (subset.isEmpty()) {
                out.add(new OptimizerCandidate("SEARCH_" + (nameId++), List.of(), knobs));
                continue;
            }

            List<List<OptimizationStage>> permutations = new ArrayList<>();
            permute(subset, 0, permutations);
            for (List<OptimizationStage> p : permutations) {
                out.add(new OptimizerCandidate("SEARCH_" + (nameId++), p, knobs));
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
        List<CpuKernelConfig> cpuDispatchProfiles = List.of(
                new CpuKernelConfig(4, 32, 32, 32, 256, 50_000, 0, 2, 2_048),
                new CpuKernelConfig(4, 32, 32, 32, 512, 100_000, 0, 4, 4_096),
                new CpuKernelConfig(4, 32, 32, 32, 2_048, 250_000, 0, 8, 8_192),
                new CpuKernelConfig(4, 64, 32, 32, 256, 1_000_000, 0, 4, 4_096),
                new CpuKernelConfig(4, 32, 32, 32, 1_000_000_000, 2_000_000, 0, 4, 4_096),
                new CpuKernelConfig(4, 32, 32, 32, 1_000_000_000, 1_000_000_000, 0, 4, 4_096)
        );

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
}
