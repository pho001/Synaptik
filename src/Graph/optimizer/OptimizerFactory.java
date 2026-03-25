package Graph.optimizer;

import Backend.ComputeEngine;
import Benchmark.OptimizationStage;
import Benchmark.OptimizerBuilder;
import Benchmark.OptimizerCandidate;
import Benchmark.OptimizerCandidateFactory;
import Benchmark.OptimizerProfileIO;
import Benchmark.TuningKnobs;
import Graph.optimizer.rules.AlgebraicRewritingRule;
import Graph.optimizer.rules.CommonSubexpressionEliminationRule;
import Graph.optimizer.rules.FuseElementWiseRule;
import Graph.optimizer.rules.MemoryOptimizerRule;

import java.nio.file.Path;
import java.util.List;

public final class OptimizerFactory {
    private static final Path PROFILE_PATH = Path.of("config", "optimizer-profile.json");
    private static final Path HW_PROFILE_PATH = Path.of("config", "optimizer-hw-profiles.tsv");
    private static final Path AUTOTUNE_BEST_TRAINING_PATH = Path.of("build", "optimizer-autotune", "best-profile-training.json");
    private static final Path AUTOTUNE_BEST_INFERENCE_PATH = Path.of("build", "optimizer-autotune", "best-profile-inference.json");

    private OptimizerFactory() {}

    public static OptimizationRule addAlgebraicRewritingRule() {
        return new AlgebraicRewritingRule();
    }

    public static OptimizationRule addCommonSubexpressionEliminationRule() {
        return new CommonSubexpressionEliminationRule(true);
    }

    public static OptimizationRule addCommonSubexpressionEliminationRuleAggressive() {
        return new CommonSubexpressionEliminationRule(false);
    }

    public static OptimizationRule addFuseElementWise() {
        return new FuseElementWiseRule(true);
    }

    public static OptimizationRule addFuseElementWiseAggressive() {
        return new FuseElementWiseRule(false);
    }

    public static OptimizationRule addMemoryOptimizerRule() {
        return new MemoryOptimizerRule();
    }

    // Bezpečný režim pro training/autograd
    public static GraphOptimizer createTrainingOptimizer() {
        GraphOptimizer optimizer = new GraphOptimizer();
        optimizer.addRule(addAlgebraicRewritingRule());
        optimizer.addRule(addCommonSubexpressionEliminationRule());
        optimizer.addRule(addFuseElementWise());
        optimizer.addRule(addMemoryOptimizerRule());
        return optimizer;
    }

    // Agresivní režim pro inference benchmarky
    public static GraphOptimizer createInferencePerformanceOptimizer() {
        try {
            OptimizerCandidate baseInference = findCandidateOrDefault(
                    OptimizerCandidateFactory.defaultCandidates(),
                    "INFERENCE_PERF",
                    List.of(OptimizationStage.AR, OptimizationStage.CSE, OptimizationStage.FUSE, OptimizationStage.MEM),
                    TuningKnobs.inferencePerfDefaults()
            );
            OptimizerCandidate effective = OptimizerProfileIO.loadRecommendedOverrideOrDefault(
                    AUTOTUNE_BEST_INFERENCE_PATH,
                    baseInference
            );
            effective = OptimizerProfileIO.loadArchitectureDefaultOverrideOrDefault(
                    "INFERENCE",
                    effective
            );
            effective = OptimizerProfileIO.loadHardwareOverrideOrDefault(
                    HW_PROFILE_PATH,
                    OptimizerProfileIO.hardwareBucketKey(),
                    "INFERENCE",
                    effective
            );
            ComputeEngine.setCpuKernelConfig(effective.knobs().kernelConfig().cpu());
            return OptimizerBuilder.build(effective);
        } catch (Exception ignored) {
            GraphOptimizer optimizer = new GraphOptimizer();
            optimizer.addRule(addAlgebraicRewritingRule());
            optimizer.addRule(addCommonSubexpressionEliminationRuleAggressive());
            optimizer.addRule(addFuseElementWiseAggressive());
            optimizer.addRule(addMemoryOptimizerRule());
            return optimizer;
        }
    }

    // Kompatibilita se stávajícím benchmarkem
    public static GraphOptimizer createRecommendedTrainingOptimizer() {
        try {
            OptimizerCandidate baseRecommended = findCandidateOrDefault(
                    OptimizerCandidateFactory.defaultCandidates(),
                    "RECOMMENDED",
                    List.of(OptimizationStage.AR, OptimizationStage.CSE, OptimizationStage.MEM),
                    TuningKnobs.trainingDefaults()
            );
            TuningKnobs tuned = OptimizerProfileIO.loadKnobsOrDefault(PROFILE_PATH, baseRecommended.knobs());
            OptimizerCandidate profileCandidate = new OptimizerCandidate(
                    baseRecommended.name(),
                    baseRecommended.stageOrder(),
                    tuned
            );
            OptimizerCandidate effective = OptimizerProfileIO.loadRecommendedOverrideOrDefault(
                    AUTOTUNE_BEST_TRAINING_PATH,
                    profileCandidate
            );
            effective = OptimizerProfileIO.loadArchitectureDefaultOverrideOrDefault(
                    "TRAINING",
                    effective
            );
            effective = OptimizerProfileIO.loadHardwareOverrideOrDefault(
                    HW_PROFILE_PATH,
                    OptimizerProfileIO.hardwareBucketKey(),
                    "TRAINING",
                    effective
            );
            ComputeEngine.setCpuKernelConfig(effective.knobs().kernelConfig().cpu());
            return OptimizerBuilder.build(effective);
        } catch (Exception ignored) {
            return createTrainingOptimizer();
        }
    }

    private static OptimizerCandidate findCandidateOrDefault(
            List<OptimizerCandidate> candidates,
            String candidateName,
            List<OptimizationStage> defaultStages,
            TuningKnobs defaultKnobs
    ) {
        for (OptimizerCandidate c : candidates) {
            if (candidateName.equals(c.name())) {
                return c;
            }
        }
        return new OptimizerCandidate(candidateName, defaultStages, defaultKnobs);
    }
}
