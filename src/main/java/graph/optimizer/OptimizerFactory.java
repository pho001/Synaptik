package graph.optimizer;

import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import graph.optimizer.cse.CommonSubexpressionEliminationRule;
import graph.optimizer.memory.MemoryOptimizerRule;
import graph.optimizer.partition.PartitionIntentRule;
import graph.SemanticForwardCanonicalizer;
import graph.optimizer.region.RegionOptimizationRule;
import graph.optimizer.rewrite.RewriteRule;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class OptimizerFactory {
    private OptimizerFactory() {}

    public static GraphOptimizer create(OptimizerConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        List<OptimizationRule> rules = new ArrayList<>(config.stageOrder().size());
        for (OptimizerStage stage : config.stageOrder()) {
            rules.add(createRule(stage, config));
        }
        return new GraphOptimizer(rules);
    }

    public static SemanticForwardCanonicalizer createSemanticForwardCanonicalizer(OptimizerConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        return new SemanticForwardCanonicalizer(config.rewrite());
    }

    public static List<OptimizationRule> createRules(OptimizerConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        List<OptimizationRule> rules = new ArrayList<>(config.stageOrder().size());
        for (OptimizerStage stage : config.stageOrder()) {
            rules.add(createRule(stage, config));
        }
        return List.copyOf(rules);
    }

    private static OptimizationRule createRule(OptimizerStage stage, OptimizerConfig config) {
        return switch (stage) {
            case AR -> new RewriteRule(config.rewrite());
            case CSE -> new CommonSubexpressionEliminationRule(config.cse());
            case PART -> new PartitionIntentRule(config.partition());
            case FUSE -> new RegionOptimizationRule(config.fuse());
            case MEM -> new MemoryOptimizerRule(graph.optimizer.memory.MemoryPlannerPolicy.fromConfig(config.memory()));
        };
    }
}
