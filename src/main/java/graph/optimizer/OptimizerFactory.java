package graph.optimizer;

import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import graph.optimizer.rules.CommonSubexpressionEliminationRule;
import graph.optimizer.rules.FuseElementWiseRule;
import graph.optimizer.rules.MemoryOptimizerRule;
import graph.optimizer.rules.RewriteRule;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class OptimizerFactory {
    private OptimizerFactory() {}

    public static GraphOptimizer create(OptimizerConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        return new GraphOptimizer(createRules(config));
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
            case AR -> new RewriteRule();
            case CSE -> new CommonSubexpressionEliminationRule(config.cse());
            case FUSE -> new FuseElementWiseRule(config.fuse());
            case MEM -> new MemoryOptimizerRule(graph.optimizer.memory.MemoryPlannerPolicy.fromConfig(config.memory()));
        };
    }
}
