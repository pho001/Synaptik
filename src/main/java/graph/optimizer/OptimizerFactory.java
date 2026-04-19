package graph.optimizer;

import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import graph.SemanticForwardCanonicalizer;
import graph.optimizer.rewrite.RewriteRule;
import graph.optimizer.rules.CommonSubexpressionEliminationRule;
import graph.optimizer.rules.FuseElementWiseRule;
import graph.optimizer.rules.MemoryOptimizerRule;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class OptimizerFactory {
    private static final int DEFAULT_FIXPOINT_ROUNDS = 4;

    private OptimizerFactory() {}

    public static GraphOptimizer create(OptimizerConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        List<OptimizationRule> iterative = new ArrayList<>(config.stageOrder().size());
        List<OptimizationRule> terminal = new ArrayList<>(1);
        for (OptimizerStage stage : config.stageOrder()) {
            OptimizationRule rule = createRule(stage, config);
            if (stage == OptimizerStage.MEM) {
                terminal.add(rule);
            } else {
                iterative.add(rule);
            }
        }
        List<OptimizationRule> rules = new ArrayList<>(iterative.size() + terminal.size());
        rules.addAll(iterative);
        rules.addAll(terminal);
        return new GraphOptimizer(rules, iterative.size(), DEFAULT_FIXPOINT_ROUNDS);
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
            case FUSE -> new FuseElementWiseRule(config.fuse());
            case MEM -> new MemoryOptimizerRule(graph.optimizer.memory.MemoryPlannerPolicy.fromConfig(config.memory()));
        };
    }
}
