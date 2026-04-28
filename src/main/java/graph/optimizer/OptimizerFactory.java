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

/**
 * Builds optimizer pipelines from {@link OptimizerConfig}.
 *
 * <p>The factory is the single place that maps configured {@link OptimizerStage} values to rule implementations:
 * rewrite ({@code AR}), common subexpression elimination ({@code CSE}), partition planning ({@code PART}), region
 * fusion ({@code FUSE}), and memory planning ({@code MEM}). The returned rules are ordered exactly as requested by the
 * configuration.
 */
public final class OptimizerFactory {
    private OptimizerFactory() {}

    /**
     * Creates a graph optimizer for the configured stage order.
     *
     * @param config optimizer configuration
     * @return optimizer with one rule per configured stage
     */
    public static GraphOptimizer create(OptimizerConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        List<OptimizationRule> rules = new ArrayList<>(config.stageOrder().size());
        for (OptimizerStage stage : config.stageOrder()) {
            rules.add(createRule(stage, config));
        }
        return new GraphOptimizer(rules);
    }

    /**
     * Creates the semantic forward canonicalizer associated with the rewrite configuration.
     *
     * @param config optimizer configuration
     * @return canonicalizer used before full graph compilation
     */
    public static SemanticForwardCanonicalizer createSemanticForwardCanonicalizer(OptimizerConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        return new SemanticForwardCanonicalizer(config.rewrite());
    }

    /**
     * Creates optimizer rules without wrapping them in a {@link GraphOptimizer}.
     *
     * @param config optimizer configuration
     * @return immutable rules in configured stage order
     */
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
            case PART -> new PartitionIntentRule(config.partition(), config.offload(), config.cpuRegion());
            case FUSE -> new RegionOptimizationRule(config.fuse(), config.cpuFusion());
            case MEM -> new MemoryOptimizerRule(graph.optimizer.memory.MemoryPlannerPolicy.fromConfig(config.memory()));
        };
    }
}
