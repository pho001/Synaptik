package graph.optimizer;

import config.compile.GraphOptimizationConfig;
import config.compile.SemanticCanonicalizationConfig;
import graph.optimizer.cf.ConstantFoldingRule;
import graph.optimizer.cleanup.CleanupFixpointRule;
import graph.optimizer.cse.CommonSubexpressionEliminationRule;
import graph.optimizer.dce.DeadCodeEliminationRule;
import graph.SemanticForwardCanonicalizer;
import graph.optimizer.rewrite.LoweringRule;
import graph.optimizer.rewrite.RewriteRule;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Builds backend-neutral graph optimization pipelines.
 */
public final class OptimizerFactory {
    private OptimizerFactory() {}

    /**
     * Creates a graph optimizer for backend-neutral graph optimization only.
     *
     * @param config graph optimization configuration
     * @return optimizer containing only graph rewrite/cleanup/lowering rules
     */
    public static GraphOptimizer create(GraphOptimizationConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        List<OptimizationRule> rules = new ArrayList<>();
        List<OptimizationRule> cleanup = new ArrayList<>(4);
        if (config.algebraicRewrite()) {
            cleanup.add(new RewriteRule(config.rewrite()));
        }
        if (config.constantFolding()) {
            cleanup.add(new ConstantFoldingRule());
        }
        if (config.commonSubexpressionElimination()) {
            cleanup.add(new CommonSubexpressionEliminationRule(config.cse()));
        }
        if (config.deadCodeElimination()) {
            cleanup.add(new DeadCodeEliminationRule());
        }
        flushCleanup(rules, cleanup);
        if (config.optionalLowering()) {
            rules.add(new LoweringRule(config.rewrite()));
        }
        return new GraphOptimizer(rules);
    }

    /**
     * Creates the semantic forward canonicalizer for the compile contract.
     *
     * @param config semantic canonicalization configuration
     * @return canonicalizer, or {@code null} when semantic canonicalization is explicitly disabled
     */
    public static SemanticForwardCanonicalizer createSemanticForwardCanonicalizer(SemanticCanonicalizationConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        return config.enabled() ? new SemanticForwardCanonicalizer(config.rewrite()) : null;
    }

    private static void flushCleanup(List<OptimizationRule> rules, List<OptimizationRule> cleanup) {
        if (cleanup.isEmpty()) {
            return;
        }
        if (cleanup.size() == 1) {
            rules.add(cleanup.getFirst());
        } else {
            rules.add(new CleanupFixpointRule(cleanup));
        }
        cleanup.clear();
    }

}
