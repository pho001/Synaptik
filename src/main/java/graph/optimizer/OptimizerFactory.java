package graph.optimizer;

import config.compile.GraphOptimizationConfig;
import config.compile.SemanticCanonicalizationConfig;
import config.optimizer.Conv2dLoweringMode;
import config.optimizer.RewriteConfig;
import graph.optimizer.simplify.ConstantFoldingRule;
import graph.optimizer.simplify.SimplificationFixpointRule;
import graph.optimizer.simplify.CommonSubexpressionEliminationRule;
import graph.optimizer.simplify.DeadCodeEliminationRule;
import graph.SemanticForwardCanonicalizer;
import graph.optimizer.rewrite.algebraic.AlgebraicSimplificationRule;
import graph.optimizer.rewrite.canonical.PiecewiseCanonicalizationRule;
import graph.optimizer.rewrite.lowering.Conv2dDagLoweringRule;
import graph.optimizer.rewrite.lowering.LinearLoweringRule;
import graph.optimizer.rewrite.lowering.LossForwardLoweringRule;

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
     * @return optimizer containing only graph rewrite/simplification/lowering rules
     */
    public static GraphOptimizer create(GraphOptimizationConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        List<OptimizationRule> rules = new ArrayList<>();
        List<OptimizationRule> simplification = new ArrayList<>(4);
        int rewriteSimplificationRules = 0;
        if (config.algebraicRewrite()) {
            rewriteSimplificationRules = addSimplificationRewriteRules(simplification, config.rewrite());
        }
        if (config.constantFolding()) {
            simplification.add(new ConstantFoldingRule());
        }
        if (config.commonSubexpressionElimination()) {
            simplification.add(new CommonSubexpressionEliminationRule(config.cse()));
        }
        if (config.deadCodeElimination()) {
            simplification.add(new DeadCodeEliminationRule());
        }
        flushSimplification(rules, simplification, rewriteSimplificationRules == simplification.size());
        if (config.optionalLowering()) {
            addLoweringRules(rules, config.rewrite());
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

    private static void flushSimplification(
            List<OptimizationRule> rules,
            List<OptimizationRule> simplification,
            boolean rewriteOnly
    ) {
        if (simplification.isEmpty()) {
            return;
        }
        if (simplification.size() == 1 || rewriteOnly) {
            rules.addAll(simplification);
        } else {
            rules.add(new SimplificationFixpointRule(simplification));
        }
        simplification.clear();
    }

    private static int addSimplificationRewriteRules(List<OptimizationRule> rules, RewriteConfig config) {
        RewriteConfig resolved = config == null ? RewriteConfig.defaults() : config;
        int added = 0;
        if (resolved.piecewiseLowering().anyEnabled()) {
            rules.add(new PiecewiseCanonicalizationRule(resolved.piecewiseLowering()));
            added++;
        }
        if (resolved.algebraic().enabled()) {
            rules.add(new AlgebraicSimplificationRule());
            added++;
        }
        return added;
    }

    private static void addLoweringRules(List<OptimizationRule> rules, RewriteConfig config) {
        RewriteConfig resolved = config == null ? RewriteConfig.defaults() : config;
        if (resolved.linearLowering().enabled()) {
            rules.add(new LinearLoweringRule());
        }
        rules.add(new LossForwardLoweringRule());
        if (resolved.conv2dLowering().mode() != Conv2dLoweringMode.OFF) {
            rules.add(new Conv2dDagLoweringRule(resolved.conv2dLowering()));
        }
    }

}
