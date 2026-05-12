package graph.optimizer.rewrite;

import config.optimizer.RewriteConfig;
import graph.optimizer.OptimizationRule;
import graph.optimizer.state.OptimizerState;

import java.util.List;
import java.util.Objects;

/**
 * Aggregate rewrite stage for algebraic and light canonical rewrites.
 *
 * <p>This rule runs configured delegates in sequence. It is the {@code AR} optimizer stage used before constant
 * folding, CSE, partition planning, fusion, and memory planning. Backend/execution lowering is deliberately not part
 * of this stage.
 */
public class RewriteRule implements OptimizationRule {
    private final List<OptimizationRule> delegates;
    private final RewriteConfig config;

    /**
     * Creates a rewrite stage with explicit delegates and default rewrite configuration.
     *
     * @param delegates rules to apply in order
     */
    public RewriteRule(List<OptimizationRule> delegates) {
        this(RewriteConfig.defaults(), delegates);
    }

    /**
     * Creates a rewrite stage with explicit configuration and delegates.
     *
     * @param config rewrite configuration, or {@code null} for defaults
     * @param delegates rules to apply in order
     */
    public RewriteRule(RewriteConfig config, List<OptimizationRule> delegates) {
        this.config = config == null ? RewriteConfig.defaults() : config;
        this.delegates = List.copyOf(Objects.requireNonNull(delegates, "delegates cannot be null"));
    }

    /**
     * Creates the default rewrite stage.
     *
     * @return rewrite rule using default configuration
     */
    public static RewriteRule defaults() {
        return new RewriteRule(RewriteConfig.defaults());
    }

    /**
     * Creates a rewrite stage from configuration.
     *
     * @param config rewrite configuration, or {@code null} for defaults
     */
    public RewriteRule(RewriteConfig config) {
        this(config, createDelegates(config));
    }

    /**
     * Returns the rewrite configuration.
     *
     * @return rewrite configuration
     */
    public RewriteConfig config() {
        return config;
    }

    private static List<OptimizationRule> createDelegates(RewriteConfig config) {
        RewriteConfig resolved = config == null ? RewriteConfig.defaults() : config;
        java.util.ArrayList<OptimizationRule> delegates = new java.util.ArrayList<>();
        addImportCanonicalizationDelegates(delegates, resolved);
        if (resolved.algebraic().enabled()) {
            delegates.add(new AlgebraicRewrite());
        }
        return List.copyOf(delegates);
    }

    private static void addImportCanonicalizationDelegates(
            java.util.ArrayList<OptimizationRule> delegates,
            RewriteConfig resolved
    ) {
        if (resolved.piecewiseLowering().anyEnabled()) {
            delegates.add(new PiecewiseLoweringRewrite(resolved.piecewiseLowering()));
        }
    }

    /**
     * Applies each configured rewrite delegate in order.
     *
     * @param state optimizer state to rewrite
     * @return rewritten optimizer state
     */
    @Override
    public OptimizerState apply(OptimizerState state) {
        OptimizerState current = state;
        for (OptimizationRule delegate : delegates) {
            current = delegate.apply(current);
        }
        return current;
    }
}
