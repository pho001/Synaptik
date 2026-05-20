package graph.optimizer;

import graph.optimizer.state.OptimizerState;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Ordered optimizer pipeline for compiled tensor graphs.
 *
 * <p>Rules are applied sequentially to an immutable {@link OptimizerState} value. Each rule may replace the graph,
 * adjust graph-level compile metadata, or append optimization trace entries. Backend ownership, region planning, and
 * memory planning are compile-planning stages outside this package. The standard optimizer pipeline is configured by
 * {@link OptimizerFactory}: semantic cleanup, algebraic rewrite, operation lowering, and cleanup fixpoint passes.
 *
 * <p>This class owns a mutable rule list. Configure a pipeline before sharing it; concurrent calls to {@link #addRule}
 * and {@link #optimize(OptimizerState)} are not supported.
 */
public final class GraphOptimizer {
    private final List<OptimizationRule> rules;

    /**
     * Creates an optimizer with the supplied rules in application order.
     *
     * @param rules optimization rules; each entry must be non-null
     */
    public GraphOptimizer(List<OptimizationRule> rules) {
        Objects.requireNonNull(rules, "rules cannot be null");
        this.rules = new ArrayList<>(rules.size());
        for (OptimizationRule rule : rules) {
            addRule(rule);
        }
    }

    /**
     * Creates an optimizer with no rules.
     */
    public GraphOptimizer() {
        this.rules = new ArrayList<>();
    }

    /**
     * Optimizes a sorted graph using its last tensor as the forward output.
     *
     * @param sortedGraph tensors in topological order
     * @return optimized graph in topological order
     */
    public List<Tensor> optimize(List<Tensor> sortedGraph) {
        Objects.requireNonNull(sortedGraph, "sortedGraph cannot be null");
        return optimize(OptimizerState.ofGraph(sortedGraph)).graph();
    }

    /**
     * Optimizes a sorted graph while preserving an explicit forward output.
     *
     * @param sortedGraph tensors in topological order
     * @param forwardOutput semantic forward result that must remain observable after rewrites
     * @return optimized graph in topological order
     */
    public List<Tensor> optimize(List<Tensor> sortedGraph, Tensor forwardOutput) {
        Objects.requireNonNull(sortedGraph, "sortedGraph cannot be null");
        Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
        return optimize(OptimizerState.ofGraph(sortedGraph, forwardOutput)).graph();
    }

    /**
     * Applies all rules to an optimizer state.
     *
     * @param initial initial state carrying graph-level compile metadata
     * @return final optimizer state after all rules
     * @throws NullPointerException if {@code initial} is {@code null} or any rule returns {@code null}
     */
    public OptimizerState optimize(OptimizerState initial) {
        Objects.requireNonNull(initial, "initial cannot be null");
        if (rules.isEmpty()) {
            return initial;
        }
        return applyRules(initial, rules);
    }

    /**
     * Adds a rule to the end of this pipeline if it is not already present.
     *
     * @param rule optimization rule to add
     * @return this optimizer for fluent construction
     */
    public GraphOptimizer addRule(OptimizationRule rule) {
        Objects.requireNonNull(rule, "rule cannot be null");
        if (!rules.contains(rule)) {
            rules.add(rule);
        }
        return this;
    }

    /**
     * Returns a defensive copy of the configured rules.
     *
     * @return rules in application order
     */
    public List<OptimizationRule> rules() {
        return List.copyOf(rules);
    }

    private static OptimizerState applyRules(OptimizerState state, List<OptimizationRule> rules) {
        OptimizerState current = state;
        for (OptimizationRule rule : rules) {
            current = Objects.requireNonNull(rule.apply(current),
                    rule.getClass().getSimpleName() + " returned null");
        }
        return current;
    }
}
