package graph.optimizer;

import graph.optimizer.state.OptimizerState;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class GraphOptimizer {
    private final List<OptimizationRule> rules;

    public GraphOptimizer(List<OptimizationRule> rules) {
        Objects.requireNonNull(rules, "rules cannot be null");
        this.rules = new ArrayList<>(rules.size());
        for (OptimizationRule rule : rules) {
            addRule(rule);
        }
    }

    public GraphOptimizer() {
        this.rules = new ArrayList<>();
    }

    public List<Tensor> optimize(List<Tensor> sortedGraph) {
        Objects.requireNonNull(sortedGraph, "sortedGraph cannot be null");
        return optimize(OptimizerState.ofGraph(sortedGraph)).graph();
    }

    public List<Tensor> optimize(List<Tensor> sortedGraph, Tensor forwardOutput) {
        Objects.requireNonNull(sortedGraph, "sortedGraph cannot be null");
        Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
        return optimize(OptimizerState.ofGraph(sortedGraph, forwardOutput)).graph();
    }

    public OptimizerState optimize(OptimizerState initial) {
        Objects.requireNonNull(initial, "initial cannot be null");
        if (rules.isEmpty()) {
            return initial;
        }
        return applyRules(initial, rules);
    }

    public GraphOptimizer addRule(OptimizationRule rule) {
        Objects.requireNonNull(rule, "rule cannot be null");
        if (!rules.contains(rule)) {
            rules.add(rule);
        }
        return this;
    }

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
