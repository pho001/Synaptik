package graph.optimizer;

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
        List<Tensor> current = sortedGraph;
        for (OptimizationRule rule : rules) {
            current = Objects.requireNonNull(rule.apply(current),
                    rule.getClass().getSimpleName() + " returned null");
        }
        return current;
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
}
