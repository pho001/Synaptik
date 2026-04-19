package graph.optimizer;

import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class GraphOptimizer {
    private final List<OptimizationRule> rules;
    private final int iterativeRuleCount;
    private final int maxFixpointRounds;

    public GraphOptimizer(List<OptimizationRule> rules) {
        this(rules, 0, 1);
    }

    GraphOptimizer(List<OptimizationRule> rules, int iterativeRuleCount, int maxFixpointRounds) {
        Objects.requireNonNull(rules, "rules cannot be null");
        this.rules = new ArrayList<>(rules.size());
        for (OptimizationRule rule : rules) {
            addRule(rule);
        }
        this.iterativeRuleCount = Math.max(0, Math.min(iterativeRuleCount, this.rules.size()));
        this.maxFixpointRounds = Math.max(1, maxFixpointRounds);
    }

    public GraphOptimizer() {
        this.rules = new ArrayList<>();
        this.iterativeRuleCount = 0;
        this.maxFixpointRounds = 1;
    }

    public List<Tensor> optimize(List<Tensor> sortedGraph) {
        Objects.requireNonNull(sortedGraph, "sortedGraph cannot be null");
        List<Tensor> current = sortedGraph;
        if (rules.isEmpty()) {
            return current;
        }

        if (iterativeRuleCount <= 0 || maxFixpointRounds <= 1) {
            return applyRules(current, rules);
        }

        List<OptimizationRule> iterativeRules = rules.subList(0, iterativeRuleCount);
        List<OptimizationRule> terminalRules = rules.subList(iterativeRuleCount, rules.size());
        for (int round = 0; round < maxFixpointRounds; round++) {
            String before = OptimizerFingerprint.of(current);
            current = applyRules(current, iterativeRules);
            String after = OptimizerFingerprint.of(current);
            if (before.equals(after)) {
                break;
            }
        }
        if (!terminalRules.isEmpty()) {
            current = applyRules(current, terminalRules);
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

    private static List<Tensor> applyRules(List<Tensor> graph, List<OptimizationRule> rules) {
        List<Tensor> current = graph;
        for (OptimizationRule rule : rules) {
            current = Objects.requireNonNull(rule.apply(current),
                    rule.getClass().getSimpleName() + " returned null");
        }
        return current;
    }
}
