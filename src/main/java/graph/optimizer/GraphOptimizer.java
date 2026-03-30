package graph.optimizer;

import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class GraphOptimizer {
    private List<OptimizationRule> rules = new ArrayList<>();
    private List<Tensor> graph=new ArrayList<>();

    public GraphOptimizer(List<OptimizationRule> rules) {
        this.rules = rules;
    }

    public GraphOptimizer() {
    }


    public List<Tensor> optimize(List<Tensor> sortedGraph) {
        for (OptimizationRule rule : rules) {
            sortedGraph = rule.apply(sortedGraph);
        }
        return sortedGraph;
    }

    public void addRule(OptimizationRule rule) {
        if (!rules.contains(rule)) {
            rules.add(rule);
        }
    }
}