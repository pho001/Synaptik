package Graph;

import Tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class GraphOptimizer {
    private List<OptimizationRule> rules=new ArrayList<>();

    public GraphOptimizer(List<OptimizationRule> rules) {
        this.rules = rules;
    }
    public GraphOptimizer() {

    }

    public List<Tensor> optimize(Tensor vertex) {
        List<Tensor> graph = new ArrayList<>();
        if (rules.size()==0){
            return vertex.topologicalSort();
        }
        for (OptimizationRule rule : rules) {
            graph = rule.apply(vertex);
        }
        return graph;
    }

    public void addRule(OptimizationRule rule) {
        if (!rules.contains(rule)) {
            rules.add(rule);
        }
    }

}