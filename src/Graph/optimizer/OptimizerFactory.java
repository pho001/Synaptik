package Graph.optimizer;

import Graph.optimizer.rules.AlgebraicRewritingRule;
import Graph.optimizer.rules.CommonSubexpressionEliminationRule;
import Graph.optimizer.rules.FuseElementWiseRule;
import Graph.optimizer.rules.MemoryOptimizerRule;

public final class OptimizerFactory {
    private OptimizerFactory() {}

    public static OptimizationRule addAlgebraicRewritingRule() {
        return new AlgebraicRewritingRule();
    }

    public static OptimizationRule addCommonSubexpressionEliminationRule() {
        return new CommonSubexpressionEliminationRule(true);
    }

    public static OptimizationRule addCommonSubexpressionEliminationRuleAggressive() {
        return new CommonSubexpressionEliminationRule(false);
    }

    public static OptimizationRule addFuseElementWise() {
        return new FuseElementWiseRule(true);
    }

    public static OptimizationRule addFuseElementWiseAggressive() {
        return new FuseElementWiseRule(false);
    }

    public static OptimizationRule addMemoryOptimizerRule() {
        return new MemoryOptimizerRule();
    }

    // Bezpečný režim pro training/autograd
    public static GraphOptimizer createTrainingOptimizer() {
        GraphOptimizer optimizer = new GraphOptimizer();
        optimizer.addRule(addAlgebraicRewritingRule());
        optimizer.addRule(addCommonSubexpressionEliminationRule());
        optimizer.addRule(addFuseElementWise());
        optimizer.addRule(addMemoryOptimizerRule());
        return optimizer;
    }

    // Agresivní režim pro inference benchmarky
    public static GraphOptimizer createInferencePerformanceOptimizer() {
        GraphOptimizer optimizer = new GraphOptimizer();
        optimizer.addRule(addAlgebraicRewritingRule());
        optimizer.addRule(addCommonSubexpressionEliminationRuleAggressive());
        optimizer.addRule(addFuseElementWiseAggressive());
        optimizer.addRule(addMemoryOptimizerRule());
        return optimizer;
    }

    // Kompatibilita se stávajícím benchmarkem
    public static GraphOptimizer createRecommendedTrainingOptimizer() {
        return createTrainingOptimizer();
    }
}
