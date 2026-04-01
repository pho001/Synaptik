package graph.optimizer;

import config.optimizer.CseConfig;
import graph.optimizer.rules.AlgebraicRewritingRule;
import graph.optimizer.rules.CommonSubexpressionEliminationRule;
import graph.optimizer.rules.FuseElementWiseRule;
import graph.optimizer.rules.MemoryOptimizerRule;

import java.util.Objects;

public final class OptimizerFactory {
    private OptimizerFactory() {}

    public static GraphOptimizer create(config.optimizer.OptimizerConfig config) {
        Objects.requireNonNull(config, "config cannot be null");

        GraphOptimizer optimizer = new GraphOptimizer();
        for (config.optimizer.OptimizerStage stage : config.stageOrder()) {
            switch (stage) {
                case AR -> optimizer.addRule(new AlgebraicRewritingRule());
                case CSE -> optimizer.addRule(new CommonSubexpressionEliminationRule(config.cse()));
                case FUSE -> optimizer.addRule(new FuseElementWiseRule(config.fuse()));
                case MEM -> optimizer.addRule(new MemoryOptimizerRule());
            }
        }
        return optimizer;
    }

    public static GraphOptimizer createNoOptimizationOptimizer() {
        return create(config.optimizer.OptimizerConfig.noOptimization());
    }

    public static GraphOptimizer createTrainingOptimizer() {
        return create(config.optimizer.OptimizerConfig.trainingDefaults());
    }

    public static GraphOptimizer createRecommendedTrainingOptimizer() {
        return createTrainingOptimizer();
    }

    public static GraphOptimizer createInferenceOptimizer() {
        return create(config.optimizer.OptimizerConfig.inferenceDefaults());
    }

    public static AlgebraicRewritingRule addAlgebraicRewritingRule() {
        return new AlgebraicRewritingRule();
    }

    public static CommonSubexpressionEliminationRule addCommonSubexpressionEliminationRule() {
        return new CommonSubexpressionEliminationRule(CseConfig.strictDefaults());
    }

    public static FuseElementWiseRule addFuseElementWise() {
        return new FuseElementWiseRule();
    }

    public static MemoryOptimizerRule addMemoryOptimizerRule() {
        return new MemoryOptimizerRule();
    }
}
