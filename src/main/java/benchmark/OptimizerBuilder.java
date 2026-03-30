package benchmark;

import graph.optimizer.GraphOptimizer;
import graph.optimizer.rules.AlgebraicRewritingRule;
import graph.optimizer.rules.CommonSubexpressionEliminationRule;
import graph.optimizer.rules.FuseElementWiseRule;
import graph.optimizer.rules.MemoryOptimizerRule;

public final class OptimizerBuilder {
    private OptimizerBuilder() {}

    public static GraphOptimizer build(OptimizerCandidate candidate) {
        GraphOptimizer optimizer = new GraphOptimizer();
        for (OptimizationStage stage : candidate.stageOrder()) {
            switch (stage) {
                case AR -> optimizer.addRule(new AlgebraicRewritingRule());
                case CSE -> optimizer.addRule(new CommonSubexpressionEliminationRule(candidate.knobs().strictCseSafety()));
                case FUSE -> optimizer.addRule(new FuseElementWiseRule(candidate.knobs().fuseConfig()));
                case MEM -> optimizer.addRule(new MemoryOptimizerRule());
            }
        }
        return optimizer;
    }
}
