package benchmark;

import config.optimizer.CseConfig;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.rules.CommonSubexpressionEliminationRule;
import graph.optimizer.rules.FuseElementWiseRule;
import graph.optimizer.rules.MemoryOptimizerRule;
import graph.optimizer.rules.RewriteRule;

public final class OptimizerBuilder {
    private OptimizerBuilder() {}

    public static GraphOptimizer build(OptimizerCandidate candidate) {
        GraphOptimizer optimizer = new GraphOptimizer();
        for (OptimizationStage stage : candidate.stageOrder()) {
            switch (stage) {
                case AR -> optimizer.addRule(new RewriteRule());
                case CSE -> optimizer.addRule(new CommonSubexpressionEliminationRule(
                        candidate.knobs().strictCseSafety()
                                ? CseConfig.strictDefaults()
                                : CseConfig.aggressiveDefaults()
                ));
                case FUSE -> optimizer.addRule(new FuseElementWiseRule(candidate.knobs().fuseConfig()));
                case MEM -> optimizer.addRule(new MemoryOptimizerRule());
            }
        }
        return optimizer;
    }
}
