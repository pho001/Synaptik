package graph.optimizer;

import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import graph.optimizer.state.OptimizerState;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OptimizerCleanupStageTest {
    @Test
    void defaultOptimizerGroupsCleanupStagesIntoFixpoint() {
        List<OptimizationRule> rules = OptimizerFactory.create(CompileConfig.inference().graphOptimization()).rules();

        assertEquals("CleanupFixpointRule", rules.getFirst().getClass().getSimpleName());
    }

    @Test
    void cleanupFixpointFoldsConstantsAndEliminatesDeadNodes() {
        Tensor two = Tensor.scalar(2.0);
        Tensor three = Tensor.scalar(3.0);
        Tensor live = two.add(three);
        Tensor dead = two.mul(three);
        OptimizerState initial = OptimizerState.ofGraph(List.of(two, three, live, dead), live);

        OptimizerState optimized = OptimizerFactory.create(
                CompileConfig.inference()
                        .withGraphOptimization(GraphOptimizationConfig.stages(true, true, true, true, false))
                        .graphOptimization()
        ).optimize(initial);

        assertFalse(optimized.graph().contains(dead));
        assertTrue(optimized.graph().stream().noneMatch(t -> t.getOperation() != null
                && t.getOperation().opType() == Operation.OpType.ADD));
        assertEquals(5.0, optimized.forwardOutput().scalarAsDouble(), 1e-12);
        assertTrue(optimized.trace().costExplanations().stream()
                .anyMatch(explanation -> "GraphCleanupCostModel".equals(explanation.modelName())
                        && "optimizer-cleanup-graph".equals(explanation.inputKind())));
        assertTrue(optimized.trace().events().stream()
                .anyMatch(event -> event.contains("cleanup-cost")));
    }
}
