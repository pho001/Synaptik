import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CompiledGraphIdempotencyTest {
    @Test
    void recompilingSameTrainingGraphDoesNotGrowBackwardGraph() {
        Tensor a = new Tensor(new double[]{2.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        a.setRequiresGrad(true);
        Tensor loss = a.mul(a);

        CompiledGraph first = CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults());
        int firstNodeCount = first.getCompiledGraphAsList().size();

        CompiledGraph second = CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults());
        int secondNodeCount = second.getCompiledGraphAsList().size();

        assertEquals(firstNodeCount, secondNodeCount);

        second.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        assertEquals(4.0d, a.getGradient().scalarAsDouble(), 1e-9);
    }

    @Test
    void recompilingAfterTrainingRunIgnoresPreviouslyPublishedSemanticGradients() {
        Tensor a = new Tensor(new double[]{2.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        a.setRequiresGrad(true);
        Tensor loss = a.mul(a);

        CompiledGraph first = CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults());
        first.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        assertEquals(4.0d, a.getGradient().scalarAsDouble(), 1e-9);

        CompiledGraph second = CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults());
        int firstNodeCount = first.getCompiledGraphAsList().size();
        int secondNodeCount = second.getCompiledGraphAsList().size();

        assertEquals(firstNodeCount, secondNodeCount);

        second.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        assertEquals(4.0d, a.getGradient().scalarAsDouble(), 1e-9);
    }
}
