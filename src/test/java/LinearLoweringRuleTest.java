import backend.ComputeBackend;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.state.OptimizerState;
import graph.optimizer.rewrite.lowering.LinearLoweringRule;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import graph.compile.intent.BackendIntentPlan;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class LinearLoweringRuleTest {
    @Test
    void lowersMatmulPlusBiasAddToLinearPrimitive() {
        Tensor input = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "weight", DataType.FLOAT64);
        Tensor bias = new Tensor(new double[]{1, 2}, new int[]{2}, null, "bias", DataType.FLOAT64);

        Tensor root = input.matmul(weight).add(bias);
        GraphOptimizer optimizer = new GraphOptimizer().addRule(new LinearLoweringRule());
        List<Tensor> optimized = optimizer.optimize(root.topologicalSort());

        Tensor optimizedRoot = optimized.stream()
                .filter(t -> t.getOperation() != null && t.getOperation().opType() == Operation.OpType.LINEAR)
                .findFirst()
                .orElse(null);
        assertNotNull(optimizedRoot);
        assertEquals(Operation.OpType.LINEAR, optimizedRoot.getOperation().opType());
        assertEquals(3, optimizedRoot.getPrevTensors().size());
    }

    @Test
    void preservesAcceleratorIntentWhenLoweringReplacesNode() {
        Tensor input = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "weight", DataType.FLOAT64);
        Tensor bias = new Tensor(new double[]{1, 2}, new int[]{2}, null, "bias", DataType.FLOAT64);

        Tensor root = input.matmul(weight).add(bias);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.of(root, ComputeBackend.GPU_METAL);

        OptimizerState optimized = new GraphOptimizer()
                .addRule(new LinearLoweringRule())
                .optimize(OptimizerState.ofGraph(root.topologicalSort(), root));

        Tensor optimizedRoot = optimized.graph().stream()
                .filter(t -> t.getOperation() != null && t.getOperation().opType() == Operation.OpType.LINEAR)
                .findFirst()
                .orElse(null);
        assertNotNull(optimizedRoot);
        BackendIntentPlan remapped = backendIntentPlan.remapThrough(optimized.rewriteMap());
        assertEquals(ComputeBackend.GPU_METAL, remapped.backend(optimizedRoot));
    }
}
