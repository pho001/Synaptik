import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CompiledGraphIdempotencyTest {
    @Test
    void recompilingSameTrainingGraphDoesNotGrowBackwardGraph() {
        Tensor a = new Tensor(new double[]{2.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        a.setRequiresGrad(true);
        Tensor loss = a.mul(a);

        CompiledGraph first = CompiledGraph.compile(loss, CompileConfig.training());
        int firstNodeCount = first.program().compiledNodes().size();

        CompiledGraph second = CompiledGraph.compile(loss, CompileConfig.training());
        int secondNodeCount = second.program().compiledNodes().size();

        assertEquals(firstNodeCount, secondNodeCount);

        second.prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);
        assertEquals(4.0d, a.getGradient().scalarAsDouble(), 1e-9);
    }

    @Test
    void recompilingAfterTrainingRunIgnoresPreviouslyPublishedSemanticGradients() {
        Tensor a = new Tensor(new double[]{2.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        a.setRequiresGrad(true);
        Tensor loss = a.mul(a);

        CompiledGraph first = CompiledGraph.compile(loss, CompileConfig.training());
        first.prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);
        assertEquals(4.0d, a.getGradient().scalarAsDouble(), 1e-9);

        CompiledGraph second = CompiledGraph.compile(loss, CompileConfig.training());
        int firstNodeCount = first.program().compiledNodes().size();
        int secondNodeCount = second.program().compiledNodes().size();

        assertEquals(firstNodeCount, secondNodeCount);

        second.prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);
        assertEquals(4.0d, a.getGradient().scalarAsDouble(), 1e-9);
    }

    @Test
    void compileDoesNotMutateOriginalInferenceForwardGraph() {
        Tensor a = new Tensor(new double[]{1.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2.0}, new int[]{1}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(new double[]{3.0}, new int[]{1}, null, "c", DataType.FLOAT64);
        Tensor add = a.add(b);
        Tensor out = add.mul(c);

        assertEquals(operations.Operation.OpType.ADD, add.getOperation().opType());
        assertEquals(operations.Operation.OpType.MUL, out.getOperation().opType());
        assertSame(add, out.getPrevTensors().getFirst());

        CompiledGraph.compile(out, CompileConfig.inference());

        assertEquals(operations.Operation.OpType.ADD, add.getOperation().opType());
        assertEquals(operations.Operation.OpType.MUL, out.getOperation().opType());
        assertSame(add, out.getPrevTensors().getFirst());
    }

    @Test
    void trainingCompileRejectsInt64BackwardRoot() {
        Tensor indices = new Tensor(new long[]{1L, 2L}, new int[]{2}, null, "indices", DataType.INT64);
        indices.setRequiresGrad(true);

        UnsupportedOperationException error = assertThrows(
                UnsupportedOperationException.class,
                () -> CompiledGraph.compile(indices, CompileConfig.training())
        );

        assertTrue(error.getMessage().contains("INT64"));
    }
}
