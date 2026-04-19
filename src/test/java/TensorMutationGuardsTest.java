import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import org.junit.jupiter.api.Test;
import operations.elementwise.binary.mul;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TensorMutationGuardsTest {
    @Test
    void prevTensorsViewIsUnmodifiable() {
        Tensor a = new Tensor(new double[]{1.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2.0}, new int[]{1}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        List<Tensor> prev = out.getPrevTensors();

        assertThrows(UnsupportedOperationException.class, () -> prev.add(a));
    }

    @Test
    void compiledGraphExportIsUnmodifiable() {
        Tensor a = new Tensor(new double[]{1.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2.0}, new int[]{1}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);
        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.noOptimization());

        List<Tensor> nodes = compiled.getCompiledGraphAsList();

        assertThrows(UnsupportedOperationException.class, () -> nodes.add(out));
    }

    @Test
    void preparedExecutionUsesCompiledSnapshotInsteadOfMutatedSemanticNodeTopology() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        TensorInternalAccess.setPrevTensors(out, List.of(a));
        TensorInternalAccess.setOperation(out, new mul());

        execution.execute(ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{7.0, 10.0}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void preparedExecutionUsesCompiledSnapshotForForwardOutputBinding() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.noOptimization());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());

        Tensor forwardOutput = compiled.getCompiledGraphAsList().stream()
                .filter(t -> Tensor.SYSTEM_FORWARD_OUTPUT_LABEL.equals(t.getLabel()))
                .findFirst()
                .orElseThrow();
        TensorInternalAccess.setPrevTensors(forwardOutput, List.of(a));

        execution.execute(ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{7.0, 10.0}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void prepareUsesCompiledSnapshotInsteadOfMutatedSemanticTopology() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.noOptimization());

        TensorInternalAccess.setPrevTensors(out, List.of(a));
        TensorInternalAccess.setOperation(out, new mul());

        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{7.0, 10.0}, out.toDoubleArrayCopy(), 1e-9);
    }
}
