import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import org.junit.jupiter.api.Test;
import operations.elementwise.binary.mul;
import tensor.DataType;
import tensor.storage.Float64Storage;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
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
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());

        List<Tensor> nodes = compiled.getCompiledGraphAsList();

        assertThrows(UnsupportedOperationException.class, () -> nodes.add(out));
    }

    @Test
    void preparedExecutionRejectsMutatedSemanticNodeTopology() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        TensorInternalAccess.setPrevTensors(out, List.of(a));
        TensorInternalAccess.setOperation(out, new mul());

        assertThrows(IllegalStateException.class, () -> execution.execute(ExecutionMode.FORWARD));
    }

    @Test
    void preparedExecutionUsesCompiledSnapshotForForwardOutputBinding() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
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
    void prepareRejectsMutatedSemanticTopology() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());

        TensorInternalAccess.setPrevTensors(out, List.of(a));
        TensorInternalAccess.setOperation(out, new mul());

        assertThrows(IllegalStateException.class, () -> compiled.prepare(RuntimeConfig.inferenceDefaults()));
    }

    @Test
    void preparedExecutionAllowsInputValueChanges() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        a.setData(new double[]{11.0, 13.0});
        b.setData(new double[]{17.0, 19.0});

        assertDoesNotThrow(() -> execution.execute(ExecutionMode.FORWARD));
        assertArrayEquals(new double[]{28.0, 32.0}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void executePreparedRejectsPlanFromDifferentCompiledGraph() {
        Tensor a = new Tensor(new double[]{1.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2.0}, new int[]{1}, null, "b", DataType.FLOAT64);
        CompiledGraph first = CompiledGraph.compile(a.add(b), CompileConfig.noGraphOptimizationBaseline());

        Tensor x = new Tensor(new double[]{3.0}, new int[]{1}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(new double[]{4.0}, new int[]{1}, null, "y", DataType.FLOAT64);
        CompiledGraph second = CompiledGraph.compile(x.add(y), CompileConfig.noGraphOptimizationBaseline());
        PreparedExecution firstPlan = first.prepare(RuntimeConfig.inferenceDefaults());

        assertThrows(IllegalArgumentException.class,
                () -> second.executePrepared(firstPlan, ExecutionMode.FORWARD));
    }

    @Test
    void replaceStorageRejectsViewLayout() {
        Tensor x = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor selected = x.select(1, 1);

        assertThrows(UnsupportedOperationException.class,
                () -> TensorInternalAccess.replaceStorage(selected, new Float64Storage(2)));
    }
}
