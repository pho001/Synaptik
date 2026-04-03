import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class BoolTensorInfrastructureTest {

    @Test
    void boolTensorConstructionAndReadback() {
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "mask", DataType.BOOL);

        assertEquals(DataType.BOOL, mask.getDataType());
        assertArrayEquals(new boolean[]{true, false, true, false}, mask.toBooleanArrayCopy());
        assertArrayEquals(new double[]{1.0, 0.0, 1.0, 0.0}, mask.toDoubleArrayCopy(), 0.0);
    }

    @Test
    void boolPermuteAndContiguousWork() {
        Tensor mask = new Tensor(new byte[]{1, 0, 0, 1, 1, 0}, new int[]{2, 3}, null, "mask", DataType.BOOL);

        Tensor permuted = mask.permute(1, 0);
        CompiledGraph.compile(permuted, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new boolean[]{true, true, false, true, false, false}, permuted.toBooleanArrayCopy());

        Tensor materialized = permuted.contiguous();
        CompiledGraph.compile(materialized, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new int[]{3, 2}, materialized.getShape());
        assertArrayEquals(new boolean[]{true, true, false, true, false, false}, materialized.toBooleanArrayCopy());
    }

    @Test
    void boolExpandTrueViewAndContiguousMaterializationWork() {
        Tensor mask = new Tensor(new byte[]{1, 0, 1}, new int[]{1, 3}, null, "mask", DataType.BOOL);

        Tensor expanded = mask.expand(2, 3);
        CompiledGraph.compile(expanded, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{0, 1}, expanded.getStrides());
        assertArrayEquals(new boolean[]{true, false, true, true, false, true}, expanded.toBooleanArrayCopy());

        Tensor contiguous = expanded.contiguous();
        CompiledGraph.compile(contiguous, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new boolean[]{true, false, true, true, false, true}, contiguous.toBooleanArrayCopy());
    }

    @Test
    void boolForwardOutputSyncWorks() {
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{2, 2}, null, "mask", DataType.BOOL);
        Tensor out = mask.forwardOutput();

        CompiledGraph.compile(out, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertEquals(DataType.BOOL, out.getDataType());
        assertArrayEquals(new boolean[]{true, false, true, true}, out.toBooleanArrayCopy());
    }

    @Test
    void boolSetDataTypeToNumericFails() {
        Tensor mask = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "mask", DataType.BOOL);
        assertThrows(UnsupportedOperationException.class, () -> mask.setDataType(DataType.FLOAT32));
    }
}
