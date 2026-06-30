import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
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
        CompiledGraph.compile(permuted, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(new boolean[]{true, true, false, true, false, false}, permuted.toBooleanArrayCopy());

        Tensor materialized = permuted.contiguous();
        CompiledGraph.compile(materialized, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(new int[]{3, 2}, materialized.getShape());
        assertArrayEquals(new boolean[]{true, true, false, true, false, false}, materialized.toBooleanArrayCopy());
    }

    @Test
    void boolExpandTrueViewAndContiguousMaterializationWork() {
        Tensor mask = new Tensor(new byte[]{1, 0, 1}, new int[]{1, 3}, null, "mask", DataType.BOOL);

        Tensor expanded = mask.expand(2, 3);
        CompiledGraph.compile(expanded, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{0, 1}, expanded.getStrides());
        assertArrayEquals(new boolean[]{true, false, true, true, false, true}, expanded.toBooleanArrayCopy());

        Tensor contiguous = expanded.contiguous();
        CompiledGraph.compile(contiguous, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(new boolean[]{true, false, true, true, false, true}, contiguous.toBooleanArrayCopy());
    }

    @Test
    void boolForwardOutputSyncWorks() {
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{2, 2}, null, "mask", DataType.BOOL);
        Tensor out = mask.forwardOutput();

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertEquals(DataType.BOOL, out.getDataType());
        assertArrayEquals(new boolean[]{true, false, true, true}, out.toBooleanArrayCopy());
    }

    @Test
    void boolSetDataTypeToNumericFails() {
        Tensor mask = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "mask", DataType.BOOL);
        assertThrows(UnsupportedOperationException.class, () -> mask.setDataType(DataType.FLOAT32));
    }

    @Test
    void logicalBoolOpsWork() {
        Tensor a = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "a", DataType.BOOL);
        Tensor b = new Tensor(new byte[]{1, 1, 0, 0}, new int[]{2, 2}, null, "b", DataType.BOOL);

        Tensor and = a.logicalAnd(b);
        Tensor or = a.logicalOr(b);
        Tensor not = a.logicalNot();

        CompiledGraph.compile(and, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(or, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(not, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new boolean[]{true, false, false, false}, and.toBooleanArrayCopy());
        assertArrayEquals(new boolean[]{true, true, true, false}, or.toBooleanArrayCopy());
        assertArrayEquals(new boolean[]{false, true, false, true}, not.toBooleanArrayCopy());
    }

    @Test
    void logicalAndBroadcasts() {
        Tensor a = new Tensor(new byte[]{1, 0, 1, 0, 1, 1}, new int[]{2, 3}, null, "a", DataType.BOOL);
        Tensor b = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "b", DataType.BOOL);

        Tensor out = a.logicalAnd(b);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new boolean[]{true, false, true, false, false, true}, out.toBooleanArrayCopy());
    }

    @Test
    void allAndAnyReductionsWork() {
        Tensor mask = new Tensor(new byte[]{
                1, 1, 0,
                1, 1, 1
        }, new int[]{2, 3}, null, "mask", DataType.BOOL);

        Tensor allAxis = mask.all(1);
        Tensor allKeep = mask.all(1, true);
        Tensor anyAxis = mask.any(1);
        Tensor anyKeep = mask.any(1, true);
        Tensor allAll = mask.all();
        Tensor anyAll = mask.any();

        CompiledGraph.compile(allAxis, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(allKeep, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(anyAxis, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(anyKeep, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(allAll, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(anyAll, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2}, allAxis.getShape());
        assertArrayEquals(new boolean[]{false, true}, allAxis.toBooleanArrayCopy());

        assertArrayEquals(new int[]{2, 1}, allKeep.getShape());
        assertArrayEquals(new boolean[]{false, true}, allKeep.toBooleanArrayCopy());

        assertArrayEquals(new int[]{2}, anyAxis.getShape());
        assertArrayEquals(new boolean[]{true, true}, anyAxis.toBooleanArrayCopy());

        assertArrayEquals(new int[]{2, 1}, anyKeep.getShape());
        assertArrayEquals(new boolean[]{true, true}, anyKeep.toBooleanArrayCopy());

        assertArrayEquals(new int[]{1}, allAll.getShape());
        assertArrayEquals(new boolean[]{false}, allAll.toBooleanArrayCopy());

        assertArrayEquals(new int[]{1}, anyAll.getShape());
        assertArrayEquals(new boolean[]{true}, anyAll.toBooleanArrayCopy());
    }
}
