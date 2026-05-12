import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import operations.index.ScatterReduction;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ScatterElementsExecutionTest {

    @Test
    void scatterElementsNoneWritesRankPreservingAxisUpdates() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, 0, 0, 2}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{1, 5, 7, 9}, new int[]{2, 2}, null, "updates", DataType.FLOAT64);
        Tensor out = data.scatterElements(indices, updates, 1);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                5, 20, 1,
                7, 50, 9
        }, out.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.SCATTER_ELEMENTS));
    }

    @Test
    void scatterElementsNoneRejectsDuplicateTargets() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 1, 0, 2}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{1, 5, 7, 9}, new int[]{2, 2}, null, "updates", DataType.FLOAT64);
        Tensor out = data.scatterElements(indices, updates, 1);

        assertThrows(IllegalArgumentException.class, () ->
                CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                        .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD));
    }

    @Test
    void scatterElementsAddAccumulatesDuplicateTargets() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 1, 0, 2}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{1, 5, 7, 9}, new int[]{2, 2}, null, "updates", DataType.FLOAT64);
        Tensor out = data.scatterElements(indices, updates, 1, ScatterReduction.ADD);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                10, 26, 30,
                47, 50, 69
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterElementsSupportsNegativeAxisAndIndices() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{-1, 0, 0, -1}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{1, 5, 7, 9}, new int[]{2, 2}, null, "updates", DataType.FLOAT64);
        Tensor out = data.scatterElements(indices, updates, -1);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                5, 20, 1,
                7, 50, 9
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterElementsSupportsNonContiguousDataAndUpdates() {
        Tensor dataBase = new Tensor(new double[]{
                10, 20,
                30, 40,
                50, 60
        }, new int[]{3, 2}, null, "dataBase", DataType.FLOAT64);
        Tensor data = dataBase.permute(1, 0);
        Tensor updatesBase = new Tensor(new double[]{
                1, 7,
                5, 9
        }, new int[]{2, 2}, null, "updatesBase", DataType.FLOAT64);
        Tensor updates = updatesBase.permute(1, 0);
        Tensor indices = new Tensor(new int[]{2, 0, 0, 2}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor out = data.scatterElements(indices, updates, 1);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                5, 30, 1,
                7, 40, 9
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterElementsMulMaxMinUseReductionPolicy() {
        Tensor data = new Tensor(new int[]{
                2, 10, 4,
                8, 3, 6
        }, new int[]{2, 3}, null, "data", DataType.INT32);
        Tensor indices = new Tensor(new int[]{1, 1, 0, 2}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new int[]{3, 4, 5, 2}, new int[]{2, 2}, null, "updates", DataType.INT32);

        Tensor mul = data.scatterElements(indices, updates, 1, ScatterReduction.MUL);
        Tensor max = data.scatterElements(indices, updates, 1, ScatterReduction.MAX);
        Tensor min = data.scatterElements(indices, updates, 1, ScatterReduction.MIN);

        CompiledGraph.compile(mul, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(max, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(min, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{2, 120, 4, 40, 3, 12}, mul.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2, 10, 4, 8, 3, 6}, max.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2, 3, 4, 5, 3, 2}, min.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterElementsSupportsBfloat16AndBoolNone() {
        Tensor bf16 = new Tensor(new short[]{
                (short) 0x4120, (short) 0x41a0, (short) 0x41f0,
                (short) 0x4220, (short) 0x4248, (short) 0x4270
        }, new int[]{2, 3}, null, "bf16", DataType.BFLOAT16);
        Tensor bf16Updates = new Tensor(new short[]{
                (short) 0x3f80, (short) 0x40a0,
                (short) 0x40e0, (short) 0x4110
        }, new int[]{2, 2}, null, "bf16Updates", DataType.BFLOAT16);
        Tensor indices = new Tensor(new int[]{2, 0, 0, 2}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor bf16Out = bf16.scatterElements(indices, bf16Updates, 1);

        CompiledGraph.compile(bf16Out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                5.0, 20.0, 1.0,
                7.0, 50.0, 9.0
        }, bf16Out.toDoubleArrayCopy(), 1e-6);

        Tensor bools = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "bools", DataType.BOOL);
        Tensor boolUpdates = new Tensor(new byte[]{0, 1}, new int[]{2, 1}, null, "boolUpdates", DataType.BOOL);
        Tensor boolIndices = new Tensor(new int[]{1, 0}, new int[]{2, 1}, null, "boolIndices", DataType.INT32);
        Tensor boolOut = bools.scatterElements(boolIndices, boolUpdates, 1);

        CompiledGraph.compile(boolOut, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new boolean[]{true, false, true, false}, boolOut.toBooleanArrayCopy());
        assertThrows(IllegalArgumentException.class,
                () -> bools.scatterElements(boolIndices, boolUpdates, 1, ScatterReduction.ADD));
    }

    @Test
    void scatterElementsRejectsShapeAndDtypeMismatches() {
        Tensor data = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 1, 2, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor wrongRank = new Tensor(new int[]{0, 1}, new int[]{2}, null, "wrongRank", DataType.INT32);
        Tensor wrongNonAxis = new Tensor(new int[]{0, 1, 2}, new int[]{1, 3}, null, "wrongNonAxis", DataType.INT32);
        Tensor wrongUpdatesShape = new Tensor(new double[]{1, 2}, new int[]{2, 1}, null, "wrongUpdatesShape", DataType.FLOAT64);
        Tensor wrongDtype = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "wrongDtype", DataType.FLOAT32);
        Tensor updates = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "updates", DataType.FLOAT64);

        assertThrows(IllegalArgumentException.class, () -> data.scatterElements(wrongRank, updates, 1));
        assertThrows(IllegalArgumentException.class, () -> data.scatterElements(wrongNonAxis, new Tensor(new double[]{1, 2, 3}, new int[]{1, 3}, null, "u", DataType.FLOAT64), 1));
        assertThrows(IllegalArgumentException.class, () -> data.scatterElements(indices, wrongUpdatesShape, 1));
        assertThrows(IllegalArgumentException.class, () -> data.scatterElements(indices, wrongDtype, 1));
    }

    @Test
    void scatterElementsBackwardForNoneZerosOverwrittenDataAndGathersUpdates() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor updates = new Tensor(new double[]{1, 5, 7, 9}, new int[]{2, 2}, null, "updates", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, 0, 0, 2}, new int[]{2, 2}, null, "indices", DataType.INT32);
        data.setRequiresGrad(true);
        updates.setRequiresGrad(true);

        Tensor out = data.scatterElements(indices, updates, 1).mul(2.0);

        CompiledGraph.compile(out, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0.0, 2.0, 0.0,
                0.0, 2.0, 0.0
        }, data.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2.0, 2.0, 2.0, 2.0}, updates.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterElementsBackwardForAddPropagatesToDataAndUpdates() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor updates = new Tensor(new double[]{1, 5, 7, 9}, new int[]{2, 2}, null, "updates", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 1, 0, 2}, new int[]{2, 2}, null, "indices", DataType.INT32);
        data.setRequiresGrad(true);
        updates.setRequiresGrad(true);

        Tensor out = data.scatterElements(indices, updates, 1, ScatterReduction.ADD).mul(3.0);

        CompiledGraph.compile(out, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{3.0, 3.0, 3.0, 3.0, 3.0, 3.0}, data.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{3.0, 3.0, 3.0, 3.0}, updates.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterElementsTrainingRejectsMulMaxMinReductions() {
        Tensor data = new Tensor(new double[]{1, 2, 3}, new int[]{1, 3}, null, "data", DataType.FLOAT64);
        Tensor updates = new Tensor(new double[]{4}, new int[]{1, 1}, null, "updates", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1}, new int[]{1, 1}, null, "indices", DataType.INT32);
        data.setRequiresGrad(true);

        assertThrows(UnsupportedOperationException.class,
                () -> data.scatterElements(indices, updates, 1, ScatterReduction.MUL));
        assertThrows(UnsupportedOperationException.class,
                () -> data.scatterElements(indices, updates, 1, ScatterReduction.MAX));
        assertThrows(UnsupportedOperationException.class,
                () -> data.scatterElements(indices, updates, 1, ScatterReduction.MIN));
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}
