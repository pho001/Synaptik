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

public class ScatterNdExecutionTest {
    @Test
    void scatterNdNoneWritesTupleIndexedElements() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 2, 1, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{1, 7}, new int[]{2}, null, "updates", DataType.FLOAT64);
        Tensor out = data.scatterNd(indices, updates);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                10, 20, 1,
                7, 50, 60
        }, out.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.SCATTER_ND));
    }

    @Test
    void scatterNdWritesTupleIndexedSlices() {
        Tensor data = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1}, new int[]{1, 1}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{40, 50, 60}, new int[]{1, 3}, null, "updates", DataType.FLOAT64);
        Tensor out = data.scatterNd(indices, updates);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                1, 2, 3,
                40, 50, 60
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterNdAcceptsProjectScalarShapeForFullRankSingleIndex() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 2}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{9}, new int[]{1}, null, "updates", DataType.FLOAT64);
        Tensor out = data.scatterNd(indices, updates);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                10, 20, 30,
                40, 50, 9
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterNdAddAccumulatesDuplicateTupleTargets() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 1, 0, 1}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{5, 7}, new int[]{2}, null, "updates", DataType.FLOAT64);
        Tensor out = data.scatterNd(indices, updates, ScatterReduction.ADD);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                10, 32, 30,
                40, 50, 60
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterNdAddSupportsBatchDimsOne() {
        Tensor data = new Tensor(new double[6], new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 1, 0, 2}, new int[]{2, 2, 1}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{2, 3, 7, 11}, new int[]{2, 2}, null, "updates", DataType.FLOAT64);
        Tensor out = data.scatterNd(indices, updates, ScatterReduction.ADD, 1);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                0, 5, 0,
                7, 0, 11
        }, out.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.SCATTER_ND));
    }

    @Test
    void scatterNdNoneRejectsDuplicateTupleTargets() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 1, 0, 1}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{5, 7}, new int[]{2}, null, "updates", DataType.FLOAT64);
        Tensor out = data.scatterNd(indices, updates);

        assertThrows(IllegalArgumentException.class, () ->
                CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                        .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD));
    }

    @Test
    void scatterNdSupportsNegativeIndicesAndNonContiguousInputs() {
        Tensor dataBase = new Tensor(new double[]{
                10, 20,
                30, 40,
                50, 60
        }, new int[]{3, 2}, null, "dataBase", DataType.FLOAT64);
        Tensor data = dataBase.permute(1, 0);
        Tensor indices = new Tensor(new int[]{0, -1}, new int[]{1, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{1}, new int[]{1}, null, "updates", DataType.FLOAT64);
        Tensor out = data.scatterNd(indices, updates);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                10, 30, 1,
                20, 40, 60
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterNdSupportsNonContiguousSliceUpdates() {
        Tensor data = new Tensor(new double[4], new int[]{2, 2}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 1}, new int[]{2, 1}, null, "indices", DataType.INT32);
        Tensor updateBase = new Tensor(new double[]{
                1, 7,
                5, 9
        }, new int[]{2, 2}, null, "updateBase", DataType.FLOAT64);
        Tensor updates = updateBase.permute(1, 0);
        Tensor out = data.scatterNd(indices, updates);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                1, 5,
                7, 9
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterNdSupportsIntReductionsBfloat16AndBoolNone() {
        Tensor ints = new Tensor(new int[]{2, 10, 4, 8}, new int[]{2, 2}, null, "ints", DataType.INT32);
        Tensor intIndices = new Tensor(new int[]{0, 1, 0, 1}, new int[]{2, 2}, null, "intIndices", DataType.INT32);
        Tensor intUpdates = new Tensor(new int[]{3, 4}, new int[]{2}, null, "intUpdates", DataType.INT32);
        Tensor mul = ints.scatterNd(intIndices, intUpdates, ScatterReduction.MUL);

        CompiledGraph.compile(mul, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{2, 120, 4, 8}, mul.toDoubleArrayCopy(), 1e-9);

        Tensor bf16 = new Tensor(new short[]{(short) 0x4120, (short) 0x41a0}, new int[]{2}, null, "bf16", DataType.BFLOAT16);
        Tensor bf16Indices = new Tensor(new int[]{1}, new int[]{1, 1}, null, "bf16Indices", DataType.INT32);
        Tensor bf16Updates = new Tensor(new short[]{(short) 0x3f80}, new int[]{1}, null, "bf16Updates", DataType.BFLOAT16);
        Tensor bf16Out = bf16.scatterNd(bf16Indices, bf16Updates);

        CompiledGraph.compile(bf16Out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{10.0, 1.0}, bf16Out.toDoubleArrayCopy(), 1e-6);

        Tensor bools = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "bools", DataType.BOOL);
        Tensor boolUpdates = new Tensor(new byte[]{1}, new int[]{1}, null, "boolUpdates", DataType.BOOL);
        Tensor boolOut = bools.scatterNd(bf16Indices, boolUpdates);

        CompiledGraph.compile(boolOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new boolean[]{true, true}, boolOut.toBooleanArrayCopy());
        assertThrows(IllegalArgumentException.class,
                () -> bools.scatterNd(bf16Indices, boolUpdates, ScatterReduction.ADD));
    }

    @Test
    void scatterNdRejectsShapeDtypeAndUnsupportedTrainingReductions() {
        Tensor data = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 1}, new int[]{1, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{9}, new int[]{1}, null, "updates", DataType.FLOAT64);

        assertThrows(IllegalArgumentException.class,
                () -> data.scatterNd(new Tensor(new int[]{0, 1, 2}, new int[]{1, 3}, null, "badTuple", DataType.INT32), updates));
        assertThrows(IllegalArgumentException.class,
                () -> data.scatterNd(indices, new Tensor(new double[]{1, 2}, new int[]{2}, null, "badUpdates", DataType.FLOAT64)));
        assertThrows(IllegalArgumentException.class,
                () -> data.scatterNd(indices, new Tensor(new float[]{1}, new int[]{1}, null, "wrongDtype", DataType.FLOAT32)));

        data.setRequiresGrad(true);
        assertThrows(UnsupportedOperationException.class, () -> data.scatterNd(indices, updates, ScatterReduction.MUL));
        assertThrows(UnsupportedOperationException.class, () -> data.scatterNd(indices, updates, ScatterReduction.MAX));
        assertThrows(UnsupportedOperationException.class, () -> data.scatterNd(indices, updates, ScatterReduction.MIN));
    }

    @Test
    void scatterNdBackwardForNoneZerosOverwrittenDataAndGathersUpdates() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 2, 1, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{1, 7}, new int[]{2}, null, "updates", DataType.FLOAT64);
        data.setRequiresGrad(true);
        updates.setRequiresGrad(true);

        Tensor out = data.scatterNd(indices, updates).mul(2.0);

        CompiledGraph.compile(out, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                2.0, 2.0, 0.0,
                0.0, 2.0, 2.0
        }, data.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2.0, 2.0}, updates.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterNdBackwardForAddPropagatesDataAndGathersUpdates() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 1, 0, 1, 1, 2}, new int[]{3, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{5, 7, 9}, new int[]{3}, null, "updates", DataType.FLOAT64);
        data.setRequiresGrad(true);
        updates.setRequiresGrad(true);

        Tensor out = data.scatterNd(indices, updates, ScatterReduction.ADD).mul(3.0);

        CompiledGraph.compile(out, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                3.0, 3.0, 3.0,
                3.0, 3.0, 3.0
        }, data.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{3.0, 3.0, 3.0}, updates.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterNdBackwardForAddSupportsBatchDimsOne() {
        Tensor data = new Tensor(new double[6], new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 1, 0, 2}, new int[]{2, 2, 1}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{2, 3, 7, 11}, new int[]{2, 2}, null, "updates", DataType.FLOAT64);
        data.setRequiresGrad(true);
        updates.setRequiresGrad(true);

        Tensor out = data.scatterNd(indices, updates, ScatterReduction.ADD, 1).mul(4.0);

        CompiledGraph.compile(out, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                4.0, 4.0, 4.0,
                4.0, 4.0, 4.0
        }, data.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{4.0, 4.0, 4.0, 4.0}, updates.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterNdBackwardForSliceUpdatesUsesGatherNdSuffixShape() {
        Tensor data = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1}, new int[]{1, 1}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{40, 50, 60}, new int[]{1, 3}, null, "updates", DataType.FLOAT64);
        data.setRequiresGrad(true);
        updates.setRequiresGrad(true);

        Tensor out = data.scatterNd(indices, updates).mul(5.0);

        CompiledGraph.compile(out, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                5.0, 5.0, 5.0,
                0.0, 0.0, 0.0
        }, data.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new int[]{1, 3}, updates.getGradient().getShape());
        assertArrayEquals(new double[]{5.0, 5.0, 5.0}, updates.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterNdBackwardSupportsNonContiguousSliceUpdates() {
        Tensor data = new Tensor(new double[4], new int[]{2, 2}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 1}, new int[]{2, 1}, null, "indices", DataType.INT32);
        Tensor updateBase = new Tensor(new double[]{
                1, 7,
                5, 9
        }, new int[]{2, 2}, null, "updateBase", DataType.FLOAT64);
        Tensor updates = updateBase.permute(1, 0);
        data.setRequiresGrad(true);
        updates.setRequiresGrad(true);

        Tensor out = data.scatterNd(indices, updates).mul(4.0);

        CompiledGraph.compile(out, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[4], data.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{4.0, 4.0, 4.0, 4.0}, updates.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterNdBackwardSupportsBfloat16NoneAndAdd() {
        Tensor dataNone = new Tensor(new short[]{(short) 0x4120, (short) 0x41a0}, new int[]{2}, null, "dataNone", DataType.BFLOAT16);
        Tensor indices = new Tensor(new int[]{1}, new int[]{1, 1}, null, "indices", DataType.INT32);
        Tensor updatesNone = new Tensor(new short[]{(short) 0x3f80}, new int[]{1}, null, "updatesNone", DataType.BFLOAT16);
        dataNone.setRequiresGrad(true);
        updatesNone.setRequiresGrad(true);

        Tensor none = dataNone.scatterNd(indices, updatesNone).mul(2.0);
        CompiledGraph.compile(none, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{2.0, 0.0}, dataNone.getGradient().toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{2.0}, updatesNone.getGradient().toDoubleArrayCopy(), 1e-6);

        Tensor dataAdd = new Tensor(new short[]{(short) 0x4120, (short) 0x41a0}, new int[]{2}, null, "dataAdd", DataType.BFLOAT16);
        Tensor updatesAdd = new Tensor(new short[]{(short) 0x3f80}, new int[]{1}, null, "updatesAdd", DataType.BFLOAT16);
        dataAdd.setRequiresGrad(true);
        updatesAdd.setRequiresGrad(true);

        Tensor add = dataAdd.scatterNd(indices, updatesAdd, ScatterReduction.ADD).mul(3.0);
        CompiledGraph.compile(add, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{3.0, 3.0}, dataAdd.getGradient().toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{3.0}, updatesAdd.getGradient().toDoubleArrayCopy(), 1e-6);
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.program().compiledNodes().stream()
                .map(graph.model.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}
