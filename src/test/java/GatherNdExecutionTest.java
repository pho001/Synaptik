import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GatherNdExecutionTest {
    @Test
    void gatherNdReadsTupleIndexedElements() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 2, 1, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor out = data.gatherNd(indices);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2}, out.getShape());
        assertArrayEquals(new double[]{30, 40}, out.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.GATHER_ND));
    }

    @Test
    void gatherNdReadsTupleIndexedSlices() {
        Tensor data = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2, 1}, null, "indices", DataType.INT32);
        Tensor out = data.gatherNd(indices);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 3}, out.getShape());
        assertArrayEquals(new double[]{
                4, 5, 6,
                1, 2, 3
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherNdBatchDimsOneReadsPerBatchSlices() {
        Tensor data = new Tensor(new double[]{
                1, 2,
                3, 4,
                5, 6,
                7, 8,
                9, 10,
                11, 12
        }, new int[]{2, 3, 2}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, 0, 1, 0}, new int[]{2, 2, 1}, null, "indices", DataType.INT32);
        Tensor out = data.gatherNd(indices, 1);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 2, 2}, out.getShape());
        assertArrayEquals(new double[]{
                5, 6,
                1, 2,
                9, 10,
                7, 8
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherNdBatchDimsTwoReadsPerBatchElements() {
        Tensor data = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                10, 11, 12
        }, new int[]{2, 2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, 0, 1, 2}, new int[]{2, 2, 1}, null, "indices", DataType.INT32);
        Tensor out = data.gatherNd(indices, 2);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 2}, out.getShape());
        assertArrayEquals(new double[]{3, 4, 8, 12}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherNdAcceptsProjectScalarShapeForFullRankSingleIndex() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 2}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor out = data.gatherNd(indices);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{1}, out.getShape());
        assertArrayEquals(new double[]{60}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherNdSupportsNegativeIndicesAndNonContiguousInput() {
        Tensor dataBase = new Tensor(new double[]{
                10, 20,
                30, 40,
                50, 60
        }, new int[]{3, 2}, null, "dataBase", DataType.FLOAT64);
        Tensor data = dataBase.permute(1, 0);
        Tensor indices = new Tensor(new int[]{0, -1, 1, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor out = data.gatherNd(indices);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{50, 20}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherNdSupportsBfloat16Int32AndBoolValues() {
        Tensor bf16 = new Tensor(new short[]{(short) 0x4120, (short) 0x41a0}, new int[]{2}, null, "bf16", DataType.BFLOAT16);
        Tensor indices = new Tensor(new int[]{1}, new int[]{1, 1}, null, "indices", DataType.INT32);
        Tensor bf16Out = bf16.gatherNd(indices);

        CompiledGraph.compile(bf16Out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{20.0}, bf16Out.toDoubleArrayCopy(), 1e-6);

        Tensor ints = new Tensor(new int[]{3, 7}, new int[]{2}, null, "ints", DataType.INT32);
        Tensor intOut = ints.gatherNd(indices);
        CompiledGraph.compile(intOut, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{7.0}, intOut.toDoubleArrayCopy(), 1e-9);

        Tensor bools = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "bools", DataType.BOOL);
        Tensor boolOut = bools.gatherNd(indices);
        CompiledGraph.compile(boolOut, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new boolean[]{false}, boolOut.toBooleanArrayCopy());
    }

    @Test
    void gatherNdBackwardAccumulatesDuplicateTupleTargets() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 1, 0, 1, 1, 2}, new int[]{3, 2}, null, "indices", DataType.INT32);
        data.setRequiresGrad(true);

        Tensor out = data.gatherNd(indices).mul(2.0);

        CompiledGraph.compile(out, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0.0, 4.0, 0.0,
                0.0, 0.0, 2.0
        }, data.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherNdBackwardAccumulatesWithinBatchDims() {
        Tensor data = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 1, 0, 2}, new int[]{2, 2, 1}, null, "indices", DataType.INT32);
        data.setRequiresGrad(true);

        Tensor out = data.gatherNd(indices, 1).mul(2.0);

        CompiledGraph.compile(out, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0.0, 4.0, 0.0,
                2.0, 0.0, 2.0
        }, data.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherNdRejectsInvalidShapeAndIndexDtype() {
        Tensor data = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "data", DataType.FLOAT64);
        Tensor tupleRankTooLarge = new Tensor(new int[]{0, 1, 2}, new int[]{1, 3}, null, "bad", DataType.INT32);
        Tensor boolIndices = new Tensor(new byte[]{1}, new int[]{1, 1}, null, "boolIndices", DataType.BOOL);
        Tensor batchMismatch = new Tensor(new int[]{0, 1}, new int[]{1, 2, 1}, null, "batchMismatch", DataType.INT32);

        assertThrows(IllegalArgumentException.class, () -> data.gatherNd(tupleRankTooLarge));
        assertThrows(IllegalArgumentException.class, () -> data.gatherNd(boolIndices));
        assertThrows(IllegalArgumentException.class, () -> data.gatherNd(batchMismatch, 1));
        assertThrows(IllegalArgumentException.class, () -> data.gatherNd(boolIndices, 1));
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}
