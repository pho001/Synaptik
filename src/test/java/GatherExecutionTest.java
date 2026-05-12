import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import operations.index.gatherGrad;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorPrimitiveBuilder;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GatherExecutionTest {

    @Test
    void gatherAxisOneSelectsOneValuePerRow() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor indices = new Tensor(new double[]{2, 0}, new int[]{2}, null, "indices", DataType.FLOAT64);
        Tensor y = x.gather(indices, 1);

        CompiledGraph compiledGraph = CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2}, y.getShape());
        assertArrayEquals(new double[]{3.0, 4.0}, y.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.GATHER));
    }

    @Test
    void gatherSupportsNonContiguousInputView() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base", DataType.FLOAT64);
        Tensor view = base.permute(1, 0);
        Tensor indices = new Tensor(new double[]{2, 0}, new int[]{2}, null, "indices", DataType.FLOAT64);
        Tensor y = view.gather(indices, 0);

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{3.0, 4.0}, y.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherBackwardScattersGradientToSelectedPositions() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        x.setRequiresGrad(true);
        Tensor indices = new Tensor(new double[]{2, 0}, new int[]{2}, null, "indices", DataType.FLOAT64);
        Tensor y = x.gather(indices, 1);

        CompiledGraph.compile(y, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0.0, 0.0, 1.0,
                1.0, 0.0, 0.0
        }, x.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherAxisUsesOnnxOutputShapeAndNegativeIndices() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, 0, -1, 1}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor y = x.gatherAxis(indices, 1);

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 2, 2}, y.getShape());
        assertArrayEquals(new double[]{
                3.0, 1.0, 3.0, 2.0,
                6.0, 4.0, 6.0, 5.0
        }, y.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline()), Operation.OpType.GATHER_AXIS));
    }

    @Test
    void gatherAxisSupportsNonContiguousDataInput() {
        Tensor base = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "base", DataType.FLOAT64);
        Tensor view = base.permute(1, 0);
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor y = view.gatherAxis(indices, 1);

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{3, 2}, y.getShape());
        assertArrayEquals(new double[]{
                4.0, 1.0,
                5.0, 2.0,
                6.0, 3.0
        }, y.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherAxisBackwardAccumulatesRepeatedIndices() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        x.setRequiresGrad(true);
        Tensor indices = new Tensor(new int[]{2, 0, 2, 1}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor y = x.gatherAxis(indices, 1);

        CompiledGraph.compile(y, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                1.0, 1.0, 2.0,
                1.0, 1.0, 2.0
        }, x.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherAxisSupportsIntAndBoolValues() {
        Tensor ints = new Tensor(new int[]{10, 20, 30}, new int[]{3}, null, "ints", DataType.INT32);
        Tensor intIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "intIndices", DataType.INT32);
        Tensor intOut = ints.gatherAxis(intIndices, 0);

        CompiledGraph.compile(intOut, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertEquals(DataType.INT32, intOut.getDataType());
        assertArrayEquals(new double[]{30.0, 10.0}, intOut.toDoubleArrayCopy(), 1e-9);

        Tensor bools = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "bools", DataType.BOOL);
        Tensor boolOut = bools.gatherAxis(intIndices, 0);

        CompiledGraph.compile(boolOut, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertEquals(DataType.BOOL, boolOut.getDataType());
        assertArrayEquals(new boolean[]{true, true}, boolOut.toBooleanArrayCopy());
    }

    @Test
    void gatherGradPrimitiveScattersIntoOriginalInputShape() {
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor outGrad = new Tensor(new double[]{1, 2}, new int[]{2}, null, "outGrad", DataType.FLOAT64);
        Tensor grad = TensorPrimitiveBuilder.binary(
                indices,
                outGrad,
                new int[]{2, 3},
                new gatherGrad(1),
                "gatherGrad",
                DataType.FLOAT64
        );

        CompiledGraph.compile(grad, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                0.0, 0.0, 1.0,
                2.0, 0.0, 0.0
        }, grad.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherGradRepeatedAxisValuesRemainLaneScoped() {
        Tensor indices = new Tensor(new int[]{2, 2}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor outGrad = new Tensor(new double[]{1, 2}, new int[]{2}, null, "outGrad", DataType.FLOAT64);
        Tensor grad = TensorPrimitiveBuilder.binary(
                indices,
                outGrad,
                new int[]{2, 3},
                new gatherGrad(1),
                "gatherGrad",
                DataType.FLOAT64
        );

        CompiledGraph.compile(grad, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                0.0, 0.0, 1.0,
                0.0, 0.0, 2.0
        }, grad.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherGradRejectsOutOfBoundsIndexAtExecution() {
        Tensor indices = new Tensor(new int[]{3, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor outGrad = new Tensor(new double[]{1, 2}, new int[]{2}, null, "outGrad", DataType.FLOAT64);
        Tensor grad = TensorPrimitiveBuilder.binary(
                indices,
                outGrad,
                new int[]{2, 3},
                new gatherGrad(1),
                "gatherGrad",
                DataType.FLOAT64
        );

        assertThrows(IllegalArgumentException.class, () ->
                CompiledGraph.compile(grad, CompileConfig.noGraphOptimizationBaseline())
                        .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD));
    }

    @Test
    void gatherRejectsNonIntegralIndices() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor indices = new Tensor(new double[]{1.5, 0.0}, new int[]{2}, null, "indices", DataType.FLOAT64);
        Tensor y = x.gather(indices, 1);

        assertThrows(IllegalArgumentException.class, () ->
                CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                        .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD));
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}
