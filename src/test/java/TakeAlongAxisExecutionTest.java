import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import operations.index.ScatterReduction;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TakeAlongAxisExecutionTest {

    @Test
    void takeAlongAxisPreservesRankAndUsesIndicesShape() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor y = x.takeAlongAxis(indices, 1);

        CompiledGraph compiledGraph = CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 2}, y.getShape());
        assertArrayEquals(new double[]{3.0, 2.0, 4.0, 4.0}, y.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.TAKE_ALONG_AXIS));
    }

    @Test
    void takeAlongAxisNormalizesNegativeIndices() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, -1, 0, 1}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor y = x.takeAlongAxis(indices, 1);

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{3.0, 3.0, 4.0, 5.0}, y.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void takeAlongAxisReadsNativeSegmentInputProducedByPriorCpuNativeOp() {
        Tensor x = new Tensor(new float[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT32);
        Tensor doubled = x.add(x);
        Tensor indices = new Tensor(new int[]{2, -1, 0, 1}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor y = doubled.takeAlongAxis(indices, 1);

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .prepare(nativeRuntime()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{6.0, 6.0, 8.0, 10.0}, y.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void takeAlongAxisBackwardScattersGradientToSelectedPositions() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        x.setRequiresGrad(true);
        Tensor indices = new Tensor(new int[]{2, 2, 0, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor y = x.takeAlongAxis(indices, 1);

        CompiledGraph compiled = CompiledGraph.compile(y, CompileConfig.training());
        compiled
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0.0, 0.0, 2.0,
                2.0, 0.0, 0.0
        }, x.getGradient().toDoubleArrayCopy(), 1e-9);
        assertFalse(containsOp(compiled, Operation.OpType.TAKE_ALONG_AXIS_GRAD));
        assertTrue(containsOp(compiled, Operation.OpType.SCATTER_ELEMENTS));
    }

    @Test
    void scatterElementsAddAccumulatesDuplicateIndicesWithinLane() {
        Tensor base = new Tensor(new double[6], new int[]{2, 3}, null, "base", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, 2, 0, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor outGrad = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "outGrad", DataType.FLOAT64);
        Tensor grad = base.scatterElements(indices, outGrad, 1, ScatterReduction.ADD);

        CompiledGraph.compile(grad, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                0.0, 0.0, 3.0,
                7.0, 0.0, 0.0
        }, grad.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterElementsAddRejectsOutOfBoundsIndexAtExecution() {
        Tensor base = new Tensor(new double[6], new int[]{2, 3}, null, "base", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, 3, 0, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor outGrad = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "outGrad", DataType.FLOAT64);
        Tensor grad = base.scatterElements(indices, outGrad, 1, ScatterReduction.ADD);

        assertThrows(IllegalArgumentException.class, () ->
                CompiledGraph.compile(grad, CompileConfig.noGraphOptimizationBaseline())
                        .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD));
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.program().compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }

    private static RuntimeConfig nativeRuntime() {
        return RuntimeConfig.inferenceDefaults().withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);
    }
}
