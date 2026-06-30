import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.options.Pool2dOptions;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class Pool2dExecutionTest {

    @Test
    void maxPool2dForwardMatchesExpectedValues() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "input", DataType.FLOAT64);

        Tensor out = input.maxPool2d(Pool2dOptions.square(2));
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{1, 1, 2, 2}, out.getShape());
        assertArrayEquals(new double[]{
                6, 8,
                14, 16
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void maxPool2dBackwardRoutesGradientToWindowMaxima() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "input", DataType.FLOAT64);
        input.setRequiresGrad(true);

        Tensor loss = input.maxPool2d(Pool2dOptions.square(2)).sum();
        CompiledGraph graph = CompiledGraph.compile(loss, CompileConfig.training());
        graph.prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0, 0, 0, 0,
                0, 1, 0, 1,
                0, 0, 0, 0,
                0, 1, 0, 1
        }, input.getGradient().toDoubleArrayCopy(), 1e-9);
        assertContainsOp(graph, Operation.OpType.UNFOLD2D);
        assertContainsOp(graph, Operation.OpType.FOLD2D);
        assertContainsOp(graph, Operation.OpType.ARGMAX);
        assertContainsOp(graph, Operation.OpType.SCATTER_ELEMENTS);
    }

    @Test
    void maxPool2dBackwardIgnoresPaddingWhenInputsAreNegative() {
        for (DataType dataType : new DataType[]{DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16}) {
            Tensor input = new Tensor(new double[]{
                    -4, -3,
                    -2, -1
            }, new int[]{1, 1, 2, 2}, null, "input", dataType);
            input.setRequiresGrad(true);

            Tensor loss = input.maxPool2d(
                    Pool2dOptions.square(2)
                            .withStride(1, 1)
                            .withPadding(1, 1)
            ).sum();
            CompiledGraph graph = CompiledGraph.compile(loss, CompileConfig.training());
            graph.prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

            assertArrayEquals(new double[]{
                    1, 2,
                    2, 4
            }, input.getGradient().toDoubleArrayCopy(), dataType == DataType.BFLOAT16 ? 1e-3 : 1e-6);
            assertContainsOp(graph, Operation.OpType.WHERE);
            assertContainsOp(graph, Operation.OpType.ARGMAX);
            assertContainsOp(graph, Operation.OpType.FOLD2D);
        }
    }

    @Test
    void avgPool2dForwardAndBackwardMatchExpectedValues() {
        Tensor forwardInput = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "input", DataType.FLOAT64);

        Tensor out = forwardInput.avgPool2d(Pool2dOptions.square(2));
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{1, 1, 2, 2}, out.getShape());
        assertArrayEquals(new double[]{
                3.5, 5.5,
                11.5, 13.5
        }, out.toDoubleArrayCopy(), 1e-9);

        Tensor backwardInput = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "input", DataType.FLOAT64);
        backwardInput.setRequiresGrad(true);

        Tensor loss = backwardInput.avgPool2d(Pool2dOptions.square(2)).sum();
        CompiledGraph backwardGraph = CompiledGraph.compile(loss, CompileConfig.training());
        backwardGraph.prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0.25, 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25, 0.25
        }, backwardInput.getGradient().toDoubleArrayCopy(), 1e-9);
        assertContainsOp(backwardGraph, Operation.OpType.UNFOLD2D);
        assertContainsOp(backwardGraph, Operation.OpType.FOLD2D);
    }

    @Test
    void avgPool2dCountIncludePadChangesBorderNormalization() {
        Tensor input = new Tensor(new double[]{4}, new int[]{1, 1, 1, 1}, null, "input", DataType.FLOAT64);

        Tensor excludePad = input.avgPool2d(Pool2dOptions.square(2).withStride(1, 1).withPadding(1, 1));
        CompiledGraph.compile(excludePad, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{4, 4, 4, 4}, excludePad.toDoubleArrayCopy(), 1e-9);

        Tensor includePad = input.avgPool2d(
                Pool2dOptions.square(2)
                        .withStride(1, 1)
                        .withPadding(1, 1)
                        .withCountIncludePad(true)
        );
        CompiledGraph.compile(includePad, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{1, 1, 1, 1}, includePad.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void pool2dSupportsFloat32AndBFloat16Execution() {
        Tensor input32 = new Tensor(new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "input32", DataType.FLOAT32);

        Tensor max32 = input32.maxPool2d(Pool2dOptions.square(2));
        CompiledGraph.compile(max32, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{6, 8, 14, 16}, max32.toDoubleArrayCopy(), 1e-6);

        short[] input16Data = new short[]{
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(1),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(2),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(3),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(4),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(5),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(6),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(7),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(8),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(9),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(10),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(11),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(12),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(13),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(14),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(15),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(16)
        };
        Tensor input16 = new Tensor(input16Data, new int[]{1, 1, 4, 4}, null, "input16", DataType.BFLOAT16);

        Tensor avg16 = input16.avgPool2d(Pool2dOptions.square(2));
        CompiledGraph.compile(avg16, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{3.5, 5.5, 11.5, 13.5}, avg16.toDoubleArrayCopy(), 1e-3);
    }

    @Test
    void pool2dRejectsAllPaddingWindowsAtGraphConstructionTime() {
        Tensor input = new Tensor(new double[]{1}, new int[]{1, 1, 1, 1}, null, "input", DataType.FLOAT64);

        assertThrows(IllegalArgumentException.class, () ->
                input.maxPool2d(Pool2dOptions.square(1).withStride(2, 2).withPadding(5, 0))
        );
        assertThrows(IllegalArgumentException.class, () ->
                input.avgPool2d(Pool2dOptions.square(1).withStride(2, 2).withPadding(0, 5))
        );
    }

    @Test
    void maxPool2dPreparedExecutionReusesWorkspaceAcrossRuns() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "input", DataType.FLOAT64);
        Tensor out = input.maxPool2d(Pool2dOptions.square(2));

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        execution.execute(ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{6, 8, 14, 16}, out.toDoubleArrayCopy(), 1e-9);

        input.setData(new double[]{
                16, 15, 14, 13,
                12, 11, 10, 9,
                8, 7, 6, 5,
                4, 3, 2, 1
        });

        execution.execute(ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{16, 14, 8, 6}, out.toDoubleArrayCopy(), 1e-9);
    }

    private static void assertContainsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        org.junit.jupiter.api.Assertions.assertTrue(containsOp(compiledGraph, opType), "Expected graph to contain " + opType);
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.program().compiledNodes().stream()
                .map(graph.model.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}
