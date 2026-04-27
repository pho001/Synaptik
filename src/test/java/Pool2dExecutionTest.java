import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
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
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

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
        CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0, 0, 0, 0,
                0, 1, 0, 1,
                0, 0, 0, 0,
                0, 1, 0, 1
        }, input.getGradient().toDoubleArrayCopy(), 1e-9);
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
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

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
        CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0.25, 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25, 0.25
        }, backwardInput.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void avgPool2dCountIncludePadChangesBorderNormalization() {
        Tensor input = new Tensor(new double[]{4}, new int[]{1, 1, 1, 1}, null, "input", DataType.FLOAT64);

        Tensor excludePad = input.avgPool2d(Pool2dOptions.square(2).withStride(1, 1).withPadding(1, 1));
        CompiledGraph.compile(excludePad, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{4, 4, 4, 4}, excludePad.toDoubleArrayCopy(), 1e-9);

        Tensor includePad = input.avgPool2d(
                Pool2dOptions.square(2)
                        .withStride(1, 1)
                        .withPadding(1, 1)
                        .withCountIncludePad(true)
        );
        CompiledGraph.compile(includePad, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
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
        CompiledGraph.compile(max32, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{6, 8, 14, 16}, max32.toDoubleArrayCopy(), 1e-6);

        short[] input16Data = new short[]{
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(1),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(2),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(3),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(4),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(5),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(6),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(7),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(8),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(9),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(10),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(11),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(12),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(13),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(14),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(15),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(16)
        };
        Tensor input16 = new Tensor(input16Data, new int[]{1, 1, 4, 4}, null, "input16", DataType.BFLOAT16);

        Tensor avg16 = input16.avgPool2d(Pool2dOptions.square(2));
        CompiledGraph.compile(avg16, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
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

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
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
}
