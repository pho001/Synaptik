import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.options.Conv2dOptions;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class Conv2dExecutionTest {

    @Test
    void conv2dForwardWithoutBiasMatchesExpectedValues() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[]{
                1, 0,
                0, -1
        }, new int[]{1, 1, 2, 2}, null, "weight", DataType.FLOAT64);

        Tensor out = input.conv2d(weight, Conv2dOptions.defaults());
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{1, 1, 2, 2}, out.getShape());
        assertArrayEquals(new double[]{
                -4, -4,
                -4, -4
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void conv2dForwardWithBiasStrideAndPaddingMatchesExpectedValues() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[]{
                1, 1,
                1, 1
        }, new int[]{1, 1, 2, 2}, null, "weight", DataType.FLOAT64);
        Tensor bias = new Tensor(new double[]{0.5}, new int[]{1}, null, "bias", DataType.FLOAT64);

        Tensor out = input.conv2d(weight, bias, Conv2dOptions.defaults().withStride(2, 2).withPadding(1, 1));
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{1, 1, 2, 2}, out.getShape());
        assertArrayEquals(new double[]{
                1.5, 5.5,
                11.5, 28.5
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void conv2dBackwardInputAndWeightMatchExpectedForSumLoss() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[]{
                1, 0,
                0, -1
        }, new int[]{1, 1, 2, 2}, null, "weight", DataType.FLOAT64);
        input.setRequiresGrad(true);
        weight.setRequiresGrad(true);

        Tensor loss = input.conv2d(weight, Conv2dOptions.defaults()).sum();
        CompiledGraph.compile(loss, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                1, 1, 0,
                1, 0, -1,
                0, -1, -1
        }, input.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{
                12, 16,
                24, 28
        }, weight.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void conv2dGroupedForwardMatchesExpectedValues() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3, 4,
                10, 20, 30, 40
        }, new int[]{1, 2, 1, 4}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[]{
                1, 1,
                2, 0
        }, new int[]{2, 1, 1, 2}, null, "weight", DataType.FLOAT64);

        Tensor out = input.conv2d(weight, Conv2dOptions.defaults().withGroups(2));
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{1, 2, 1, 3}, out.getShape());
        assertArrayEquals(new double[]{
                3, 5, 7,
                20, 40, 60
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void conv2dSupportsFloat32AndBFloat16Execution() {
        Tensor input32 = new Tensor(new float[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, null, "input32", DataType.FLOAT32);
        Tensor weight32 = new Tensor(new float[]{
                1, 0,
                0, -1
        }, new int[]{1, 1, 2, 2}, null, "weight32", DataType.FLOAT32);
        Tensor out32 = input32.conv2d(weight32, Conv2dOptions.defaults());
        CompiledGraph.compile(out32, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{-4, -4, -4, -4}, out32.toDoubleArrayCopy(), 1e-6);

        short[] input16Data = new short[]{
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(1),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(2),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(3),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(4),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(5),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(6),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(7),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(8),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(9)
        };
        short[] weight16Data = new short[]{
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(1),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(0),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(0),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(-1)
        };
        Tensor input16 = new Tensor(input16Data, new int[]{1, 1, 3, 3}, null, "input16", DataType.BFLOAT16);
        Tensor weight16 = new Tensor(weight16Data, new int[]{1, 1, 2, 2}, null, "weight16", DataType.BFLOAT16);
        Tensor out16 = input16.conv2d(weight16, Conv2dOptions.defaults());
        CompiledGraph.compile(out16, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{-4, -4, -4, -4}, out16.toDoubleArrayCopy(), 1e-3);
    }
}
