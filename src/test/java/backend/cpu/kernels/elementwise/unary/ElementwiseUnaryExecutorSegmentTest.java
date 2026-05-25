package backend.cpu.kernels.elementwise.unary;

import tensor.dtype.TensorDTypeOps;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ElementwiseUnaryExecutorSegmentTest {
    @Test
    void denseMemorySegmentExecutionRunsUnaryKernelOperationsByDtype() {
        float[] inputF32 = {1.0f, -2.0f, 3.5f};
        float[] outF32 = new float[3];

        ElementwiseUnaryExecutor.runDenseSegment(
                new CpuNegKernel(),
                DataType.FLOAT32,
                MemorySegment.ofArray(inputF32),
                MemorySegment.ofArray(outF32),
                3,
                null
        );

        assertArrayEquals(new float[]{-1.0f, 2.0f, -3.5f}, outF32, 0.0f);

        double[] inputF64 = {2.0d, -4.0d, 0.5d};
        double[] outF64 = new double[3];

        ElementwiseUnaryExecutor.runDenseSegment(
                new CpuInvKernel(),
                DataType.FLOAT64,
                MemorySegment.ofArray(inputF64),
                MemorySegment.ofArray(outF64),
                3,
                null
        );

        assertArrayEquals(new double[]{0.5d, -0.25d, 2.0d}, outF64, 0.0d);

        short[] inputBF16 = bf16(1.0f, -2.0f, 3.5f);
        short[] outBF16 = new short[3];

        ElementwiseUnaryExecutor.runDenseSegment(
                new CpuReluKernel(),
                DataType.BFLOAT16,
                MemorySegment.ofArray(inputBF16),
                MemorySegment.ofArray(outBF16),
                3,
                null
        );

        assertArrayEquals(new double[]{1.0d, 0.0d, 3.5d}, toDouble(outBF16), 0.0d);
    }

    @Test
    void denseMemorySegmentExecutionRunsScalarUnaryKernelOperationsByDtype() {
        float[] inputF32 = {1.0f, -2.0f, 3.5f};
        float[] outF32 = new float[3];

        ElementwiseUnaryExecutor.runDenseSegment(
                new CpuMulScalarKernel(),
                0.25d,
                0.25f,
                DataType.FLOAT32,
                MemorySegment.ofArray(inputF32),
                MemorySegment.ofArray(outF32),
                3,
                null
        );

        assertArrayEquals(new float[]{0.25f, -0.5f, 0.875f}, outF32, 0.0f);

        short[] inputBF16 = bf16(1.0f, -2.0f, 3.5f);
        short[] outBF16 = new short[3];

        ElementwiseUnaryExecutor.runDenseSegment(
                new CpuClampMinKernel(),
                0.0d,
                0.0f,
                DataType.BFLOAT16,
                MemorySegment.ofArray(inputBF16),
                MemorySegment.ofArray(outBF16),
                3,
                null
        );

        assertArrayEquals(new double[]{1.0d, 0.0d, 3.5d}, toDouble(outBF16), 0.0d);
    }

    private static short[] bf16(float... values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }

    private static double[] toDouble(short[] values) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(values[i]);
        }
        return out;
    }
}
