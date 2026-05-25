package backend.cpu.kernels.elementwise.unary.memorysegmentloops;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.elementwise.unary.CpuClampMinKernel;
import backend.cpu.kernels.elementwise.unary.CpuInvKernel;
import backend.cpu.kernels.elementwise.unary.CpuMulScalarKernel;
import backend.cpu.kernels.elementwise.unary.CpuNegKernel;
import backend.cpu.kernels.elementwise.unary.CpuReluKernel;
import backend.cpu.kernels.storage.CpuStorageBindings;
import backend.cpu.kernels.storage.CpuStorageView;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class UnaryMemorySegmentLoopsTest {
    @Test
    void denseMemorySegmentLoopRunsUnaryKernelsByDtype() {
        float[] inputF32 = {1.0f, -2.0f, 3.5f};
        float[] outF32 = new float[3];

        UnaryMemorySegmentLoops.runSegmentDense(new CpuNegKernel(), bindings(
                DataType.FLOAT32,
                MemorySegment.ofArray(inputF32),
                MemorySegment.ofArray(outF32),
                3
        ));

        assertArrayEquals(new float[]{-1.0f, 2.0f, -3.5f}, outF32, 0.0f);

        double[] inputF64 = {2.0d, -4.0d, 0.5d};
        double[] outF64 = new double[3];

        UnaryMemorySegmentLoops.runSegmentDense(new CpuInvKernel(), bindings(
                DataType.FLOAT64,
                MemorySegment.ofArray(inputF64),
                MemorySegment.ofArray(outF64),
                3
        ));

        assertArrayEquals(new double[]{0.5d, -0.25d, 2.0d}, outF64, 0.0d);

        short[] inputBF16 = bf16(1.0f, -2.0f, 3.5f);
        short[] outBF16 = new short[3];

        UnaryMemorySegmentLoops.runSegmentDense(new CpuReluKernel(), bindings(
                DataType.BFLOAT16,
                MemorySegment.ofArray(inputBF16),
                MemorySegment.ofArray(outBF16),
                3
        ));

        assertArrayEquals(new double[]{1.0d, 0.0d, 3.5d}, toDouble(outBF16), 0.0d);
    }

    @Test
    void denseMemorySegmentLoopRunsScalarUnaryKernelsByDtype() {
        float[] inputF32 = {1.0f, -2.0f, 3.5f};
        float[] outF32 = new float[3];

        UnaryMemorySegmentLoops.runSegmentDense(new CpuMulScalarKernel(), 0.25d, 0.25f, bindings(
                DataType.FLOAT32,
                MemorySegment.ofArray(inputF32),
                MemorySegment.ofArray(outF32),
                3
        ));

        assertArrayEquals(new float[]{0.25f, -0.5f, 0.875f}, outF32, 0.0f);

        short[] inputBF16 = bf16(1.0f, -2.0f, 3.5f);
        short[] outBF16 = new short[3];

        UnaryMemorySegmentLoops.runSegmentDense(new CpuClampMinKernel(), 0.0d, 0.0f, bindings(
                DataType.BFLOAT16,
                MemorySegment.ofArray(inputBF16),
                MemorySegment.ofArray(outBF16),
                3
        ));

        assertArrayEquals(new double[]{1.0d, 0.0d, 3.5d}, toDouble(outBF16), 0.0d);
    }

    private static CpuStorageBindings bindings(DataType dtype, MemorySegment input, MemorySegment output, int size) {
        return new CpuStorageBindings(
                List.of(view(dtype, input, size)),
                view(dtype, output, size)
        );
    }

    private static CpuStorageView view(DataType dtype, MemorySegment segment, int size) {
        return CpuStorageView.segment(dtype, segment, new int[]{size}, new int[]{1}, 0, size);
    }

    private static short[] bf16(float... values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = CpuDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }

    private static double[] toDouble(short[] values) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = CpuDTypeOps.fromBFloat16Bits(values[i]);
        }
        return out;
    }
}
