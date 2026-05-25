package backend.cpu.kernels.elementwise.binary;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.storage.CpuStorageBindings;
import backend.cpu.kernels.storage.CpuStorageView;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class BinaryStorageLoopsTest {
    @Test
    void denseMemorySegmentLoopRunsBinaryKernelsByDtype() {
        float[] leftF32 = {1.0f, -2.0f, 3.5f};
        float[] rightF32 = {4.0f, 5.0f, -1.5f};
        float[] outF32 = new float[3];

        BinaryStorageLoops.runSegmentDense(new CpuSubKernel(), bindings(
                DataType.FLOAT32,
                MemorySegment.ofArray(leftF32),
                MemorySegment.ofArray(rightF32),
                MemorySegment.ofArray(outF32),
                3
        ));

        assertArrayEquals(new float[]{-3.0f, -7.0f, 5.0f}, outF32, 0.0f);

        double[] leftF64 = {1.0d, 4.0d, 9.0d};
        double[] rightF64 = {2.0d, 0.5d, -1.0d};
        double[] outF64 = new double[3];

        BinaryStorageLoops.runSegmentDense(new CpuPowTensorKernel(), bindings(
                DataType.FLOAT64,
                MemorySegment.ofArray(leftF64),
                MemorySegment.ofArray(rightF64),
                MemorySegment.ofArray(outF64),
                3
        ));

        assertArrayEquals(new double[]{1.0d, 2.0d, 1.0d / 9.0d}, outF64, 1.0e-12d);

        short[] leftBF16 = bf16(1.0f, -2.0f, 3.5f);
        short[] rightBF16 = bf16(4.0f, 5.0f, -1.5f);
        short[] outBF16 = new short[3];

        BinaryStorageLoops.runSegmentDense(new CpuMulKernel(), bindings(
                DataType.BFLOAT16,
                MemorySegment.ofArray(leftBF16),
                MemorySegment.ofArray(rightBF16),
                MemorySegment.ofArray(outBF16),
                3
        ));

        assertArrayEquals(new double[]{4.0d, -10.0d, -5.25d}, toDouble(outBF16), 0.0d);
    }

    private static CpuStorageBindings bindings(DataType dtype, MemorySegment left, MemorySegment right, MemorySegment output, int size) {
        return new CpuStorageBindings(
                List.of(view(dtype, left, size), view(dtype, right, size)),
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
