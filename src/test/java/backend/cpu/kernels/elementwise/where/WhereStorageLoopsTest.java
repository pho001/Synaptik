package backend.cpu.kernels.elementwise.where;

import tensor.dtype.TensorDTypeOps;
import backend.cpu.storage.CpuStorageBindings;
import backend.cpu.storage.CpuStorageView;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class WhereStorageLoopsTest {
    @Test
    void denseMemorySegmentLoopRunsWhereByDtype() {
        byte[] condition = {1, 0, 1};

        float[] trueF32 = {1.0f, 2.0f, 3.0f};
        float[] falseF32 = {-1.0f, -2.0f, -3.0f};
        float[] outF32 = new float[3];

        WhereStorageLoops.runSegmentDense(new CpuWhereKernel(), condition, bindings(
                DataType.FLOAT32,
                MemorySegment.ofArray(trueF32),
                MemorySegment.ofArray(falseF32),
                MemorySegment.ofArray(outF32),
                3
        ));

        assertArrayEquals(new float[]{1.0f, -2.0f, 3.0f}, outF32, 0.0f);

        double[] trueF64 = {1.0d, 2.0d, 3.0d};
        double[] falseF64 = {-1.0d, -2.0d, -3.0d};
        double[] outF64 = new double[3];

        WhereStorageLoops.runSegmentDense(new CpuWhereKernel(), condition, bindings(
                DataType.FLOAT64,
                MemorySegment.ofArray(trueF64),
                MemorySegment.ofArray(falseF64),
                MemorySegment.ofArray(outF64),
                3
        ));

        assertArrayEquals(new double[]{1.0d, -2.0d, 3.0d}, outF64, 0.0d);

        short[] trueBF16 = bf16(1.0f, 2.0f, 3.5f);
        short[] falseBF16 = bf16(-1.0f, -2.0f, -3.5f);
        short[] outBF16 = new short[3];

        WhereStorageLoops.runSegmentDense(new CpuWhereKernel(), condition, bindings(
                DataType.BFLOAT16,
                MemorySegment.ofArray(trueBF16),
                MemorySegment.ofArray(falseBF16),
                MemorySegment.ofArray(outBF16),
                3
        ));

        assertArrayEquals(new double[]{1.0d, -2.0d, 3.5d}, toDouble(outBF16), 0.0d);
    }

    private static CpuStorageBindings bindings(DataType dtype, MemorySegment ifTrue, MemorySegment ifFalse, MemorySegment output, int size) {
        return new CpuStorageBindings(
                List.of(view(dtype, ifTrue, size), view(dtype, ifFalse, size)),
                view(dtype, output, size)
        );
    }

    private static CpuStorageView view(DataType dtype, MemorySegment segment, int size) {
        return CpuStorageView.segment(dtype, segment, new int[]{size}, new int[]{1}, 0, size);
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
