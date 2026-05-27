package backend.cpu.kernels.elementwise.unary;

import backend.cpu.storage.CpuStorageView;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class StorageAwareScalarUnaryElementwiseKernelSegmentTest {
    @Test
    void denseMemorySegmentExecutionRunsScalarUnaryKernelOperationsByDtype() {
        float[] inputF32 = {1.0f, -2.0f, 3.5f};
        float[] outF32 = new float[3];

        new CpuMulScalarKernel().runSegmentF32(
                MemorySegment.ofArray(inputF32),
                0.25f,
                MemorySegment.ofArray(outF32),
                0,
                3
        );

        assertArrayEquals(new float[]{0.25f, -0.5f, 0.875f}, outF32, 0.0f);

        short[] inputBF16 = bf16(1.0f, -2.0f, 3.5f);
        short[] outBF16 = new short[3];

        new CpuClampMinKernel().runSegmentBF16(
                MemorySegment.ofArray(inputBF16),
                0.0f,
                MemorySegment.ofArray(outBF16),
                0,
                3
        );

        assertArrayEquals(new double[]{1.0d, 0.0d, 3.5d}, toDouble(outBF16), 0.0d);
    }

    @Test
    void indexedSegmentLoopHandlesStridedSegmentInputView() {
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float[] out = new float[6];
        CpuStorageView inputView = CpuStorageView.segment(
                DataType.FLOAT32,
                MemorySegment.ofArray(input),
                new int[]{3, 2},
                new int[]{1, 3},
                0,
                6
        );
        CpuStorageView outView = CpuStorageView.segment(
                DataType.FLOAT32,
                MemorySegment.ofArray(out),
                new int[]{3, 2},
                new int[]{2, 1},
                0,
                6
        );
        var layout = StorageAwareScalarUnaryElementwiseKernel.ScalarUnaryStorageLayout.from(inputView, outView);

        new CpuMulScalarKernel().runIndexedSegmentF32(
                inputView.requireSegment(),
                -2.0f,
                outView.requireSegment(),
                layout,
                0,
                6
        );

        assertArrayEquals(new float[]{-2.0f, -8.0f, -4.0f, -10.0f, -6.0f, -12.0f}, out, 0.0f);
    }

    @Test
    void indexedMixedLoopHandlesArrayInputAndSegmentOutput() {
        float[] input = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
        float[] out = new float[6];
        CpuStorageView inputView = CpuStorageView.array(
                DataType.FLOAT32,
                input,
                new int[]{3, 2},
                new int[]{1, 3},
                0,
                6
        );
        CpuStorageView outView = CpuStorageView.segment(
                DataType.FLOAT32,
                MemorySegment.ofArray(out),
                new int[]{3, 2},
                new int[]{2, 1},
                0,
                6
        );
        var layout = StorageAwareScalarUnaryElementwiseKernel.ScalarUnaryStorageLayout.from(inputView, outView);

        new CpuClampMaxKernel().runIndexedMixedF32(inputView, 0.0f, outView, layout, 0, 6);

        assertArrayEquals(new float[]{-1.0f, 0.0f, 0.0f, -5.0f, -3.0f, 0.0f}, out, 0.0f);
    }

    @Test
    void indexedArrayBfloat16ToFloatUsesContinuationValues() {
        short[] input = bf16(1.0f, 2.0f, 3.0f, 4.0f);
        float[] continuation = {1.25f, 2.25f, 3.25f, 4.25f};
        float[] out = new float[4];
        CpuStorageView inputView = CpuStorageView.array(
                DataType.BFLOAT16,
                input,
                new int[]{2, 2},
                new int[]{1, 2},
                0,
                4
        );
        CpuStorageView outView = CpuStorageView.array(
                DataType.BFLOAT16,
                new short[4],
                new int[]{2, 2},
                new int[]{2, 1},
                0,
                4
        );
        var layout = StorageAwareScalarUnaryElementwiseKernel.ScalarUnaryStorageLayout.from(inputView, outView);

        new CpuPowKernel().runIndexedArrayBF16ToFloat(input, continuation, 2.0f, out, layout, 0, 4);

        assertArrayEquals(
                new float[]{1.5625f, 10.5625f, 5.0625f, 18.0625f},
                out,
                0.0f
        );
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
