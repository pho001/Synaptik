package backend.cpu.kernels.elementwise.binary.array;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuExecutionMode;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.storage.CpuStorageBindings;
import backend.cpu.kernels.storage.CpuStorageView;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class AddArrayLoopsTest {
    @Test
    void denseArrayLoopUsesExistingSpecializedAddKernels() {
        double[] leftF64 = {1.0, -2.0, 3.5};
        double[] rightF64 = {4.0, 5.0, -1.5};
        double[] outF64 = new double[3];

        AddArrayLoops.runDense(bindings(DataType.FLOAT64, leftF64, rightF64, outF64), scalarHints(3), null, null);

        assertArrayEquals(new double[]{5.0, 3.0, 2.0}, outF64, 0.0);

        float[] leftF32 = {1.0f, -2.0f, 3.5f};
        float[] rightF32 = {4.0f, 5.0f, -1.5f};
        float[] outF32 = new float[3];

        AddArrayLoops.runDense(bindings(DataType.FLOAT32, leftF32, rightF32, outF32), scalarHints(3), null, null);

        assertArrayEquals(new float[]{5.0f, 3.0f, 2.0f}, outF32, 0.0f);

        short[] leftBF16 = bf16(1.0f, -2.0f, 3.5f);
        short[] rightBF16 = bf16(4.0f, 5.0f, -1.5f);
        short[] outBF16 = new short[3];

        AddArrayLoops.runDense(bindings(DataType.BFLOAT16, leftBF16, rightBF16, outBF16), scalarHints(3), null, null);

        assertArrayEquals(new double[]{5.0, 3.0, 2.0}, toDouble(outBF16), 0.0);
    }

    private static CpuStorageBindings bindings(DataType dtype, Object left, Object right, Object output) {
        int size = java.lang.reflect.Array.getLength(output);
        return new CpuStorageBindings(
                List.of(arrayView(dtype, left, size), arrayView(dtype, right, size)),
                arrayView(dtype, output, size)
        );
    }

    private static CpuStorageView arrayView(DataType dtype, Object array, int size) {
        return CpuStorageView.array(dtype, array, new int[]{size}, new int[]{1}, 0, size);
    }

    private static ResolvedDispatchHints scalarHints(int size) {
        return new ResolvedDispatchHints(size, CpuExecutionMode.SCALAR, size, size, 1, 1, false);
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
