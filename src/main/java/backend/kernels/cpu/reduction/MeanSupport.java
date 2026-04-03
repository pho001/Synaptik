package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuDTypeOps;
import tensor.Tensor;

public final class MeanSupport {
    private MeanSupport() {
    }

    public static void divideF64(Tensor node, int divisor) {
        double[] out = node.getFloat64Data();
        double scale = 1.0 / divisor;
        for (int i = 0; i < out.length; i++) {
            out[i] *= scale;
        }
    }

    public static void divideF32(Tensor node, int divisor) {
        float[] out = node.getFloat32Data();
        float scale = 1.0f / divisor;
        for (int i = 0; i < out.length; i++) {
            out[i] *= scale;
        }
    }

    public static void divideF16(Tensor node, int divisor) {
        short[] out = node.getFloat16Data();
        float scale = 1.0f / divisor;
        for (int i = 0; i < out.length; i++) {
            float value = CpuDTypeOps.fromHalfBits(out[i]) * scale;
            out[i] = CpuDTypeOps.toHalfBits(value);
        }
    }
}
