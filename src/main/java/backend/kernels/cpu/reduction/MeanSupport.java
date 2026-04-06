package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuDTypeOps;
import tensor.Tensor;

public final class MeanSupport {
    private MeanSupport() {
    }

    public static void divideF64(Tensor node, int divisor) {
        double[] out = node.getFloat64Data();
        double scale = 1.0 / divisor;
        int baseOffset = node.getStorageOffsetUnsafe();
        for (int i = 0; i < node.getFlatDataSize(); i++) {
            out[baseOffset + i] *= scale;
        }
    }

    public static void divideF32(Tensor node, int divisor) {
        float[] out = node.getFloat32Data();
        float scale = 1.0f / divisor;
        int baseOffset = node.getStorageOffsetUnsafe();
        for (int i = 0; i < node.getFlatDataSize(); i++) {
            out[baseOffset + i] *= scale;
        }
    }

    public static void divideF16(Tensor node, int divisor) {
        short[] out = node.getBFloat16Data();
        float scale = 1.0f / divisor;
        int baseOffset = node.getStorageOffsetUnsafe();
        for (int i = 0; i < node.getFlatDataSize(); i++) {
            int idx = baseOffset + i;
            float value = CpuDTypeOps.fromBFloat16Bits(out[idx]) * scale;
            out[idx] = CpuDTypeOps.toBFloat16Bits(value);
        }
    }
}
