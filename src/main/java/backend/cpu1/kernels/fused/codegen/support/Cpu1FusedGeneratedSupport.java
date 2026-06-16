package backend.cpu1.kernels.fused.codegen.support;

import tensor.dtype.TensorDTypeOps;

/**
 * Primitive helpers callable from generated cpu1 fused kernels.
 */
public final class Cpu1FusedGeneratedSupport {
    public static final int ABI_VERSION = 3;

    private Cpu1FusedGeneratedSupport() {
    }

    public static boolean bool(byte value) {
        return value != 0;
    }

    public static float bf16ToFloat(short value) {
        return TensorDTypeOps.fromBFloat16Bits(value);
    }

    public static short floatToBf16(float value) {
        return TensorDTypeOps.toBFloat16Bits(value);
    }
}
