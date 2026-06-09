package backend.cpu1.kernels.fused.codegen.support;

/**
 * Primitive helpers callable from generated cpu1 fused kernels.
 */
public final class Cpu1FusedGeneratedSupport {
    public static final int ABI_VERSION = 1;

    private Cpu1FusedGeneratedSupport() {
    }

    public static boolean bool(byte value) {
        return value != 0;
    }
}
