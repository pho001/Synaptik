package backend.cpu1.kernels.fused.codegen.support;

/**
 * Math helpers callable from generated cpu1 fused kernels.
 */
public final class Cpu1FusedMathSupport {
    private Cpu1FusedMathSupport() {
    }

    public static float reluF32(float value) {
        return Math.max(0.0f, value);
    }

    public static double reluF64(double value) {
        return Math.max(0.0d, value);
    }

    public static float minF32(float left, float right) {
        return Math.min(left, right);
    }

    public static double minF64(double left, double right) {
        return Math.min(left, right);
    }

    public static float maxF32(float left, float right) {
        return Math.max(left, right);
    }

    public static double maxF64(double left, double right) {
        return Math.max(left, right);
    }

    public static float absF32(float value) {
        return Math.abs(value);
    }

    public static double absF64(double value) {
        return Math.abs(value);
    }
}
