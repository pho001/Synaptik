package backend.cpu.kernels.elementwise.unary.support;

public final class CpuPowSupport {
    private CpuPowSupport() {}

    public static double applyF64(double value, double exponent) {
        if (exponent == 0.0d) {
            return 1.0d;
        }
        if (exponent == 1.0d) {
            return value;
        }
        if (exponent == 2.0d) {
            return value * value;
        }
        if (exponent == 0.5d) {
            return Math.sqrt(value);
        }
        if (exponent == -1.0d) {
            return 1.0d / value;
        }
        return Math.pow(value, exponent);
    }

    public static float applyF32(float value, float exponent) {
        if (exponent == 0.0f) {
            return 1.0f;
        }
        if (exponent == 1.0f) {
            return value;
        }
        if (exponent == 2.0f) {
            return value * value;
        }
        if (exponent == 0.5f) {
            return (float) Math.sqrt(value);
        }
        if (exponent == -1.0f) {
            return 1.0f / value;
        }
        return (float) Math.pow(value, exponent);
    }
}
