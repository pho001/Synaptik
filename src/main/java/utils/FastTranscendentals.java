package utils;

public final class FastTranscendentals {
    private static final boolean FORCE_EXACT_TRANSCENDENTALS =
            Boolean.parseBoolean(System.getProperty("cg.math.forceExactTranscendentals", "false"));

    private static final float FAST_EXP_F32_A = 12_102_203.1615614f;
    private static final float FAST_EXP_F32_B = 1_064_866_805.0f;

    private static final double FAST_EXP_F64_A = 1_512_775.3951951857d;
    private static final double FAST_EXP_F64_B = 1_072_632_447.0d;

    private FastTranscendentals() {}

    public static float fastExpF32(float x) {
        if (FORCE_EXACT_TRANSCENDENTALS) {
            return (float) Math.exp(x);
        }
        if (Float.isNaN(x)) {
            return Float.NaN;
        }
        if (x <= -87.0f) {
            return 0.0f;
        }
        if (x >= 88.0f) {
            return Float.POSITIVE_INFINITY;
        }
        int bits = (int) (x * FAST_EXP_F32_A + FAST_EXP_F32_B);
        return Float.intBitsToFloat(bits);
    }

    public static double fastExpF64(double x) {
        if (FORCE_EXACT_TRANSCENDENTALS) {
            return Math.exp(x);
        }
        if (Double.isNaN(x)) {
            return Double.NaN;
        }
        if (x <= -709.0d) {
            return 0.0d;
        }
        if (x >= 709.0d) {
            return Double.POSITIVE_INFINITY;
        }
        long bits = (long) (x * FAST_EXP_F64_A + FAST_EXP_F64_B);
        return Double.longBitsToDouble(bits << 32);
    }

    public static float fastTanhF32(float x) {
        if (FORCE_EXACT_TRANSCENDENTALS) {
            return (float) Math.tanh(x);
        }
        float ax = Math.abs(x);
        if (ax > 5.0f) {
            return Math.copySign(1.0f, x);
        }
        float x2 = x * x;
        return x * (27.0f + x2) / (27.0f + 9.0f * x2);
    }

    public static double fastTanhF64(double x) {
        if (FORCE_EXACT_TRANSCENDENTALS) {
            return Math.tanh(x);
        }
        double ax = Math.abs(x);
        if (ax > 5.0d) {
            return Math.copySign(1.0d, x);
        }
        double x2 = x * x;
        return x * (27.0d + x2) / (27.0d + 9.0d * x2);
    }
}
