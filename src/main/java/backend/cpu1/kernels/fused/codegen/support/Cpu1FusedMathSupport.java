package backend.cpu1.kernels.fused.codegen.support;

import utils.FastTranscendentals;
import utils.SpecialFunctions;

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

    public static float expF32(float value) {
        return (float) Math.exp(value);
    }

    public static double expF64(double value) {
        return Math.exp(value);
    }

    public static float fastExpF32(float value) {
        return FastTranscendentals.fastExpF32(value);
    }

    public static double fastExpF64(double value) {
        return FastTranscendentals.fastExpF64(value);
    }

    public static float logF32(float value) {
        return (float) Math.log(value);
    }

    public static double logF64(double value) {
        return Math.log(value);
    }

    public static float tanhF32(float value) {
        return (float) Math.tanh(value);
    }

    public static double tanhF64(double value) {
        return Math.tanh(value);
    }

    public static float fastTanhF32(float value) {
        return FastTranscendentals.fastTanhF32(value);
    }

    public static double fastTanhF64(double value) {
        return FastTranscendentals.fastTanhF64(value);
    }

    public static float erfF32(float value) {
        return SpecialFunctions.erf(value);
    }

    public static double erfF64(double value) {
        return SpecialFunctions.erf(value);
    }

    public static float sqrtF32(float value) {
        return (float) Math.sqrt(value);
    }

    public static double sqrtF64(double value) {
        return Math.sqrt(value);
    }

    public static float sigmoidF32(float value) {
        return 1.0f / (1.0f + (float) Math.exp(-value));
    }

    public static double sigmoidF64(double value) {
        return 1.0d / (1.0d + Math.exp(-value));
    }

    public static float floorF32(float value) {
        return (float) Math.floor(value);
    }

    public static double floorF64(double value) {
        return Math.floor(value);
    }

    public static float ceilF32(float value) {
        return (float) Math.ceil(value);
    }

    public static double ceilF64(double value) {
        return Math.ceil(value);
    }

    public static float signF32(float value) {
        return value > 0.0f ? 1.0f : (value < 0.0f ? -1.0f : 0.0f);
    }

    public static double signF64(double value) {
        return value > 0.0d ? 1.0d : (value < 0.0d ? -1.0d : 0.0d);
    }

    public static float powF32(float value, float exponent) {
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

    public static double powF64(double value, double exponent) {
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
}
