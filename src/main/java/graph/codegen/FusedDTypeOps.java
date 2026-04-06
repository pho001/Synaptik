package graph.codegen;

import utils.FastExp;

public final class FusedDTypeOps {
    public static final int MODE_F64 = 0;
    public static final int MODE_F32 = 1;
    public static final int MODE_BF16 = 2;

    private FusedDTypeOps() {}

    public static double add(double a, double b, int mode) {
        return cast(cast(a, mode) + cast(b, mode), mode);
    }

    public static double sub(double a, double b, int mode) {
        return cast(cast(a, mode) - cast(b, mode), mode);
    }

    public static double mul(double a, double b, int mode) {
        return cast(cast(a, mode) * cast(b, mode), mode);
    }

    public static double div(double a, double b, int mode) {
        return cast(cast(a, mode) / cast(b, mode), mode);
    }

    public static double min(double a, double b, int mode) {
        return cast(Math.min(cast(a, mode), cast(b, mode)), mode);
    }

    public static double max(double a, double b, int mode) {
        return cast(Math.max(cast(a, mode), cast(b, mode)), mode);
    }

    public static double neg(double a, int mode) {
        return cast(-cast(a, mode), mode);
    }

    public static double inv(double a, int mode) {
        return cast(1.0 / cast(a, mode), mode);
    }

    public static double log(double a, int mode) {
        return cast(Math.log(cast(a, mode)), mode);
    }

    public static double exp(double a, int mode) {
        return exp(a, mode, false);
    }

    public static double exp(double a, int mode, boolean useFastExpApprox) {
        if (useFastExpApprox) {
            return fastExp(a, mode);
        }
        return cast(Math.exp(cast(a, mode)), mode);
    }

    public static double fastExp(double a, int mode) {
        return switch (mode) {
            case MODE_F64 -> FastExp.fastExpF64(a);
            case MODE_F32 -> (double) FastExp.fastExpF32((float) a);
            case MODE_BF16 -> cast(FastExp.fastExpF32((float) a), MODE_BF16);
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static double tanh(double a, int mode) {
        return tanh(a, mode, false);
    }

    public static double tanh(double a, int mode, boolean useFastTanhApprox) {
        if (useFastTanhApprox) {
            return fastTanh(a, mode);
        }
        return cast(Math.tanh(cast(a, mode)), mode);
    }

    public static double fastTanh(double a, int mode) {
        return switch (mode) {
            case MODE_F64 -> FastExp.fastTanhF64(a);
            case MODE_F32 -> (double) FastExp.fastTanhF32((float) a);
            case MODE_BF16 -> cast(FastExp.fastTanhF32((float) a), MODE_BF16);
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static double sqrt(double a, int mode) {
        return cast(Math.sqrt(cast(a, mode)), mode);
    }

    public static double abs(double a, int mode) {
        return cast(Math.abs(cast(a, mode)), mode);
    }

    public static double pow(double a, double exponent, int mode) {
        return cast(Math.pow(cast(a, mode), exponent), mode);
    }

    public static double mulScalar(double a, double scalar, int mode) {
        return cast(cast(a, mode) * cast(scalar, mode), mode);
    }

    public static double relu(double a, int mode) {
        return cast(Math.max(cast(a, mode), 0.0), mode);
    }

    public static double clampMin(double a, double minValue, int mode) {
        return cast(Math.max(cast(a, mode), cast(minValue, mode)), mode);
    }

    public static double clampMax(double a, double maxValue, int mode) {
        return cast(Math.min(cast(a, mode), cast(maxValue, mode)), mode);
    }

    public static double sigmoid(double a, int mode) {
        double x = cast(a, mode);
        return cast(1.0 / (1.0 + Math.exp(-x)), mode);
    }

    public static double noop(double a, int mode) {
        return cast(a, mode);
    }

    public static double cast(double value, int mode) {
        return switch (mode) {
            case MODE_F64 -> value;
            case MODE_F32 -> (double) ((float) value);
            case MODE_BF16 -> bfloat16BitsToFloat(floatToBFloat16Bits((float) value));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    private static float bfloat16BitsToFloat(short halfBits) {
        int floatBits = (halfBits & 0xFFFF) << 16;
        return Float.intBitsToFloat(floatBits);
    }

    private static short floatToBFloat16Bits(float value) {
        int bits = Float.floatToIntBits(value);
        if (Float.isNaN(value)) {
            return (short) 0x7FC0;
        }
        int upper = bits >>> 16;
        int lower = bits & 0xFFFF;
        if (lower > 0x8000 || (lower == 0x8000 && (upper & 0x1) != 0)) {
            upper++;
        }
        return (short) upper;
    }
}
