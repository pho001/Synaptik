package backend.cpu.kernels;

import tensor.DataType;
import tensor.BFloat16Bits;
import tensor.Tensor;

public final class CpuDTypeOps {
    static final int MODE_F64 = 0;
    static final int MODE_F32 = 1;
    static final int MODE_BF16 = 2;

    private CpuDTypeOps() {}

    public static int modeFor(Tensor node) {
        DataType dataType = node == null ? DataType.FLOAT64 : node.getDataType();
        if (dataType == null) return MODE_F64;
        return switch (dataType) {
            case FLOAT64 -> MODE_F64;
            case FLOAT32 -> MODE_F32;
            case BFLOAT16 -> MODE_BF16;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("INT32/INT64/BOOL are not supported by CpuDTypeOps.");
        };
    }

    public static boolean isF64(int mode) {
        return mode == MODE_F64;
    }

    public static double cast(double value, int mode) {
        return switch (mode) {
            case MODE_F64 -> value;
            case MODE_F32 -> (double) ((float) value);
            case MODE_BF16 -> fromBFloat16Bits(toBFloat16Bits((float) value));
            default -> throw new IllegalArgumentException("Unsupported dtype mode: " + mode);
        };
    }

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
        return cast(Math.exp(cast(a, mode)), mode);
    }

    public static double tanh(double a, int mode) {
        return cast(Math.tanh(cast(a, mode)), mode);
    }

    public static double sqrt(double a, int mode) {
        return cast(Math.sqrt(cast(a, mode)), mode);
    }

    public static double pow(double a, double exponent, int mode) {
        double x = cast(a, mode);
        double e = cast(exponent, mode);
        if (e == 0.0d) {
            return cast(1.0d, mode);
        }
        if (e == 1.0d) {
            return x;
        }
        if (e == 2.0d) {
            return cast(x * x, mode);
        }
        if (e == 0.5d) {
            return cast(Math.sqrt(x), mode);
        }
        if (e == -1.0d) {
            return cast(1.0d / x, mode);
        }
        return cast(Math.pow(x, e), mode);
    }

    public static double mulScalar(double a, double scalar, int mode) {
        return cast(cast(a, mode) * cast(scalar, mode), mode);
    }

    public static double relu(double a, int mode) {
        return cast(Math.max(cast(a, mode), 0.0), mode);
    }

    public static double sigmoid(double a, int mode) {
        double x = cast(a, mode);
        return cast(1.0 / (1.0 + Math.exp(-x)), mode);
    }

    public static float fromBFloat16Bits(short halfBits) {
        return BFloat16Bits.toFloat(halfBits);
    }

    public static short toBFloat16Bits(float value) {
        return BFloat16Bits.fromFloat(value);
    }
}
