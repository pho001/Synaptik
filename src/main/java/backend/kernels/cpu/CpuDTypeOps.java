package backend.kernels.cpu;

import tensor.DataType;
import tensor.Tensor;

public final class CpuDTypeOps {
    static final int MODE_F64 = 0;
    static final int MODE_F32 = 1;
    static final int MODE_F16 = 2;

    private CpuDTypeOps() {}

    public static int modeFor(Tensor node) {
        DataType dataType = node == null ? DataType.FLOAT64 : node.getDataType();
        if (dataType == null) return MODE_F64;
        return switch (dataType) {
            case FLOAT64 -> MODE_F64;
            case FLOAT32 -> MODE_F32;
            case FLOAT16 -> MODE_F16;
        };
    }

    public static boolean isF64(int mode) {
        return mode == MODE_F64;
    }

    public static double cast(double value, int mode) {
        return switch (mode) {
            case MODE_F64 -> value;
            case MODE_F32 -> (double) ((float) value);
            case MODE_F16 -> fromHalfBits(toHalfBits((float) value));
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
        return cast(Math.pow(cast(a, mode), cast(exponent, mode)), mode);
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

    public static float fromHalfBits(short halfBits) {
        return halfBitsToFloat(halfBits);
    }

    public static short toHalfBits(float value) {
        return floatToHalfBits(value);
    }

    private static float halfBitsToFloat(short halfBits) {
        int bits = halfBits & 0xFFFF;
        int sign = (bits >>> 15) & 0x1;
        int exponent = (bits >>> 10) & 0x1F;
        int mantissa = bits & 0x3FF;

        int floatSign = sign << 31;
        int floatExponent;
        int floatMantissa;

        if (exponent == 0) {
            if (mantissa == 0) {
                return Float.intBitsToFloat(floatSign);
            }
            int normalizedMantissa = mantissa;
            int adjustedExponent = -14;
            while ((normalizedMantissa & 0x400) == 0) {
                normalizedMantissa <<= 1;
                adjustedExponent--;
            }
            normalizedMantissa &= 0x3FF;
            floatExponent = (adjustedExponent + 127) << 23;
            floatMantissa = normalizedMantissa << 13;
            return Float.intBitsToFloat(floatSign | floatExponent | floatMantissa);
        }

        if (exponent == 0x1F) {
            floatExponent = 0xFF << 23;
            floatMantissa = mantissa << 13;
            return Float.intBitsToFloat(floatSign | floatExponent | floatMantissa);
        }

        floatExponent = ((exponent - 15) + 127) << 23;
        floatMantissa = mantissa << 13;
        return Float.intBitsToFloat(floatSign | floatExponent | floatMantissa);
    }

    private static short floatToHalfBits(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exponent = (bits >>> 23) & 0xFF;
        int mantissa = bits & 0x7FFFFF;

        if (exponent == 0xFF) {
            if (mantissa == 0) {
                return (short) (sign | 0x7C00);
            }
            int nanMantissa = mantissa >>> 13;
            if (nanMantissa == 0) {
                nanMantissa = 1;
            }
            return (short) (sign | 0x7C00 | nanMantissa);
        }

        int halfExponent = exponent - 127 + 15;
        if (halfExponent >= 0x1F) {
            return (short) (sign | 0x7C00);
        }
        if (halfExponent <= 0) {
            if (halfExponent < -10) {
                return (short) sign;
            }
            mantissa |= 0x800000;
            int shift = 14 - halfExponent;
            int halfMantissa = mantissa >>> shift;
            int remainder = mantissa & ((1 << shift) - 1);
            int halfway = 1 << (shift - 1);
            if (remainder > halfway || (remainder == halfway && (halfMantissa & 0x1) != 0)) {
                halfMantissa++;
            }
            return (short) (sign | halfMantissa);
        }

        int halfMantissa = mantissa >>> 13;
        int remainder = mantissa & 0x1FFF;
        if (remainder > 0x1000 || (remainder == 0x1000 && (halfMantissa & 0x1) != 0)) {
            halfMantissa++;
            if (halfMantissa == 0x400) {
                halfMantissa = 0;
                halfExponent++;
                if (halfExponent >= 0x1F) {
                    return (short) (sign | 0x7C00);
                }
            }
        }
        return (short) (sign | (halfExponent << 10) | halfMantissa);
    }
}
