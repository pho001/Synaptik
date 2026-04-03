package tensor;

public class Float16Storage implements TensorStorage {
    private final short[] data;

    public Float16Storage(int size) {
        this.data = new short[size];
    }

    public Float16Storage(short[] data) {
        this.data = data;
    }

    @Override
    public DataType getType() {
        return DataType.FLOAT16;
    }

    @Override
    public int getSize() {
        return data.length;
    }

    public short getFloat16BitsAt(int flatIndex) {
        return data[flatIndex];
    }

    public void setFloat16BitsAt(int flatIndex, short value) {
        data[flatIndex] = value;
    }

    public short[] getShortArray() {
        return data;
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
