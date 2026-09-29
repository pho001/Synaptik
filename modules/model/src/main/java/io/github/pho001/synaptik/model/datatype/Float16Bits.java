package io.github.pho001.synaptik.model.datatype;

/**
 * Converts Java floating-point values to and from raw IEEE-754 binary16 bit patterns.
 *
 * <p>Binary16 has one sign bit, a five-bit exponent, and ten fraction bits. Finite conversion
 * uses round-to-nearest with ties-to-even; signed zero and infinities retain sign and
 * classification. Every Java NaN converts to canonical quiet binary16 NaN {@code 0x7E00}, while
 * raw binary16 NaN payloads are preserved when expanding to binary32. This stateless utility owns
 * no tensor storage.</p>
 */
public final class Float16Bits {
    private static final short CANONICAL_NAN = (short) 0x7E00;

    /** Prevents instantiation of this stateless bit-conversion utility. */
    private Float16Bits() {
    }

    /**
     * Expands a raw binary16 pattern into an IEEE-754 binary32 value.
     *
     * <p>Every 16-bit pattern is accepted. Signed zero, infinities, and NaN payload bits are
     * preserved in the widened representation.</p>
     *
     * @param bits raw binary16 bits
     * @return binary32 value represented by the expanded bits
     */
    public static float toFloat(short bits) {
        int raw = Short.toUnsignedInt(bits);
        int sign = (raw & 0x8000) << 16;
        int exponent = (raw >>> 10) & 0x1F;
        int fraction = raw & 0x03FF;
        if (exponent == 0) {
            if (fraction == 0) return Float.intBitsToFloat(sign);
            int e = -14;
            int normalized = fraction;
            while ((normalized & 0x0400) == 0) {
                normalized <<= 1;
                e--;
            }
            return Float.intBitsToFloat(sign | ((e + 127) << 23) | ((normalized & 0x03FF) << 13));
        }
        if (exponent == 0x1F) {
            return Float.intBitsToFloat(sign | 0x7F80_0000 | (fraction << 13));
        }
        return Float.intBitsToFloat(sign | ((exponent + 112) << 23) | (fraction << 13));
    }

    /**
     * Rounds an IEEE-754 binary32 value to raw binary16 bits using round-to-nearest, ties-to-even.
     *
     * <p>Finite underflow is gradual and overflow produces signed infinity. Every Java NaN is
     * normalized to canonical positive quiet binary16 NaN {@code 0x7E00}.</p>
     *
     * @param value binary32 value to convert
     * @return raw binary16 bits stored in a Java {@code short}
     */
    public static short fromFloat(float value) {
        int bits = Float.floatToRawIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exponent = (bits >>> 23) & 0xFF;
        int fraction = bits & 0x7F_FFFF;
        if (exponent == 0xFF) {
            return (short) (fraction == 0 ? sign | 0x7C00 : CANONICAL_NAN);
        }
        if (exponent == 0 && fraction == 0) {
            return (short) sign;
        }
        long significand = exponent == 0 ? fraction : 0x80_0000L | fraction;
        int scale = exponent == 0 ? -149 : exponent - 150;
        return roundFiniteBinary(sign, significand, scale);
    }

    /**
     * Rounds an IEEE-754 binary64 value directly to raw binary16 bits.
     *
     * <p>The conversion does not pass through binary32, so a value immediately above or below a
     * binary16 midpoint cannot be changed by double rounding. Finite values use round-to-nearest,
     * ties-to-even with gradual underflow and signed-infinity overflow. Signed zero and infinity
     * retain their signs. Every Java NaN is normalized to canonical positive quiet binary16 NaN
     * {@code 0x7E00}.</p>
     *
     * @param value binary64 value to convert
     * @return raw binary16 bits stored in a Java {@code short}
     */
    public static short fromDouble(double value) {
        long bits = Double.doubleToRawLongBits(value);
        int sign = (int) ((bits >>> 48) & 0x8000);
        int exponent = (int) ((bits >>> 52) & 0x7FF);
        long fraction = bits & 0x000F_FFFF_FFFF_FFFFL;
        if (exponent == 0x7FF) {
            return (short) (fraction == 0 ? sign | 0x7C00 : CANONICAL_NAN);
        }
        if (exponent == 0 && fraction == 0) {
            return (short) sign;
        }
        long significand = exponent == 0 ? fraction : (1L << 52) | fraction;
        int scale = exponent == 0 ? -1074 : exponent - 1075;
        return roundFiniteBinary(sign, significand, scale);
    }

    private static short roundFiniteBinary(int sign, long significand, int scale) {
        int highest = 63 - Long.numberOfLeadingZeros(significand);
        int unbiasedExponent = highest + scale;
        if (unbiasedExponent > 15) {
            return (short) (sign | 0x7C00);
        }
        if (unbiasedExponent >= -14) {
            long rounded = roundRight(significand, highest - 10);
            if (rounded == 0x800L) {
                rounded = 0x400L;
                unbiasedExponent++;
            }
            if (unbiasedExponent > 15) {
                return (short) (sign | 0x7C00);
            }
            return (short) (sign
                    | ((unbiasedExponent + 15) << 10)
                    | ((int) rounded & 0x03FF));
        }
        long rounded = roundRight(significand, -(scale + 24));
        if (rounded >= 0x400L) {
            return (short) (sign | 0x0400);
        }
        return (short) (sign | (int) rounded);
    }

    private static long roundRight(long value, int shift) {
        if (shift <= 0) return value << -shift;
        if (shift >= Long.SIZE) return 0L;
        long retained = value >>> shift;
        long discarded = value & ((1L << shift) - 1L);
        long midpoint = 1L << (shift - 1);
        int comparison = Long.compareUnsigned(discarded, midpoint);
        if (comparison > 0 || (comparison == 0 && (retained & 1L) != 0L)) retained++;
        return retained;
    }
}
