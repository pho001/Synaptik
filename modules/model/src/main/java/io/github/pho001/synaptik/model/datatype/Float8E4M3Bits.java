package io.github.pho001.synaptik.model.datatype;

/**
 * Converts raw OFP8 E4M3 (bias 7, finite maximum 448, no infinity) values.
 *
 * <p>Numeric narrowing uses non-saturating round-to-nearest, ties-to-even. Raw decoding preserves
 * signed zeros and the represented NaN sign and fraction. This utility owns no storage.</p>
 */
public final class Float8E4M3Bits {
    private Float8E4M3Bits() {
    }

    /**
     * Expands every raw byte pattern exactly to binary32, including represented NaN bits.
     *
     * @param bits raw OFP8 storage; all 256 patterns are accepted
     * @return exact represented binary32 value, without NaN canonicalization
     */
    public static float toFloat(byte bits) {
        int raw = bits & 0xFF;
        int sign = (raw & 0x80) << 24;
        int exponent = (raw >>> 3) & 15;
        int fraction = raw & 7;
        if ((raw & 0x7F) == 0x7F) {
            return Float.intBitsToFloat(sign | 0x7F80_0000 | (fraction << 20));
        }
        if (exponent == 0) {
            float magnitude = Math.scalb((float) fraction, -9);
            return Float.intBitsToFloat(sign | Float.floatToRawIntBits(magnitude));
        }
        return Float.intBitsToFloat(sign | ((exponent + 120) << 23)
                | (fraction << 20));
    }

    /**
     * Rounds a binary32 value once to OFP8. NaNs become {@code 0x7F}.
     *
     * @param value binary32 input, including signed zeros, infinities and NaNs
     * @return raw rounded byte; infinity and out-of-range rounding produce canonical NaN
     */
    public static byte fromFloat(float value) {
        return fromDouble(value);
    }

    /**
     * Rounds binary64 directly to OFP8, without an intermediate binary32 rounding.
     *
     * @param value binary64 input, including signed zeros, infinities and NaNs
     * @return raw rounded byte; NaNs become {@code 0x7F} and magnitudes above the inclusive 464 tie become NaN
     */
    public static byte fromDouble(double value) {
        if (Double.isNaN(value)) return (byte) 0x7F;
        int sign = (int) (Double.doubleToRawLongBits(value) >>> 56) & 0x80;
        double magnitude = Math.abs(value);
        if (magnitude > 464.0d) return (byte) 0x7F;
        if (magnitude < 0x1.0p-6) {
            return (byte) (sign | (int) Math.rint(Math.scalb(magnitude, 9)));
        }
        int exponent = Math.getExponent(magnitude);
        int significand = (int) Math.rint(Math.scalb(magnitude, 3 - exponent));
        if (significand == 16) {
            significand = 8;
            exponent++;
        }
        return (byte) (sign | ((exponent + 7) << 3) | (significand & 7));
    }
}
