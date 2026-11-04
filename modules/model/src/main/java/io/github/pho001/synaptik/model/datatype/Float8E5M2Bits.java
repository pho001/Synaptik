package io.github.pho001.synaptik.model.datatype;

/**
 * Converts raw OFP8 E5M2 (bias 15, finite maximum 57344) values.
 *
 * <p>Numeric narrowing uses non-saturating round-to-nearest, ties-to-even. Raw decoding preserves
 * signed zeros and the represented NaN sign and fraction. This utility owns no storage.</p>
 */
public final class Float8E5M2Bits {
    private Float8E5M2Bits() {
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
        int exponent = (raw >>> 2) & 31;
        int fraction = raw & 3;
        if (exponent == 31) {
            return Float.intBitsToFloat(sign | 0x7F80_0000 | (fraction << 21));
        }
        if (exponent == 0) {
            float magnitude = Math.scalb((float) fraction, -16);
            return Float.intBitsToFloat(sign | Float.floatToRawIntBits(magnitude));
        }
        return Float.intBitsToFloat(sign | ((exponent + 112) << 23)
                | (fraction << 21));
    }

    /**
     * Rounds a binary32 value once to OFP8. NaNs become {@code 0x7E}.
     *
     * @param value binary32 input, including signed zeros, infinities and NaNs
     * @return raw rounded byte; overflow produces signed infinity
     */
    public static byte fromFloat(float value) {
        return fromDouble(value);
    }

    /**
     * Rounds binary64 directly to OFP8, without an intermediate binary32 rounding.
     *
     * @param value binary64 input, including signed zeros, infinities and NaNs
     * @return raw rounded byte; NaNs become {@code 0x7E} and magnitudes at or above the 61440 overflow tie become signed infinity
     */
    public static byte fromDouble(double value) {
        if (Double.isNaN(value)) return (byte) 0x7E;
        int sign = (int) (Double.doubleToRawLongBits(value) >>> 56) & 0x80;
        double magnitude = Math.abs(value);
        if (magnitude >= 61440.0d) return (byte) (sign | 0x7C);
        if (magnitude < 0x1.0p-14) {
            return (byte) (sign | (int) Math.rint(Math.scalb(magnitude, 16)));
        }
        int exponent = Math.getExponent(magnitude);
        int significand = (int) Math.rint(Math.scalb(magnitude, 2 - exponent));
        if (significand == 8) {
            significand = 4;
            exponent++;
        }
        return (byte) (sign | ((exponent + 15) << 2) | (significand & 3));
    }
}
