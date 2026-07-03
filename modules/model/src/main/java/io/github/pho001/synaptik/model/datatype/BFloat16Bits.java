package io.github.pho001.synaptik.model.datatype;

/**
 * Converts scalar values between Java {@code float} and raw BFLOAT16 bit patterns.
 *
 * <p>BFLOAT16 preserves the sign and eight-bit exponent of IEEE-754 binary32 while retaining its
 * most significant seven fraction bits. Conversion from {@code float} uses round-to-nearest with
 * ties to even. This stateless utility does not allocate or own tensor storage.</p>
 */
public final class BFloat16Bits {
    private static final short CANONICAL_NAN = (short) 0x7FC0;

    /** Prevents instantiation of this stateless bit-conversion utility. */
    private BFloat16Bits() {
    }

    /**
     * Expands a raw BFLOAT16 bit pattern into an IEEE-754 binary32 value.
     *
     * <p>The conversion places the unsigned 16-bit input in the most significant half of the
     * binary32 representation and fills the remaining fraction bits with zero. Signed zero,
     * infinities, and NaN classifications are preserved.</p>
     *
     * @param bits raw BFLOAT16 bits; every 16-bit pattern is accepted
     * @return binary32 value represented by the expanded bits; BFLOAT16 NaN patterns produce a
     *     Java NaN
     */
    public static float toFloat(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << Short.SIZE);
    }

    /**
     * Rounds an IEEE-754 binary32 value to a raw BFLOAT16 bit pattern.
     *
     * <p>Finite values use round-to-nearest with ties to even. Signed zero and infinities retain
     * their sign and classification. Every Java NaN is normalized to the canonical quiet BFLOAT16
     * NaN pattern {@code 0x7FC0}.</p>
     *
     * @param value binary32 value to convert; finite values, infinities, and NaNs are accepted
     * @return raw BFLOAT16 bits stored in a Java {@code short}
     */
    public static short fromFloat(float value) {
        if (Float.isNaN(value)) {
            return CANONICAL_NAN;
        }

        int floatBits = Float.floatToRawIntBits(value);
        int upper = floatBits >>> Short.SIZE;
        int discarded = floatBits & 0xFFFF;
        if (discarded > 0x8000 || (discarded == 0x8000 && (upper & 1) != 0)) {
            upper++;
        }
        return (short) upper;
    }
}
