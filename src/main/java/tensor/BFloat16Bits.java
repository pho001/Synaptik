package tensor;

/**
 * Backend-neutral helpers for converting between Java float values and raw BFLOAT16 bit patterns.
 */
public final class BFloat16Bits {
    private BFloat16Bits() {
    }

    public static float toFloat(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    public static short fromFloat(float value) {
        if (Float.isNaN(value)) {
            return (short) 0x7FC0;
        }
        int bits = Float.floatToIntBits(value);
        int upper = bits >>> 16;
        int lower = bits & 0xFFFF;
        if (lower > 0x8000 || (lower == 0x8000 && (upper & 0x1) != 0)) {
            upper++;
        }
        return (short) upper;
    }
}
