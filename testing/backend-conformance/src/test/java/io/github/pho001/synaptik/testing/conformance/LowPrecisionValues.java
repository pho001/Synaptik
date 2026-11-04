package io.github.pho001.synaptik.testing.conformance;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

/** Test-only represented-value oracle; FP8 rounding searches the finite codebook, not production bits. */
final class LowPrecisionValues {
    static final List<DataType> TYPES = List.of(DataType.BFLOAT16, DataType.FLOAT16,
            DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2);

    private LowPrecisionValues() { }

    /**
     * Decodes a raw cell using the format's sign, biased exponent and fraction definition.
     * @param type low floating storage format
     * @param bits unsigned raw cell bits
     * @return represented value, including signed zero and nonfinite classes
     */
    static float decode(DataType type, int bits) {
        if (type == DataType.BFLOAT16) return Float.intBitsToFloat(bits << 16);
        if (type == DataType.FLOAT16) return Float.float16ToFloat((short) bits);
        boolean e4 = type == DataType.FLOAT8_E4M3;
        int fractionBits = e4 ? 3 : 2, bias = e4 ? 7 : 15;
        int magnitude = bits & 127, fraction = magnitude & ((1 << fractionBits) - 1);
        int exponent = magnitude >> fractionBits;
        if (e4 ? magnitude == 127 : exponent == 31)
            return !e4 && fraction == 0 ? Math.copySign(Float.POSITIVE_INFINITY, (bits & 128) == 0 ? 1 : -1)
                    : Float.NaN;
        float value = Math.scalb(exponent == 0 ? fraction : (1 << fractionBits) + fraction,
                (exponent == 0 ? 1 : exponent) - bias - fractionBits);
        return Math.copySign(value, (bits & 128) == 0 ? 1 : -1);
    }

    /**
     * Rounds once to nearest, resolving equal-distance codes by their even trailing significand bit.
     * @param type low floating storage format
     * @param value F32 working result
     * @return unsigned raw narrowed cell bits, with canonical arithmetic NaNs
     */
    static int encode(DataType type, float value) {
        if (type == DataType.FLOAT16) return Float.floatToFloat16(value) & 65535;
        if (type == DataType.BFLOAT16) {
            int raw = Float.floatToRawIntBits(value);
            return Float.isNaN(value) ? 0x7fc0 : ((raw + 0x7fff + ((raw >>> 16) & 1)) >>> 16);
        }
        boolean e4 = type == DataType.FLOAT8_E4M3;
        if (Float.isNaN(value)) return e4 ? 127 : 126;
        int sign = Float.floatToRawIntBits(value) < 0 ? 128 : 0;
        double magnitude = Math.abs((double) value);
        if (e4 ? magnitude > 464 : magnitude >= 61440) return e4 ? 127 : sign | 124;
        int best = 0, last = e4 ? 126 : 123;
        double distance = Double.POSITIVE_INFINITY;
        for (int code = 0; code <= last; code++) {
            double candidate = Math.abs(magnitude - decode(type, code));
            if (candidate < distance || candidate == distance && (code & 1) == 0) {
                distance = candidate;
                best = code;
            }
        }
        return sign | best;
    }

    /**
     * Reads an unsigned stored cell without numeric conversion.
     * @param type low floating storage format
     * @param segment live storage covering the cell
     * @param index physical element index
     * @return unsigned raw cell bits
     */
    static int raw(DataType type, MemorySegment segment, long index) {
        return type.byteWidth() == 1 ? segment.getAtIndex(ValueLayout.JAVA_BYTE, index) & 255
                : segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, index * 2) & 65535;
    }

    /**
     * Writes a represented F32 value using this independent narrowing oracle.
     * @param type low floating storage format
     * @param segment live writable storage
     * @param index physical element index
     * @param value numeric value to round
     */
    static void put(DataType type, MemorySegment segment, long index, float value) {
        int bits = encode(type, value);
        if (type.byteWidth() == 1) segment.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) bits);
        else segment.set(ValueLayout.JAVA_SHORT_UNALIGNED, index * 2, (short) bits);
    }
}
