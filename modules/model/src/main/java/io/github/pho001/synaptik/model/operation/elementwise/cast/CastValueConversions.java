package io.github.pho001.synaptik.model.operation.elementwise.cast;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.datatype.Float8E4M3Bits;
import io.github.pho001.synaptik.model.datatype.Float8E5M2Bits;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import java.util.Objects;

/**
 * Converts one exact scalar value according to the backend-independent {@link CastKind#CAST}
 * value contract.
 *
 * <p>All source/target pairs among the nine Model data types are valid. Same-type conversion
 * returns the exact source object. Every finite floating or integer source converted to a
 * floating target rounds directly to that target with round-to-nearest, ties-to-even; no
 * low conversion passes through a lossy binary32 intermediate. Floating-to-integral conversion
 * truncates toward zero and saturates, integral narrowing retains low bits, and numeric-to-Boolean
 * conversion is false only for either signed floating zero or integer zero. Lossy floating
 * conversion produces the target's positive canonical quiet NaN. Lossless floating widening
 * preserves the source NaN sign, quiet/signaling bit, and complete fraction by left alignment.</p>
 *
 * <p>This stateless utility is a scalar semantic oracle for tests and cold verification. It does
 * not inspect Tensor storage, evaluate an expression, advertise backend support, or provide a
 * backend element-loop implementation.</p>
 */
public final class CastValueConversions {
    /** Prevents instantiation of this stateless conversion utility. */
    private CastValueConversions() {
    }

    /**
     * Converts one exact scalar to the requested target data type.
     *
     * <p>Signed floating zero is preserved when the target is floating; gradual underflow may
     * produce a target subnormal or signed zero. Overflow produces signed infinity except in
     * E4M3, where it produces canonical NaN. NaN converts to zero for an integral target and to true for a Boolean target.
     * Floating infinities saturate for an integral target. INT32-to-INT64 conversion sign-extends,
     * and INT64-to-INT32 conversion retains the low 32 two's-complement bits. BOOL converts to
     * positive numeric zero or one. Inexact, overflow, underflow, saturation, and modulo outcomes
     * are returned normally.</p>
     *
     * @param source non-null exact source scalar; it is returned by identity for a same-type cast
     *     and is otherwise not retained or mutated
     * @param targetDataType non-null requested target data type
     * @return the exact {@code source} reference for a same-type request; otherwise a non-null new
     *     scalar whose data type is exactly {@code targetDataType}
     * @throws NullPointerException if {@code source} or {@code targetDataType} is null, checked in
     *     that order with the parameter name as the message
     */
    public static ScalarValue convert(ScalarValue source, DataType targetDataType) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(targetDataType, "targetDataType");
        if (source.dataType() == targetDataType) return source;
        return switch (source.dataType()) {
            case FLOAT64 -> fromFloat64(source.float64Value(), targetDataType);
            case FLOAT32 -> fromFloat32(source.float32Value(), targetDataType);
            case BFLOAT16 -> fromBFloat16(source.bfloat16Bits(), targetDataType);
            case FLOAT16 -> fromFloat16(source.float16Bits(), targetDataType);
            case FLOAT8_E4M3 -> fromFloat8(source.float8E4m3Bits(), true, targetDataType);
            case FLOAT8_E5M2 -> fromFloat8(source.float8E5m2Bits(), false, targetDataType);
            case INT64 -> fromInt64(source.int64Value(), targetDataType);
            case INT32 -> fromInt32(source.int32Value(), targetDataType);
            case BOOL -> fromBoolean(source.booleanValue(), targetDataType);
        };
    }

    private static ScalarValue fromFloat64(double value, DataType target) {
        long bits = Double.doubleToRawLongBits(value);
        return switch (target) {
            case FLOAT64 -> failSame();
            case FLOAT32 -> ScalarValue.float32(isFloat64NaN(bits)
                    ? Float.intBitsToFloat(0x7FC0_0000)
                    : (float) value);
            case BFLOAT16 -> ScalarValue.bfloat16Bits(float64ToBFloat16(bits));
            case FLOAT16 -> ScalarValue.float16Bits(Float16Bits.fromDouble(value));
            case FLOAT8_E4M3 -> ScalarValue.float8E4m3Bits(Float8E4M3Bits.fromDouble(value));
            case FLOAT8_E5M2 -> ScalarValue.float8E5m2Bits(Float8E5M2Bits.fromDouble(value));
            case INT64 -> ScalarValue.int64((long) value);
            case INT32 -> ScalarValue.int32((int) value);
            case BOOL -> ScalarValue.bool((bits & 0x7FFF_FFFF_FFFF_FFFFL) != 0);
        };
    }

    private static ScalarValue fromFloat32(float value, DataType target) {
        int bits = Float.floatToRawIntBits(value);
        return switch (target) {
            case FLOAT64 -> ScalarValue.float64(
                    isFloat32NaN(bits)
                            ? Double.longBitsToDouble(float32NaNToFloat64(bits))
                            : (double) value);
            case FLOAT32 -> failSame();
            case BFLOAT16 -> ScalarValue.bfloat16Bits(BFloat16Bits.fromFloat(value));
            case FLOAT16 -> ScalarValue.float16Bits(Float16Bits.fromFloat(value));
            case FLOAT8_E4M3 -> ScalarValue.float8E4m3(value);
            case FLOAT8_E5M2 -> ScalarValue.float8E5m2(value);
            case INT64 -> ScalarValue.int64((long) value);
            case INT32 -> ScalarValue.int32((int) value);
            case BOOL -> ScalarValue.bool((bits & 0x7FFF_FFFF) != 0);
        };
    }

    private static ScalarValue fromBFloat16(short value, DataType target) {
        int bits = value & 0xFFFF;
        float expanded = BFloat16Bits.toFloat(value);
        return switch (target) {
            case FLOAT64 -> ScalarValue.float64(
                    isBFloat16NaN(bits)
                            ? Double.longBitsToDouble(bfloat16NaNToFloat64(bits))
                            : (double) expanded);
            case FLOAT32 -> ScalarValue.float32(expanded);
            case BFLOAT16 -> failSame();
            case FLOAT16 -> ScalarValue.float16Bits(Float16Bits.fromFloat(expanded));
            case FLOAT8_E4M3 -> ScalarValue.float8E4m3(expanded);
            case FLOAT8_E5M2 -> ScalarValue.float8E5m2(expanded);
            case INT64 -> ScalarValue.int64((long) expanded);
            case INT32 -> ScalarValue.int32((int) expanded);
            case BOOL -> ScalarValue.bool((bits & 0x7FFF) != 0);
        };
    }

    private static ScalarValue fromFloat16(short value, DataType target) {
        int bits = value & 0xFFFF;
        float expanded = Float16Bits.toFloat(value);
        return switch (target) {
            case FLOAT64 -> ScalarValue.float64(
                    isFloat16NaN(bits)
                            ? Double.longBitsToDouble(float16NaNToFloat64(bits))
                            : (double) expanded);
            case FLOAT32 -> ScalarValue.float32(expanded);
            case BFLOAT16 -> ScalarValue.bfloat16Bits(
                    isFloat16NaN(bits) ? (short) 0x7FC0 : BFloat16Bits.fromFloat(expanded));
            case FLOAT16 -> failSame();
            case FLOAT8_E4M3 -> ScalarValue.float8E4m3(expanded);
            case FLOAT8_E5M2 -> ScalarValue.float8E5m2(expanded);
            case INT64 -> ScalarValue.int64((long) expanded);
            case INT32 -> ScalarValue.int32((int) expanded);
            case BOOL -> ScalarValue.bool((bits & 0x7FFF) != 0);
        };
    }

    private static ScalarValue fromFloat8(byte value, boolean e4m3, DataType target) {
        int raw = value & 0xFF;
        float expanded = e4m3 ? Float8E4M3Bits.toFloat(value) : Float8E5M2Bits.toFloat(value);
        int floatBits = Float.floatToRawIntBits(expanded);
        boolean nan = e4m3 ? (raw & 0x7F) == 0x7F : (raw & 0x7F) > 0x7C;
        return switch (target) {
            case FLOAT64 -> ScalarValue.float64(nan
                    ? Double.longBitsToDouble(float32NaNToFloat64(floatBits))
                    : (double) expanded);
            case FLOAT32 -> ScalarValue.float32(expanded);
            case BFLOAT16 -> ScalarValue.bfloat16Bits(nan
                    ? (short) (floatBits >>> 16) : BFloat16Bits.fromFloat(expanded));
            case FLOAT16 -> ScalarValue.float16Bits(nan
                    ? (short) (((raw & 0x80) << 8) | 0x7C00
                            | ((raw & (e4m3 ? 7 : 3)) << (e4m3 ? 7 : 8)))
                    : Float16Bits.fromFloat(expanded));
            case FLOAT8_E4M3 -> e4m3 ? failSame() : ScalarValue.float8E4m3(expanded);
            case FLOAT8_E5M2 -> e4m3 ? ScalarValue.float8E5m2(expanded) : failSame();
            case INT64 -> ScalarValue.int64((long) expanded);
            case INT32 -> ScalarValue.int32((int) expanded);
            case BOOL -> ScalarValue.bool((raw & 0x7F) != 0);
        };
    }

    private static ScalarValue fromInt64(long value, DataType target) {
        return switch (target) {
            case FLOAT64 -> ScalarValue.float64((double) value);
            case FLOAT32 -> ScalarValue.float32((float) value);
            case BFLOAT16 -> ScalarValue.bfloat16Bits(integerToBFloat16(value));
            case FLOAT16 -> ScalarValue.float16Bits(integerToFloat16(value));
            case FLOAT8_E4M3 -> ScalarValue.float8E4m3Bits(integerToFloat8(value, true));
            case FLOAT8_E5M2 -> ScalarValue.float8E5m2Bits(integerToFloat8(value, false));
            case INT64 -> failSame();
            case INT32 -> ScalarValue.int32((int) value);
            case BOOL -> ScalarValue.bool(value != 0);
        };
    }

    private static ScalarValue fromInt32(int value, DataType target) {
        return switch (target) {
            case FLOAT64 -> ScalarValue.float64((double) value);
            case FLOAT32 -> ScalarValue.float32((float) value);
            case BFLOAT16 -> ScalarValue.bfloat16Bits(integerToBFloat16(value));
            case FLOAT16 -> ScalarValue.float16Bits(integerToFloat16(value));
            case FLOAT8_E4M3 -> ScalarValue.float8E4m3Bits(integerToFloat8(value, true));
            case FLOAT8_E5M2 -> ScalarValue.float8E5m2Bits(integerToFloat8(value, false));
            case INT64 -> ScalarValue.int64(value);
            case INT32 -> failSame();
            case BOOL -> ScalarValue.bool(value != 0);
        };
    }

    private static ScalarValue fromBoolean(boolean value, DataType target) {
        int numeric = value ? 1 : 0;
        return switch (target) {
            case FLOAT64 -> ScalarValue.float64(numeric);
            case FLOAT32 -> ScalarValue.float32(numeric);
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) (value ? 0x3F80 : 0));
            case FLOAT16 -> ScalarValue.float16Bits((short) (value ? 0x3C00 : 0));
            case FLOAT8_E4M3 -> ScalarValue.float8E4m3Bits((byte) (value ? 0x38 : 0));
            case FLOAT8_E5M2 -> ScalarValue.float8E5m2Bits((byte) (value ? 0x3C : 0));
            case INT64 -> ScalarValue.int64(numeric);
            case INT32 -> ScalarValue.int32(numeric);
            case BOOL -> failSame();
        };
    }

    private static byte integerToFloat8(long value, boolean e4m3) {
        // Only this bounded exact-in-binary64 range can produce a finite FP8 result.
        if (e4m3) {
            if (value < -464 || value > 464) return (byte) 0x7F;
            return Float8E4M3Bits.fromDouble((double) value);
        }
        if (value <= -61440) return (byte) 0xFC;
        if (value >= 61440) return (byte) 0x7C;
        return Float8E5M2Bits.fromDouble((double) value);
    }

    private static short integerToFloat16(long value) {
        if (value == 0) return 0;
        int sign = value < 0 ? 0x8000 : 0;
        long magnitude = value < 0 ? -value : value;
        int exponent = 63 - Long.numberOfLeadingZeros(magnitude);
        if (exponent > 15) return (short) (sign | 0x7C00);
        long rounded = exponent <= 10
                ? magnitude << (10 - exponent)
                : roundRight(magnitude, exponent - 10);
        if (rounded == 0x800) { rounded = 0x400; exponent++; }
        if (exponent > 15) return (short) (sign | 0x7C00);
        return (short) (sign | ((exponent + 15) << 10) | ((int) rounded & 0x3FF));
    }

    private static short float64ToBFloat16(long sourceBits) {
        int sign = (int) ((sourceBits >>> 48) & 0x8000);
        int rawExponent = (int) ((sourceBits >>> 52) & 0x7FF);
        long fraction = sourceBits & 0x000F_FFFF_FFFF_FFFFL;
        if (rawExponent == 0x7FF) return (short) (fraction == 0 ? sign | 0x7F80 : 0x7FC0);
        if (rawExponent == 0 && fraction == 0) return (short) sign;
        long significand = rawExponent == 0 ? fraction : (1L << 52) | fraction;
        int scale = rawExponent == 0 ? -1074 : rawExponent - 1075;
        int highest = 63 - Long.numberOfLeadingZeros(significand);
        int exponent = highest + scale;
        if (exponent < -126) return (short) (sign | roundRight(significand, -(scale + 133)));
        long rounded = roundRight(significand, highest - 7);
        if (rounded == 0x100) { rounded = 0x80; exponent++; }
        if (exponent > 127) return (short) (sign | 0x7F80);
        return (short) (sign | ((exponent + 127) << 7) | ((int) rounded & 0x7F));
    }

    private static short integerToBFloat16(long value) {
        if (value == 0) return 0;
        int sign = value < 0 ? 0x8000 : 0;
        long magnitude = value < 0 ? -value : value;
        int exponent = 63 - Long.numberOfLeadingZeros(magnitude);
        long rounded = exponent <= 7
                ? magnitude << (7 - exponent)
                : roundRight(magnitude, exponent - 7);
        if (rounded == 0x100) { rounded = 0x80; exponent++; }
        return (short) (sign | ((exponent + 127) << 7) | ((int) rounded & 0x7F));
    }

    private static long roundRight(long value, int shift) {
        if (shift <= 0) return value << -shift;
        if (shift >= 64) return 0;
        long retained = value >>> shift;
        long discarded = value & ((1L << shift) - 1);
        long midpoint = 1L << (shift - 1);
        int comparison = Long.compareUnsigned(discarded, midpoint);
        if (comparison > 0 || (comparison == 0 && (retained & 1) != 0)) {
            retained++;
        }
        return retained;
    }
    private static boolean isFloat64NaN(long bits) {
        return (bits & 0x7FF0_0000_0000_0000L) == 0x7FF0_0000_0000_0000L
                && (bits & 0x000F_FFFF_FFFF_FFFFL) != 0;
    }

    private static boolean isFloat32NaN(int bits) {
        return (bits & 0x7F80_0000) == 0x7F80_0000
                && (bits & 0x007F_FFFF) != 0;
    }

    private static boolean isBFloat16NaN(int bits) {
        return (bits & 0x7F80) == 0x7F80 && (bits & 0x007F) != 0;
    }

    private static boolean isFloat16NaN(int bits) {
        return (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0;
    }

    private static long float32NaNToFloat64(int bits) {
        return ((long) (bits & 0x8000_0000) << 32)
                | 0x7FF0_0000_0000_0000L
                | ((long) (bits & 0x007F_FFFF) << 29);
    }

    private static long bfloat16NaNToFloat64(int bits) {
        return ((long) (bits & 0x8000) << 48)
                | 0x7FF0_0000_0000_0000L
                | ((long) (bits & 0x007F) << 45);
    }

    private static long float16NaNToFloat64(int bits) {
        return ((long) (bits & 0x8000) << 48)
                | 0x7FF0_0000_0000_0000L
                | ((long) (bits & 0x03FF) << 42);
    }

    private static <T> T failSame() {
        throw new AssertionError("same-type conversion bypassed identity return");
    }
}
