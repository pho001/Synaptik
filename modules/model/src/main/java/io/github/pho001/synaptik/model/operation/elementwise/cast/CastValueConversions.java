package io.github.pho001.synaptik.model.operation.elementwise.cast;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import java.util.Objects;

/**
 * Converts one exact scalar value according to the backend-independent {@link CastKind#CAST}
 * value contract.
 *
 * <p>All current source and target pairs are accepted. Same-type conversion returns the exact
 * source object. Every finite floating or integer source converted to a floating target rounds
 * directly to that target with round-to-nearest, ties-to-even; in particular, conversion to
 * BFLOAT16 does not pass through FLOAT32. Floating-to-integer conversion truncates toward zero
 * and saturates, integral narrowing retains low bits, and numeric-to-Boolean conversion is false
 * only for zero. Lossy floating narrowing produces positive canonical quiet NaN bits
 * {@code 0x7FC00000} for FLOAT32 or {@code 0x7FC0} for BFLOAT16. Lossless floating widening
 * preserves the source NaN sign, quiet/signaling state, and complete fraction by left alignment.</p>
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
     * produce a target subnormal or signed zero, while floating overflow produces signed
     * infinity. NaN converts to zero for an integral target and to true for a Boolean target.
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
        if (source.dataType() == targetDataType) {
            return source;
        }

        return switch (source.dataType()) {
            case FLOAT64 -> fromFloat64(source.float64Value(), targetDataType);
            case FLOAT32 -> fromFloat32(source.float32Value(), targetDataType);
            case BFLOAT16 -> fromBFloat16(source.bfloat16Bits(), targetDataType);
            case INT64 -> fromInt64(source.int64Value(), targetDataType);
            case INT32 -> fromInt32(source.int32Value(), targetDataType);
            case BOOL -> fromBoolean(source.booleanValue(), targetDataType);
        };
    }

    private static ScalarValue fromFloat64(double value, DataType targetDataType) {
        long bits = Double.doubleToRawLongBits(value);
        return switch (targetDataType) {
            case FLOAT64 -> throw sameTypeDispatchFailure();
            case FLOAT32 -> ScalarValue.float32(isFloat64NaN(bits)
                    ? Float.intBitsToFloat(0x7FC0_0000)
                    : (float) value);
            case BFLOAT16 -> ScalarValue.bfloat16Bits(float64ToBFloat16(bits));
            case INT64 -> ScalarValue.int64((long) value);
            case INT32 -> ScalarValue.int32((int) value);
            case BOOL -> ScalarValue.bool((bits & 0x7FFF_FFFF_FFFF_FFFFL) != 0L);
        };
    }

    private static ScalarValue fromFloat32(float value, DataType targetDataType) {
        int bits = Float.floatToRawIntBits(value);
        return switch (targetDataType) {
            case FLOAT64 -> ScalarValue.float64(isFloat32NaN(bits)
                    ? Double.longBitsToDouble(float32NaNToFloat64(bits))
                    : (double) value);
            case FLOAT32 -> throw sameTypeDispatchFailure();
            case BFLOAT16 -> ScalarValue.bfloat16Bits(BFloat16Bits.fromFloat(value));
            case INT64 -> ScalarValue.int64((long) value);
            case INT32 -> ScalarValue.int32((int) value);
            case BOOL -> ScalarValue.bool((bits & 0x7FFF_FFFF) != 0);
        };
    }

    private static ScalarValue fromBFloat16(short value, DataType targetDataType) {
        int bits = value & 0xFFFF;
        float expanded = BFloat16Bits.toFloat(value);
        return switch (targetDataType) {
            case FLOAT64 -> ScalarValue.float64(isBFloat16NaN(bits)
                    ? Double.longBitsToDouble(bfloat16NaNToFloat64(bits))
                    : (double) expanded);
            case FLOAT32 -> ScalarValue.float32(expanded);
            case BFLOAT16 -> throw sameTypeDispatchFailure();
            case INT64 -> ScalarValue.int64((long) expanded);
            case INT32 -> ScalarValue.int32((int) expanded);
            case BOOL -> ScalarValue.bool((bits & 0x7FFF) != 0);
        };
    }

    private static ScalarValue fromInt64(long value, DataType targetDataType) {
        return switch (targetDataType) {
            case FLOAT64 -> ScalarValue.float64((double) value);
            case FLOAT32 -> ScalarValue.float32((float) value);
            case BFLOAT16 -> ScalarValue.bfloat16Bits(integerToBFloat16(value));
            case INT64 -> throw sameTypeDispatchFailure();
            case INT32 -> ScalarValue.int32((int) value);
            case BOOL -> ScalarValue.bool(value != 0L);
        };
    }

    private static ScalarValue fromInt32(int value, DataType targetDataType) {
        return switch (targetDataType) {
            case FLOAT64 -> ScalarValue.float64((double) value);
            case FLOAT32 -> ScalarValue.float32((float) value);
            case BFLOAT16 -> ScalarValue.bfloat16Bits(integerToBFloat16(value));
            case INT64 -> ScalarValue.int64(value);
            case INT32 -> throw sameTypeDispatchFailure();
            case BOOL -> ScalarValue.bool(value != 0);
        };
    }

    private static ScalarValue fromBoolean(boolean value, DataType targetDataType) {
        int numeric = value ? 1 : 0;
        return switch (targetDataType) {
            case FLOAT64 -> ScalarValue.float64(numeric);
            case FLOAT32 -> ScalarValue.float32(numeric);
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) (value ? 0x3F80 : 0));
            case INT64 -> ScalarValue.int64(numeric);
            case INT32 -> ScalarValue.int32(numeric);
            case BOOL -> throw sameTypeDispatchFailure();
        };
    }

    private static short float64ToBFloat16(long sourceBits) {
        int sign = (int) ((sourceBits >>> 48) & 0x8000);
        int rawExponent = (int) ((sourceBits >>> 52) & 0x7FF);
        long fraction = sourceBits & 0x000F_FFFF_FFFF_FFFFL;
        if (rawExponent == 0x7FF) {
            return (short) (fraction == 0L ? sign | 0x7F80 : 0x7FC0);
        }
        if (rawExponent == 0 && fraction == 0L) {
            return (short) sign;
        }

        long significand;
        int binaryScale;
        if (rawExponent == 0) {
            significand = fraction;
            binaryScale = -1074;
        } else {
            significand = (1L << 52) | fraction;
            binaryScale = rawExponent - 1075;
        }

        int significandBits = Long.SIZE - Long.numberOfLeadingZeros(significand);
        int exponent = significandBits - 1 + binaryScale;
        if (exponent < -126) {
            long subnormal = roundRightShift(significand, -binaryScale - 133);
            return (short) (sign | (int) subnormal);
        }

        long roundedSignificand = roundRightShift(significand, significandBits - 8);
        if (roundedSignificand == 0x100L) {
            roundedSignificand = 0x80L;
            exponent++;
        }
        if (exponent > 127) {
            return (short) (sign | 0x7F80);
        }
        return (short) (sign | ((exponent + 127) << 7) | ((int) roundedSignificand & 0x7F));
    }

    private static short integerToBFloat16(long value) {
        if (value == 0L) {
            return 0;
        }

        int sign = value < 0L ? 0x8000 : 0;
        long magnitude = value < 0L ? -value : value;
        int exponent = Long.SIZE - 1 - Long.numberOfLeadingZeros(magnitude);
        long roundedSignificand = exponent <= 7
                ? magnitude << (7 - exponent)
                : roundRightShift(magnitude, exponent - 7);
        if (roundedSignificand == 0x100L) {
            roundedSignificand = 0x80L;
            exponent++;
        }
        return (short) (sign | ((exponent + 127) << 7) | ((int) roundedSignificand & 0x7F));
    }

    private static long roundRightShift(long value, int shift) {
        if (shift <= 0) {
            return value << -shift;
        }
        if (shift >= Long.SIZE - 1) {
            return 0L;
        }

        long retained = value >>> shift;
        long discarded = value & ((1L << shift) - 1L);
        long midpoint = 1L << (shift - 1);
        if (discarded > midpoint || (discarded == midpoint && (retained & 1L) != 0L)) {
            retained++;
        }
        return retained;
    }

    private static boolean isFloat64NaN(long bits) {
        return (bits & 0x7FF0_0000_0000_0000L) == 0x7FF0_0000_0000_0000L
                && (bits & 0x000F_FFFF_FFFF_FFFFL) != 0L;
    }

    private static boolean isFloat32NaN(int bits) {
        return (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0;
    }

    private static boolean isBFloat16NaN(int bits) {
        return (bits & 0x7F80) == 0x7F80 && (bits & 0x007F) != 0;
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

    private static AssertionError sameTypeDispatchFailure() {
        return new AssertionError("same-type conversion bypassed identity return");
    }
}
