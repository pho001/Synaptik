package io.github.pho001.synaptik.model.datatype;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.operation.elementwise.cast.CastValueConversions;
import org.junit.jupiter.api.Test;

class Float8BitsTest {
    @Test
    void decodesEveryByteAndPreservesEveryRawScalarAndSameTypeCast() {
        for (boolean e4 : new boolean[] {true, false}) {
            int fractionBits = e4 ? 3 : 2;
            int bias = e4 ? 7 : 15;
            for (int raw = 0; raw < 256; raw++) {
                int magnitude = raw & 0x7F;
                int fraction = magnitude & ((1 << fractionBits) - 1);
                int exponent = magnitude >>> fractionBits;
                boolean nan = e4 ? magnitude == 127 : magnitude > 124;
                boolean infinity = !e4 && magnitude == 124;
                float decoded = decode((byte) raw, e4);
                int sign = (raw & 0x80) << 24;
                int expected;
                if (nan || infinity) {
                    expected = sign | 0x7F80_0000 | (fraction << (23 - fractionBits));
                } else {
                    double value = exponent == 0
                            ? Math.scalb((double) fraction, 1 - bias - fractionBits)
                            : Math.scalb((double) ((1 << fractionBits) + fraction),
                                    exponent - bias - fractionBits);
                    expected = sign | Float.floatToRawIntBits((float) value);
                }
                assertEquals(expected, Float.floatToRawIntBits(decoded), "decode " + raw);
                assertEquals(nan ? (e4 ? 127 : 126) : raw, encode(decoded, e4));
                ScalarValue scalar = scalar((byte) raw, e4);
                assertEquals((byte) raw, e4 ? scalar.float8E4m3Bits() : scalar.float8E5m2Bits());
                assertSame(scalar, CastValueConversions.convert(scalar, scalar.dataType()));
                assertEquals(expected, Float.floatToRawIntBits(
                        CastValueConversions.convert(scalar, DataType.FLOAT32).float32Value()));
                if (nan) {
                    long doubleBits = ((long) sign << 32) | 0x7FF0_0000_0000_0000L
                            | ((long) fraction << (52 - fractionBits));
                    assertEquals(doubleBits, Double.doubleToRawLongBits(
                            CastValueConversions.convert(scalar, DataType.FLOAT64).float64Value()));
                    int halfBits = ((raw & 0x80) << 8) | 0x7C00
                            | (fraction << (10 - fractionBits));
                    assertEquals(halfBits, Short.toUnsignedInt(
                            CastValueConversions.convert(scalar, DataType.FLOAT16).float16Bits()));
                    assertEquals(expected >>> 16, Short.toUnsignedInt(
                            CastValueConversions.convert(scalar, DataType.BFLOAT16).bfloat16Bits()));
                }
            }
        }
    }

    @Test
    void roundsEveryFiniteMidpointAndItsAdjacentBinary64ValuesWithoutDoubleRounding() {
        for (boolean e4 : new boolean[] {true, false}) {
            int last = e4 ? 126 : 123;
            for (int lower = 0; lower < last; lower++) {
                double midpoint = ((double) decode((byte) lower, e4)
                        + decode((byte) (lower + 1), e4)) * 0.5d;
                int even = (lower & 1) == 0 ? lower : lower + 1;
                assertConverted(lower, Math.nextDown(midpoint), e4);
                assertConverted(even, midpoint, e4);
                assertConverted(lower + 1, Math.nextUp(midpoint), e4);
                assertConverted(lower | 0x80, Math.nextUp(-midpoint), e4);
                assertConverted(even | 0x80, -midpoint, e4);
                assertConverted((lower + 1) | 0x80, Math.nextDown(-midpoint), e4);
                assertEquals(even, encode((float) midpoint, e4));
                assertEquals(lower, encode(Math.nextDown((float) midpoint), e4));
                assertEquals(lower + 1, encode(Math.nextUp((float) midpoint), e4));
            }
        }
    }

    @Test
    void handlesOverflowTiesInfinitiesNaNsAndIntegerConversionDirectly() {
        assertConverted(126, 464.0d, true);
        assertConverted(126, Math.nextDown(464.0d), true);
        assertConverted(127, Math.nextUp(464.0d), true);
        assertConverted(254, -464.0d, true);
        assertConverted(127, Math.nextDown(-464.0d), true);
        assertConverted(123, Math.nextDown(61440.0d), false);
        assertConverted(124, 61440.0d, false);
        assertConverted(251, Math.nextUp(-61440.0d), false);
        assertConverted(252, -61440.0d, false);
        for (boolean e4 : new boolean[] {true, false}) {
            assertConverted(e4 ? 127 : 124, Double.POSITIVE_INFINITY, e4);
            assertConverted(e4 ? 127 : 252, Double.NEGATIVE_INFINITY, e4);
            assertConverted(e4 ? 127 : 126, Double.longBitsToDouble(0xFFF0_0000_0000_0042L), e4);
            assertConverted(0, Double.MIN_VALUE, e4);
            assertConverted(128, -Double.MIN_VALUE, e4);
            for (long value : new long[] {Long.MIN_VALUE, Long.MAX_VALUE, -61440, -464, -465,
                    0, 1, 17, 463, 464, 465, 61439, 61440}) {
                ScalarValue converted = CastValueConversions.convert(ScalarValue.int64(value),
                        e4 ? DataType.FLOAT8_E4M3 : DataType.FLOAT8_E5M2);
                int raw = Byte.toUnsignedInt(e4 ? converted.float8E4m3Bits() : converted.float8E5m2Bits());
                assertEquals(encode((double) value, e4), raw);
            }
        }
        assertNotEquals(ScalarValue.float8E4m3Bits((byte) 0x7F),
                ScalarValue.float8E4m3Bits((byte) 0xFF));
        assertNotEquals(ScalarValue.float8E5m2Bits((byte) 0x7D),
                ScalarValue.float8E5m2Bits((byte) 0x7E));
        assertThrows(IllegalStateException.class,
                () -> ScalarValue.float8E4m3(1.0f).float8E5m2Bits());
    }

    private static void assertConverted(int expected, double value, boolean e4) {
        assertEquals(expected, encode(value, e4));
        ScalarValue converted = CastValueConversions.convert(ScalarValue.float64(value),
                e4 ? DataType.FLOAT8_E4M3 : DataType.FLOAT8_E5M2);
        assertEquals(expected, Byte.toUnsignedInt(e4 ? converted.float8E4m3Bits() : converted.float8E5m2Bits()));
    }

    private static ScalarValue scalar(byte raw, boolean e4) {
        return e4 ? ScalarValue.float8E4m3Bits(raw) : ScalarValue.float8E5m2Bits(raw);
    }

    private static float decode(byte raw, boolean e4) {
        return e4 ? Float8E4M3Bits.toFloat(raw) : Float8E5M2Bits.toFloat(raw);
    }

    private static int encode(double value, boolean e4) {
        return Byte.toUnsignedInt(e4 ? Float8E4M3Bits.fromDouble(value) : Float8E5M2Bits.fromDouble(value));
    }

    private static int encode(float value, boolean e4) {
        return Byte.toUnsignedInt(e4 ? Float8E4M3Bits.fromFloat(value) : Float8E5M2Bits.fromFloat(value));
    }
}
