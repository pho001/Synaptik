package io.github.pho001.synaptik.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class BFloat16BitsTest {
    @Test
    void convertsExactlyRepresentableFiniteValues() {
        assertBFloat16Bits(0x0000, 0.0f);
        assertBFloat16Bits(0x8000, -0.0f);
        assertBFloat16Bits(0x3F80, 1.0f);
        assertBFloat16Bits(0xC020, -2.5f);
        assertEquals(
                Float.floatToRawIntBits(-0.0f),
                Float.floatToRawIntBits(BFloat16Bits.toFloat((short) 0x8000)));
    }

    @Test
    void preservesInfinityClassificationAndSign() {
        assertBFloat16Bits(0x7F80, Float.POSITIVE_INFINITY);
        assertBFloat16Bits(0xFF80, Float.NEGATIVE_INFINITY);
        assertEquals(
                Float.floatToRawIntBits(Float.POSITIVE_INFINITY),
                Float.floatToRawIntBits(BFloat16Bits.toFloat((short) 0x7F80)));
        assertEquals(
                Float.floatToRawIntBits(Float.NEGATIVE_INFINITY),
                Float.floatToRawIntBits(BFloat16Bits.toFloat((short) 0xFF80)));
    }

    @Test
    void roundsFiniteValuesToNearestWithTiesToEven() {
        assertConversion(0x3F80, 0x3F80_7FFF);
        assertConversion(0x3F80, 0x3F80_8000);
        assertConversion(0x3F81, 0x3F80_8001);
        assertConversion(0x3F82, 0x3F81_8000);
    }

    @Test
    void canonicalizesEveryFloatNan() {
        assertEquals(0x7FC0, unsigned(BFloat16Bits.fromFloat(Float.NaN)));
        assertEquals(
                0x7FC0,
                unsigned(BFloat16Bits.fromFloat(Float.intBitsToFloat(0xFFC0_0001))));
    }

    @Test
    void expandsBFloat16NanPatternsToJavaNan() {
        assertTrue(Float.isNaN(BFloat16Bits.toFloat((short) 0x7FC0)));
        assertTrue(Float.isNaN(BFloat16Bits.toFloat((short) 0x7F81)));
        assertTrue(Float.isNaN(BFloat16Bits.toFloat((short) 0xFFC1)));
    }

    @Test
    void roundTripsEverySelectedNonNanBFloat16Pattern() {
        short[] patterns = {
            (short) 0x0000,
            (short) 0x8000,
            (short) 0x0001,
            (short) 0x3F80,
            (short) 0xBF80,
            (short) 0x7F7F,
            (short) 0xFF7F,
            (short) 0x7F80,
            (short) 0xFF80
        };

        for (short pattern : patterns) {
            assertEquals(
                    unsigned(pattern),
                    unsigned(BFloat16Bits.fromFloat(BFloat16Bits.toFloat(pattern))));
        }
    }

    private static void assertBFloat16Bits(int expectedBits, float value) {
        assertEquals(expectedBits, unsigned(BFloat16Bits.fromFloat(value)));
    }

    private static void assertConversion(int expectedBits, int floatBits) {
        assertEquals(
                expectedBits,
                unsigned(BFloat16Bits.fromFloat(Float.intBitsToFloat(floatBits))));
    }

    private static int unsigned(short bits) {
        return Short.toUnsignedInt(bits);
    }
}
