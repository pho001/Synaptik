package io.github.pho001.synaptik.model.datatype;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class Float16BitsTest {
    @Test
    void decodesEveryRawPatternExactlyAndRoundTripsNonNaNs() {
        for (int raw = 0; raw <= 0xFFFF; raw++) {
            short bits = (short) raw;
            float value = Float16Bits.toFloat(bits);
            int exponent = (raw >>> 10) & 0x1F;
            int fraction = raw & 0x03FF;
            if (exponent == 0x1F && fraction != 0) {
                int expected = ((raw & 0x8000) << 16)
                        | 0x7F80_0000
                        | (fraction << 13);
                assertEquals(expected, Float.floatToRawIntBits(value));
                assertEquals(0x7E00, Short.toUnsignedInt(Float16Bits.fromFloat(value)));
            } else {
                assertEquals(
                        Float.floatToRawIntBits(Float.float16ToFloat(bits)),
                        Float.floatToRawIntBits(value));
                assertEquals(raw, Short.toUnsignedInt(Float16Bits.fromFloat(value)));
            }
        }
    }

    @Test
    void handlesBoundariesAndTiesToEven() {
        assertEquals(0x0000, Short.toUnsignedInt(Float16Bits.fromFloat(0.0f)));
        assertEquals(0x8000, Short.toUnsignedInt(Float16Bits.fromFloat(-0.0f)));
        assertEquals(0x0001, Short.toUnsignedInt(Float16Bits.fromFloat(Float16Bits.toFloat((short) 0x0001))));
        assertEquals(0x03FF, Short.toUnsignedInt(Float16Bits.fromFloat(Float16Bits.toFloat((short) 0x03FF))));
        assertEquals(0x0400, Short.toUnsignedInt(Float16Bits.fromFloat(Float16Bits.toFloat((short) 0x0400))));
        assertEquals(0x7BFF, Short.toUnsignedInt(Float16Bits.fromFloat(Float16Bits.toFloat((short) 0x7BFF))));
        assertEquals(0x7C00, Short.toUnsignedInt(Float16Bits.fromFloat(Float.POSITIVE_INFINITY)));
        assertEquals(0xFC00, Short.toUnsignedInt(Float16Bits.fromFloat(Float.NEGATIVE_INFINITY)));
        assertEquals(0x3C00, Short.toUnsignedInt(Float16Bits.fromFloat(Float.intBitsToFloat(0x3F80_1000))));
        assertEquals(0x3C01, Short.toUnsignedInt(Float16Bits.fromFloat(Float.intBitsToFloat(0x3F80_1800))));
        assertEquals(0x7E00, Short.toUnsignedInt(Float16Bits.fromFloat(Float.NaN)));
        assertEquals(0x0000,
                Short.toUnsignedInt(Float16Bits.fromFloat(Math.scalb(1.0f, -64))));
        assertEquals(0x8000,
                Short.toUnsignedInt(Float16Bits.fromFloat(-Math.scalb(1.0f, -64))));
    }

    @Test
    void roundsBinary64DirectlyAtEveryPositiveAndNegativeBinary16Boundary() {
        for (int lowerBits = 0; lowerBits < 0x7BFF; lowerBits++) {
            double lower = Float.float16ToFloat((short) lowerBits);
            double upper = Float.float16ToFloat((short) (lowerBits + 1));
            double midpoint = (lower + upper) * 0.5d;
            int tie = (lowerBits & 1) == 0 ? lowerBits : lowerBits + 1;

            assertFromDouble(lowerBits, Math.nextDown(midpoint));
            assertFromDouble(tie, midpoint);
            assertFromDouble(lowerBits + 1, Math.nextUp(midpoint));
            assertFromDouble(0x8000 | lowerBits, Math.nextUp(-midpoint));
            assertFromDouble(0x8000 | tie, -midpoint);
            assertFromDouble(0x8000 | lowerBits + 1, Math.nextDown(-midpoint));
        }

        double overflowMidpoint = 65_520.0d;
        assertFromDouble(0x7BFF, Math.nextDown(overflowMidpoint));
        assertFromDouble(0x7C00, overflowMidpoint);
        assertFromDouble(0x7C00, Math.nextUp(overflowMidpoint));
        assertFromDouble(0xFBFF, Math.nextUp(-overflowMidpoint));
        assertFromDouble(0xFC00, -overflowMidpoint);
        assertFromDouble(0xFC00, Math.nextDown(-overflowMidpoint));
        assertFromDouble(0x0000, Math.scalb(1.0d, -35));
        assertFromDouble(0x8000, -Math.scalb(1.0d, -35));
        assertFromDouble(
                0x7E00, Double.longBitsToDouble(0xFFF0_0000_0000_0042L));
    }

    private static void assertFromDouble(int expectedBits, double value) {
        assertEquals(expectedBits, Short.toUnsignedInt(Float16Bits.fromDouble(value)));
    }
}
