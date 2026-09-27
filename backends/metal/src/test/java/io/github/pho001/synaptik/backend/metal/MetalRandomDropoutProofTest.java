package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import java.math.BigInteger;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** Machine-checked integer proof obligations and fixed vectors for Task 0065. */
class MetalRandomDropoutProofTest {
    private static final long KEY_BIAS = 0x9e37_79b9_7f4a_7c15L;
    private static final long M1 = 0xbf58_476d_1ce4_e5b9L;
    private static final long M2 = 0x94d0_49bb_1331_11ebL;

    @Test
    void exactThresholdHasEveryNamedBinary64Boundary() {
        long nextAboveTwoMinus53 = Double.doubleToRawLongBits(0x1.0000000000001p-53);
        long largestBelowOne = Double.doubleToRawLongBits(Math.nextDown(1.0d));
        long[][] vectors = {
            {Double.doubleToRawLongBits(0.0d), 0L},
            {Double.doubleToRawLongBits(-0.0d), 0L},
            {1L, 1L},
            {Double.doubleToRawLongBits(0x1.0p-54), 1L},
            {Double.doubleToRawLongBits(0x1.0p-53), 1L},
            {nextAboveTwoMinus53, 2L},
            {Double.doubleToRawLongBits(0.5d), 1L << 52},
            {largestBelowOne, (1L << 53) - 1L}
        };
        for (long[] vector : vectors) {
            assertEquals(vector[1], MetalMpsGraphProgram.dropoutThreshold(vector[0]),
                    Long.toHexString(vector[0]));
        }
    }

    @Test
    void integerThresholdMatchesIndependentBigIntegerOracleAcrossBinary64Domain() {
        SplittableRandom random = new SplittableRandom(0x0065_0065_0065L);
        for (int index = 0; index < 200_000; index++) {
            double probability = random.nextDouble();
            long bits = Double.doubleToRawLongBits(probability);
            assertEquals(oracleThreshold(bits), MetalMpsGraphProgram.dropoutThreshold(bits));
        }
    }

    @Test
    void complementNarrowsAfterBinary64SubtractionRatherThanBeforeIt() {
        long probabilityBits = 0x3fef_fedc_6e3c_8059L;
        assertEquals(0x3911_c8e2,
                MetalMpsGraphProgram.dropoutComplementBits(probabilityBits));
        float probabilityFirst = (float) Double.longBitsToDouble(probabilityBits);
        assertEquals(0x3911_d000, Float.floatToRawIntBits(1.0f - probabilityFirst));
        assertNotEquals(Float.floatToRawIntBits(1.0f - probabilityFirst),
                MetalMpsGraphProgram.dropoutComplementBits(probabilityBits));
    }

    @Test
    void splitMixKeyCounterBitstreamMatchesFixedUnsignedVectorsIncludingWraparound() {
        assertArrayEquals(words(
                "48218226ff3cd4bf", "ea8568d2e45fd6cb", "a559650848dc2883",
                "7c1141ab3f530058", "4a4ebce365728b7f", "2cd1e7f222744ee3"),
                words(0L, 0L, 6));
        assertArrayEquals(words(
                "19420d43f429a92b", "d015649381301eed", "ad8cc7c8686c1578",
                "61d7b0175c5541db", "c9a25d85b6d2ff1b", "aee10726c9636efd"),
                words(0x0123_4567_89ab_cdefL, 0xfedc_ba98_7654_3210L, 6));
        assertArrayEquals(words(
                "82d1a3576a8a7a2b", "a6112c3025616794", "e8ba9f99ca933538",
                "445018e305810b78", "19438ae6b813b33d", "64b174495128d35e"),
                words(-1L, -3L, 6));
        assertEquals(3L, -3L + 6L, "counter advancement is modulo 2^64");
        assertVector(0L, 0L, "48218226ff3cd4bf", "09043044dfe79a");
        assertVector(0L, 1L, "ea8568d2e45fd6cb", "1d50ad1a5c8bfa");
        assertVector(1L, 0L, "dce423fc82c0d5b8", "1b9c847f90581a");
        assertVector(-1L, -1L, "e8ba9f99ca933538", "1d1753f3395266");
        assertVector(0x1234L, 7L, "3e4cf5a0c9489779", "07c99eb4192912");
    }

    private static void assertVector(
            long key, long counter, String expectedWord, String expectedTop53) {
        long word = words(key, counter, 1)[0];
        assertEquals(Long.parseUnsignedLong(expectedWord, 16), word);
        assertEquals(Long.parseUnsignedLong(expectedTop53, 16), word >>> 11);
    }

    private static long oracleThreshold(long bits) {
        long magnitude = bits & Long.MAX_VALUE;
        if (magnitude == 0L) return 0L;
        int exponent = (int) (magnitude >>> 52) & 0x7ff;
        long fraction = magnitude & 0x000f_ffff_ffff_ffffL;
        BigInteger numerator;
        int shift;
        if (exponent == 0) {
            numerator = BigInteger.valueOf(fraction);
            shift = -1021;
        } else {
            numerator = BigInteger.valueOf((1L << 52) | fraction);
            shift = exponent - 1022;
        }
        BigInteger result;
        if (shift >= 0) {
            result = numerator.shiftLeft(shift);
        } else {
            BigInteger denominator = BigInteger.ONE.shiftLeft(-shift);
            result = numerator.add(denominator).subtract(BigInteger.ONE).divide(denominator);
        }
        return result.longValueExact();
    }

    private static long[] words(long key, long counter, int count) {
        long[] result = new long[count];
        long keyOffset = mix64(key + KEY_BIAS);
        for (int index = 0; index < count; index++) {
            result[index] = mix64(counter + index + keyOffset);
        }
        return result;
    }

    private static long mix64(long word) {
        word = (word ^ (word >>> 30)) * M1;
        word = (word ^ (word >>> 27)) * M2;
        return word ^ (word >>> 31);
    }

    private static long[] words(String... hexadecimal) {
        long[] result = new long[hexadecimal.length];
        for (int index = 0; index < hexadecimal.length; index++) {
            result[index] = Long.parseUnsignedLong(hexadecimal[index], 16);
        }
        return result;
    }
}
