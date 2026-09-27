package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** Machine-checked integer and window-geometry parts of the Task 0064 kernel proof. */
class MetalConvolutionPoolingProofTest {
    @Test
    void everyBfloat16WordHasExactNanPartitionAndMonotoneMaximumKey() {
        List<Integer> numbers = new ArrayList<>();
        int nanCount = 0;
        for (int word = 0; word <= 0xffff; word++) {
            boolean expectedNaN = Float.isNaN(Float.intBitsToFloat(word << Short.SIZE));
            assertEquals(expectedNaN, isNan16(word));
            if (expectedNaN) nanCount++; else numbers.add(word);
        }
        assertEquals(254, nanCount);
        numbers.sort(Comparator.comparingInt(MetalConvolutionPoolingProofTest::key16));
        for (int index = 1; index < numbers.size(); index++) {
            float previous = Float.intBitsToFloat(numbers.get(index - 1) << Short.SIZE);
            float current = Float.intBitsToFloat(numbers.get(index) << Short.SIZE);
            assertTrue(Float.compare(previous, current) < 0,
                    Integer.toHexString(numbers.get(index - 1)) + " before "
                            + Integer.toHexString(numbers.get(index)));
        }
    }

    @Test
    void integerMaximumComparatorMatchesFloatingOrderAcrossBoundariesAndRandomWords() {
        int[] boundaries32 = {
            0xffc54321, 0xff800001, 0xff800000, 0xff7fffff, 0x80000000,
            0, 1, 0x007fffff, 0x00800000, 0x7f7fffff, 0x7f800000,
            0x7f800001, 0x7fa12345
        };
        long[] boundaries64 = {
            0xfff8_0000_0000_5678L, 0xfff0_0000_0000_0001L,
            0xfff0_0000_0000_0000L, 0xffef_ffff_ffff_ffffL,
            Long.MIN_VALUE, 0L, 1L, 0x000f_ffff_ffff_ffffL,
            0x0010_0000_0000_0000L, 0x7fef_ffff_ffff_ffffL,
            0x7ff0_0000_0000_0000L, 0x7ff0_0000_0000_0001L,
            0x7ff8_0000_0000_1234L
        };
        for (int left : boundaries32) for (int right : boundaries32) {
            assertEquals(expectedGreater32(left, right), greater32(left, right));
        }
        for (long left : boundaries64) for (long right : boundaries64) {
            assertEquals(expectedGreater64(left, right), greater64(left, right));
        }
        SplittableRandom random = new SplittableRandom(0x64_0064_0064L);
        for (int index = 0; index < 200_000; index++) {
            int left32 = random.nextInt();
            int right32 = random.nextInt();
            assertEquals(expectedGreater32(left32, right32), greater32(left32, right32));
            long left64 = random.nextLong();
            long right64 = random.nextLong();
            assertEquals(expectedGreater64(left64, right64), greater64(left64, right64));
        }
    }

    @Test
    void literalFloorAndCeilGridsMatchEnumeratedOriginsIncludingAllPaddingWindows() {
        for (long input = 1; input <= 8; input++) {
            for (long kernel = 1; kernel <= 5; kernel++) {
                for (long stride = 1; stride <= 4; stride++) {
                    for (long padding = 0; padding <= 5; padding++) {
                        for (long dilation = 1; dilation <= 3; dilation++) {
                            long effective = dilation * (kernel - 1) + 1;
                            long padded = input + 2 * padding;
                            if (padded < effective) continue;
                            for (boolean ceil : List.of(false, true)) {
                                long numerator = padded - effective;
                                long formula = numerator / stride
                                        + (ceil && numerator % stride != 0 ? 1 : 0) + 1;
                                long enumerated = 0;
                                long terminal = numerator;
                                if (ceil && numerator % stride != 0) {
                                    terminal += stride - numerator % stride;
                                }
                                for (long offset = 0; offset <= terminal; offset += stride) {
                                    long origin = offset - padding;
                                    long inside = 0;
                                    long outside = 0;
                                    for (long sample = 0; sample < kernel; sample++) {
                                        long coordinate = origin + sample * dilation;
                                        if (coordinate >= 0 && coordinate < input) inside++;
                                        else outside++;
                                    }
                                    assertEquals(kernel, inside + outside);
                                    enumerated++;
                                }
                                assertEquals(formula, enumerated);
                            }
                        }
                    }
                }
            }
        }
    }

    private static boolean isNan16(int word) {
        return (word & 0x7f80) == 0x7f80 && (word & 0x007f) != 0;
    }

    private static int key16(int word) {
        return (word & 0x8000) != 0 ? ~word & 0xffff : word ^ 0x8000;
    }

    private static int key32(int word) {
        return word < 0 ? ~word : word ^ Integer.MIN_VALUE;
    }

    private static long key64(long word) {
        return word < 0 ? ~word : word ^ Long.MIN_VALUE;
    }

    private static boolean greater32(int candidate, int best) {
        boolean candidateNan = Float.isNaN(Float.intBitsToFloat(candidate));
        boolean bestNan = Float.isNaN(Float.intBitsToFloat(best));
        if (bestNan) return false;
        if (candidateNan) return true;
        return Integer.compareUnsigned(key32(candidate), key32(best)) > 0;
    }

    private static boolean greater64(long candidate, long best) {
        boolean candidateNan = Double.isNaN(Double.longBitsToDouble(candidate));
        boolean bestNan = Double.isNaN(Double.longBitsToDouble(best));
        if (bestNan) return false;
        if (candidateNan) return true;
        return Long.compareUnsigned(key64(candidate), key64(best)) > 0;
    }

    private static boolean expectedGreater32(int candidate, int best) {
        float candidateValue = Float.intBitsToFloat(candidate);
        float bestValue = Float.intBitsToFloat(best);
        if (Float.isNaN(bestValue)) return false;
        if (Float.isNaN(candidateValue)) return true;
        return Float.compare(candidateValue, bestValue) > 0;
    }

    private static boolean expectedGreater64(long candidate, long best) {
        double candidateValue = Double.longBitsToDouble(candidate);
        double bestValue = Double.longBitsToDouble(best);
        if (Double.isNaN(bestValue)) return false;
        if (Double.isNaN(candidateValue)) return true;
        return Double.compare(candidateValue, bestValue) > 0;
    }
}
