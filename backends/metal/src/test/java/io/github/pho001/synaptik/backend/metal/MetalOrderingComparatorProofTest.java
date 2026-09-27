package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** Machine-checked parts of the complete-domain raw-word ordering proof. */
class MetalOrderingComparatorProofTest {
    @Test
    void everyBfloat16WordHasTheExactNanPartitionAndBijectiveMonotoneKey() {
        boolean[] keys = new boolean[1 << Short.SIZE];
        List<Integer> nonNaNs = new ArrayList<>();
        int nanCount = 0;
        for (int word = 0; word <= 0xffff; word++) {
            boolean expectedNaN = (word & 0x7f80) == 0x7f80 && (word & 0x007f) != 0;
            boolean representedNaN = Float.isNaN(Float.intBitsToFloat(word << Short.SIZE));
            assertEquals(expectedNaN, representedNaN, Integer.toHexString(word));
            if (expectedNaN) nanCount++; else nonNaNs.add(word);
            int key = (int) key(DataType.BFLOAT16, word);
            assertFalse(keys[key], "duplicate key " + Integer.toHexString(key));
            keys[key] = true;
        }
        assertEquals(254, nanCount);
        for (boolean represented : keys) assertTrue(represented);

        nonNaNs.sort(Comparator.comparingLong(word -> key(DataType.BFLOAT16, word)));
        for (int index = 1; index < nonNaNs.size(); index++) {
            float previous = Float.intBitsToFloat(nonNaNs.get(index - 1) << Short.SIZE);
            float current = Float.intBitsToFloat(nonNaNs.get(index) << Short.SIZE);
            assertTrue(Float.compare(previous, current) < 0,
                    Integer.toHexString(nonNaNs.get(index - 1)) + " before "
                            + Integer.toHexString(nonNaNs.get(index)));
        }
    }

    @Test
    void floatingClassifiersAndKeysCoverExponentFractionAndSignPartitions() {
        int[] fractions32 = {0, 1, 0x0040_0000, 0x007f_ffff};
        List<Integer> finite32 = new ArrayList<>();
        for (int sign : new int[] {0, Integer.MIN_VALUE}) {
            for (int exponent = 0; exponent <= 0xff; exponent++) {
                for (int fraction : fractions32) {
                    int word = sign | exponent << 23 | fraction;
                    boolean expectedNaN = exponent == 0xff && fraction != 0;
                    assertEquals(expectedNaN, isNaN(DataType.FLOAT32, Integer.toUnsignedLong(word)));
                    assertEquals(expectedNaN, Float.isNaN(Float.intBitsToFloat(word)));
                    if (!expectedNaN) finite32.add(word);
                }
            }
        }
        finite32.sort(Comparator.comparingLong(word -> key(
                DataType.FLOAT32, Integer.toUnsignedLong(word))));
        for (int index = 1; index < finite32.size(); index++) {
            assertTrue(Float.compare(
                    Float.intBitsToFloat(finite32.get(index - 1)),
                    Float.intBitsToFloat(finite32.get(index))) <= 0);
        }

        long[] fractions64 = {0, 1, 0x0008_0000_0000_0000L, 0x000f_ffff_ffff_ffffL};
        List<Long> finite64 = new ArrayList<>();
        for (long sign : new long[] {0, Long.MIN_VALUE}) {
            for (int exponent = 0; exponent <= 0x7ff; exponent++) {
                for (long fraction : fractions64) {
                    long word = sign | (long) exponent << 52 | fraction;
                    boolean expectedNaN = exponent == 0x7ff && fraction != 0;
                    assertEquals(expectedNaN, isNaN(DataType.FLOAT64, word));
                    assertEquals(expectedNaN, Double.isNaN(Double.longBitsToDouble(word)));
                    if (!expectedNaN) finite64.add(word);
                }
            }
        }
        finite64.sort((left, right) -> Long.compareUnsigned(
                key(DataType.FLOAT64, left), key(DataType.FLOAT64, right)));
        for (int index = 1; index < finite64.size(); index++) {
            assertTrue(Double.compare(
                    Double.longBitsToDouble(finite64.get(index - 1)),
                    Double.longBitsToDouble(finite64.get(index))) <= 0);
        }
    }

    @Test
    void signedToggleMatchesSignedOrderAtExtremaAndAcrossDeterministicDomainSamples() {
        int[] intBoundary = {
            Integer.MIN_VALUE, Integer.MIN_VALUE + 1, -2, -1, 0, 1, 2,
            Integer.MAX_VALUE - 1, Integer.MAX_VALUE
        };
        for (int left : intBoundary) for (int right : intBoundary) {
            assertEquals(Integer.signum(Integer.compare(left, right)), Integer.signum(
                    Integer.compareUnsigned(left ^ Integer.MIN_VALUE,
                            right ^ Integer.MIN_VALUE)));
        }
        long[] longBoundary = {
            Long.MIN_VALUE, Long.MIN_VALUE + 1, -2, -1, 0, 1, 2,
            Long.MAX_VALUE - 1, Long.MAX_VALUE
        };
        for (long left : longBoundary) for (long right : longBoundary) {
            assertEquals(Integer.signum(Long.compare(left, right)), Integer.signum(
                    Long.compareUnsigned(left ^ Long.MIN_VALUE, right ^ Long.MIN_VALUE)));
        }
        SplittableRandom random = new SplittableRandom(0x63_0063_0063L);
        for (int index = 0; index < 100_000; index++) {
            int left32 = random.nextInt();
            int right32 = random.nextInt();
            assertEquals(Integer.signum(Integer.compare(left32, right32)), Integer.signum(
                    Integer.compareUnsigned(left32 ^ Integer.MIN_VALUE,
                            right32 ^ Integer.MIN_VALUE)));
            long left64 = random.nextLong();
            long right64 = random.nextLong();
            assertEquals(Integer.signum(Long.compare(left64, right64)), Integer.signum(
                    Long.compareUnsigned(left64 ^ Long.MIN_VALUE,
                            right64 ^ Long.MIN_VALUE)));
        }
    }

    @Test
    void allSmallBoundarySequencesHaveUniqueStableRanksTopKCompactionAndArgTies() {
        for (DataType type : DataType.values()) {
            long[] alphabet = alphabet(type);
            for (int length = 1; length <= 4; length++) {
                long combinations = 1;
                for (int ignored = 0; ignored < length; ignored++) {
                    combinations = Math.multiplyExact(combinations, alphabet.length);
                }
                for (long encoded = 0; encoded < combinations; encoded++) {
                    long state = encoded;
                    long[] sequence = new long[length];
                    for (int index = 0; index < length; index++) {
                        sequence[index] = alphabet[(int) (state % alphabet.length)];
                        state /= alphabet.length;
                    }
                    verifySequence(type, sequence);
                }
            }
        }
    }

    private static void verifySequence(DataType type, long[] sequence) {
        for (boolean ascending : List.of(false, true)) {
            int[] order = order(type, sequence, ascending);
            boolean[] seen = new boolean[sequence.length];
            for (int rank = 0; rank < order.length; rank++) {
                int coordinate = order[rank];
                assertFalse(seen[coordinate]);
                seen[coordinate] = true;
                if (rank > 0) {
                    int previous = order[rank - 1];
                    assertTrue(compare(type, sequence[previous], previous,
                            sequence[coordinate], coordinate, ascending) < 0);
                }
            }
            int[] ranks = new int[sequence.length];
            for (int source = 0; source < sequence.length; source++) {
                for (int candidate = 0; candidate < sequence.length; candidate++) {
                    if (compare(
                            type,
                            sequence[candidate],
                            candidate,
                            sequence[source],
                            source,
                            ascending) < 0) {
                        ranks[source]++;
                    }
                }
            }
            for (boolean sorted : List.of(false, true)) {
                for (int k = 1; k <= sequence.length; k++) {
                    int[] expected = Arrays.copyOf(order, k);
                    if (!sorted) Arrays.sort(expected);
                    int[] placed = new int[k];
                    Arrays.fill(placed, -1);
                    for (int source = 0; source < sequence.length; source++) {
                        if (ranks[source] >= k) continue;
                        int position = ranks[source];
                        if (!sorted) {
                            position = 0;
                            for (int earlier = 0; earlier < source; earlier++) {
                                if (ranks[earlier] < k) position++;
                            }
                        }
                        assertEquals(-1, placed[position]);
                        placed[position] = source;
                    }
                    assertArrayEquals(expected, placed);
                }
            }
        }
        if (type != DataType.BOOL) {
            for (boolean minimum : List.of(false, true)) {
                for (boolean last : List.of(false, true)) {
                    int selected = arg(type, sequence, minimum, last);
                    for (int index = 0; index < sequence.length; index++) {
                        if (isNaN(type, sequence[selected])) {
                            assertTrue(isNaN(type, sequence[index]) || index != selected);
                        } else {
                            assertFalse(isNaN(type, sequence[index]));
                            int relation = semanticCompare(type, sequence[index], sequence[selected]);
                            assertTrue(minimum ? relation >= 0 : relation <= 0);
                        }
                    }
                    for (int index = 0; index < sequence.length; index++) {
                        if (semanticCompare(type, sequence[index], sequence[selected]) == 0
                                && isNaN(type, sequence[index])
                                        == isNaN(type, sequence[selected])) {
                            assertTrue(last ? index <= selected : index >= selected);
                        }
                    }
                }
            }
        }
    }

    private static int[] order(DataType type, long[] sequence, boolean ascending) {
        Integer[] boxed = new Integer[sequence.length];
        for (int index = 0; index < sequence.length; index++) boxed[index] = index;
        Arrays.sort(boxed, (left, right) -> compare(
                type, sequence[left], left, sequence[right], right, ascending));
        int[] result = new int[sequence.length];
        for (int index = 0; index < sequence.length; index++) result[index] = boxed[index];
        return result;
    }

    private static int compare(
            DataType type, long left, int leftIndex, long right, int rightIndex,
            boolean ascending) {
        boolean leftNaN = isNaN(type, left);
        boolean rightNaN = isNaN(type, right);
        if (leftNaN || rightNaN) {
            if (leftNaN != rightNaN) return leftNaN ? 1 : -1;
            return Integer.compare(leftIndex, rightIndex);
        }
        int comparison = semanticCompare(type, left, right);
        if (!ascending) comparison = -comparison;
        return comparison != 0 ? comparison : Integer.compare(leftIndex, rightIndex);
    }

    private static int arg(DataType type, long[] sequence, boolean minimum, boolean last) {
        int selected = 0;
        for (int index = 1; index < sequence.length; index++) {
            boolean candidateNaN = isNaN(type, sequence[index]);
            boolean selectedNaN = isNaN(type, sequence[selected]);
            int comparison = semanticCompare(type, sequence[index], sequence[selected]);
            if (candidateNaN && (!selectedNaN || last)
                    || !candidateNaN && !selectedNaN
                            && ((minimum && comparison < 0) || (!minimum && comparison > 0)
                                    || comparison == 0 && last)) {
                selected = index;
            }
        }
        return selected;
    }

    private static int semanticCompare(DataType type, long left, long right) {
        boolean leftNaN = isNaN(type, left);
        boolean rightNaN = isNaN(type, right);
        if (leftNaN || rightNaN) {
            if (leftNaN == rightNaN) return 0;
            return leftNaN ? 1 : -1;
        }
        return Long.compareUnsigned(key(type, left), key(type, right));
    }

    private static long key(DataType type, long word) {
        return switch (type) {
            case FLOAT64 -> (word & Long.MIN_VALUE) != 0 ? ~word : word ^ Long.MIN_VALUE;
            case INT64 -> word ^ Long.MIN_VALUE;
            case FLOAT32 -> Integer.toUnsignedLong(
                    (int) word < 0 ? ~(int) word : (int) word ^ Integer.MIN_VALUE);
            case INT32 -> Integer.toUnsignedLong((int) word ^ Integer.MIN_VALUE);
            case BFLOAT16 -> {
                int narrowed = (int) word & 0xffff;
                yield ((narrowed & 0x8000) != 0 ? ~narrowed : narrowed ^ 0x8000) & 0xffffL;
            }
            case BOOL -> word;
        };
    }

    private static boolean isNaN(DataType type, long word) {
        return switch (type) {
            case FLOAT64 -> (word & 0x7ff0_0000_0000_0000L) == 0x7ff0_0000_0000_0000L
                    && (word & 0x000f_ffff_ffff_ffffL) != 0;
            case FLOAT32 -> (word & 0x7f80_0000L) == 0x7f80_0000L
                    && (word & 0x007f_ffffL) != 0;
            case BFLOAT16 -> (word & 0x7f80L) == 0x7f80L && (word & 0x007fL) != 0;
            case INT32, INT64, BOOL -> false;
        };
    }

    private static long[] alphabet(DataType type) {
        return switch (type) {
            case FLOAT64 -> new long[] {
                0xfff8_0000_0000_0001L, Long.MIN_VALUE, 0,
                0xbff0_0000_0000_0000L, 0x3ff0_0000_0000_0000L,
                0x7ff0_0000_0000_0001L
            };
            case FLOAT32 -> new long[] {
                0xffc0_0001L, 0x8000_0000L, 0, 0xbf80_0000L,
                0x3f80_0000L, 0x7f80_0001L
            };
            case BFLOAT16 -> new long[] {0xffc1, 0x8000, 0, 0xbf80, 0x3f80, 0x7f81};
            case INT32 -> new long[] {Integer.MIN_VALUE, -1, 0, 1, Integer.MAX_VALUE};
            case INT64 -> new long[] {Long.MIN_VALUE, -1, 0, 1, Long.MAX_VALUE};
            case BOOL -> new long[] {0, 1};
        };
    }
}
