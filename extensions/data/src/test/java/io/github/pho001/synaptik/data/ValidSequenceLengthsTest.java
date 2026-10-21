package io.github.pho001.synaptik.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** Verifies the bounded immutable host-prefix API independently of any tensor or time extent. */
final class ValidSequenceLengthsTest {
    @Test
    void exposesOnlyTheApprovedFinalValueApi() throws NoSuchMethodException {
        assertTrue(Modifier.isFinal(ValidSequenceLengths.class.getModifiers()));
        assertEquals(0, ValidSequenceLengths.class.getConstructors().length);
        assertEquals(1, ValidSequenceLengths.class.getDeclaredConstructors().length);
        assertTrue(Modifier.isPrivate(
                ValidSequenceLengths.class.getDeclaredConstructors()[0].getModifiers()));
        assertEquals(Set.of("of", "batchSize", "length", "toLongArray", "equals", "hashCode", "toString"),
                Arrays.stream(ValidSequenceLengths.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .map(method -> method.getName()).collect(Collectors.toSet()));
        assertTrue(ValidSequenceLengths.class.getMethod("of", long[].class).isVarArgs());
        assertTrue(Modifier.isStatic(
                ValidSequenceLengths.class.getMethod("of", long[].class).getModifiers()));
        assertEquals(1, ValidSequenceLengths.class.getDeclaredFields().length);
        assertEquals(long[].class, ValidSequenceLengths.class.getDeclaredFields()[0].getType());
    }

    @Test
    void retainsOriginalOrderIncludingZeroAndMaximumLong() {
        var lengths = ValidSequenceLengths.of(3, 0, Long.MAX_VALUE, 1);
        assertEquals(4, lengths.batchSize());
        assertEquals(3, lengths.length(0));
        assertEquals(0, lengths.length(1));
        assertEquals(Long.MAX_VALUE, lengths.length(2));
        assertEquals(1, lengths.length(3));
    }

    @Test
    void copiesSourceAndEveryExportWithoutChangingEqualityHashOrDiagnostics() {
        long[] source = {3, 0, 2};
        var lengths = ValidSequenceLengths.of(source);
        var equal = ValidSequenceLengths.of(3, 0, 2);
        int hash = lengths.hashCode();
        String text = lengths.toString();
        source[0] = -1;
        long[] first = lengths.toLongArray();
        long[] second = lengths.toLongArray();
        assertNotSame(source, first);
        assertNotSame(first, second);
        first[1] = 99;
        assertArrayEquals(new long[] {3, 0, 2}, second);
        assertArrayEquals(new long[] {3, 0, 2}, lengths.toLongArray());
        assertEquals(equal, lengths);
        assertEquals(hash, lengths.hashCode());
        assertEquals(text, lengths.toString());
    }

    @Test
    void emptyLengthsHaveFreshExportsAndStructuralValueSemantics() {
        long[] source = {};
        var empty = ValidSequenceLengths.of(source);
        assertEquals(0, empty.batchSize());
        assertArrayEquals(source, empty.toLongArray());
        assertNotSame(source, empty.toLongArray());
        assertNotSame(empty.toLongArray(), empty.toLongArray());
        assertEquals(empty, ValidSequenceLengths.of());
        assertEquals(empty.hashCode(), ValidSequenceLengths.of().hashCode());
        assertThrows(IndexOutOfBoundsException.class, () -> empty.length(0));
        assertThrows(IndexOutOfBoundsException.class, () -> empty.length(-1));
    }

    @Test
    void rejectsNullByParameterName() {
        assertEquals("lengths", assertThrows(NullPointerException.class,
                () -> ValidSequenceLengths.of((long[]) null)).getMessage());
    }

    @Test
    void rejectsFirstNegativeRowWithoutMutatingSource() {
        long[] source = {2, -1, Long.MIN_VALUE};
        String message = assertThrows(IllegalArgumentException.class,
                () -> ValidSequenceLengths.of(source)).getMessage();
        assertTrue(message.contains("row 1"));
        assertArrayEquals(new long[] {2, -1, Long.MIN_VALUE}, source);
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ValidSequenceLengths.of(-1)).getMessage().contains("row 0"));
    }

    @Test
    void invalidIndicesNeverWrapOrNarrow() {
        var lengths = ValidSequenceLengths.of(3, 2);
        for (int row : new int[] {Integer.MIN_VALUE, -2, -1, 2, 3, Integer.MAX_VALUE}) {
            assertThrows(IndexOutOfBoundsException.class, () -> lengths.length(row));
        }
    }

    @Test
    void equalityIsOrderedStructuralAndTypeSensitive() {
        var first = ValidSequenceLengths.of(1, 0, 2);
        var equal = ValidSequenceLengths.of(1, 0, 2);
        assertEquals(first, first);
        assertEquals(first, equal);
        assertEquals(equal, first);
        assertEquals(first.hashCode(), equal.hashCode());
        assertNotEquals(first, ValidSequenceLengths.of(2, 0, 1));
        assertNotEquals(first, ValidSequenceLengths.of(1, 0));
        assertFalse(first.equals(null));
        assertFalse(first.equals(new long[] {1, 0, 2}));
    }

    @Test
    void diagnosticsDescribeOnlyOrderedMetadata() {
        String text = ValidSequenceLengths.of(0, Long.MAX_VALUE, 2).toString();
        assertTrue(text.contains("ValidSequenceLengths"));
        assertTrue(text.contains("[0, " + Long.MAX_VALUE + ", 2]"));
        assertFalse(text.contains("Tensor"));
        assertFalse(text.contains("storage"));
    }
}
