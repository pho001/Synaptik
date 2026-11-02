package io.github.pho001.synaptik.model.operation.scan;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import java.util.List;
import org.junit.jupiter.api.Test;

class ScanSemanticsTest {
    @Test
    void lengthHeaderSnapshotsExplicitRolesAndSeparatesActualFromFormalCounts() {
        var axes = new java.util.ArrayList<>(List.of(java.util.OptionalInt.of(0), java.util.OptionalInt.of(1), java.util.OptionalInt.empty()));
        var attrs = new ScanAttrs(0, 2, 3, 4, 5, 6, true, true, 1, axes);
        axes.clear();
        assertEquals(10, attrs.inputCount());
        assertEquals(13, attrs.outputCount());
        assertEquals(3, attrs.sequenceRowAxes().size());
        assertThrows(UnsupportedOperationException.class, () -> attrs.sequenceRowAxes().clear());
        for (int rows : new int[] {-1, 3}) assertThrows(IllegalArgumentException.class,
                () -> new ScanAttrs(0, 2, 1, 0, 0, 0, false, true, rows, List.of(java.util.OptionalInt.of(0))));
        assertThrows(IllegalArgumentException.class, () -> new ScanAttrs(0, 1, 1, 0, 0, 0, false, false, 0, List.of(java.util.OptionalInt.empty())));
        assertThrows(IllegalArgumentException.class, () -> new ScanAttrs(0, 1, 1, 0, 0, 0, false, true, 1, List.of()));
        assertThrows(IllegalArgumentException.class,
                () -> new ScanAttrs(0, 1, 1, 0, 0, 0, false, true, 1, List.of(java.util.OptionalInt.of(-1))));
        for (int axis : new int[] {0, 1, 2, 5, Integer.MAX_VALUE}) {
            var declared = new ScanAttrs(0, 0, 1, 0, 1, 0, false, true, 0,
                    List.of(java.util.OptionalInt.of(axis)));
            assertEquals(java.util.OptionalInt.of(axis), declared.sequenceRowAxes().getFirst());
        }
        assertThrows(NullPointerException.class, () -> new ScanAttrs(0, 1, 1, 0, 0, 0, false, true, 1, null));
        assertThrows(NullPointerException.class, () -> new ScanAttrs(0, 1, 1, 0, 0, 0, false, true, 1, java.util.Arrays.asList((java.util.OptionalInt) null)));
        assertThrows(ArithmeticException.class, () -> new ScanAttrs(0, 1, 1, Integer.MAX_VALUE - 2, 0, 0, false, true, 1, List.of(java.util.OptionalInt.of(0))));
    }
    @Test
    void scalarHeaderAndStructuralSignatureRetainFixedCountMeaning() {
        ScanAttrs attrs = new ScanAttrs(0, 2, 3, 4, 5, 6, true);
        Operation op = new Operation(ScanKind.SCAN, attrs);
        assertEquals(9, attrs.inputCount());
        assertEquals(13, attrs.outputCount());
        assertSame(attrs, op.attrs());
        assertSame(ScanKind.SCAN.signatures(), ScanKind.SCAN.signatures());
        op.signature().validateOccurrence(9, 13);
        assertThrows(IllegalArgumentException.class, () -> new Operation(ScanKind.SCAN, NoOperationAttrs.INSTANCE));
        assertThrows(UnsupportedOperationException.class, () -> ScanKind.SCAN.signatures().clear());
    }

    @Test
    void explicitLengthModeSeparatesZeroRowCarriesFromAllValidAndChecksFullCounts() {
        for (boolean reverse : List.of(false, true)) {
            var all = new ScanAttrs(0, 0, 1, 0, 1, 0, reverse);
            var length = new ScanAttrs(0, 0, 1, 0, 1, 0, reverse, true, 0,
                    List.of(java.util.OptionalInt.of(0)));
            assertFalse(all.lengthAware());
            assertTrue(length.lengthAware());
            assertEquals(1, all.inputCount());
            assertEquals(2, length.inputCount());
            assertEquals(1, length.outputCount());
            new Operation(ScanKind.SCAN, all).signature().validateOccurrence(1, 1);
        }
        assertThrows(IllegalArgumentException.class, () -> new ScanAttrs(0, -1, 1, 0, 1, 0, false));
        assertThrows(IllegalArgumentException.class, () -> new ScanAttrs(0, 1, 1, 0, 0, 0,
                false, false, 1, List.of()));
        assertThrows(IllegalArgumentException.class, () -> new ScanAttrs(0, 0, 1, 0, 0, 1, false));
        assertThrows(ArithmeticException.class, () -> new ScanAttrs(0, 0, 1, Integer.MAX_VALUE,
                1, 0, false));
        assertThrows(ArithmeticException.class, () -> new ScanAttrs(0, 0, 1, 0,
                1, Integer.MAX_VALUE, false));
    }

    @Test
    void rejectsInvalidCountsAndCheckedTotalOverflow() {
        assertThrows(IllegalArgumentException.class, () -> new ScanAttrs(-1, 1, 1, 0, 0, 0, false));
        assertThrows(IllegalArgumentException.class, () -> new ScanAttrs(0, 0, 1, 0, 0, 0, false));
        assertThrows(IllegalArgumentException.class, () -> new ScanAttrs(0, 1, 0, 0, 0, 0, false));
        assertThrows(IllegalArgumentException.class, () -> new ScanAttrs(0, 1, 1, -1, 0, 0, false));
        assertThrows(IllegalArgumentException.class, () -> new ScanAttrs(0, 1, 1, 0, -1, 0, false));
        assertThrows(IllegalArgumentException.class, () -> new ScanAttrs(0, 1, 1, 0, 0, -1, false));
        assertThrows(ArithmeticException.class, () -> new ScanAttrs(0, Integer.MAX_VALUE, 1, 0, 0, 0, false));
        assertThrows(ArithmeticException.class, () -> new ScanAttrs(0, 1, 1, 0, Integer.MAX_VALUE, 0, false));
    }
}
