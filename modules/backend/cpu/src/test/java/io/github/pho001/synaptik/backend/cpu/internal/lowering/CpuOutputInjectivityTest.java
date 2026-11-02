package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import java.util.HashSet;
import org.junit.jupiter.api.Test;

class CpuOutputInjectivityTest {
    @Test void largeInterleavedResiduesAndAxisPermutationsAreInjective() {
        assertTrue(CpuOutputInjectivity.isInjective(
                new long[]{200_001, 2, 3}, new long[]{8, 3, 2}));
        assertTrue(CpuOutputInjectivity.isInjective(
                new long[]{3, 200_001, 2}, new long[]{2, 8, 3}));
        assertTrue(CpuOutputInjectivity.isInjective(
                new long[]{200_001, 1, 2, 3}, new long[]{8, 0, 3, 2}));
    }

    @Test void residuePeriodBoundaryDoesNotAdmitRealCollisions() {
        assertFalse(CpuOutputInjectivity.isInjective(
                new long[]{200_001, 3, 3}, new long[]{8, 3, 2}));
        assertFalse(CpuOutputInjectivity.isInjective(
                new long[]{200_001, 2, 3}, new long[]{7, 3, 2}));
        assertFalse(CpuOutputInjectivity.isInjective(
                new long[]{200_001, 2, 3}, new long[]{8, 0, 2}));
        assertFalse(CpuOutputInjectivity.isInjective(
                new long[]{200_001, 2, 3}, new long[]{8, 2, 2}));
    }

    @Test void emptyScalarSingletonAndCheckedBoundsRetainTheirMeaning() {
        assertTrue(CpuOutputInjectivity.isInjective(new long[0], new long[0]));
        assertTrue(CpuOutputInjectivity.isInjective(new long[]{1, 1}, new long[]{0, Long.MAX_VALUE}));
        assertTrue(CpuOutputInjectivity.isInjective(
                new long[]{Long.MAX_VALUE, 2, 0}, new long[]{Long.MAX_VALUE, 1, 0}));
        assertThrows(ArithmeticException.class, () -> CpuOutputInjectivity.isInjective(
                new long[]{2}, new long[]{Long.MAX_VALUE}));
        assertThrows(ArithmeticException.class, () -> CpuOutputInjectivity.isInjective(
                new long[]{Long.MAX_VALUE, 2}, new long[]{1, 0}));
        assertThrows(IllegalArgumentException.class, () -> CpuOutputInjectivity.isInjective(
                new long[]{2}, new long[]{-1}));
        assertThrows(IllegalArgumentException.class, () -> CpuOutputInjectivity.isInjective(
                new long[]{-1}, new long[]{1}));
        assertThrows(IllegalArgumentException.class, () -> CpuOutputInjectivity.isInjective(
                new long[]{2}, new long[0]));
    }

    @Test void boundedProofAndExactFallbackAgreeWithIndependentCoordinateEnumeration() {
        for (int a = 1; a <= 3; a++) for (int b = 1; b <= 3; b++) for (int c = 1; c <= 3; c++) {
            for (int x = 0; x <= 7; x++) for (int y = 0; y <= 7; y++) for (int z = 0; z <= 7; z++) {
                var addresses = new HashSet<Long>();
                boolean unique = true;
                for (int i = 0; i < a; i++) for (int j = 0; j < b; j++) for (int k = 0; k < c; k++) {
                    unique &= addresses.add((long) i * x + (long) j * y + (long) k * z);
                }
                assertEquals(unique, CpuOutputInjectivity.isInjective(
                        new long[]{a, b, c}, new long[]{x, y, z}),
                        "extents=" + a + "," + b + "," + c + " strides=" + x + "," + y + "," + z);
            }
        }
    }
}
