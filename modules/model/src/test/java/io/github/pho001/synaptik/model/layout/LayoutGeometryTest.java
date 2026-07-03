package io.github.pho001.synaptik.model.layout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import org.junit.jupiter.api.Test;

class LayoutGeometryTest {
    @Test
    void calculatesCanonicalStridesForScalarOrdinarySingletonAndEmptyShapes() {
        assertArrayEquals(new long[0], LayoutGeometry.canonicalStrides(Shape.scalar()));
        assertArrayEquals(
                new long[] {3, 1}, LayoutGeometry.canonicalStrides(Shape.of(2, 3)));
        assertArrayEquals(
                new long[] {3, 3, 1}, LayoutGeometry.canonicalStrides(Shape.of(2, 1, 3)));
        assertArrayEquals(
                new long[] {0, 4, 1}, LayoutGeometry.canonicalStrides(Shape.of(2, 0, 4)));
    }

    @Test
    void canonicalStridesCheckRequiredProductsButNotTotalElementCount() {
        assertArrayEquals(
                new long[] {2, 1},
                LayoutGeometry.canonicalStrides(Shape.of(Long.MAX_VALUE, 2)));
        assertArrayEquals(
                new long[] {0, Long.MAX_VALUE, 1},
                LayoutGeometry.canonicalStrides(Shape.of(2, 0, Long.MAX_VALUE)));

        assertThrows(
                ArithmeticException.class,
                () -> LayoutGeometry.canonicalStrides(Shape.of(1, Long.MAX_VALUE, 2)));
    }

    @Test
    void rejectsDynamicShapeGeometry() {
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));

        assertThrows(IllegalArgumentException.class, () -> LayoutGeometry.canonicalStrides(dynamic));
        assertThrows(
                IllegalArgumentException.class,
                () -> LayoutGeometry.hasBroadcastZeroStride(dynamic, new long[] {0, 1}));
        assertThrows(
                IllegalArgumentException.class,
                () -> LayoutGeometry.referencedElementSpan(dynamic, new long[] {3, 1}, 0));
    }

    @Test
    void classifiesCanonicalBeforeBroadcastAndOtherwiseUsesStrideFacts() {
        assertEquals(
                LayoutKind.DENSE_CONTIGUOUS,
                LayoutGeometry.classify(Shape.of(2, 3), new long[] {3, 1}, 0));
        assertEquals(
                LayoutKind.DENSE_WITH_OFFSET,
                LayoutGeometry.classify(Shape.of(2, 3), new long[] {3, 1}, 4));
        assertEquals(
                LayoutKind.DENSE_CONTIGUOUS,
                LayoutGeometry.classify(Shape.of(2, 0, 4), new long[] {0, 4, 1}, 0));
        assertEquals(
                LayoutKind.BROADCAST_ZERO_STRIDE,
                LayoutGeometry.classify(Shape.of(2, 3), new long[] {0, 1}, 0));
        assertEquals(
                LayoutKind.STRIDED,
                LayoutGeometry.classify(Shape.of(2, 3), new long[] {1, 2}, 0));
        assertEquals(
                LayoutKind.STRIDED,
                LayoutGeometry.classify(Shape.of(1, 3), new long[] {0, 1}, 0));
    }

    @Test
    void distinguishesRawAndBroadcastZeroStrides() {
        assertTrue(LayoutGeometry.hasZeroStride(new long[] {0, 1}));
        assertFalse(LayoutGeometry.hasZeroStride(new long[] {3, 1}));
        assertTrue(LayoutGeometry.hasBroadcastZeroStride(Shape.of(2, 3), new long[] {0, 1}));
        assertFalse(LayoutGeometry.hasBroadcastZeroStride(Shape.of(1, 3), new long[] {0, 1}));
        assertFalse(LayoutGeometry.hasBroadcastZeroStride(Shape.of(0, 3), new long[] {0, 1}));
    }

    @Test
    void calculatesReferencedSpanForDenseOffsetStridedBroadcastScalarAndEmptyLayouts() {
        assertEquals(6, LayoutGeometry.referencedElementSpan(Shape.of(2, 3), new long[] {3, 1}, 0));
        assertEquals(6, LayoutGeometry.referencedElementSpan(Shape.of(3), new long[] {1}, 3));
        assertEquals(7, LayoutGeometry.referencedElementSpan(Shape.of(2, 2), new long[] {5, 1}, 0));
        assertEquals(3, LayoutGeometry.referencedElementSpan(Shape.of(2, 3), new long[] {0, 1}, 0));
        assertEquals(1, LayoutGeometry.referencedElementSpan(Shape.scalar(), new long[0], 0));
        assertEquals(5, LayoutGeometry.referencedElementSpan(Shape.scalar(), new long[0], 4));
        assertEquals(0, LayoutGeometry.referencedElementSpan(Shape.of(2, 0, 4), new long[] {0, 4, 1}, Long.MAX_VALUE));
    }

    @Test
    void referencedSpanChecksMultiplicationAndAdditionOverflow() {
        assertThrows(
                ArithmeticException.class,
                () -> LayoutGeometry.referencedElementSpan(
                        Shape.of(Long.MAX_VALUE), new long[] {2}, 0));
        assertThrows(
                ArithmeticException.class,
                () -> LayoutGeometry.referencedElementSpan(Shape.scalar(), new long[0], Long.MAX_VALUE));
        assertThrows(
                ArithmeticException.class,
                () -> LayoutGeometry.referencedElementSpan(Shape.of(2), new long[] {1}, Long.MAX_VALUE));
    }
}
