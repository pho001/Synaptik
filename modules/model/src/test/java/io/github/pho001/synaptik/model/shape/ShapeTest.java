package io.github.pho001.synaptik.model.shape;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.OptionalLong;
import org.junit.jupiter.api.Test;

class ShapeTest {
    @Test
    void scalarIsCanonicalRankZeroWithOneElementAndNoAxis() {
        Shape scalar = Shape.scalar();

        assertSame(scalar, Shape.of());
        assertSame(scalar, Shape.ofDimensions());
        assertEquals(0, scalar.rank());
        assertTrue(scalar.dimensions().isEmpty());
        assertTrue(scalar.isFullyStatic());
        assertEquals(OptionalLong.of(1), scalar.knownElementCount());
        assertArrayEquals(new long[0], scalar.toLongArray());
        assertThrows(IndexOutOfBoundsException.class, () -> scalar.normalizeAxis(0));
        assertThrows(IndexOutOfBoundsException.class, () -> scalar.normalizeAxis(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> scalar.dimension(0));
    }

    @Test
    void staticShapeSupportsZeroAndCheckedElementCounts() {
        Shape ordinary = Shape.of(2, 3, 4);
        Shape empty = Shape.of(2, 0, 4);

        assertEquals(OptionalLong.of(24), ordinary.knownElementCount());
        assertEquals(OptionalLong.of(0), empty.knownElementCount());
        assertArrayEquals(new long[] {2, 0, 4}, empty.toLongArray());
    }

    @Test
    void zeroDimensionAvoidsIrrelevantOverflow() {
        Shape empty = Shape.of(Long.MAX_VALUE, 2, 0);

        assertEquals(OptionalLong.of(0), empty.knownElementCount());
    }

    @Test
    void nonZeroElementCountOverflowFails() {
        Shape overflowing = Shape.of(Long.MAX_VALUE, 2);

        assertThrows(ArithmeticException.class, overflowing::knownElementCount);
    }

    @Test
    void dynamicShapeReportsUnknownCountAndRejectsStaticExtraction() {
        Shape shape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(4));

        assertFalse(shape.isFullyStatic());
        assertEquals(OptionalLong.empty(), shape.knownElementCount());
        assertThrows(IllegalStateException.class, shape::toLongArray);
    }

    @Test
    void normalizesPositiveAndNegativeAxes() {
        Shape shape = Shape.of(2, 3, 4);

        assertEquals(0, shape.normalizeAxis(0));
        assertEquals(1, shape.normalizeAxis(1));
        assertEquals(2, shape.normalizeAxis(2));
        assertEquals(0, shape.normalizeAxis(-3));
        assertEquals(1, shape.normalizeAxis(-2));
        assertEquals(2, shape.normalizeAxis(-1));
        assertEquals(new StaticDimension(3), shape.dimension(1));
        assertEquals(new StaticDimension(4), shape.dimension(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> shape.normalizeAxis(3));
        assertThrows(IndexOutOfBoundsException.class, () -> shape.normalizeAxis(-4));
    }

    @Test
    void constructionAndAccessDefensivelyProtectShapeState() {
        long[] sizes = {2, 3};
        Shape staticShape = Shape.of(sizes);
        sizes[0] = 99;

        Dimension[] dimensions = {
            new DynamicDimension("batch"), new StaticDimension(3)
        };
        Shape dynamicShape = Shape.ofDimensions(dimensions);
        dimensions[0] = new StaticDimension(99);

        long[] extracted = staticShape.toLongArray();
        extracted[0] = 77;

        assertArrayEquals(new long[] {2, 3}, staticShape.toLongArray());
        assertEquals(new DynamicDimension("batch"), dynamicShape.dimension(0));
        assertThrows(
                UnsupportedOperationException.class,
                () -> dynamicShape.dimensions().add(new StaticDimension(1)));
    }

    @Test
    void rejectsNullAndNegativeConstructionInputs() {
        assertThrows(NullPointerException.class, () -> Shape.of((long[]) null));
        assertThrows(NullPointerException.class, () -> Shape.ofDimensions((Dimension[]) null));
        assertThrows(
                NullPointerException.class,
                () -> Shape.ofDimensions(new StaticDimension(1), null));
        assertThrows(IllegalArgumentException.class, () -> Shape.of(2, -1));
    }

    @Test
    void equalityHashingAndDiagnosticsAreStructuralAndOrdered() {
        Shape first = Shape.ofDimensions(
                new DynamicDimension("N"), new StaticDimension(0), new StaticDimension(4));
        Shape equal = Shape.ofDimensions(
                new DynamicDimension("N"), new StaticDimension(0), new StaticDimension(4));
        Shape reordered = Shape.ofDimensions(
                new StaticDimension(0), new DynamicDimension("N"), new StaticDimension(4));

        assertEquals(first, equal);
        assertEquals(first.hashCode(), equal.hashCode());
        assertFalse(first.equals(reordered));
        assertEquals("Shape[N, 0, 4]", first.toString());
        assertEquals("Shape[]", Shape.scalar().toString());
    }
}
