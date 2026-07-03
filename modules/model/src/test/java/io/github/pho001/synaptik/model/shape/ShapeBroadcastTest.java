package io.github.pho001.synaptik.model.shape;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ShapeBroadcastTest {
    @Test
    void broadcastsScalarInEitherOperandPosition() {
        Shape matrix = Shape.of(2, 3);

        assertEquals(matrix, ShapeBroadcast.broadcast(Shape.scalar(), matrix));
        assertEquals(matrix, ShapeBroadcast.broadcast(matrix, Shape.scalar()));
        assertEquals(Shape.scalar(), ShapeBroadcast.broadcast(Shape.scalar(), Shape.scalar()));
    }

    @Test
    void rightAlignsDifferentRanks() {
        Shape vector = Shape.of(3);
        Shape tensor = Shape.of(2, 1, 3);

        assertEquals(Shape.of(2, 1, 3), ShapeBroadcast.broadcast(vector, tensor));
        assertEquals(Shape.of(2, 1, 3), ShapeBroadcast.broadcast(tensor, vector));
    }

    @Test
    void expandsSingletonDimensionsAcrossBothInputs() {
        Shape left = Shape.of(2, 1, 3);
        Shape right = Shape.of(1, 4, 3);

        assertEquals(Shape.of(2, 4, 3), ShapeBroadcast.broadcast(left, right));
        assertEquals(Shape.of(2, 4, 3), ShapeBroadcast.broadcast(right, left));
    }

    @Test
    void preservesZeroWhenBroadcastAgainstSingleton() {
        Shape empty = Shape.of(0, 3);
        Shape singleton = Shape.of(1, 3);

        assertEquals(Shape.of(0, 3), ShapeBroadcast.broadcast(empty, singleton));
        assertEquals(Shape.of(0, 3), ShapeBroadcast.broadcast(singleton, empty));
    }

    @Test
    void preservesEqualSymbolsAndBroadcastsTheirSingletonAxes() {
        Shape left = Shape.ofDimensions(
                new DynamicDimension("N"), new StaticDimension(1));
        Shape right = Shape.ofDimensions(
                new DynamicDimension("N"), new StaticDimension(4));

        assertEquals(
                Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(4)),
                ShapeBroadcast.broadcast(left, right));
    }

    @Test
    void broadcastsSingletonToDynamicDimension() {
        Shape singleton = Shape.of(1);
        Shape dynamic = Shape.ofDimensions(new DynamicDimension("N"));

        assertEquals(dynamic, ShapeBroadcast.broadcast(singleton, dynamic));
        assertEquals(dynamic, ShapeBroadcast.broadcast(dynamic, singleton));
    }

    @Test
    void rejectsIncompatibleStaticDimensions() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ShapeBroadcast.broadcast(Shape.of(2, 3), Shape.of(2, 4)));
        assertThrows(
                IllegalArgumentException.class,
                () -> ShapeBroadcast.broadcast(Shape.of(0), Shape.of(2)));
    }

    @Test
    void rejectsUnprovableDynamicCombinations() {
        Shape batch = Shape.ofDimensions(new DynamicDimension("batch"));
        Shape sequence = Shape.ofDimensions(new DynamicDimension("sequence"));

        assertThrows(
                IllegalArgumentException.class,
                () -> ShapeBroadcast.broadcast(batch, sequence));
        assertThrows(
                IllegalArgumentException.class,
                () -> ShapeBroadcast.broadcast(batch, Shape.of(2)));
        assertThrows(
                IllegalArgumentException.class,
                () -> ShapeBroadcast.broadcast(Shape.of(2), batch));
    }

    @Test
    void rejectsNullShapes() {
        assertThrows(
                NullPointerException.class,
                () -> ShapeBroadcast.broadcast(null, Shape.scalar()));
        assertThrows(
                NullPointerException.class,
                () -> ShapeBroadcast.broadcast(Shape.scalar(), null));
    }
}
