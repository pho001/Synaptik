package io.github.pho001.synaptik.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class LayoutDescriptorTest {
    @Test
    void createsCanonicalContiguousDescriptor() {
        LayoutDescriptor descriptor = LayoutDescriptor.contiguous(Shape.of(2, 3));

        assertEquals(2, descriptor.rank());
        assertEquals(LayoutKind.DENSE_CONTIGUOUS, descriptor.kind());
        assertArrayEquals(new long[] {3, 1}, descriptor.strides());
        assertEquals(0, descriptor.storageOffset());
        assertFalse(descriptor.isView());
        assertTrue(descriptor.isContiguous());
        assertFalse(descriptor.hasStorageOffset());
        assertFalse(descriptor.hasZeroStride());
        assertFalse(descriptor.isBroadcast());
        assertEquals(6, descriptor.referencedElementSpan());
    }

    @Test
    void supportsDenseOffsetAndIndependentViewMetadata() {
        LayoutDescriptor nonView = LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 2, false);
        LayoutDescriptor view = LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 2, true);

        assertEquals(LayoutKind.DENSE_WITH_OFFSET, nonView.kind());
        assertTrue(nonView.isContiguous());
        assertTrue(nonView.hasStorageOffset());
        assertFalse(nonView.isView());
        assertTrue(view.isView());
        assertEquals(8, view.referencedElementSpan());
    }

    @Test
    void createsStridedAndBroadcastDescriptors() {
        LayoutDescriptor strided = LayoutDescriptor.of(
                Shape.of(2, 3), new long[] {1, 2}, 0, false);
        LayoutDescriptor singletonZeroStride = LayoutDescriptor.of(
                Shape.of(1, 3), new long[] {0, 1}, 0, false);
        LayoutDescriptor broadcast = LayoutDescriptor.of(
                Shape.of(2, 3), new long[] {0, 1}, 0, true);

        assertEquals(LayoutKind.STRIDED, strided.kind());
        assertFalse(strided.isContiguous());
        assertFalse(strided.hasZeroStride());
        assertFalse(strided.isView());
        assertEquals(6, strided.referencedElementSpan());

        assertEquals(LayoutKind.STRIDED, singletonZeroStride.kind());
        assertTrue(singletonZeroStride.hasZeroStride());
        assertFalse(singletonZeroStride.isBroadcast());
        assertFalse(singletonZeroStride.isView());

        assertEquals(LayoutKind.BROADCAST_ZERO_STRIDE, broadcast.kind());
        assertTrue(broadcast.hasZeroStride());
        assertTrue(broadcast.isBroadcast());
        assertTrue(broadcast.isView());
        assertEquals(3, broadcast.referencedElementSpan());
    }

    @Test
    void canonicalEmptyLayoutRemainsDenseDespiteRawZeroStride() {
        LayoutDescriptor descriptor = LayoutDescriptor.contiguous(Shape.of(2, 0, 4));

        assertEquals(LayoutKind.DENSE_CONTIGUOUS, descriptor.kind());
        assertArrayEquals(new long[] {0, 4, 1}, descriptor.strides());
        assertTrue(descriptor.hasZeroStride());
        assertFalse(descriptor.isBroadcast());
        assertEquals(0, descriptor.referencedElementSpan());
    }

    @Test
    void scalarHasNoAxesAndReferencesOneElement() {
        LayoutDescriptor scalar = LayoutDescriptor.contiguous(Shape.scalar());

        assertEquals(0, scalar.rank());
        assertArrayEquals(new long[0], scalar.strides());
        assertEquals(1, scalar.referencedElementSpan());
        assertThrows(IndexOutOfBoundsException.class, () -> scalar.stride(0));
        assertThrows(IndexOutOfBoundsException.class, () -> scalar.stride(-1));
    }

    @Test
    void supportsPositiveAndNegativeStrideAxes() {
        LayoutDescriptor descriptor = LayoutDescriptor.contiguous(Shape.of(2, 3, 4));

        assertEquals(12, descriptor.stride(0));
        assertEquals(4, descriptor.stride(1));
        assertEquals(1, descriptor.stride(2));
        assertEquals(12, descriptor.stride(-3));
        assertEquals(4, descriptor.stride(-2));
        assertEquals(1, descriptor.stride(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> descriptor.stride(3));
        assertThrows(IndexOutOfBoundsException.class, () -> descriptor.stride(-4));
        assertThrows(IndexOutOfBoundsException.class, () -> descriptor.stride(Integer.MIN_VALUE));
    }

    @Test
    void rejectsNullDynamicMismatchedNegativeAndUnmarkedBroadcastInputs() {
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));

        assertThrows(NullPointerException.class, () -> LayoutDescriptor.contiguous(null));
        assertThrows(NullPointerException.class, () -> LayoutDescriptor.of(null, new long[0], 0, false));
        assertThrows(NullPointerException.class, () -> LayoutDescriptor.of(Shape.scalar(), null, 0, false));
        assertThrows(IllegalArgumentException.class, () -> LayoutDescriptor.contiguous(dynamic));
        assertThrows(
                IllegalArgumentException.class,
                () -> LayoutDescriptor.of(dynamic, new long[] {3, 1}, 0, false));
        assertThrows(
                IllegalArgumentException.class,
                () -> LayoutDescriptor.of(Shape.of(2, 3), new long[] {1}, 0, false));
        assertThrows(
                IllegalArgumentException.class,
                () -> LayoutDescriptor.of(Shape.of(2, 3), new long[] {-1, 1}, 0, true));
        assertThrows(
                IllegalArgumentException.class,
                () -> LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, -1, true));
        assertThrows(
                IllegalArgumentException.class,
                () -> LayoutDescriptor.of(Shape.of(2, 3), new long[] {0, 1}, 0, false));
    }

    @Test
    void detectsCanonicalAndReferencedSpanOverflow() {
        assertThrows(
                ArithmeticException.class,
                () -> LayoutDescriptor.contiguous(Shape.of(1, Long.MAX_VALUE, 2)));
        assertThrows(
                ArithmeticException.class,
                () -> LayoutDescriptor.of(Shape.scalar(), new long[0], Long.MAX_VALUE, false));
        assertThrows(
                ArithmeticException.class,
                () -> LayoutDescriptor.of(Shape.of(Long.MAX_VALUE), new long[] {2}, 0, true));
    }

    @Test
    void defensivelyCopiesInputAndOutputArrays() {
        long[] supplied = {3, 1};
        LayoutDescriptor descriptor = LayoutDescriptor.of(Shape.of(2, 3), supplied, 0, false);
        supplied[0] = 99;

        long[] returned = descriptor.strides();
        returned[1] = 99;

        assertArrayEquals(new long[] {3, 1}, descriptor.strides());
    }

    @Test
    void equalityHashingAndDiagnosticTextAreStructural() {
        LayoutDescriptor first = LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 2, true);
        LayoutDescriptor equal = LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 2, true);
        LayoutDescriptor differentView = LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 2, false);
        LayoutDescriptor differentGeometry = LayoutDescriptor.of(Shape.of(2, 3), new long[] {1, 2}, 2, true);

        assertEquals(first, equal);
        assertEquals(first.hashCode(), equal.hashCode());
        assertNotEquals(first, differentView);
        assertNotEquals(first, differentGeometry);
        assertNotEquals(first, null);

        String text = first.toString();
        assertTrue(text.contains("kind=DENSE_WITH_OFFSET"));
        assertTrue(text.contains("rank=2"));
        assertTrue(text.contains("strides=[3, 1]"));
        assertTrue(text.contains("storageOffset=2"));
        assertTrue(text.contains("view=true"));
        assertTrue(text.contains("referencedElementSpan=8"));
    }
}
