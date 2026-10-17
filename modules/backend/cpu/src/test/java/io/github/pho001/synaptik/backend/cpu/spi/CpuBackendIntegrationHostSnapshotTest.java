package io.github.pho001.synaptik.backend.cpu.spi;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.lang.foreign.MemorySegment;
import java.util.Optional;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

/** Exercises canonical snapshots using only supported CPU, Model, and Runtime types. */
final class CpuBackendIntegrationHostSnapshotTest {
    @Test
    void copiesFreshDetachedBytesWithoutChangingBorrowedOwnership() {
        float[] source = {1.0f, -0.0f};
        TensorDescriptor descriptor = descriptor(DataType.FLOAT32, Shape.of(2));
        try (CpuBackendIntegration integration = CpuBackendIntegration.open()) {
            BufferRepresentation representation = integration.borrow(new MemorySegmentStorage(
                    DataType.FLOAT32, source.length, MemorySegment.ofArray(source)));
            byte[] first = integration.copyToCanonicalHostBytes(representation, descriptor, 8);
            byte[] second = integration.copyToCanonicalHostBytes(representation, descriptor, 8);
            assertArrayEquals(new byte[] {0x3f, (byte) 0x80, 0, 0, (byte) 0x80, 0, 0, 0}, first);
            assertNotSame(first, second);
            first[0] = 0;
            assertEquals(1.0f, source[0]);
            assertEquals((byte) 0x3f, second[0]);
            representation.close();
            assertArrayEquals(new byte[] {0x3f, (byte) 0x80, 0, 0, (byte) 0x80, 0, 0, 0}, second);
        }
    }
    @Test
    void borrowsAndPublishesRawFloat16ShortsWithoutConvertingThem() {
        short[] source = {(short) 0x0000, (short) 0x7e01, (short) 0x8000,
                (short) 0xfc01, (short) 0x3c00, (short) 0x3555, (short) 0x0001,
                (short) 0xffff};
        Shape shape = Shape.of(2, 2);
        TensorDescriptor descriptor = new TensorDescriptor(DataType.FLOAT16, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {3, 2}, 1, true)), false);
        try (CpuBackendIntegration integration = CpuBackendIntegration.open()) {
            BufferRepresentation representation = integration.borrow(new MemorySegmentStorage(
                    DataType.FLOAT16, source.length, MemorySegment.ofArray(source)));
            try {
                assertArrayEquals(new byte[] {0x7e, 0x01, (byte) 0xfc, 0x01,
                                0x3c, 0x00, 0x00, 0x01},
                        integration.copyToCanonicalHostBytes(representation, descriptor, 8));
                TensorDescriptor bfloat = new TensorDescriptor(DataType.BFLOAT16, shape,
                        Optional.of(LayoutDescriptor.of(shape, new long[] {3, 2}, 1, true)),
                        false);
                assertThrows(IllegalArgumentException.class, () ->
                        integration.copyToCanonicalHostBytes(representation, bfloat, 8));
            } finally {
                representation.close();
            }
        }
    }


    @Test
    void preservesAdapterClosedPrecedenceAndSupportsIndependentConcurrentCalls() throws Exception {
        TensorDescriptor descriptor = descriptor(DataType.INT32, Shape.of(1));
        CpuBackendIntegration closed = CpuBackendIntegration.open();
        closed.close();
        IllegalStateException failure = assertThrows(IllegalStateException.class,
                () -> closed.copyToCanonicalHostBytes(null, null, -1));
        assertEquals("CPU backend integration is closed", failure.getMessage());

        try (CpuBackendIntegration integration = CpuBackendIntegration.open();
                var executor = Executors.newFixedThreadPool(2)) {
            var first = integration.borrow(new MemorySegmentStorage(DataType.INT32, 1,
                    MemorySegment.ofArray(new int[] {0x01020304})));
            var second = integration.borrow(new MemorySegmentStorage(DataType.INT32, 1,
                    MemorySegment.ofArray(new int[] {0x05060708})));
            var firstCopy = executor.submit(() ->
                    integration.copyToCanonicalHostBytes(first, descriptor, 4));
            var secondCopy = executor.submit(() ->
                    integration.copyToCanonicalHostBytes(second, descriptor, 4));
            assertArrayEquals(new byte[] {1, 2, 3, 4}, firstCopy.get());
            assertArrayEquals(new byte[] {5, 6, 7, 8}, secondCopy.get());
        }
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
    }
}
