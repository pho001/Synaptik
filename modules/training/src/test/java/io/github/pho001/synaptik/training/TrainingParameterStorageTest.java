package io.github.pho001.synaptik.training;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.training.TestParameterSource.Binding;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Optional;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

final class TrainingParameterStorageTest {
    @Test
    void completeSourceValidationPrecedesParameterAllocationAndArenaProbeKeepsPrecedence() {
        Pair source = heapPair();
        var old = source.first.value().hostStorage().orElseThrow();
        source.second.value().clearHostStorage();
        try (Arena delegate = Arena.ofShared()) {
            CountingArena arena = new CountingArena(delegate, 0);
            assertThrows(IllegalStateException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(source, arena));
            assertEquals(1, arena.probes);
            assertEquals(0, arena.parameterAllocations);
            assertSame(old, source.first.value().hostStorage().orElseThrow());
        }
        try (Arena confined = Arena.ofConfined()) {
            // Destination sharing wins absent source storage, as before the generic cutover.
            assertEquals("destination Arena must be shareable", assertThrows(IllegalArgumentException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(source, confined)).getMessage());
        }
    }

    @Test
    void laterAllocationFailureInstallsNothingAndDoesNotCloseCallerArena() {
        Pair source = heapPair();
        var first = source.first.value().hostStorage().orElseThrow();
        var second = source.second.value().hostStorage().orElseThrow();
        try (Arena delegate = Arena.ofShared()) {
            CountingArena arena = new CountingArena(delegate, 2);
            assertThrows(IllegalStateException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(source, arena));
            assertEquals(1, arena.probes);
            assertEquals(2, arena.parameterAllocations);
            assertSame(first, source.first.value().hostStorage().orElseThrow());
            assertSame(second, source.second.value().hostStorage().orElseThrow());
            assertTrue(delegate.scope().isAlive());
        }
    }

    @Test
    void livePreinstallationFailureInstallsNothing() {
        Pair source = heapPair();
        var first = source.first.value().hostStorage().orElseThrow();
        var second = source.second.value().hostStorage().orElseThrow();
        TrainingParameters guarded = new TrainingParameters() {
            int checks;
            @Override public List<Entry> entries() { return source.entries(); }
            @Override public void validateUnchanged() {
                if (++checks == 2) throw new IllegalStateException("late owner failure");
                source.validateUnchanged();
            }
        };
        try (Arena arena = Arena.ofShared()) {
            assertEquals("late owner failure", assertThrows(IllegalStateException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(guarded, arena)).getMessage());
            assertSame(first, source.first.value().hostStorage().orElseThrow());
            assertSame(second, source.second.value().hostStorage().orElseThrow());
        }
    }

    @Test
    void laterCopyFailureInstallsNothing() {
        Arena sourceArena = Arena.ofShared();
        try (Arena destination = Arena.ofShared()) {
            Tensor firstTensor = tensor(new MemorySegmentStorage(DataType.FLOAT32, 1,
                    sourceArena.allocate(Float.BYTES, Float.BYTES)), 1);
            Tensor secondTensor = tensor(new MemorySegmentStorage(DataType.FLOAT32, 1,
                    sourceArena.allocate(Float.BYTES, Float.BYTES)), 1);
            Pair source = new Pair(firstTensor, secondTensor);
            var first = firstTensor.hostStorage().orElseThrow();
            var second = secondTensor.hostStorage().orElseThrow();
            Arena failing = new Arena() {
                int allocations;
                @Override public MemorySegment.Scope scope() { return destination.scope(); }
                @Override public void close() { throw new AssertionError("caller Arena closed"); }
                @Override public MemorySegment allocate(long size, long alignment) {
                    MemorySegment allocated = destination.allocate(size, alignment);
                    if (size > 0 && ++allocations == 2) sourceArena.close();
                    return allocated;
                }
            };
            assertThrows(IllegalStateException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(source, failing));
            assertSame(first, firstTensor.hostStorage().orElseThrow());
            assertSame(second, secondTensor.hostStorage().orElseThrow());
            assertTrue(destination.scope().isAlive());
        } finally {
            if (sourceArena.scope().isAlive()) sourceArena.close();
        }
    }

    @Test
    void laterInstallFailureRestoresExactAssociationsInReverseOrderAndSuppressesRollbackFailure() {
        for (boolean rollbackFails : List.of(false, true)) {
            Pair source = heapPair();
            Tensor firstTensor = source.first.value();
            Tensor secondTensor = source.second.value();
            var first = firstTensor.hostStorage().orElseThrow();
            var second = secondTensor.hostStorage().orElseThrow();
            List<String> events = new ArrayList<>();
            IllegalStateException failure = new IllegalStateException("later installation");
            IllegalStateException rollback = new IllegalStateException("rollback failure");
            try (Arena arena = Arena.ofShared()) {
                assertSame(failure, assertThrows(IllegalStateException.class,
                        () -> TrainingParameterStorage.copyToSharedNative(source, arena, (tensor, storage) -> {
                            if (tensor == secondTensor) {
                                events.add("second install");
                                throw failure;
                            }
                            if (storage == first) {
                                events.add("first rollback");
                                tensor.replaceHostStorage(storage);
                                if (rollbackFails) throw rollback;
                            } else {
                                events.add("first install");
                                tensor.replaceHostStorage(storage);
                            }
                        })));
                assertEquals(List.of("first install", "second install", "first rollback"), events);
                assertSame(first, firstTensor.hostStorage().orElseThrow());
                assertSame(second, secondTensor.hostStorage().orElseThrow());
                assertArrayEquals(rollbackFails ? new Throwable[] {rollback} : new Throwable[0],
                        failure.getSuppressed());
                assertEquals(0x3f800000, first.segment().get(ValueLayout.JAVA_INT_UNALIGNED, 0));
                assertEquals(0x40000000, second.segment().get(ValueLayout.JAVA_INT_UNALIGNED, 0));
                assertTrue(arena.scope().isAlive());
                source.validateUnchanged();
            }
        }
    }

    private static Pair heapPair() {
        return new Pair(tensor(new MemorySegmentStorage(DataType.FLOAT32, 1,
                MemorySegment.ofArray(bytes(0x3f800000))), 1),
                tensor(new MemorySegmentStorage(DataType.FLOAT32, 1,
                        MemorySegment.ofArray(bytes(0x40000000))), 1));
    }

    /** Counts the zero-byte probe separately from real destination allocations. */
    private static final class CountingArena implements Arena {
        private final Arena delegate;
        private final int failAt;
        private int probes;
        private int parameterAllocations;

        private CountingArena(Arena delegate, int failAt) {
            this.delegate = delegate;
            this.failAt = failAt;
        }

        @Override public MemorySegment allocate(long size, long alignment) {
            if (size == 0) probes++;
            else if (++parameterAllocations == failAt) throw new IllegalStateException("allocation failure");
            return delegate.allocate(size, alignment);
        }

        @Override public MemorySegment.Scope scope() { return delegate.scope(); }
        @Override public void close() { throw new AssertionError("copy must not close caller Arena"); }
    }

    @Test
    void rejectsEmptyParameterDiscovery() {
        TestParameterSource empty = new TestParameterSource() { };
        try (Arena arena = Arena.ofShared()) {
            assertThrows(IllegalStateException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(empty, arena));
        }
    }

    @Test
    void copiesExactRepresentedBitsAndKeepsEveryIdentityAndSource() {
        byte[] firstBytes = bytes(0x80000000, 0x3f800001);
        byte[] secondBytes = bytes(0x40400000);
        Tensor first = tensor(new MemorySegmentStorage(
                DataType.FLOAT32, 2, MemorySegment.ofArray(firstBytes)), 2);
        Tensor second = tensor(new MemorySegmentStorage(
                DataType.FLOAT32, 1, MemorySegment.ofArray(secondBytes).asReadOnly()), 1);
        Pair module = new Pair(first, second);
        Binding firstWrapper = module.first;
        Binding secondWrapper = module.second;
        HostTensorStorage oldFirst = first.hostStorage().orElseThrow();
        HostTensorStorage oldSecond = second.hostStorage().orElseThrow();

        try (Arena arena = Arena.ofShared()) {
            TrainingParameterStorage.copyToSharedNative(module, arena);
            assertSame(firstWrapper, module.first);
            assertSame(secondWrapper, module.second);
            assertSame(first, firstWrapper.value());
            assertSame(second, secondWrapper.value());
            assertArrayEquals(firstBytes, oldFirst.segment().toArray(ValueLayout.JAVA_BYTE));
            assertArrayEquals(secondBytes, oldSecond.segment().toArray(ValueLayout.JAVA_BYTE));
            assertCopied(first, firstBytes, 2);
            assertCopied(second, secondBytes, 1);
            assertTrue(oldFirst.isAlive());
            assertTrue(oldSecond.isAlive());
        }
        assertFalse(first.hostStorage().orElseThrow().isAlive());
    }

    @Test
    void zeroByteArenaProbeDistinguishesSharedAndConfinedScopes() {
        Thread other = Thread.ofVirtual().unstarted(() -> { });
        try (Arena shared = Arena.ofShared(); Arena confined = Arena.ofConfined()) {
            assertTrue(shared.allocate(0, 1).isAccessibleBy(other));
            assertFalse(confined.allocate(0, 1).isAccessibleBy(other));
        }
        Arena dead = Arena.ofShared();
        dead.close();
        assertFalse(dead.scope().isAlive());
        assertThrows(IllegalStateException.class, () -> dead.allocate(0, 1));
    }

    @Test
    void copiesFiniteRawBitsAtEverySupportedWidth() {
        assertWidth(DataType.FLOAT64,
                ByteBuffer.allocate(8).order(ByteOrder.nativeOrder())
                        .putLong(0x8000000000000000L).array());
        assertWidth(DataType.FLOAT32, bytes(0x80000000));
        assertWidth(DataType.BFLOAT16,
                ByteBuffer.allocate(4).order(ByteOrder.nativeOrder())
                        .putShort((short) 0x8000).putShort((short) 0x0001).array(), 2);
        assertWidth(DataType.FLOAT16,
                ByteBuffer.allocate(4).order(ByteOrder.nativeOrder())
                        .putShort((short) 0x8000).putShort((short) 0x0001).array(), 2);
    }

    @Test
    void copiesExistingReadOnlyNativeSourceWithoutClosingItsArena() {
        try (Arena sourceArena = Arena.ofShared(); Arena destinationArena = Arena.ofShared()) {
            MemorySegment source = sourceArena.allocate(Float.BYTES, Float.BYTES);
            source.set(ValueLayout.JAVA_INT, 0, 0x80000000);
            Tensor value = tensor(new MemorySegmentStorage(
                    DataType.FLOAT32, 1, source.asReadOnly()), 1);
            HostTensorStorage original = value.hostStorage().orElseThrow();
            TestParameterSource module = new TestParameterSource() {
                { parameter("value", value); }
            };
            TrainingParameterStorage.copyToSharedNative(module, destinationArena);
            assertEquals(0x80000000,
                    value.hostStorage().orElseThrow().segment().get(ValueLayout.JAVA_INT, 0));
            assertTrue(original.isReadOnly());
            assertTrue(original.isAlive());
            assertTrue(destinationArena.scope().isAlive());
        }
    }

    @Test
    void copiesUnalignedNativeSourceIntoAlignedDestination() {
        try (Arena sourceArena = Arena.ofShared(); Arena destinationArena = Arena.ofShared()) {
            MemorySegment unaligned = sourceArena.allocate(5, 1).asSlice(1, Float.BYTES);
            unaligned.set(ValueLayout.JAVA_INT_UNALIGNED, 0, 0x3f800001);
            Tensor value = tensor(new MemorySegmentStorage(DataType.FLOAT32, 1, unaligned), 1);
            TestParameterSource module = new TestParameterSource() {
                { parameter("value", value); }
            };
            TrainingParameterStorage.copyToSharedNative(module, destinationArena);
            MemorySegment destination = value.hostStorage().orElseThrow().segment();
            assertEquals(0, Math.floorMod(destination.address(), Float.BYTES));
            assertEquals(0x3f800001, destination.get(ValueLayout.JAVA_INT, 0));
            assertEquals(0x3f800001, unaligned.get(ValueLayout.JAVA_INT_UNALIGNED, 0));
        }
    }

    @Test
    void badSecondParameterLeavesBothAssociationsAndBytesUntouched() {
        byte[] firstBytes = bytes(0x80000000);
        byte[] invalid = bytes(0x7f800000);
        Pair module = new Pair(
                tensor(new MemorySegmentStorage(
                        DataType.FLOAT32, 1, MemorySegment.ofArray(firstBytes)), 1),
                tensor(new MemorySegmentStorage(
                        DataType.FLOAT32, 1, MemorySegment.ofArray(invalid)), 1));
        HostTensorStorage first = module.first.value().hostStorage().orElseThrow();
        HostTensorStorage second = module.second.value().hostStorage().orElseThrow();
        try (Arena arena = Arena.ofShared()) {
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(module, arena));
        }
        assertSame(first, module.first.value().hostStorage().orElseThrow());
        assertSame(second, module.second.value().hostStorage().orElseThrow());
        assertArrayEquals(bytes(0x80000000), firstBytes);
        assertArrayEquals(bytes(0x7f800000), invalid);
    }

    @Test
    void duplicateTensorAndOverlappingSourcesAreRejected() {
        Tensor value = tensor(new MemorySegmentStorage(
                DataType.FLOAT32, 1, MemorySegment.ofArray(bytes(0x3f800000))), 1);
        Pair duplicate = new Pair(value, value);
        try (Arena arena = Arena.ofShared()) {
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(duplicate, arena));
        }
        HostTensorStorage source = value.hostStorage().orElseThrow();
        Pair overlap = new Pair(value, tensor(source, 1));
        try (Arena arena = Arena.ofShared()) {
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(overlap, arena));
        }
        assertSame(source, value.hostStorage().orElseThrow());
        assertSame(source, overlap.second.value().hostStorage().orElseThrow());
    }

    @Test
    void distinctPartiallyOverlappingSlicesLeaveBothAssociationsAndBytesUntouched() {
        try (Arena sourceArena = Arena.ofShared(); Arena destinationArena = Arena.ofShared()) {
            MemorySegment backing = sourceArena.allocate(3L * Float.BYTES, Float.BYTES);
            backing.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 1.0f);
            backing.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 2.0f);
            backing.setAtIndex(ValueLayout.JAVA_FLOAT, 2, 3.0f);
            HostTensorStorage firstSource = new MemorySegmentStorage(
                    DataType.FLOAT32, 2, backing.asSlice(0, 2L * Float.BYTES));
            HostTensorStorage secondSource = new MemorySegmentStorage(
                    DataType.FLOAT32, 2, backing.asSlice(Float.BYTES, 2L * Float.BYTES));
            assertNotSame(firstSource, secondSource);
            assertNotSame(firstSource.segment(), secondSource.segment());
            Pair module = new Pair(tensor(firstSource, 2), tensor(secondSource, 2));
            byte[] firstBytes = firstSource.segment().toArray(ValueLayout.JAVA_BYTE);
            byte[] secondBytes = secondSource.segment().toArray(ValueLayout.JAVA_BYTE);

            assertThrows(IllegalArgumentException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(module, destinationArena));

            assertSame(firstSource, module.first.value().hostStorage().orElseThrow());
            assertSame(secondSource, module.second.value().hostStorage().orElseThrow());
            assertArrayEquals(firstBytes, firstSource.segment().toArray(ValueLayout.JAVA_BYTE));
            assertArrayEquals(secondBytes, secondSource.segment().toArray(ValueLayout.JAVA_BYTE));
        }
    }

    @Test
    void foreignThreadConfinedDestinationPropagatesWrongThreadBeforeRebinding() throws Exception {
        Pair module = new Pair(
                tensor(new MemorySegmentStorage(
                        DataType.FLOAT32, 1, MemorySegment.ofArray(bytes(0x3f800000))), 1),
                tensor(new MemorySegmentStorage(
                        DataType.FLOAT32, 1, MemorySegment.ofArray(bytes(0x40000000))), 1));
        HostTensorStorage firstSource = module.first.value().hostStorage().orElseThrow();
        HostTensorStorage secondSource = module.second.value().hostStorage().orElseThrow();
        try (var owner = Executors.newSingleThreadExecutor()) {
            Arena foreignArena = owner.submit(Arena::ofConfined).get();
            try {
                assertThrows(WrongThreadException.class,
                        () -> TrainingParameterStorage.copyToSharedNative(module, foreignArena));
                assertSame(firstSource, module.first.value().hostStorage().orElseThrow());
                assertSame(secondSource, module.second.value().hostStorage().orElseThrow());
            } finally {
                owner.submit(foreignArena::close).get();
            }
        }
    }

    @Test
    void deadOrConfinedDestinationAndMissingOrDeadSourceFailBeforeRebinding() {
        Tensor value = tensor(new MemorySegmentStorage(
                DataType.FLOAT32, 1, MemorySegment.ofArray(bytes(0x3f800000))), 1);
        Pair module = new Pair(value, tensor(new MemorySegmentStorage(
                DataType.FLOAT32, 1, MemorySegment.ofArray(bytes(0x40000000))), 1));
        HostTensorStorage original = value.hostStorage().orElseThrow();
        try (Arena confined = Arena.ofConfined()) {
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(module, confined));
        }
        Arena dead = Arena.ofShared();
        dead.close();
        assertThrows(IllegalStateException.class,
                () -> TrainingParameterStorage.copyToSharedNative(module, dead));
        assertSame(original, value.hostStorage().orElseThrow());

        module.second.value().clearHostStorage();
        try (Arena arena = Arena.ofShared()) {
            assertThrows(IllegalStateException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(module, arena));
        }
        assertSame(original, value.hostStorage().orElseThrow());

        Arena sourceArena = Arena.ofShared();
        MemorySegmentStorage nativeSource = new MemorySegmentStorage(
                DataType.FLOAT32, 1, sourceArena.allocate(Float.BYTES, Float.BYTES));
        nativeSource.segment().set(ValueLayout.JAVA_FLOAT, 0, 2f);
        module.second.value().replaceHostStorage(nativeSource);
        sourceArena.close();
        try (Arena arena = Arena.ofShared()) {
            assertThrows(IllegalStateException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(module, arena));
        }
        assertSame(original, value.hostStorage().orElseThrow());
    }

    private static void assertCopied(Tensor tensor, byte[] expected, long count) {
        HostTensorStorage storage = tensor.hostStorage().orElseThrow();
        assertEquals(count, storage.elementCapacity());
        assertEquals(expected.length, storage.byteSize());
        assertFalse(storage.isReadOnly());
        assertTrue(storage.segment().heapBase().isEmpty());
        assertArrayEquals(expected, storage.segment().toArray(ValueLayout.JAVA_BYTE));
    }

    private static void assertWidth(DataType type, byte[] raw) {
        assertWidth(type, raw, 1);
    }

    private static void assertWidth(DataType type, byte[] raw, long count) {
        Shape shape = Shape.of(count);
        Tensor value = TensorFactory.create(new TensorDescriptor(
                        type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), true),
                Optional.empty(), Optional.of(new MemorySegmentStorage(
                        type, count, MemorySegment.ofArray(raw))));
        TestParameterSource module = new TestParameterSource() {
            { parameter("value", value); }
        };
        try (Arena arena = Arena.ofShared()) {
            TrainingParameterStorage.copyToSharedNative(module, arena);
            assertCopied(value, raw, count);
        }
    }

    private static Tensor tensor(HostTensorStorage storage, long count) {
        Shape shape = Shape.of(count);
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.contiguous(shape)), true),
                Optional.empty(), Optional.of(storage));
    }

    private static byte[] bytes(int... bits) {
        ByteBuffer result = ByteBuffer.allocate(bits.length * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int value : bits) {
            result.putInt(value);
        }
        return result.array();
    }

    private static final class Pair extends TestParameterSource {
        private final Binding first;
        private final Binding second;

        private Pair(Tensor first, Tensor second) {
            this.first = parameter("first", first);
            this.second = parameter("second", second);
        }
    }
}
