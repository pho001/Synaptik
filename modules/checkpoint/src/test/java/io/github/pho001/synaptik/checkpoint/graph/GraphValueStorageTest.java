package io.github.pho001.synaptik.checkpoint.graph;

import static io.github.pho001.synaptik.checkpoint.graph.GraphCheckpointTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/** Host admission and bit preservation are independent of backend or Engine capability. */
final class GraphValueStorageTest {
    @Test
    void readOnlyUnalignedNativeAndHeapSourcesCopyAllSevenCarriersWithDetachedLifetime() {
        for (DataType type : DataType.values()) {
            byte[] bits = raw(type, 3);
            GraphCheckpoint.Source copied;
            try (Arena arena = Arena.ofConfined()) {
                Tensor seed = tensor(type, Shape.of(3), false, false, bits);
                MemorySegment unaligned = arena.allocate(bits.length + 1, 1).asSlice(1, bits.length);
                unaligned.copyFrom(seed.hostStorage().orElseThrow().segment());
                Tensor nativeState = prototype(type, Shape.of(3), false, false);
                nativeState.replaceHostStorage(new MemorySegmentStorage(type, 3, unaligned.asReadOnly()));
                copied = GraphCheckpoint.copy(source(nativeState));
                assertArrayEquals(bits, GraphValueStorage.encode(copied.outputs().getFirst().descriptor(),
                        copied.outputs().getFirst().hostStorage().orElseThrow().segment()));
                Tensor heap = prototype(type, Shape.of(3), false, false);
                heap.replaceHostStorage(new MemorySegmentStorage(type, 3,
                        seed.hostStorage().orElseThrow().segment().asReadOnly()));
                assertArrayEquals(bits, GraphValueStorage.encode(GraphCheckpoint.copy(source(heap)).outputs().getFirst().descriptor(),
                        GraphCheckpoint.copy(source(heap)).outputs().getFirst().hostStorage().orElseThrow().segment()));
            }
            Tensor result = copied.outputs().getFirst();
            assertTrue(result.hostStorage().orElseThrow().isAlive());
            assertArrayEquals(bits, GraphValueStorage.encode(result.descriptor(), result.hostStorage().orElseThrow().segment()));
            Class<?> carrier = switch (type) {
                case FLOAT64 -> double[].class; case FLOAT32 -> float[].class;
                case INT64 -> long[].class; case INT32 -> int[].class;
                case BFLOAT16, FLOAT16 -> short[].class; case BOOL -> byte[].class;
            };
            assertEquals(carrier, result.hostStorage().orElseThrow().segment().heapBase().orElseThrow().getClass());
        }
    }

    @Test
    void rejectsAbsentDeadAndInaccessibleStorageEvenForEmptyValues() throws Exception {
        for (Shape shape : List.of(Shape.of(1), Shape.of(0))) {
            Tensor missing = prototype(DataType.FLOAT32, shape, false, false);
            assertThrows(IllegalStateException.class, () -> GraphCheckpoint.copy(source(missing)));
            Tensor state = prototype(DataType.FLOAT32, shape, false, false);
            Arena arena = Arena.ofConfined();
            long count = shape.knownElementCount().orElseThrow();
            state.replaceHostStorage(new MemorySegmentStorage(DataType.FLOAT32, count, arena.allocate(count * 4, 1)));
            AtomicReference<Throwable> failure = new AtomicReference<>();
            Thread thread = Thread.ofPlatform().start(() -> {
                try { GraphCheckpoint.copy(source(state)); }
                catch (Throwable caught) { failure.set(caught); }
            });
            thread.join();
            assertInstanceOf(IllegalStateException.class, failure.get());
            arena.close();
            assertThrows(IllegalStateException.class, () -> GraphCheckpoint.copy(source(state)));
        }
    }

    @Test
    void admissionOfEveryIdentityPrecedesBoolValidationAndCopy() {
        Tensor invalidBool = tensor(DataType.BOOL, Shape.of(1), false, false, new byte[] {2});
        Tensor missing = prototype(DataType.BOOL, Shape.of(1), false, false);
        var source = new GraphCheckpoint.Source(List.of(), List.of(invalidBool, missing), List.of(
                leaf("first", GraphCheckpoint.Role.CONSTANT, invalidBool),
                leaf("late", GraphCheckpoint.Role.CONSTANT, missing)));
        assertThrows(IllegalStateException.class, () -> GraphCheckpoint.copy(source));
        missing.replaceHostStorage(new MemorySegmentStorage(DataType.BOOL, 1, MemorySegment.ofArray(new byte[1])));
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.copy(source));
        assertEquals(2, invalidBool.hostStorage().orElseThrow().segment().get(ValueLayout.JAVA_BYTE, 0));
    }

    @Test
    void shortStorageNoncanonicalLayoutAndCheckedUnresolvedGeometryFail() {
        Tensor shortValue = prototype(DataType.FLOAT32, Shape.of(2), false, false);
        shortValue.replaceHostStorage(new MemorySegmentStorage(DataType.FLOAT32, 1, MemorySegment.ofArray(new float[1])));
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.copy(source(shortValue)));
        Shape shape = Shape.of(2, 2);
        for (LayoutDescriptor layout : List.of(
                LayoutDescriptor.of(shape, new long[] {2, 1}, 1, false),
                LayoutDescriptor.of(shape, new long[] {2, 1}, 0, true),
                LayoutDescriptor.of(shape, new long[] {1, 2}, 0, false))) {
            Tensor bad = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape, Optional.of(layout), false));
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(source(bad), List.of()));
        }
        Tensor huge = prototype(DataType.FLOAT64, Shape.of(Long.MAX_VALUE), false, false);
        assertThrows(ArithmeticException.class, () -> GraphCheckpoint.copy(source(huge)));
        Tensor oversized = prototype(DataType.BOOL, Shape.of(Integer.MAX_VALUE), false, false);
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.copy(source(oversized)));
        // Rebind has no implicit file/JVM payload bound and performs no storage gate.
        assertSame(oversized, GraphCheckpoint.rebind(source(oversized), List.of()).getFirst());
    }

    @Test
    void overlappingDistinctIdentitiesAndExcessCapacityAreNotDeduplicatedOrPersisted() {
        MemorySegment shared = MemorySegment.ofArray(new int[] {10, 20, 30});
        Tensor a = prototype(DataType.INT32, Shape.of(2), false, false);
        Tensor b = prototype(DataType.INT32, Shape.of(2), false, false);
        a.replaceHostStorage(new MemorySegmentStorage(DataType.INT32, 3, shared));
        b.replaceHostStorage(new MemorySegmentStorage(DataType.INT32, 2, shared.asSlice(4, 8)));
        var source = new GraphCheckpoint.Source(List.of(), List.of(a, b), List.of(
                leaf("a", GraphCheckpoint.Role.CONSTANT, a), leaf("b", GraphCheckpoint.Role.BUFFER, b)));
        var copy = GraphCheckpoint.copy(source);
        Tensor ca = copy.outputs().get(0), cb = copy.outputs().get(1);
        assertNotSame(ca, cb);
        assertEquals(8, ca.hostStorage().orElseThrow().byteSize());
        assertEquals(8, cb.hostStorage().orElseThrow().byteSize());
        assertEquals(20, cb.hostStorage().orElseThrow().segment().get(ValueLayout.JAVA_INT_UNALIGNED, 0));
        ca.hostStorage().orElseThrow().segment().set(ValueLayout.JAVA_INT_UNALIGNED, 4, 99);
        assertEquals(20, shared.get(ValueLayout.JAVA_INT_UNALIGNED, 4));
        assertEquals(20, cb.hostStorage().orElseThrow().segment().get(ValueLayout.JAVA_INT_UNALIGNED, 0));
    }

    /**
     * Creates a one-state pass-through source; zero-node persistence is not execution support.
     * @param state caller-owned leaf, possibly unreadable for negative tests
     * @return borrowed source with one explicit constant registration
     */
    private static GraphCheckpoint.Source source(Tensor state) {
        return new GraphCheckpoint.Source(List.of(), List.of(state),
                List.of(leaf("state", GraphCheckpoint.Role.CONSTANT, state)));
    }
}
