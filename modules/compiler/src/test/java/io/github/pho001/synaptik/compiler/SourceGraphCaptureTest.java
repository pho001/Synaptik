package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class SourceGraphCaptureTest {
    @Test
    void alignsBothLeafInsertionSitesInTheSameTraversalWithoutMergingOccurrences() {
        Tensor first = leaf(DataType.FLOAT32, Shape.of(2), true);
        Tensor second = leaf(DataType.FLOAT32, Shape.of(2), true);
        Tensor rootLeaf = leaf(DataType.FLOAT32, Shape.of(2), true);
        Tensor shared = second.add(first);
        Tensor equalButDistinct = second.add(first);
        Tensor repeated = shared.mul(shared);
        var requested = new ArrayList<>(List.of(repeated, rootLeaf, equalButDistinct, first, shared));

        var source = GraphCompilationPort.captureSource(requested);
        assertEquals(GraphCapture.capture(requested), source.graph());
        requested.clear();

        assertEquals(List.of(second, first, rootLeaf), source.leaves());
        assertEquals(List.of(new ValueId(0), new ValueId(1), new ValueId(4)), source.graph().inputs());
        assertEquals(List.of(new ValueId(3), new ValueId(4), new ValueId(5), new ValueId(1), new ValueId(2)),
                source.graph().outputs());
        assertEquals(3, source.graph().nodes().size());
        assertEquals(List.of(new ValueId(2), new ValueId(2)), source.graph().nodes().get(1).inputs());
        assertSame(shared.descriptor(), source.graph().values().get(2).descriptor());
        assertSame(shared.provenance().orElseThrow().operation(), source.graph().nodes().getFirst().operation());
        assertThrows(UnsupportedOperationException.class, () -> source.leaves().clear());
        for (int index = 0; index < source.leaves().size(); index++) {
            int position = index;
            assertSame(source.leaves().get(index).descriptor(), source.graph().values().stream()
                    .filter(value -> value.id().equals(source.graph().inputs().get(position)))
                    .findFirst().orElseThrow().descriptor());
        }
    }

    @Test
    void snapshotsDirectConstructionAndRequiresExactUniqueProvenanceFreeLeaves() {
        Tensor x = leaf(DataType.FLOAT32, Shape.scalar(), true);
        Tensor y = leaf(DataType.FLOAT32, Shape.scalar(), true);
        var source = GraphCompilationPort.captureSource(List.of(x.add(y)));
        var leaves = new ArrayList<>(source.leaves());
        var direct = new GraphCompilationPort.SourceCapture(source.graph(), leaves);
        leaves.clear();
        assertEquals(source, direct);
        assertThrows(IllegalArgumentException.class,
                () -> new GraphCompilationPort.SourceCapture(source.graph(), List.of(x, x)));
        assertThrows(IllegalArgumentException.class,
                () -> new GraphCompilationPort.SourceCapture(source.graph(), List.of(x.neg(), y)));
        assertThrows(IllegalArgumentException.class,
                () -> new GraphCompilationPort.SourceCapture(source.graph(), List.of(x)));
        assertThrows(NullPointerException.class,
                () -> new GraphCompilationPort.SourceCapture(null, source.leaves()));
        assertThrows(NullPointerException.class,
                () -> new GraphCompilationPort.SourceCapture(source.graph(), null));
        assertThrows(NullPointerException.class,
                () -> new GraphCompilationPort.SourceCapture(source.graph(), Arrays.asList(x, null)));
    }

    @Test
    void keepsRawUnresolvedResultsAndResolvedOrUnresolvedTranspose() {
        Shape matrix = Shape.of(2, 3);
        for (Optional<LayoutDescriptor> layout : List.of(Optional.<LayoutDescriptor>empty(),
                Optional.of(LayoutDescriptor.of(matrix, new long[] {0, 2}, 5, true)))) {
            Tensor x = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, matrix, layout, true));
            Tensor transpose = x.permute(1, 0);
            Tensor result = transpose.neg();
            var source = GraphCompilationPort.captureSource(List.of(result, transpose));
            assertSame(result.descriptor(), source.graph().values().getLast().descriptor());
            assertTrue(result.descriptor().layout().isEmpty());
            assertEquals(layout.isPresent(), transpose.descriptor().layout().isPresent());
            var rebuilt = GraphCompilationPort.reconstructSource(source.graph(), source.leaves());
            assertEquals(result.descriptor(), rebuilt.getFirst().descriptor());
            assertEquals(transpose.descriptor(), rebuilt.get(1).descriptor());
            assertSame(rebuilt.get(1), rebuilt.getFirst().provenance().orElseThrow().inputs().getFirst());
        }
    }

    @Test
    void rejectsInvalidBoundariesAndUnsupportedOrSymbolicSourceButRawCaptureStaysStructural() {
        Tensor x = leaf(DataType.FLOAT32, Shape.scalar(), true);
        assertThrows(NullPointerException.class, () -> GraphCompilationPort.captureSource(null));
        assertThrows(NullPointerException.class,
                () -> GraphCompilationPort.captureSource(Arrays.asList(x, null)));
        assertThrows(IllegalArgumentException.class, () -> GraphCompilationPort.captureSource(List.of()));
        assertThrows(IllegalArgumentException.class, () -> GraphCompilationPort.captureSource(List.of(x, x)));
        for (Tensor unsupported : List.of(x.abs(), x.sum(),
                leaf(DataType.FLOAT32, Shape.of(2, 3), false).permute(0, 1),
                leaf(DataType.FLOAT32, Shape.of(2, 3, 4), false).permute(2, 1, 0))) {
            assertDoesNotThrow(() -> GraphCapture.capture(List.of(unsupported)));
            assertThrows(IllegalArgumentException.class,
                    () -> GraphCompilationPort.captureSource(List.of(unsupported)));
        }
        Tensor symbolic = leaf(DataType.FLOAT32, Shape.ofDimensions(new DynamicDimension("N")), false);
        assertDoesNotThrow(() -> GraphCapture.capture(List.of(symbolic)));
        assertThrows(IllegalArgumentException.class,
                () -> GraphCompilationPort.captureSource(List.of(symbolic)));
        Tensor overflow = leaf(DataType.FLOAT32, Shape.of(Long.MAX_VALUE, 2), false);
        var failure = assertThrows(IllegalArgumentException.class,
                () -> GraphCompilationPort.captureSource(List.of(overflow)));
        assertInstanceOf(ArithmeticException.class, failure.getCause());
    }

    @Test
    void neverRequiresLiveReadableStorageAndDoesNotRebindOrMutateOriginals() {
        Tensor x;
        MemorySegmentStorage storage;
        try (Arena arena = Arena.ofConfined()) {
            storage = new MemorySegmentStorage(DataType.FLOAT32, 1, arena.allocate(4));
            x = TensorFactory.create(new TensorDescriptor(
                    DataType.FLOAT32, Shape.scalar(), Optional.empty(), true), Optional.empty(), Optional.of(storage));
        }
        assertFalse(storage.isAlive());
        Tensor expression = x.mul(x).neg();
        var source = GraphCompilationPort.captureSource(List.of(expression));
        var rebuilt = GraphCompilationPort.reconstructSource(source.graph(), source.leaves());
        assertSame(storage, x.hostStorage().orElseThrow());
        assertSame(x, source.leaves().getFirst());
        assertTrue(rebuilt.getFirst().hostStorage().isEmpty());
        assertTrue(expression.hostStorage().isEmpty());

        Tensor stored = TensorFactory.scalar(-0.0f, Optional.empty(), true);
        var originalStorage = stored.hostStorage().orElseThrow();
        byte[] before = originalStorage.segment().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
        var live = GraphCompilationPort.captureSource(List.of(stored.add(stored)));
        GraphCompilationPort.reconstructSource(live.graph(), live.leaves());
        assertSame(originalStorage, stored.hostStorage().orElseThrow());
        assertArrayEquals(before, originalStorage.segment().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
    }

    private static Tensor leaf(DataType type, Shape shape, boolean grad) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), grad));
    }
}
