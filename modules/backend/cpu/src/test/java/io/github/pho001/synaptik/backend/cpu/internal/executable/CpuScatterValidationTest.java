package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Prepared validation regressions where logical index rank differs from scratch capacity. */
class CpuScatterValidationTest {
    @Test void rankOneIndicesValidateStridedOffsetStorageBeforeRankTwoUpdates() {
        for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
            var data = desc(DataType.FLOAT32, Shape.of(2, 3));
            var indices = new TensorDescriptor(indexType, Shape.of(3), Optional.of(
                    LayoutDescriptor.of(Shape.of(3), new long[] {2}, 1, true)), false);
            var executable = executable(new Operation(AxisScatterKind.SCATTER_ADD,
                    new IndexAxisAttrs(1)), data, indices, desc(DataType.FLOAT32, Shape.of(2, 3)));
            float[] base = {10, 20, 30, 40, 50, 60}, updates = {1, 2, 3, 4, 5, 6};
            float[] output = new float[6];
            MemorySegment indexBacking = indexType == DataType.INT32
                    ? MemorySegment.ofArray(new int[8]) : MemorySegment.ofArray(new long[8]);
            // The binding slice and layout offset both contribute; padding is intentionally invalid.
            indexBacking.fill((byte) 0x7f);
            MemorySegment indexSlice = indexBacking.asSlice(indexType.byteWidth(),
                    6L * indexType.byteWidth());
            int[] logical = {2, 0, 2};
            for (int i = 0; i < 3; i++) putIndex(indexType, indexSlice, 1 + 2 * i, logical[i]);
            byte[] validIndices = indexBacking.toArray(ValueLayout.JAVA_BYTE);
            try (RunState state = state(executable, List.of(
                    borrow(DataType.FLOAT32, MemorySegment.ofArray(base)), borrow(indexType, indexSlice),
                    borrow(DataType.FLOAT32, MemorySegment.ofArray(updates)),
                    borrow(DataType.FLOAT32, MemorySegment.ofArray(output))))) {
                var bound = executable.bind(state);
                for (int repeat = 0; repeat < 2; repeat++) {
                    bound.execute();
                    assertArrayEquals(addOracle(base, logical, updates), output);
                }
                for (int bad : new int[] {-1, 3}) {
                    putIndex(indexType, indexSlice, 5, bad);
                    Arrays.fill(output, Float.intBitsToFloat(0x7fc01234));
                    byte[] sentinel = MemorySegment.ofArray(output).toArray(ValueLayout.JAVA_BYTE);
                    var failure = assertThrows(IndexOutOfBoundsException.class, bound::execute);
                    assertEquals("SCATTER_ADD index at logical position 2 for data axis 1 is out of bounds: value="
                            + bad + ", extent=3", failure.getMessage());
                    assertArrayEquals(sentinel, MemorySegment.ofArray(output).toArray(ValueLayout.JAVA_BYTE));
                    putIndex(indexType, indexSlice, 5, 2);
                    bound.execute();
                    assertArrayEquals(addOracle(base, logical, updates), output);
                }
                assertArrayEquals(validIndices, indexBacking.toArray(ValueLayout.JAVA_BYTE));
                assertArrayEquals(new float[] {10, 20, 30, 40, 50, 60}, base);
                assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6}, updates);
            }
        }
    }

    @Test void scalarAndEmptyIndicesRetainTheirAdmittedDomains() {
        for (boolean scalar : List.of(true, false)) {
            var data = desc(DataType.FLOAT32, Shape.of(2, 3));
            var executable = executable(new Operation(AxisScatterKind.SCATTER_ADD,
                    new IndexAxisAttrs(1)), data, desc(DataType.INT64, scalar ? Shape.scalar() : Shape.of(0)),
                    desc(DataType.FLOAT32, scalar ? Shape.of(2) : Shape.of(2, 0)));
            float[] base = {10, 20, 30, 40, 50, 60}, output = new float[6];
            try (RunState state = state(executable, List.of(
                    borrow(DataType.FLOAT32, MemorySegment.ofArray(base)),
                    borrow(DataType.INT64, MemorySegment.ofArray(scalar ? new long[] {1} : new long[0])),
                    borrow(DataType.FLOAT32, MemorySegment.ofArray(scalar ? new float[] {2, 5} : new float[0])),
                    borrow(DataType.FLOAT32, MemorySegment.ofArray(output))))) {
                var bound = executable.bind(state);
                for (int repeat = 0; repeat < 2; repeat++) {
                    bound.execute();
                    assertArrayEquals(scalar ? new float[] {10, 22, 30, 40, 55, 60} : base, output);
                }
            }
        }
    }

    @Test void ndBoundsAndTupleUniquenessReuseLargerUpdateScratch() {
        var data = desc(DataType.FLOAT32, Shape.of(3, 2, 2));
        var executable = executable(new Operation(ScatterNdKind.SCATTER_ND,
                new ScatterNdAttrs(0, ScatterReduction.NONE)), data,
                desc(DataType.INT64, Shape.of(2, 1)), desc(DataType.FLOAT32, Shape.of(2, 2, 2)));
        long[] indices = {2, 0};
        float[] base = new float[12], updates = {1, 2, 3, 4, 5, 6, 7, 8}, output = new float[12];
        try (RunState state = state(executable, List.of(
                borrow(DataType.FLOAT32, MemorySegment.ofArray(base)),
                borrow(DataType.INT64, MemorySegment.ofArray(indices)),
                borrow(DataType.FLOAT32, MemorySegment.ofArray(updates)),
                borrow(DataType.FLOAT32, MemorySegment.ofArray(output))))) {
            var bound = executable.bind(state);
            for (int repeat = 0; repeat < 2; repeat++) {
                bound.execute();
                assertArrayEquals(new float[] {5, 6, 7, 8, 0, 0, 0, 0, 1, 2, 3, 4}, output);
                for (int bad : new int[] {-1, 3, 2}) {
                    indices[1] = bad;
                    Arrays.fill(output, -71);
                    RuntimeException failure = bad == 2
                            ? assertThrows(IllegalArgumentException.class, bound::execute)
                            : assertThrows(IndexOutOfBoundsException.class, bound::execute);
                    assertEquals(bad == 2
                            ? "SCATTER_ND duplicate target tuple at logical tuple position 1; first addressed at logical tuple position 0"
                            : "SCATTER_ND index at logical position 1 for data axis 0 is out of bounds: value="
                                    + bad + ", extent=3", failure.getMessage());
                    for (float cell : output) assertEquals(-71, cell);
                }
                indices[1] = 0;
            }
            bound.execute();
            assertArrayEquals(new float[] {5, 6, 7, 8, 0, 0, 0, 0, 1, 2, 3, 4}, output);
        }
    }

    @Test void elementsBoundsStillPrecedeUniquenessAndResetBetweenCalls() {
        var data = desc(DataType.FLOAT32, Shape.of(2, 3));
        var executable = executable(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                new ScatterElementsAttrs(1, ScatterReduction.NONE)), data,
                desc(DataType.INT32, Shape.of(2, 2)), desc(DataType.FLOAT32, Shape.of(2, 2)));
        int[] indices = {0, 0, 1, 3};
        float[] output = {-71, -71, -71, -71, -71, -71};
        try (RunState state = state(executable, List.of(
                borrow(DataType.FLOAT32, MemorySegment.ofArray(new float[6])),
                borrow(DataType.INT32, MemorySegment.ofArray(indices)),
                borrow(DataType.FLOAT32, MemorySegment.ofArray(new float[] {1, 2, 3, 4})),
                borrow(DataType.FLOAT32, MemorySegment.ofArray(output))))) {
            var bound = executable.bind(state);
            assertEquals("SCATTER_ELEMENTS index at logical position 3 for data axis 1 is out of bounds: value=3, extent=3",
                    assertThrows(IndexOutOfBoundsException.class, bound::execute).getMessage());
            indices[3] = 2;
            assertEquals("SCATTER_ELEMENTS duplicate target at logical update position 1; first addressed at logical update position 0",
                    assertThrows(IllegalArgumentException.class, bound::execute).getMessage());
            assertArrayEquals(new float[] {-71, -71, -71, -71, -71, -71}, output);
            indices[1] = 2;
            for (int repeat = 0; repeat < 2; repeat++) {
                bound.execute();
                assertArrayEquals(new float[] {1, 0, 2, 0, 3, 4}, output);
            }
        }
    }

    private static float[] addOracle(float[] base, int[] indices, float[] updates) {
        float[] expected = base.clone();
        for (int row = 0; row < 2; row++)
            for (int i = 0; i < indices.length; i++)
                expected[row * 3 + indices[i]] += updates[row * indices.length + i];
        return expected;
    }

    private static void putIndex(DataType type, MemorySegment segment, int position, int value) {
        if (type == DataType.INT32) segment.setAtIndex(ValueLayout.JAVA_INT, position, value);
        else segment.setAtIndex(ValueLayout.JAVA_LONG, position, value);
    }

    private static TensorDescriptor desc(DataType type, Shape shape) {
        return CpuScatterLoweringTest.desc(type, shape);
    }

    private static CpuPreparedExecutable executable(Operation operation, TensorDescriptor data,
            TensorDescriptor indices, TensorDescriptor updates) {
        var base = CpuScatterLoweringTest.context(operation, List.of(0, 1, 2),
                List.of(data, indices, updates), data);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.FLOAT_ARRAY,
                                indices.dataType() == DataType.INT32 ? CarrierAccess.INT_ARRAY : CarrierAccess.LONG_ARRAY,
                                CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY)));
        return CpuPartitionFinalizerTest.finalizeExecutable(new CpuPartitionPreparer().analyze(context), Optional.empty());
    }

    private static CpuBorrowedBuffer borrow(DataType type, MemorySegment segment) {
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type, segment.byteSize() / type.byteWidth(), segment));
    }

    private static RunState state(CpuPreparedExecutable executable, List<CpuBorrowedBuffer> buffers) {
        return new RunState(executable.memoryPlan(), buffers.stream().map(buffer -> List.of(
                new BufferRepresentationBinding(buffer, RunResourceOwnership.BORROWED))).toList(), List.of());
    }
}
