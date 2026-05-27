package backend.cpu.kernels.index;

import backend.cpu.storage.CpuStorageView;
import operations.index.ScatterReduction;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ScatterStorageViewLoopsTest {
    @Test
    void scatterElementsF32UsesRuntimeSegmentBaseUpdatesAndOutput() {
        Tensor data = new Tensor(new float[]{0.0f, 0.0f, 0.0f, 0.0f}, new int[]{2, 2}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[]{1, -2}, new int[]{2, 1}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new float[]{0.0f, 0.0f}, new int[]{2, 1}, null, "updates", DataType.FLOAT32);
        Tensor out = new Tensor(new float[]{0.0f, 0.0f, 0.0f, 0.0f}, new int[]{2, 2}, null, "out", DataType.FLOAT32);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dataSegment = arena.allocate(JAVA_FLOAT, 4);
            MemorySegment updatesSegment = arena.allocate(JAVA_FLOAT, 2);
            MemorySegment outSegment = arena.allocate(JAVA_FLOAT, 4);
            writeF32(dataSegment, 10.0f, 20.0f, 30.0f, 40.0f);
            writeF32(updatesSegment, 1.0f, 2.0f);

            ScatterElementsLoops.scatterElementsF32(
                    data,
                    indices,
                    updates,
                    out,
                    segmentView(data, dataSegment),
                    arrayView(indices, new int[]{1, -2}),
                    segmentView(updates, updatesSegment),
                    segmentView(out, outSegment),
                    1,
                    ScatterReduction.ADD);

            assertArrayEquals(new float[]{10.0f, 21.0f, 32.0f, 40.0f}, readF32(outSegment, 4), 1.0e-6f);
        }
    }

    @Test
    void scatterNdI64UsesSegmentIndicesAndRuntimeBase() {
        Tensor data = new Tensor(new long[]{0L, 0L, 0L, 0L}, new int[]{2, 2}, null, "data", DataType.INT64);
        Tensor indices = new Tensor(new int[]{0, 1}, new int[]{2, 1}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new long[]{0L, 0L, 0L, 0L}, new int[]{2, 2}, null, "updates", DataType.INT64);
        Tensor out = new Tensor(new long[]{0L, 0L, 0L, 0L}, new int[]{2, 2}, null, "out", DataType.INT64);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dataSegment = arena.allocate(JAVA_LONG, 4);
            MemorySegment indicesSegment = arena.allocate(JAVA_INT, 2);
            MemorySegment updatesSegment = arena.allocate(JAVA_LONG, 4);
            MemorySegment outSegment = arena.allocate(JAVA_LONG, 4);
            writeI64(dataSegment, 5L, 1L, 2L, 8L);
            writeI32(indicesSegment, 0, 1);
            writeI64(updatesSegment, 10L, 20L, 30L, 40L);

            ScatterNdLoops.scatterNdI64(
                    data,
                    indices,
                    updates,
                    out,
                    segmentView(data, dataSegment),
                    segmentView(indices, indicesSegment),
                    segmentView(updates, updatesSegment),
                    segmentView(out, outSegment),
                    ScatterReduction.MAX,
                    0);

            assertArrayEquals(new long[]{10L, 20L, 30L, 40L}, readI64(outSegment, 4));
        }
    }

    private static CpuStorageView arrayView(Tensor tensor, Object array) {
        return CpuStorageView.array(
                tensor.getDataType(),
                array,
                tensor.getShapeUnsafe(),
                tensor.getStridesUnsafe(),
                tensor.getStorageOffsetUnsafe(),
                tensor.getFlatDataSize());
    }

    private static CpuStorageView segmentView(Tensor tensor, MemorySegment segment) {
        return CpuStorageView.segment(
                tensor.getDataType(),
                segment,
                tensor.getShapeUnsafe(),
                tensor.getStridesUnsafe(),
                tensor.getStorageOffsetUnsafe(),
                tensor.getFlatDataSize());
    }

    private static void writeF32(MemorySegment segment, float... values) {
        for (int i = 0; i < values.length; i++) {
            segment.set(JAVA_FLOAT, (long) i * Float.BYTES, values[i]);
        }
    }

    private static float[] readF32(MemorySegment segment, int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = segment.get(JAVA_FLOAT, (long) i * Float.BYTES);
        }
        return values;
    }

    private static void writeI32(MemorySegment segment, int... values) {
        for (int i = 0; i < values.length; i++) {
            segment.set(JAVA_INT, (long) i * Integer.BYTES, values[i]);
        }
    }

    private static void writeI64(MemorySegment segment, long... values) {
        for (int i = 0; i < values.length; i++) {
            segment.set(JAVA_LONG, (long) i * Long.BYTES, values[i]);
        }
    }

    private static long[] readI64(MemorySegment segment, int length) {
        long[] values = new long[length];
        for (int i = 0; i < length; i++) {
            values[i] = segment.get(JAVA_LONG, (long) i * Long.BYTES);
        }
        return values;
    }
}
