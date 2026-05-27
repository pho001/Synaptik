package backend.cpu.kernels.index;

import backend.cpu.storage.CpuStorageView;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class IndexStorageViewLoopsTest {
    @Test
    void gatherAxisF32ReadsStridedSegmentIndicesFromStorageView() {
        Tensor input = new Tensor(new float[6], new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor out = new Tensor(new float[8], new int[]{2, 2, 2}, null, "out", DataType.FLOAT32);
        float[] inputData = new float[]{10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f};
        float[] outData = new float[8];

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment indicesSegment = arena.allocate(JAVA_INT, 6);
            writeI32(indicesSegment, 0, 2, -1, 0, 0, 1);

            GatherAxisLoops.gatherAxisF32(
                    input,
                    indices,
                    out,
                    arrayView(input, inputData),
                    segmentView(indices, indicesSegment, new int[]{3, 1}, 1),
                    arrayView(out, outData),
                    1);
        }

        assertArrayEquals(new float[]{
                12.0f, 12.0f, 10.0f, 11.0f,
                22.0f, 22.0f, 20.0f, 21.0f
        }, outData, 1.0e-6f);
    }

    @Test
    void takeAlongAxisF64RejectsFractionalSegmentFloatingIndex() {
        Tensor input = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "input", DataType.FLOAT64);
        Tensor indices = new Tensor(new float[4], new int[]{2, 2}, null, "indices", DataType.FLOAT32);
        Tensor out = new Tensor(new double[4], new int[]{2, 2}, null, "out", DataType.FLOAT64);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment indicesSegment = arena.allocate(JAVA_FLOAT, 4);
            writeF32(indicesSegment, 2.0f, 1.5f, 0.0f, 1.0f);

            assertThrows(IllegalArgumentException.class, () -> TakeAlongAxisLoops.takeAlongAxisF64(
                    input,
                    indices,
                    out,
                    arrayView(input, new double[]{1, 2, 3, 4, 5, 6}),
                    segmentView(indices, indicesSegment, indices.getStridesUnsafe(), indices.getStorageOffsetUnsafe()),
                    arrayView(out, new double[4]),
                    1));
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

    private static CpuStorageView segmentView(Tensor tensor, MemorySegment segment, int[] strides, int storageOffset) {
        return CpuStorageView.segment(
                tensor.getDataType(),
                segment,
                tensor.getShapeUnsafe(),
                strides,
                storageOffset,
                tensor.getFlatDataSize());
    }

    private static void writeI32(MemorySegment segment, int... values) {
        for (int i = 0; i < values.length; i++) {
            segment.set(JAVA_INT, (long) i * Integer.BYTES, values[i]);
        }
    }

    private static void writeF32(MemorySegment segment, float... values) {
        for (int i = 0; i < values.length; i++) {
            segment.set(JAVA_FLOAT, (long) i * Float.BYTES, values[i]);
        }
    }
}
