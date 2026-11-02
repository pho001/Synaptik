package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Device regressions for output-owned scatter membership, duplicates and pre-write validation. */
class MetalGeneralScatterNativeTest {
    @Test
    void gatherCotangentAcceptsMiddleAxisAndMultidimensionalIndices() {
        for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
            try (Execution e = new Execution(List.of(
                    value(DataType.FLOAT32, 2, 3, 2), value(indexType, 2, 2),
                    value(DataType.FLOAT32, 2, 2, 2, 2), value(DataType.FLOAT32, 2, 3, 2)),
                    List.of(MetalMpsGraphProgram.Node.scatterAdd(0, 1, 2, 3, 1)))) {
                e.upload(0, floats(10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10));
                e.upload(1, indices(indexType, 2, 0, 2, 1));
                e.upload(2, floats(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
                e.run();
                assertArrayEquals(floats(13, 14, 17, 18, 16, 18, 21, 22, 25, 26, 32, 34), e.output());
            }
        }
    }

    @Test
    void scalarIndicesAndRankSixteenDataKeepEveryNonSelectedCoordinate() {
        long[] shape = new long[16];
        Arrays.fill(shape, 1);
        shape[14] = 3;
        shape[15] = 2;
        long[] updateShape = new long[15];
        Arrays.fill(updateShape, 1);
        updateShape[14] = 2;
        try (Execution e = new Execution(List.of(value(DataType.FLOAT32, shape),
                value(DataType.INT64), value(DataType.FLOAT32, updateShape),
                value(DataType.FLOAT32, shape)),
                List.of(MetalMpsGraphProgram.Node.scatterAdd(0, 1, 2, 3, 14)))) {
            e.upload(0, floats(10, 20, 30, 40, 50, 60));
            e.upload(1, indices(DataType.INT64, 1));
            e.upload(2, floats(3, 4));
            e.run();
            assertArrayEquals(floats(10, 20, 33, 44, 50, 60), e.output());
        }
    }

    @Test
    void scatterElementsCombinesBaseAndEveryDuplicateExactlyOnce() {
        byte[][] expected = {floats(5, 2, 9, 10, 20, 2), floats(4, 2, 24, 16, 420, 2),
                floats(2, 2, 4, 8, 7, 2), floats(1, 2, 2, 2, 2, 2)};
        for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
            for (int reduction = 2; reduction <= 5; reduction++) {
                try (Execution e = new Execution(List.of(value(DataType.FLOAT32, 2, 3),
                        value(indexType, 2, 4), value(DataType.FLOAT32, 2, 4),
                        value(DataType.FLOAT32, 2, 3)), List.of(elements(1, reduction, 1)))) {
                    e.upload(0, floats(2, 2, 2, 2, 2, 2));
                    e.upload(1, indices(indexType, 0, 0, 2, 2, 1, 1, 1, 0));
                    e.upload(2, floats(1, 2, 3, 4, 5, 6, 7, 8));
                    e.run();
                    assertArrayEquals(expected[reduction - 2], e.output(), indexType + "/" + reduction);
                }
            }
        }
    }

    @Test
    void scatterNdKeepsBatchesAndSuffixSlicesSeparateWithDuplicateTuples() {
        byte[][] expected = {floats(7, 8, 6, 8, 2, 2, 11, 12, 2, 2, 20, 22),
                floats(10, 12, 6, 16, 2, 2, 18, 20, 2, 2, 154, 192),
                floats(5, 6, 3, 4, 2, 2, 9, 10, 2, 2, 11, 12),
                floats(2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2)};
        for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
            for (int reduction = 2; reduction <= 5; reduction++) {
                try (Execution e = new Execution(List.of(value(DataType.FLOAT32, 2, 3, 2),
                        value(indexType, 2, 3, 1), value(DataType.FLOAT32, 2, 3, 2),
                        value(DataType.FLOAT32, 2, 3, 2)), List.of(nd(1, reduction, 1)))) {
                    e.upload(0, floats(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2));
                    e.upload(1, indices(indexType, 1, 1, 0, 2, 0, 2));
                    e.upload(2, floats(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
                    e.run();
                    assertArrayEquals(expected[reduction - 2], e.output(), indexType + "/" + reduction);
                }
            }
        }
    }

    @Test
    void affineBaseIndexAndUpdateRolesAreReadByTheirActualStorageCoordinates() {
        var values = List.of(view(DataType.FLOAT32, new long[] {2, 3}, new long[] {4, 1}, 1),
                view(DataType.INT64, new long[] {2, 2}, new long[] {3, 1}, 1),
                view(DataType.FLOAT32, new long[] {2, 2}, new long[] {1, 3}, 1),
                value(DataType.FLOAT32, 2, 3));
        try (Execution e = new Execution(values, List.of(elements(1, 2, 1)))) {
            e.upload(0, floats(99, 10, 20, 30, 99, 40, 50, 60));
            e.upload(1, indices(DataType.INT64, 99, 0, 0, 99, 2, 2));
            e.upload(2, floats(99, 1, 3, 99, 2, 4));
            e.run();
            assertArrayEquals(floats(13, 20, 30, 40, 50, 67), e.output());
        }
    }

    @Test
    void producedIndicesAreValidatedBeforeScatterWritesAndValidDuplicatesRemainLegal() {
        for (boolean tuple : List.of(false, true)) {
            var indexShape = tuple ? new long[] {2, 1} : new long[] {2};
            var values = List.of(value(DataType.FLOAT32, 3), value(DataType.INT64, indexShape),
                    value(DataType.FLOAT32, 2), value(DataType.FLOAT32, 3),
                    value(DataType.INT64, indexShape));
            var scatter = tuple ? nd(0, 2, 4) : MetalMpsGraphProgram.Node.scatterAdd(0, 4, 2, 3, 0);
            try (Execution e = new Execution(values,
                    List.of(MetalMpsGraphProgram.Node.contiguous(1, 4), scatter))) {
                e.upload(0, floats(10, 20, 30));
                e.upload(1, indices(DataType.INT64, 1, 3));
                e.upload(2, floats(2, 3));
                byte[] sentinel = words(0x5a5a5a5a, 0x5a5a5a5a, 0x5a5a5a5a);
                e.upload(3, sentinel);
                assertThrows(IndexOutOfBoundsException.class, e::run);
                assertArrayEquals(sentinel, e.output());
                assertArrayEquals(floats(10, 20, 30), e.download(0));
                e.upload(1, indices(DataType.INT64, 1, 1));
                e.run();
                assertArrayEquals(floats(10, 25, 30), e.output());
            }
        }
    }

    @Test
    void replacementChecksAllBoundsThenUniquenessBeforeAnyOutputWrite() {
        for (boolean tuple : List.of(false, true)) {
            var indexShape = tuple ? new long[] {2, 1} : new long[] {2};
            try (Execution e = new Execution(List.of(value(DataType.FLOAT32, 3),
                    value(DataType.INT32, indexShape), value(DataType.FLOAT32, 2),
                    value(DataType.FLOAT32, 3)), List.of(tuple ? nd(0, 1, 1) : elements(0, 1, 1)))) {
                e.upload(0, floats(10, 20, 30));
                e.upload(2, floats(2, 3));
                byte[] sentinel = words(0x5a5a5a5a, 0x5a5a5a5a, 0x5a5a5a5a);
                e.upload(3, sentinel);
                e.upload(1, indices(DataType.INT32, 1, 3));
                assertThrows(IndexOutOfBoundsException.class, e::run);
                assertArrayEquals(sentinel, e.output());
                e.upload(1, indices(DataType.INT32, 1, 1));
                assertThrows(IllegalArgumentException.class, e::run);
                assertArrayEquals(sentinel, e.output());
                assertArrayEquals(floats(10, 20, 30), e.download(0));
                e.upload(1, indices(DataType.INT32, 2, 0));
                e.run();
                assertArrayEquals(floats(3, 20, 2), e.output());
            }
        }
    }

    @Test
    void extremaKeepStoredSubnormalsSignedZerosAndFirstNanWhileUntouchedWordsStayRaw() {
        for (int reduction : new int[] {4, 5}) {
            try (Execution e = new Execution(List.of(value(DataType.FLOAT32, 5),
                    value(DataType.INT32, 4), value(DataType.FLOAT32, 4),
                    value(DataType.FLOAT32, 5)), List.of(elements(0, reduction, 1)))) {
                e.upload(0, words(0x80000000, 1, 0x7fc12345, 0x7f812345, 0x80000000));
                e.upload(1, indices(DataType.INT32, 0, 1, 2, 2));
                e.upload(2, words(0, 0, 0x3f800000, 0x7fc54321));
                e.run();
                int[] result = decodedWords(e.output());
                assertEquals(reduction == 4 ? 0 : 0x80000000, result[0]);
                assertEquals(reduction == 4 ? 1 : 0, result[1]);
                assertTrue(Float.isNaN(Float.intBitsToFloat(result[2])));
                assertEquals(0x7f812345, result[3]);
                assertEquals(0x80000000, result[4]);
            }
        }
    }

    @Test
    void mulIncludesZerosAndNonfiniteFactorsAndNeverTouchesUnaddressedPayloads() {
        try (Execution e = new Execution(List.of(value(DataType.FLOAT32, 4),
                value(DataType.INT32, 4), value(DataType.FLOAT32, 4),
                value(DataType.FLOAT32, 4)), List.of(elements(0, 3, 1)))) {
            e.upload(0, words(0x40000000, 0x7f800000, 0x7fc12345, 0x80000000));
            e.upload(1, indices(DataType.INT32, 0, 0, 1, 1));
            e.upload(2, floats(0, 3, 0, 4));
            e.run();
            int[] result = decodedWords(e.output());
            assertEquals(0, result[0]);
            assertTrue(Float.isNaN(Float.intBitsToFloat(result[1])));
            assertEquals(0x7fc12345, result[2]);
            assertEquals(0x80000000, result[3]);
        }
    }

    @Test
    void integralArithmeticWrapsAtExactWidthAndExtremaUseSignedOrder() {
        for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
            long max = type == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
            long min = type == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
            long[][] expected = {{min, 8, min + 1, 7}, {-2, 8, 0, 7}, {max, 4, 1, 7}, {1, 2, min, 7}};
            for (int reduction = 2; reduction <= 5; reduction++) {
                try (Execution e = new Execution(List.of(value(type, 4), value(DataType.INT64, 3),
                        value(type, 3), value(type, 4)), List.of(elements(0, reduction, 1)))) {
                    e.upload(0, indices(type, max, 2, min, 7));
                    e.upload(1, indices(DataType.INT64, 0, 1, 2));
                    e.upload(2, indices(type, reduction == 3 ? 2 : 1, reduction == 2 ? 6 : 4, reduction == 3 ? 2 : 1));
                    e.run();
                    assertArrayEquals(indices(type, expected[reduction - 2]), e.output(), type + "/" + reduction);
                }
            }
        }
    }

    @Test
    void scatterNdMultiAxisTuplesAndSingleScalarUpdateUseTheExactSuffixProjection() {
        try (Execution e = new Execution(List.of(value(DataType.FLOAT32, 2, 3, 2),
                value(DataType.INT64, 3, 2), value(DataType.FLOAT32, 3, 2),
                value(DataType.FLOAT32, 2, 3, 2)), List.of(nd(0, 2, 1)))) {
            e.upload(0, floats(10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10));
            e.upload(1, indices(DataType.INT64, 1, 2, 0, 0, 1, 2));
            e.upload(2, floats(1, 2, 3, 4, 5, 6));
            e.run();
            assertArrayEquals(floats(13, 14, 10, 10, 10, 10, 10, 10, 10, 10, 16, 18), e.output());
        }
        try (Execution e = new Execution(List.of(value(DataType.FLOAT32, 2, 3, 2),
                value(DataType.INT64, 3), value(DataType.FLOAT32),
                value(DataType.FLOAT32, 2, 3, 2)), List.of(nd(0, 2, 1)))) {
            e.upload(0, floats(10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10));
            e.upload(1, indices(DataType.INT64, 1, 1, 0));
            e.upload(2, floats(3));
            e.run();
            assertArrayEquals(floats(10, 10, 10, 10, 10, 10, 10, 10, 13, 10, 10, 10), e.output());
        }
    }

    @Test
    void float64ExtremaUseRawStoredComparisonWithoutFloat64Arithmetic() {
        for (int reduction : new int[] {4, 5}) {
            try (Execution e = new Execution(List.of(value(DataType.FLOAT64, 4),
                    value(DataType.INT64, 3), value(DataType.FLOAT64, 3),
                    value(DataType.FLOAT64, 4)), List.of(elements(0, reduction, 1)))) {
                e.upload(0, indices(DataType.INT64, Long.MIN_VALUE, 1, 0x7ff123456789abcdL, 0xfff123456789abcdL));
                e.upload(1, indices(DataType.INT64, 0, 1, 2));
                e.upload(2, indices(DataType.INT64, 0, 0, 0x3ff0000000000000L));
                e.run();
                assertArrayEquals(indices(DataType.INT64, reduction == 4 ? 0 : Long.MIN_VALUE,
                        reduction == 4 ? 1 : 0, 0x7ff123456789abcdL, 0xfff123456789abcdL), e.output());
            }
        }
    }

    @Test
    void broadcastBaseReadsRemainLegalWhenNonbroadcastAxesAreInjective() {
        try (Execution e = new Execution(List.of(
                view(DataType.FLOAT32, new long[] {2, 2, 2}, new long[] {0, 2, 1}, 0),
                value(DataType.INT32, 1), value(DataType.FLOAT32, 1, 2, 2),
                value(DataType.FLOAT32, 2, 2, 2)),
                List.of(MetalMpsGraphProgram.Node.scatterAdd(0, 1, 2, 3, 0)))) {
            e.upload(0, floats(10, 20, 30, 40));
            e.upload(1, indices(DataType.INT32, 0));
            e.upload(2, floats(1, 2, 3, 4));
            e.run();
            assertArrayEquals(floats(11, 22, 33, 44, 10, 20, 30, 40), e.output());
        }
    }

    private static MetalMpsGraphProgram.Node elements(int axis, int reduction, int indexValue) {
        return MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.SCATTER_ELEMENTS,
                new int[] {0, indexValue, 2}, new int[] {3},
                MetalMpsGraphProgram.AttributeKind.SCATTER_ELEMENTS, new long[] {axis, reduction});
    }

    private static MetalMpsGraphProgram.Node nd(int batch, int reduction, int indexValue) {
        return MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.SCATTER_ND,
                new int[] {0, indexValue, 2}, new int[] {3},
                MetalMpsGraphProgram.AttributeKind.SCATTER_ND, new long[] {batch, reduction});
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(DataType type, long... shape) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, shape, false);
    }

    private static MetalMpsGraphProgram.ValueDescriptor view(DataType type, long[] shape, long[] strides, long offset) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, shape,
                Optional.of(LayoutDescriptor.of(Shape.of(shape), strides, offset, true)), false, false);
    }

    private static byte[] floats(float... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.nativeOrder());
        for (float value : values) bytes.putFloat(value);
        return bytes.array();
    }

    private static byte[] words(int... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Integer.BYTES).order(ByteOrder.nativeOrder());
        for (int value : values) bytes.putInt(value);
        return bytes.array();
    }

    private static int[] decodedWords(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
        int[] values = new int[bytes.length / Integer.BYTES];
        for (int i = 0; i < values.length; i++) values[i] = buffer.getInt();
        return values;
    }

    private static byte[] indices(DataType type, long... values) {
        ByteBuffer bytes = ByteBuffer.allocate(Math.toIntExact(values.length * type.byteWidth()))
                .order(ByteOrder.nativeOrder());
        for (long value : values) {
            if (type == DataType.INT32) bytes.putInt((int) value);
            else bytes.putLong(value);
        }
        return bytes.array();
    }

    /** Keeps the complete custom value table stable across validation failures and retries. */
    private static final class Execution implements AutoCloseable {
        private final MetalNativeApi api;
        private final MetalNativeApi.Handle context;
        private final MetalNativeApi.Handle executable;
        private final List<MetalNativeApi.Handle> buffers = new ArrayList<>();
        private final List<MetalMpsGraphProgram.ValueDescriptor> values;

        private Execution(List<MetalMpsGraphProgram.ValueDescriptor> values, List<MetalMpsGraphProgram.Node> nodes) {
            String library = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
            assumeTrue(library != null && !library.isBlank(), "native device library not configured");
            this.values = values;
            api = MetalNativeApi.open(Path.of(library));
            MetalNativeApi.Handle createdContext = null;
            MetalNativeApi.Handle createdExecutable = null;
            try {
                createdContext = api.createContext();
                createdExecutable = api.createProgramExecutable(createdContext, values,
                        new MetalMpsGraphProgram(nodes), new int[] {0, 1, 2}, new int[] {3},
                        MetalPreparedRoute.CUSTOM_PROGRAM);
                for (var value : values) buffers.add(api.createBuffer(createdContext, value.byteCount()));
                context = createdContext;
                executable = createdExecutable;
            } catch (RuntimeException | Error failure) {
                for (int i = buffers.size(); i-- > 0;) api.releaseBuffer(buffers.get(i));
                if (createdExecutable != null) api.releaseExecutable(createdExecutable);
                if (createdContext != null) api.releaseContext(createdContext);
                api.close();
                throw failure;
            }
        }

        private void upload(int value, byte[] bytes) {
            assertEquals(values.get(value).byteCount(), bytes.length);
            try (Arena arena = Arena.ofConfined()) {
                var source = arena.allocate(bytes.length, 1);
                source.copyFrom(MemorySegment.ofArray(bytes));
                api.upload(buffers.get(value), 0, source, bytes.length);
            }
        }

        private byte[] download(int value) {
            long size = values.get(value).byteCount();
            try (Arena arena = Arena.ofConfined()) {
                var target = arena.allocate(size, 1);
                api.download(buffers.get(value), 0, target, size);
                return target.toArray(JAVA_BYTE);
            }
        }

        private byte[] output() { return download(3); }

        private void run() {
            try (Arena arena = Arena.ofConfined()) {
                var table = arena.allocate(ADDRESS, buffers.size());
                for (int i = 0; i < buffers.size(); i++) table.setAtIndex(ADDRESS, i, buffers.get(i).carrier());
                var targets = arena.allocate(ADDRESS);
                targets.setAtIndex(ADDRESS, 0, buffers.get(3).carrier());
                api.runExecutable(executable, buffers.size(), table, 1, targets);
            }
        }

        @Override
        public void close() {
            for (int i = buffers.size(); i-- > 0;) api.releaseBuffer(buffers.get(i));
            api.releaseExecutable(executable);
            api.releaseContext(context);
            api.close();
        }
    }
}
