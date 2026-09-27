package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalScatterAddNativeTest {
    private static final int SENTINEL = 0x5a5a5a5a;

    @Test
    void customKernelRetainsDuplicatesUsesOrdinalOrderAndRawCopiesUnaddressedWords() {
        Path library = configuredLibrary();
        for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
            Execution execution = open(library, indexType);
            try {
                uploadInts(execution.api(), execution.buffers().get(0),
                        Float.floatToRawIntBits(1.0e20f),
                        0x80000000,
                        0x00000001,
                        0x7f812345,
                        0x7fc54321);
                uploadIndices(execution.api(), execution.buffers().get(1), indexType, 0, 0, 4);
                uploadInts(execution.api(), execution.buffers().get(2),
                        Float.floatToRawIntBits(-1.0e20f),
                        Float.floatToRawIntBits(3.0f),
                        Float.floatToRawIntBits(1.0f));
                run(execution);
                int[] actual =
                        downloadInts(execution.api(), execution.buffers().get(3), 5);
                assertEquals(Float.floatToRawIntBits(3.0f), actual[0]);
                assertEquals(0x80000000, actual[1]);
                assertEquals(0x00000001, actual[2]);
                assertEquals(0x7f812345, actual[3]);
                assertTrue(Float.isNaN(Float.intBitsToFloat(actual[4])));
            } finally {
                execution.close();
            }
        }
    }

    @Test
    void addressedSubnormalInfinityAndNanResultsStayInsideAuthorizedClasses() {
        Path library = configuredLibrary();
        for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
            Execution execution = open(library, indexType);
            try {
                uploadInts(execution.api(), execution.buffers().get(0),
                        0x00000001,
                        0x7f800000,
                        0x7f812345,
                        0x80000000,
                        0x7fc54321);
                uploadIndices(execution.api(), execution.buffers().get(1), indexType, 0, 1, 2);
                uploadInts(execution.api(), execution.buffers().get(2),
                        0x00000001,
                        Float.floatToRawIntBits(1.0f),
                        Float.floatToRawIntBits(1.0f));
                run(execution);
                int[] actual =
                        downloadInts(execution.api(), execution.buffers().get(3), 5);
                assertTrue(actual[0] == 0x00000000
                        || actual[0] == 0x00000001
                        || actual[0] == 0x00000002);
                assertEquals(0x7f800000, actual[1]);
                assertTrue(Float.isNaN(Float.intBitsToFloat(actual[2])));
                assertEquals(0x80000000, actual[3]);
                assertEquals(0x7fc54321, actual[4]);
            } finally {
                execution.close();
            }
        }
    }

    @Test
    void completeCpuIndexScanRejectsBeforeAnyOutputMutation() {
        Path library = configuredLibrary();
        for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
            Execution execution = open(library, indexType);
            try {
                uploadInts(execution.api(), execution.buffers().get(0), 1, 2, 3, 4, 5);
                uploadIndices(execution.api(), execution.buffers().get(1), indexType, 0, 1, 5);
                uploadInts(execution.api(), execution.buffers().get(2), 10, 20, 30);
                uploadInts(execution.api(), execution.buffers().get(3),
                        SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL);
                assertThrows(MetalNativeApi.NativeFailure.class, () -> run(execution));
                assertArrayEquals(
                        new int[] {SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL},
                        downloadInts(execution.api(), execution.buffers().get(3), 5));
                uploadIndices(execution.api(), execution.buffers().get(1), indexType, 0, -1, 2);
                assertThrows(MetalNativeApi.NativeFailure.class, () -> run(execution));
                assertArrayEquals(
                        new int[] {SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL},
                        downloadInts(execution.api(), execution.buffers().get(3), 5));
            } finally {
                execution.close();
            }
        }
    }

    @Test
    void runtimeRejectsAliasedValueBuffersBeforeMutation() {
        Execution execution = open(configuredLibrary(), DataType.INT32);
        try {
            uploadInts(execution.api(), execution.buffers().get(0), 1, 2, 3, 4, 5);
            uploadIndices(
                    execution.api(), execution.buffers().get(1), DataType.INT32, 0, 1, 2);
            uploadInts(execution.api(), execution.buffers().get(3),
                    SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL);
            assertThrows(MetalNativeApi.NativeFailure.class, () -> {
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment table = arena.allocate(ADDRESS, 4);
                    table.setAtIndex(ADDRESS, 0, execution.buffers().get(0).carrier());
                    table.setAtIndex(ADDRESS, 1, execution.buffers().get(1).carrier());
                    table.setAtIndex(ADDRESS, 2, execution.buffers().get(3).carrier());
                    table.setAtIndex(ADDRESS, 3, execution.buffers().get(3).carrier());
                    MemorySegment outputs = arena.allocate(ADDRESS);
                    outputs.setAtIndex(ADDRESS, 0, execution.buffers().get(3).carrier());
                    execution.api().runExecutable(
                            execution.executable(), 4, table, 1, outputs);
                }
            });
            assertArrayEquals(
                    new int[] {SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL},
                    downloadInts(execution.api(), execution.buffers().get(3), 5));
        } finally {
            execution.close();
        }
    }

    @Test
    void JavaPreflightRejectsAlternateProfileRouteGradientFeedAndAliasing() {
        var validProgram = program(0, 1, 2, 3);
        var values = values(DataType.INT32, false);
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.ACCELERATOR,
                values,
                validProgram,
                new int[] {0, 1, 2},
                new int[] {3},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.STRICT_IEEE,
                        values,
                        validProgram,
                        new int[] {0, 1, 2},
                        new int[] {3},
                        MetalPreparedRoute.CUSTOM_PROGRAM));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        values,
                        validProgram,
                        new int[] {0, 1, 2},
                        new int[] {3},
                        MetalPreparedRoute.MPSGRAPH));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        values(DataType.INT32, true),
                        validProgram,
                        new int[] {0, 1, 2},
                        new int[] {3},
                        MetalPreparedRoute.CUSTOM_PROGRAM));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        values,
                        validProgram,
                        new int[] {0, 2},
                        new int[] {3},
                        MetalPreparedRoute.CUSTOM_PROGRAM));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        values,
                        program(0, 1, 0, 3),
                        new int[] {0, 1, 2},
                        new int[] {3},
                        MetalPreparedRoute.CUSTOM_PROGRAM));
    }

    private static Execution open(Path library, DataType indexType) {
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createMpsGraphExecutable(
                    context,
                    NumericalProfile.ACCELERATOR,
                    values(indexType, false),
                    program(0, 1, 2, 3),
                    new int[] {0, 1, 2},
                    new int[] {3},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            buffers.add(api.createBuffer(context, 5L * Float.BYTES));
            buffers.add(api.createBuffer(context, 3L * indexType.byteWidth()));
            buffers.add(api.createBuffer(context, 3L * Float.BYTES));
            buffers.add(api.createBuffer(context, 5L * Float.BYTES));
            return new Execution(api, context, executable, buffers);
        } catch (RuntimeException failure) {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
            throw failure;
        }
    }

    private static MetalMpsGraphProgram program(int data, int indices, int updates, int output) {
        return new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.scatterAdd(data, indices, updates, output, 0)));
    }

    private static List<MetalMpsGraphProgram.ValueDescriptor> values(
            DataType indexType, boolean dataGradient) {
        return List.of(
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {5}, dataGradient),
                new MetalMpsGraphProgram.ValueDescriptor(
                        indexType, new long[] {3}, false),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {3}, false),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {5}, false));
    }

    private static void run(Execution execution) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment table = arena.allocate(ADDRESS, execution.buffers().size());
            for (int index = 0; index < execution.buffers().size(); index++) {
                table.setAtIndex(ADDRESS, index, execution.buffers().get(index).carrier());
            }
            MemorySegment outputs = arena.allocate(ADDRESS);
            outputs.setAtIndex(ADDRESS, 0, execution.buffers().get(3).carrier());
            execution.api().runExecutable(
                    execution.executable(), execution.buffers().size(), table, 1, outputs);
        }
    }

    private static void uploadIndices(
            MetalNativeApi api,
            MetalNativeApi.Handle buffer,
            DataType type,
            long... values) {
        byte[] bytes = new byte[Math.toIntExact((long) values.length * type.byteWidth())];
        ByteBuffer encoded = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
        for (long value : values) {
            if (type == DataType.INT32) encoded.putInt(Math.toIntExact(value));
            else encoded.putLong(value);
        }
        upload(api, buffer, bytes);
    }

    private static void uploadInts(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int... values) {
        byte[] bytes = new byte[values.length * Integer.BYTES];
        ByteBuffer encoded = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
        for (int value : values) encoded.putInt(value);
        upload(api, buffer, bytes);
    }

    private static void upload(
            MetalNativeApi api, MetalNativeApi.Handle buffer, byte[] bytes) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes.length, 1L);
            MemorySegment.copy(bytes, 0, source, JAVA_BYTE, 0L, bytes.length);
            api.upload(buffer, 0L, source, bytes.length);
        }
    }

    private static int[] downloadInts(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int count) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate((long) count * Integer.BYTES, Integer.BYTES);
            api.download(buffer, 0L, target, target.byteSize());
            ByteBuffer decoded = ByteBuffer.wrap(target.toArray(JAVA_BYTE))
                    .order(ByteOrder.nativeOrder());
            int[] result = new int[count];
            for (int index = 0; index < count; index++) result[index] = decoded.getInt();
            return result;
        }
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }

    private record Execution(
            MetalNativeApi api,
            MetalNativeApi.Handle context,
            MetalNativeApi.Handle executable,
            List<MetalNativeApi.Handle> buffers) implements AutoCloseable {
        @Override
        public void close() {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            api.releaseExecutable(executable);
            api.releaseContext(context);
            api.close();
        }
    }
}
