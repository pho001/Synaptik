package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
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

class MetalBoolNativeTest {
    private static final int[] CLASSIFICATION_BITS = {
        0x00000000, 0x80000000, 0x00000001, 0x007fffff,
        0x00800000, 0x7f7fffff, 0x7f800000, 0xff800000,
        0x7fc12345, 0xffc54321, 0x7f812345, 0xff812346
    };
    private static final byte[] FINITE = {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};
    private static final byte[] NAN = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
    private static final byte[] INF = {0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0};
    private static final byte[] LEFT = {1, 0};
    private static final byte[] RIGHT = {1, 0, 1};
    private static final byte[] AND = {1, 0, 1, 0, 0, 0};
    private static final byte[] OR = {1, 1, 1, 1, 0, 1};
    private static final byte[] NOT_FINITE = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    private static final int[] TRUE_BITS = {0x7fc12345, 0x80000000};
    private static final int[] FALSE_BITS = {0x7f800000, 0xffc54321, 0x00000001};
    private static final int[] WHERE_BITS = {
        0x7fc12345, 0xffc54321, 0x7fc12345,
        0x7f800000, 0xffc54321, 0x00000001
    };
    private static final int[] DIRECT_WHERE_BITS = {
        0x7fc12345, 0xffc54321, 0x7fc12345,
        0x80000000, 0xffc54321, 0x80000000
    };

    @Test
    void productionCustomProgramExecutesAllSevenExactBoolOperationsAndRejectsNonCanonicalIngress() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            var program = new MetalMpsGraphProgram(List.of(
                    node(MetalMpsGraphProgram.NodeKind.IS_FINITE, new int[] {0}, 1),
                    node(MetalMpsGraphProgram.NodeKind.IS_NAN, new int[] {0}, 10),
                    node(MetalMpsGraphProgram.NodeKind.IS_INF, new int[] {0}, 11),
                    node(MetalMpsGraphProgram.NodeKind.LOGICAL_AND, new int[] {2, 3}, 4),
                    node(MetalMpsGraphProgram.NodeKind.LOGICAL_OR, new int[] {2, 3}, 5),
                    node(MetalMpsGraphProgram.NodeKind.LOGICAL_NOT, new int[] {1}, 6),
                    node(MetalMpsGraphProgram.NodeKind.WHERE, new int[] {4, 7, 8}, 9)));
            List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                    value(DataType.FLOAT32, 3, 4), value(DataType.BOOL, 3, 4),
                    value(DataType.BOOL, rankSixteen(2, 1)), value(DataType.BOOL, 3),
                    value(DataType.BOOL, rankSixteen(2, 3)),
                    value(DataType.BOOL, rankSixteen(2, 3)),
                    value(DataType.BOOL, 3, 4),
                    value(DataType.FLOAT32, rankSixteen(2, 1)),
                    value(DataType.FLOAT32, 3),
                    value(DataType.FLOAT32, rankSixteen(2, 3)),
                    value(DataType.BOOL, 3, 4), value(DataType.BOOL, 3, 4));
            int[] feeds = {0, 2, 3, 7, 8};
            int[] targets = {1, 4, 5, 6, 9, 10, 11};
            executable = api.createMpsGraphExecutable(
                    context, NumericalProfile.STRICT_IEEE, values, program,
                    feeds, targets, MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var descriptor : values) {
                buffers.add(api.createBuffer(context, descriptor.byteCount()));
            }
            upload(api, buffers.get(0), ints(CLASSIFICATION_BITS));
            upload(api, buffers.get(2), LEFT);
            upload(api, buffers.get(3), RIGHT);
            upload(api, buffers.get(7), ints(TRUE_BITS));
            upload(api, buffers.get(8), ints(FALSE_BITS));

            runCustom(api, executable, buffers, targets);
            assertArrayEquals(FINITE, download(api, buffers.get(1), FINITE.length));
            assertArrayEquals(NAN, download(api, buffers.get(10), NAN.length));
            assertArrayEquals(INF, download(api, buffers.get(11), INF.length));
            assertArrayEquals(AND, download(api, buffers.get(4), AND.length));
            assertArrayEquals(OR, download(api, buffers.get(5), OR.length));
            assertArrayEquals(NOT_FINITE, download(api, buffers.get(6), NOT_FINITE.length));
            assertArrayEquals(WHERE_BITS, bytesToInts(download(
                    api, buffers.get(9), WHERE_BITS.length * Integer.BYTES)));

            upload(api, buffers.get(3), new byte[] {1, 2, 1});
            fill(api, buffers.get(9), WHERE_BITS.length * Integer.BYTES, (byte) 0x5a);
            MetalNativeApi.Handle retainedExecutable = executable;
            assertThrows(MetalNativeApi.NativeFailure.class,
                    () -> runCustom(api, retainedExecutable, buffers, targets));
            byte[] sentinel = new byte[WHERE_BITS.length * Integer.BYTES];
            java.util.Arrays.fill(sentinel, (byte) 0x5a);
            assertArrayEquals(sentinel, download(api, buffers.get(9), sentinel.length));
        } finally {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    @Test
    void directMpsGraphCandidateExecutesEveryAuditedBoolSelector() {
        Path library = configuredLibrary();
        for (NumericalProfile profile : NumericalProfile.values()) {
            assertArrayEquals(FINITE, executeDirect(
                    library, profile,
                    node(MetalMpsGraphProgram.NodeKind.IS_FINITE, new int[] {0}, 1),
                    List.of(value(DataType.FLOAT32, 3, 4), value(DataType.BOOL, 3, 4)),
                    List.of(ints(CLASSIFICATION_BITS))));
            assertArrayEquals(NAN, executeDirect(
                    library, profile,
                    node(MetalMpsGraphProgram.NodeKind.IS_NAN, new int[] {0}, 1),
                    List.of(value(DataType.FLOAT32, 3, 4), value(DataType.BOOL, 3, 4)),
                    List.of(ints(CLASSIFICATION_BITS))));
            assertArrayEquals(INF, executeDirect(
                    library, profile,
                    node(MetalMpsGraphProgram.NodeKind.IS_INF, new int[] {0}, 1),
                    List.of(value(DataType.FLOAT32, 3, 4), value(DataType.BOOL, 3, 4)),
                    List.of(ints(CLASSIFICATION_BITS))));
            assertArrayEquals(AND, executeDirect(
                    library, profile,
                    node(MetalMpsGraphProgram.NodeKind.LOGICAL_AND, new int[] {0, 1}, 2),
                    List.of(value(DataType.BOOL, 2, 1), value(DataType.BOOL, 3),
                            value(DataType.BOOL, 2, 3)),
                    List.of(LEFT, RIGHT)));
            assertArrayEquals(OR, executeDirect(
                    library, profile,
                    node(MetalMpsGraphProgram.NodeKind.LOGICAL_OR, new int[] {0, 1}, 2),
                    List.of(value(DataType.BOOL, 2, 1), value(DataType.BOOL, 3),
                            value(DataType.BOOL, 2, 3)),
                    List.of(LEFT, RIGHT)));
            assertArrayEquals(NOT_FINITE, executeDirect(
                    library, profile,
                    node(MetalMpsGraphProgram.NodeKind.LOGICAL_NOT, new int[] {0}, 1),
                    List.of(value(DataType.BOOL, 3, 4), value(DataType.BOOL, 3, 4)),
                    List.of(FINITE)));
            assertArrayEquals(ints(DIRECT_WHERE_BITS), executeDirect(
                    library, profile,
                    node(MetalMpsGraphProgram.NodeKind.WHERE, new int[] {0, 1, 2}, 3),
                    List.of(value(DataType.BOOL, 3), value(DataType.FLOAT32, 2, 1),
                            value(DataType.FLOAT32, 1, 3), value(DataType.FLOAT32, 2, 3)),
                    List.of(RIGHT, ints(TRUE_BITS), ints(FALSE_BITS))));
        }
    }

    private static byte[] executeDirect(
            Path library,
            NumericalProfile profile,
            MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            List<byte[]> feedBytes) {
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var inputs = new ArrayList<MetalNativeApi.Handle>();
        MetalNativeApi.Handle output = null;
        try {
            context = api.createContext();
            var program = new MetalMpsGraphProgram(List.of(node));
            int[] feeds = java.util.stream.IntStream.range(0, feedBytes.size()).toArray();
            int target = values.size() - 1;
            executable = api.createMpsGraphExecutable(
                    context, profile, values, program, feeds, new int[] {target},
                    MetalPreparedRoute.MPSGRAPH);
            for (int index = 0; index < feedBytes.size(); index++) {
                MetalNativeApi.Handle buffer = api.createBuffer(context, feedBytes.get(index).length);
                inputs.add(buffer);
                upload(api, buffer, feedBytes.get(index));
            }
            output = api.createBuffer(context, values.get(target).byteCount());
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size());
                for (int index = 0; index < inputs.size(); index++) {
                    inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
                }
                MemorySegment outputAddresses = arena.allocate(ADDRESS);
                outputAddresses.set(ADDRESS, 0L, output.carrier());
                api.runExecutable(
                        executable, inputs.size(), inputAddresses, 1, outputAddresses);
            }
            return download(api, output, Math.toIntExact(values.get(target).byteCount()));
        } finally {
            if (output != null) api.releaseBuffer(output);
            for (int index = inputs.size(); index-- > 0;) api.releaseBuffer(inputs.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static void runCustom(
            MetalNativeApi api,
            MetalNativeApi.Handle executable,
            List<MetalNativeApi.Handle> buffers,
            int[] targets) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment inputAddresses = arena.allocate(ADDRESS, buffers.size());
            for (int index = 0; index < buffers.size(); index++) {
                inputAddresses.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
            }
            MemorySegment outputAddresses = arena.allocate(ADDRESS, targets.length);
            for (int index = 0; index < targets.length; index++) {
                outputAddresses.setAtIndex(ADDRESS, index, buffers.get(targets[index]).carrier());
            }
            api.runExecutable(
                    executable, buffers.size(), inputAddresses, targets.length, outputAddresses);
        }
    }

    private static MetalMpsGraphProgram.Node node(
            MetalMpsGraphProgram.NodeKind kind, int[] inputs, int output) {
        return MetalMpsGraphProgram.Node.generic(
                kind, inputs, new int[] {output},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(
            DataType dataType, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(dataType, dimensions, false);
    }

    private static long[] rankSixteen(long first, long last) {
        long[] dimensions = new long[MetalMpsGraphProgram.MAX_RANK];
        java.util.Arrays.fill(dimensions, 1L);
        dimensions[0] = first;
        dimensions[dimensions.length - 1] = last;
        return dimensions;
    }


    private static byte[] ints(int[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int value : values) bytes.putInt(value);
        return bytes.array();
    }

    private static int[] bytesToInts(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
        int[] values = new int[bytes.length / Integer.BYTES];
        for (int index = 0; index < values.length; index++) values[index] = buffer.getInt();
        return values;
    }

    private static void upload(
            MetalNativeApi api, MetalNativeApi.Handle buffer, byte[] values) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(values.length, 1L);
            MemorySegment.copy(values, 0, source, JAVA_BYTE, 0L, values.length);
            api.upload(buffer, 0L, source, values.length);
        }
    }

    private static void fill(
            MetalNativeApi api, MetalNativeApi.Handle buffer, long bytes, byte value) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes, 1L);
            source.fill(value);
            api.upload(buffer, 0L, source, bytes);
        }
    }

    private static byte[] download(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int bytes) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(bytes, 1L);
            api.download(buffer, 0L, target, bytes);
            return target.toArray(JAVA_BYTE);
        }
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }
}
