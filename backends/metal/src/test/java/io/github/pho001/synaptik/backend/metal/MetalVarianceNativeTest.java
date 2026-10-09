package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalVarianceNativeTest {
    @Test
    void customKernelProducesPositiveZeroForEveryFiniteSingletonAndNanForSpecials() {
        Path library = configuredLibrary();
        int[] finite = {
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x007fffff,
            0x807fffff,
            0x00800000,
            0x80800000,
            0x3fc00000,
            0xbfc00000,
            0x7f7fffff,
            0xff7fffff
        };
        for (boolean keepDimensions : List.of(false, true)) {
            for (int word : finite) {
                assertEquals(0x00000000, execute(library, word, keepDimensions));
            }
            for (int word : new int[] {0x7f800000, 0xff800000, 0x7fc12345, 0xff812345}) {
                int actual = execute(library, word, keepDimensions);
                assertTrue((actual & 0x7f800000) == 0x7f800000
                                && (actual & 0x007fffff) != 0,
                        () -> "expected NaN, but got 0x" + Integer.toHexString(actual));
            }
        }
    }

    @Test
    void customAbiRejectsAlternateRouteAttributeGradientTypeAndGeometry() {
        var valid = program(false, List.of(0), 0L);
        var values = values(new long[] {1L}, false, false, false, DataType.FLOAT32);
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                values,
                valid,
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                values(new long[] {1L}, false, true, false, DataType.FLOAT32),
                program(true, List.of(0), 0L),
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.CUSTOM_PROGRAM);

        assertRejected(values, valid,
                MetalPreparedRoute.MPSGRAPH);
        assertRejected(values,
                program(false, List.of(), 0L), MetalPreparedRoute.CUSTOM_PROGRAM);
        assertRejected(values(new long[] {1L, 1L}, false, false, false, DataType.FLOAT32),
                program(false, List.of(0), 0L), MetalPreparedRoute.CUSTOM_PROGRAM);
        assertRejected(values(new long[] {2L}, false, false, false, DataType.FLOAT32),
                program(false, List.of(0), 0L), MetalPreparedRoute.CUSTOM_PROGRAM);
        assertRejected(values,
                program(false, List.of(0), 1L), MetalPreparedRoute.CUSTOM_PROGRAM);
        assertRejected(values(new long[] {1L}, false, false, false, DataType.FLOAT32),
                program(false, List.of(1), 0L), MetalPreparedRoute.CUSTOM_PROGRAM);
        assertRejected(values(new long[] {1L}, true, false, false, DataType.FLOAT32),
                valid, MetalPreparedRoute.CUSTOM_PROGRAM);
        assertRejected(values(new long[] {1L}, false, false, true, DataType.FLOAT32),
                valid, MetalPreparedRoute.CUSTOM_PROGRAM);
        assertRejected(values(new long[] {1L}, false, false, false, DataType.FLOAT64),
                valid, MetalPreparedRoute.CUSTOM_PROGRAM);
        assertRejected(values(new long[] {1L}, false, true, false, DataType.FLOAT32),
                valid, MetalPreparedRoute.CUSTOM_PROGRAM);
    }

    @Test
    void customRunRejectsAliasedValueBuffersBeforeMutation() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        MetalNativeApi.Handle buffer = null;
        try {
            context = api.createContext();
            executable = api.createProgramExecutable(context,
            values(new long[] {1L}, false, false, false, DataType.FLOAT32),
            program(false, List.of(0), 0L),
            new int[] {0},
            new int[] {1},
            MetalPreparedRoute.CUSTOM_PROGRAM);
            buffer = api.createBuffer(context, Integer.BYTES);
            upload(api, buffer, 0x3fc00000);
            MetalNativeApi.Handle retainedExecutable = executable;
            MetalNativeApi.Handle retainedBuffer = buffer;
            assertThrows(MetalNativeApi.NativeFailure.class, () -> {
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment table = arena.allocate(ADDRESS, 2);
                    table.setAtIndex(ADDRESS, 0, retainedBuffer.carrier());
                    table.setAtIndex(ADDRESS, 1, retainedBuffer.carrier());
                    MemorySegment outputs = arena.allocate(ADDRESS);
                    outputs.setAtIndex(ADDRESS, 0, retainedBuffer.carrier());
                    api.runExecutable(retainedExecutable, 2, table, 1, outputs);
                }
            });
            assertEquals(0x3fc00000, download(api, buffer));
        } finally {
            if (buffer != null) api.releaseBuffer(buffer);
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static void assertRejected(
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            MetalMpsGraphProgram program,
            MetalPreparedRoute route) {
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        values, program, new int[] {0}, new int[] {1}, route));
    }

    private static int execute(Path library, int word, boolean keepDimensions) {
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createProgramExecutable(context,
            values(new long[] {1L}, false, keepDimensions, false, DataType.FLOAT32),
            program(keepDimensions, List.of(0), 0L),
            new int[] {0},
            new int[] {1},
            MetalPreparedRoute.CUSTOM_PROGRAM);
            buffers.add(api.createBuffer(context, Integer.BYTES));
            buffers.add(api.createBuffer(context, Integer.BYTES));
            upload(api, buffers.getFirst(), word);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment table = arena.allocate(ADDRESS, 2);
                table.setAtIndex(ADDRESS, 0, buffers.get(0).carrier());
                table.setAtIndex(ADDRESS, 1, buffers.get(1).carrier());
                MemorySegment outputs = arena.allocate(ADDRESS);
                outputs.setAtIndex(ADDRESS, 0, buffers.get(1).carrier());
                api.runExecutable(executable, 2, table, 1, outputs);
            }
            return download(api, buffers.get(1));
        } finally {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static MetalMpsGraphProgram program(
            boolean keepDimensions, List<Integer> axes, long correction) {
        return new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.statisticalReduction(
                MetalMpsGraphProgram.NodeKind.VARIANCE,
                0,
                1,
                axes,
                keepDimensions,
                correction)));
    }

    private static List<MetalMpsGraphProgram.ValueDescriptor> values(
            long[] inputShape,
            boolean inputGradient,
            boolean keepDimensions,
            boolean outputGradient,
            DataType inputType) {
        return List.of(
                new MetalMpsGraphProgram.ValueDescriptor(
                        inputType, inputShape, inputGradient),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32,
                        keepDimensions ? new long[] {1L} : new long[0],
                        outputGradient));
    }

    private static void upload(MetalNativeApi api, MetalNativeApi.Handle buffer, int value) {
        byte[] bytes = ByteBuffer.allocate(Integer.BYTES)
                .order(ByteOrder.nativeOrder())
                .putInt(value)
                .array();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes.length, 1L);
            MemorySegment.copy(bytes, 0, source, JAVA_BYTE, 0L, bytes.length);
            api.upload(buffer, 0L, source, bytes.length);
        }
    }

    private static int download(MetalNativeApi api, MetalNativeApi.Handle buffer) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(Integer.BYTES, Integer.BYTES);
            api.download(buffer, 0L, target, Integer.BYTES);
            return ByteBuffer.wrap(target.toArray(JAVA_BYTE))
                    .order(ByteOrder.nativeOrder())
                    .getInt();
        }
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }
}
