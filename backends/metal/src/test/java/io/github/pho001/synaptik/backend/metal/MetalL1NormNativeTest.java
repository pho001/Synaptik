package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertEquals;
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

class MetalL1NormNativeTest {
    @Test
    void customKernelExecutesSingletonAbsAndOrdinalLeftFoldForBothOutputForms() {
        Path library = configuredLibrary();
        assertEquals(0x7f812345, execute(library, new int[] {0xff812345}, false));
        assertEquals(0x00000001, execute(library, new int[] {0x80000001}, true));
        assertEquals(
                0x4b800000,
                execute(library, new int[] {0xcb800000, 0xbf800000, 0xbf800000}, false));
        assertEquals(
                0x7f800000,
                execute(library, new int[] {0xff800000, 0x3f800000}, true));
    }

    @Test
    void customAbiRejectsEveryAlternateReductionFormGradientTypeAndGeometry() {
        var valid = program(4, false, MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, List.of(0));
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                NumericalProfile.ACCELERATOR,
                values(4, false, false, DataType.FLOAT32),
                valid,
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.CUSTOM_PROGRAM);

        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        values(4, false, false, DataType.FLOAT32),
                        valid,
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.MPSGRAPH));

        for (var malformed : List.of(
                program(4, false, MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS, List.of(0)),
                program(4, false, MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, List.of()))) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                            NumericalProfile.ACCELERATOR,
                            values(4, false, false, DataType.FLOAT32),
                            malformed,
                            new int[] {0},
                            new int[] {1},
                            MetalPreparedRoute.CUSTOM_PROGRAM));
        }
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        values(4, true, false, DataType.FLOAT32),
                        valid,
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.CUSTOM_PROGRAM));
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        values(4, false, false, DataType.FLOAT64),
                        valid,
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.CUSTOM_PROGRAM));
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        values(0xffff_ffffL / Float.BYTES + 1L, false, false, DataType.FLOAT32),
                        program(
                                0xffff_ffffL / Float.BYTES + 1L,
                                false,
                                MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                                List.of(0)),
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.CUSTOM_PROGRAM));
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
            var program = program(1, false, MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, List.of(0));
            executable = api.createProgramExecutable(context,
            NumericalProfile.ACCELERATOR,
            values(1, false, false, DataType.FLOAT32),
            program,
            new int[] {0},
            new int[] {1},
            MetalPreparedRoute.CUSTOM_PROGRAM);
            buffer = api.createBuffer(context, Integer.BYTES);
            upload(api, buffer, 0xc1200000);
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
            assertEquals(0xc1200000, download(api, buffer));
        } finally {
            if (buffer != null) api.releaseBuffer(buffer);
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static int execute(Path library, int[] words, boolean keepDimensions) {
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            var program = program(
                    words.length,
                    keepDimensions,
                    MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                    List.of(0));
            executable = api.createProgramExecutable(context,
            NumericalProfile.ACCELERATOR,
            values(words.length, false, keepDimensions, DataType.FLOAT32),
            program,
            new int[] {0},
            new int[] {1},
            MetalPreparedRoute.CUSTOM_PROGRAM);
            buffers.add(api.createBuffer(context, (long) words.length * Integer.BYTES));
            buffers.add(api.createBuffer(context, Integer.BYTES));
            upload(api, buffers.get(0), words);
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
            long extent,
            boolean keepDimensions,
            MetalMpsGraphProgram.ReductionForm form,
            List<Integer> axes) {
        return new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.reduction(
                MetalMpsGraphProgram.NodeKind.L1_NORM,
                0,
                1,
                form,
                axes,
                keepDimensions)));
    }

    private static List<MetalMpsGraphProgram.ValueDescriptor> values(
            long extent,
            boolean inputGradient,
            boolean keepDimensions,
            DataType inputType) {
        return List.of(
                new MetalMpsGraphProgram.ValueDescriptor(
                        inputType, new long[] {extent}, inputGradient),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32,
                        keepDimensions ? new long[] {1L} : new long[0],
                        false));
    }

    private static void upload(MetalNativeApi api, MetalNativeApi.Handle buffer, int... values) {
        byte[] bytes = new byte[values.length * Integer.BYTES];
        ByteBuffer encoded = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
        for (int value : values) encoded.putInt(value);
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
