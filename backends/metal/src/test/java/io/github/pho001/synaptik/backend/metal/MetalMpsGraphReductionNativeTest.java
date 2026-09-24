package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalMpsGraphReductionNativeTest {
    @Test
    void schemaSevenEncodesClosedReductionFormsAndPreflightKeepsStrictClosed() {
        var sum = MetalMpsGraphProgram.Node.reduction(
                MetalMpsGraphProgram.NodeKind.SUM,
                2,
                5,
                MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                List.of(2, 0),
                true);
        ByteBuffer record = ByteBuffer.wrap(
                new MetalMpsGraphProgram(List.of(sum)).encodedNodeRecords())
                .order(ByteOrder.BIG_ENDIAN);
        assertEquals(7, MetalMpsGraphProgram.SCHEMA_VERSION);
        assertEquals(13, record.getInt());
        assertEquals(4, record.getInt());
        assertEquals(2, record.getInt());
        assertEquals(-1, record.getInt());
        assertEquals(5, record.getInt());
        assertEquals(2, record.getInt());
        assertEquals(3, record.getInt());
        assertEquals(1, record.getInt());
        assertEquals(2L, record.getLong());
        assertEquals(0L, record.getLong());
        while (record.hasRemaining()) assertEquals(0L, record.getLong());

        var full = new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.reduction(
                MetalMpsGraphProgram.NodeKind.MEAN,
                0,
                1,
                MetalMpsGraphProgram.ReductionForm.FULL,
                List.of(),
                false)));
        int[] ranks = {2, 0};
        long[] dimensions = dimensions(new long[][] {{2, 2}, {}});
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.ACCELERATOR,
                ranks,
                dimensions,
                full,
                new int[] {0},
                new int[] {1});
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.STRICT_IEEE,
                        ranks,
                        dimensions,
                        full,
                        new int[] {0},
                        new int[] {1}));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        new int[] {0, 0},
                        dimensions(new long[][] {{}, {}}),
                        full,
                        new int[] {0},
                        new int[] {1}));

        var wrongShape = new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.reduction(
                MetalMpsGraphProgram.NodeKind.SUM,
                0,
                1,
                MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                List.of(1),
                false)));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        new int[] {2, 1},
                        dimensions(new long[][] {{2, 2}, {1}}),
                        wrongShape,
                        new int[] {0},
                        new int[] {1}));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.MEAN,
                        0,
                        1,
                        MetalMpsGraphProgram.ReductionForm.FULL,
                        List.of(),
                        true));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.sumToShape(0, 1, new long[] {2, 0}));
    }

    @Test
    void realNativeGraphExecutesScalarMeanIdentityAndSumToShapeWithDirectTargets() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        MetalNativeApi api = MetalNativeApi.open(Path.of(configured).toAbsolutePath().normalize());
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            var program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.SUM,
                            0,
                            1,
                            MetalMpsGraphProgram.ReductionForm.FULL,
                            List.of(),
                            false),
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.MEAN,
                            0,
                            2,
                            MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                            List.of(1),
                            true),
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.SUM,
                            0,
                            3,
                            MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                            List.of(),
                            false),
                    MetalMpsGraphProgram.Node.sumToShape(0, 4, new long[] {1, 2})));
            int[] ranks = {2, 0, 2, 2, 2};
            long[] dimensions = dimensions(new long[][] {
                {2, 2}, {}, {2, 1}, {2, 2}, {1, 2}
            });
            executable = api.createMpsGraphExecutable(
                    context,
                    NumericalProfile.ACCELERATOR,
                    ranks,
                    dimensions,
                    program,
                    new int[] {0},
                    new int[] {4, 1, 3, 2});
            int[][] initial = {
                bits(1.0f, 2.0f, 3.0f, 4.0f),
                new int[2],
                new int[1],
                new int[4],
                new int[2]
            };
            for (int[] values : initial) {
                MetalNativeApi.Handle buffer = api.createBuffer(
                        context, Math.multiplyExact((long) values.length, Integer.BYTES));
                buffers.add(buffer);
                upload(api, buffer, values);
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputs = arena.allocate(ADDRESS, 1);
                inputs.setAtIndex(ADDRESS, 0, buffers.get(0).carrier());
                MemorySegment outputs = arena.allocate(ADDRESS, 4);
                outputs.setAtIndex(ADDRESS, 0, buffers.get(1).carrier());
                outputs.setAtIndex(ADDRESS, 1, buffers.get(2).carrier());
                outputs.setAtIndex(ADDRESS, 2, buffers.get(3).carrier());
                outputs.setAtIndex(ADDRESS, 3, buffers.get(4).carrier());
                for (int repetition = 0; repetition < 2; repetition++) {
                    api.runExecutable(executable, 1, inputs, 4, outputs);
                    assertArrayEquals(bits(4.0f, 6.0f), download(api, buffers.get(1), 2));
                    assertArrayEquals(bits(10.0f), download(api, buffers.get(2), 1));
                    assertArrayEquals(initial[0], download(api, buffers.get(3), 4));
                    assertArrayEquals(bits(1.5f, 3.5f), download(api, buffers.get(4), 2));
                    assertArrayEquals(initial[0], download(api, buffers.get(0), 4));
                }
            }
        } finally {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    @Test
    void oneTermSumAndEqualShapeSumToShapePreserveRawBits() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        MetalNativeApi api = MetalNativeApi.open(Path.of(configured).toAbsolutePath().normalize());
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            var program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.SUM,
                            0,
                            1,
                            MetalMpsGraphProgram.ReductionForm.FULL,
                            List.of(),
                            false),
                    MetalMpsGraphProgram.Node.sumToShape(0, 2, new long[] {1})));
            executable = api.createMpsGraphExecutable(
                    context,
                    NumericalProfile.ACCELERATOR,
                    new int[] {1, 0, 1},
                    dimensions(new long[][] {{1}, {}, {1}}),
                    program,
                    new int[] {0},
                    new int[] {1, 2});
            int signalingNaN = 0x7f800123;
            for (int count : new int[] {1, 1, 1}) {
                buffers.add(api.createBuffer(context, (long) count * Integer.BYTES));
            }
            upload(api, buffers.get(0), new int[] {signalingNaN});
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputs = arena.allocate(ADDRESS, 1);
                inputs.setAtIndex(ADDRESS, 0, buffers.get(0).carrier());
                MemorySegment outputs = arena.allocate(ADDRESS, 2);
                outputs.setAtIndex(ADDRESS, 0, buffers.get(1).carrier());
                outputs.setAtIndex(ADDRESS, 1, buffers.get(2).carrier());
                api.runExecutable(executable, 1, inputs, 2, outputs);
            }
            assertArrayEquals(new int[] {signalingNaN}, download(api, buffers.get(1), 1));
            assertArrayEquals(new int[] {signalingNaN}, download(api, buffers.get(2), 1));
        } finally {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static int[] bits(float... values) {
        int[] result = new int[values.length];
        for (int index = 0; index < values.length; index++) {
            result[index] = Float.floatToRawIntBits(values[index]);
        }
        return result;
    }

    private static void upload(MetalNativeApi api, MetalNativeApi.Handle buffer, int[] bits) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(
                    Math.multiplyExact((long) bits.length, Integer.BYTES), Integer.BYTES);
            for (int index = 0; index < bits.length; index++) {
                source.setAtIndex(JAVA_INT, index, bits[index]);
            }
            api.upload(buffer, 0L, source, source.byteSize());
        }
    }

    private static int[] download(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int count) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(
                    Math.multiplyExact((long) count, Integer.BYTES), Integer.BYTES);
            api.download(buffer, 0L, target, target.byteSize());
            int[] result = new int[count];
            for (int index = 0; index < count; index++) {
                result[index] = target.getAtIndex(JAVA_INT, index);
            }
            return result;
        }
    }

    private static long[] dimensions(long[][] shapes) {
        long[] dimensions = new long[Math.multiplyExact(shapes.length, 16)];
        for (int value = 0; value < shapes.length; value++) {
            System.arraycopy(shapes[value], 0, dimensions, value * 16, shapes[value].length);
        }
        return dimensions;
    }
}
