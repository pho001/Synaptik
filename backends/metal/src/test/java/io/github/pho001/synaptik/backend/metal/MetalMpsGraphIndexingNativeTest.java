package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalMpsGraphIndexingNativeTest {
    private static final int[] DATA_BITS = {
        0x00000000, 0x80000000, 0x00000001, 0x7fc12345,
        0xffc54321, 0x7f800000, 0x80000001, 0x3f800000
    };
    private static final byte SENTINEL = (byte) 0x5a;

    @Test
    void realGatherMovesRawFloat32BitsAndOneHotWritesExactBoolBytesBeforeBoundsFailures() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var inputs = new ArrayList<MetalNativeApi.Handle>();
        var outputs = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.gather(0, 1, 2, 1),
                    MetalMpsGraphProgram.Node.oneHot(3, 4, 4)));
            long[][] shapes = {{2, 4}, {3}, {2, 3}, {3}, {3, 4}};
            executable = api.createMpsGraphExecutable(context, NumericalProfile.STRICT_IEEE, MetalTestProgram.descriptors(ranks(shapes), dimensions(shapes), program), program, new int[] {0, 1, 3}, new int[] {2, 4}, MetalPreparedRoute.MPSGRAPH);

            MetalNativeApi.Handle data = api.createBuffer(context, 8L * Integer.BYTES);
            MetalNativeApi.Handle gatherIndices = api.createBuffer(context, 3L * Integer.BYTES);
            MetalNativeApi.Handle oneHotIndices = api.createBuffer(context, 3L * Integer.BYTES);
            MetalNativeApi.Handle gathered = api.createBuffer(context, 6L * Integer.BYTES);
            MetalNativeApi.Handle oneHot = api.createBuffer(context, 12L);
            inputs.add(data);
            inputs.add(gatherIndices);
            inputs.add(oneHotIndices);
            outputs.add(gathered);
            outputs.add(oneHot);

            uploadInts(api, data, DATA_BITS);
            uploadInts(api, gatherIndices, new int[] {3, 0, 2});
            uploadInts(api, oneHotIndices, new int[] {2, 0, 3});
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size());
                MemorySegment outputAddresses = arena.allocate(ADDRESS, outputs.size());
                for (int index = 0; index < inputs.size(); index++) {
                    inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
                }
                for (int index = 0; index < outputs.size(); index++) {
                    outputAddresses.setAtIndex(ADDRESS, index, outputs.get(index).carrier());
                }

                api.runExecutable(
                        executable, inputs.size(), inputAddresses,
                        outputs.size(), outputAddresses);
                assertArrayEquals(new int[] {
                    0x7fc12345, 0x00000000, 0x00000001,
                    0x3f800000, 0xffc54321, 0x80000001
                }, downloadInts(api, gathered, 6));
                assertArrayEquals(new byte[] {
                    0, 0, 1, 0,
                    1, 0, 0, 0,
                    0, 0, 0, 1
                }, downloadBytes(api, oneHot, 12));
                assertArrayEquals(DATA_BITS, downloadInts(api, data, DATA_BITS.length));

                uploadInts(api, gatherIndices, new int[] {3, 4, 0});
                uploadInts(api, oneHotIndices, new int[] {2, 0, 3});
                fill(api, gathered, 6L * Integer.BYTES, SENTINEL);
                fill(api, oneHot, 12L, SENTINEL);
                assertBoundsFailure(api, executable, inputAddresses, outputAddresses);
                assertFilled(downloadBytes(api, gathered, 6 * Integer.BYTES), SENTINEL);
                assertFilled(downloadBytes(api, oneHot, 12), SENTINEL);

                uploadInts(api, gatherIndices, new int[] {3, 0, 2});
                uploadInts(api, oneHotIndices, new int[] {2, -1, 3});
                assertBoundsFailure(api, executable, inputAddresses, outputAddresses);
                assertFilled(downloadBytes(api, gathered, 6 * Integer.BYTES), SENTINEL);
                assertFilled(downloadBytes(api, oneHot, 12), SENTINEL);
            }
        } finally {
            for (int index = outputs.size(); index-- > 0;) api.releaseBuffer(outputs.get(index));
            for (int index = inputs.size(); index-- > 0;) api.releaseBuffer(inputs.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    @Test
    void realReplacementScatterMovesRawBitsAndRejectsBoundsAndDuplicatesBeforeWrites() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var inputs = new ArrayList<MetalNativeApi.Handle>();
        var outputs = new ArrayList<MetalNativeApi.Handle>();
        int[] dataBits = {
            0x00000000, 0x80000000, 0x00000001,
            0x7fa12345, 0xffa54321, 0x3f800000
        };
        int[] updateBits = {0x7fa22222, 0xffa33333, 0x80000001, 0x7f800000};
        try {
            context = api.createContext();
            MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.scatterElements(0, 1, 2, 3, 1)));
            long[][] shapes = {{2, 3}, {2, 2}, {2, 2}, {2, 3}};
            executable = api.createMpsGraphExecutable(context, NumericalProfile.STRICT_IEEE, MetalTestProgram.descriptors(ranks(shapes), dimensions(shapes), program), program, new int[] {0, 1, 2}, new int[] {3}, MetalPreparedRoute.MPSGRAPH);

            MetalNativeApi.Handle data = api.createBuffer(context, 6L * Integer.BYTES);
            MetalNativeApi.Handle indices = api.createBuffer(context, 4L * Integer.BYTES);
            MetalNativeApi.Handle updates = api.createBuffer(context, 4L * Integer.BYTES);
            MetalNativeApi.Handle output = api.createBuffer(context, 6L * Integer.BYTES);
            inputs.add(data);
            inputs.add(indices);
            inputs.add(updates);
            outputs.add(output);
            uploadInts(api, data, dataBits);
            uploadInts(api, indices, new int[] {2, 0, 1, 2});
            uploadInts(api, updates, updateBits);

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size());
                MemorySegment outputAddresses = arena.allocate(ADDRESS, outputs.size());
                for (int index = 0; index < inputs.size(); index++) {
                    inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
                }
                outputAddresses.setAtIndex(ADDRESS, 0, output.carrier());

                api.runExecutable(executable, 3, inputAddresses, 1, outputAddresses);
                assertArrayEquals(new int[] {
                    updateBits[1], dataBits[1], updateBits[0],
                    dataBits[3], updateBits[2], updateBits[3]
                }, downloadInts(api, output, 6));
                assertArrayEquals(dataBits, downloadInts(api, data, 6));
                assertArrayEquals(new int[] {2, 0, 1, 2}, downloadInts(api, indices, 4));
                assertArrayEquals(updateBits, downloadInts(api, updates, 4));

                uploadInts(api, indices, new int[] {2, 3, 1, 2});
                fill(api, output, 6L * Integer.BYTES, SENTINEL);
                assertRangeFailure(api, executable, 3, inputAddresses, 1, outputAddresses);
                assertFilled(downloadBytes(api, output, 6 * Integer.BYTES), SENTINEL);
                assertArrayEquals(dataBits, downloadInts(api, data, 6));
                assertArrayEquals(new int[] {2, 3, 1, 2}, downloadInts(api, indices, 4));
                assertArrayEquals(updateBits, downloadInts(api, updates, 4));

                uploadInts(api, indices, new int[] {1, 1, 1, 2});
                assertRangeFailure(api, executable, 3, inputAddresses, 1, outputAddresses);
                assertFilled(downloadBytes(api, output, 6 * Integer.BYTES), SENTINEL);
                assertArrayEquals(dataBits, downloadInts(api, data, 6));
                assertArrayEquals(new int[] {1, 1, 1, 2}, downloadInts(api, indices, 4));
                assertArrayEquals(updateBits, downloadInts(api, updates, 4));
            }
        } finally {
            for (int index = outputs.size(); index-- > 0;) api.releaseBuffer(outputs.get(index));
            for (int index = inputs.size(); index-- > 0;) api.releaseBuffer(inputs.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    @Test
    void realUnfoldAxisMapsOverlapAndTailWithExactFloat32BitsWithoutMutatingInput() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        MetalNativeApi.Handle input = null;
        MetalNativeApi.Handle output = null;
        int[] inputBits = {
            0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x7f800000, 0xff800000,
            0x7fc12345, 0xffc54321, 0x7f812345, 0xff854321, 0x3f800000, 0xbf800000
        };
        int[] expected = {
            inputBits[0], inputBits[1], inputBits[2],
            inputBits[2], inputBits[3], inputBits[4],
            inputBits[6], inputBits[7], inputBits[8],
            inputBits[8], inputBits[9], inputBits[10]
        };
        try {
            context = api.createContext();
            long[][] shapes = {{2, 6}, {2, 2, 3}};
            executable = api.createMpsGraphExecutable(context, NumericalProfile.STRICT_IEEE, MetalTestProgram.descriptors(ranks(shapes), dimensions(shapes), new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.unfoldAxis(0, 1, 1, 3, 2)))), new MetalMpsGraphProgram(List.of(
            MetalMpsGraphProgram.Node.unfoldAxis(0, 1, 1, 3, 2))), new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH);
            input = api.createBuffer(context, (long) inputBits.length * Integer.BYTES);
            output = api.createBuffer(context, (long) expected.length * Integer.BYTES);
            uploadInts(api, input, inputBits);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddress = arena.allocate(ADDRESS);
                MemorySegment outputAddress = arena.allocate(ADDRESS);
                inputAddress.setAtIndex(ADDRESS, 0, input.carrier());
                outputAddress.setAtIndex(ADDRESS, 0, output.carrier());
                api.runExecutable(executable, 1, inputAddress, 1, outputAddress);
            }
            assertArrayEquals(expected, downloadInts(api, output, expected.length));
            assertArrayEquals(inputBits, downloadInts(api, input, inputBits.length));
        } finally {
            if (output != null) api.releaseBuffer(output);
            if (input != null) api.releaseBuffer(input);
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static void assertRangeFailure(
            MetalNativeApi api,
            MetalNativeApi.Handle executable,
            int inputCount,
            MemorySegment inputs,
            int outputCount,
            MemorySegment outputs) {
        MetalNativeApi.NativeFailure failure = assertThrows(
                MetalNativeApi.NativeFailure.class,
                () -> api.runExecutable(
                        executable, inputCount, inputs, outputCount, outputs));
        assertEquals(MetalNativeApi.Status.RANGE_OUT_OF_BOUNDS, failure.status());
    }

    private static void assertBoundsFailure(
            MetalNativeApi api,
            MetalNativeApi.Handle executable,
            MemorySegment inputs,
            MemorySegment outputs) {
        MetalNativeApi.NativeFailure failure = assertThrows(
                MetalNativeApi.NativeFailure.class,
                () -> api.runExecutable(executable, 3, inputs, 2, outputs));
        assertEquals(MetalNativeApi.Status.RANGE_OUT_OF_BOUNDS, failure.status());
    }

    private static void uploadInts(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int[] values) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(
                    Math.multiplyExact((long) values.length, Integer.BYTES), Integer.BYTES);
            for (int index = 0; index < values.length; index++) {
                source.setAtIndex(JAVA_INT, index, values[index]);
            }
            api.upload(buffer, 0L, source, source.byteSize());
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

    private static int[] downloadInts(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int count) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate((long) count * Integer.BYTES, Integer.BYTES);
            api.download(buffer, 0L, target, target.byteSize());
            int[] values = new int[count];
            for (int index = 0; index < count; index++) {
                values[index] = target.getAtIndex(JAVA_INT, index);
            }
            return values;
        }
    }

    private static byte[] downloadBytes(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int count) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(count, 1L);
            api.download(buffer, 0L, target, count);
            return target.toArray(JAVA_BYTE);
        }
    }

    private static void assertFilled(byte[] values, byte expected) {
        for (byte value : values) assertEquals(expected, value);
    }

    private static int[] ranks(long[][] shapes) {
        int[] result = new int[shapes.length];
        for (int index = 0; index < shapes.length; index++) result[index] = shapes[index].length;
        return result;
    }

    private static long[] dimensions(long[][] shapes) {
        long[] result = new long[Math.multiplyExact(shapes.length, 16)];
        for (int value = 0; value < shapes.length; value++) {
            System.arraycopy(shapes[value], 0, result, value * 16, shapes[value].length);
        }
        return result;
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }
}
