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
            executable = api.createMpsGraphExecutable(
                    context,
                    NumericalProfile.STRICT_IEEE,
                    ranks(shapes),
                    dimensions(shapes),
                    program,
                    new int[] {0, 1, 3},
                    new int[] {2, 4});

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
