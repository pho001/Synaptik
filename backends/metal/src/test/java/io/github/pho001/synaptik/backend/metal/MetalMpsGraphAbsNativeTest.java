package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class MetalMpsGraphAbsNativeTest {
    private static final int[] INPUT_BITS = {
        0x00000000, 0x80000000,
        0x00000001, 0x80000001,
        0x00000002, 0x80000002,
        0x007fffff, 0x807fffff,
        0x00800000, 0x80800000,
        0x00800001, 0x80800001,
        0x3f800000, 0xbf800000,
        0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000,
        0x7fc12345, 0xffc54321,
        0x7f812345, 0xff854321
    };

    @Test
    void JavaPreflightClosesAbsProfileAndRejectsSelectedMpsGraphViewNodes() {

        int[] ranks = {1, 1};
        long[] dimensions = dimensions(2, INPUT_BITS.length);
        MetalMpsGraphProgram unary = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.abs(0, 1)));
        for (NumericalProfile profile : NumericalProfile.values()) {
            MetalNativeApi.MpsGraphExecutableAbi.validateCreate(profile, MetalTestProgram.descriptors(ranks, dimensions, unary), unary, new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH);
        }

        MetalMpsGraphProgram viewToAbs = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE,
                        0,
                        1,
                        new long[] {INPUT_BITS.length}),
                MetalMpsGraphProgram.Node.abs(1, 2)));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(NumericalProfile.STRICT_IEEE, MetalTestProgram.descriptors(new int[] {1, 1, 1}, dimensions(3, INPUT_BITS.length), viewToAbs), viewToAbs, new int[] {0}, new int[] {2}, MetalPreparedRoute.MPSGRAPH));

        MetalMpsGraphProgram canonicalizedAbs = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE,
                        0,
                        1,
                        new long[] {INPUT_BITS.length}),
                MetalMpsGraphProgram.Node.contiguous(1, 2),
                MetalMpsGraphProgram.Node.abs(2, 3)));
    assertThrows(
        IllegalArgumentException.class,
        () ->
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(NumericalProfile.STRICT_IEEE, MetalTestProgram.descriptors(new int[] {1, 1, 1, 1}, dimensions(4, INPUT_BITS.length), canonicalizedAbs), canonicalizedAbs, new int[] {0}, new int[] {3}, MetalPreparedRoute.MPSGRAPH));
    }

    @Test
    void realAbsIsExactAcrossProfilesTopologyReuseConcurrencyContextsAndClose() throws Exception {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle firstContext = null;
        MetalNativeApi.Handle secondContext = null;
        try {
            firstContext = api.createContext();
            secondContext = api.createContext();
            for (NumericalProfile profile : NumericalProfile.values()) {
                runProfile(api, firstContext, profile, 2);
                runProfile(api, secondContext, profile, 1);
            }

            MetalNativeApi.Handle concurrentContext = firstContext;
            var executor = Executors.newFixedThreadPool(4);
            try {
                var calls = new ArrayList<java.util.concurrent.Callable<Boolean>>();
                for (int call = 0; call < 12; call++) {
                    NumericalProfile profile = call % 2 == 0
                            ? NumericalProfile.STRICT_IEEE
                            : NumericalProfile.ACCELERATOR;
                    calls.add(() -> {
                        runProfile(api, concurrentContext, profile, 1);
                        return true;
                    });
                }
                for (var future : executor.invokeAll(calls)) {
                    assertTrue(future.get(30, TimeUnit.SECONDS));
                }
            } finally {
                executor.shutdownNow();
                assertTrue(executor.awaitTermination(30, TimeUnit.SECONDS));
            }
        } finally {
            if (secondContext != null) api.releaseContext(secondContext);
            if (firstContext != null) api.releaseContext(firstContext);
            api.close();
        }

        MetalDeviceContext device = MetalDeviceContext.open(library);
        MetalBufferRepresentation buffer = device.createBuffer(Integer.BYTES);
        buffer.close();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(JAVA_INT);
            assertThrows(IllegalStateException.class,
                    () -> buffer.upload(0L, source, 0L, Integer.BYTES));
        }
        device.close();
        assertTrue(device.isClosed());
        assertThrows(IllegalStateException.class, () -> device.createBuffer(Integer.BYTES));
    }

    private static void runProfile(
            MetalNativeApi api,
            MetalNativeApi.Handle context,
            NumericalProfile profile,
            int repetitions) {
        boolean accelerator = profile == NumericalProfile.ACCELERATOR;
        MetalMpsGraphProgram program = accelerator
                ? new MetalMpsGraphProgram(List.of(
                        MetalMpsGraphProgram.Node.binary(
                                MetalMpsGraphProgram.NodeKind.ADD, 0, 1, 2),
                        MetalMpsGraphProgram.Node.abs(2, 3),
                        MetalMpsGraphProgram.Node.abs(0, 4)))
                : new MetalMpsGraphProgram(List.of(
                        MetalMpsGraphProgram.Node.abs(0, 1),
                        MetalMpsGraphProgram.Node.neg(1, 2),
                        MetalMpsGraphProgram.Node.abs(2, 3),
                        MetalMpsGraphProgram.Node.abs(0, 4)));
        int valueCount = 5;
        int[] feeds = accelerator ? new int[] {0, 1} : new int[] {0};
        int[] targets = accelerator ? new int[] {2, 3, 4} : new int[] {1, 3, 4};
        MetalNativeApi.Handle executable = null;
        var inputs = new ArrayList<MetalNativeApi.Handle>();
        var outputs = new ArrayList<MetalNativeApi.Handle>();
        long byteCount = Math.multiplyExact((long) INPUT_BITS.length, Integer.BYTES);
        try {
            executable = api.createMpsGraphExecutable(context, profile, MetalTestProgram.descriptors(new int[] {1, 1, 1, 1, 1}, dimensions(valueCount, INPUT_BITS.length), program), program, feeds, targets, MetalPreparedRoute.MPSGRAPH);
            MetalNativeApi.Handle input = api.createBuffer(context, byteCount);
            inputs.add(input);
            upload(api, input, INPUT_BITS);
            if (accelerator) {
                MetalNativeApi.Handle zeros = api.createBuffer(context, byteCount);
                inputs.add(zeros);
                upload(api, zeros, new int[INPUT_BITS.length]);
            }
            for (int target : targets) {
                outputs.add(api.createBuffer(context, byteCount));
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size());
                MemorySegment outputAddresses = arena.allocate(ADDRESS, outputs.size());
                for (int index = 0; index < inputs.size(); index++) {
                    inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
                }
                for (int index = 0; index < outputs.size(); index++) {
                    outputAddresses.setAtIndex(ADDRESS, index, outputs.get(index).carrier());
                }
                MetalNativeApi.Handle runExecutable = executable;
                if (accelerator) {
                    inputAddresses.setAtIndex(
                            ADDRESS, 1, inputs.getFirst().carrier());
                    api.runExecutable(
                            runExecutable,
                            inputs.size(),
                            inputAddresses,
                            outputs.size(),
                            outputAddresses);
                    inputAddresses.setAtIndex(
                            ADDRESS, 1, inputs.get(1).carrier());
                }
                for (MetalNativeApi.Handle output : outputs) {
                    upload(api, output, INPUT_BITS);
                }
                outputAddresses.setAtIndex(
                        ADDRESS, 0, inputs.getFirst().carrier());
                MetalNativeApi.NativeFailure inputOutputAlias = assertThrows(
                        MetalNativeApi.NativeFailure.class,
                        () -> api.runExecutable(
                                runExecutable,
                                inputs.size(),
                                inputAddresses,
                                outputs.size(),
                                outputAddresses));
                assertEquals(
                        MetalNativeApi.Status.INCOMPATIBLE_RESOURCE,
                        inputOutputAlias.status());
                assertArrayEquals(
                        INPUT_BITS,
                        download(api, inputs.getFirst()),
                        "input/output alias rejection must precede input mutation");
                for (MetalNativeApi.Handle output : outputs) {
                    assertArrayEquals(
                            INPUT_BITS,
                            download(api, output),
                            "input/output alias rejection must precede target mutation");
                }
                outputAddresses.setAtIndex(
                        ADDRESS, 0, outputs.getFirst().carrier());
                outputAddresses.setAtIndex(ADDRESS, 1, outputs.getFirst().carrier());
                MetalNativeApi.NativeFailure aliasFailure = assertThrows(
                        MetalNativeApi.NativeFailure.class,
                        () -> api.runExecutable(
                                runExecutable,
                                inputs.size(),
                                inputAddresses,
                                outputs.size(),
                                outputAddresses));
                assertEquals(
                        MetalNativeApi.Status.INCOMPATIBLE_RESOURCE,
                        aliasFailure.status());
                for (MetalNativeApi.Handle output : outputs) {
                    assertArrayEquals(INPUT_BITS, download(api, output),
                            "output alias rejection must precede mutation");
                }
                outputAddresses.setAtIndex(ADDRESS, 1, outputs.get(1).carrier());
                for (int repetition = 0; repetition < repetitions; repetition++) {
                    api.runExecutable(
                            executable,
                            inputs.size(),
                            inputAddresses,
                            outputs.size(),
                            outputAddresses);
                    int[][] actual = new int[outputs.size()][];
                    for (int output = 0; output < outputs.size(); output++) {
                        actual[output] = download(api, outputs.get(output));
                    }
                    assertExactMagnitude(INPUT_BITS, actual[2], profile + " direct ABS");
                    if (accelerator) {
                        assertExactMagnitude(actual[0], actual[1],
                                "accelerator binary-to-ABS");
                    } else {
                        assertExactMagnitude(INPUT_BITS, actual[0], "strict first ABS target");
                        assertExactMagnitude(INPUT_BITS, actual[1], "strict ABS/NEG/ABS");
                    }
                    assertArrayEquals(INPUT_BITS, download(api, input),
                            profile + " input preservation");
                }
            }
        } finally {
            for (int index = outputs.size(); index-- > 0;) api.releaseBuffer(outputs.get(index));
            for (int index = inputs.size(); index-- > 0;) api.releaseBuffer(inputs.get(index));
            if (executable != null) api.releaseExecutable(executable);
        }
    }

    private static void assertExactMagnitude(int[] inputs, int[] outputs, String label) {
        assertEquals(inputs.length, outputs.length, label);
        for (int index = 0; index < inputs.length; index++) {
            if (isNaN(inputs[index])) {
                assertTrue(isNaN(outputs[index]), label + " NaN lane " + index);
            } else {
                assertEquals(inputs[index] & 0x7fffffff, outputs[index],
                        label + " lane " + index);
            }
        }
    }

    private static boolean isNaN(int bits) {
        return (bits & 0x7f800000) == 0x7f800000
                && (bits & 0x007fffff) != 0;
    }

    private static void upload(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int[] bits) {
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
            MetalNativeApi api, MetalNativeApi.Handle buffer) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(
                    Math.multiplyExact((long) INPUT_BITS.length, Integer.BYTES), Integer.BYTES);
            api.download(buffer, 0L, target, target.byteSize());
            int[] bits = new int[INPUT_BITS.length];
            for (int index = 0; index < bits.length; index++) {
                bits[index] = target.getAtIndex(JAVA_INT, index);
            }
            return bits;
        }
    }

    private static long[] dimensions(int valueCount, int extent) {
        long[] dimensions = new long[Math.multiplyExact(valueCount, 16)];
        for (int value = 0; value < valueCount; value++) {
            dimensions[value * 16] = extent;
        }
        return dimensions;
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }
}
