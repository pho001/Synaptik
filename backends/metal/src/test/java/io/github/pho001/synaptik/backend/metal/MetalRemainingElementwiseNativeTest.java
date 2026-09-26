package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class MetalRemainingElementwiseNativeTest {
    private static final int[] INPUT = {
        0x00000000, 0x80000000,
        0x00000001, 0x80000001,
        0x007fffff, 0x807fffff,
        0x00800000, 0x80800000,
        0x3f7fffff, 0xbf7fffff,
        0x3f800000, 0xbf800000,
        0x3f800001, 0xbf800001,
        0x3fc00000, 0xbfc00000,
        0x40000000, 0xc0000000,
        0x4affffff, 0xcaffffff,
        0x4b000000, 0xcb000000,
        0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000,
        0x7fc12345, 0xffc54321,
        0x7f812345, 0xff812346
    };
    private static final int[] FLOOR = {
        0x00000000, 0x80000000,
        0x00000000, 0xbf800000,
        0x00000000, 0xbf800000,
        0x00000000, 0xbf800000,
        0x00000000, 0xbf800000,
        0x3f800000, 0xbf800000,
        0x3f800000, 0xc0000000,
        0x3f800000, 0xc0000000,
        0x40000000, 0xc0000000,
        0x4afffffe, 0xcb000000,
        0x4b000000, 0xcb000000,
        0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000,
        0x7fc12345, 0xffc54321,
        0x7f812345, 0xff812346
    };
    private static final int[] CEIL = {
        0x00000000, 0x80000000,
        0x3f800000, 0x80000000,
        0x3f800000, 0x80000000,
        0x3f800000, 0x80000000,
        0x3f800000, 0x80000000,
        0x3f800000, 0xbf800000,
        0x40000000, 0xbf800000,
        0x40000000, 0xbf800000,
        0x40000000, 0xc0000000,
        0x4b000000, 0xcafffffe,
        0x4b000000, 0xcb000000,
        0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000,
        0x7fc12345, 0xffc54321,
        0x7f812345, 0xff812346
    };

    @Test
    void productionCustomProgramExecutesAllFourExactRawOperationsUnderBothProfiles() {
        Path library = configuredLibrary();
        for (NumericalProfile profile : NumericalProfile.values()) {
            var program = new MetalMpsGraphProgram(List.of(
                    unary(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 1),
                    unary(MetalMpsGraphProgram.NodeKind.CEIL, 0, 2),
                    unary(MetalMpsGraphProgram.NodeKind.SIGN, 0, 3),
                    unary(MetalMpsGraphProgram.NodeKind.RELU, 0, 4)));
            List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                    value(INPUT.length), value(INPUT.length), value(INPUT.length),
                    value(INPUT.length), value(INPUT.length));
            List<int[]> actual = executeCustom(
                    library, profile, program, values, new int[] {0},
                    new int[] {1, 2, 3, 4}, List.of(INPUT));
            assertModelWords(FLOOR, actual.get(0));
            assertModelWords(CEIL, actual.get(1));
            int[] sign = new int[INPUT.length];
            int[] relu = new int[INPUT.length];
            for (int index = 0; index < INPUT.length; index++) {
                int word = INPUT[index];
                if (isNaN(word) || (word & 0x7fff_ffff) == 0) sign[index] = word;
                else sign[index] = word < 0 ? 0xbf80_0000 : 0x3f80_0000;
                relu[index] = isNaN(word) ? word : word < 0 ? 0 : word;
            }
            assertModelWords(sign, actual.get(2));
            assertModelWords(relu, actual.get(3));
        }
    }

    @Test
    void directMpsGraphCandidatesRemainForceableForAllFourExactRawOperations() {
        Path library = configuredLibrary();
        int[] ordinary = bits(-2.5f, -0.0f, 0.0f, 1.25f, 4.0f);
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (MetalMpsGraphProgram.NodeKind kind : List.of(
                    MetalMpsGraphProgram.NodeKind.FLOOR,
                    MetalMpsGraphProgram.NodeKind.CEIL,
                    MetalMpsGraphProgram.NodeKind.SIGN,
                    MetalMpsGraphProgram.NodeKind.RELU)) {
                int[] output = NonProductionStructuralFixture.executeDirect(
                        library, profile, unary(kind, 0, 1),
                        List.of(value(ordinary.length), value(ordinary.length)),
                        List.of(ordinary));
                assertEquals(ordinary.length, output.length, kind.name());
            }
        }
    }

    @Test
    void everyBlockedCatalogRecipeCreatesRunsAndClosesOnlyThroughTheRawFixture() {
        Path library = configuredLibrary();
        int[] input = bits(0.25f, 0.5f, 1.0f, 2.0f);
        int[] exponent = bits(2.0f, 2.0f, 2.0f, 2.0f);
        int outputWords = input.length;

        int[] tensorPow = NonProductionStructuralFixture.executeDirect(
                library, NumericalProfile.ACCELERATOR,
                binary(MetalMpsGraphProgram.NodeKind.TENSOR_POW, 0, 1, 2),
                List.of(value(outputWords), value(outputWords), value(outputWords)),
                List.of(input, exponent));
        assertEquals(outputWords, tensorPow.length);

        for (MetalMpsGraphProgram.NodeKind kind : List.of(
                MetalMpsGraphProgram.NodeKind.SCALAR_ADD,
                MetalMpsGraphProgram.NodeKind.SCALAR_SUB,
                MetalMpsGraphProgram.NodeKind.SCALAR_MUL,
                MetalMpsGraphProgram.NodeKind.SCALAR_DIV,
                MetalMpsGraphProgram.NodeKind.SCALAR_POW)) {
            int[] output = NonProductionStructuralFixture.executeDirect(
                    library, NumericalProfile.ACCELERATOR,
                    scalar(kind, 0, 1, Float.floatToRawIntBits(2.0f)),
                    List.of(value(outputWords), value(outputWords)), List.of(input));
            assertEquals(outputWords, output.length, kind.name());
        }

        for (MetalMpsGraphProgram.NodeKind kind : List.of(
                MetalMpsGraphProgram.NodeKind.RECIPROCAL,
                MetalMpsGraphProgram.NodeKind.LOG,
                MetalMpsGraphProgram.NodeKind.LOG1P,
                MetalMpsGraphProgram.NodeKind.EXPM1,
                MetalMpsGraphProgram.NodeKind.ERF,
                MetalMpsGraphProgram.NodeKind.SQRT,
                MetalMpsGraphProgram.NodeKind.RSQRT,
                MetalMpsGraphProgram.NodeKind.TANH,
                MetalMpsGraphProgram.NodeKind.GELU,
                MetalMpsGraphProgram.NodeKind.GELU_TANH_APPROXIMATION,
                MetalMpsGraphProgram.NodeKind.SILU)) {
            int[] output = NonProductionStructuralFixture.executeDirect(
                    library, NumericalProfile.ACCELERATOR, unary(kind, 0, 1),
                    List.of(value(outputWords), value(outputWords)), List.of(input));
            assertEquals(outputWords, output.length, kind.name());
        }
    }

    @Test
    void mixedCustomProgramExecutesRawAndNestedMpsGraphNodesWithDirectPublication() {
        Path library = configuredLibrary();
        int[] input = bits(-1.5f, -0.25f, 0.0f, 1.5f);
        var program = new MetalMpsGraphProgram(List.of(
                unary(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 1),
                unary(MetalMpsGraphProgram.NodeKind.ABS, 1, 2),
                unary(MetalMpsGraphProgram.NodeKind.SIGN, 2, 3)));
        List<int[]> actual = executeCustom(
                library, NumericalProfile.STRICT_IEEE, program,
                List.of(value(4), value(4), value(4), value(4)),
                new int[] {0}, new int[] {1, 2, 3}, List.of(input));
        assertArrayEquals(bits(-2.0f, -1.0f, 0.0f, 1.0f), actual.get(0));
        assertArrayEquals(bits(2.0f, 1.0f, 0.0f, 1.0f), actual.get(1));
        assertArrayEquals(bits(1.0f, 1.0f, 0.0f, 1.0f), actual.get(2));
    }

    @Test
    void scalarWireRawBitsShapeAndTensorPrimitiveAllDriftInCanonicalImageDigests()
            throws NoSuchAlgorithmException {
        Set<String> digests = new HashSet<>();
        for (MetalMpsGraphProgram.NodeKind kind : List.of(
                MetalMpsGraphProgram.NodeKind.SCALAR_ADD,
                MetalMpsGraphProgram.NodeKind.SCALAR_SUB,
                MetalMpsGraphProgram.NodeKind.SCALAR_MUL,
                MetalMpsGraphProgram.NodeKind.SCALAR_DIV,
                MetalMpsGraphProgram.NodeKind.SCALAR_POW)) {
            digests.add(digestImage(
                    new MetalMpsGraphProgram(List.of(scalar(kind, 0, 1, 0x3f80_0000))),
                    List.of(value(4), value(4)), new int[] {0}, new int[] {1}));
        }
        digests.add(digestImage(
                new MetalMpsGraphProgram(List.of(scalar(
                        MetalMpsGraphProgram.NodeKind.SCALAR_ADD, 0, 1, 0x3f80_0001))),
                List.of(value(4), value(4)), new int[] {0}, new int[] {1}));
        digests.add(digestImage(
                new MetalMpsGraphProgram(List.of(scalar(
                        MetalMpsGraphProgram.NodeKind.SCALAR_ADD, 0, 1, 0x3f80_0000))),
                List.of(value(2, 2), value(2, 2)), new int[] {0}, new int[] {1}));
        digests.add(digestImage(
                new MetalMpsGraphProgram(List.of(binary(
                        MetalMpsGraphProgram.NodeKind.ADD, 0, 1, 2))),
                List.of(value(4), value(4), value(4)), new int[] {0, 1}, new int[] {2}));
        assertEquals(8, digests.size());
    }

    private static String digestImage(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets) throws NoSuchAlgorithmException {
        try (Arena arena = Arena.ofConfined()) {
            byte[] image = program.encodeNative(
                    arena, values, feeds, targets, MetalPreparedRoute.MPSGRAPH).toArray(JAVA_BYTE);
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(image));
        }
    }

    private static List<int[]> executeCustom(
            Path library,
            NumericalProfile profile,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<int[]> feedWords) {
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createMpsGraphExecutable(
                    context, profile, values, program, feeds, targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) buffers.add(api.createBuffer(context, value.byteCount()));
            for (int index = 0; index < feeds.length; index++)
                upload(api, buffers.get(feeds[index]), feedWords.get(index));
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++)
                    inputAddresses.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                MemorySegment outputAddresses = arena.allocate(ADDRESS, targets.length);
                for (int index = 0; index < targets.length; index++)
                    outputAddresses.setAtIndex(
                            ADDRESS, index, buffers.get(targets[index]).carrier());
                api.runExecutable(
                        executable, buffers.size(), inputAddresses,
                        targets.length, outputAddresses);
            }
            List<int[]> output = new ArrayList<>(targets.length);
            for (int target : targets)
                output.add(download(api, buffers.get(target), values.get(target).byteCount()));
            return List.copyOf(output);
        } finally {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static final class NonProductionStructuralFixture {
        private NonProductionStructuralFixture() {}

        static int[] executeDirect(
                Path library,
                NumericalProfile profile,
                MetalMpsGraphProgram.Node node,
                List<MetalMpsGraphProgram.ValueDescriptor> values,
                List<int[]> feedWords) {
            MetalNativeApi api = MetalNativeApi.open(library);
            MetalNativeApi.Handle context = null;
            MetalNativeApi.Handle executable = null;
            var inputs = new ArrayList<MetalNativeApi.Handle>();
            MetalNativeApi.Handle output = null;
            try {
                context = api.createContext();
                var program = new MetalMpsGraphProgram(List.of(node));
                int[] feeds = java.util.stream.IntStream.range(0, feedWords.size()).toArray();
                int target = values.size() - 1;
                executable = api.createMpsGraphExecutable(
                        context, profile, values, program, feeds, new int[] {target},
                        MetalPreparedRoute.MPSGRAPH);
                for (int[] words : feedWords) {
                    MetalNativeApi.Handle input = api.createBuffer(
                            context, Math.multiplyExact((long) words.length, Integer.BYTES));
                    inputs.add(input);
                    upload(api, input, words);
                }
                output = api.createBuffer(context, values.get(target).byteCount());
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size());
                    for (int index = 0; index < inputs.size(); index++)
                        inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
                    MemorySegment outputAddresses = arena.allocate(ADDRESS);
                    outputAddresses.set(ADDRESS, 0L, output.carrier());
                    api.runExecutable(
                            executable, inputs.size(), inputAddresses, 1, outputAddresses);
                }
                return download(api, output, values.get(target).byteCount());
            } finally {
                if (output != null) api.releaseBuffer(output);
                for (int index = inputs.size(); index-- > 0;) api.releaseBuffer(inputs.get(index));
                if (executable != null) api.releaseExecutable(executable);
                if (context != null) api.releaseContext(context);
                api.close();
            }
        }
    }

    private static MetalMpsGraphProgram.Node unary(
            MetalMpsGraphProgram.NodeKind kind, int input, int output) {
        return MetalMpsGraphProgram.Node.generic(
                kind, new int[] {input}, new int[] {output},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
    }

    private static MetalMpsGraphProgram.Node binary(
            MetalMpsGraphProgram.NodeKind kind, int left, int right, int output) {
        return MetalMpsGraphProgram.Node.generic(
                kind, new int[] {left, right}, new int[] {output},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
    }

    private static MetalMpsGraphProgram.Node scalar(
            MetalMpsGraphProgram.NodeKind kind, int input, int output, int rawBits) {
        return MetalMpsGraphProgram.Node.generic(
                kind, new int[] {input}, new int[] {output},
                MetalMpsGraphProgram.AttributeKind.SCALAR_VALUE,
                new long[] {1L, Integer.toUnsignedLong(rawBits)});
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, dimensions, false);
    }

    private static int[] bits(float... values) {
        int[] bits = new int[values.length];
        for (int index = 0; index < values.length; index++)
            bits[index] = Float.floatToRawIntBits(values[index]);
        return bits;
    }

    private static void assertModelWords(int[] expected, int[] actual) {
        assertEquals(expected.length, actual.length);
        for (int index = 0; index < expected.length; index++) {
            if (isNaN(expected[index])) assertTrue(isNaN(actual[index]), "NaN lane " + index);
            else assertEquals(expected[index], actual[index], "lane " + index);
        }
    }

    private static boolean isNaN(int word) {
        return (word & 0x7f80_0000) == 0x7f80_0000 && (word & 0x007f_ffff) != 0;
    }

    private static void upload(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int[] words) {
        ByteBuffer encoded = ByteBuffer.allocate(Math.multiplyExact(words.length, Integer.BYTES))
                .order(ByteOrder.nativeOrder());
        for (int word : words) encoded.putInt(word);
        byte[] bytes = encoded.array();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes.length, 1L);
            MemorySegment.copy(bytes, 0, source, JAVA_BYTE, 0L, bytes.length);
            api.upload(buffer, 0L, source, bytes.length);
        }
    }

    private static int[] download(
            MetalNativeApi api, MetalNativeApi.Handle buffer, long byteCount) {
        int bytes = Math.toIntExact(byteCount);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(bytes, 1L);
            api.download(buffer, 0L, target, bytes);
            ByteBuffer bufferBytes = ByteBuffer.wrap(target.toArray(JAVA_BYTE))
                    .order(ByteOrder.nativeOrder());
            int[] words = new int[bytes / Integer.BYTES];
            for (int index = 0; index < words.length; index++) words[index] = bufferBytes.getInt();
            return words;
        }
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }
}
