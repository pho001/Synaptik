package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.EngineMixedOwnerTestAccess;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Public, CPU-free Engine evidence for exact Metal ordering and arg-extrema execution. */
final class EngineOrderingMetalIntegrationTest {
    @Test
    void cpuFreeEnginePublishesEveryCarrierRoleAcrossProfilesReuseAndSessions() {
        Path library = configuredMetalLibrary();
        for (NumericalProfile profile : NumericalProfile.values()) {
            try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
                builder.numericalProfile(profile);
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    for (DataType type : DataType.values()) {
                        long[] raw = corpus(type);
                        Tensor input = nativeTensor(type, Shape.of(2, 4), raw, arena);
                        Tensor sorted = input.sort(1, true);
                        Tensor argsorted = input.argsort(1, false);
                        var largest = input.topK(2, 1, true, false);
                        var smallest = input.topK(2, 1, false, true);
                        Tensor nestedValues = largest.values().sort(1, false);
                        Tensor fanoutIndices = sorted.argsort(1, true);
                        var outputs = new ArrayList<Tensor>(List.of(
                                sorted,
                                argsorted,
                                largest.values(),
                                largest.indices(),
                                smallest.values(),
                                smallest.indices(),
                                nestedValues,
                                fanoutIndices));
                        if (type != DataType.BOOL) {
                            outputs.add(input.argMax(
                                    1, false, ArgExtremaTiePolicy.FIRST_INDEX));
                            outputs.add(input.argMin(
                                    1, false, ArgExtremaTiePolicy.LAST_INDEX));
                        }
                        var compiled = engine.compile(outputs);
                        assertEquals(List.of("metal"),
                                EngineMixedOwnerTestAccess.partitionOwners(compiled));
                        List<byte[]> expected = expected(type, raw);
                        try (var first = engine.session(compiled);
                                var independent = engine.session(compiled)) {
                            assertResults(first.run(List.of(input)), expected);
                            assertResults(first.run(List.of(input)), expected);
                            assertResults(independent.run(List.of(input)), expected);
                        }
                    }
                }
            }
        }
    }

    @Test
    void topKCompanionRemainsCorrectWhenOnlyOnePublicRoleIsRequested() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                long[] raw = corpus(DataType.FLOAT32);
                Tensor input = nativeTensor(DataType.FLOAT32, Shape.of(2, 4), raw, arena);
                var top = input.topK(2, 1, true, false);
                var valuesOnly = engine.compile(List.of(top.values()));
                var indicesOnly = engine.compile(List.of(top.indices()));
                List<byte[]> all = expected(DataType.FLOAT32, raw);
                try (var valuesSession = engine.session(valuesOnly);
                        var indicesSession = engine.session(indicesOnly);
                        RunResult values = valuesSession.run(List.of(input));
                        RunResult indices = indicesSession.run(List.of(input))) {
                    assertEquals(1, values.resultCount());
                    assertEquals(1, indices.resultCount());
                    assertPublication(values, 0, all.get(2));
                    assertPublication(indices, 0, all.get(3));
                }
            }
        }
    }
    @Test
    void cpuFreeEngineRejectsOverLimitAndGeneratedOrderingBackwardBeforePreparation() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Tensor overLimit = TensorFactory.create(
                        new TensorDescriptor(
                                DataType.BOOL,
                                Shape.of(0x1_0000_0000L),
                                Optional.of(LayoutDescriptor.contiguous(
                                        Shape.of(0x1_0000_0000L))),
                                false),
                        Optional.empty(),
                        Optional.empty());
                IllegalStateException overLimitFailure = assertThrows(
                        IllegalStateException.class,
                        () -> engine.compile(List.of(overLimit.sort(0))));
                assertTrue(overLimitFailure.getMessage().contains(
                        "no hard-eligible backend is available for ownership selection"));

                Tensor input = nativeTensor(
                        DataType.FLOAT32, Shape.of(2, 4), corpus(DataType.FLOAT32), arena);
                Tensor sortSeed = nativeTensor(
                        DataType.FLOAT32, Shape.of(2, 4),
                        new long[] {
                            0x3f80_0000L, 0x3f80_0000L, 0x3f80_0000L, 0x3f80_0000L,
                            0x3f80_0000L, 0x3f80_0000L, 0x3f80_0000L, 0x3f80_0000L
                        }, false, arena);
                assertThrows(
                        IllegalStateException.class,
                        () -> engine.compile(
                                List.of(input.sort(1)),
                                List.of(sortSeed),
                                List.of(input)));
                var top = input.topK(2, 1);
                Tensor topSeed = nativeTensor(
                        DataType.FLOAT32, Shape.of(2, 2),
                        new long[] {0x3f80_0000L, 0x3f80_0000L, 0x3f80_0000L, 0x3f80_0000L},
                        false, arena);
                assertThrows(
                        IllegalStateException.class,
                        () -> engine.compile(
                                List.of(top.values()),
                                List.of(topSeed),
                                List.of(input)));
            }
        }
    }


    private static List<byte[]> expected(DataType type, long[] raw) {
        List<byte[]> expected = new ArrayList<>();
        long[] descending = orderedValues(type, raw, false);
        expected.add(words(type, descending));
        expected.add(longWords(orderedIndices(type, raw, true)));
        TopSelection largest = top(type, raw, 2, false, false);
        expected.add(words(type, largest.values()));
        expected.add(longWords(largest.indices()));
        TopSelection smallest = top(type, raw, 2, true, true);
        expected.add(words(type, smallest.values()));
        expected.add(longWords(smallest.indices()));
        expected.add(words(type, orderedValues(type, largest.values(), true, 2)));
        expected.add(longWords(orderedIndices(type, descending, false)));
        if (type != DataType.BOOL) {
            expected.add(longWords(arg(type, raw, false, false)));
            expected.add(longWords(arg(type, raw, true, true)));
        }
        return expected;
    }

    private static long[] orderedValues(
            DataType type, long[] values, boolean ascending) {
        return orderedValues(type, values, ascending, 4);
    }

    private static long[] orderedValues(
            DataType type, long[] values, boolean ascending, int width) {
        long[] output = new long[values.length];
        for (int base = 0; base < values.length; base += width) {
            int[] order = order(type, values, base, width, ascending);
            for (int rank = 0; rank < width; rank++) {
                output[base + rank] = values[base + order[rank]];
            }
        }
        return output;
    }

    private static long[] orderedIndices(DataType type, long[] values, boolean ascending) {
        int width = 4;
        long[] output = new long[values.length];
        for (int base = 0; base < values.length; base += width) {
            int[] order = order(type, values, base, width, ascending);
            for (int rank = 0; rank < width; rank++) output[base + rank] = order[rank];
        }
        return output;
    }

    private static TopSelection top(
            DataType type, long[] values, int k, boolean ascending, boolean sorted) {
        long[] selectedValues = new long[(values.length / 4) * k];
        long[] selectedIndices = new long[selectedValues.length];
        for (int base = 0, target = 0; base < values.length; base += 4) {
            int[] order = order(type, values, base, 4, ascending);
            int[] selected = Arrays.copyOf(order, k);
            if (!sorted) Arrays.sort(selected);
            for (int index : selected) {
                selectedValues[target] = values[base + index];
                selectedIndices[target++] = index;
            }
        }
        return new TopSelection(selectedValues, selectedIndices);
    }

    private static long[] arg(
            DataType type, long[] values, boolean minimum, boolean last) {
        long[] output = new long[values.length / 4];
        for (int base = 0; base < values.length; base += 4) {
            int selected = 0;
            for (int index = 1; index < 4; index++) {
                long candidate = values[base + index];
                long best = values[base + selected];
                boolean candidateNaN = isNaN(type, candidate);
                boolean bestNaN = isNaN(type, best);
                int comparison = semanticCompare(type, candidate, best);
                if (candidateNaN && (!bestNaN || last)
                        || !candidateNaN && !bestNaN
                                && ((minimum && comparison < 0) || (!minimum && comparison > 0)
                                        || comparison == 0 && last)) {
                    selected = index;
                }
            }
            output[base / 4] = selected;
        }
        return output;
    }

    private static int[] order(
            DataType type, long[] values, int base, int width, boolean ascending) {
        Integer[] boxed = new Integer[width];
        for (int index = 0; index < width; index++) boxed[index] = index;
        Arrays.sort(boxed, (left, right) -> {
            boolean leftNaN = isNaN(type, values[base + left]);
            boolean rightNaN = isNaN(type, values[base + right]);
            if (leftNaN || rightNaN) {
                if (leftNaN != rightNaN) return leftNaN ? 1 : -1;
                return Integer.compare(left, right);
            }
            int comparison = semanticCompare(type, values[base + left], values[base + right]);
            if (!ascending) comparison = -comparison;
            return comparison != 0 ? comparison : Integer.compare(left, right);
        });
        int[] result = new int[width];
        for (int index = 0; index < width; index++) result[index] = boxed[index];
        return result;
    }

    private static int semanticCompare(DataType type, long left, long right) {
        boolean leftNaN = isNaN(type, left);
        boolean rightNaN = isNaN(type, right);
        if (leftNaN || rightNaN) {
            if (leftNaN == rightNaN) return 0;
            return leftNaN ? 1 : -1;
        }
        return Long.compareUnsigned(orderingKey(type, left), orderingKey(type, right));
    }

    private static boolean isNaN(DataType type, long word) {
        return switch (type) {
            case FLOAT64 -> (word & 0x7ff0_0000_0000_0000L) == 0x7ff0_0000_0000_0000L
                    && (word & 0x000f_ffff_ffff_ffffL) != 0;
            case FLOAT32 -> (word & 0x7f80_0000L) == 0x7f80_0000L
                    && (word & 0x007f_ffffL) != 0;
            case BFLOAT16 -> (word & 0x7f80L) == 0x7f80L && (word & 0x007fL) != 0;
            case INT32, INT64, BOOL -> false;
        };
    }

    private static long orderingKey(DataType type, long word) {
        return switch (type) {
            case FLOAT64 -> (word & Long.MIN_VALUE) != 0 ? ~word : word ^ Long.MIN_VALUE;
            case INT64 -> word ^ Long.MIN_VALUE;
            case FLOAT32 -> Integer.toUnsignedLong(
                    (int) word < 0 ? ~(int) word : (int) word ^ Integer.MIN_VALUE);
            case INT32 -> Integer.toUnsignedLong((int) word ^ Integer.MIN_VALUE);
            case BFLOAT16 -> {
                int narrowed = (int) word & 0xffff;
                yield ((narrowed & 0x8000) != 0 ? ~narrowed : narrowed ^ 0x8000) & 0xffffL;
            }
            case BOOL -> word;
        };
    }

    private static long[] corpus(DataType type) {
        return switch (type) {
            case FLOAT64 -> new long[] {
                0x0000_0000_0000_0001L, 0x8000_0000_0000_0001L,
                0x7ff0_0000_0000_0000L, 0xfff0_0000_0000_0000L,
                0x7ff0_0000_0000_1234L, 0xfff8_0000_0000_5678L,
                0x7fef_ffff_ffff_ffffL, 0x0010_0000_0000_0000L
            };
            case FLOAT32 -> new long[] {
                0x0000_0001L, 0x8000_0001L, 0x7f80_0000L, 0xff80_0000L,
                0x7fa1_2345L, 0xffc5_4321L, 0x7f7f_ffffL, 0x0080_0000L
            };
            case BFLOAT16 -> new long[] {
                0x0001L, 0x8001L, 0x7f80L, 0xff80L,
                0x7f81L, 0xffc2L, 0x7f7fL, 0x0080L
            };
            case INT32 -> new long[] {
                Integer.MIN_VALUE, Integer.MAX_VALUE, -1, 0,
                1, Integer.MIN_VALUE, Integer.MAX_VALUE, 2
            };
            case INT64 -> new long[] {
                Long.MIN_VALUE, Long.MAX_VALUE, -1, 0,
                1, Long.MIN_VALUE, Long.MAX_VALUE, 2
            };
            case BOOL -> new long[] {1, 0, 0, 1, 0, 1, 1, 0};
        };
    }

    private static Tensor nativeTensor(
            DataType type, Shape shape, long[] raw, Arena arena) {
        return nativeTensor(type, shape, raw, type.isDifferentiable(), arena);
    }

    private static Tensor nativeTensor(
            DataType type, Shape shape, long[] raw, boolean requiresGrad, Arena arena) {
        byte[] bytes = encodedWords(type, ByteOrder.nativeOrder(), raw);
        MemorySegment segment = arena.allocate(bytes.length, Math.max(1, type.byteWidth()));
        segment.copyFrom(MemorySegment.ofArray(bytes));
        TensorDescriptor descriptor = new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
        return TensorFactory.create(descriptor, Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, raw.length, segment)));
    }

    private static byte[] words(DataType type, long... words) {
        return encodedWords(type, ByteOrder.BIG_ENDIAN, words);
    }

    private static byte[] encodedWords(DataType type, ByteOrder order, long... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * type.byteWidth()).order(order);
        for (long word : words) {
            switch (type) {
                case FLOAT64, INT64 -> bytes.putLong(word);
                case FLOAT32, INT32 -> bytes.putInt((int) word);
                case BFLOAT16 -> bytes.putShort((short) word);
                case BOOL -> bytes.put((byte) word);
            }
        }
        return bytes.array();
    }

    private static byte[] longWords(long... words) {
        return words(DataType.INT64, words);
    }

    private static void assertResults(RunResult result, List<byte[]> expected) {
        try (result) {
            assertEquals(expected.size(), result.resultCount());
            for (int index = 0; index < expected.size(); index++) {
                assertPublication(result, index, expected.get(index));
            }
        }
    }

    private static void assertPublication(RunResult result, int index, byte[] expected) {
        ByteBuffer bytes = result.materialize(
                result.publications().get(index), expected.length).bytes();
        byte[] actual = new byte[expected.length];
        bytes.get(actual);
        assertArrayEquals(expected, actual, "publication " + index);
    }

    private static Path configuredMetalLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        Assumptions.assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is required for real Metal integration");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(library), "configured Metal library must exist: " + library);
        return library;
    }

    private record TopSelection(long[] values, long[] indices) {}
}
