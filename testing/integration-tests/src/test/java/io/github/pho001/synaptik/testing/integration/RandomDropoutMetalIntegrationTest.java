package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.EngineMixedOwnerTestAccess;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.DropoutResult;
import io.github.pho001.synaptik.model.tensor.GraphRngState;
import io.github.pho001.synaptik.model.tensor.RecurrentScan;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.layers.Dropout;
import io.github.pho001.synaptik.nn.module.ForwardContext;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** CPU-free public Engine evidence for Metal explicit-state RNG and dropout. */
final class RandomDropoutMetalIntegrationTest {
    private static final long KEY_BIAS = 0x9e37_79b9_7f4a_7c15L;
    private static final long M1 = 0xbf58_476d_1ce4_e5b9L;
    private static final long M2 = 0x94d0_49bb_1331_11ebL;

    @Test
    void forwardReplayBranchChainingReuseAndConcurrentSessionsAreMetalOnly() throws Exception {
        Path library = configuredMetalLibrary();
        int[] inputBits = {
            0x8000_0000,
            Float.floatToRawIntBits(1.0f),
            Float.floatToRawIntBits(-2.0f),
            Float.floatToRawIntBits(4.0f),
            0x7f80_0000,
            0xff80_0000,
            0x8000_0000,
            Float.floatToRawIntBits(3.0f)
        };
        long key = 0L;
        long counter = 0L;
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Tensor input = nativeTensor(
                        DataType.FLOAT32, Shape.of(inputBits.length), false,
                        integerWords(inputBits), arena);
                GraphRngState state = GraphRngState.initial(key, counter);
                DropoutResult first = input.dropout(0.5d, state);
                DropoutResult branch = input.dropout(0.25d, state);
                DropoutResult chained = first.output().dropout(0.5d, first.nextState());
                var compiled = engine.compile(List.of(
                        first.output(), branch.output(), chained.output()));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));

                Oracle firstExpected = oracle(inputBits, key, counter, 0.5d);
                Oracle branchExpected = oracle(inputBits, key, counter, 0.25d);
                Oracle chainedExpected = oracle(
                        firstExpected.output(), key, counter + inputBits.length, 0.5d);
                List<byte[]> expected = List.of(
                        floatWords(firstExpected.output()),
                        floatWords(branchExpected.output()),
                        floatWords(chainedExpected.output()));
                try (var firstSession = engine.session(compiled);
                        var independent = engine.session(compiled)) {
                    assertResults(firstSession.run(List.of(input)), expected);
                    assertResults(firstSession.run(List.of(input)), expected);
                    assertResults(independent.run(List.of(input)), expected);
                }

                Callable<List<byte[]>> invocation = () -> {
                    try (var session = engine.session(compiled);
                            var result = session.run(List.of(input))) {
                        return materialize(result, expected);
                    }
                };
                try (var executor = Executors.newFixedThreadPool(4)) {
                    var futures = executor.invokeAll(
                            java.util.Collections.nCopies(8, invocation));
                    for (var future : futures) {
                        List<byte[]> actual = future.get();
                        for (int index = 0; index < expected.size(); index++) {
                            assertArrayEquals(expected.get(index), actual.get(index));
                        }
                    }
                }
            }
        }
    }

    @Test
    void backwardConsumesTheSavedMaskAndDropsExceptionalCotangentsToPositiveZero() {
        Path library = configuredMetalLibrary();
        int count = 8;
        int[] zeros = new int[count];
        int[] seedBits = {
            0x7fc1_2345,
            Float.floatToRawIntBits(1.0f),
            Float.floatToRawIntBits(-2.0f),
            0x7f80_0000,
            0xff80_0000,
            0x7fc5_4321,
            Float.floatToRawIntBits(3.0f),
            Float.floatToRawIntBits(-4.0f)
        };
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Tensor input = nativeTensor(
                        DataType.FLOAT32, Shape.of(count), true, integerWords(zeros), arena);
                Tensor seed = nativeTensor(
                        DataType.FLOAT32, Shape.of(count), false,
                        integerWords(seedBits), arena);
                DropoutResult dropout = input.dropout(0.5d, GraphRngState.initial(0L, 0L));
                var compiled = engine.compile(
                        List.of(dropout.output()), List.of(seed), List.of(input));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));

                Oracle forward = oracle(zeros, 0L, 0L, 0.5d);
                int[] expectedGradient = new int[count];
                float scale = 2.0f;
                for (int index = 0; index < count; index++) {
                    expectedGradient[index] = forward.mask()[index] == 0
                            ? 0
                            : Float.floatToRawIntBits(
                                    Float.intBitsToFloat(seedBits[index]) * scale);
                }
                try (var session = engine.session(compiled)) {
                    assertResults(session.run(List.of(seed, input)), List.of(
                            floatWords(forward.output()),
                            floatWords(expectedGradient)));
                }
            }
        }
    }

    @Test
    void lowPrecisionBackwardReusesSavedMaskAndDeterministicStateOnMetal() {
        Path library = configuredMetalLibrary();
        float[] values = {1, 2, 3, 4, 5, 6, 7, 8};
        byte[] mask = oracle(new int[values.length], 0L, 0L, 0.5d).mask();
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                    Tensor input = nativeTensor(
                            type, Shape.of(values.length), true,
                            lowNativeWords(type, values), arena);
                    Tensor seed = nativeTensor(
                            type, Shape.of(values.length), false,
                            lowNativeWords(type, values), arena);
                    DropoutResult dropout =
                            input.dropout(0.5d, GraphRngState.initial(0L, 0L));
                    var compiled = engine.compile(
                            List.of(dropout.output()), List.of(seed), List.of(input));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled),
                            type.name());
                    float[] expected = new float[values.length];
                    for (int index = 0; index < values.length; index++) {
                        expected[index] = mask[index] == 0 ? 0.0f : values[index] * 2.0f;
                    }
                    List<byte[]> publications = List.of(
                            lowCanonicalWords(type, expected),
                            lowCanonicalWords(type, expected));
                    try (var session = engine.session(compiled)) {
                        assertResults(session.run(List.of(seed, input)), publications);
                        assertResults(session.run(List.of(seed, input)), publications);
                    }
                }
            }
        }
    }

    @Test
    void evaluationBuildsNoDropoutAndUnsupportedProfilesCarriersLimitsAndRecurrenceFailEarly() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared()) {
            Tensor value = nativeTensor(
                    DataType.FLOAT32, Shape.of(2), false,
                    integerWords(new int[] {
                        Float.floatToRawIntBits(1.0f), Float.floatToRawIntBits(2.0f)
                    }), arena);
            GraphRngState state = GraphRngState.initial(7L, 11L);
            Dropout layer = new Dropout(0.5d);
            var evaluation = layer.forward(
                    value, state, new ForwardContext(ForwardMode.EVALUATION));
            assertSame(value, evaluation.output());
            assertSame(state, evaluation.nextState());
            assertTrue(evaluation.output().provenance().isEmpty(),
                    "evaluation bypass contains no wire-101 occurrence");
            var training = layer.forward(
                    value, state, new ForwardContext(ForwardMode.TRAINING));
            assertSame(DropoutKind.DROPOUT,
                    training.output().provenance().orElseThrow().operation().kind());

            try (Engine.Builder strictBuilder = Engine.builder()) {
                strictBuilder.numericalProfile(NumericalProfile.STRICT_IEEE);
                strictBuilder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine strict = strictBuilder.build()) {
                    assertThrows(IllegalStateException.class,
                            () -> strict.compile(List.of(training.output())));
                }
            }

            try (Engine.Builder acceleratorBuilder = Engine.builder()) {
                acceleratorBuilder.numericalProfile(NumericalProfile.ACCELERATOR);
                acceleratorBuilder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine accelerator = acceleratorBuilder.build()) {
                    Tensor float64 = nativeTensor(
                            DataType.FLOAT64, Shape.of(2), false,
                            longWords(
                                    Double.doubleToRawLongBits(1.0d),
                                    Double.doubleToRawLongBits(2.0d)), arena);
                    assertThrows(IllegalStateException.class, () -> accelerator.compile(List.of(
                            float64.dropout(0.5d, GraphRngState.initial(0L, 0L)).output())));

                    Shape overShape = Shape.of(0x1_0000_0000L);
                    Tensor over = TensorFactory.create(
                            new TensorDescriptor(
                                    DataType.FLOAT32,
                                    overShape,
                                    Optional.of(LayoutDescriptor.contiguous(overShape)),
                                    false),
                            Optional.empty(),
                            Optional.empty());
                    assertThrows(IllegalStateException.class, () -> accelerator.compile(List.of(
                            over.dropout(0.5d, GraphRngState.initial(0L, 0L)).output())));

                    Tensor recurrentInput = nativeTensor(
                            DataType.FLOAT32, Shape.of(1, 1, 1), true,
                            integerWords(new int[] {Float.floatToRawIntBits(1.0f)}), arena);
                    Tensor lengths = nativeTensor(
                            DataType.INT64, Shape.of(1), false, longWords(1L), arena);
                    Tensor hidden = nativeTensor(
                            DataType.FLOAT32, Shape.of(1, 1), true,
                            integerWords(new int[] {0}), arena);
                    Tensor inputWeight = nativeTensor(
                            DataType.FLOAT32, Shape.of(1, 1), true,
                            integerWords(new int[] {Float.floatToRawIntBits(1.0f)}), arena);
                    Tensor hiddenWeight = nativeTensor(
                            DataType.FLOAT32, Shape.of(1, 1), true,
                            integerWords(new int[] {Float.floatToRawIntBits(1.0f)}), arena);
                    var recurrent = RecurrentScan.rnn(
                            recurrentInput,
                            lengths,
                            hidden,
                            inputWeight,
                            hiddenWeight,
                            RecurrentDirection.FORWARD);
                    assertThrows(IllegalStateException.class,
                            () -> accelerator.compile(List.of(recurrent.outputs())));
                    Tensor recurrentSeed = nativeTensor(
                            DataType.FLOAT32, Shape.of(1, 1, 1), false,
                            integerWords(new int[] {Float.floatToRawIntBits(1.0f)}), arena);
                    assertThrows(IllegalArgumentException.class,
                            () -> accelerator.compile(
                                    List.of(recurrent.outputs()),
                                    List.of(recurrentSeed),
                                    List.of(recurrentInput)));
                }
            }
        }
    }

    private static Oracle oracle(
            int[] input, long key, long counter, double probability) {
        long threshold = threshold(Double.doubleToRawLongBits(probability));
        float complement = (float) (1.0d - probability);
        float scale = 1.0f / complement;
        int[] output = new int[input.length];
        byte[] mask = new byte[input.length];
        long keyOffset = mix64(key + KEY_BIAS);
        for (int index = 0; index < input.length; index++) {
            long word = mix64(counter + index + keyOffset);
            boolean keep = (word >>> 11) >= threshold;
            mask[index] = keep ? (byte) 1 : 0;
            output[index] = keep
                    ? Float.floatToRawIntBits(Float.intBitsToFloat(input[index]) * scale)
                    : 0;
        }
        return new Oracle(output, mask);
    }

    private static long threshold(long bits) {
        long magnitude = bits & Long.MAX_VALUE;
        if (magnitude == 0L) return 0L;
        long exponent = magnitude >>> 52;
        if (exponent == 0L) return 1L;
        long significand = (1L << 52) | (magnitude & 0x000f_ffff_ffff_ffffL);
        long shift = 1022L - exponent;
        if (shift == 0L) return significand;
        if (shift >= 63L) return 1L;
        long denominator = 1L << shift;
        return (significand + denominator - 1L) / denominator;
    }

    private static long mix64(long word) {
        word = (word ^ (word >>> 30)) * M1;
        word = (word ^ (word >>> 27)) * M2;
        return word ^ (word >>> 31);
    }

    private static Tensor nativeTensor(
            DataType type,
            Shape shape,
            boolean requiresGrad,
            byte[] nativeBytes,
            Arena arena) {
        MemorySegment segment = arena.allocate(nativeBytes.length, Math.max(1, type.byteWidth()));
        segment.copyFrom(MemorySegment.ofArray(nativeBytes));
        TensorDescriptor descriptor = new TensorDescriptor(
                type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        type, nativeBytes.length / type.byteWidth(), segment)));
    }

    private static byte[] integerWords(int[] words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int word : words) bytes.putInt(word);
        return bytes.array();
    }

    private static byte[] longWords(long... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Long.BYTES)
                .order(ByteOrder.nativeOrder());
        for (long word : words) bytes.putLong(word);
        return bytes.array();
    }

    private static byte[] floatWords(int[] words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Integer.BYTES)
                .order(ByteOrder.BIG_ENDIAN);
        for (int word : words) bytes.putInt(word);
        return bytes.array();
    }

    private static byte[] lowNativeWords(DataType type, float[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Short.BYTES)
                .order(ByteOrder.nativeOrder());
        for (float value : values) {
            bytes.putShort(type == DataType.BFLOAT16
                    ? BFloat16Bits.fromFloat(value)
                    : Float16Bits.fromFloat(value));
        }
        return bytes.array();
    }

    private static byte[] lowCanonicalWords(DataType type, float[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Short.BYTES)
                .order(ByteOrder.BIG_ENDIAN);
        for (float value : values) {
            bytes.putShort(type == DataType.BFLOAT16
                    ? BFloat16Bits.fromFloat(value)
                    : Float16Bits.fromFloat(value));
        }
        return bytes.array();
    }

    private static void assertResults(RunResult result, List<byte[]> expected) {
        try (result) {
            List<byte[]> actual = materialize(result, expected);
            for (int index = 0; index < expected.size(); index++) {
                assertArrayEquals(expected.get(index), actual.get(index), "publication " + index);
            }
        }
    }

    private static List<byte[]> materialize(RunResult result, List<byte[]> expected) {
        assertEquals(expected.size(), result.resultCount());
        var actual = new ArrayList<byte[]>(expected.size());
        for (int index = 0; index < expected.size(); index++) {
            ByteBuffer bytes = result.materialize(
                    result.publications().get(index), expected.get(index).length).bytes();
            byte[] copy = new byte[expected.get(index).length];
            bytes.get(copy);
            actual.add(copy);
        }
        return List.copyOf(actual);
    }

    private static Path configuredMetalLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        Assumptions.assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is required for real Metal integration");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(library), "configured Metal library must exist: " + library);
        return library;
    }

    private record Oracle(int[] output, byte[] mask) {}
}
