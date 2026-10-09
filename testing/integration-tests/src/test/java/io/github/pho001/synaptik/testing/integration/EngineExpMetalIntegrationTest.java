package io.github.pho001.synaptik.testing.integration;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.EngineMixedOwnerTestAccess;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Public, CPU-free Engine execution of direct FLOAT32 and typed low custom EXP routes. */
final class EngineExpMetalIntegrationTest {
    @Test
    void publicEnginePreparesAndReusesFloat32ExpWithoutMutatingInput() {
        Path library = configuredLibrary();
        float[] samples = {
            -2.0f, -1.0f, -0.0f, 0.0f, 0.5f, 1.0f, 2.0f,
            Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN
        };
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                MemorySegment storage = arena.allocate(samples.length * Float.BYTES, Float.BYTES);
                MemorySegment.copy(MemorySegment.ofArray(samples), 0,
                        storage, 0, storage.byteSize());
                Tensor input = tensor(DataType.FLOAT32, Shape.of(samples.length),
                        false, storage, samples.length);
                var compiled = engine.compile(List.of(input.exp()));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));
                try (var session = engine.session(compiled)) {
                    assertExp(session.run(List.of(input)), samples);
                    assertExp(session.run(List.of(input)), samples);
                }
                for (int index = 0; index < samples.length; index++) {
                    assertEquals(Float.floatToRawIntBits(samples[index]),
                            storage.getAtIndex(JAVA_INT, index), "caller input " + index);
                }
                Tensor gradientInput = tensor(DataType.FLOAT32,
                        Shape.of(samples.length), true, storage, samples.length);
                assertThrows(IllegalStateException.class,
                        () -> engine.compile(List.of(gradientInput.exp())));
            }
        }
    }

    @Test
    void explicitLowCastThenFloat32ExpRunsInsideOneCustomMetalPartition() {
        Path library = configuredLibrary();
        float[] samples = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                    MemorySegment storage = arena.allocate(
                            samples.length * Short.BYTES, Short.BYTES);
                    short[] original = new short[samples.length];
                    for (int index = 0; index < samples.length; index++) {
                        original[index] = low == DataType.BFLOAT16
                                ? (short) (Float.floatToRawIntBits(samples[index]) >>> 16)
                                : Float16Bits.fromFloat(samples[index]);
                        storage.setAtIndex(JAVA_SHORT, index, original[index]);
                    }
                    Tensor input = tensor(low, Shape.of(samples.length), false,
                            storage, samples.length);
                    var compiled = engine.compile(List.of(input.cast(DataType.FLOAT32).exp()));
                    assertEquals(List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled), low.toString());
                    try (var session = engine.session(compiled)) {
                        assertExp(session.run(List.of(input)), samples);
                        assertExp(session.run(List.of(input)), samples);
                    }
                    for (int index = 0; index < samples.length; index++) {
                        assertEquals(original[index], storage.getAtIndex(JAVA_SHORT, index),
                                low + " input " + index);
                    }
                }
            }
        }
    }

    @Test
    void publicEngineRunsAndReusesIndependentLowExpWithoutMutatingCallerStorage() {
        Path library = configuredLibrary();
        float[] samples = {-12.0f, -2.0f, -1.0f, -0.0f, 0.0f, 0.5f,
                1.0f, 2.0f, 12.0f, Float.NEGATIVE_INFINITY,
                Float.POSITIVE_INFINITY, Float.NaN};
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                    MemorySegment storage = arena.allocate(
                            samples.length * Short.BYTES, Short.BYTES);
                    short[] original = new short[samples.length];
                    for (int index = 0; index < samples.length; index++) {
                        original[index] = low == DataType.BFLOAT16
                                ? BFloat16Bits.fromFloat(samples[index])
                                : Float16Bits.fromFloat(samples[index]);
                        storage.setAtIndex(JAVA_SHORT, index, original[index]);
                    }
                    Tensor input = tensor(low, Shape.of(samples.length), false,
                            storage, samples.length);
                    var compiled = engine.compile(List.of(input.exp()));
                    assertEquals(List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled), low.name());
                    try (var session = engine.session(compiled)) {
                        for (int repeat = 0; repeat < 2; repeat++) {
                            try (RunResult result = session.run(List.of(input))) {
                                ByteBuffer bytes = result.materialize(
                                        result.publications().getFirst(),
                                        (long) samples.length * Short.BYTES).bytes();
                                for (int index = 0; index < samples.length; index++) {
                                    short bits = bytes.getShort();
                                    float actual = low == DataType.BFLOAT16
                                            ? BFloat16Bits.toFloat(bits)
                                            : Float16Bits.toFloat(bits);
                                    float widened = low == DataType.BFLOAT16
                                            ? BFloat16Bits.toFloat(original[index])
                                            : Float16Bits.toFloat(original[index]);
                                    if (Float.isNaN(widened)) {
                                        assertTrue(Float.isNaN(actual), low + " NaN");
                                    } else if (widened == 0.0f) {
                                        assertEquals(1.0f, actual, low + " signed zero");
                                    } else if (widened == Float.NEGATIVE_INFINITY) {
                                        assertEquals(0, Short.toUnsignedInt(bits), low + " -inf");
                                    } else if (widened == Float.POSITIVE_INFINITY
                                            || low == DataType.FLOAT16 && widened == 12.0f) {
                                        assertEquals(Float.POSITIVE_INFINITY, actual, low + " overflow");
                                    } else {
                                        double reference = StrictMath.exp((double) widened);
                                        assertTrue(Float.isFinite(actual) && actual > 0.0f,
                                                low + " input=" + widened);
                                        if (low == DataType.FLOAT16 && widened == -12.0f) {
                                            assertTrue(Short.toUnsignedInt(bits) > 0
                                                    && Short.toUnsignedInt(bits) < 0x0400);
                                        } else if (reference >= (low == DataType.BFLOAT16
                                                ? BFloat16Bits.toFloat((short) 0x0080)
                                                : Float16Bits.toFloat((short) 0x0400))) {
                                            assertTrue(Math.abs(actual - reference) / reference
                                                    <= (low == DataType.BFLOAT16
                                                            ? 0.0040 : 0.00055),
                                                    low + " input=" + widened);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for (int index = 0; index < samples.length; index++) {
                        assertEquals(original[index], storage.getAtIndex(JAVA_SHORT, index),
                                low + " input " + index);
                    }
                    Tensor gradientInput = tensor(low, Shape.of(samples.length), true,
                            storage, samples.length);
                    assertThrows(IllegalStateException.class,
                            () -> engine.compile(List.of(gradientInput.exp())), low.name());
                }
            }
        }
    }

    private static void assertExp(RunResult result, float[] inputs) {
        try (result) {
            assertEquals(1, result.resultCount());
            ByteBuffer bytes = result.materialize(result.publications().getFirst(),
                    (long) inputs.length * Float.BYTES).bytes();
            for (int index = 0; index < inputs.length; index++) {
                float input = inputs[index];
                float actual = bytes.getFloat();
                if (Float.isNaN(input)) {
                    assertTrue(Float.isNaN(actual));
                } else if (input == 0.0f) {
                    assertEquals(0x3f800000, Float.floatToRawIntBits(actual));
                } else if (input == Float.NEGATIVE_INFINITY) {
                    assertEquals(0, Float.floatToRawIntBits(actual));
                } else if (input == Float.POSITIVE_INFINITY) {
                    assertEquals(Float.POSITIVE_INFINITY, actual);
                } else {
                    float reference = (float) StrictMath.exp((double) input);
                    assertTrue(Float.isFinite(actual) && actual > 0.0f);
                    assertTrue(Math.abs((double) actual - reference) / reference <= 2e-6,
                            "EXP finite lane " + index);
                }
            }
        }
    }

    private static Tensor tensor(DataType type, Shape shape, boolean grad,
            MemorySegment storage, int elements) {
        return TensorFactory.create(new TensorDescriptor(type, shape,
                        Optional.of(LayoutDescriptor.contiguous(shape)), grad),
                Optional.empty(), Optional.of(new MemorySegmentStorage(
                        type, elements, storage)));
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        if (configured == null || configured.isBlank()) {
            throw new IllegalStateException("SYNAPTIK_METAL_TEST_LIBRARY is required");
        }
        Path library = Path.of(configured).toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(library), "source-matched Metal library is missing");
        return library;
    }
}
