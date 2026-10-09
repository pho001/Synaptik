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

/** Public CPU-free Engine execution of direct FLOAT32 and explicit low-cast SIGMOID. */
final class EngineSigmoidMetalIntegrationTest {
    @Test
    void directFloat32SigmoidReusesPreparedSessionAndPreservesInput() {
        float[] samples = {-2.0f, -1.0f, -0.0f, 0.0f, 1.0f, 2.0f,
                Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN};
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(configuredLibrary())));
            try (Engine engine = builder.build()) {
                MemorySegment storage = arena.allocate(samples.length * Float.BYTES, Float.BYTES);
                MemorySegment.copy(MemorySegment.ofArray(samples), 0,
                        storage, 0, storage.byteSize());
                Tensor input = tensor(DataType.FLOAT32, Shape.of(samples.length), false,
                        storage, samples.length);
                var compiled = engine.compile(List.of(input.sigmoid()));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));
                try (var session = engine.session(compiled)) {
                    assertSigmoid(session.run(List.of(input)), samples);
                    assertSigmoid(session.run(List.of(input)), samples);
                }
                for (int index = 0; index < samples.length; index++) {
                    assertEquals(Float.floatToRawIntBits(samples[index]),
                            storage.getAtIndex(JAVA_INT, index));
                }
                Tensor gradientInput = tensor(DataType.FLOAT32, Shape.of(samples.length), true,
                        storage, samples.length);
                assertThrows(IllegalStateException.class,
                        () -> engine.compile(List.of(gradientInput.sigmoid())));
            }
        }
    }

    @Test
    void explicitLowCastAllowsOnlyFloat32SigmoidInsideCustomPartition() {
        float[] samples = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(configuredLibrary())));
                try (Engine engine = builder.build()) {
                    MemorySegment storage = arena.allocate(samples.length * Short.BYTES,
                            Short.BYTES);
                    short[] original = new short[samples.length];
                    for (int index = 0; index < samples.length; index++) {
                        original[index] = low == DataType.BFLOAT16
                                ? (short) (Float.floatToRawIntBits(samples[index]) >>> 16)
                                : Float16Bits.fromFloat(samples[index]);
                        storage.setAtIndex(JAVA_SHORT, index, original[index]);
                    }
                    Tensor input = tensor(low, Shape.of(samples.length), false,
                            storage, samples.length);
                    assertThrows(IllegalStateException.class,
                            () -> engine.compile(List.of(input.sigmoid())));
                    var compiled = engine.compile(List.of(
                            input.cast(DataType.FLOAT32).sigmoid()));
                    assertEquals(List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled));
                    try (var session = engine.session(compiled)) {
                        assertSigmoid(session.run(List.of(input)), samples);
                        assertSigmoid(session.run(List.of(input)), samples);
                    }
                    for (int index = 0; index < original.length; index++) {
                        assertEquals(original[index], storage.getAtIndex(JAVA_SHORT, index));
                    }
                }
            }
        }
    }

    private static void assertSigmoid(RunResult result, float[] inputs) {
        try (result) {
            ByteBuffer bytes = result.materialize(result.publications().getFirst(),
                    (long) inputs.length * Float.BYTES).bytes();
            for (int index = 0; index < inputs.length; index++) {
                float input = inputs[index];
                float actual = bytes.getFloat();
                if (Float.isNaN(input)) {
                    assertTrue(Float.isNaN(actual));
                } else if (input == 0.0f) {
                    assertEquals(0x3f000000, Float.floatToRawIntBits(actual));
                } else if (input == Float.NEGATIVE_INFINITY) {
                    assertEquals(0, Float.floatToRawIntBits(actual));
                } else if (input == Float.POSITIVE_INFINITY) {
                    assertEquals(0x3f800000, Float.floatToRawIntBits(actual));
                } else {
                    double exponent = StrictMath.exp(input < 0.0f
                            ? (double) input : -(double) input);
                    double reference = input < 0.0f
                            ? exponent / (1.0 + exponent) : 1.0 / (1.0 + exponent);
                    assertTrue(Float.isFinite(actual) && actual > 0.0f && actual < 1.0f);
                    assertTrue(Math.abs(actual - reference) / reference <= 2e-6,
                            "SIGMOID finite lane " + index + " input=" + input);
                }
            }
        }
    }

    private static Tensor tensor(DataType type, Shape shape, boolean grad,
            MemorySegment storage, int elements) {
        return TensorFactory.create(new TensorDescriptor(type, shape,
                        Optional.of(LayoutDescriptor.contiguous(shape)), grad),
                Optional.empty(), Optional.of(new MemorySegmentStorage(type, elements, storage)));
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        if (configured == null || configured.isBlank()) {
            throw new IllegalStateException("SYNAPTIK_METAL_TEST_LIBRARY is required");
        }
        Path library = Path.of(configured).toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(library));
        return library;
    }
}
