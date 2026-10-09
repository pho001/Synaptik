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

/** Public CPU-free Engine execution of direct FLOAT32 and both typed low SIGMOID steps. */
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
    void lowAndExplicitCastFloat32SigmoidShareCustomPartition() {
        float[] samples = {-20.0f, -12.0f, -2.0f, -1.0f, -0.0f, 0.0f, 1.0f, 2.0f,
                Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN};
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
                                ? BFloat16Bits.fromFloat(samples[index])
                                : Float16Bits.fromFloat(samples[index]);
                        storage.setAtIndex(JAVA_SHORT, index, original[index]);
                    }
                    Tensor input = tensor(low, Shape.of(samples.length), false,
                            storage, samples.length);
                    var compiled = engine.compile(List.of(
                            input.sigmoid(), input.cast(DataType.FLOAT32).sigmoid()));
                    assertEquals(List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled));
                    try (var session = engine.session(compiled)) {
                        assertMixedSigmoid(session.run(List.of(input)), low, original);
                        assertMixedSigmoid(session.run(List.of(input)), low, original);
                    }
                    for (int index = 0; index < original.length; index++) {
                        assertEquals(original[index], storage.getAtIndex(JAVA_SHORT, index));
                    }
                }
            }
        }
    }

    private static void assertMixedSigmoid(RunResult result, DataType low, short[] inputs) {
        try (result) {
            assertEquals(2, result.publications().size());
            ByteBuffer lowBytes = result.materialize(result.publications().get(0),
                    (long) inputs.length * Short.BYTES).bytes();
            ByteBuffer f32Bytes = result.materialize(result.publications().get(1),
                    (long) inputs.length * Float.BYTES).bytes();
            for (int index = 0; index < inputs.length; index++) {
                float input = low == DataType.BFLOAT16
                        ? BFloat16Bits.toFloat(inputs[index])
                        : Float16Bits.toFloat(inputs[index]);
                short actualBits = lowBytes.getShort();
                float actual = low == DataType.BFLOAT16
                        ? BFloat16Bits.toFloat(actualBits)
                        : Float16Bits.toFloat(actualBits);
                float f32 = f32Bytes.getFloat();
                if (Float.isNaN(input)) {
                    assertTrue(Float.isNaN(actual) && Float.isNaN(f32));
                } else if (input == 0.0f) {
                    assertEquals(0.5f, actual);
                    assertEquals(0.5f, f32);
                } else if (input == Float.NEGATIVE_INFINITY) {
                    assertEquals(0.0f, actual);
                    assertEquals(0.0f, f32);
                } else if (input == Float.POSITIVE_INFINITY) {
                    assertEquals(1.0f, actual);
                    assertEquals(1.0f, f32);
                } else {
                    double exponent = StrictMath.exp(input < 0.0f
                            ? (double) input : -(double) input);
                    double reference = input < 0.0f
                            ? exponent / (1.0 + exponent) : 1.0 / (1.0 + exponent);
                    double threshold = low == DataType.BFLOAT16 ? 0.0040 : 0.00055;
                    double q = low == DataType.BFLOAT16
                            ? BFloat16Bits.toFloat((short) 1) : Float16Bits.toFloat((short) 1);
                    double normal = low == DataType.BFLOAT16
                            ? BFloat16Bits.toFloat((short) 0x0080)
                            : Float16Bits.toFloat((short) 0x0400);
                    if (reference < normal) {
                        assertTrue(actual >= 0.0f && actual < normal
                                        && Math.abs(actual - reference) <= q,
                                low + " subnormal lane " + index + " input=" + input);
                        if (low == DataType.FLOAT16 && input == -12.0f) {
                            assertTrue(actual > 0.0f, "FLOAT16 -12 must retain subnormal");
                        }
                        if (low == DataType.FLOAT16 && input == -20.0f) {
                            assertEquals(0.0f, actual, "FLOAT16 -20 underflows to +0");
                        }
                    } else if (reference < 1.0 - q) {
                        assertTrue(Math.abs(actual - reference) / reference <= threshold,
                                low + " ordinary lane " + index + " input=" + input);
                    } else {
                        assertTrue(actual > 0.0f && actual <= 1.0f,
                                low + " saturation lane " + index + " input=" + input);
                    }
                    assertTrue(Math.abs(f32 - reference) / reference <= 2e-6,
                            low + " F32 lane " + index + " input=" + input);
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
