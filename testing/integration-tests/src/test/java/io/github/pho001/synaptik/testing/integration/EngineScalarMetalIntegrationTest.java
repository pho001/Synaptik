package io.github.pho001.synaptik.testing.integration;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.EngineMixedOwnerTestAccess;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Public scalar analysis, preparation, execution, and publication through existing Metal routes. */
final class EngineScalarMetalIntegrationTest {
    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "BFLOAT16", "FLOAT16"})
    void standaloneReluAndClampPreserveExactBoundariesAndGradientMetadata(DataType type) {
        standaloneScalars(type, false);
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "BFLOAT16", "FLOAT16"})
    void standaloneMultiplyPreservesArithmeticBoundariesAndGradientMetadata(DataType type) {
        standaloneScalars(type, true);
    }

    private static void standaloneScalars(DataType type, boolean arithmetic) {
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library())));
            try (Engine engine = builder.build()) {
                for (boolean grad : List.of(false, true)) {
                    MemorySegment storage = arena.allocate(type.byteWidth(), type.byteWidth());
                    Tensor input = tensor(type, grad, storage);
                    List<Tensor> outputs = arithmetic ? List.of(
                            input.mul(scalar(type, -2.0f)),
                            input.mul(scalar(type, Float.POSITIVE_INFINITY)),
                            input.mul(scalar(type, -0.0f)),
                            input.mul(scalar(type, 1.5f))) : List.of(input.relu(),
                            input.clamp(scalar(type, -1.0f), scalar(type, 1.0f)),
                            input.clamp(scalar(type, -0.0f), scalar(type, 0.0f)),
                            input.clamp(scalar(type, 0.0f), scalar(type, -0.0f)),
                            input.clamp(scalar(type, Float.NaN), scalar(type, 1.0f)));
                    outputs.forEach(output -> {
                        assertEquals(Shape.scalar(), output.descriptor().shape());
                        assertEquals(grad, output.descriptor().requiresGrad());
                    });
                    var compiled = engine.compile(outputs);
                    assertEquals(List.of("metal"), EngineMixedOwnerTestAccess.partitionOwners(compiled));
                    try (var session = engine.prepare(compiled)) {
                        for (float sample : samples(type)) {
                            write(storage, type, sample);
                            float inputValue = represented(type, sample);
                            float[] expected = arithmetic ? new float[] {inputValue * -2.0f,
                                    inputValue * Float.POSITIVE_INFINITY, inputValue * -0.0f,
                                    inputValue * 1.5f} : new float[] {Math.max(inputValue, 0.0f),
                                    clamp(inputValue, -1.0f, 1.0f), clamp(inputValue, -0.0f, 0.0f),
                                    clamp(inputValue, 0.0f, -0.0f), Float.NaN};
                            try (var result = engine.run(session, List.of(input))) {
                                assertEquals(expected.length, result.publications().size());
                                for (int index = 0; index < expected.length; index++) {
                                    // MUL alone permits same-sign input DAZ/result FTZ. Exact
                                    // RELU/CLAMP must preserve represented subnormal candidates.
                                    int actual = read(result, index, type);
                                    // At subnormal * infinity, allowed input DAZ also permits
                                    // NaN from same-signed zero * infinity, not only infinity.
                                    if (arithmetic && index == 1 && inputValue != 0.0f
                                            && Math.abs(inputValue) < minimumNormal(type)
                                            && Float.isNaN(fromBits(type, actual))) continue;
                                    assertValue(type, expected[index], actual, arithmetic,
                                            type + " grad=" + grad + " input=" + inputValue + " output=" + index);
                                }
                            }
                            assertEquals(bits(type, sample), storedBits(storage, type), "borrowed input");
                        }
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "BFLOAT16", "FLOAT16"})
    void composedScalarMultiplyReluClampReusesSession(DataType type) {
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library())));
            try (Engine engine = builder.build()) {
                for (boolean grad : List.of(false, true)) {
                    MemorySegment storage = arena.allocate(type.byteWidth(), type.byteWidth());
                    Tensor input = tensor(type, grad, storage);
                    Tensor output = input.mul(scalar(type, -2.0f)).relu()
                            .clamp(scalar(type, 0.0f), scalar(type, 3.0f));
                    var compiled = engine.compile(List.of(output));
                    assertEquals(List.of("metal"), EngineMixedOwnerTestAccess.partitionOwners(compiled));
                    try (var session = engine.prepare(compiled)) {
                        for (float sample : new float[] {-2.0f, -0.5f, -0.0f, 0.0f, 0.5f, 2.0f,
                                Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN}) {
                            write(storage, type, sample);
                            float multiplied = represented(type, sample * -2.0f);
                            float expected = clamp(Math.max(multiplied, 0.0f), 0.0f, 3.0f);
                            try (var result = engine.run(session, List.of(input))) {
                                assertValue(type, expected, read(result, 0, type), false,
                                        type + " composed input=" + sample);
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    void metalFirstScalarCpuExpThenMetalReluKeepsMixedOwnership() {
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library())));
            builder.takeOwnership(CpuBackendIntegration.open());
            try (Engine engine = builder.build()) {
                MemorySegment storage = arena.allocate(Float.BYTES, Float.BYTES);
                Tensor input = tensor(DataType.FLOAT32, false, storage);
                var compiled = engine.compile(List.of(input.exp().relu()));
                assertEquals(List.of("cpu", "metal"), EngineMixedOwnerTestAccess.partitionOwners(compiled));
                try (var session = engine.prepare(compiled)) {
                    for (float sample : new float[] {-2.0f, -0.0f, 0.0f, 2.0f,
                            Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN}) {
                        write(storage, DataType.FLOAT32, sample);
                        try (var result = engine.run(session, List.of(input))) {
                            float actual = Float.intBitsToFloat(read(result, 0, DataType.FLOAT32));
                            float expected = (float) StrictMath.exp(sample);
                            if (Float.isFinite(expected) && expected != 0.0f) {
                                assertTrue(Math.abs(actual - expected) / expected <= 2e-6,
                                        "CPU EXP finite relative error for " + sample);
                            } else assertValue(DataType.FLOAT32, expected,
                                    Float.floatToRawIntBits(actual), false, "mixed special " + sample);
                        }
                        assertEquals(bits(DataType.FLOAT32, sample), storedBits(storage, DataType.FLOAT32));
                    }
                }
            }
        }
    }

    private static float[] samples(DataType type) {
        float least = type == DataType.FLOAT32 ? Float.MIN_VALUE
                : type == DataType.BFLOAT16 ? BFloat16Bits.toFloat((short) 1)
                        : Float16Bits.toFloat((short) 1);
        return new float[] {-2.5f, -1.0f, -0.0f, 0.0f, 0.5f, 1.0f, 2.5f, 1.0078125f,
                Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN, least, -least};
    }

    private static float clamp(float input, float lower, float upper) {
        return Math.min(Math.max(input, lower), upper);
    }

    private static void assertValue(DataType type, float expected, int actualBits,
            boolean arithmetic, String context) {
        float actual = fromBits(type, actualBits);
        if (Float.isNaN(expected)) {
            assertTrue(Float.isNaN(actual), context);
        } else if (arithmetic && expected != 0.0f && Float.isFinite(expected)
                && Math.abs(expected) < minimumNormal(type)) {
            int zero = bits(type, Math.copySign(0.0f, expected));
            assertTrue(actualBits == bits(type, expected) || actualBits == zero, context);
        } else assertEquals(bits(type, expected), actualBits, context);
    }

    private static float minimumNormal(DataType type) {
        return type == DataType.FLOAT16 ? 0x1.0p-14f : Float.MIN_NORMAL;
    }

    private static int read(RunResult result, int index, DataType type) {
        ByteBuffer bytes = result.materialize(result.publications().get(index), type.byteWidth()).bytes();
        return type == DataType.FLOAT32 ? bytes.getInt() : Short.toUnsignedInt(bytes.getShort());
    }

    private static void write(MemorySegment storage, DataType type, float value) {
        if (type == DataType.FLOAT32) storage.set(JAVA_INT, 0, bits(type, value));
        else storage.set(JAVA_SHORT, 0, (short) bits(type, value));
    }

    private static int storedBits(MemorySegment storage, DataType type) {
        return type == DataType.FLOAT32 ? storage.get(JAVA_INT, 0)
                : Short.toUnsignedInt(storage.get(JAVA_SHORT, 0));
    }

    private static int bits(DataType type, float value) {
        return switch (type) {
            case FLOAT32 -> Float.floatToRawIntBits(value);
            case BFLOAT16 -> Short.toUnsignedInt(BFloat16Bits.fromFloat(value));
            case FLOAT16 -> Short.toUnsignedInt(Float16Bits.fromFloat(value));
            default -> throw new AssertionError(type);
        };
    }

    private static float fromBits(DataType type, int bits) {
        return switch (type) {
            case FLOAT32 -> Float.intBitsToFloat(bits);
            case BFLOAT16 -> BFloat16Bits.toFloat((short) bits);
            case FLOAT16 -> Float16Bits.toFloat((short) bits);
            default -> throw new AssertionError(type);
        };
    }

    private static float represented(DataType type, float value) {
        return fromBits(type, bits(type, value));
    }

    private static ScalarValue scalar(DataType type, float value) {
        return switch (type) {
            case FLOAT32 -> ScalarValue.float32(value);
            case BFLOAT16 -> ScalarValue.bfloat16(value);
            case FLOAT16 -> ScalarValue.float16(value);
            default -> throw new AssertionError(type);
        };
    }

    private static Tensor tensor(DataType type, boolean grad, MemorySegment storage) {
        Shape shape = Shape.scalar();
        return TensorFactory.create(new TensorDescriptor(type, shape,
                        Optional.of(LayoutDescriptor.contiguous(shape)), grad), Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, 1, storage)));
    }

    private static Path library() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        if (configured == null || configured.isBlank()) {
            throw new IllegalStateException("SYNAPTIK_METAL_TEST_LIBRARY is required");
        }
        Path path = Path.of(configured).toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(path));
        return path;
    }
}
