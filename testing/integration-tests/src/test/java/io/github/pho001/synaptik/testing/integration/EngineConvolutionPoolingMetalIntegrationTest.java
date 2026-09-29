package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalTraceObserver;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.EngineMixedOwnerTestAccess;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool1dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool1dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CopyOnWriteArrayList;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** CPU-free public Engine evidence for Task 0064 convolution and pooling. */
final class EngineConvolutionPoolingMetalIntegrationTest {
    @Test
    void conv1dAndPool1dCompositionsExecuteAsOneMetalOwnedProgram() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                MutableTensor mutableInput =
                        mutableTensor(arena, Shape.of(1, 1, 4), 1, 2, 3, 4);
                Tensor input = mutableInput.tensor();
                Tensor weight = tensor(arena, Shape.of(1, 1, 2), 1, 1);
                Tensor bias = tensor(arena, Shape.of(1), 0.5f);
                Tensor convolution = input.conv1d(weight, bias, Conv1dAttrs.defaults());
                Tensor maximum = input.maxPool1d(
                        new MaxPool1dAttrs(2, 1, 0, 1, false));
                Tensor average = input.averagePool1d(
                        new AveragePool1dAttrs(2, 1, 0, 1, false));

                var compiled = engine.compile(List.of(convolution, maximum, average));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));
                try (var firstSession = engine.session(compiled);
                        var secondSession = engine.session(compiled)) {
                    List<byte[]> original = List.of(
                            floats(3.5f, 5.5f, 7.5f),
                            floats(2, 3, 4),
                            floats(1.5f, 2.5f, 3.5f));
                    List<byte[]> changed = List.of(
                            floats(7.5f, 5.5f, 3.5f),
                            floats(4, 3, 2),
                            floats(3.5f, 2.5f, 1.5f));
                    try (RunResult firstResult =
                            firstSession.run(List.of(bias, weight, input))) {
                        writeFloats(mutableInput.storage(), 4, 3, 2, 1);
                        try (RunResult secondResult =
                                secondSession.run(List.of(input, weight, bias))) {
                            assertOpenResults(secondResult, changed);
                            assertOpenResults(firstResult, original);
                        }
                    }
                    assertResults(
                            firstSession.run(List.of(weight, bias, input)), changed);
                }
            }
        }
    }

    @Test
    void groupedConv2dConv3dAndAllPoolDimensionsPublishThroughMetalOnlyEngine() {
        Path library = configuredMetalLibrary();
        List<ObservedTrace> events = new CopyOnWriteArrayList<>();
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library), traceCollector(events)));
            try (Engine engine = builder.build()) {
                Tensor input2d = tensor(
                        arena, Shape.of(1, 2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8);
                Tensor weight2d = tensor(arena, Shape.of(2, 1, 1, 1), 2, -1);
                Tensor bias2d = tensor(arena, Shape.of(2), 1, 10);
                Tensor conv2d = input2d.conv2d(
                        weight2d, bias2d, new Conv2dAttrs(1, 1, 0, 0, 1, 1, 2));
                Tensor max2d = conv2d.maxPool2d(
                        new MaxPool2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false));
                Tensor average2d = conv2d.averagePool2d(
                        new AveragePool2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false));

                Tensor input3d = tensor(arena, Shape.of(1, 1, 2, 1, 2), 1, 2, 3, 4);
                Tensor weight3d = tensor(arena, Shape.of(1, 1, 1, 1, 1), 2);
                Tensor bias3d = tensor(arena, Shape.of(1), 1);
                Tensor conv3d = input3d.conv3d(weight3d, bias3d, Conv3dAttrs.defaults());
                Tensor max3d = conv3d.maxPool3d(
                        new MaxPool3dAttrs(2, 1, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false));
                Tensor average3d = conv3d.averagePool3d(
                        new AveragePool3dAttrs(
                                2, 1, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false));
                Tensor mixedInput = bfloatTensor(
                        arena, Shape.of(1, 1, 1, 2), 1.5f, -2);
                Tensor mixedWeight = tensor(arena, Shape.of(1, 1, 1, 1), 2);
                Tensor mixedConv = mixedInput.conv2d(mixedWeight, Conv2dAttrs.defaults());

                var outputs = List.of(
                        conv2d, max2d, average2d, conv3d, max3d, average3d, mixedConv);
                var compiled = engine.compile(outputs);
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));
                try (var session = engine.session(compiled)) {
                    assertEquals(2, events.size(), "structural and outcome preparation events");
                    ObservedTrace preparation = events.getFirst();
                    assertEquals("PREPARE", preparation.phase());
                    assertEquals(
                            "CUSTOM_KERNEL",
                            enumName(component(preparation.payload(), "route")),
                            "convolution and pooling graph uses the custom route");
                    assertResults(session.run(List.of(
                            bias3d,
                            mixedInput,
                            input2d,
                            weight3d,
                            mixedWeight,
                            weight2d,
                            input3d,
                            bias2d)), List.of(
                            floats(3, 5, 7, 9, 5, 4, 3, 2),
                            floats(9, 5),
                            floats(6, 3.5f),
                            floats(3, 5, 7, 9),
                            floats(9),
                            floats(6),
                            floats(3, -4)));
                }
            }
        }
    }

    @Test
    void generatedAveragePoolBackwardIsOwnedOnlyForNonOverlappingFoldGeometry() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Tensor input = tensor(
                        arena, Shape.of(1, 1, 2, 2), true, 1, 2, 3, 4);
                Tensor seed = tensor(arena, Shape.of(1, 1, 1, 1), false, 1);
                Tensor average = input.averagePool2d(
                        new AveragePool2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false));
                var compiled = engine.compile(
                        List.of(average), List.of(seed), List.of(input));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));
                try (var session = engine.session(compiled)) {
                    assertResults(session.run(List.of(seed, input)), List.of(
                            floats(2.5f),
                            floats(0.25f, 0.25f, 0.25f, 0.25f)));
                }

                Tensor input3d = tensor(
                        arena, Shape.of(1, 1, 2, 2, 2), true,
                        1, 2, 3, 4, 5, 6, 7, 8);
                Tensor seed3d = tensor(
                        arena, Shape.of(1, 1, 2, 2, 2), false,
                        1, 1, 1, 1, 1, 1, 1, 1);
                Tensor average3d = input3d.averagePool3d(
                        new AveragePool3dAttrs(
                                1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, false));
                var compiled3d = engine.compile(
                        List.of(average3d), List.of(seed3d), List.of(input3d));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled3d));
                try (var session = engine.session(compiled3d)) {
                    assertResults(session.run(List.of(input3d, seed3d)), List.of(
                            floats(1, 2, 3, 4, 5, 6, 7, 8),
                            floats(1, 1, 1, 1, 1, 1, 1, 1)));
                }

                Tensor overlappingInput = tensor(
                        arena, Shape.of(1, 1, 3, 3), true,
                        1, 2, 3, 4, 5, 6, 7, 8, 9);
                Tensor overlapping = overlappingInput.averagePool2d(
                        new AveragePool2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false));
                Tensor overlappingSeed = tensor(
                        arena, Shape.of(1, 1, 2, 2), false, 1, 1, 1, 1);
                assertThrows(IllegalStateException.class,
                        () -> engine.compile(
                                List.of(overlapping),
                                List.of(overlappingSeed),
                                List.of(overlappingInput)));
            }
        }
    }

    @Test
    void generatedConv2dBackwardOwnsSeparateAndJointFloat32SourceRoles() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Tensor inputTarget = tensor(
                        arena, Shape.of(1, 1, 2, 2), true, 1, 2, 3, 4);
                Tensor inputWeight = tensor(arena, Shape.of(1, 1, 1, 1), false, 2);
                Tensor inputBias = tensor(arena, Shape.of(1), false, 1);
                Tensor inputOutput =
                        inputTarget.conv2d(inputWeight, inputBias, Conv2dAttrs.defaults());
                Tensor inputSeed = tensor(
                        arena, Shape.of(1, 1, 2, 2), false, 1, 1, 1, 1);
                var inputCompiled = engine.compile(
                        List.of(inputOutput), List.of(inputSeed), List.of(inputTarget));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(inputCompiled));
                try (var session = engine.session(inputCompiled)) {
                    assertResults(
                            session.run(List.of(inputBias, inputSeed, inputTarget, inputWeight)),
                            List.of(
                                    floats(3, 5, 7, 9),
                                    floats(2, 2, 2, 2)));
                }

                Tensor weightInput = tensor(
                        arena, Shape.of(1, 1, 2, 2), false, 1, 2, 3, 4);
                Tensor weightTarget = tensor(arena, Shape.of(1, 1, 1, 1), true, 2);
                Tensor weightBias = tensor(arena, Shape.of(1), false, 1);
                Tensor weightOutput =
                        weightInput.conv2d(weightTarget, weightBias, Conv2dAttrs.defaults());
                Tensor weightSeed = tensor(
                        arena, Shape.of(1, 1, 2, 2), false, 1, 1, 1, 1);
                var weightCompiled = engine.compile(
                        List.of(weightOutput), List.of(weightSeed), List.of(weightTarget));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(weightCompiled));
                try (var session = engine.session(weightCompiled)) {
                    assertResults(
                            session.run(List.of(weightTarget, weightInput, weightSeed, weightBias)),
                            List.of(
                                    floats(3, 5, 7, 9),
                                    floats(10)));
                }

                Tensor biasInput = tensor(
                        arena, Shape.of(1, 1, 2, 2), false, 1, 2, 3, 4);
                Tensor biasWeight = tensor(arena, Shape.of(1, 1, 1, 1), false, 2);
                Tensor biasTarget = tensor(arena, Shape.of(1), true, 1);
                Tensor biasOutput =
                        biasInput.conv2d(biasWeight, biasTarget, Conv2dAttrs.defaults());
                Tensor biasSeed = tensor(
                        arena, Shape.of(1, 1, 2, 2), false, 1, 1, 1, 1);
                var biasCompiled = engine.compile(
                        List.of(biasOutput), List.of(biasSeed), List.of(biasTarget));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(biasCompiled));
                try (var session = engine.session(biasCompiled)) {
                    assertResults(
                            session.run(List.of(biasTarget, biasInput, biasWeight, biasSeed)),
                            List.of(
                                    floats(3, 5, 7, 9),
                                    floats(4)));
                }

                Tensor jointInput = tensor(
                        arena, Shape.of(1, 1, 2, 2), true, 1, 2, 3, 4);
                Tensor jointWeight = tensor(arena, Shape.of(1, 1, 1, 1), true, 2);
                Tensor jointOutput = jointInput.conv2d(jointWeight, Conv2dAttrs.defaults());
                Tensor jointSeed = tensor(
                        arena, Shape.of(1, 1, 2, 2), false, 1, 1, 1, 1);
                var jointCompiled = engine.compile(
                        List.of(jointOutput),
                        List.of(jointSeed),
                        List.of(jointInput, jointWeight));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(jointCompiled));
                try (var session = engine.session(jointCompiled)) {
                    assertResults(
                            session.run(List.of(jointInput, jointWeight, jointSeed)),
                            List.of(
                                    floats(2, 4, 6, 8),
                                    floats(2, 2, 2, 2),
                                    floats(10)));
                }
            }
        }
    }

    @Test
    void generatedMaximumPoolBackwardClosesWhileConv3dRemainsCompilerFailClosed() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Tensor poolInput = tensor(
                        arena, Shape.of(1, 1, 2, 2), true, 1, 2, 3, 4);
                Tensor maximum = poolInput.maxPool2d(
                        new MaxPool2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false));
                Tensor poolSeed = tensor(arena, Shape.of(1, 1, 1, 1), false, 1);
                var maximumCompiled = engine.compile(
                        List.of(maximum), List.of(poolSeed), List.of(poolInput));
                assertEquals(
                        List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(maximumCompiled));
                try (var session = engine.session(maximumCompiled)) {
                    assertResults(
                            session.run(List.of(poolInput, poolSeed)),
                            List.of(floats(4), floats(0, 0, 0, 1)));
                }

                Tensor convInput = tensor(
                        arena, Shape.of(1, 1, 1, 1, 1), true, 2);
                Tensor convWeight = tensor(
                        arena, Shape.of(1, 1, 1, 1, 1), false, 3);
                Tensor convolution = convInput.conv3d(convWeight, Conv3dAttrs.defaults());
                Tensor convSeed = tensor(
                        arena, Shape.of(1, 1, 1, 1, 1), false, 1);
                IllegalArgumentException failure = assertThrows(
                        IllegalArgumentException.class,
                        () -> engine.compile(
                                List.of(convolution), List.of(convSeed), List.of(convInput)));
                assertTrue(failure.getMessage().contains(
                        "Conv3d is forward-only until Compiler task 0006C closes its gradients"));
            }
        }
    }

    @Test
    void strictProfileExecutesMaximumPoolAndRejectsConvolutionAndAveragePool() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.STRICT_IEEE);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Tensor input = tensor(arena, Shape.of(1, 1, 1, 2), 1, 2);
                Tensor maximum = input.maxPool2d(
                        new MaxPool2dAttrs(1, 2, 1, 1, 0, 0, 1, 1, false));
                var maximumCompiled = engine.compile(List.of(maximum));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(maximumCompiled));
                try (var session = engine.session(maximumCompiled)) {
                    assertResults(session.run(List.of(input)), List.of(floats(2)));
                }

                Tensor average = input.averagePool2d(
                        new AveragePool2dAttrs(1, 2, 1, 1, 0, 0, 1, 1, false));
                assertThrows(IllegalStateException.class,
                        () -> engine.compile(List.of(average)));

                Tensor weight = tensor(arena, Shape.of(1, 1, 1, 1), 2);
                Tensor convolution = input.conv2d(weight, Conv2dAttrs.defaults());
                assertThrows(IllegalStateException.class,
                        () -> engine.compile(List.of(convolution)));
            }
        }
    }

    private static Tensor tensor(Arena arena, Shape shape, float... values) {
        return tensor(arena, shape, false, values);
    }

    private static Tensor tensor(
            Arena arena, Shape shape, boolean requiresGrad, float... values) {
        return mutableTensor(arena, shape, requiresGrad, values).tensor();
    }

    private static MutableTensor mutableTensor(
            Arena arena, Shape shape, float... values) {
        return mutableTensor(arena, shape, false, values);
    }

    private static MutableTensor mutableTensor(
            Arena arena, Shape shape, boolean requiresGrad, float... values) {
        assertEquals(shape.knownElementCount().orElseThrow(), values.length);
        MemorySegment storage = arena.allocate(
                Math.multiplyExact(values.length, Float.BYTES), Float.BYTES);
        writeFloats(storage, values);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
        Tensor tensor = TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.FLOAT32, values.length, storage)));
        return new MutableTensor(tensor, storage);
    }

    private static void writeFloats(MemorySegment storage, float... values) {
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment.copy(source, 0, storage, 0, source.byteSize());
    }

    private static Tensor bfloatTensor(Arena arena, Shape shape, float... values) {
        assertEquals(shape.knownElementCount().orElseThrow(), values.length);
        MemorySegment storage = arena.allocate(
                Math.multiplyExact(values.length, Short.BYTES), Short.BYTES);
        for (int index = 0; index < values.length; index++) {
            storage.setAtIndex(
                    ValueLayout.JAVA_SHORT,
                    index,
                    (short) (Float.floatToRawIntBits(values[index]) >>> 16));
        }
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.BFLOAT16,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.BFLOAT16, values.length, storage)));
    }

    private static byte[] floats(float... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES)
                .order(ByteOrder.BIG_ENDIAN);
        for (float value : values) bytes.putFloat(value);
        return bytes.array();
    }

    private static void assertResults(RunResult result, List<byte[]> expected) {
        try (result) {
            assertOpenResults(result, expected);
        }
    }

    private static void assertOpenResults(RunResult result, List<byte[]> expected) {
        assertEquals(expected.size(), result.resultCount());
        for (int index = 0; index < expected.size(); index++) {
            ByteBuffer bytes = result.materialize(
                    result.publications().get(index), expected.get(index).length).bytes();
            byte[] actual = new byte[bytes.remaining()];
            bytes.get(actual);
            assertArrayEquals(expected.get(index), actual, "publication " + index);
        }
    }

    private static MetalTraceObserver traceCollector(List<ObservedTrace> events) {
        return (MetalTraceObserver) Proxy.newProxyInstance(
                MetalTraceObserver.class.getClassLoader(),
                new Class<?>[] {MetalTraceObserver.class},
                (proxy, method, arguments) -> {
                    if ("onEvent".equals(method.getName())) {
                        Object event = arguments[0];
                        events.add(new ObservedTrace(
                                enumName(component(event, "phase")),
                                component(event, "payload")));
                        return null;
                    }
                    return switch (method.getName()) {
                        case "equals" -> proxy == arguments[0];
                        case "hashCode" -> System.identityHashCode(proxy);
                        case "toString" -> "Task0064 Metal trace collector";
                        default -> throw new AssertionError(
                                "unexpected observer method: " + method.getName());
                    };
                });
    }

    private static Object component(Object value, String name) {
        try {
            return value.getClass().getMethod(name).invoke(value);
        } catch (InvocationTargetException exception) {
            throw new AssertionError(
                    "trace component invocation failed: " + name, exception.getCause());
        } catch (ReflectiveOperationException exception) {
            throw new AssertionError("trace component is unavailable: " + name, exception);
        }
    }

    private static String enumName(Object value) {
        return ((Enum<?>) value).name();
    }

    private record MutableTensor(Tensor tensor, MemorySegment storage) {}

    private record ObservedTrace(String phase, Object payload) {}

    private static Path configuredMetalLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        Assumptions.assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is required for real Metal integration");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(library), "configured Metal library must exist: " + library);
        return library;
    }
}
