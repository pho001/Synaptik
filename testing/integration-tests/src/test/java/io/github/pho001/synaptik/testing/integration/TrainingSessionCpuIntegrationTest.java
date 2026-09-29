package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import io.github.pho001.synaptik.training.GradientMode;
import io.github.pho001.synaptik.training.Sgd;
import io.github.pho001.synaptik.training.TrainingSession;
import io.github.pho001.synaptik.training.TrainingState;
import io.github.pho001.synaptik.training.TrainingStep;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Exercises the public training lifecycle through the real CPU Engine composition. */
final class TrainingSessionCpuIntegrationTest {
    @Test
    void repeatedPreparedStepsConvergeAndDetachedResultsOutliveAllOwners() {
        TrainingStep retained;
        float finalValue;
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            ScalarModule module = new ScalarModule(scalar(arena, -4.0f, true));
            Tensor target = scalar(arena, 2.5f, false);
            Tensor difference = module.value().value().sub(target);
            Tensor objective = difference.mul(difference);

            try (TrainingSession session = TrainingSession.open(
                    engine, module, objective, new Sgd(0.1d))) {
                var compiled = session.compiledGraph();
                float firstLoss = Float.NaN;
                retained = null;
                for (int step = 0; step < 40; step++) {
                    retained = session.run(List.of(target), GradientMode.RESET_AND_STEP);
                    assertSame(compiled, session.compiledGraph());
                    if (step == 0) {
                        firstLoss = retained.objective().bytes().getFloat();
                    }
                }
                finalValue = module.value().value().hostStorage().orElseThrow().segment()
                        .get(ValueLayout.JAVA_FLOAT, 0);
                assertTrue(Math.abs(finalValue - 2.5f) < 0.001f, () -> "value=" + finalValue);
                assertTrue(retained.objective().bytes().getFloat() < firstLoss);
                assertEquals(40, retained.executionNumber());
                assertEquals(40, retained.optimizerStep().orElseThrow());
                assertEquals(40, session.executions());
                assertEquals(40, session.optimizerSteps());
            }
        }

        assertTrue(Math.abs(finalValue - 2.5f) < 0.001f);
        assertTrue(retained.objective().bytes().getFloat() < 1e-5f);
    }

    @Test
    void bfloat16TrainingConvergesAccumulatesFailsAtomicallyAndResumes() {
        assertLowPrecisionTrainingLifecycle(DataType.BFLOAT16);
    }

    @Test
    void float16TrainingConvergesAccumulatesFailsAtomicallyAndResumes() {
        assertLowPrecisionTrainingLifecycle(DataType.FLOAT16);
    }

    @Test
    void gatherDataGradientKeepsItsCpuResultAfterCanonicalZeroRepresentation() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor data = tensor(arena, DataType.FLOAT32, true, 1, 2, 3, 4);
            Tensor indices = tensor(arena, DataType.INT64, false, 0, 0, 2);
            Tensor seed = tensor(arena, DataType.FLOAT32, false, 10, 20, 30);
            var compiled = engine.compile(
                    List.of(data.gather(indices, 0)), List.of(seed), List.of(data));
            try (var session = engine.session(compiled);
                    var result = session.run(List.of(data, indices, seed))) {
                var gradient = result.materialize(result.publications().get(1), 4L * Float.BYTES)
                        .bytes();
                assertEquals(30.0f, gradient.getFloat(), 0.0f);
                assertEquals(0.0f, gradient.getFloat(), 0.0f);
                assertEquals(30.0f, gradient.getFloat(), 0.0f);
                assertEquals(0.0f, gradient.getFloat(), 0.0f);
            }
        }
    }

    @Test
    void configuredMetalMixedTrainingSucceeds() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            builder.takeOwnership(CpuBackendIntegration.open());
            try (Engine engine = builder.build()) {
                ScalarModule module = new ScalarModule(vectorOne(arena, 2.0f, true));
                Tensor input = vectorOne(arena, 3.0f, false);
                Tensor objective = module.value().value().neg().add(input).sum();
                try (TrainingSession session = TrainingSession.open(
                        engine, module, objective, new Sgd(0.1d))) {
                    TrainingStep step = session.run(
                            List.of(input), GradientMode.RESET_AND_STEP);
                    assertEquals(2.1f, parameterValue(module), 1e-6f);
                    assertEquals(1, step.optimizerStep().orElseThrow());
                    assertEquals(1, session.executions());
                    assertEquals(1, session.optimizerSteps());
                }
            }
        }
    }

    @Test
    void metalOnlyUnsupportedOperationIsClassifiedBeforeMutation() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                ScalarModule module = new ScalarModule(vectorOne(arena, 2.0f, true));
                Tensor input = vectorOne(arena, 3.0f, false);
                Tensor objective = module.value().value().neg().add(input).sum();

                IllegalStateException unsupported = assertThrows(
                        IllegalStateException.class,
                        () -> TrainingSession.open(engine, module, objective, new Sgd(0.1d)));

                assertTrue(unsupported.getMessage().contains(
                        "no hard-eligible backend is available for ownership selection"));
                assertEquals(2.0f, parameterValue(module), 0.0f);
            }
        }
    }

    private static void assertLowPrecisionTrainingLifecycle(DataType dataType) {
        try (Arena arena = Arena.ofShared(); Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(CpuBackendIntegration.open());
            try (Engine engine = builder.build()) {
                Tensor target = scalar(arena, 2.5f, false);
                Tensor sourceParameter = lowScalar(arena, dataType, -4.0f, true);
                Tensor resumedParameter = lowScalar(arena, dataType, -4.0f, true);
                ScalarModule sourceModule = new ScalarModule(sourceParameter);
                ScalarModule resumedModule = new ScalarModule(resumedParameter);
                Tensor sourceDifference =
                        sourceParameter.cast(DataType.FLOAT32).sub(target);
                Tensor resumedDifference =
                        resumedParameter.cast(DataType.FLOAT32).sub(target);
                Sgd optimizer = new Sgd(0.05d, 0.25d, 0.0d, false);

                try (TrainingSession source = TrainingSession.open(
                                engine,
                                sourceModule,
                                sourceDifference.mul(sourceDifference),
                                optimizer);
                        TrainingSession resumed = TrainingSession.open(
                                engine,
                                resumedModule,
                                resumedDifference.mul(resumedDifference),
                                optimizer)) {
                    short initialBits = rawLow(sourceParameter);
                    source.run(List.of(target), GradientMode.ACCUMULATE);
                    TrainingStep accumulated =
                            source.run(List.of(target), GradientMode.ACCUMULATE);
                    assertTrue(accumulated.optimizerStep().isEmpty());
                    assertEquals(2, accumulated.accumulatedGradientRuns());
                    assertEquals(initialBits, rawLow(sourceParameter));

                    TrainingState checkpoint = source.state();
                    TrainingState.ParameterState pending =
                            checkpoint.parameters().getFirst();
                    assertEquals(Short.BYTES, pending.parameterBytes().remaining());
                    assertEquals(Float.BYTES, pending.masterParameterBytes().remaining());
                    assertEquals(-4.0f, pending.masterParameterBytes().getFloat(), 0.0f);
                    assertEquals(
                            -26.0f,
                            pending.accumulatedGradientBytes().getFloat(),
                            0.0f);

                    resumed.restore(checkpoint);
                    TrainingStep sourceUpdate =
                            source.run(List.of(target), GradientMode.ACCUMULATE_AND_STEP);
                    TrainingStep resumedUpdate =
                            resumed.run(List.of(target), GradientMode.ACCUMULATE_AND_STEP);
                    assertEquals(1, sourceUpdate.optimizerStep().orElseThrow());
                    assertEquals(1, resumedUpdate.optimizerStep().orElseThrow());
                    assertEquals(0, sourceUpdate.accumulatedGradientRuns());
                    assertEquals(0, resumedUpdate.accumulatedGradientRuns());
                    assertTrue(rawLow(sourceParameter) != initialBits);
                    assertEquals(rawLow(sourceParameter), rawLow(resumedParameter));
                    assertEquals(
                            -2.05f,
                            source.state().parameters().getFirst()
                                    .masterParameterBytes().getFloat(),
                            1e-6f);
                    assertStateEquals(source.state(), resumed.state());

                    for (int step = 0; step < 100; step++) {
                        source.run(List.of(target), GradientMode.RESET_AND_STEP);
                        resumed.run(List.of(target), GradientMode.RESET_AND_STEP);
                    }
                    float finalValue = lowValue(sourceParameter);
                    float tolerance = dataType == DataType.BFLOAT16 ? 0.02f : 0.002f;
                    assertTrue(Math.abs(finalValue - 2.5f) <= tolerance);
                    assertEquals(rawLow(sourceParameter), rawLow(resumedParameter));
                    assertStateEquals(source.state(), resumed.state());
                }

                Tensor badParameter = lowScalar(arena, dataType, 1.0f, true);
                Tensor nonFinite = scalar(arena, Float.NaN, false);
                ScalarModule badModule = new ScalarModule(badParameter);
                Tensor badObjective =
                        badParameter.cast(DataType.FLOAT32).mul(nonFinite);
                try (TrainingSession session = TrainingSession.open(
                        engine, badModule, badObjective, new Sgd(0.1d))) {
                    TrainingState before = session.state();
                    short rawBefore = rawLow(badParameter);
                    assertThrows(
                            ArithmeticException.class,
                            () -> session.run(
                                    List.of(nonFinite), GradientMode.RESET_AND_STEP));
                    assertEquals(rawBefore, rawLow(badParameter));
                    assertStateEquals(before, session.state());
                    assertEquals(0, session.executions());
                    assertEquals(0, session.optimizerSteps());
                }
            }
        }
    }

    private static void assertStateEquals(TrainingState expected, TrainingState actual) {
        assertEquals(expected.optimizer(), actual.optimizer());
        assertEquals(expected.executions(), actual.executions());
        assertEquals(expected.optimizerSteps(), actual.optimizerSteps());
        assertEquals(expected.accumulatedGradientRuns(), actual.accumulatedGradientRuns());
        assertEquals(expected.parameters().size(), actual.parameters().size());
        for (int index = 0; index < expected.parameters().size(); index++) {
            TrainingState.ParameterState left = expected.parameters().get(index);
            TrainingState.ParameterState right = actual.parameters().get(index);
            assertEquals(left.path(), right.path());
            assertEquals(left.dataType(), right.dataType());
            assertEquals(left.shape(), right.shape());
            assertArrayEquals(bytes(left.parameterBytes()), bytes(right.parameterBytes()));
            assertArrayEquals(
                    bytes(left.masterParameterBytes()),
                    bytes(right.masterParameterBytes()));
            assertArrayEquals(bytes(left.momentumBytes()), bytes(right.momentumBytes()));
            assertArrayEquals(
                    bytes(left.accumulatedGradientBytes()),
                    bytes(right.accumulatedGradientBytes()));
        }
    }

    private static byte[] bytes(ByteBuffer buffer) {
        byte[] result = new byte[buffer.remaining()];
        buffer.get(result);
        return result;
    }

    private static short rawLow(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment()
                .get(ValueLayout.JAVA_SHORT, 0);
    }

    private static float lowValue(Tensor tensor) {
        short bits = rawLow(tensor);
        return tensor.descriptor().dataType() == DataType.BFLOAT16
                ? BFloat16Bits.toFloat(bits)
                : Float16Bits.toFloat(bits);
    }

    private static float parameterValue(ScalarModule module) {
        return module.value().value().hostStorage().orElseThrow().segment()
                .get(ValueLayout.JAVA_FLOAT, 0);
    }

    private static Path configuredMetalLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        Assumptions.assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is required for real Metal integration");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        Assumptions.assumeTrue(Files.isRegularFile(library),
                "configured Metal library must exist: " + library);
        return library;
    }


    private static Tensor scalar(Arena arena, float value, boolean requiresGrad) {
        Shape shape = Shape.scalar();
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
        MemorySegmentStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 1, arena.allocate(Float.BYTES, Float.BYTES));
        storage.segment().set(ValueLayout.JAVA_FLOAT, 0, value);
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
    }

    private static Tensor lowScalar(
            Arena arena, DataType dataType, float value, boolean requiresGrad) {
        Shape shape = Shape.scalar();
        TensorDescriptor descriptor = new TensorDescriptor(
                dataType,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
        MemorySegmentStorage storage = new MemorySegmentStorage(
                dataType, 1, arena.allocate(Short.BYTES, Short.BYTES));
        short bits = dataType == DataType.BFLOAT16
                ? BFloat16Bits.fromFloat(value)
                : Float16Bits.fromFloat(value);
        storage.segment().set(ValueLayout.JAVA_SHORT, 0, bits);
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
    }

    private static Tensor vectorOne(Arena arena, float value, boolean requiresGrad) {
        Shape shape = Shape.of(1);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
        MemorySegmentStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 1, arena.allocate(Float.BYTES, Float.BYTES));
        storage.segment().set(ValueLayout.JAVA_FLOAT, 0, value);
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
    }

    private static Tensor tensor(
            Arena arena, DataType dataType, boolean requiresGrad, long... values) {
        Shape shape = Shape.of(values.length);
        TensorDescriptor descriptor = new TensorDescriptor(
                dataType,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
        MemorySegmentStorage storage = new MemorySegmentStorage(
                dataType,
                values.length,
                arena.allocate(Math.multiplyExact(values.length, dataType.byteWidth()),
                        dataType.byteWidth()));
        for (int index = 0; index < values.length; index++) {
            if (dataType == DataType.FLOAT32) {
                storage.segment().setAtIndex(
                        ValueLayout.JAVA_FLOAT, index, (float) values[index]);
            } else if (dataType == DataType.INT64) {
                storage.segment().setAtIndex(ValueLayout.JAVA_LONG, index, values[index]);
            } else {
                throw new IllegalArgumentException("unsupported fixture type " + dataType);
            }
        }
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
    }

    private static final class ScalarModule extends Module {
        private final Parameter value;

        private ScalarModule(Tensor value) {
            this.value = parameter("value", value);
        }

        private Parameter value() {
            return value;
        }
    }
}
