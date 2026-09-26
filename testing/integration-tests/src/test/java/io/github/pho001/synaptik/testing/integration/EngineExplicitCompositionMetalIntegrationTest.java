package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalTraceObserver;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.EngineMixedOwnerTestAccess;
import io.github.pho001.synaptik.engine.InferenceSession;
import io.github.pho001.synaptik.engine.ModelAutotuningPreparation;
import io.github.pho001.synaptik.engine.ModelAutotuningRequest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Exercises real CPU-free Metal and mixed CPU/Metal public Engine composition. */
final class EngineExplicitCompositionMetalIntegrationTest {
    @Test
    void publicEngineLifecycleCollectorObservesPreparationAndRepeatedInvocations() {
        Path library = configuredMetalLibrary();
        List<ObservedTrace> events = new CopyOnWriteArrayList<>();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library), traceCollector(events)));
            try (Engine engine = builder.build()) {
                Tensor input = nativeTensorBits(
                        descriptor(Shape.of(2)),
                        arena,
                        Float.floatToRawIntBits(1.0f),
                        Float.floatToRawIntBits(-2.0f));
                var compiled = engine.compile(List.of(input.neg()));
                assertTrue(events.isEmpty(), "compile emits no Metal preparation or run event");

                try (InferenceSession session = engine.session(compiled)) {
                    assertEquals(1, events.size());
                    try (var first = session.run(List.of(input))) {
                        assertRawBits(
                                first.materialize(
                                        first.publications().getFirst(), 2L * Float.BYTES).bytes(),
                                new int[] {
                                    Float.floatToRawIntBits(-1.0f),
                                    Float.floatToRawIntBits(2.0f)
                                });
                    }
                    try (var second = session.run(List.of(input))) {
                        assertRawBits(
                                second.materialize(
                                        second.publications().getFirst(), 2L * Float.BYTES).bytes(),
                                new int[] {
                                    Float.floatToRawIntBits(-1.0f),
                                    Float.floatToRawIntBits(2.0f)
                                });
                    }
                }
            }
        }

        assertEquals(3, events.size());
        assertEquals(List.of(0L, 1L, 2L), events.stream()
                .map(ObservedTrace::eventId)
                .toList());
        Object preparation = events.get(0).payload();
        assertEquals("PREPARE", events.get(0).phase());
        assertEquals("INFO", events.get(0).level());
        assertEquals(0L, idValue(component(preparation, "backendId")));
        assertEquals(0L, idValue(component(preparation, "deviceId")));
        assertEquals(0L, idValue(component(preparation, "preparedUnitId")));
        assertEquals("SUCCEEDED", enumName(component(preparation, "status")));
        assertEquals("STRICT_IEEE", enumName(component(preparation, "profile")));
        assertEquals("CUSTOM_KERNEL", enumName(component(preparation, "route")));
        assertEquals("NOT_QUERIED", enumName(component(preparation, "cacheStatus")));
        assertEquals("SUCCESS", nativeStatusKind(preparation));
        assertEquals(0, nativeStatusCode(preparation));

        for (int index = 1; index < events.size(); index++) {
            Object invocation = events.get(index).payload();
            assertEquals("RUN", events.get(index).phase());
            assertEquals("INFO", events.get(index).level());
            assertEquals(0L, idValue(component(invocation, "preparedUnitId")));
            assertEquals(index - 1L, idValue(component(invocation, "invocationId")));
            assertEquals("SUCCEEDED", enumName(component(invocation, "status")));
            assertEquals("STRICT_IEEE", enumName(component(invocation, "profile")));
            assertEquals("CUSTOM_KERNEL", enumName(component(invocation, "route")));
            assertEquals("SUCCESS", nativeStatusKind(invocation));
            assertEquals(0, nativeStatusCode(invocation));
        }
    }

    @Test
    void subnormalBinaryGraphIsCpuOwnedOrRejectedBeforeMetalPreparation() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared()) {
            TensorDescriptor descriptor = descriptor(Shape.of(2));
            Tensor left = nativeTensorBits(
                    descriptor, arena, 0x00000001, 0x80000001);
            Tensor right = nativeTensorBits(
                    descriptor, arena, 0x00000001, 0x80000001);
            Tensor output = left.add(right);

            try (Engine.Builder builder = Engine.builder()) {
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    IllegalStateException failure = assertThrows(
                            IllegalStateException.class,
                            () -> engine.compile(List.of(output)));
                    assertTrue(failure.getMessage().contains(
                            "no hard-eligible backend is available for ownership selection"));
                }
            }

            try (Engine.Builder builder = Engine.builder()) {
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                builder.takeOwnership(CpuBackendIntegration.open());
                try (Engine engine = builder.build()) {
                    var compiled = engine.compile(List.of(output));
                    assertEquals(
                            List.of("cpu"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled));
                    try (InferenceSession session = engine.session(compiled);
                            var result = session.run(List.of(left, right))) {
                        assertEquals(1, result.resultCount());
                        ByteBuffer canonical = result.materialize(
                                result.publications().getFirst(), 8L).bytes();
                        assertEquals(0x00000002, canonical.getInt());
                        assertEquals(0x80000002, canonical.getInt());
                    }
                }
            }
        }
    }

    @Test
    void cpuFreeAcceleratorMetalRunsAllBinaryOperationsWithinBoundedRawBitSets() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                int[] matrixBits = {
                    0x00800000, 0x80800000, 0x00000001, 0x80000001,
                    0x7f7fffff, 0x7f800000, 0x00000000, 0x80000000
                };
                int[] rowBits = {
                    Float.floatToRawIntBits(0.5f),
                    Float.floatToRawIntBits(2.0f),
                    0x00000001,
                    0x7f7fffff
                };
                int[] columnBits = {
                    Float.floatToRawIntBits(1.0f),
                    Float.floatToRawIntBits(-0.5f)
                };
                int[] scalarBits = {Float.floatToRawIntBits(2.0f)};
                Tensor matrix = nativeTensorBits(
                        descriptor(Shape.of(2, 4)), arena, matrixBits);
                Tensor row = nativeTensorBits(
                        descriptor(Shape.of(4)), arena, rowBits);
                Tensor column = nativeTensorBits(
                        descriptor(Shape.of(2, 1)), arena, columnBits);
                Tensor scalar = nativeTensorBits(
                        descriptor(Shape.of(1)), arena, scalarBits);
                Tensor added = matrix.add(row);
                Tensor subtracted = row.sub(matrix);
                Tensor multiplied = matrix.mul(row);
                Tensor divided = matrix.div(scalar);
                Tensor cancellation = matrix.sub(matrix);
                Tensor chained = added.mul(row);
                Tensor reverseDivided = scalar.div(matrix);
                var compiled = engine.compile(List.of(
                        added,
                        subtracted,
                        multiplied,
                        divided,
                        cancellation,
                        chained,
                        reverseDivided));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));
                List<Tensor> inputs = List.of(matrix, row, scalar);

                InferenceSession first = engine.session(compiled);
                try (InferenceSession independent = engine.session(compiled)) {
                    Object firstPublication;
                    try (var initial = first.run(inputs)) {
                        assertBinaryResults(
                                initial, matrixBits, rowBits, scalarBits);
                        firstPublication = EngineMixedOwnerTestAccess
                                .runOwnedIdentities(initial)
                                .publications()
                                .getFirst();
                    }
                    try (var reused = first.run(inputs)) {
                        assertBinaryResults(
                                reused, matrixBits, rowBits, scalarBits);
                        assertNotSame(firstPublication, EngineMixedOwnerTestAccess
                                .runOwnedIdentities(reused)
                                .publications()
                                .getFirst());
                    }
                    try (var separate = independent.run(inputs)) {
                        assertBinaryResults(
                                separate, matrixBits, rowBits, scalarBits);
                    }
                    first.close();
                    assertTrue(first.isClosed());
                    assertThrows(
                            IllegalStateException.class,
                            () -> first.run(inputs));
                } finally {
                    first.close();
                }

                assertEquals(Shape.of(2, 1), column.descriptor().shape());
                Tensor columnBroadcast = matrix.add(column);
                var columnCompiled = engine.compile(List.of(columnBroadcast));
                try (InferenceSession session = engine.session(columnCompiled);
                        var result = session.run(List.of(matrix, column))) {
                    int[] actual = rawBits(result.materialize(
                            result.publications().getFirst(),
                            (long) matrixBits.length * Integer.BYTES).bytes(),
                            matrixBits.length);
                    for (int lane = 0; lane < matrixBits.length; lane++) {
                        assertAcceleratorAllowed(
                                BinaryArithmeticKind.ADD,
                                matrixBits[lane],
                                columnBits[lane / 4],
                                actual[lane],
                                "column broadcast lane=" + lane);
                    }
                }
            }
        }
    }

    @Test
    void cpuFreeAcceleratorMetalRunsCanonicalReductionsAndScalarMaterialization()
            throws Exception {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared()) {
            Tensor input = nativeTensor(
                    descriptor(Shape.of(2, 3, 4)),
                    arena,
                    1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f, 16.0f,
                    17.0f, 18.0f, 19.0f, 20.0f,
                    21.0f, 22.0f, 23.0f, 24.0f);
            Tensor fullSum = input.sum();
            Tensor fullMean = input.mean();
            Tensor axisMean = input.mean(1);
            Tensor axisSumKeep = input.sum(new int[] {2}, true);
            Tensor multiMean = input.mean(new int[] {2, 0}, false);
            Tensor emptyIdentity = input.sum(new int[0]);
            Tensor sumTo = input.sumToShape(Shape.of(1, 4));
            Tensor composed = axisMean.abs().add(sumTo);
            List<Tensor> publications = List.of(
                    fullSum,
                    fullMean,
                    axisMean,
                    axisSumKeep,
                    multiMean,
                    emptyIdentity,
                    sumTo,
                    composed);

            try (Engine.Builder strictBuilder = Engine.builder()) {
                strictBuilder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine strictEngine = strictBuilder.build()) {
                    IllegalStateException failure = assertThrows(
                            IllegalStateException.class,
                            () -> strictEngine.compile(publications));
                    assertTrue(failure.getMessage().contains(
                            "no hard-eligible backend is available for ownership selection"));
                }
            }

            try (Engine.Builder builder = Engine.builder()) {
                builder.numericalProfile(NumericalProfile.ACCELERATOR);
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    var compiled = engine.compile(publications);
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled));
                    List<Tensor> inputs = List.of(input);

                    InferenceSession first = engine.session(compiled);
                    try (InferenceSession independent = engine.session(compiled)) {
                        Object firstPublication;
                        try (var initial = first.run(inputs)) {
                            assertReductionResults(initial);
                            firstPublication = EngineMixedOwnerTestAccess
                                    .runOwnedIdentities(initial)
                                    .publications()
                                    .getFirst();
                        }
                        try (var reused = first.run(inputs)) {
                            assertReductionResults(reused);
                            assertNotSame(firstPublication, EngineMixedOwnerTestAccess
                                    .runOwnedIdentities(reused)
                                    .publications()
                                    .getFirst());
                        }

                        CountDownLatch ready = new CountDownLatch(2);
                        CountDownLatch start = new CountDownLatch(1);
                        try (var executor = Executors.newFixedThreadPool(2)) {
                            var left = executor.submit(() -> {
                                ready.countDown();
                                start.await();
                                try (var result = first.run(inputs)) {
                                    assertReductionResults(result);
                                }
                                return null;
                            });
                            var right = executor.submit(() -> {
                                ready.countDown();
                                start.await();
                                try (var result = independent.run(inputs)) {
                                    assertReductionResults(result);
                                }
                                return null;
                            });
                            assertTrue(ready.await(10, TimeUnit.SECONDS));
                            start.countDown();
                            left.get(30, TimeUnit.SECONDS);
                            right.get(30, TimeUnit.SECONDS);
                        }

                        first.close();
                        assertTrue(first.isClosed());
                        assertThrows(IllegalStateException.class, () -> first.run(inputs));
                    } finally {
                        first.close();
                    }
                }
            }
        }
    }

    @Test
    void cpuFreeMetalEngineExecutesExactAbsLifecycleUnderBothProfiles() {
        Path library = configuredMetalLibrary();
        int[] inputBits = {
            0x00000000, 0x80000000,
            0x00000001, 0x80000001,
            0x007fffff, 0x807fffff,
            0x00800000, 0x80800000,
            0x00800001, 0x80800001,
            0x3f800000, 0xbf800000,
            0x7f7fffff, 0xff7fffff,
            0x7f800000, 0xff800000,
            0x7fc12345, 0xffc54321,
            0x7f812345, 0xff854321
        };
        for (NumericalProfile profile : NumericalProfile.values()) {
            try (Arena arena = Arena.ofShared();
                    Engine.Builder builder = Engine.builder()) {
                builder.numericalProfile(profile);
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    Tensor input = nativeTensorBits(
                            descriptor(Shape.of(inputBits.length)), arena, inputBits);
                    Tensor direct = input.abs();
                    Tensor intermediate;
                    Tensor composed;
                    List<Tensor> publications;
                    List<Tensor> inputs;
                    if (profile == NumericalProfile.STRICT_IEEE) {
                        intermediate = direct;
                        composed = input.reshape(2, inputBits.length / 2)
                                .contiguous()
                                .abs()
                                .neg()
                                .abs();
                        publications = List.of(direct, composed);
                        inputs = List.of(input);
                    } else {
                        Tensor zeros = nativeTensorBits(
                                descriptor(Shape.of(inputBits.length)),
                                arena,
                                new int[inputBits.length]);
                        intermediate = input.add(zeros);
                        composed = intermediate.abs();
                        publications = List.of(direct, intermediate, composed);
                        inputs = List.of(input, zeros);
                    }
                    var compiled = engine.compile(publications);
                    assertEquals(List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled));

                    InferenceSession reused = engine.session(compiled);
                    try (InferenceSession independent = engine.session(compiled)) {
                        Object firstPublication;
                        try (var first = reused.run(inputs)) {
                            assertAbsEngineResults(profile, first, inputBits);
                            firstPublication = EngineMixedOwnerTestAccess
                                    .runOwnedIdentities(first)
                                    .publications()
                                    .getFirst();
                        }
                        try (var second = reused.run(inputs)) {
                            assertAbsEngineResults(profile, second, inputBits);
                            assertNotSame(firstPublication, EngineMixedOwnerTestAccess
                                    .runOwnedIdentities(second)
                                    .publications()
                                    .getFirst());
                        }
                        try (var separate = independent.run(inputs)) {
                            assertAbsEngineResults(profile, separate, inputBits);
                        }
                        reused.close();
                        assertTrue(reused.isClosed());
                        assertThrows(IllegalStateException.class, () -> reused.run(inputs));
                    } finally {
                        reused.close();
                    }
                }
            }
        }
    }
    @Test
    void cpuFreeMetalEngineRunsAllRemainingExactUnaryOperationsUnderBothProfiles()
            throws Exception {
        Path library = configuredMetalLibrary();
        int[] inputBits = {
            0x00000000, 0x80000000,
            0x00000001, 0x80000001,
            0x007fffff, 0x807fffff,
            0x00800000, 0x80800000,
            0x3f7fffff, 0xbf7fffff,
            0x3fc00000, 0xbfc00000,
            0x7f7fffff, 0xff7fffff,
            0x7f800000, 0xff800000,
            0x7fc12345, 0xff812346
        };
        for (NumericalProfile profile : NumericalProfile.values()) {
            List<ObservedTrace> events = new CopyOnWriteArrayList<>();
            try (Arena arena = Arena.ofShared();
                    Engine.Builder builder = Engine.builder()) {
                builder.numericalProfile(profile);
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library), traceCollector(events)));
                try (Engine engine = builder.build()) {
                    Tensor input = nativeTensorBits(
                            descriptor(Shape.of(inputBits.length)), arena, inputBits);
                    MemorySegment inputBytes = ((MemorySegmentStorage)
                            input.hostStorage().orElseThrow()).segment();
                    var compiled = engine.compile(List.of(
                            input.floor(), input.ceil(), input.sign(), input.relu()));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled));
                    assertTrue(events.isEmpty(), "compile performs no native preparation");

                    InferenceSession first = engine.session(compiled);
                    try (InferenceSession second = engine.session(compiled)) {
                        assertEquals(2, events.size());
                        for (ObservedTrace event : events) {
                            assertEquals("PREPARE", event.phase());
                            assertEquals(
                                    "CUSTOM_KERNEL",
                                    enumName(component(event.payload(), "route")));
                        }

                        assertThrows(
                                IllegalArgumentException.class,
                                () -> first.run(List.of()));
                        try (var recovered = first.run(List.of(input))) {
                            assertRemainingExactUnaryResults(recovered, inputBits);
                        }
                        try (var repeated = first.run(List.of(input))) {
                            assertRemainingExactUnaryResults(repeated, inputBits);
                        }

                        try (var executor = Executors.newFixedThreadPool(2)) {
                            var left = executor.submit(() -> {
                                try (var result = first.run(List.of(input))) {
                                    assertRemainingExactUnaryResults(result, inputBits);
                                }
                            });
                            var right = executor.submit(() -> {
                                try (var result = second.run(List.of(input))) {
                                    assertRemainingExactUnaryResults(result, inputBits);
                                }
                            });
                            left.get(30, TimeUnit.SECONDS);
                            right.get(30, TimeUnit.SECONDS);
                        }

                        first.close();
                        assertTrue(first.isClosed());
                        assertThrows(IllegalStateException.class, () -> first.run(List.of(input)));
                    } finally {
                        first.close();
                    }
                    assertArrayEquals(inputBits, inputBytes.toArray(ValueLayout.JAVA_INT));
                }
            }
        }
    }


    @Test
    void acceleratorMetalEngineRunsNoGradScalarArithmeticAndReciprocal() {
        Path library = configuredMetalLibrary();
        int[] inputBits = {
            0x3f80_0000, 0xc020_0000, 0x0000_0000,
            0x8000_0000, 0x0000_0001, 0x8000_0001,
            0x7f80_0000, 0xff80_0000, 0x7fc1_2345
        };
        int[] scalarBits = {
            0x0000_0000, 0x8000_0000,
            0x0000_0001, 0x8000_0001,
            0x7f80_0000, 0xff80_0000,
            0x7fc1_2345, 0xff81_2346
        };
        List<ObservedTrace> events = new CopyOnWriteArrayList<>();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library), traceCollector(events)));
            try (Engine engine = builder.build()) {
                Tensor input = nativeTensorBits(
                        descriptor(Shape.of(3, 3)), arena, inputBits);
                MemorySegment inputBytes = ((MemorySegmentStorage)
                        input.hostStorage().orElseThrow()).segment();
                List<Tensor> publications = new ArrayList<>();
                for (int scalarBit : scalarBits) {
                    ScalarValue scalar = ScalarValue.float32(
                            Float.intBitsToFloat(scalarBit));
                    publications.add(input.add(scalar));
                    publications.add(input.sub(scalar));
                    publications.add(input.mul(scalar));
                    publications.add(input.div(scalar));
                }
                publications.add(input.reciprocal());
                var compiled = engine.compile(publications);
                assertEquals(
                        List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));
                try (InferenceSession session = engine.session(compiled)) {
                    assertEquals(1, events.size());
                    assertEquals("PREPARE", events.getFirst().phase());
                    assertEquals(
                            "GRAPH_EXECUTABLE",
                            enumName(component(events.getFirst().payload(), "route")));
                    try (var first = session.run(List.of(input))) {
                        assertNoGradScalarResults(first, inputBits, scalarBits);
                    }
                    try (var repeated = session.run(List.of(input))) {
                        assertNoGradScalarResults(repeated, inputBits, scalarBits);
                    }
                    assertArrayEquals(inputBits, inputBytes.toArray(ValueLayout.JAVA_INT));
                }

                Tensor gradInput = nativeTensorBits(
                        descriptor(Shape.of(3, 3), true), arena, inputBits);
                IllegalStateException failure = assertThrows(
                        IllegalStateException.class,
                        () -> engine.compile(List.of(
                                gradInput.add(ScalarValue.float32(2.0f)),
                                gradInput.reciprocal())));
                assertTrue(failure.getMessage().contains(
                        "no hard-eligible backend is available for ownership selection"));
            }
        }
    }

    @Test
    void cpuFreeMetalEngineRunsExactInt32GatherAndBoolOneHotWithBoundsErrors() {
        Path library = configuredMetalLibrary();
        List<ObservedTrace> events = new CopyOnWriteArrayList<>();
        int[] dataBits = {
            0x00000000, 0x80000000, 0x00000001, 0x7fc12345,
            0xffc54321, 0x7f800000, 0x80000001, 0x3f800000
        };
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library), traceCollector(events)));
            try (Engine engine = builder.build()) {
                Tensor data = nativeTensorBits(
                        descriptor(Shape.of(2, 4)), arena, dataBits);
                Tensor gatherIndices = nativeIntTensor(
                        Shape.of(3), arena, 3, 0, 2);
                Tensor oneHotIndices = nativeIntTensor(
                        Shape.of(3), arena, 2, 0, 3);
                MemorySegment gatherIndexBytes = ((MemorySegmentStorage)
                        gatherIndices.hostStorage().orElseThrow()).segment();
                MemorySegment oneHotIndexBytes = ((MemorySegmentStorage)
                        oneHotIndices.hostStorage().orElseThrow()).segment();
                Tensor gathered = data.gather(gatherIndices, 1);
                Tensor oneHot = oneHotIndices.oneHot(4);
                var compiled = engine.compile(List.of(gathered, oneHot));
                assertEquals(
                        List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));

                try (InferenceSession session = engine.session(compiled);
                        var result = session.run(
                                List.of(data, gatherIndices, oneHotIndices))) {
                    assertEquals(2, result.resultCount());
                    assertRawBits(
                            result.materialize(
                                    result.publications().get(0),
                                    6L * Integer.BYTES).bytes(),
                            new int[] {
                                0x7fc12345, 0x00000000, 0x00000001,
                                0x3f800000, 0xffc54321, 0x80000001
                            });
                    ByteBuffer boolBytes = result.materialize(
                            result.publications().get(1), 12L).bytes();
                    byte[] actual = new byte[boolBytes.remaining()];
                    boolBytes.get(actual);
                    assertArrayEquals(new byte[] {
                        0, 0, 1, 0,
                        1, 0, 0, 0,
                        0, 0, 0, 1
                    }, actual);
                }

                gatherIndexBytes.setAtIndex(ValueLayout.JAVA_INT, 1, 4);
                try (InferenceSession session = engine.session(compiled)) {
                    IndexOutOfBoundsException failure = assertThrows(
                            IndexOutOfBoundsException.class,
                            () -> session.run(
                                    List.of(data, gatherIndices, oneHotIndices)));
                    assertEquals(
                            "GATHER index at logical position 1 for data axis 1"
                                    + " is out of bounds: value=4, extent=4",
                            failure.getMessage());
                    ObservedTrace event = events.getLast();
                    Object outcome = event.payload();
                    assertEquals("RUN", event.phase());
                    assertEquals("ERROR", event.level());
                    assertEquals("FAILED", enumName(component(outcome, "status")));
                    assertEquals("RANGE_OUT_OF_BOUNDS", nativeStatusKind(outcome));
                    assertEquals(5, nativeStatusCode(outcome));
                }

                gatherIndexBytes.setAtIndex(ValueLayout.JAVA_INT, 1, 0);
                oneHotIndexBytes.setAtIndex(ValueLayout.JAVA_INT, 1, -1);
                try (InferenceSession session = engine.session(compiled)) {
                    IndexOutOfBoundsException failure = assertThrows(
                            IndexOutOfBoundsException.class,
                            () -> session.run(
                                    List.of(data, gatherIndices, oneHotIndices)));
                    assertEquals(
                            "ONE_HOT index at logical position 1"
                                    + " is out of bounds: value=-1, depth=4",
                            failure.getMessage());
                    ObservedTrace event = events.getLast();
                    Object outcome = event.payload();
                    assertEquals("RUN", event.phase());
                    assertEquals("ERROR", event.level());
                    assertEquals("FAILED", enumName(component(outcome, "status")));
                    assertEquals("RANGE_OUT_OF_BOUNDS", nativeStatusKind(outcome));
                    assertEquals(5, nativeStatusCode(outcome));
                }
            }
        }
    }

    @Test
    void cpuFreeMetalEngineRunsExactReplacementScatterWithBothFailureClasses() {
        Path library = configuredMetalLibrary();
        int[] dataBits = {
            0x00000000, 0x80000000, 0x00000001,
            0x7fa12345, 0xffa54321, 0x3f800000
        };
        int[] updateBits = {0x7fa22222, 0xffa33333, 0x80000001, 0x7f800000};
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Tensor data = nativeTensorBits(
                        descriptor(Shape.of(2, 3)), arena, dataBits);
                Tensor indices = nativeIntTensor(
                        Shape.of(2, 2), arena, 2, 0, 1, 2);
                Tensor updates = nativeTensorBits(
                        descriptor(Shape.of(2, 2)), arena, updateBits);
                MemorySegment dataBytes = ((MemorySegmentStorage)
                        data.hostStorage().orElseThrow()).segment();
                MemorySegment indexBytes = ((MemorySegmentStorage)
                        indices.hostStorage().orElseThrow()).segment();
                MemorySegment updateBytes = ((MemorySegmentStorage)
                        updates.hostStorage().orElseThrow()).segment();
                Tensor scattered = data.scatterElements(indices, updates, 1);
                var compiled = engine.compile(List.of(scattered));
                assertEquals(
                        List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));

                try (InferenceSession session = engine.session(compiled);
                        var result = session.run(List.of(data, indices, updates))) {
                    assertRawBits(
                            result.materialize(
                                    result.publications().getFirst(),
                                    6L * Integer.BYTES).bytes(),
                            new int[] {
                                updateBits[1], dataBits[1], updateBits[0],
                                dataBits[3], updateBits[2], updateBits[3]
                            });
                }
                assertArrayEquals(dataBits, dataBytes.toArray(ValueLayout.JAVA_INT));
                assertArrayEquals(
                        new int[] {2, 0, 1, 2},
                        indexBytes.toArray(ValueLayout.JAVA_INT));
                assertArrayEquals(updateBits, updateBytes.toArray(ValueLayout.JAVA_INT));

                indexBytes.setAtIndex(ValueLayout.JAVA_INT, 1, 3);
                try (InferenceSession session = engine.session(compiled)) {
                    IndexOutOfBoundsException failure = assertThrows(
                            IndexOutOfBoundsException.class,
                            () -> session.run(List.of(data, indices, updates)));
                    assertEquals(
                            "SCATTER_ELEMENTS index at logical position 1 for data axis 1"
                                    + " is out of bounds: value=3, extent=3",
                            failure.getMessage());
                }
                assertArrayEquals(dataBits, dataBytes.toArray(ValueLayout.JAVA_INT));
                assertArrayEquals(updateBits, updateBytes.toArray(ValueLayout.JAVA_INT));

                indexBytes.setAtIndex(ValueLayout.JAVA_INT, 0, 1);
                indexBytes.setAtIndex(ValueLayout.JAVA_INT, 1, 1);
                try (InferenceSession session = engine.session(compiled)) {
                    IllegalArgumentException failure = assertThrows(
                            IllegalArgumentException.class,
                            () -> session.run(List.of(data, indices, updates)));
                    assertEquals(
                            "SCATTER_ELEMENTS duplicate target at logical update position 1;"
                                    + " first addressed at logical update position 0",
                            failure.getMessage());
                }
                assertArrayEquals(dataBits, dataBytes.toArray(ValueLayout.JAVA_INT));
                assertArrayEquals(
                        new int[] {1, 1, 1, 2},
                        indexBytes.toArray(ValueLayout.JAVA_INT));
                assertArrayEquals(updateBits, updateBytes.toArray(ValueLayout.JAVA_INT));
            }
        }
    }

    @Test
    void cpuFreeMetalEngineRunsExactUnfoldAxisUnderBothProfiles() {
        Path library = configuredMetalLibrary();
        int[] inputBits = {
            0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x7f800000, 0xff800000,
            0x7fc12345, 0xffc54321, 0x7f812345, 0xff854321, 0x3f800000, 0xbf800000
        };
        int[] expected = {
            inputBits[0], inputBits[1], inputBits[2],
            inputBits[2], inputBits[3], inputBits[4],
            inputBits[6], inputBits[7], inputBits[8],
            inputBits[8], inputBits[9], inputBits[10]
        };
        for (NumericalProfile profile : NumericalProfile.values()) {
            try (Arena arena = Arena.ofShared();
                    Engine.Builder builder = Engine.builder()) {
                builder.numericalProfile(profile);
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    Tensor input = nativeTensorBits(
                            descriptor(Shape.of(2, 6)), arena, inputBits);
                    MemorySegment inputBytes = ((MemorySegmentStorage)
                            input.hostStorage().orElseThrow()).segment();
                    var compiled = engine.compile(List.of(input.unfold(1, 3, 2)));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled));
                    try (InferenceSession session = engine.session(compiled);
                            var result = session.run(List.of(input))) {
                        assertRawBits(
                                result.materialize(
                                        result.publications().getFirst(),
                                        (long) expected.length * Integer.BYTES).bytes(),
                                expected);
                    }
                    assertArrayEquals(inputBits, inputBytes.toArray(ValueLayout.JAVA_INT));
                }
            }
        }
    }

    @Test
    void task0060PublicEngineRunsExactReplacementFoldsAndAggregateReductions()
            throws Exception {
        Path library = configuredMetalLibrary();
        int[] dataBits = {0x80000000, 0x7fc12345, 0x7f800000, 0x80000001};
        int[] updateBits = {0x00000000, 0xffc54321};
        int[] foldAxisBits = {0x00000001, 0x80000001, 0x7f812345, 0xff854321};
        int[] fold2dBits = {0x80000000, 0x00000000, 0x7fc00041, 0xff800000};
        int[] fold3dBits = {
            0x00000001, 0x80000001, 0x7f800000, 0xff800000,
            0x7fc00042, 0xffc00043, 0x3f800000, 0xbf800000
        };
        for (NumericalProfile profile : NumericalProfile.values()) {
            try (Arena arena = Arena.ofShared();
                    Engine.Builder builder = Engine.builder()) {
                builder.numericalProfile(profile);
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    Tensor data = nativeTensorBits(
                            descriptor(Shape.of(4)), arena, dataBits);
                    Tensor indices = nativeLongTensor(Shape.of(2, 1), arena, 3, 1);
                    Tensor updates = nativeTensorBits(
                            descriptor(Shape.of(2)), arena, updateBits);
                    Tensor foldAxis = nativeTensorBits(
                            descriptor(Shape.of(2, 2)), arena, foldAxisBits);
                    Tensor fold2d = nativeTensorBits(
                            descriptor(Shape.of(1, 4, 1)), arena, fold2dBits);
                    Tensor fold3d = nativeTensorBits(
                            descriptor(Shape.of(1, 8, 1)), arena, fold3dBits);
                    Tensor products = nativeIntTensor(
                            Shape.of(2, 2), arena, Integer.MAX_VALUE, 2, -3, 4);
                    Tensor booleans = nativeBoolTensor(
                            Shape.of(2, 2), arena, 1, 1, 0, 1);
                    Window2dAttrs window2d =
                            new Window2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false);
                    Window3dAttrs window3d =
                            new Window3dAttrs(
                                    2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, false);
                    List<Tensor> publications = List.of(
                            data.scatterNd(indices, updates),
                            data.sliceUpdate(
                                    updates,
                                    new long[] {3},
                                    new int[] {0},
                                    new long[] {-2}),
                            data.sliceUpdate(updates, Shape.of(1)),
                            foldAxis.foldAxis(0, 4, 2),
                            fold2d.fold2d(Shape.of(1, 1, 2, 2), window2d),
                            fold3d.fold3d(Shape.of(1, 1, 2, 2, 2), window3d),
                            products.prod(1),
                            products.prod(new int[0]),
                            booleans.all(1),
                            booleans.any(0),
                            booleans.all(new int[0]));
                    var compiled = engine.compile(publications);
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled));
                    List<Tensor> inputs = List.of(
                            data, indices, updates, foldAxis, fold2d, fold3d, products, booleans);
                    try (InferenceSession session = engine.session(compiled);
                            InferenceSession independent = engine.session(compiled)) {
                        try (var first = session.run(inputs)) {
                            assertTask0060PublicResults(
                                    first,
                                    dataBits,
                                    updateBits,
                                    foldAxisBits,
                                    fold2dBits,
                                    fold3dBits);
                        }
                        try (var repeated = session.run(inputs)) {
                            assertTask0060PublicResults(
                                    repeated,
                                    dataBits,
                                    updateBits,
                                    foldAxisBits,
                                    fold2dBits,
                                    fold3dBits);
                        }
                        try (var separate = independent.run(inputs)) {
                            assertTask0060PublicResults(
                                    separate,
                                    dataBits,
                                    updateBits,
                                    foldAxisBits,
                                    fold2dBits,
                                    fold3dBits);
                        }
                        MemorySegment indexBytes = ((MemorySegmentStorage)
                                indices.hostStorage().orElseThrow()).segment();
                        indexBytes.setAtIndex(ValueLayout.JAVA_LONG, 1, 3L);
                        assertThrows(RuntimeException.class, () -> session.run(inputs));
                        indexBytes.setAtIndex(ValueLayout.JAVA_LONG, 1, 1L);
                        indexBytes.setAtIndex(ValueLayout.JAVA_LONG, 0, 4L);
                        assertThrows(RuntimeException.class, () -> session.run(inputs));
                        indexBytes.setAtIndex(ValueLayout.JAVA_LONG, 0, 3L);
                        try (var recovered = session.run(inputs)) {
                            assertTask0060PublicResults(
                                    recovered,
                                    dataBits,
                                    updateBits,
                                    foldAxisBits,
                                    fold2dBits,
                                    fold3dBits);
                        }


                        CountDownLatch ready = new CountDownLatch(2);
                        CountDownLatch start = new CountDownLatch(1);
                        try (var executor = Executors.newFixedThreadPool(2)) {
                            var left = executor.submit(() -> {
                                ready.countDown();
                                start.await();
                                try (var result = session.run(inputs)) {
                                    assertTask0060PublicResults(
                                            result,
                                            dataBits,
                                            updateBits,
                                            foldAxisBits,
                                            fold2dBits,
                                            fold3dBits);
                                }
                                return null;
                            });
                            var right = executor.submit(() -> {
                                ready.countDown();
                                start.await();
                                try (var result = session.run(inputs)) {
                                    assertTask0060PublicResults(
                                            result,
                                            dataBits,
                                            updateBits,
                                            foldAxisBits,
                                            fold2dBits,
                                            fold3dBits);
                                }
                                return null;
                            });
                            assertTrue(ready.await(10, TimeUnit.SECONDS));
                            start.countDown();
                            left.get(30, TimeUnit.SECONDS);
                            right.get(30, TimeUnit.SECONDS);
                        }
                    }
                    assertArrayEquals(
                            dataBits,
                            ((MemorySegmentStorage) data.hostStorage().orElseThrow())
                                    .segment().toArray(ValueLayout.JAVA_INT));
                    assertArrayEquals(
                            updateBits,
                            ((MemorySegmentStorage) updates.hostStorage().orElseThrow())
                                    .segment().toArray(ValueLayout.JAVA_INT));
                    assertArrayEquals(
                            new long[] {3, 1},
                            ((MemorySegmentStorage) indices.hostStorage().orElseThrow())
                                    .segment().toArray(ValueLayout.JAVA_LONG));
                    assertArrayEquals(
                            new int[] {Integer.MAX_VALUE, 2, -3, 4},
                            ((MemorySegmentStorage) products.hostStorage().orElseThrow())
                                    .segment().toArray(ValueLayout.JAVA_INT));
                    assertArrayEquals(
                            new byte[] {1, 1, 0, 1},
                            ((MemorySegmentStorage) booleans.hostStorage().orElseThrow())
                                    .segment().toArray(ValueLayout.JAVA_BYTE));
                }
            }
        }
    }

    @Test
    void task0060PublicEngineRejectsIntegralAndBoolFolds() {
        Path library = configuredMetalLibrary();
        Window2dAttrs window2d =
                new Window2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false);
        Window3dAttrs window3d =
                new Window3dAttrs(2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, false);
        for (NumericalProfile profile : NumericalProfile.values()) {
            try (Arena arena = Arena.ofShared();
                    Engine.Builder builder = Engine.builder()) {
                builder.numericalProfile(profile);
                Tensor integralAxis =
                        nativeIntTensor(Shape.of(2, 2), arena, 1, 2, 3, 4);
                Tensor integral2d =
                        nativeIntTensor(Shape.of(1, 4, 1), arena, 1, 2, 3, 4);
                Tensor integral3d = nativeIntTensor(
                        Shape.of(1, 8, 1), arena, 1, 2, 3, 4, 5, 6, 7, 8);
                Tensor boolAxis =
                        nativeBoolTensor(Shape.of(2, 2), arena, 0, 1, 1, 0);
                Tensor unsupportedAxis = integralAxis.foldAxis(0, 4, 2);

                IllegalArgumentException integral2dFailure = assertThrows(
                        IllegalArgumentException.class,
                        () -> integral2d.fold2d(Shape.of(1, 1, 2, 2), window2d));
                assertTrue(integral2dFailure.getMessage().contains(
                        "fold2d requires floating input: INT32"));
                IllegalArgumentException integral3dFailure = assertThrows(
                        IllegalArgumentException.class,
                        () -> integral3d.fold3d(Shape.of(1, 1, 2, 2, 2), window3d));
                assertTrue(integral3dFailure.getMessage().contains(
                        "fold3d requires floating input: INT32"));
                IllegalArgumentException boolFailure = assertThrows(
                        IllegalArgumentException.class,
                        () -> boolAxis.foldAxis(0, 4, 2));
                assertTrue(boolFailure.getMessage().contains(
                        "foldAxis requires floating or integral input: BOOL"));

                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    IllegalStateException failure = assertThrows(
                            IllegalStateException.class,
                            () -> engine.compile(List.of(unsupportedAxis)));
                    assertTrue(failure.getMessage().contains(
                            "no hard-eligible backend is available for ownership selection"));
                }
            }
        }
    }

    @Test
    void task0059PublicEngineRunsEveryAdmittedCastIndexAndLayoutOperation()
            throws Exception {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Tensor matrix = nativeTensor(
                        descriptor(Shape.of(2, 3)), arena, 1, 2, 3, 4, 5, 6);
                Tensor indices = nativeIntTensor(
                        Shape.of(2, 2), arena, 2, 0, 1, 1);
                Tensor ndIndices = nativeIntTensor(Shape.of(2, 1), arena, 1, 0);
                Tensor image2d = nativeTensor(
                        descriptor(Shape.of(1, 1, 2, 2)), arena, 1, 2, 3, 4);
                Tensor image3d = nativeTensor(
                        descriptor(Shape.of(1, 1, 2, 2, 2)),
                        arena, 1, 2, 3, 4, 5, 6, 7, 8);
                Tensor scalarInt = nativeIntTensor(Shape.scalar(), arena, 7);
                Tensor scalar = nativeTensor(descriptor(Shape.scalar()), arena, 7);
                Tensor vector = nativeTensor(descriptor(Shape.of(3)), arena, 10, 20, 30);
                Tensor scalarNdIndex = nativeIntTensor(Shape.of(1), arena, 1);
                Window2dAttrs window2d =
                        new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false);
                Window3dAttrs window3d =
                        new Window3dAttrs(
                                2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
                List<Tensor> publications = List.of(
                        indices.cast(DataType.INT64),
                        matrix.gatherElements(indices, -1),
                        matrix.gatherNd(ndIndices),
                        matrix.pad(
                                new long[] {0, 1},
                                new long[] {0, 1},
                                ScalarValue.float32(-0.0f)),
                        Tensor.concat(-2, matrix, matrix),
                        Tensor.stack(-3, matrix, matrix),
                        matrix.tile(1, 2),
                        image2d.unfold2d(window2d, ScalarValue.float32(-0.0f)),
                        image3d.unfold3d(window3d, ScalarValue.float32(-0.0f)),
                        scalarInt.cast(DataType.INT64),
                        scalar.pad(new long[0], new long[0], ScalarValue.float32(-0.0f)),
                        scalar.tile(),
                        Tensor.stack(0, scalar, scalar),
                        vector.gatherNd(scalarNdIndex));
                var compiled = engine.compile(publications);
                assertEquals(
                        List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));
                List<Tensor> inputs = List.of(
                        matrix,
                        indices,
                        ndIndices,
                        image2d,
                        image3d,
                        scalarInt,
                        scalar,
                        vector,
                        scalarNdIndex);

                InferenceSession session = engine.session(compiled);
                try (InferenceSession independent = engine.session(compiled)) {
                    try (var first = session.run(inputs)) {
                        assertTask0059PublicResults(first);
                        assertEquals(
                                first.resultCount(),
                                new HashSet<>(EngineMixedOwnerTestAccess
                                        .runOwnedIdentities(first)
                                        .publications()).size());
                    }
                    try (var repeated = session.run(inputs)) {
                        assertTask0059PublicResults(repeated);
                    }
                    try (var separate = independent.run(inputs)) {
                        assertTask0059PublicResults(separate);
                    }

                    CountDownLatch ready = new CountDownLatch(2);
                    CountDownLatch start = new CountDownLatch(1);
                    try (var executor = Executors.newFixedThreadPool(2)) {
                        var left = executor.submit(() -> {
                            ready.countDown();
                            start.await();
                            try (var result = session.run(inputs)) {
                                assertTask0059PublicResults(result);
                            }
                            return null;
                        });
                        var right = executor.submit(() -> {
                            ready.countDown();
                            start.await();
                            try (var result = session.run(inputs)) {
                                assertTask0059PublicResults(result);
                            }
                            return null;
                        });
                        assertTrue(ready.await(10, TimeUnit.SECONDS));
                        start.countDown();
                        left.get(30, TimeUnit.SECONDS);
                        right.get(30, TimeUnit.SECONDS);
                    }
                } finally {
                    session.close();
                }
                assertTrue(session.isClosed());
                assertThrows(IllegalStateException.class, () -> session.run(inputs));
            }
        }
    }

    @Test
    void task0059PublicGatherFailuresReportExactAxesForBothIndexWidths() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Tensor data = nativeTensor(
                        descriptor(Shape.of(2, 3)), arena, 1, 2, 3, 4, 5, 6);
                Tensor elementIndices = nativeIntTensor(
                        Shape.of(2, 2), arena, 2, 0, 1, 1);
                Tensor ndIndices = nativeLongTensor(Shape.of(2, 1), arena, 1, 0);
                MemorySegment elementBytes = ((MemorySegmentStorage)
                        elementIndices.hostStorage().orElseThrow()).segment();
                MemorySegment ndBytes = ((MemorySegmentStorage)
                        ndIndices.hostStorage().orElseThrow()).segment();
                var compiled = engine.compile(List.of(
                        data.gatherElements(elementIndices, 1),
                        data.gatherNd(ndIndices)));
                assertEquals(
                        List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));

                elementBytes.setAtIndex(ValueLayout.JAVA_INT, 1, -1);
                try (InferenceSession session = engine.session(compiled)) {
                    IndexOutOfBoundsException failure = assertThrows(
                            IndexOutOfBoundsException.class,
                            () -> session.run(List.of(data, elementIndices, ndIndices)));
                    assertEquals(
                            "GATHER_ELEMENTS index at logical position 1 for data axis 1"
                                    + " is out of bounds: value=-1, extent=3",
                            failure.getMessage());
                }

                elementBytes.setAtIndex(ValueLayout.JAVA_INT, 1, 0);
                ndBytes.setAtIndex(ValueLayout.JAVA_LONG, 1, 2L);
                try (InferenceSession session = engine.session(compiled)) {
                    IndexOutOfBoundsException failure = assertThrows(
                            IndexOutOfBoundsException.class,
                            () -> session.run(List.of(data, elementIndices, ndIndices)));
                    assertEquals(
                            "GATHER_ND index at logical position 1 for data axis 0"
                                    + " is out of bounds: value=2, extent=2",
                            failure.getMessage());
                }
            }
        }
    }

    @Test
    void cpuFreeMetalEngineRunsConcurrentAffineCallsWithIsolatedRunOwnership()
            throws Exception {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                int[] matrixBits = {
                        0x00000000, 0x80000000, 0x00000001,
                        0x80012345, 0x7f800000, 0xffc54321
                };
                int[] rowBits = {0x00000001, 0x80000000, 0x7fc12345};
                int[] singletonBits = {
                        0x007fffff, 0x807fffff, 0x00800000,
                        0x80800000, 0x7f7fffff, 0xff7fffff
                };
                Tensor matrix = nativeTensorBits(
                        descriptor(Shape.of(2, 3)), arena, matrixBits);
                Tensor row = nativeTensorBits(
                        descriptor(Shape.of(1, 3)), arena, rowBits);
                Tensor singleton = nativeTensorBits(
                        descriptor(Shape.of(2, 1, 3)), arena, singletonBits);

                var compiled = engine.compile(List.of(
                        matrix.reshape(3, 2),
                        row.expand(2, 3),
                        matrix.permute(1, 0),
                        matrix.expandDims(1),
                        singleton.squeeze(1)));
                assertEquals(3, compiled.inputs().size());

                List<int[]> expected = List.of(
                        matrixBits,
                        new int[] {
                                rowBits[0], rowBits[1], rowBits[2],
                                rowBits[0], rowBits[1], rowBits[2]
                        },
                        new int[] {
                                matrixBits[0], matrixBits[3],
                                matrixBits[1], matrixBits[4],
                                matrixBits[2], matrixBits[5]
                        },
                        matrixBits,
                        singletonBits);
                List<Tensor> inputs = List.of(matrix, row, singleton);
                try (InferenceSession first = engine.session(compiled);
                        InferenceSession second = engine.session(compiled);
                        var executor = Executors.newFixedThreadPool(2)) {
                    CountDownLatch start = new CountDownLatch(1);
                    var leftFuture = executor.submit(() -> {
                        start.await();
                        return first.run(inputs);
                    });
                    var rightFuture = executor.submit(() -> {
                        start.await();
                        return first.run(inputs);
                    });
                    start.countDown();
                    io.github.pho001.synaptik.engine.RunResult left =
                            leftFuture.get(10, TimeUnit.SECONDS);
                    io.github.pho001.synaptik.engine.RunResult right =
                            rightFuture.get(10, TimeUnit.SECONDS);
                    try {
                        assertAffineResults(left, expected);
                        assertAffineResults(right, expected);
                        var leftOwned =
                                EngineMixedOwnerTestAccess.runOwnedIdentities(left);
                        var rightOwned =
                                EngineMixedOwnerTestAccess.runOwnedIdentities(right);
                        assertEquals(5, leftOwned.publications().size());
                        assertEquals(5, rightOwned.publications().size());
                        assertEquals(1, leftOwned.workspaces().size());
                        assertEquals(1, rightOwned.workspaces().size());
                        for (Object leftTarget : leftOwned.publications()) {
                            for (Object rightTarget : rightOwned.publications()) {
                                assertNotSame(leftTarget, rightTarget,
                                        "concurrent runs must own distinct affine targets");
                            }
                        }
                        assertNotSame(
                                leftOwned.workspaces().getFirst(),
                                rightOwned.workspaces().getFirst(),
                                "concurrent runs must own distinct address workspaces");

                        left.close();
                        assertTrue(left.isClosed());
                        assertFalse(right.isClosed());
                        assertAffineResults(right, expected);
                        assertEquals("run result is closed",
                                assertThrows(
                                        IllegalStateException.class,
                                        () -> left.materialize(
                                                left.publications().getFirst(), 24L))
                                        .getMessage());
                    } finally {
                        right.close();
                        left.close();
                    }
                    assertTrue(right.isClosed());

                    assertAffineResults(first, inputs, expected);
                    assertAffineResults(second, inputs, expected);
                }
            }
        }
    }

    @Test
    void cpuFreeAcceleratorPublishesAffineViewsAndRunsMixedExactComposition() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                int[] inputBits = {
                    Float.floatToRawIntBits(1.0f),
                    Float.floatToRawIntBits(-2.0f),
                    Float.floatToRawIntBits(3.0f),
                    Float.floatToRawIntBits(-4.0f),
                    Float.floatToRawIntBits(5.0f),
                    Float.floatToRawIntBits(-6.0f)
                };
                Tensor input = nativeTensorBits(
                        descriptor(Shape.of(6)), arena, inputBits);
                Tensor reshaped = input.reshape(2, 1, 3);
                Tensor expanded = reshaped.expand(2, 4, 3);
                Tensor permuted = expanded.permute(1, 0, 2);
                Tensor rankEdited = permuted.expandDims(2);
                Tensor finalView = rankEdited.squeeze(2);
                Tensor contiguous = finalView.contiguous();
                Tensor negated = contiguous.neg();
                Tensor absolute = negated.abs();
                Tensor added = absolute.add(contiguous.abs());
                Tensor reduced = added.sum(2);
                var compiled = engine.compile(List.of(
                        reshaped,
                        expanded,
                        permuted,
                        rankEdited,
                        finalView,
                        contiguous,
                        negated,
                        absolute,
                        added,
                        reduced));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));

                int[] expandedBits = new int[24];
                for (int batch = 0; batch < 2; batch++) {
                    for (int repeat = 0; repeat < 4; repeat++) {
                        System.arraycopy(
                                inputBits, batch * 3, expandedBits,
                                (batch * 4 + repeat) * 3, 3);
                    }
                }
                int[] permutedBits = new int[24];
                for (int repeat = 0; repeat < 4; repeat++) {
                    for (int batch = 0; batch < 2; batch++) {
                        System.arraycopy(
                                inputBits, batch * 3, permutedBits,
                                (repeat * 2 + batch) * 3, 3);
                    }
                }
                int[] negatedBits = permutedBits.clone();
                int[] absoluteBits = permutedBits.clone();
                int[] addedBits = permutedBits.clone();
                for (int index = 0; index < negatedBits.length; index++) {
                    negatedBits[index] ^= 0x80000000;
                    absoluteBits[index] &= 0x7fffffff;
                    addedBits[index] = Float.floatToRawIntBits(
                            2.0f * Math.abs(Float.intBitsToFloat(permutedBits[index])));
                }
                int[] reducedBits = {
                    Float.floatToRawIntBits(12.0f), Float.floatToRawIntBits(30.0f),
                    Float.floatToRawIntBits(12.0f), Float.floatToRawIntBits(30.0f),
                    Float.floatToRawIntBits(12.0f), Float.floatToRawIntBits(30.0f),
                    Float.floatToRawIntBits(12.0f), Float.floatToRawIntBits(30.0f)
                };
                List<int[]> expected = List.of(
                        inputBits,
                        expandedBits,
                        permutedBits,
                        permutedBits,
                        permutedBits,
                        permutedBits,
                        negatedBits,
                        absoluteBits,
                        addedBits,
                        reducedBits);
                try (InferenceSession session = engine.session(compiled);
                        var result = session.run(List.of(input))) {
                    assertEquals(expected.size(), result.resultCount());
                    for (int index = 0; index < expected.size(); index++) {
                        ByteBuffer bytes = result.materialize(
                                result.publications().get(index),
                                Math.multiplyExact(expected.get(index).length, Integer.BYTES))
                                .bytes();
                        assertRawBits(bytes, expected.get(index));
                    }
                }
            }
        }
    }

    @Test
    void enginePreflightRejectsMetalAffineViewBeforeClosedBackendPreparation() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            MetalBackendIntegration metal = MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library));
            builder.takeOwnership(metal);
            builder.takeOwnership(CpuBackendIntegration.open());
            try (Engine engine = builder.build()) {
                Tensor input = nativeTensorBits(
                        descriptor(Shape.of(1, 3)),
                        arena,
                        0x00000000,
                        0x00000001,
                        0x7fc12345);
                var compiled = engine.compile(List.of(input.expand(2, 3).abs()));
                assertEquals(
                        List.of("metal", "cpu"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));

                metal.close();
                IllegalArgumentException failure = assertThrows(
                        IllegalArgumentException.class,
                        () -> engine.session(compiled));
                assertTrue(failure.getMessage().startsWith(
                        "unsupported cross-owner transfer from metal to cpu for "));
            }
        }
    }

    @Test
    void canonicalCrossOwnerTransfersRunAtRankZeroOneAndSixteen() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            builder.takeOwnership(CpuBackendIntegration.open());
            try (Engine engine = builder.build()) {
                List<Shape> rankBounds = List.of(
                        Shape.of(2),
                        Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2));
                for (Shape shape : rankBounds) {
                    Tensor input = nativeTensor(
                            descriptor(shape), arena, -1.25f, 2.5f);
                    var compiled = engine.compile(List.of(
                            input.neg().exp(),
                            input.exp().neg()));
                    assertEquals(
                            List.of("metal", "cpu", "metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled));
                    try (var session = engine.session(compiled);
                            var result = session.run(List.of(input))) {
                        assertCanonical(
                                result.materialize(
                                        result.publications().get(0), 2L * Float.BYTES).bytes(),
                                strictExp(1.25f), strictExp(-2.5f));
                        assertCanonical(
                                result.materialize(
                                        result.publications().get(1), 2L * Float.BYTES).bytes(),
                                -strictExp(-1.25f), -strictExp(2.5f));
                    }
                }
                Tensor reductionInput =
                        nativeTensor(descriptor(Shape.of(2)), arena, 1.25f, 2.75f);
                var scalarTransfer = engine.compile(List.of(reductionInput.sum().exp()));
                assertEquals(List.of("metal", "cpu"),
                        EngineMixedOwnerTestAccess.partitionOwners(scalarTransfer));
                try (var session = engine.session(scalarTransfer);
                        var result = session.run(List.of(reductionInput))) {
                    assertCanonical(
                            result.materialize(
                                    result.publications().getFirst(), Float.BYTES).bytes(),
                            strictExp(4.0f));
                }
            }
        }
    }

    @Test
    void realMetalLifecycleMixedOwnerTransfersAndCpuTuning(@TempDir Path directory) {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            builder.takeOwnership(CpuBackendIntegration.open());
            try (Engine engine = builder.build()) {
                Shape shape = Shape.of(2);
                TensorDescriptor descriptor = new TensorDescriptor(
                        DataType.FLOAT32,
                        shape,
                        Optional.of(LayoutDescriptor.contiguous(shape)),
                        false);
                Tensor input = nativeTensor(
                        descriptor, arena, 1.25f, -2.5f);

                var metalCompiled = engine.compile(List.of(input.neg()));
                try (var session = engine.session(metalCompiled)) {
                    for (int run = 0; run < 2; run++) {
                        try (var result = session.run(List.of(input))) {
                            ByteBuffer canonical = result.materialize(
                                    result.publications().getFirst(), 8L).bytes();
                            assertEquals(Float.floatToRawIntBits(-1.25f),
                                    canonical.getInt());
                            assertEquals(Float.floatToRawIntBits(2.5f),
                                    canonical.getInt());
                        }
                    }
                }

                assertMixedResult(
                        engine,
                        engine.compile(List.of(input.exp().neg())),
                        List.of("cpu", "metal"),
                        List.of(input),
                        -strictExp(1.25f),
                        -strictExp(-2.5f));
                assertMixedResult(
                        engine,
                        engine.compile(List.of(input.neg().exp())),
                        List.of("metal", "cpu"),
                        List.of(input),
                        strictExp(-1.25f),
                        strictExp(2.5f));
                assertMixedResult(
                        engine,
                        engine.compile(List.of(input.neg().exp().neg())),
                        List.of("metal", "cpu", "metal"),
                        List.of(input),
                        -strictExp(-1.25f),
                        -strictExp(2.5f));

                Tensor cpuInput = TensorFactory.create(
                        descriptor,
                        Optional.empty(),
                        Optional.of(new MemorySegmentStorage(
                                DataType.FLOAT32, 2, arena.allocate(8, Float.BYTES))));
                var cpuCompiled = engine.compile(List.of(cpuInput.exp()));
                ModelAutotuningRequest cpuRequest = tuningRequest(
                        directory,
                        cpuInput,
                        ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC);
                ModelAutotuningPreparation preparation =
                        engine.prepareTuned(cpuCompiled, cpuRequest);
                assertEquals(ModelAutotuningPreparation.Outcome.SAFE_HEURISTIC_FALLBACK,
                        preparation.outcome());
                try (var tuned = preparation.preparedExecution();
                        var result = engine.run(tuned, List.of(cpuInput))) {
                    assertEquals(1, result.resultCount());
                }
                Tensor capturedBase = input.exp();
                Tensor metalPublication = capturedBase.neg();
                Tensor cpuPublication = metalPublication.exp();
                assertCapturedAdaptersSurviveRegistryPoison(
                        engine,
                        engine.compile(List.of(metalPublication, cpuPublication)),
                        List.of("cpu", "metal", "cpu"),
                        List.of(input));
            }
        }
    }

    @Test
    void publicMetalRouteTuningPreparesAndRunsSingletonNeg(@TempDir Path directory) {
        Path library = configuredMetalLibrary();
        Path workloadCache = directory.resolve("metal-workload.bin");
        Path modelPlanCache = directory.resolve("metal-model-plan.bin");
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Tensor input = nativeTensor(
                        descriptor(Shape.of(2)), arena, 1.25f, -2.5f);
                var compiled = engine.compile(List.of(input.neg()));
                var config = new ModelAutotuningConfig(
                        ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                        new ModelAutotuningConfig.Budget(1, 2, 0, 1),
                        new ModelAutotuningConfig.RepresentativeProfileIdentity(
                                1, new byte[] {2}),
                        ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT,
                        workloadCache,
                        new ModelAutotuningConfig.CompletePlanBudget(1, 0, 1, 2L, 8L),
                        modelPlanCache);
                var request = new ModelAutotuningRequest(
                        config,
                        new ModelAutotuningRequest.ModelIdentity(1, new byte[] {1}),
                        List.of(input));

                ModelAutotuningPreparation preparation =
                        engine.prepareTuned(compiled, request);
                assertEquals(ModelAutotuningPreparation.Outcome.TUNED,
                        preparation.outcome());
                var evidence = preparation.evidence().orElseThrow();
                assertEquals(1, evidence.workloads().size());
                var local = evidence.workloads().getFirst();
                assertEquals(ModelAutotuningPreparation.Source.MEASURED, local.source());
                assertEquals(ModelAutotuningPreparation.ReuseScope.SESSION,
                        local.compatibility().reuseScope());
                assertEquals(2, local.candidates().size());
                assertEquals(ModelAutotuningPreparation.Source.MEASURED,
                        evidence.completePlan().source());
                assertEquals(ModelAutotuningPreparation.ReuseScope.SESSION,
                        evidence.completePlan().compatibility().reuseScope());
                assertEquals(1, evidence.completePlan().candidates().size());
                assertFalse(Files.exists(workloadCache));
                assertFalse(Files.exists(modelPlanCache));

                try (var tuned = preparation.preparedExecution();
                        var result = engine.run(tuned, List.of(input))) {
                    assertEquals(1, result.resultCount());
                    assertCanonical(
                            result.materialize(
                                    result.publications().getFirst(), 8L).bytes(),
                            -1.25f, 2.5f);
                }
            }
        }
    }

    @Test
    void cpuFreeAcceleratorMetalRunsRankTwoMatmulLinearAndSeededGradients() {
        Path library = configuredMetalLibrary();
        float[] leftValues = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float[] rightValues = {
                7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f,
                15.0f, 16.0f, 17.0f, 18.0f
        };
        float[] weightValues = {
                7.0f, 11.0f, 15.0f,
                8.0f, 12.0f, 16.0f,
                9.0f, 13.0f, 17.0f,
                10.0f, 14.0f, 18.0f
        };
        float[] seedValues = {
                1.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f
        };
        try (Arena arena = Arena.ofShared()) {
            Tensor left = nativeTensor(
                    descriptor(Shape.of(2, 3), true), arena, leftValues);
            Tensor right = nativeTensor(
                    descriptor(Shape.of(3, 4), true), arena, rightValues);
            Tensor weight = nativeTensor(
                    descriptor(Shape.of(4, 3)), arena, weightValues);
            Tensor seed = nativeTensor(
                    descriptor(Shape.of(2, 4)), arena, seedValues);
            Tensor output = left.matmul(right);
            Tensor linear = left.linear(weight);

            try (Engine.Builder strictBuilder = Engine.builder()) {
                strictBuilder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine strictEngine = strictBuilder.build()) {
                    IllegalStateException failure = assertThrows(
                            IllegalStateException.class,
                            () -> strictEngine.compile(List.of(output)));
                    assertTrue(failure.getMessage().contains(
                            "no hard-eligible backend is available for ownership selection"));
                }
            }

            try (Engine.Builder builder = Engine.builder()) {
                builder.numericalProfile(NumericalProfile.ACCELERATOR);
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    var gradientCompiled = engine.compile(
                            List.of(output), List.of(seed), List.of(left, right));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(gradientCompiled));
                    assertEquals(
                            List.of(left.id(), right.id(), seed.id()),
                            gradientCompiled.inputs().stream()
                                    .map(input -> input.tensorId())
                                    .toList());
                    assertMatmulGradientGraph(
                            EngineMixedOwnerTestAccess.compileArtifacts(gradientCompiled),
                            left,
                            right,
                            seed);

                    List<Tensor> gradientInputs = List.of(seed, right, left);
                    InferenceSession reused = engine.session(gradientCompiled);
                    try (InferenceSession independent = engine.session(gradientCompiled)) {
                        for (int run = 0; run < 2; run++) {
                            try (var result = reused.run(gradientInputs)) {
                                assertMatmulGradientResults(result);
                            }
                        }
                        try (var result = independent.run(gradientInputs)) {
                            assertMatmulGradientResults(result);
                        }
                        reused.close();
                        assertTrue(reused.isClosed());
                        assertThrows(
                                IllegalStateException.class,
                                () -> reused.run(gradientInputs));
                    } finally {
                        reused.close();
                    }

                    var linearCompiled = engine.compile(List.of(linear));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(linearCompiled));
                    CompiledGraphModel linearGraph =
                            EngineMixedOwnerTestAccess.compileArtifacts(linearCompiled).graph();
                    assertEquals(
                            List.of(AxisTransformKind.PERMUTE, MatmulKind.MATMUL),
                            linearGraph.nodes().stream()
                                    .map(node -> node.operation().kind())
                                    .toList());
                    try (InferenceSession session = engine.session(linearCompiled);
                            var result = session.run(List.of(left, weight))) {
                        assertPublication(result, 0,
                                74.0f, 80.0f, 86.0f, 92.0f,
                                173.0f, 188.0f, 203.0f, 218.0f);
                    }

                    assertTensorBitsUnchanged(left, leftValues);
                    assertTensorBitsUnchanged(right, rightValues);
                    assertTensorBitsUnchanged(weight, weightValues);
                    assertTensorBitsUnchanged(seed, seedValues);
                }
            }
        }
    }
    @Test
    void publicMetalEngineRunsGeneralMatmulPromotionAndBatchedGradientDomain() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared()) {
            Tensor vectorLeft = nativeTensor(
                    descriptor(Shape.of(3)), arena, 1, 2, 3);
            Tensor vectorRight = nativeTensor(
                    descriptor(Shape.of(3)), arena, 4, 5, 6);
            Tensor mixedSource = nativeBfloatTensor(
                    Shape.of(2, 2), arena, 1, 3, 2, 4);
            Tensor mixedLeft = mixedSource.permute(1, 0);
            Tensor mixedRight = nativeTensor(
                    descriptor(Shape.of(2, 1)), arena, 5, 6);
            Tensor batchedLeft = nativeTensor(
                    descriptor(Shape.of(2, 2, 3), true),
                    arena,
                    1, 2, 3, 4, 5, 6,
                    7, 8, 9, 10, 11, 12);
            Tensor batchedLinearLeft = nativeTensor(
                    descriptor(Shape.of(2, 2, 3)),
                    arena,
                    1, 2, 3, 4, 5, 6,
                    7, 8, 9, 10, 11, 12);
            Tensor batchedRight = nativeTensor(
                    descriptor(Shape.of(1, 3, 2), true),
                    arena,
                    1, 0, 0, 1, 1, 1);
            Tensor seed = nativeTensor(
                    descriptor(Shape.of(2, 2, 2)),
                    arena,
                    1, 1, 1, 1, 1, 1, 1, 1);
            Tensor batched = batchedLeft.matmul(batchedRight);
            Tensor explicitSource = nativeTensor(
                    descriptor(Shape.of(3, 2)), arena, 1, 4, 2, 5, 3, 6);
            Tensor explicitRight = nativeTensor(
                    descriptor(Shape.of(3, 1)), arena, 1, 10, 100);
            Tensor nestedLeft = nativeTensor(
                    descriptor(Shape.of(2, 2)), arena, 1.25f, 2, 3, 4);
            Tensor nestedRight = nativeTensor(
                    descriptor(Shape.of(2, 2)), arena, 1, 0, 0, 1);
            Tensor linearWeight = nativeTensor(
                    descriptor(Shape.of(2, 3)), arena, 1, 0, 1, 0, 1, 1);
            Tensor vectorMatrixLeft = nativeTensor(
                    descriptor(Shape.of(3), true), arena, 1, 2, 3);
            Tensor vectorMatrixRight = nativeTensor(
                    descriptor(Shape.of(3, 2), true), arena, 1, 2, 3, 4, 5, 6);
            Tensor vectorMatrixSeed = nativeTensor(
                    descriptor(Shape.of(2)), arena, 1, 10);
            Tensor matrixVectorLeft = nativeTensor(
                    descriptor(Shape.of(2, 3), true), arena, 1, 2, 3, 4, 5, 6);
            Tensor matrixVectorRight = nativeTensor(
                    descriptor(Shape.of(3), true), arena, 1, 2, 3);
            Tensor matrixVectorSeed = nativeTensor(
                    descriptor(Shape.of(2)), arena, 1, 10);
            Tensor dotLeft = nativeTensor(
                    descriptor(Shape.of(3), true), arena, 1, 2, 3);
            Tensor dotRight = nativeTensor(
                    descriptor(Shape.of(3), true), arena, 4, 5, 6);
            Tensor dotSeed = nativeTensor(
                    descriptor(Shape.scalar()), arena, 2);
            Tensor linearBias = nativeTensor(
                    descriptor(Shape.of(2)), arena, 10, 20);

            try (Engine.Builder builder = Engine.builder()) {
                builder.numericalProfile(NumericalProfile.ACCELERATOR);
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    var forward = engine.compile(List.of(
                            vectorLeft.matmul(vectorRight),
                            mixedLeft.matmul(mixedRight)));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(forward));
                    try (InferenceSession session = engine.session(forward);
                            var result = session.run(List.of(
                                    vectorLeft, vectorRight, mixedSource, mixedRight))) {
                        assertPublication(result, 0, 32);
                        assertPublication(result, 1, 17, 39);
                    }

                    Tensor dot = dotLeft.matmul(dotRight);
                    var dotGradients = engine.compile(
                            List.of(dot),
                            List.of(dotSeed),
                            List.of(dotLeft, dotRight));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(dotGradients));
                    try (InferenceSession session = engine.session(dotGradients);
                            var result = session.run(List.of(dotSeed, dotRight, dotLeft))) {
                        assertPublication(result, 0, 32);
                        assertPublication(result, 1, 8, 10, 12);
                        assertPublication(result, 2, 2, 4, 6);
                    }

                    var gradients = engine.compile(
                            List.of(batched),
                            List.of(seed),
                            List.of(batchedLeft, batchedRight));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(gradients));
                    try (InferenceSession session = engine.session(gradients);
                            var result = session.run(List.of(
                                    seed, batchedRight, batchedLeft))) {
                        assertEquals(3, result.resultCount());
                        assertPublication(
                                result, 0,
                                4, 5, 10, 11,
                                16, 17, 22, 23);
                        assertPublication(
                                result, 1,
                                1, 1, 2, 1, 1, 2,
                                1, 1, 2, 1, 1, 2);
                        assertPublication(
                                result, 2,
                                22, 22, 26, 26, 30, 30);
                    }

                    var composed = engine.compile(List.of(
                            explicitSource.permute(1, 0).matmul(explicitRight),
                            nestedLeft.matmul(nestedRight).floor(),
                            batchedLinearLeft.linear(linearWeight, linearBias)));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(composed));
                    try (InferenceSession session = engine.session(composed);
                            var result = session.run(List.of(
                                    explicitSource,
                                    explicitRight,
                                    nestedLeft,
                                    nestedRight,
                                    batchedLinearLeft,
                                    linearWeight,
                                    linearBias))) {
                        assertPublication(result, 0, 321, 654);
                        assertPublication(result, 1, 1, 2, 3, 4);
                        assertPublication(
                                result, 2,
                                14, 25, 20, 31,
                                26, 37, 32, 43);
                    }

                    Tensor vectorMatrix = vectorMatrixLeft.matmul(vectorMatrixRight);
                    var vectorMatrixGradients = engine.compile(
                            List.of(vectorMatrix),
                            List.of(vectorMatrixSeed),
                            List.of(vectorMatrixLeft, vectorMatrixRight));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(vectorMatrixGradients));
                    try (InferenceSession session = engine.session(vectorMatrixGradients);
                            var result = session.run(List.of(
                                    vectorMatrixSeed, vectorMatrixRight, vectorMatrixLeft))) {
                        assertPublication(result, 0, 22, 28);
                        assertPublication(result, 1, 21, 43, 65);
                        assertPublication(result, 2, 1, 10, 2, 20, 3, 30);
                    }

                    Tensor matrixVector = matrixVectorLeft.matmul(matrixVectorRight);
                    var matrixVectorGradients = engine.compile(
                            List.of(matrixVector),
                            List.of(matrixVectorSeed),
                            List.of(matrixVectorLeft, matrixVectorRight));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(matrixVectorGradients));
                    try (InferenceSession session = engine.session(matrixVectorGradients);
                            var result = session.run(List.of(
                                    matrixVectorSeed, matrixVectorRight, matrixVectorLeft))) {
                        assertPublication(result, 0, 14, 32);
                        assertPublication(result, 1, 1, 2, 3, 10, 20, 30);
                        assertPublication(result, 2, 41, 52, 63);
                    }
                }
            }

            Tensor intLeftSource = nativeIntTensor(
                    Shape.of(2, 1), arena, Integer.MAX_VALUE, 2);
            Tensor intLeft = intLeftSource.permute(1, 0);
            Tensor intRight = nativeIntTensor(Shape.of(2, 1), arena, 2, 3);
            Tensor promotedLeft = nativeIntTensor(Shape.of(1, 2), arena, -2, 3);
            Tensor promotedRight = nativeLongTensor(
                    Shape.of(2, 1), arena, Long.MAX_VALUE, 3);
            try (Engine.Builder builder = Engine.builder()) {
                builder.numericalProfile(NumericalProfile.ACCELERATOR);
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    var compiled = engine.compile(List.of(
                            intLeft.matmul(intRight),
                            promotedLeft.matmul(promotedRight)));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled));
                    try (InferenceSession session = engine.session(compiled);
                            var result = session.run(List.of(
                                    intLeftSource, intRight, promotedLeft, promotedRight))) {
                        ByteBuffer intResult = result.materialize(
                                result.publications().get(0), Integer.BYTES).bytes();
                        ByteBuffer longResult = result.materialize(
                                result.publications().get(1), Long.BYTES).bytes();
                        assertEquals(4, intResult.getInt());
                        assertEquals(11L, longResult.getLong());
                    }
                }
            }

            try (Engine.Builder builder = Engine.builder()) {
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    var compiled = engine.compile(List.of(
                            intLeft.matmul(intRight),
                            promotedLeft.matmul(promotedRight)));
                    assertEquals(
                            List.of("metal"),
                            EngineMixedOwnerTestAccess.partitionOwners(compiled));
                    try (InferenceSession session = engine.session(compiled);
                            var result = session.run(List.of(
                                    intLeftSource, intRight, promotedLeft, promotedRight))) {
                        ByteBuffer intResult = result.materialize(
                                result.publications().get(0), Integer.BYTES).bytes();
                        ByteBuffer longResult = result.materialize(
                                result.publications().get(1), Long.BYTES).bytes();
                        assertEquals(4, intResult.getInt());
                        assertEquals(11L, longResult.getLong());
                    }
                }
            }
        }
    }
    @Test
    void publicMetalOnlyRejectsEveryExcludedMatmulTypeProfileGradientAndLayoutForm() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared()) {
            Tensor floatLeft = nativeTensor(
                    descriptor(Shape.of(2, 2)), arena, 1, 2, 3, 4);
            Tensor floatRight = nativeTensor(
                    descriptor(Shape.of(2, 1)), arena, 5, 6);
            Tensor bfloatLeft = nativeBfloatTensor(
                    Shape.of(2, 2), arena, 1, 2, 3, 4);
            Tensor bfloatRight = nativeBfloatTensor(
                    Shape.of(2, 1), arena, 5, 6);
            Tensor mixedGradientLeft = nativeBfloatTensor(
                    Shape.of(2, 2), true, arena, 1, 2, 3, 4);
            Tensor doubleLeft = nativeDoubleTensor(
                    Shape.of(2, 2), arena, 1, 2, 3, 4);
            Tensor doubleRight = nativeDoubleTensor(
                    Shape.of(2, 1), arena, 5, 6);
            Tensor bfloatGradientSeed = nativeBfloatTensor(
                    Shape.of(2, 1), arena, 1, 1);
            Tensor doubleGradientLeft = nativeDoubleTensor(
                    Shape.of(2, 2), true, arena, 1, 2, 3, 4);
            Tensor doubleGradientSeed = nativeDoubleTensor(
                    Shape.of(2, 1), arena, 1, 1);
            Shape zeroLeftShape = Shape.of(2, 0);
            Shape zeroRightShape = Shape.of(0, 1);
            Tensor zeroLeft = symbolic(new TensorDescriptor(
                    DataType.FLOAT32,
                    zeroLeftShape,
                    Optional.of(LayoutDescriptor.contiguous(zeroLeftShape)),
                    false));
            Tensor zeroRight = symbolic(new TensorDescriptor(
                    DataType.FLOAT32,
                    zeroRightShape,
                    Optional.of(LayoutDescriptor.contiguous(zeroRightShape)),
                    false));
            Shape forgedShape = Shape.of(2, 2);
            Tensor forgedLayout = symbolic(new TensorDescriptor(
                    DataType.FLOAT32,
                    forgedShape,
                    Optional.of(LayoutDescriptor.of(
                            forgedShape, new long[] {3, 1}, 0L, true)),
                    false));

            try (Engine.Builder builder = Engine.builder()) {
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    assertThrows(
                            IllegalStateException.class,
                            () -> engine.compile(List.of(floatLeft.matmul(floatRight))),
                            "strict floating MATMUL has no Metal fallback");
                }
            }

            try (Engine.Builder builder = Engine.builder()) {
                builder.numericalProfile(NumericalProfile.ACCELERATOR);
                builder.takeOwnership(MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library)));
                try (Engine engine = builder.build()) {
                    assertThrows(
                            IllegalStateException.class,
                            () -> engine.compile(List.of(bfloatLeft.matmul(bfloatRight))));
                    assertThrows(
                            IllegalStateException.class,
                            () -> engine.compile(List.of(doubleLeft.matmul(doubleRight))));
                    assertThrows(
                            IllegalStateException.class,
                            () -> engine.compile(
                                    List.of(mixedGradientLeft.matmul(bfloatRight)),
                                    List.of(bfloatGradientSeed),
                                    List.of(mixedGradientLeft)));
                    assertThrows(
                            IllegalStateException.class,
                            () -> engine.compile(
                                    List.of(doubleGradientLeft.matmul(doubleRight)),
                                    List.of(doubleGradientSeed),
                                    List.of(doubleGradientLeft)));
                    assertThrows(
                            IllegalStateException.class,
                            () -> engine.compile(List.of(
                                    mixedGradientLeft.matmul(floatRight))));
                    assertThrows(
                            IllegalStateException.class,
                            () -> engine.compile(List.of(zeroLeft.matmul(zeroRight))));
                    assertThrows(
                            IllegalStateException.class,
                            () -> engine.compile(List.of(forgedLayout.matmul(floatRight))));
                }
            }
        }
    }



    private static void assertMatmulGradientGraph(
            CompileArtifacts artifacts, Tensor left, Tensor right, Tensor seed) {
        CompiledGraphModel graph = artifacts.graph();
        ValueId leftInput = inputValue(artifacts, left);
        ValueId rightInput = inputValue(artifacts, right);
        ValueId seedInput = inputValue(artifacts, seed);

        CompiledNode forward = producer(
                graph, artifacts.publication().forwardBindings().getFirst().valueId());
        assertEquals(MatmulKind.MATMUL, forward.operation().kind());
        assertEquals(List.of(leftInput, rightInput), forward.inputs());
        assertEquals(GraphPhase.FORWARD, graph.nodePhases().get(forward.id()));

        var gradients = artifacts.publication().gradientBindings();
        assertEquals(List.of(left.id(), right.id()),
                gradients.stream().map(binding -> binding.target()).toList());
        assertEquals(7, graph.nodes().size());
        assertEquals(6, graph.nodes().stream()
                .filter(node -> graph.nodePhases().get(node.id()) == GraphPhase.BACKWARD)
                .count());

        CompiledNode leftBoundary = producer(graph, gradients.get(0).valueId());
        assertShapeRestoration(leftBoundary);
        CompiledNode leftGradient = producer(graph, leftBoundary.inputs().getFirst());
        assertEquals(MatmulKind.MATMUL, leftGradient.operation().kind());
        assertEquals(seedInput, leftGradient.inputs().get(0));
        assertExactTranspose(graph, leftGradient.inputs().get(1), rightInput);

        CompiledNode rightBoundary = producer(graph, gradients.get(1).valueId());
        assertShapeRestoration(rightBoundary);
        CompiledNode rightGradient = producer(graph, rightBoundary.inputs().getFirst());
        assertEquals(MatmulKind.MATMUL, rightGradient.operation().kind());
        assertExactTranspose(graph, rightGradient.inputs().get(0), leftInput);
        assertEquals(seedInput, rightGradient.inputs().get(1));
    }

    private static void assertShapeRestoration(CompiledNode boundary) {
        assertEquals(AggregateReductionKind.SUM, boundary.operation().kind());
        assertTrue(boundary.operation().attrs() instanceof SumToShapeAttrs);
    }

    private static void assertExactTranspose(
            CompiledGraphModel graph, ValueId transposeOutput, ValueId expectedInput) {
        CompiledNode transpose = producer(graph, transposeOutput);
        assertEquals(AxisTransformKind.PERMUTE, transpose.operation().kind());
        assertEquals(List.of(expectedInput), transpose.inputs());
        assertEquals(
                List.of(1, 0),
                ((PermutationAttrs) transpose.operation().attrs()).axes());
        assertEquals(GraphPhase.BACKWARD, graph.nodePhases().get(transpose.id()));
    }

    private static ValueId inputValue(CompileArtifacts artifacts, Tensor tensor) {
        return artifacts.constants().bindableInputBindings().stream()
                .filter(binding -> binding.tensorId().equals(tensor.id()))
                .map(binding -> binding.valueId())
                .findFirst()
                .orElseThrow();
    }

    private static CompiledNode producer(CompiledGraphModel graph, ValueId output) {
        return graph.nodes().stream()
                .filter(node -> node.outputs().contains(output))
                .findFirst()
                .orElseThrow();
    }

    private static void assertMatmulGradientResults(
            io.github.pho001.synaptik.engine.RunResult result) {
        assertEquals(3, result.resultCount());
        assertEquals(
                List.of(
                        io.github.pho001.synaptik.engine.RunResult.Role.FORWARD,
                        io.github.pho001.synaptik.engine.RunResult.Role.GRADIENT,
                        io.github.pho001.synaptik.engine.RunResult.Role.GRADIENT),
                result.publications().stream()
                        .map(publication -> publication.role())
                        .toList());
        assertPublication(result, 0,
                74.0f, 80.0f, 86.0f, 92.0f,
                173.0f, 188.0f, 203.0f, 218.0f);
        assertPublication(result, 1,
                90.0f, 130.0f, 170.0f,
                226.0f, 330.0f, 434.0f);
        assertPublication(result, 2,
                21.0f, 26.0f, 31.0f, 36.0f,
                27.0f, 34.0f, 41.0f, 48.0f,
                33.0f, 42.0f, 51.0f, 60.0f);
    }

    private static void assertTensorBitsUnchanged(Tensor tensor, float[] expected) {
        MemorySegmentStorage storage =
                (MemorySegmentStorage) tensor.hostStorage().orElseThrow();
        for (int index = 0; index < expected.length; index++) {
            assertEquals(
                    Float.floatToRawIntBits(expected[index]),
                    storage.segment().getAtIndex(ValueLayout.JAVA_INT, index));
        }
    }

    private static void assertBinaryResults(
            io.github.pho001.synaptik.engine.RunResult result,
            int[] matrix,
            int[] row,
            int[] scalar) {
        assertEquals(7, result.resultCount());
        var ownership = EngineMixedOwnerTestAccess.runOwnedIdentities(result);
        assertEquals(7, ownership.publications().size());
        assertEquals(1, ownership.workspaces().size());
        BinaryArithmeticKind[] kinds = {
            BinaryArithmeticKind.ADD,
            BinaryArithmeticKind.SUB,
            BinaryArithmeticKind.MUL,
            BinaryArithmeticKind.DIV,
            BinaryArithmeticKind.SUB,
            BinaryArithmeticKind.MUL,
            BinaryArithmeticKind.DIV
        };
        int[][] actual = new int[kinds.length][];
        for (int target = 0; target < kinds.length; target++) {
            actual[target] = rawBits(result.materialize(
                    result.publications().get(target),
                    (long) matrix.length * Integer.BYTES).bytes(),
                    matrix.length);
        }
        for (int target = 0; target < kinds.length; target++) {
            for (int lane = 0; lane < matrix.length; lane++) {
                int column = lane % row.length;
                int left = switch (target) {
                    case 0, 2, 3, 4 -> matrix[lane];
                    case 1 -> row[column];
                    case 5 -> actual[0][lane];
                    case 6 -> scalar[0];
                    default -> throw new AssertionError();
                };
                int right = switch (target) {
                    case 0, 2, 5 -> row[column];
                    case 1, 4, 6 -> matrix[lane];
                    case 3 -> scalar[0];
                    default -> throw new AssertionError();
                };
                assertAcceleratorAllowed(
                        kinds[target],
                        left,
                        right,
                        actual[target][lane],
                        "publication=" + target + " lane=" + lane);
            }
        }
    }

    private static void assertAcceleratorAllowed(
            BinaryArithmeticKind kind,
            int declaredLeft,
            int declaredRight,
            int observed,
            String label) {
        Set<Integer> allowed = new HashSet<>();
        boolean allowsNaN = false;
        for (int left : dazChoices(declaredLeft)) {
            for (int right : dazChoices(declaredRight)) {
                int result = evaluate(kind, left, right);
                if (isNaN(result)) {
                    allowsNaN = true;
                } else {
                    allowed.add(result);
                }
                if (isSubnormal(result)) {
                    allowed.add(0x00000000);
                    allowed.add(0x80000000);
                }
                if ((kind == BinaryArithmeticKind.ADD || kind == BinaryArithmeticKind.SUB)
                        && (result & 0x7fffffff) == 0) {
                    allowed.add(0x00000000);
                    allowed.add(0x80000000);
                }
            }
        }
        assertTrue(
                isNaN(observed) ? allowsNaN : allowed.contains(observed),
                label + " kind=" + kind
                        + " left=0x" + Integer.toHexString(declaredLeft)
                        + " right=0x" + Integer.toHexString(declaredRight)
                        + " observed=0x" + Integer.toHexString(observed));
    }

    private static int[] dazChoices(int bits) {
        return isSubnormal(bits)
                ? new int[] {bits, bits & 0x80000000}
                : new int[] {bits};
    }

    private static int evaluate(
            BinaryArithmeticKind kind, int leftBits, int rightBits) {
        float left = Float.intBitsToFloat(leftBits);
        float right = Float.intBitsToFloat(rightBits);
        float result = switch (kind) {
            case ADD -> left + right;
            case SUB -> left - right;
            case MUL -> left * right;
            case DIV -> left / right;
            default -> throw new AssertionError("unexpected binary kind " + kind);
        };
        return Float.floatToRawIntBits(result);
    }

    private static boolean isSubnormal(int bits) {
        int magnitude = bits & 0x7fffffff;
        return magnitude != 0 && magnitude < 0x00800000;
    }

    private static boolean isNaN(int bits) {
        return (bits & 0x7f800000) == 0x7f800000
                && (bits & 0x007fffff) != 0;
    }

    private static void assertNoGradScalarResults(
            io.github.pho001.synaptik.engine.RunResult result,
            int[] inputBits,
            int[] scalarBits) {
        BinaryArithmeticKind[] kinds = {
            BinaryArithmeticKind.ADD,
            BinaryArithmeticKind.SUB,
            BinaryArithmeticKind.MUL,
            BinaryArithmeticKind.DIV
        };
        assertEquals(scalarBits.length * kinds.length + 1, result.resultCount());
        int publication = 0;
        for (int scalar : scalarBits) {
            for (BinaryArithmeticKind kind : kinds) {
                int[] actual = rawBits(
                        result.materialize(
                                result.publications().get(publication),
                                Math.multiplyExact(
                                        (long) inputBits.length,
                                        Integer.BYTES)).bytes(),
                        inputBits.length);
                for (int lane = 0; lane < inputBits.length; lane++) {
                    assertAcceleratorAllowed(
                            kind,
                            inputBits[lane],
                            scalar,
                            actual[lane],
                            "scalar=0x" + Integer.toHexString(scalar)
                                    + " publication=" + publication
                                    + " lane=" + lane);
                }
                publication++;
            }
        }
        int[] reciprocal = rawBits(
                result.materialize(
                        result.publications().get(publication),
                        Math.multiplyExact((long) inputBits.length, Integer.BYTES)).bytes(),
                inputBits.length);
        for (int lane = 0; lane < inputBits.length; lane++) {
            assertAcceleratorAllowed(
                    BinaryArithmeticKind.DIV,
                    0x3f80_0000,
                    inputBits[lane],
                    reciprocal[lane],
                    "reciprocal lane=" + lane);
        }
    }

    private static void assertRemainingExactUnaryResults(
            io.github.pho001.synaptik.engine.RunResult result,
            int[] inputBits) {
        assertEquals(4, result.resultCount());
        int[][] actual = new int[4][];
        for (int publication = 0; publication < actual.length; publication++) {
            actual[publication] = rawBits(
                    result.materialize(
                            result.publications().get(publication),
                            Math.multiplyExact((long) inputBits.length, Integer.BYTES)).bytes(),
                    inputBits.length);
        }
        for (int lane = 0; lane < inputBits.length; lane++) {
            int input = inputBits[lane];
            if (isNaN(input)) {
                for (int publication = 0; publication < actual.length; publication++) {
                    assertTrue(
                            isNaN(actual[publication][lane]),
                            "publication=" + publication + " NaN lane=" + lane);
                }
                continue;
            }
            float value = Float.intBitsToFloat(input);
            int expectedFloor = Float.floatToRawIntBits((float) StrictMath.floor(value));
            int expectedCeil = Float.floatToRawIntBits((float) StrictMath.ceil(value));
            int expectedSign = (input & 0x7fff_ffff) == 0
                    ? input
                    : input < 0 ? 0xbf80_0000 : 0x3f80_0000;
            int expectedRelu = input < 0 ? 0 : input;
            assertEquals(expectedFloor, actual[0][lane], "FLOOR lane=" + lane);
            assertEquals(expectedCeil, actual[1][lane], "CEIL lane=" + lane);
            assertEquals(expectedSign, actual[2][lane], "SIGN lane=" + lane);
            assertEquals(expectedRelu, actual[3][lane], "RELU lane=" + lane);
        }
    }

    private static void assertAbsEngineResults(
            NumericalProfile profile,
            io.github.pho001.synaptik.engine.RunResult result,
            int[] inputBits) {
        int expectedPublications = profile == NumericalProfile.STRICT_IEEE ? 2 : 3;
        assertEquals(expectedPublications, result.resultCount());
        int[][] actual = new int[expectedPublications][];
        for (int publication = 0; publication < expectedPublications; publication++) {
            actual[publication] = rawBits(result.materialize(
                    result.publications().get(publication),
                    Math.multiplyExact((long) inputBits.length, Integer.BYTES)).bytes(),
                    inputBits.length);
        }
        assertExactAbs(inputBits, actual[0], profile + " direct ABS");
        if (profile == NumericalProfile.STRICT_IEEE) {
            assertExactAbs(inputBits, actual[1], "strict affine/CONTIGUOUS/ABS/NEG/ABS");
        } else {
            for (int lane = 0; lane < inputBits.length; lane++) {
                assertAcceleratorAllowed(
                        BinaryArithmeticKind.ADD,
                        inputBits[lane],
                        0,
                        actual[1][lane],
                        "accelerator ABS composition lane=" + lane);
            }
            assertExactAbs(actual[1], actual[2], "accelerator binary-to-ABS");
        }
    }

    private static void assertExactAbs(int[] inputs, int[] outputs, String label) {
        assertEquals(inputs.length, outputs.length, label);
        for (int lane = 0; lane < inputs.length; lane++) {
            if (isNaN(inputs[lane])) {
                assertTrue(isNaN(outputs[lane]), label + " NaN lane " + lane);
            } else {
                assertEquals(inputs[lane] & 0x7fffffff, outputs[lane],
                        label + " lane " + lane);
            }
        }
    }

    private static int[] rawBits(ByteBuffer bytes, int count) {
        int[] bits = new int[count];
        for (int index = 0; index < count; index++) {
            bits[index] = bytes.getInt();
        }
        return bits;
    }

    private static void assertMixedResult(
            Engine engine,
            io.github.pho001.synaptik.engine.CompiledGraph compiled,
            List<String> expectedOwners,
            List<Tensor> inputs,
            float first,
            float second) {
        assertEquals(
                expectedOwners,
                EngineMixedOwnerTestAccess.partitionOwners(compiled));
        try (var session = engine.session(compiled)) {
            for (int run = 0; run < 2; run++) {
                try (var result = session.run(inputs)) {
                    ByteBuffer canonical = result.materialize(
                            result.publications().getFirst(), 8L).bytes();
                    assertEquals(Float.floatToRawIntBits(first), canonical.getInt());
                    assertEquals(Float.floatToRawIntBits(second), canonical.getInt());
                }
            }
        }
    }

    private static void assertCapturedAdaptersSurviveRegistryPoison(
            Engine engine,
            io.github.pho001.synaptik.engine.CompiledGraph compiled,
            List<String> expectedOwners,
            List<Tensor> inputs) {
        assertEquals(
                expectedOwners,
                EngineMixedOwnerTestAccess.partitionOwners(compiled));
        try (var session = engine.session(compiled)) {
            EngineMixedOwnerTestAccess.poisonBackendLookup(engine);
            for (int run = 0; run < 2; run++) {
                try (var result = session.run(inputs)) {
                    assertEquals(2, result.resultCount());
                    assertCanonical(
                            result.materialize(result.publications().get(0), 8L).bytes(),
                            -strictExp(1.25f),
                            -strictExp(-2.5f));
                    assertCanonical(
                            result.materialize(result.publications().get(1), 8L).bytes(),
                            strictExp(-strictExp(1.25f)),
                            strictExp(-strictExp(-2.5f)));
                }
            }
        }
    }

    private static float strictExp(float value) {
        return (float) StrictMath.exp(value);
    }

    @Test
    void cpuFreeMetalRunsExactBoolMixedCustomProgramWithoutFallback() throws Exception {
        Path library = configuredMetalLibrary();
        List<ObservedTrace> events = new CopyOnWriteArrayList<>();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library), traceCollector(events)));
            try (Engine engine = builder.build()) {
                Tensor classified = nativeTensorBits(
                        descriptor(Shape.of(2, 3)),
                        arena,
                        0x00000000,
                        0x7f800000,
                        0x7fc12345,
                        0xff800000,
                        0x3f800000,
                        0x7f812345);
                Tensor boolColumn = nativeBoolTensor(Shape.of(2, 1), arena, 1, 0);
                Tensor boolRow = nativeBoolTensor(Shape.of(3), arena, 1, 0, 1);
                Tensor trueBranch = nativeTensorBits(
                        descriptor(Shape.of(2, 1)),
                        arena,
                        0x7fc12345,
                        0x80000000);
                Tensor falseBranch = nativeTensorBits(
                        descriptor(Shape.of(1, 3)),
                        arena,
                        0x3f800000,
                        0xffc54321,
                        0x00000001);

                Tensor finite = classified.isFinite();
                Tensor nan = classified.isNaN();
                Tensor infinite = classified.isInf();
                Tensor conjunction = finite.logicalAnd(boolColumn);
                Tensor disjunction = nan.logicalOr(boolRow);
                Tensor negatedInfinite = infinite.logicalNot();
                Tensor selected = Tensor.where(conjunction, trueBranch, falseBranch);
                Tensor mixed = Tensor.where(disjunction, classified.neg(), classified);
                var compiled = engine.compile(List.of(
                        finite,
                        nan,
                        infinite,
                        conjunction,
                        disjunction,
                        negatedInfinite,
                        selected,
                        mixed));
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));
                assertTrue(events.isEmpty(), "compile neither prepares nor probes a fallback");

                try (InferenceSession session = engine.session(compiled)) {
                    assertEquals(1, events.size());
                    assertEquals("PREPARE", events.getFirst().phase());
                    assertEquals("CUSTOM_KERNEL",
                            enumName(component(events.getFirst().payload(), "route")));

                    List<Tensor> canonicalInputs =
                            List.of(classified, boolColumn, boolRow, trueBranch, falseBranch);
                    try (var result = session.run(canonicalInputs)) {
                        assertBoolPublication(result, 0, 1, 0, 0, 0, 1, 0);
                        assertBoolPublication(result, 1, 0, 0, 1, 0, 0, 1);
                        assertBoolPublication(result, 2, 0, 1, 0, 1, 0, 0);
                        assertBoolPublication(result, 3, 1, 0, 0, 0, 0, 0);
                        assertBoolPublication(result, 4, 1, 0, 1, 1, 0, 1);
                        assertBoolPublication(result, 5, 1, 0, 1, 0, 1, 1);
                        assertRawBits(
                                result.materialize(
                                        result.publications().get(6),
                                        6L * Float.BYTES).bytes(),
                                new int[] {
                                    0x7fc12345,
                                    0xffc54321,
                                    0x00000001,
                                    0x3f800000,
                                    0xffc54321,
                                    0x00000001
                                });
                        assertRawBits(
                                result.materialize(
                                        result.publications().get(7),
                                        6L * Float.BYTES).bytes(),
                                new int[] {
                                    0x80000000,
                                    0x7f800000,
                                    0xffc12345,
                                    0x7f800000,
                                    0x3f800000,
                                    0xff812345
                                });
                    }

                    int eventsBeforeRejection = events.size();
                    Tensor invalidBoolRow = nativeBoolTensor(Shape.of(3), arena, 1, 2, 1);
                    assertThrows(
                            RuntimeException.class,
                            () -> {
                                try (var ignored = session.run(List.of(
                                        classified,
                                        boolColumn,
                                        invalidBoolRow,
                                        trueBranch,
                                        falseBranch))) {
                                    // A successful result would violate canonical BOOL ingress.
                                }
                            });
                    assertEquals(
                            eventsBeforeRejection,
                            events.size(),
                            "canonical BOOL ingress fails before native invocation or fallback");

                    try (var recovered = session.run(canonicalInputs)) {
                        assertBoolPublication(recovered, 3, 1, 0, 0, 0, 0, 0);
                    }
                    Object recovery = events.getLast().payload();
                    assertEquals("SUCCEEDED", enumName(component(recovery, "status")));
                    assertEquals("CUSTOM_KERNEL", enumName(component(recovery, "route")));
                }
            }
        }
    }

    @Test
    void cpuFreeAcceleratorMetalRunsTask0052CustomPartitionsThroughPublicEngine()
            throws Exception {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.numericalProfile(NumericalProfile.ACCELERATOR);
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Tensor left = nativeTensor(
                        descriptor(Shape.of(2, 3)), arena,
                        1.0f, -2.0f, 3.0f, 4.0f, 0.0f, -1.0f);
                Tensor right = nativeTensor(
                        descriptor(Shape.of(2, 3)), arena,
                        0.0f, -2.0f, 5.0f, 3.0f, 0.0f, -2.0f);
                List<Tensor> publications = List.of(
                        left.greaterThan(right),
                        left.greaterOrEqual(right),
                        left.lessThan(right),
                        left.lessOrEqual(right),
                        left.equalTo(right),
                        left.notEqualTo(right),
                        left.minimum(right),
                        left.maximum(right),
                        left.minimum(ScalarValue.float32(0.5f)),
                        left.maximum(ScalarValue.float32(-0.5f)),
                        left.clamp(ScalarValue.float32(-1.0f), ScalarValue.float32(2.0f)),
                        left.min(),
                        left.max(1),
                        left.min(new int[] {1, 0}, false),
                        left.max(new int[0], false),
                        left.cumSum(1, false, false),
                        left.cumSum(1, true, false),
                        left.cumSum(1, false, true),
                        left.cumSum(1, true, true),
                        left.cumProd(1, false, false),
                        left.cumProd(1, true, false),
                        left.cumProd(1, false, true),
                        left.cumProd(1, true, true),
                        left.neg().maximum(right).abs());
                var compiled = engine.compile(publications);
                assertEquals(List.of("metal"),
                        EngineMixedOwnerTestAccess.partitionOwners(compiled));
                try (InferenceSession session = engine.session(compiled);
                        InferenceSession independent = engine.session(compiled)) {
                    for (int repetition = 0; repetition < 2; repetition++) {
                        try (var result = session.run(List.of(left, right))) {
                            assertTask0052Results(result);
                        }
                    }
                    try (var executor = Executors.newFixedThreadPool(2)) {
                        var first = executor.submit(() -> {
                            try (var result = session.run(List.of(left, right))) {
                                assertTask0052Results(result);
                            }
                        });
                        var second = executor.submit(() -> {
                            try (var result = independent.run(List.of(left, right))) {
                                assertTask0052Results(result);
                            }
                        });
                        first.get(10, TimeUnit.SECONDS);
                        second.get(10, TimeUnit.SECONDS);
                    }
                }
            }
        }
    }

    private static void assertTask0052Results(
            io.github.pho001.synaptik.engine.RunResult result) {
        int[][] booleans = {
            {1, 0, 0, 1, 0, 1},
            {1, 1, 0, 1, 1, 1},
            {0, 0, 1, 0, 0, 0},
            {0, 1, 1, 0, 1, 0},
            {0, 1, 0, 0, 1, 0},
            {1, 0, 1, 1, 0, 1}
        };
        for (int publication = 0; publication < booleans.length; publication++) {
            ByteBuffer bytes = result.materialize(
                    result.publications().get(publication), 6L).bytes();
            for (int expected : booleans[publication]) {
                assertEquals(expected, Byte.toUnsignedInt(bytes.get()));
            }
        }
        assertPublication(result, 6, 0.0f, -2.0f, 3.0f, 3.0f, 0.0f, -2.0f);
        assertPublication(result, 7, 1.0f, -2.0f, 5.0f, 4.0f, 0.0f, -1.0f);
        assertPublication(result, 8, 0.5f, -2.0f, 0.5f, 0.5f, 0.0f, -1.0f);
        assertPublication(result, 9, 1.0f, -0.5f, 3.0f, 4.0f, 0.0f, -0.5f);
        assertPublication(result, 10, 1.0f, -1.0f, 2.0f, 2.0f, 0.0f, -1.0f);
        assertPublication(result, 11, -2.0f);
        assertPublication(result, 12, 3.0f, 4.0f);
        assertPublication(result, 13, -2.0f);
        assertPublication(result, 14, 1.0f, -2.0f, 3.0f, 4.0f, 0.0f, -1.0f);
        assertPublication(result, 15, 1.0f, -1.0f, 2.0f, 4.0f, 4.0f, 3.0f);
        assertPublication(result, 16, 0.0f, 1.0f, -1.0f, 0.0f, 4.0f, 4.0f);
        assertPublication(result, 17, 2.0f, 1.0f, 3.0f, 3.0f, -1.0f, -1.0f);
        assertPublication(result, 18, 1.0f, 3.0f, 0.0f, -1.0f, -1.0f, 0.0f);
        assertPublication(result, 19, 1.0f, -2.0f, -6.0f, 4.0f, 0.0f, -0.0f);
        assertPublication(result, 20, 1.0f, 1.0f, -2.0f, 1.0f, 4.0f, 0.0f);
        assertPublication(result, 21, -6.0f, -6.0f, 3.0f, -0.0f, -0.0f, -1.0f);
        assertPublication(result, 22, -6.0f, 3.0f, 1.0f, -0.0f, -1.0f, 1.0f);
        assertPublication(result, 23, 0.0f, 2.0f, 5.0f, 3.0f, 0.0f, 1.0f);
    }

    private static void assertReductionResults(
            io.github.pho001.synaptik.engine.RunResult result) {
        assertEquals(8, result.resultCount());
        assertPublication(result, 0, 300.0f);
        assertPublication(result, 1, 12.5f);
        assertPublication(result, 2,
                5.0f, 6.0f, 7.0f, 8.0f,
                17.0f, 18.0f, 19.0f, 20.0f);
        assertPublication(result, 3,
                10.0f, 26.0f, 42.0f,
                58.0f, 74.0f, 90.0f);
        assertPublication(result, 4, 8.5f, 12.5f, 16.5f);
        assertPublication(result, 5,
                1.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f,
                13.0f, 14.0f, 15.0f, 16.0f,
                17.0f, 18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f, 24.0f);
        assertPublication(result, 6, 66.0f, 72.0f, 78.0f, 84.0f);
        assertPublication(result, 7,
                71.0f, 78.0f, 85.0f, 92.0f,
                83.0f, 90.0f, 97.0f, 104.0f);
    }

    private static void assertPublication(
            io.github.pho001.synaptik.engine.RunResult result,
            int publicationIndex,
            float... expected) {
        assertCanonical(
                result.materialize(
                        result.publications().get(publicationIndex),
                        Math.multiplyExact((long) expected.length, Float.BYTES))
                        .bytes(),
                expected);
    }

    private static void assertBoolPublication(
            io.github.pho001.synaptik.engine.RunResult result,
            int publicationIndex,
            int... expected) {
        ByteBuffer bytes = result.materialize(
                result.publications().get(publicationIndex), expected.length).bytes();
        for (int value : expected) {
            assertEquals(value, Byte.toUnsignedInt(bytes.get()));
        }
    }


    private static void assertTask0060PublicResults(
            io.github.pho001.synaptik.engine.RunResult result,
            int[] data,
            int[] updates,
            int[] foldAxis,
            int[] fold2d,
            int[] fold3d) {
        assertEquals(11, result.resultCount());
        int[] replacement = {data[0], updates[1], data[2], updates[0]};
        assertRawBits(result.materialize(
                result.publications().get(0), 4L * Integer.BYTES).bytes(), replacement);
        assertRawBits(result.materialize(
                result.publications().get(1), 4L * Integer.BYTES).bytes(), replacement);
        assertRawBits(
                result.materialize(
                        result.publications().get(2), 4L * Integer.BYTES).bytes(),
                new int[] {data[0], updates[0], updates[1], data[3]});
        assertRawBits(result.materialize(
                result.publications().get(3), 4L * Integer.BYTES).bytes(), foldAxis);
        assertRawBits(result.materialize(
                result.publications().get(4), 4L * Integer.BYTES).bytes(), fold2d);
        assertRawBits(result.materialize(
                result.publications().get(5), 8L * Integer.BYTES).bytes(), fold3d);

        ByteBuffer product = result.materialize(
                result.publications().get(6), 2L * Integer.BYTES).bytes();
        assertEquals(-2, product.getInt());
        assertEquals(-12, product.getInt());
        ByteBuffer productIdentity = result.materialize(
                result.publications().get(7), 4L * Integer.BYTES).bytes();
        for (int expected : new int[] {Integer.MAX_VALUE, 2, -3, 4}) {
            assertEquals(expected, productIdentity.getInt());
        }
        assertBoolPublication(result, 8, 1, 0);
        assertBoolPublication(result, 9, 1, 1);
        assertBoolPublication(result, 10, 1, 1, 0, 1);
    }

    private static void assertTask0059PublicResults(
            io.github.pho001.synaptik.engine.RunResult result) {
        assertEquals(14, result.resultCount());
        ByteBuffer cast = result.materialize(
                result.publications().getFirst(), 4L * Long.BYTES).bytes();
        for (long value : new long[] {2, 0, 1, 1}) {
            assertEquals(value, cast.getLong());
        }
        assertPublication(result, 1, 3, 1, 5, 5);
        assertPublication(result, 2, 4, 5, 6, 1, 2, 3);
        assertPublication(result, 3, -0.0f, 1, 2, 3, -0.0f, -0.0f, 4, 5, 6, -0.0f);
        assertPublication(result, 4, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6);
        assertPublication(result, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6);
        assertPublication(result, 6, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6);
        assertPublication(result, 7, 1, 2, 3, 4);
        assertPublication(result, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        assertEquals(7L, result.materialize(
                result.publications().get(9), Long.BYTES).bytes().getLong());
        assertPublication(result, 10, 7);
        assertPublication(result, 11, 7);
        assertPublication(result, 12, 7, 7);
        assertPublication(result, 13, 20);
    }

    private static void assertAffineResults(
            InferenceSession session, List<Tensor> inputs, List<int[]> expected) {
        try (var result = session.run(inputs)) {
            assertAffineResults(result, expected);
        }
    }

    private static void assertAffineResults(
            io.github.pho001.synaptik.engine.RunResult result, List<int[]> expected) {
        assertEquals(expected.size(), result.resultCount());
        for (int index = 0; index < expected.size(); index++) {
            ByteBuffer canonical = result.materialize(
                    result.publications().get(index),
                    Math.multiplyExact((long) expected.get(index).length, Float.BYTES)).bytes();
            for (int bits : expected.get(index)) {
                assertEquals(bits, canonical.getInt());
            }
        }
    }

    private static void assertRawBits(ByteBuffer canonical, int[] expected) {
        for (int bits : expected) {
            assertEquals(bits, canonical.getInt());
        }
    }

    private static void assertCanonical(ByteBuffer canonical, float... expected) {
        for (float value : expected) {
            assertEquals(Float.floatToRawIntBits(value), canonical.getInt());
        }
    }


    private static TensorDescriptor descriptor(Shape shape) {
        return descriptor(shape, false);
    }

    private static TensorDescriptor descriptor(Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
    }

    private static Tensor nativeTensorBits(
            TensorDescriptor descriptor, Arena arena, int... bits) {
        var segment = arena.allocate(
                Math.multiplyExact(bits.length, Integer.BYTES), Integer.BYTES);
        for (int index = 0; index < bits.length; index++) {
            segment.setAtIndex(ValueLayout.JAVA_INT, index, bits[index]);
        }
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.FLOAT32, bits.length, segment)));
    }
    private static Tensor nativeTensor(
            TensorDescriptor descriptor, Arena arena, float... values) {
        var segment = arena.allocate(
                Math.multiplyExact(values.length, Float.BYTES), Float.BYTES);
        for (int index = 0; index < values.length; index++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, index, values[index]);
        }
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.FLOAT32, values.length, segment)));
    }

    private static Tensor nativeIntTensor(Shape shape, Arena arena, int... values) {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.INT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        var segment = arena.allocate(
                Math.multiplyExact(values.length, Integer.BYTES), Integer.BYTES);
        for (int index = 0; index < values.length; index++) {
            segment.setAtIndex(ValueLayout.JAVA_INT, index, values[index]);
        }
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.INT32, values.length, segment)));
    }
    private static Tensor nativeBfloatTensor(
            Shape shape, Arena arena, float... values) {
        return nativeBfloatTensor(shape, false, arena, values);
    }

    private static Tensor nativeBfloatTensor(
            Shape shape, boolean requiresGrad, Arena arena, float... values) {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.BFLOAT16,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
        var segment = arena.allocate(
                Math.multiplyExact(values.length, Short.BYTES), Short.BYTES);
        for (int index = 0; index < values.length; index++) {
            segment.setAtIndex(
                    ValueLayout.JAVA_SHORT,
                    index,
                    (short) (Float.floatToRawIntBits(values[index]) >>> 16));
        }
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.BFLOAT16, values.length, segment)));
    }

    private static Tensor nativeDoubleTensor(
            Shape shape, Arena arena, double... values) {
        return nativeDoubleTensor(shape, false, arena, values);
    }

    private static Tensor nativeDoubleTensor(
            Shape shape, boolean requiresGrad, Arena arena, double... values) {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT64,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
        var segment = arena.allocate(
                Math.multiplyExact(values.length, Double.BYTES), Double.BYTES);
        for (int index = 0; index < values.length; index++) {
            segment.setAtIndex(ValueLayout.JAVA_DOUBLE, index, values[index]);
        }
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.FLOAT64, values.length, segment)));
    }

    private static Tensor symbolic(TensorDescriptor descriptor) {
        return TensorFactory.create(descriptor, Optional.empty(), Optional.empty());
    }


    private static Tensor nativeLongTensor(Shape shape, Arena arena, long... values) {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.INT64,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        var segment = arena.allocate(
                Math.multiplyExact(values.length, Long.BYTES), Long.BYTES);
        for (int index = 0; index < values.length; index++) {
            segment.setAtIndex(ValueLayout.JAVA_LONG, index, values[index]);
        }
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.INT64, values.length, segment)));
    }

    private static Tensor nativeBoolTensor(Shape shape, Arena arena, int... values) {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.BOOL,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        var segment = arena.allocate(values.length, 1);
        for (int index = 0; index < values.length; index++) {
            segment.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) values[index]);
        }
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.BOOL, values.length, segment)));
    }

    private static ModelAutotuningRequest tuningRequest(
            Path directory,
            Tensor representative,
            ModelAutotuningConfig.FallbackPolicy fallbackPolicy) {
        var config = new ModelAutotuningConfig(
                ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new ModelAutotuningConfig.Budget(1, 4, 0, 1),
                new ModelAutotuningConfig.RepresentativeProfileIdentity(1, new byte[] {2}),
                fallbackPolicy,
                directory.resolve("workload.bin"),
                new ModelAutotuningConfig.CompletePlanBudget(1, 0, 1, 1L, 8L),
                directory.resolve("model-plan.bin"));
        return new ModelAutotuningRequest(
                config,
                new ModelAutotuningRequest.ModelIdentity(1, new byte[] {1}),
                List.of(representative));
    }

    private static MetalTraceObserver traceCollector(List<ObservedTrace> events) {
        return (MetalTraceObserver) Proxy.newProxyInstance(
                MetalTraceObserver.class.getClassLoader(),
                new Class<?>[] {MetalTraceObserver.class},
                (proxy, method, arguments) -> {
                    if ("onEvent".equals(method.getName())) {
                        Object event = arguments[0];
                        events.add(new ObservedTrace(
                                idValue(component(event, "id")),
                                enumName(component(event, "phase")),
                                enumName(component(event, "level")),
                                component(event, "payload")));
                        return null;
                    }
                    return switch (method.getName()) {
                        case "equals" -> proxy == arguments[0];
                        case "hashCode" -> System.identityHashCode(proxy);
                        case "toString" -> "Metal trace collector";
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

    private static long idValue(Object id) {
        return (long) component(id, "value");
    }

    private static String enumName(Object value) {
        return ((Enum<?>) value).name();
    }

    private static String nativeStatusKind(Object outcome) {
        return enumName(component(nativeStatus(outcome), "kind"));
    }

    private static int nativeStatusCode(Object outcome) {
        return (int) component(nativeStatus(outcome), "code");
    }

    private static Object nativeStatus(Object outcome) {
        return ((Optional<?>) component(outcome, "nativeStatus")).orElseThrow();
    }

    private record ObservedTrace(long eventId, String phase, String level, Object payload) {}

    private static Path configuredMetalLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        Assumptions.assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is required for real Metal integration");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        Assumptions.assumeTrue(Files.isRegularFile(library),
                "configured Metal library must exist: " + library);
        return library;
    }
}
