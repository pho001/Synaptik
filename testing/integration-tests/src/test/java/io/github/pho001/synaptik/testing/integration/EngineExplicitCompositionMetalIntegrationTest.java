package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.EngineMixedOwnerTestAccess;
import io.github.pho001.synaptik.engine.InferenceSession;
import io.github.pho001.synaptik.engine.ModelAutotuningPreparation;
import io.github.pho001.synaptik.engine.ModelAutotuningRequest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Exercises real CPU-free Metal and mixed CPU/Metal public Engine composition. */
final class EngineExplicitCompositionMetalIntegrationTest {
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
    void metalOnlyEnginePublishesComposedViewsAndRunsContiguousBarrierIntoNeg() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                int[] inputBits = {
                    0x00000000,
                    0x80000000,
                    0x00000001,
                    0xff800000,
                    0x7fc12345,
                    0xff812345
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
                var compiled = engine.compile(List.of(
                        reshaped,
                        expanded,
                        permuted,
                        rankEdited,
                        finalView,
                        contiguous,
                        negated));
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
                for (int index = 0; index < negatedBits.length; index++) {
                    negatedBits[index] ^= 0x80000000;
                }
                List<int[]> expected = List.of(
                        inputBits,
                        expandedBits,
                        permutedBits,
                        permutedBits,
                        permutedBits,
                        permutedBits,
                        negatedBits);
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
                        descriptor(Shape.of(2, 3)),
                        arena,
                        0x00000000,
                        0x80000000,
                        0x00000001,
                        0x7f800000,
                        0xff800000,
                        0x7fc12345);
                var compiled = engine.compile(List.of(input.reshape(3, 2).abs()));
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
    void canonicalCrossOwnerTransfersStillRunAtRankOneAndRankSixteen() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
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
                            input.neg().abs(),
                            input.abs().neg()));
                    try (var session = engine.session(compiled);
                            var result = session.run(List.of(input))) {
                        assertCanonical(
                                result.materialize(
                                        result.publications().get(0), 8L).bytes(),
                                1.25f,
                                2.5f);
                        assertCanonical(
                                result.materialize(
                                        result.publications().get(1), 8L).bytes(),
                                -1.25f,
                                -2.5f);
                    }
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
                        engine.compile(List.of(input.abs().neg())),
                        List.of(input),
                        -1.25f,
                        -2.5f);
                assertMixedResult(
                        engine,
                        engine.compile(List.of(input.neg().abs())),
                        List.of(input),
                        1.25f,
                        2.5f);
                assertMixedResult(
                        engine,
                        engine.compile(List.of(input.neg().abs().neg())),
                        List.of(input),
                        -1.25f,
                        -2.5f);

                Tensor cpuInput = TensorFactory.create(
                        descriptor,
                        Optional.empty(),
                        Optional.of(new MemorySegmentStorage(
                                DataType.FLOAT32, 2, arena.allocate(8, Float.BYTES))));
                var cpuCompiled = engine.compile(List.of(cpuInput.abs()));
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
                Tensor noStorage = TensorFactory.create(
                        descriptor, Optional.empty(), Optional.empty());
                var metalForTuning = engine.compile(List.of(noStorage.neg()));
                ModelAutotuningRequest metalRequest = tuningRequest(
                        directory, noStorage,
                        ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC);
                assertEquals("model autotuning requires a CPU-owned partition plan",
                        assertThrows(IllegalStateException.class,
                                () -> engine.prepareTuned(metalForTuning, metalRequest)).getMessage());
                Tensor capturedBase = input.abs();
                Tensor metalPublication = capturedBase.neg();
                Tensor cpuPublication = metalPublication.abs();
                assertCapturedAdaptersSurviveRegistryPoison(
                        engine,
                        engine.compile(List.of(metalPublication, cpuPublication)),
                        List.of(input));
            }
        }
    }


    private static void assertMixedResult(
            Engine engine,
            io.github.pho001.synaptik.engine.CompiledGraph compiled,
            List<Tensor> inputs,
            float first,
            float second) {
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
            List<Tensor> inputs) {
        try (var session = engine.session(compiled)) {
            EngineMixedOwnerTestAccess.poisonBackendLookup(engine);
            for (int run = 0; run < 2; run++) {
                try (var result = session.run(inputs)) {
                    assertEquals(2, result.resultCount());
                    assertCanonical(
                            result.materialize(result.publications().get(0), 8L).bytes(),
                            -1.25f,
                            -2.5f);
                    assertCanonical(
                            result.materialize(result.publications().get(1), 8L).bytes(),
                            1.25f,
                            2.5f);
                }
            }
        }
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
        return new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
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
