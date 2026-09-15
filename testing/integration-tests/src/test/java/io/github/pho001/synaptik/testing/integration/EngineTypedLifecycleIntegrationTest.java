package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Exercises the ordinary public Engine lifecycle through the real CPU composition. */
final class EngineTypedLifecycleIntegrationTest {
    @Test
    void runsForwardWithReversedLogicalInputOrderAndMetadataOnlyResult() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor left = tensor(arena, 2, false);
            Tensor right = tensor(arena, 2, false);
            Tensor leftOutput = left.contiguous();
            Tensor rightOutput = right.contiguous();

            var compiled = engine.compile(List.of(leftOutput, rightOutput));
            assertEquals(List.of(left.id(), right.id()),
                    compiled.inputs().stream().map(input -> input.tensorId()).toList());
            var prepared = engine.prepare(compiled);
            assertSame(compiled, prepared.compiledGraph());
            RunResult result = engine.run(prepared, List.of(right, left));
            List<RunResult.Publication> publications = result.publications();
            try (result) {
                assertEquals(2, result.resultCount());
                assertEquals(List.of(leftOutput.id(), rightOutput.id()),
                        publications.stream().map(RunResult.Publication::tensorId).toList());
                assertEquals(List.of(leftOutput.descriptor(), rightOutput.descriptor()),
                        publications.stream().map(RunResult.Publication::descriptor).toList());
                assertEquals(List.of(RunResult.Role.FORWARD, RunResult.Role.FORWARD),
                        publications.stream().map(RunResult.Publication::role).toList());
                assertEquals(List.of(0, 1),
                        publications.stream().map(RunResult.Publication::index).toList());
                assertTrue(publications.stream()
                        .allMatch(publication -> publication.derivativeOrder().isEmpty()));
                assertTrue(publications.stream()
                        .allMatch(publication -> publication.targetIndex().isEmpty()));
                assertTrue(publications.stream().noneMatch(RunResult.Publication::isClosed));
            }
            assertTrue(result.isClosed());
            assertTrue(publications.stream().allMatch(RunResult.Publication::isClosed));
            assertEquals(leftOutput.id(), publications.getFirst().tensorId());
        }
    }

    @Test
    void runsExplicitSeedAndKeepsRepeatedGradientOccurrences() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor left = tensor(arena, 2, true);
            Tensor right = tensor(arena, 2, true);
            Tensor seedLeaf = tensor(arena, 2, false);
            Tensor leftOutput = left.contiguous();
            Tensor rightOutput = right.contiguous();
            Tensor seed = seedLeaf.contiguous();

            var compiled = engine.compile(
                    List.of(leftOutput, rightOutput), List.of(seed, seed), List.of(left, right));
            assertEquals(List.of(left.id(), right.id(), seedLeaf.id()),
                    compiled.inputs().stream().map(input -> input.tensorId()).toList());
            var prepared = engine.prepare(compiled);
            RunResult first = engine.run(prepared, List.of(seedLeaf, right, left));
            RunResult second = engine.run(prepared, List.of(left, right, seedLeaf));
            List<RunResult.Publication> firstPublications = first.publications();
            List<RunResult.Publication> secondPublications = second.publications();
            try (first; second) {
                assertNotSame(first, second);
                assertEquals(4, first.resultCount());
                assertEquals(List.of(RunResult.Role.FORWARD, RunResult.Role.FORWARD,
                                RunResult.Role.GRADIENT,
                                RunResult.Role.GRADIENT),
                        firstPublications.stream().map(RunResult.Publication::role).toList());
                assertEquals(List.of(leftOutput.id(), rightOutput.id(), left.id(), right.id()),
                        firstPublications.stream().map(RunResult.Publication::tensorId).toList());
                assertEquals(List.of(0, 1, 2, 3),
                        firstPublications.stream().map(RunResult.Publication::index).toList());
                assertTrue(firstPublications.get(0).derivativeOrder().isEmpty());
                assertTrue(firstPublications.get(1).derivativeOrder().isEmpty());
                assertEquals(1, firstPublications.get(2).derivativeOrder().orElseThrow());
                assertEquals(1, firstPublications.get(3).derivativeOrder().orElseThrow());
                assertTrue(firstPublications.get(0).targetIndex().isEmpty());
                assertTrue(firstPublications.get(1).targetIndex().isEmpty());
                assertEquals(0, firstPublications.get(2).targetIndex().orElseThrow());
                assertEquals(1, firstPublications.get(3).targetIndex().orElseThrow());
                assertNotSame(firstPublications.get(2), firstPublications.get(3));
                assertNotSame(firstPublications.get(2), secondPublications.get(2));
                assertFalse(first.isClosed());
            }
            assertTrue(first.isClosed());
            assertTrue(second.isClosed());
            assertTrue(firstPublications.stream().allMatch(RunResult.Publication::isClosed));
            assertTrue(secondPublications.stream().allMatch(RunResult.Publication::isClosed));
        }
    }

    @Test
    void preservesCurrentZeroNodePreparationRejection() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor leaf = tensor(arena, 1, false);
            var compiled = engine.compile(List.of(leaf));
            assertThrows(IllegalArgumentException.class, () -> engine.prepare(compiled));
        }
    }

    private static Tensor tensor(Arena arena, long elementCount, boolean requiresGrad) {
        Shape shape = Shape.of(elementCount);
        TensorDescriptor descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
        var segment = arena.allocate(
                Math.multiplyExact(elementCount, DataType.FLOAT32.byteWidth()), Float.BYTES);
        var storage = new MemorySegmentStorage(DataType.FLOAT32, elementCount, segment);
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
    }
}
