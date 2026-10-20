package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.nn.layers.Linear;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.api.io.TempDir;

/** Cold graph load joins the existing explicit Engine path, not a new Model or execution facade. */
final class GraphCheckpointEngineIntegrationTest {
    @TempDir Path directory;

    @Test
    void linearLoadOutlivesSourceAndMatchesForwardAndOrdinaryAutogradForTwoFreshInputs() throws Exception {
        ArtifactLimits limits = new ArtifactLimits(100_000, 10_000, 10, 32, 10_000, 90_000);
        Path file = directory.resolve("linear.graph");
        try (Arena arena = Arena.ofConfined()) {
            Tensor weight = nativeValue(arena, Shape.of(2, 2), 1, -2, 3, 1);
            Tensor bias = nativeValue(arena, Shape.of(2), 0.5f, -0.5f);
            Tensor prototype = TensorFactory.create(descriptor(Shape.of(1, 2)));
            Tensor output = new Linear(weight, bias).forward(prototype);
            GraphCheckpoint.save(file, new GraphCheckpoint.Source(List.of(prototype), List.of(output), List.of(
                    new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight),
                    new GraphCheckpoint.Leaf("bias", GraphCheckpoint.Role.PARAMETER, bias))), limits);
        }
        var loaded = GraphCheckpoint.load(file, limits);
        Files.delete(file);
        assertTrue(loaded.inputs().getFirst().hostStorage().isEmpty());
        for (var leaf : loaded.leaves()) assertFalse(leaf.tensor().hostStorage().orElseThrow().segment().isNative());
        try (Engine engine = Engine.standard()) {
            for (float[] batch : List.of(new float[] {1, 2}, new float[] {-3, 4})) {
                Tensor input = heapValue(Shape.of(1, 2), batch);
                Tensor directWeight = heapValue(Shape.of(2, 2), 1, -2, 3, 1);
                Tensor directBias = heapValue(Shape.of(2), 0.5f, -0.5f);
                Tensor direct = new Linear(directWeight, directBias).forward(input);
                Tensor rebound = GraphCheckpoint.rebind(loaded, List.of(input)).getFirst();
                assertEquals(engine.compute(direct).bytes(), engine.compute(rebound).bytes());
                var expected = engine.backward(direct.sum(), List.of(input, directWeight, directBias), 1000);
                var actual = engine.backward(rebound.sum(), List.of(input,
                        loaded.leaves().get(0).tensor(), loaded.leaves().get(1).tensor()), 1000);
                assertEquals(expected.objective().bytes(), actual.objective().bytes());
                for (int i = 0; i < 3; i++) assertEquals(expected.gradients().get(i).bytes(), actual.gradients().get(i).bytes());
                assertEquals(batch[0] - 2 * batch[1] + 0.5f, engine.compute(rebound).bytes().getFloat(0));
            }
            GraphCheckpoint.save(file, loaded, limits);
            var second = GraphCheckpoint.load(file, limits);
            assertNotSame(loaded.leaves().getFirst().tensor(), second.leaves().getFirst().tensor());
        }
    }

    @ParameterizedTest
    @EnumSource(value = UnaryElementwiseKind.class, names = {"RELU", "TANH", "SIGMOID"})
    void restoredActivationMatchesIndependentOrdinaryForwardAndBackward(UnaryElementwiseKind kind) throws Exception {
        ArtifactLimits limits = new ArtifactLimits(100_000, 10_000, 10, 32, 10_000, 90_000);
        Path file = directory.resolve("activation.graph");
        try (Arena arena = Arena.ofConfined()) {
            Tensor weight = nativeValue(arena, Shape.of(2, 2), 0.5f, -0.25f, 0.25f, 0.5f);
            Tensor prototype = TensorFactory.create(descriptor(Shape.of(1, 2)));
            Tensor output = activate(new Linear(weight).forward(prototype), kind);
            GraphCheckpoint.save(file, new GraphCheckpoint.Source(List.of(prototype), List.of(output),
                    List.of(new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight))), limits);
        }
        var loaded = GraphCheckpoint.load(file, limits);
        try (Engine engine = Engine.standard()) {
            for (float[] batch : List.of(new float[] {1, 2}, new float[] {-0.5f, 1})) {
                Tensor input = heapValue(Shape.of(1, 2), batch);
                Tensor weight = heapValue(Shape.of(2, 2), 0.5f, -0.25f, 0.25f, 0.5f);
                Tensor direct = activate(new Linear(weight).forward(input), kind);
                Tensor rebuilt = GraphCheckpoint.rebind(loaded, List.of(input)).getFirst();
                assertEquals(kind, rebuilt.provenance().orElseThrow().operation().kind());
                var forward = engine.compute(direct).bytes();
                var actualForward = engine.compute(rebuilt).bytes();
                for (int lane = 0; lane < 2; lane++)
                    assertEquals(forward.getFloat(lane * 4), actualForward.getFloat(lane * 4), 1e-5);
                var expected = engine.backward(direct.sum(), List.of(input, weight), 1000);
                var actual = engine.backward(rebuilt.sum(), List.of(input, loaded.leaves().getFirst().tensor()), 1000);
                assertEquals(expected.objective().bytes().getFloat(), actual.objective().bytes().getFloat(), 1e-5);
                for (int p = 0; p < 2; p++) {
                    var a = actual.gradients().get(p).bytes();
                    var e = expected.gradients().get(p).bytes();
                    for (int lane = 0; lane < e.remaining() / 4; lane++)
                        assertEquals(e.getFloat(lane * 4), a.getFloat(lane * 4), 1e-5);
                }
            }
        }
    }

    /**
     * @param input ordinary floating source expression
     * @param kind selected activation, independent of cold reconstruction
     * @return ordinary Model activation expression
     */
    private static Tensor activate(Tensor input, UnaryElementwiseKind kind) {
        return switch (kind) {
            case RELU -> input.relu(); case TANH -> input.tanh(); case SIGMOID -> input.sigmoid();
            default -> throw new AssertionError(kind);
        };
    }

    /**
     * @param shape static logical geometry
     * @return exact gradient-eligible canonical FLOAT32 descriptor
     */
    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.contiguous(shape)), true);
    }

    /**
     * @param shape static geometry matching values
     * @param values logical row-major FLOAT32 lanes
     * @return ordinary mutable heap-backed test input/state
     */
    private static Tensor heapValue(Shape shape, float... values) {
        return TensorFactory.create(descriptor(shape), Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, values.length, MemorySegment.ofArray(values))));
    }

    /**
     * @param arena caller-owned source lifetime, closed before load
     * @param shape static geometry matching values
     * @param values logical row-major FLOAT32 lanes
     * @return borrowed native test state
     */
    private static Tensor nativeValue(Arena arena, Shape shape, float... values) {
        MemorySegment storage = arena.allocate(values.length * 4L, 4);
        storage.copyFrom(MemorySegment.ofArray(values));
        return TensorFactory.create(descriptor(shape), Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, values.length, storage)));
    }
}
