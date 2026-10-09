package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.ScalarObjectiveBackwardResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Exercises profile-free arithmetic through reusable forward and generated-backward CPU runs. */
final class EngineNumericalSemanticsIntegrationTest {
    private static final TensorDescriptor DESCRIPTOR = new TensorDescriptor(
            DataType.FLOAT32,
            Shape.of(3),
            Optional.of(LayoutDescriptor.contiguous(Shape.of(3))),
            true);

    @Test
    void forwardBackwardAndReusableSessionsPreserveDeclaredResults() {
        execute();
    }

    private static void execute() {
        try (Arena arena = Arena.ofConfined();
                Engine engine = Engine.builder()
                        .takeOwnership(CpuBackendIntegration.open())
                        .build()) {
            Tensor input = input(arena);
            Tensor forward = input.mul(input).add(input.neg());
            var compiled = engine.compile(List.of(forward));
            float[] first;
            float[] second;
            try (var session = engine.session(compiled)) {
                try (var result = session.run(List.of(input))) {
                    first = floats(result.materialize(
                            result.publications().getFirst(), 3L * Float.BYTES));
                }
                try (var result = session.run(List.of(input))) {
                    second = floats(result.materialize(
                            result.publications().getFirst(), 3L * Float.BYTES));
                }
            }
            assertArrayEquals(new float[] {0.3125f, 8.75f, 12.0f}, first);
            assertArrayEquals(first, second);

            Tensor scalar = scalarInput(arena, 3.0f);
            Tensor objective = scalar.mul(scalar).add(scalar.neg());
            ScalarObjectiveBackwardResult backward =
                    engine.backward(objective, List.of(scalar), 2L * Float.BYTES);
            float objectiveValue = backward.objective().bytes().asFloatBuffer().get();
            float[] gradient = floats(backward.gradients().getFirst());
            assertEquals(6.0f, objectiveValue);
            assertArrayEquals(new float[] {5.0f}, gradient);
        }
    }

    private static float[] floats(HostTensorValue value) {
        float[] actual = new float[Math.toIntExact(value.elementCount())];
        value.bytes().asFloatBuffer().get(actual);
        return actual;
    }

    private static Tensor input(Arena arena) {
        float[] values = {1.25f, -2.5f, 4.0f};
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment storage = arena.allocate(source.byteSize(), Float.BYTES);
        MemorySegment.copy(source, 0, storage, 0, source.byteSize());
        return TensorFactory.create(DESCRIPTOR, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, values.length, storage)));
    }

    private static Tensor scalarInput(Arena arena, float value) {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.scalar(),
                Optional.of(LayoutDescriptor.contiguous(Shape.scalar())),
                true);
        MemorySegment storage = arena.allocate(Float.BYTES, Float.BYTES);
        storage.set(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, value);
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, 1, storage)));
    }

}
