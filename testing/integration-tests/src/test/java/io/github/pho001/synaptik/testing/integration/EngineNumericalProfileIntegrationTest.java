package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Exercises the graph-wide numerical profile through the real ordinary CPU lifecycle. */
final class EngineNumericalProfileIntegrationTest {
    private static final TensorDescriptor DESCRIPTOR = new TensorDescriptor(
            DataType.FLOAT32,
            Shape.of(3),
            Optional.of(LayoutDescriptor.contiguous(Shape.of(3))),
            false);

    @Test
    void strictStandardAndExplicitSessionsRunWhileAcceleratorCompilationFailsClosed() {
        assertStrictSession(Engine.standard());
        assertStrictSession(Engine.builder()
                .numericalProfile(NumericalProfile.STRICT_IEEE)
                .takeOwnership(CpuBackendIntegration.open())
                .build());

        try (Arena arena = Arena.ofConfined();
                Engine accelerator = Engine.builder()
                        .takeOwnership(CpuBackendIntegration.open())
                        .numericalProfile(NumericalProfile.ACCELERATOR)
                        .build()) {
            Tensor input = input(arena);
            assertThrows(IllegalStateException.class,
                    () -> accelerator.compile(List.of(input.neg())));
        }
    }

    private static void assertStrictSession(Engine engine) {
        try (Arena arena = Arena.ofConfined(); engine) {
            Tensor input = input(arena);
            var compiled = engine.compile(List.of(input.neg()));
            try (var session = engine.session(compiled);
                    var result = session.run(List.of(input))) {
                HostTensorValue output = result.materialize(result.publications().getFirst(), 12);
                float[] actual = new float[3];
                output.bytes().asFloatBuffer().get(actual);
                assertArrayEquals(new float[] {-1.25f, 2.5f, -4.0f}, actual);
            }
        }
    }

    private static Tensor input(Arena arena) {
        float[] values = {1.25f, -2.5f, 4.0f};
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment storage = arena.allocate(source.byteSize(), Float.BYTES);
        MemorySegment.copy(source, 0, storage, 0, source.byteSize());
        return TensorFactory.create(DESCRIPTOR, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, values.length, storage)));
    }

}
