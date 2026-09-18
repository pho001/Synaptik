package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Executes the public dimensional-convolution Tensor and Engine lifecycle on CPU. */
final class EngineConvolutionIntegrationTest {
    @Test
    void executesNcwConv1dThroughReusableAndOneShotPublicLifecycles() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor input = tensor(arena, Shape.of(1, 1, 4), 1, 2, 3, 4);
            Tensor weight = tensor(arena, Shape.of(1, 1, 2), 1, 1);
            Tensor bias = tensor(arena, Shape.of(1), 0.5f);
            Tensor output = input.conv1d(weight, bias, Conv1dAttrs.defaults());

            assertPublicExecution(
                    engine,
                    output,
                    List.of(bias, weight, input),
                    Shape.of(1, 1, 3),
                    true,
                    new float[] {3.5f, 5.5f, 7.5f});
        }
    }

    @Test
    void executesGroupedNchwConv2dThroughReusableAndOneShotPublicLifecycles() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor input = tensor(arena, Shape.of(1, 2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8);
            Tensor weight = tensor(arena, Shape.of(2, 1, 1, 1), 2, -1);
            Tensor bias = tensor(arena, Shape.of(2), 1, 10);
            Tensor output = input.conv2d(
                    weight, bias, new Conv2dAttrs(1, 1, 0, 0, 1, 1, 2));

            assertPublicExecution(
                    engine,
                    output,
                    List.of(bias, weight, input),
                    Shape.of(1, 2, 2, 2),
                    false,
                    new float[] {3, 5, 7, 9, 5, 4, 3, 2});
        }
    }

    @Test
    void executesGroupedNcdhwConv3dForwardThroughReusableAndOneShotPublicLifecycles() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor input = tensor(
                    arena, Shape.of(1, 2, 2, 1, 2), 1, 2, 3, 4, 5, 6, 7, 8);
            Tensor weight = tensor(arena, Shape.of(2, 1, 1, 1, 1), 2, -1);
            Tensor bias = tensor(arena, Shape.of(2), 1, 10);
            Tensor output = input.conv3d(
                    weight,
                    bias,
                    new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, 2));

            assertPublicExecution(
                    engine,
                    output,
                    List.of(bias, weight, input),
                    Shape.of(1, 2, 2, 1, 2),
                    false,
                    new float[] {3, 5, 7, 9, 5, 4, 3, 2});
        }
    }

    private static void assertPublicExecution(
            Engine engine,
            Tensor output,
            List<Tensor> nonCompilerInputOrder,
            Shape expectedShape,
            boolean expectedView,
            float[] expectedValues) {
        assertTrue(output.descriptor().layout().isEmpty());
        var compiled = engine.compile(List.of(output));
        var prepared = engine.prepare(compiled);
        try (var result = engine.run(prepared, nonCompilerInputOrder)) {
            assertEquals(1, result.resultCount());
            var publication = result.publications().getFirst();
            assertEquals(output.id(), publication.tensorId());
            assertEquals(expectedShape, publication.descriptor().shape());
            LayoutDescriptor contiguous = LayoutDescriptor.contiguous(expectedShape);
            assertEquals(
                    LayoutDescriptor.of(
                            expectedShape,
                            contiguous.strides(),
                            contiguous.storageOffset(),
                            expectedView),
                    publication.descriptor().layout().orElseThrow());
            HostTensorValue materialized = result.materialize(
                    publication, Math.multiplyExact(expectedValues.length, Float.BYTES));
            assertEquals(expectedShape, materialized.shape());
            assertArrayEquals(expectedValues, floats(materialized));
        }

        HostTensorValue computed = engine.compute(output);
        assertEquals(expectedShape, computed.shape());
        assertArrayEquals(expectedValues, floats(computed));
    }

    private static Tensor tensor(Arena arena, Shape shape, float... values) {
        assertEquals(shape.knownElementCount().orElseThrow(), values.length);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment segment = arena.allocate(source.byteSize(), Float.BYTES);
        MemorySegment.copy(source, 0, segment, 0, source.byteSize());
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.FLOAT32, values.length, segment)));
    }

    private static float[] floats(HostTensorValue value) {
        FloatBuffer buffer = value.bytes().asFloatBuffer();
        float[] result = new float[buffer.remaining()];
        buffer.get(result);
        return result;
    }
}
