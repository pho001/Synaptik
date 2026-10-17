package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Public CPU composition regressions without changing shared Engine or Compiler fixtures. */
final class EngineCpuComposedValidationIntegrationTest {
    @Test void expSoftmaxRejectsProducedInfinityJustLikeDirectInputAcrossSessionRuns() {
        try (var arena = Arena.ofShared(); var engine = Engine.standard()) {
            var storage = arena.allocate(2L * Float.BYTES, Float.BYTES);
            var input = tensor(DataType.FLOAT32, Shape.of(2), storage);
            for (boolean composed : List.of(false, true)) {
                Tensor output = (composed ? input.exp() : input).softmax(0);
                try (var session = engine.prepare(engine.compile(List.of(output)))) {
                    for (int iteration = 0; iteration < 2; iteration++) {
                        storage.setAtIndex(ValueLayout.JAVA_FLOAT, 0, composed ? 0 : 1);
                        storage.setAtIndex(ValueLayout.JAVA_FLOAT, 1,
                                composed ? 1000 : Float.POSITIVE_INFINITY);
                        assertThrows(IllegalArgumentException.class, () -> engine.run(session, List.of(input)));
                        storage.setAtIndex(ValueLayout.JAVA_FLOAT, 1, composed ? 0 : 1);
                        try (var result = engine.run(session, List.of(input))) {
                            var bytes = result.materialize(result.publications().getFirst(), 8).bytes();
                            assertEquals(.5f, bytes.getFloat());
                            assertEquals(.5f, bytes.getFloat());
                        }
                    }
                }
            }
        }
    }

    @Test void nonScalarLossesComposeWithArithmeticAndStridedCallerBoundaries() {
        try (var arena = Arena.ofShared(); var engine = Engine.standard()) {
            for (boolean strided : List.of(false, true)) {
                var prediction = floats(arena, Shape.of(2, 3), strided, -1, -2, -3, -3, -2, -1);
                var target = floats(arena, Shape.of(2, 3), strided, 0, 0, 1, 1, 0, 0);
                assertArrayEquals(new float[] {-1, -4, -4, -4, -4, -1}, run(engine,
                        prediction.neg().meanSquaredError(target, LossReduction.NONE).neg(),
                        List.of(prediction, target), 6), 0f);
                float loss = (float) Math.log(1 + Math.exp(-1) + Math.exp(-2));
                assertArrayEquals(new float[] {-loss, -loss}, run(engine,
                        prediction.neg().categoricalCrossEntropyWithLogits(target, 1,
                                LossReduction.NONE).neg(), List.of(prediction, target), 2), 1e-6f);
            }
        }
    }

    @Test void computedIndexTargetsAreRevalidatedOnEachPublicRun() {
        try (var arena = Arena.ofShared(); var engine = Engine.standard()) {
            Tensor logits = floats(arena, Shape.of(2, 3), false, 0, 0, 0, 0, 0, 0);
            var targetStorage = arena.allocate(8, Float.BYTES);
            Tensor source = tensor(DataType.FLOAT32, Shape.of(2), targetStorage);
            for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
                Tensor output = logits.categoricalCrossEntropyWithLogits(source.cast(type),
                        1, LossReduction.NONE);
                try (var session = engine.prepare(engine.compile(List.of(output)))) {
                    for (int iteration = 0; iteration < 2; iteration++) {
                        targetStorage.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 1);
                        targetStorage.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 3);
                        assertThrows(IndexOutOfBoundsException.class,
                                () -> engine.run(session, List.of(logits, source)));
                        targetStorage.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 2);
                        try (var result = engine.run(session, List.of(logits, source))) {
                            var bytes = result.materialize(result.publications().getFirst(), 8).bytes();
                            assertEquals((float) Math.log(3), bytes.getFloat(), 1e-6f);
                            assertEquals((float) Math.log(3), bytes.getFloat(), 1e-6f);
                        }
                    }
                }
            }
        }
    }

    @Test void computedBooleanMaskRunsBeforeItsReductionConsumer() {
        try (var arena = Arena.ofShared(); var engine = Engine.standard()) {
            Tensor input = floats(arena, Shape.of(2, 2), false, 10, 20, 30, 40);
            var maskStorage = arena.allocate(4, 1);
            MemorySegment.copy(MemorySegment.ofArray(new byte[] {0, 1, 1, 0}), 0, maskStorage, 0, 4);
            Tensor mask = tensor(DataType.BOOL, Shape.of(2, 2), maskStorage);
            assertArrayEquals(new float[] {10, 40}, run(engine, input.sum(1, mask.logicalNot()),
                    List.of(input, mask), 2));
        }
    }

    private static float[] run(Engine engine, Tensor output, List<Tensor> inputs, int count) {
        try (var session = engine.prepare(engine.compile(List.of(output)));
                var result = engine.run(session, inputs)) {
            var bytes = result.materialize(result.publications().getFirst(),
                    count * (long) Float.BYTES).bytes();
            float[] values = new float[count];
            for (int i = 0; i < count; i++) values[i] = bytes.getFloat();
            return values;
        }
    }

    private static Tensor floats(Arena arena, Shape shape, boolean strided, float... values) {
        var layout = strided ? LayoutDescriptor.of(shape, new long[] {8, 2}, 1, true)
                : LayoutDescriptor.contiguous(shape);
        long count = layout.referencedElementSpan();
        var segment = arena.allocate(count * Float.BYTES, Float.BYTES);
        for (int i = 0; i < values.length; i++) {
            long address = strided ? 1 + i / 3 * 8L + i % 3 * 2L : i;
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, address, values[i]);
        }
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(layout), false), Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, count, segment)));
    }

    private static Tensor tensor(DataType type, Shape shape, MemorySegment storage) {
        return TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false), Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, shape.knownElementCount().orElseThrow(),
                        storage)));
    }
}
