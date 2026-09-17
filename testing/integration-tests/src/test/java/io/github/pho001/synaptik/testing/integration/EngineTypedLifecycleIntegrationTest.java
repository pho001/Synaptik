package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.RunResult;
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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
    void materializesCanonicalDetachedValuesForAllCurrentDataTypes() {
        HostTensorValue retained;
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            retained = assertMaterializes(arena, engine, DataType.FLOAT64,
                    new double[] {1.5, -2.25}, bytes(16).putDouble(1.5).putDouble(-2.25).array());
            assertMaterializes(arena, engine, DataType.FLOAT32,
                    new float[] {1.5f, -2.25f}, bytes(8).putFloat(1.5f).putFloat(-2.25f).array());
            assertMaterializes(arena, engine, DataType.BFLOAT16,
                    new short[] {(short) 0x3fc0, (short) 0xc010},
                    bytes(4).putShort((short) 0x3fc0).putShort((short) 0xc010).array());
            assertMaterializes(arena, engine, DataType.INT64,
                    new long[] {1, -2}, bytes(16).putLong(1).putLong(-2).array());
            assertMaterializes(arena, engine, DataType.INT32,
                    new int[] {1, -2}, bytes(8).putInt(1).putInt(-2).array());
            assertMaterializes(arena, engine, DataType.BOOL,
                    new byte[] {0, 1}, new byte[] {0, 1});
        }
        assertArrayEquals(bytes(16).putDouble(1.5).putDouble(-2.25).array(), read(retained));
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
                HostTensorValue firstGradient = first.materialize(firstPublications.get(2), 8);
                HostTensorValue aliasedGradient = first.materialize(firstPublications.get(3), 8);
                assertNotSame(firstGradient, aliasedGradient);
                assertArrayEquals(read(firstGradient), read(aliasedGradient));
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

    @Test
    void computeDiscoversInputsAndReturnsOrderedDetachedValuesWithOneAggregateLimit() {
        HostTensorValue retained;
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor left = floatTensor(arena, new float[] {1.5f, -2.25f});
            Tensor right = floatTensor(arena, new float[] {3.25f, 4.5f});
            Tensor leftOutput = left.contiguous();
            Tensor rightOutput = right.contiguous();

            retained = engine.compute(leftOutput);
            assertArrayEquals(bytes(8).putFloat(1.5f).putFloat(-2.25f).array(), read(retained));

            List<HostTensorValue> values = engine.compute(
                    List.of(left.contiguous(), rightOutput, left.contiguous()), 24);
            assertEquals(3, values.size());
            assertArrayEquals(bytes(8).putFloat(1.5f).putFloat(-2.25f).array(),
                    read(values.get(0)));
            assertArrayEquals(bytes(8).putFloat(3.25f).putFloat(4.5f).array(),
                    read(values.get(1)));
            assertArrayEquals(bytes(8).putFloat(1.5f).putFloat(-2.25f).array(),
                    read(values.get(2)));
            assertThrows(UnsupportedOperationException.class, () -> values.add(retained));
            assertEquals(
                    "total canonical byte count exceeds maximumTotalBytes: required=24, maximum=23",
                    assertThrows(IllegalArgumentException.class, () -> engine.compute(
                            List.of(left.contiguous(), rightOutput, left.contiguous()), 23))
                            .getMessage());
            assertTrue(left.hostStorage().orElseThrow().isAlive());
            assertTrue(right.hostStorage().orElseThrow().isAlive());
        }
        assertArrayEquals(bytes(8).putFloat(1.5f).putFloat(-2.25f).array(), read(retained));
    }

    @Test
    void backwardDiscoversScalarInputAndReturnsDetachedObjectiveAndPositiveOneGradient() {
        ScalarObjectiveBackwardResult retained;
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor input = scalarFloatTensor(arena, 2.5f, true);
            Tensor objective = input.contiguous();

            assertEquals(
                    "total canonical byte count exceeds maximumTotalBytes: required=8, maximum=7",
                    assertThrows(IllegalArgumentException.class,
                            () -> engine.backward(objective, List.of(input), 7)).getMessage());
            retained = engine.backward(objective, List.of(input), 8);
            assertArrayEquals(bytes(4).putFloat(2.5f).array(), read(retained.objective()));
            assertEquals(1, retained.gradients().size());
            assertArrayEquals(bytes(4).putFloat(1.0f).array(),
                    read(retained.gradients().getFirst()));
            assertNotSame(retained.objective(), retained.gradients().getFirst());
            assertTrue(input.hostStorage().orElseThrow().isAlive());
        }
        assertArrayEquals(bytes(4).putFloat(2.5f).array(), read(retained.objective()));
        assertArrayEquals(bytes(4).putFloat(1.0f).array(),
                read(retained.gradients().getFirst()));
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

    private static Tensor floatTensor(Arena arena, float[] values) {
        Shape shape = Shape.of(values.length);
        TensorDescriptor descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment segment = arena.allocate(source.byteSize(), Float.BYTES);
        MemorySegment.copy(source, 0, segment, 0, source.byteSize());
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, values.length, segment)));
    }

    private static Tensor scalarFloatTensor(
            Arena arena, float value, boolean requiresGrad) {
        Shape shape = Shape.scalar();
        TensorDescriptor descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
        MemorySegment source = MemorySegment.ofArray(new float[] {value});
        MemorySegment segment = arena.allocate(source.byteSize(), Float.BYTES);
        MemorySegment.copy(source, 0, segment, 0, source.byteSize());
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, 1, segment)));
    }

    private static HostTensorValue assertMaterializes(
            Arena arena, Engine engine, DataType dataType, Object values, byte[] expected) {
        Shape shape = Shape.of(2);
        TensorDescriptor descriptor = new TensorDescriptor(dataType, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        MemorySegment source = switch (dataType) {
            case FLOAT64 -> MemorySegment.ofArray((double[]) values);
            case FLOAT32 -> MemorySegment.ofArray((float[]) values);
            case BFLOAT16 -> MemorySegment.ofArray((short[]) values);
            case INT64 -> MemorySegment.ofArray((long[]) values);
            case INT32 -> MemorySegment.ofArray((int[]) values);
            case BOOL -> MemorySegment.ofArray((byte[]) values);
        };
        MemorySegment segment = arena.allocate(source.byteSize(), dataType.byteWidth());
        MemorySegment.copy(source, 0, segment, 0, source.byteSize());
        var storage = new MemorySegmentStorage(dataType, 2, segment);
        Tensor input = TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
        Tensor output = input.contiguous();
        HostTensorValue value;
        try (RunResult result = engine.run(engine.prepare(engine.compile(List.of(output))),
                List.of(input))) {
            value = result.materialize(result.publications().getFirst(), expected.length);
            assertSame(dataType, value.dataType());
            assertSame(output.descriptor().shape(), value.shape());
            assertArrayEquals(expected, read(value));
            assertEquals(
                    "canonical byte count exceeds maximumBytes: required=" + expected.length
                            + ", maximum=" + (expected.length - 1),
                    assertThrows(IllegalArgumentException.class, () -> result.materialize(
                            result.publications().getFirst(), expected.length - 1)).getMessage());
        }
        assertArrayEquals(expected, read(value));
        return value;
    }

    private static ByteBuffer bytes(int size) {
        return ByteBuffer.allocate(size).order(ByteOrder.BIG_ENDIAN);
    }

    private static byte[] read(HostTensorValue value) {
        byte[] bytes = new byte[Math.toIntExact(value.byteSize())];
        value.bytes().get(bytes);
        return bytes;
    }
}
