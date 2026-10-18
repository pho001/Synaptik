package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.nn.checkpoint.ModelCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.layers.Activation;
import io.github.pho001.synaptik.nn.layers.Embedding;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.module.DeclarativeModel;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import io.github.pho001.synaptik.nn.module.ModelSpecification.ActivationKind;
import io.github.pho001.synaptik.nn.module.Sequential;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Ordinary save/automatic-load/fresh-CPU workflow, with no manual definition or Arena on load. */
final class WholeModelCheckpointEngineIntegrationTest {
    @TempDir Path directory;

    @Test
    void ordinarySequentialSurvivesSourceClosureAndExecutesVariableBatchesThenResaves() throws Exception {
        Path file = directory.resolve("ordinary.chk");
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            var source = new Sequential(List.of(
                    new Linear(nativeTensor(arena, Shape.of(2, 2), true, 1, -2, 3, 1),
                            nativeTensor(arena, Shape.of(2), true, 0.5f, -0.5f)),
                    new Activation(ActivationKind.RELU)));
            source.train();
            ModelCheckpoint.save(file, source, engine);
        }
        DeclarativeModel loaded = ModelCheckpoint.load(file);
        Files.delete(file);
        assertEquals(ForwardMode.EVALUATION, loaded.mode());
        assertHeapState(loaded);
        try (Arena arena = Arena.ofShared(); Engine fresh = Engine.standard()) {
            for (Shape shape : List.of(Shape.of(2), Shape.of(1, 2), Shape.of(5, 2), Shape.of(2, 3, 2))) {
                int count = Math.toIntExact(shape.knownElementCount().orElseThrow());
                float[] input = pattern(count);
                float[] expected = new float[count];
                for (int i = 0; i < count; i += 2) {
                    expected[i] = Math.max(0, input[i] - 2 * input[i + 1] + 0.5f);
                    expected[i + 1] = Math.max(0, 3 * input[i] + input[i + 1] - 0.5f);
                }
                assertArrayEquals(expected, values(fresh.compute(loaded.forward(nativeTensor(arena, shape, false, input)))), 1e-6f);
            }
            ModelCheckpoint.save(file, loaded, fresh);
        }
        var resaved = ModelCheckpoint.load(file);
        assertEquals(loaded.describe().root(), resaved.describe().root());
        assertHeapState(resaved);
    }

    @Test
    void exactWeightAliasesSurviveClosureComputeResaveAndIndependentLoad() throws Exception {
        Path file = directory.resolve("tied.chk");
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor shared = nativeTensor(arena, Shape.of(2, 2), true, 1, 0, 0, 2);
            ModelCheckpoint.save(file, new Sequential(List.of(new Linear(shared), new Linear(shared))), engine);
        }
        var loaded = ModelCheckpoint.load(file);
        Tensor first = loaded.parametersRecursively().get("0.weight").value();
        assertSame(first, loaded.parametersRecursively().get("1.weight").value());
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            assertArrayEquals(new float[] {1, 8, -3, 16}, values(engine.compute(loaded.forward(
                    nativeTensor(arena, Shape.of(2, 2), false, 1, 2, -3, 4)))), 0);
            ModelCheckpoint.save(file, loaded, engine);
        }
        var second = ModelCheckpoint.load(file);
        assertEquals(loaded.describe().aliases(), second.describe().aliases());
        assertNotSame(first, second.parametersRecursively().get("0.weight").value());
        assertSame(second.parametersRecursively().get("0.weight").value(),
                second.parametersRecursively().get("1.weight").value());
    }

    @Test
    void embeddingKeepsIntegralInputContractAndExecutesAfterAutomaticLoad() throws Exception {
        Path file = directory.resolve("embedding.chk");
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            ModelCheckpoint.save(file, new Sequential(List.of(new Embedding(
                    nativeTensor(arena, Shape.of(3, 2), true, 1, 2, 3, 4, 5, 6)))), engine);
        }
        var loaded = ModelCheckpoint.load(file);
        try (Engine engine = Engine.standard()) {
            Tensor int32 = integral(new int[] {2, 0}, Shape.of(2));
            Tensor int64 = integral(new long[] {1, 2, 0, 1}, Shape.of(2, 2));
            assertArrayEquals(new float[] {5, 6, 1, 2}, values(engine.compute(loaded.forward(int32))), 0);
            assertArrayEquals(new float[] {3, 4, 5, 6, 1, 2, 3, 4}, values(engine.compute(loaded.forward(int64))), 0);
            ModelCheckpoint.save(file, loaded, engine);
        }
        assertEquals(loaded.describe().root(), ModelCheckpoint.load(file).describe().root());
    }

    private static void assertHeapState(DeclarativeModel model) {
        for (var state : model.stateDictionary().entries()) {
            assertFalse(state.value().hostStorage().orElseThrow().segment().isNative());
            assertTrue(state.value().descriptor().requiresGrad());
            assertTrue(state.value().provenance().isEmpty());
        }
    }

    private static Tensor nativeTensor(Arena arena, Shape shape, boolean grad, float... values) {
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment storage = arena.allocate(source.byteSize(), 4); storage.copyFrom(source);
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad), Optional.empty(),
                Optional.of(new MemorySegmentStorage(DataType.FLOAT32, values.length, storage)));
    }

    private static Tensor integral(Object array, Shape shape) {
        boolean wide = array instanceof long[];
        MemorySegment segment = wide ? MemorySegment.ofArray((long[]) array) : MemorySegment.ofArray((int[]) array);
        DataType type = wide ? DataType.INT64 : DataType.INT32;
        return TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false), Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, shape.knownElementCount().orElseThrow(), segment)));
    }

    private static float[] values(HostTensorValue value) {
        var buffer = value.bytes().asFloatBuffer();
        float[] result = new float[buffer.remaining()]; buffer.get(result); return result;
    }

    private static float[] pattern(int count) {
        float[] result = new float[count];
        for (int i = 0; i < count; i++) result[i] = (i % 7 - 3) * 0.25f;
        return result;
    }
}
