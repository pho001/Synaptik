package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.nn.module.OwnedModelSourceTest.*;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Fresh positional batches, full signatures, preparation reuse and detached publication order. */
final class OwnedModelExecutionTest {
    @TempDir Path directory;

    @Test void twoInputConvenienceIsPositionalGenericCachedAndColdLoaded() throws Exception {
        Tensor a = prototype(), b = prototype(), w = scalar(2, true);
        Path file = directory.resolve("two-input");
        try (var model = Model.graph(List.of(b, a), List.of(a.mul(w).add(b)), List.of(parameter("weight", w)))) {
            model.save(file);
            assertNull(field(field(model, "execution"), "engine"));
        }
        try (var model = Model.loadGraph(file); Engine borrowed = Engine.standard()) {
            assertThrows(NullPointerException.class, () -> model.compute((Tensor) null, scalar(3, false)));
            assertThrows(NullPointerException.class, () -> model.compute(scalar(3, false), (Tensor) null));
            assertNull(field(field(model, "execution"), "engine"));
            assertSame(model, model.withEngine(borrowed));
            var value = model.compute(scalar(5, false), scalar(3, false));
            assertEquals(11, value.bytes().getFloat());
            Object prepared = field(field(model, "execution"), "inference");
            var poison = Engine.class.getDeclaredMethod("poisonBackendLookupForTesting");
            poison.setAccessible(true); poison.invoke(borrowed);
            assertEquals(14, model.compute(scalar(4, false), scalar(5, false)).bytes().getFloat());
            assertEquals(14, model.compute(List.of(scalar(4, false), scalar(5, false))).getFirst().bytes().getFloat());
            assertSame(prepared, field(field(model, "execution"), "inference"));
            assertThrows(IllegalStateException.class, () -> model.withEngine(borrowed));
            assertEquals(0, openResults(borrowed));
            model.close(); assertFalse(borrowed.isClosed());
            assertEquals(11, value.bytes().getFloat());
            assertTrue(assertThrows(IllegalStateException.class, () -> model.compute((Tensor) null, (Tensor) null))
                    .getMessage().contains("closed"));
        }
        for (int count : List.of(0, 1, 3)) {
            List<Tensor> inputs = java.util.stream.IntStream.range(0, count).mapToObj(i -> prototype()).toList();
            try (var model = Model.graph(inputs, List.of(w.neg()), List.of(parameter("weight", w)))) {
                assertThrows(IllegalArgumentException.class, () -> model.compute((Tensor) null, (Tensor) null));
                assertNull(field(field(model, "execution"), "engine"));
                assertFalse((Boolean) field(field(model, "execution"), "frozen"));
            }
        }
        try (var model = Model.graph(List.of(a, b), List.of(a.neg(), b.neg()), List.of())) {
            assertThrows(IllegalArgumentException.class, () -> model.compute((Tensor) null, (Tensor) null));
        }
        try (var generic = Model.define(t -> (String text) -> text)) {
            assertThrows(UnsupportedOperationException.class, () -> generic.compute((Tensor) null, (Tensor) null));
        }
    }

    @Test void twoInputConvenienceValidatesUnusedFullDescriptorWithoutFreezeOrEngine() throws Exception {
        Tensor used = prototype(), unused = prototype();
        try (var model = Model.graph(List.of(used, unused), List.of(used.neg()), List.of())) {
            for (TensorDescriptor mismatch : List.of(
                    new TensorDescriptor(DataType.FLOAT64, unused.descriptor().shape(), unused.descriptor().layout(), false),
                    new TensorDescriptor(DataType.FLOAT32, unused.descriptor().shape(), Optional.empty(), false),
                    new TensorDescriptor(DataType.FLOAT32, unused.descriptor().shape(), unused.descriptor().layout(), true),
                    ModelBuilderTest.descriptor(false, 1))) {
                assertThrows(IllegalArgumentException.class, () -> model.compute(scalar(2, false), TensorFactory.create(mismatch)));
            }
            assertThrows(IllegalStateException.class, () -> model.compute(scalar(2, false), unused));
            assertNull(field(field(model, "execution"), "engine"));
            assertFalse((Boolean) field(field(model, "execution"), "frozen"));
            assertEquals(-2, model.compute(scalar(2, false), scalar(99, false)).bytes().getFloat());
        }
    }

    @Test void finalCompiledOrderNotSourceOrderAndUnusedSlotsStillValidate() throws Exception {
        Tensor a = prototype(), b = prototype(), unused = prototype(), w = scalar(2, true);
        try (var model = Model.graph(List.of(unused, b, a),
                List.of(w.mul(a).add(b), b.neg()), List.of(parameter("weight", w)))) {
            var first = model.compute(List.of(scalar(99, false), scalar(5, false), scalar(3, false)));
            assertEquals(List.of(11f, -5f), first.stream().map(v -> v.bytes().getFloat()).toList());
            Object state = field(model, "execution"), prepared = field(state, "inference");
            var poison = Engine.class.getDeclaredMethod("poisonBackendLookupForTesting");
            poison.setAccessible(true);
            poison.invoke(field(state, "engine"));
            var second = model.compute(List.of(scalar(-99, false), scalar(7, false), scalar(4, false)));
            assertEquals(List.of(15f, -7f), second.stream().map(v -> v.bytes().getFloat()).toList());
            assertSame(prepared, field(state, "inference"));
            assertThrows(IllegalStateException.class,
                    () -> model.compute(List.of(a, b, w)));
            assertThrows(IllegalStateException.class,
                    () -> model.compute(List.of(unused, scalar(1, false), scalar(2, false))));
            assertEquals(11, first.getFirst().bytes().getFloat());
        }
    }

    @Test void everyDescriptorComponentRejectsIncludingUnusedSlotWithoutFreeze() throws Exception {
        Tensor used = prototype(), unused = prototype();
        try (var model = Model.graph(List.of(used, unused), List.of(used.neg()), List.of());
                Engine selected = Engine.standard()) {
            var descriptor = used.descriptor();
            for (TensorDescriptor mismatch : List.of(
                    new TensorDescriptor(DataType.FLOAT64, descriptor.shape(), descriptor.layout(), false),
                    new TensorDescriptor(DataType.FLOAT32, descriptor.shape(), Optional.empty(), false),
                    new TensorDescriptor(DataType.FLOAT32, descriptor.shape(), descriptor.layout(), true))) {
                assertThrows(IllegalArgumentException.class, () -> model.compute(
                        List.of(scalar(1, false), TensorFactory.create(mismatch))));
            }
            assertSame(model, model.withEngine(selected));
            assertThrows(IllegalArgumentException.class, () -> model.compute(scalar(1, false)));
            assertThrows(IllegalArgumentException.class, () -> model.compute(
                    List.of(scalar(1, false), scalar(2, false)), -1));
            assertSame(model, model.withEngine(selected));
            assertEquals(-3, model.compute(List.of(scalar(3, false), scalar(9, false))).getFirst().bytes().getFloat());
            assertThrows(IllegalStateException.class, () -> model.withEngine(selected));
        }
    }

    @Test void aggregateBoundsCheckedAndResultsNeverRetained() throws Exception {
        Tensor x = prototype();
        var model = Model.graph(List.of(x), List.of(x.neg(), x.add(x)), List.of());
        var state = field(model, "execution");
        assertThrows(IllegalArgumentException.class, () -> model.compute(List.of(scalar(2, false)), 7));
        var result = model.compute(List.of(scalar(3, false)), 8);
        Engine engine = (Engine) field(state, "engine");
        assertEquals(0, openResults(engine));
        assertEquals(List.of(-3f, 6f), result.stream().map(v -> v.bytes().getFloat()).toList());
        model.close();
        assertTrue(engine.isClosed());
        assertEquals(-3, result.getFirst().bytes().getFloat());
    }

    @Test void detachedValuesOutliveBorrowedNativeBatchAndEngine() {
        Tensor x = prototype();
        io.github.pho001.synaptik.engine.HostTensorValue value;
        try (var model = Model.graph(List.of(x), List.of(x.neg()), List.of());
                Engine engine = Engine.standard()) {
            model.withEngine(engine);
            try (Arena batch = Arena.ofShared()) {
                var memory = batch.allocate(4, 4);
                memory.set(ValueLayout.JAVA_FLOAT, 0, 7);
                Tensor input = TensorFactory.create(x.descriptor(), Optional.empty(),
                        Optional.of(new MemorySegmentStorage(DataType.FLOAT32, 1, memory)));
                value = model.compute(input);
            }
            model.close();
            assertFalse(engine.isClosed());
        }
        assertEquals(-7, value.bytes().getFloat());
    }

    @Test void saveBeforeComputeAndLoadHaveSameBoundaryAndNoEngine() throws Exception {
        Tensor w = scalar(2, true), x = prototype(), y = prototype();
        Path file = directory.resolve("model");
        try (var model = Model.graph(List.of(x, y), List.of(w.mul(y).add(x), x.neg()),
                List.of(parameter("layer.weight", w)))) {
            model.save(file);
        }
        assertTrue(assertThrows(IllegalArgumentException.class, () -> Model.load(file))
                .getMessage().contains("loadGraph"));
        try (var loaded = Model.loadGraph(file)) {
            assertNull(field(field(loaded, "execution"), "engine"));
            assertEquals(List.of(11f, -3f), loaded.compute(List.of(scalar(3, false), scalar(4, false)))
                    .stream().map(v -> v.bytes().getFloat()).toList());
        }
    }

    @Test void coldForwardComposesProducedInputsWithoutExecutionOrChangingSource() throws Exception {
        Tensor x = prototype();
        try (var model = Model.graph(List.of(x), List.of(x.neg()), List.of())) {
            Tensor replacement = prototype().neg().contiguous();
            Tensor output = model.forward(List.of(replacement)).getFirst();
            assertSame(replacement, output.provenance().orElseThrow().inputs().getFirst());
            assertNull(field(field(model, "execution"), "engine"));
            assertEquals(-2, model.compute(scalar(2, false)).bytes().getFloat());
        }
    }

    @Test void unusedSlotsStillRequireLiveAccessibleCapacityBeforeSetup() throws Exception {
        Tensor x = prototype(), unused = prototype();
        try (var model = Model.graph(List.of(x, unused), List.of(x.neg()), List.of())) {
            Tensor dead;
            try (Arena arena = Arena.ofShared()) {
                dead = TensorFactory.create(unused.descriptor(), Optional.empty(),
                        Optional.of(new MemorySegmentStorage(DataType.FLOAT32, 1, arena.allocate(4, 4))));
            }
            assertThrows(IllegalStateException.class, () -> model.compute(List.of(scalar(1, false), dead)));
            assertFalse((Boolean) field(field(model, "execution"), "frozen"));
            try (Arena arena = Arena.ofConfined(); var pool = java.util.concurrent.Executors.newSingleThreadExecutor()) {
                Tensor confined = TensorFactory.create(unused.descriptor(), Optional.empty(),
                        Optional.of(new MemorySegmentStorage(DataType.FLOAT32, 1, arena.allocate(4, 4))));
                pool.submit(() -> assertThrows(IllegalStateException.class,
                        () -> model.compute(List.of(scalar(1, false), confined)))).get();
            }
        }
    }
}
