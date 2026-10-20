package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.nn.module.OwnedModelSourceTest.field;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import io.github.pho001.synaptik.training.Sgd;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.Arena;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.random.RandomGeneratorFactory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Public construction, deterministic private acquisition, exact signatures and save isolation. */
final class ModelBuilderTest {
    @TempDir Path directory;

    static TensorDescriptor descriptor(boolean grad, long... axes) {
        Shape shape = Shape.of(axes);
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }

    static Tensor values(boolean grad, long[] axes, float... values) {
        return TensorFactory.fromFlatArray(descriptor(grad, axes), Optional.empty(), values);
    }

    static Tensor input(long... axes) { return TensorFactory.create(descriptor(false, axes)); }

    static byte[] bytes(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE);
    }

    static GraphCheckpoint.Source source(Model<?, ?> model) throws Exception {
        return (GraphCheckpoint.Source) field(field(field(model, "execution"), "owned"), "source");
    }

    @Test void exactClosedPublicSurfaceAndRecipeExample() {
        assertTrue(Modifier.isFinal(ModelBuilder.class.getModifiers()));
        assertEquals(0, ModelBuilder.class.getConstructors().length);
        assertEquals(Set.of("input", "seed", "linear", "relu", "tanh", "sigmoid", "conv1d", "conv2d",
                "conv3d", "maxPool2d", "flatten", "rnn", "gru", "lstm", "lastTimeStep", "build"),
                Arrays.stream(ModelBuilder.class.getDeclaredMethods())
                        .filter(m -> Modifier.isPublic(m.getModifiers())).map(m -> m.getName())
                        .collect(Collectors.toSet()));
        try (var model = Model.builder().input(DataType.FLOAT32, 32, 784)
                .seed(42).linear(128).relu().linear(10, false).build()) {
            assertEquals(Shape.of(32, 10), model.forward(input(32, 784)).descriptor().shape());
            assertEquals(List.of("layers.0.weight", "layers.0.bias", "layers.2.weight"),
                    model.stateDictionary().entries().stream().map(StateEntry::path).toList());
            assertEquals(ForwardMode.EVALUATION, model.mode());
        }
    }

    @Test void buildsAreFreshDeterministicAndIndependentOfLaterBuilderEdits() throws Exception {
        long[] axes = {2, 3};
        var builder = Model.builder().input(DataType.FLOAT32, axes).linear(3).relu().linear(3);
        axes[1] = 99;
        assertSame(builder, builder.seed(7));
        try (var first = builder.build(); var second = builder.build()) {
            var a = first.stateDictionary().entries(); var b = second.stateDictionary().entries();
            for (int i = 0; i < a.size(); i++) {
                assertNotSame(a.get(i).value(), b.get(i).value());
                assertNotSame(a.get(i).value().hostStorage().orElseThrow(), b.get(i).value().hostStorage().orElseThrow());
                assertArrayEquals(bytes(a.get(i).value()), bytes(b.get(i).value()));
            }
            for (int i : List.of(1, 3)) for (byte lane : bytes(a.get(i).value())) assertEquals(0, lane);
            var expected = ParameterInitializers.glorotUniform(Shape.of(3, 3), DataType.FLOAT32,
                    RandomGeneratorFactory.of("L64X128MixRandom").create(7 + 0x9E3779B97F4A7C15L * 2));
            assertArrayEquals(bytes(expected), bytes(a.get(2).value()));
            assertFalse(Arrays.equals(bytes(a.get(0).value()), bytes(a.get(2).value())));
            assertNull(field(field(first, "execution"), "engine"));
            builder.linear(1).seed(-1);
            assertEquals(Shape.of(2, 3), first.forward(input(2, 3)).descriptor().shape());
            assertEquals(3, first.children().get("layers").children().size());
            assertThrows(UnsupportedOperationException.class, () -> first.children().clear());
        }
        try (var zero = Model.builder().input(DataType.FLOAT32, 1, 2).linear(2).build();
                var explicit = Model.builder().seed(0).input(DataType.FLOAT32, 1, 2).linear(2).build()) {
            assertArrayEquals(bytes(zero.stateDictionary().entries().getFirst().value()),
                    bytes(explicit.stateDictionary().entries().getFirst().value()));
        }
    }

    @Test void suppliedLeavesCopyOncePreserveAliasesAndNeverMutateCaller() throws Exception {
        Tensor w = values(true, new long[]{2, 2}, 1, 0, 0, 1);
        Tensor b = values(true, new long[]{2}, .25f, -.5f);
        var ws = w.hostStorage().orElseThrow(); byte[] before = bytes(w), bias = bytes(b);
        try (var model = Model.builder().input(DataType.FLOAT32, 1, 2).linear(w, b).linear(w).build()) {
            var entries = model.stateDictionary().entries();
            assertSame(entries.get(0).value(), entries.get(2).value());
            assertNotSame(w, entries.get(0).value());
            assertNotSame(ws, entries.get(0).value().hostStorage().orElseThrow());
            assertArrayEquals(before, bytes(entries.get(0).value()));
            assertArrayEquals(bias, bytes(entries.get(1).value()));
            model.compute(values(false, new long[]{1, 2}, 2, 3));
            assertThrows(IllegalArgumentException.class, () -> model.configureTraining(descriptor(false, 1, 2),
                    ModelBuilderExecutionTest::objective, new Sgd(.01))); // Training alias policy is unchanged.
        }
        assertSame(ws, w.hostStorage().orElseThrow());
        assertArrayEquals(before, bytes(w)); assertArrayEquals(bias, bytes(b));
    }

    @Test void invalidRecipesAndGeometryRejectWithoutPublishingPartialModels() {
        assertThrows(IllegalStateException.class, () -> Model.builder().linear(2));
        assertThrows(IllegalStateException.class, () -> Model.builder().build());
        assertThrows(IllegalStateException.class, () -> Model.builder().input(DataType.FLOAT32, 1).build());
        for (long[] axes : List.of(new long[]{}, new long[]{0}, new long[]{-1})) {
            assertThrows(IllegalArgumentException.class, () -> Model.builder().input(DataType.FLOAT32, axes));
        }
        assertThrows(IllegalArgumentException.class, () -> Model.builder().input(DataType.INT32, 1));
        var b = Model.builder().input(DataType.FLOAT32, 1, 2);
        assertThrows(IllegalStateException.class, () -> b.input(DataType.FLOAT32, 1));
        assertThrows(IllegalArgumentException.class, () -> b.linear(0));
        assertThrows(IllegalArgumentException.class, () -> b.conv3d(1, 0, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> b.maxPool2d(-1, 2));
        assertThrows(IllegalArgumentException.class, () -> b.rnn(0));
        for (var invalid : List.of(Model.builder().input(DataType.FLOAT32, 1).flatten(),
                Model.builder().input(DataType.FLOAT32, 1, 2).lastTimeStep(),
                Model.builder().input(DataType.FLOAT32, 1, 2).lstm(2),
                Model.builder().input(DataType.FLOAT32, 1, 1, 2, 2).conv2d(1, 3, 3))) {
            assertThrows(IllegalArgumentException.class, invalid::build);
        }
        assertThrows(ArithmeticException.class, () -> Model.builder().input(DataType.FLOAT32, Long.MAX_VALUE, 2));
        Tensor good = values(true, new long[]{2, 2}, 1, 0, 0, 1);
        assertThrows(IllegalArgumentException.class, () -> b.linear(good.neg()));
        assertThrows(IllegalArgumentException.class, () -> b.linear(values(false, new long[]{2, 2}, 1, 0, 0, 1)));
        var unavailable = b.linear(TensorFactory.create(good.descriptor()));
        assertThrows(IllegalStateException.class, unavailable::build);
        assertArrayEquals(new float[]{1, 0, 0, 1}, good.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_FLOAT));
    }

    @Test void cnnAndStackedRecurrentShapesAndFlattenAxisAreExplicit() {
        try (var cnn = Model.builder().input(DataType.FLOAT32, 32, 1, 28, 28)
                .conv2d(32, 3, 3).relu().maxPool2d(2, 2).conv2d(64, 3, 3).relu()
                .maxPool2d(2, 2).flatten().linear(128).relu().linear(10).build();
                var stack = Model.builder().input(DataType.FLOAT32, 20, 32, 8).lstm(128).lstm(64).build();
                var last = Model.builder().input(DataType.FLOAT32, 20, 32, 8).lstm(128).lstm(64)
                        .lastTimeStep().linear(10).build();
                var flatten = Model.builder().input(DataType.FLOAT32, 3, 2, 4).flatten().build()) {
            assertEquals(Shape.of(32, 10), cnn.forward(input(32, 1, 28, 28)).descriptor().shape());
            assertEquals(Shape.of(20, 32, 64), stack.forward(input(20, 32, 8)).descriptor().shape());
            assertEquals(Shape.of(32, 10), last.forward(input(20, 32, 8)).descriptor().shape());
            assertEquals(Shape.of(3, 8), flatten.forward(input(3, 2, 4)).descriptor().shape());
        }
    }

    @Test void unsupportedSaveDoesNoFileWorkAndDoesNotFreezeOrDisableTraining() throws Exception {
        Tensor w = values(true, new long[]{1, 1}, .75f);
        try (var model = Model.builder().input(DataType.FLOAT32, 1, 1, 1).flatten().linear(w).build();
                Engine engine = Engine.standard()) {
            Path absent = directory.resolve("absent"), sentinel = directory.resolve("sentinel");
            Files.writeString(sentinel, "unchanged");
            for (Path path : List.of(absent, sentinel)) {
                assertThrows(IllegalArgumentException.class, () -> model.save(path));
                assertNull(field(field(model, "execution"), "engine"));
            }
            assertFalse(Files.exists(absent)); assertEquals("unchanged", Files.readString(sentinel));
            try (var files = Files.list(directory)) { assertEquals(List.of(sentinel), files.toList()); }
            assertSame(model, model.withEngine(engine));
            Tensor batch = values(false, new long[]{1, 1, 1}, 2);
            assertEquals(1.5f, model.compute(batch).bytes().getFloat());
            model.configureTraining(descriptor(false, 1, 1), ModelBuilderExecutionTest::objective, new Sgd(.01));
            assertTrue(model.train(batch, values(false, new long[]{1, 1}, .3f)).objective().bytes().getFloat() > 0);
            assertTrue(Float.isFinite(model.compute(batch).bytes().getFloat()));
            assertThrows(IllegalArgumentException.class, () -> model.save(sentinel));
            assertEquals("unchanged", Files.readString(sentinel));
        }
        assertEquals(.75f, w.hostStorage().orElseThrow().segment().get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
    }

    @Test void coldProducedInputRetainsOwnedWeightsAndCachedGraphAndLiveGuards() throws Exception {
        try (var model = Model.builder().input(DataType.FLOAT32, 1, 2).linear(2).relu().build()) {
            var source = source(model); Tensor x = input(1, 2).abs().contiguous();
            assertEquals(source.outputs().getFirst().descriptor(), model.forward(x).descriptor());
            assertSame(source, source(model));
            assertNull(field(field(model, "execution"), "engine"));
            assertThrows(IllegalArgumentException.class, () -> model.forward(input(2, 2)));
            model.compute(values(false, new long[]{1, 2}, 1, 2));
            var prepared = field(field(model, "execution"), "inference");
            model.forward(x);
            model.compute(values(false, new long[]{1, 2}, -1, 3));
            assertSame(prepared, field(field(model, "execution"), "inference"));
            assertSame(source.outputs().getFirst(), source(model).outputs().getFirst());
            model.children().get("layers").train();
            assertThrows(IllegalStateException.class, () -> model.forward(x));
            assertThrows(IllegalStateException.class, () -> model.compute(values(false, new long[]{1, 2}, 1, 2)));
        }
    }

    @Test void rawNativeStateIsAcquiredBeforeExpressionsAndOutlivesCallerArena() {
        Model<Tensor, Tensor> model;
        try (Arena arena = Arena.ofConfined()) {
            var memory = arena.allocate(4, 4);
            memory.set(ValueLayout.JAVA_FLOAT, 0, .75f);
            Tensor weight = TensorFactory.create(descriptor(true, 1, 1), Optional.empty(), Optional.of(
                    new MemorySegmentStorage(DataType.FLOAT32, 1, memory.asReadOnly())));
            model = Model.builder().input(DataType.FLOAT32, 1, 1).linear(weight).build();
            assertSame(memory.asReadOnly().scope(), weight.hostStorage().orElseThrow().segment().scope());
        }
        try (model) {
            assertEquals(1.5f, model.compute(values(false, new long[]{1, 1}, 2)).bytes().getFloat());
            model.configureTraining(descriptor(false, 1, 1), ModelBuilderExecutionTest::objective, new Sgd(.01));
            model.train(values(false, new long[]{1, 1}, 2), values(false, new long[]{1, 1}, .3f));
        }
    }

    @Test void unsupportedConvolutionPoolAndRecurrentSavesPreserveTargetsAndInference() throws Exception {
        var builders = List.of(Model.builder().input(DataType.FLOAT32, 1, 1, 3, 3).conv2d(1, 2, 2),
                Model.builder().input(DataType.FLOAT32, 1, 1, 2, 2).maxPool2d(2, 2),
                Model.builder().input(DataType.FLOAT32, 2, 1, 1).gru(1).lastTimeStep());
        var batches = List.of(values(false, new long[]{1, 1, 3, 3}, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                values(false, new long[]{1, 1, 2, 2}, 1, 2, 3, 4),
                values(false, new long[]{2, 1, 1}, .2f, -.3f));
        for (int i = 0; i < builders.size(); i++) {
            try (var model = builders.get(i).build(); Engine engine = Engine.standard()) {
                Path absent = directory.resolve("absent" + i), sentinel = directory.resolve("sentinel" + i);
                Files.writeString(sentinel, "unchanged");
                assertThrows(IllegalArgumentException.class, () -> model.save(absent));
                assertThrows(IllegalArgumentException.class, () -> model.save(sentinel));
                assertFalse(Files.exists(absent));
                assertEquals("unchanged", Files.readString(sentinel));
                assertNull(field(field(model, "execution"), "engine"));
                model.withEngine(engine);
                var result = model.compute(batches.get(i));
                assertTrue(Float.isFinite(result.bytes().getFloat()));
                assertThrows(IllegalArgumentException.class, () -> model.save(sentinel));
                ModelBuilderExecutionTest.sameValues(result, model.compute(batches.get(i)), 0);
            }
        }
        try (var files = Files.list(directory)) { assertEquals(3, files.count()); }
    }
}
