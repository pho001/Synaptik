package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint.Leaf;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint.Role;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.Linear;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Cold source acquisition, final state installation, raw values and old generic compatibility. */
final class OwnedModelSourceTest {
    @TempDir Path directory;

    static Tensor scalar(float value, boolean gradient) {
        Tensor tensor = TensorFactory.allocate(new TensorDescriptor(DataType.FLOAT32,
                Shape.scalar(), Optional.of(LayoutDescriptor.contiguous(Shape.scalar())), gradient));
        tensor.hostStorage().orElseThrow().segment().set(ValueLayout.JAVA_FLOAT_UNALIGNED, 0, value);
        return tensor;
    }

    static Tensor prototype() { return TensorFactory.create(scalar(0, false).descriptor()); }
    static float number(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0);
    }
    static Leaf parameter(String path, Tensor tensor) { return new Leaf(path, Role.PARAMETER, tensor); }

    static Object field(Object object, String name) throws Exception {
        Class<?> owner = object.getClass();
        while (owner != null) {
            try {
                Field field = owner.getDeclaredField(name);
                field.setAccessible(true);
                return field.get(object);
            } catch (NoSuchFieldException absent) { owner = owner.getSuperclass(); }
        }
        throw new NoSuchFieldException(name);
    }

    static int openResults(io.github.pho001.synaptik.engine.Engine engine) throws Exception {
        return ((java.util.Collection<?>) field(field(engine, "delegate"), "openResults")).size();
    }

    static final class Weight extends Module {
        final Parameter weight;
        Weight(Tensor tensor) { weight = parameter("weight", tensor); }
    }

    @Test void definitionIsOneShotEvalAndClosureCapturedWeightIsAcquired() throws Exception {
        Tensor external = scalar(2, true);
        var storage = external.hostStorage().orElseThrow();
        AtomicInteger definitions = new AtomicInteger(), forwards = new AtomicInteger();
        AtomicReference<Weight> child = new AtomicReference<>();
        AtomicReference<Topology> collector = new AtomicReference<>();
        try (var model = Model.defineTensor(prototype().descriptor(), topology -> {
            definitions.incrementAndGet(); collector.set(topology);
            child.set(topology.addModule("layer", new Weight(external)));
            return input -> {
                forwards.incrementAndGet();
                assertEquals(ForwardMode.EVALUATION, child.get().mode());
                return external.mul(input); // deliberately captures the original, not the wrapper
            };
        })) {
            Tensor owned = model.stateDictionary().entries().getFirst().value();
            assertNotSame(external, owned);
            assertNotSame(storage, owned.hostStorage().orElseThrow());
            assertSame(storage, external.hostStorage().orElseThrow());
            assertEquals(2, number(external));
            assertSame(owned, child.get().weight.value());
            Tensor expression = model.forward(prototype());
            assertTrue(expression.provenance().orElseThrow().inputs().contains(owned));
            assertFalse(expression.provenance().orElseThrow().inputs().contains(external));
            assertEquals(1, definitions.get()); assertEquals(1, forwards.get());
            assertThrows(IllegalStateException.class, () -> collector.get().addModule(null, null));
            assertNull(field(field(model, "execution"), "engine"));
            model.save(directory.resolve("cold"));
            assertNull(field(field(model, "execution"), "engine"));
        }
    }

    @Test void initializesDeferredLayerOnceBeforeAcquisition() {
        AtomicReference<Linear> layer = new AtomicReference<>();
        Tensor signature = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(1, 2), Optional.empty(), false));
        try (var model = Model.defineTensor(signature.descriptor(), topology -> {
            layer.set(topology.addModule("linear", new Linear(1, false, DataType.FLOAT32,
                    ParameterInitialization.zeros(),
                    java.util.random.RandomGeneratorFactory.of("L64X128MixRandom"), 7)));
            return layer.get()::forward;
        })) {
            assertEquals(List.of("linear.weight"), model.stateDictionary().entries().stream()
                    .map(StateEntry::path).toList());
            assertEquals(Shape.of(1, 2), model.stateDictionary().entries().getFirst().value().descriptor().shape());
        }
    }

    @Test void constantsAreDiscoveredWithoutInferringParameterRoleAndAvoidPathCollisions() throws Exception {
        Tensor registered = scalar(2, true), constant = scalar(3, true);
        try (var model = Model.defineTensor(prototype().descriptor(), topology -> {
            topology.addModule("__constant0", new Weight(registered));
            return x -> registered.mul(x).add(constant);
        })) {
            assertEquals(1, model.stateDictionary().entries().size());
            var source = (GraphCheckpoint.Source) field(field(field(model, "execution"), "owned"), "source");
            assertEquals(List.of("__constant0.weight", "__constant1"),
                    source.leaves().stream().map(Leaf::name).toList());
            assertEquals(Role.CONSTANT, source.leaves().getLast().role());
            assertTrue(source.leaves().getLast().tensor().descriptor().requiresGrad());
        }
    }

    @Test void strictInstallationPreservesAliasesAndRawValues() throws Exception {
        Tensor external = scalar(Float.intBitsToFloat(0xffc01234), true);
        try (var model = Model.graph(List.of(prototype()), List.of(external.neg()),
                List.of(parameter("a.weight", external), parameter("b.weight", external)))) {
            var entries = model.stateDictionary().entries();
            assertSame(entries.getFirst().value(), entries.getLast().value());
            assertNotSame(external, entries.getFirst().value());
            assertEquals(0xffc01234, Float.floatToRawIntBits(number(entries.getFirst().value())));
            model.save(directory.resolve("alias"));
            try (var loaded = Model.loadGraph(directory.resolve("alias"))) {
                assertSame(loaded.stateDictionary().entries().getFirst().value(),
                        loaded.stateDictionary().entries().getLast().value());
            }
        }
    }

    @Test void rejectsNoncanonicalPathsRolesUnreachableStateAndUnsupportedSource() {
        Tensor x = prototype(), weight = scalar(2, true), buffer = scalar(3, false);
        assertThrows(IllegalArgumentException.class, () -> Model.graph(List.of(x),
                List.of(weight.add(buffer)), List.of(new Leaf("buffer", Role.BUFFER, buffer),
                        parameter("weight", weight))));
        assertThrows(IllegalArgumentException.class, () -> Model.graph(List.of(x), List.of(weight.neg()),
                List.of(parameter("a", weight), new Leaf("b", Role.CONSTANT, weight))));
        assertThrows(IllegalArgumentException.class, () -> Model.graph(List.of(x), List.of(x.neg()),
                List.of(parameter("weight", weight))));
        assertThrows(IllegalArgumentException.class, () -> Model.graph(List.of(x), List.of(x.abs()), List.of()));
        assertThrows(IllegalArgumentException.class, () -> Model.graph(List.of(x), List.of(weight.neg()),
                List.of(parameter("a", weight), parameter("a.b", weight))));
        assertThrows(IllegalArgumentException.class, () -> Model.defineTensor(x.descriptor(), t -> {
            t.addModule("unused", new Weight(weight)); return Tensor::neg;
        }));
    }

    @Test void sevenRawTypesAndZeroNodeDataPersistWithoutEngine() throws Exception {
        for (DataType type : DataType.values()) {
            var shape = Shape.of(2);
            var descriptor = new TensorDescriptor(type, shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)), false);
            byte[] bytes = new byte[2 * type.byteWidth()];
            if (type == DataType.BOOL) { bytes[0] = 1; }
            else java.util.Arrays.fill(bytes, (byte) 0xa5);
            var tensor = TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                    new MemorySegmentStorage(type, 2, MemorySegment.ofArray(bytes))));
            Path path = directory.resolve(type.name());
            try (var model = Model.graph(List.of(), List.of(tensor),
                    List.of(new Leaf("state", Role.BUFFER, tensor)))) {
                model.save(path);
                assertNull(field(field(model, "execution"), "engine"));
            }
            try (var model = Model.loadGraph(path)) {
                assertArrayEquals(bytes, model.stateDictionary().entries().getFirst().value()
                        .hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE));
                assertThrows(IllegalArgumentException.class, () -> model.compute(List.of()));
            }
            assertTrue(assertThrows(IllegalArgumentException.class, () -> Model.load(path))
                    .getMessage().contains("loadGraph"));
        }
    }

    @Test void bothConstructionRoutesAndLoadedForwardRemainColdForAllThreeActivations() throws Exception {
        Tensor x = prototype();
        for (var activation : List.<java.util.function.UnaryOperator<Tensor>>of(Tensor::relu, Tensor::tanh, Tensor::sigmoid)) {
            Tensor expected = activation.apply(x);
            try (var defined = Model.defineTensor(x.descriptor(), topology -> activation::apply);
                    var graph = Model.graph(List.of(x), List.of(expected), List.of())) {
                for (var model : List.of(defined, graph)) {
                    Tensor result = model == defined ? defined.forward(x) : graph.forward(List.of(x)).getFirst();
                    assertEquals(expected.provenance().orElseThrow().operation(), result.provenance().orElseThrow().operation());
                    assertSame(x, result.provenance().orElseThrow().inputs().getFirst());
                    assertNull(field(field(model, "execution"), "engine"));
                    Path path = directory.resolve("active"); model.save(path);
                    try (var loaded = Model.load(path)) {
                        assertEquals(expected.provenance().orElseThrow().operation(),
                                loaded.forward(x).provenance().orElseThrow().operation());
                        assertNull(field(field(loaded, "execution"), "engine"));
                    }
                }
            }
        }
    }

    @Test void genericUnsupportedMethodsPrecedeArgumentsAndCloseIsNoop() {
        var model = Model.define(t -> (String value) -> value);
        assertThrows(UnsupportedOperationException.class, () -> model.compute((Tensor) null));
        assertThrows(UnsupportedOperationException.class, () -> model.compute((List<Tensor>) null));
        assertThrows(UnsupportedOperationException.class, () -> model.withEngine(null));
        assertThrows(UnsupportedOperationException.class, () -> model.configureTraining(
                (TensorDescriptor) null, null, null));
        assertThrows(UnsupportedOperationException.class, () -> model.train((Tensor) null, null));
        assertThrows(UnsupportedOperationException.class, () -> model.save(null));
        model.close(); model.close();
        assertEquals("retained", model.forward("retained"));
    }
}
