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
import io.github.pho001.synaptik.model.tensor.RecurrentScan;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
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

    @Test void deepNamespaceRoundtripPreservesSourceStateRolesAliasesAndCanonicalBytes() throws Exception {
        String prefix = "a.".repeat(12_000) + "層🦉.";
        Tensor input = prototype(), weight = scalar(-0.0f, true), buffer = scalar(3, false);
        Tensor constant = scalar(7, false);
        var leaves = List.of(parameter(prefix + "weight", weight), parameter(prefix + "alias", weight),
                new Leaf(prefix + "buffer", Role.BUFFER, buffer), new Leaf("constant", Role.CONSTANT, constant));
        Path first = directory.resolve("deep"), second = directory.resolve("deep-again");
        try (var model = Model.graph(List.of(input), List.of(input.mul(weight).add(buffer).add(constant)), leaves)) {
            assertDeepState(model, prefix);
            model.save(first);
        }
        try (var loaded = Model.loadGraph(first)) {
            assertDeepState(loaded, prefix);
            Tensor replacement = prototype();
            Tensor output = loaded.forward(List.of(replacement)).getFirst();
            var outer = output.provenance().orElseThrow().inputs();
            assertEquals(7, number(outer.get(1)));
            var addition = outer.getFirst().provenance().orElseThrow().inputs();
            assertSame(loaded.stateDictionary().entries().getLast().value(), addition.get(1));
            var product = addition.getFirst().provenance().orElseThrow().inputs();
            assertSame(replacement, product.getFirst());
            assertSame(loaded.stateDictionary().entries().getFirst().value(), product.get(1));
            loaded.save(second);
        }
        assertArrayEquals(java.nio.file.Files.readAllBytes(first), java.nio.file.Files.readAllBytes(second));
    }

    private static void assertDeepState(Model<List<Tensor>, List<Tensor>> model, String prefix) {
        var entries = model.stateDictionary().entries();
        assertEquals(List.of(prefix + "weight", prefix + "alias", prefix + "buffer"),
                entries.stream().map(StateEntry::path).toList());
        assertEquals(List.of(StateKind.PARAMETER, StateKind.PARAMETER, StateKind.BUFFER),
                entries.stream().map(StateEntry::kind).toList());
        assertSame(entries.getFirst().value(), entries.get(1).value());
        assertEquals(0x80000000, Float.floatToRawIntBits(number(entries.getFirst().value())));
        assertEquals(3, number(entries.getLast().value()));
    }

    @Test void nestedMutationDiagnosticsIdentifyCapturedModuleAndStatePaths() {
        for (String mutation : List.of("mode", "topology", "wrapper", "tensor", "storage")) {
            Tensor external = scalar(2, true);
            var child = new MutableSourceModule(external);
            try (var model = Model.defineTensor(prototype().descriptor(), topology -> {
                topology.addModule("encoder", child);
                return x -> x.mul(external);
            })) {
                String expected;
                switch (mutation) {
                    case "mode" -> { child.train(); expected = "mode at encoder"; }
                    case "topology" -> {
                        child.attach();
                        expected = "topology at encoder";
                    }
                    case "wrapper" -> {
                        child.addParameter();
                        expected = "parameter wrapper at encoder.extra";
                    }
                    case "tensor" -> {
                        child.weight.replace(scalar(5, true));
                        expected = "Tensor binding at encoder.weight";
                    }
                    default -> {
                        child.weight.value().replaceHostStorage(scalar(5, true).hostStorage().orElseThrow());
                        expected = "storage association at encoder.weight";
                    }
                }
                var failure = assertThrows(IllegalStateException.class, () -> model.forward(prototype()));
                assertTrue(failure.getMessage().contains(expected), failure.getMessage());
                child.eval();
                assertThrows(IllegalStateException.class, () -> model.forward(prototype()));
            }
        }
    }

    @Test void controlNamesRoundtripUnchangedButInvalidationDiagnosticsStaySingleLine() throws Exception {
        String name = "encoder\n\r\t\u0000\u0085\u2028\u2029\"\\end";
        String shown = "encoder" + "?".repeat(9) + "end";
        for (String mutation : List.of("mode", "tensor")) {
            Tensor external = scalar(2, true);
            var child = new MutableSourceModule(external);
            Path saved = directory.resolve("controls-" + mutation);
            try (var model = Model.defineTensor(prototype().descriptor(), topology -> {
                topology.addModule(name, child);
                return x -> x.mul(external);
            })) {
                assertEquals(name + ".weight", model.stateDictionary().entries().getFirst().path());
                model.save(saved);
                try (var loaded = Model.loadGraph(saved)) {
                    assertEquals(name + ".weight", loaded.stateDictionary().entries().getFirst().path());
                    Tensor input = prototype();
                    assertSame(input, loaded.forward(List.of(input)).getFirst()
                            .provenance().orElseThrow().inputs().getFirst());
                    Path again = directory.resolve("controls-again-" + mutation);
                    loaded.save(again);
                    assertArrayEquals(java.nio.file.Files.readAllBytes(saved),
                            java.nio.file.Files.readAllBytes(again));
                }
                String category, location;
                if (mutation.equals("mode")) {
                    child.train();
                    category = "mode";
                    location = shown;
                } else {
                    child.weight.replace(scalar(5, true));
                    category = "Tensor binding";
                    location = shown + ".weight";
                }
                var failure = assertThrows(IllegalStateException.class, () -> model.forward(prototype()));
                assertEquals("owned Model source changed: " + category + " at " + location
                        + "; construct a replacement Model", failure.getMessage());
                child.eval();
                assertThrows(IllegalStateException.class, () -> model.forward(prototype()));
            }
        }
    }

    @Test void deepValidNamespaceInvalidationBoundsModuleAndStatePathsWithoutSplittingSurrogates() {
        String prefix = "a.".repeat(79) + "x🦉." + "b.".repeat(12_000);
        String path = prefix + "weight";
        String shown = "a.".repeat(79) + "x...";
        for (String mutation : List.of("mode", "tensor")) {
            Tensor input = prototype(), weight = scalar(2, true);
            try (var model = Model.graph(List.of(input), List.of(input.mul(weight)),
                    List.of(parameter(path, weight)))) {
                assertEquals(path, model.stateDictionary().entries().getFirst().path());
                assertNotSame(weight, model.stateDictionary().entries().getFirst().value());
                Tensor replacement = prototype();
                assertSame(replacement, model.forward(List.of(replacement)).getFirst()
                        .provenance().orElseThrow().inputs().getFirst());
                String category;
                Module deepest = model;
                while (!deepest.children().isEmpty()) deepest = deepest.children().values().iterator().next();
                if (mutation.equals("mode")) {
                    deepest.train();
                    category = "mode";
                } else {
                    model.parametersRecursively().get(path).replace(scalar(5, true));
                    category = "Tensor binding";
                }
                var failure = assertThrows(IllegalStateException.class,
                        () -> model.forward(List.of(prototype())));
                assertEquals("owned Model source changed: " + category + " at " + shown
                        + "; construct a replacement Model", failure.getMessage());
                deepest.eval();
                assertThrows(IllegalStateException.class, () -> model.forward(List.of(prototype())));
            }
        }
    }

    private static final class MutableSourceModule extends Module {
        final Parameter weight;
        MutableSourceModule(Tensor tensor) { weight = parameter("weight", tensor); }
        void attach() { child("later", new Weight(scalar(1, true))); }
        void addParameter() { parameter("extra", scalar(1, true)); }
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

    /**
     * Checks all fixed signatures through Model.graph/save/loadGraph/cold forward with fresh lengths.
     * Invalid prototype length values are intentionally ignored, never copied or saved.
     * @throws Exception if a supported cold workflow or lifecycle-field inspection fails
     */
    @Test void fixedScansSaveCloseColdLoadAndRebindFreshRuntimeLengthsWithoutEngine() throws Exception {
        for (String family : List.of("rnn", "gru", "lstm")) for (boolean bias : List.of(false, true))
            for (var direction : RecurrentDirection.values()) {
                int gates = family.equals("lstm") ? 4 : family.equals("gru") ? 3 : 1;
                Tensor x = TensorFactory.create(scanDescriptor(DataType.FLOAT32, false, 2, 1, 1));
                Tensor lengths = TensorFactory.fromFlatArray(scanDescriptor(DataType.INT64, false, 1),
                        Optional.empty(), new long[]{Long.MAX_VALUE}); // intentionally not a valid runtime length
                Tensor h = TensorFactory.create(scanDescriptor(DataType.FLOAT32, false, 1, 1));
                Tensor c = TensorFactory.create(h.descriptor());
                Tensor w = TensorFactory.fromFlatArray(scanDescriptor(DataType.FLOAT32, true, gates, 1),
                        Optional.empty(), new float[gates]);
                Tensor u = TensorFactory.fromFlatArray(w.descriptor(), Optional.empty(), new float[gates]);
                Tensor b = TensorFactory.fromFlatArray(scanDescriptor(DataType.FLOAT32, true, gates),
                        Optional.empty(), new float[gates]);
                List<Tensor> inputs = family.equals("lstm") ? List.of(x, lengths, h, c) : List.of(x, lengths, h);
                var leaves = new java.util.ArrayList<>(List.of(parameter("w", w), parameter("u", u)));
                if (bias) leaves.add(parameter("bias", b));
                List<Tensor> outputs;
                if (family.equals("lstm")) {
                    var scan = bias ? RecurrentScan.lstm(x, lengths, h, c, w, u, b, direction)
                            : RecurrentScan.lstm(x, lengths, h, c, w, u, direction);
                    outputs = List.of(scan.finalCell(), scan.outputs(), h);
                } else {
                    var scan = family.equals("rnn")
                            ? (bias ? RecurrentScan.rnn(x, lengths, h, w, u, b, direction) : RecurrentScan.rnn(x, lengths, h, w, u, direction))
                            : (bias ? RecurrentScan.gru(x, lengths, h, w, u, b, direction) : RecurrentScan.gru(x, lengths, h, w, u, direction));
                    outputs = List.of(scan.finalHidden(), scan.outputs(), h);
                }
                Path file = directory.resolve("scan"), resaved = directory.resolve("scan-again");
                try (var model = Model.graph(inputs, outputs, leaves)) {
                    assertNull(field(field(model, "execution"), "engine"));
                    model.save(file);
                    assertNull(field(field(model, "execution"), "engine"));
                }
                lengths.clearHostStorage();
                assertThrows(IllegalArgumentException.class, () -> Model.load(file));
                try (var loaded = Model.loadGraph(file)) {
                    var source = (GraphCheckpoint.Source) field(field(field(loaded, "execution"), "owned"), "source");
                    for (Tensor prototype : source.inputs()) assertTrue(prototype.hostStorage().isEmpty());
                    Tensor freshX = TensorFactory.create(x.descriptor());
                    Tensor freshLengths = TensorFactory.fromFlatArray(lengths.descriptor(), Optional.empty(), new long[]{0});
                    Tensor freshState = TensorFactory.create(h.descriptor());
                    var fresh = family.equals("lstm") ? List.of(freshX, freshLengths, freshState, freshState)
                            : List.of(freshX, freshLengths, freshState);
                    var rebound = loaded.forward(fresh);
                    var producer = rebound.getFirst().provenance().orElseThrow().producer();
                    assertSame(rebound.getFirst(), producer.output(family.equals("lstm") ? 2 : 1));
                    assertSame(rebound.get(1), producer.output(0));
                    assertSame(freshState, rebound.get(2));
                    assertSame(freshLengths, producer.inputs().get(1));
                    assertSame(freshX, producer.inputs().getFirst());
                    if (family.equals("lstm")) assertSame(producer.inputs().get(2), producer.inputs().get(3));
                    assertNull(field(field(loaded, "execution"), "engine"));
                    loaded.save(resaved);
                    assertArrayEquals(java.nio.file.Files.readAllBytes(file), java.nio.file.Files.readAllBytes(resaved));
                }
            }
    }

    @Test void staticLengthGraphAcquisitionAndExplicitForwardStayEngineFree() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long batch : List.of(0L, 1L, 3L)) for (long time : List.of(0L, 3L))
                for (boolean reverse : List.of(false, true)) {
                    Tensor h = TensorFactory.create(scanDescriptor(type, false, batch, 2));
                    Tensor x = TensorFactory.create(scanDescriptor(type, false, batch, 2));
                    Tensor w = TensorFactory.create(scanDescriptor(type, true, 2));
                    Tensor initial = TensorFactory.create(h.descriptor());
                    Tensor sequence = TensorFactory.create(scanDescriptor(type, false, time, batch, 2));
                    Tensor lengths;
                    MemorySegmentStorage control;
                    try (var arena = java.lang.foreign.Arena.ofConfined()) {
                        control = new MemorySegmentStorage(DataType.INT64, batch, arena.allocate(batch * 8, 8));
                        lengths = TensorFactory.create(scanDescriptor(DataType.INT64, false, batch),
                                Optional.empty(), Optional.of(control));
                    }
                    Tensor weight = type == DataType.FLOAT32
                            ? TensorFactory.fromFlatArray(w.descriptor(), Optional.empty(),
                                    new float[]{Float.intBitsToFloat(0xffc00123), -0.0f})
                            : TensorFactory.fromFlatArray(w.descriptor(), Optional.empty(),
                                    new double[]{Double.longBitsToDouble(0xfff8000000000123L), -0.0});
                    byte[] raw = weight.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE);
                    var body = new io.github.pho001.synaptik.model.tensor.ScanBody(List.of(h), List.of(x), List.of(w),
                            List.of(h.add(x).add(w).tanh()), List.of(h));
                    var scan = io.github.pho001.synaptik.model.tensor.Scan.apply(List.of(initial), List.of(sequence),
                            List.of(weight), lengths, 1, List.of(java.util.OptionalInt.of(0)), body, reverse);
                    var roots = new java.util.ArrayList<>(scan.finalCarry());
                    roots.addAll(scan.outputs()); roots.addAll(scan.residuals());
                    try (var model = Model.graph(List.of(sequence, lengths, initial), roots,
                            List.of(parameter("a.weight", weight), parameter("b.weight", weight)))) {
                        assertNull(field(field(model, "execution"), "engine"));
                        var source = (GraphCheckpoint.Source) field(field(field(model, "execution"), "owned"), "source");
                        assertTrue(source.inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
                        Tensor owned = source.leaves().getFirst().tensor();
                        assertNotSame(weight, owned); assertSame(owned, source.leaves().getLast().tensor());
                        assertArrayEquals(raw, owned.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE));
                        Tensor fresh = TensorFactory.create(sequence.descriptor()), freshLengths = TensorFactory.create(lengths.descriptor());
                        Tensor freshInitial = TensorFactory.create(initial.descriptor());
                        var rebound = model.forward(List.of(fresh, freshLengths, freshInitial));
                        var p = rebound.getFirst().provenance().orElseThrow().producer();
                        assertSame(freshInitial, p.inputs().getFirst()); assertSame(fresh, p.inputs().get(1));
                        assertSame(owned, p.inputs().get(2)); assertSame(freshLengths, p.inputs().getLast());
                        assertEquals(scan.finalCarry().getFirst().descriptor(), rebound.getFirst().descriptor());
                        for (int port = 0; port < p.outputCount(); port++) assertSame(p.output(port), rebound.get(port));
                        assertNull(field(field(model, "execution"), "engine"));
                    }
                    assertSame(control, lengths.hostStorage().orElseThrow()); assertFalse(control.isAlive());
                    assertArrayEquals(raw, weight.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE));
                }
    }

    @Test void staticLengthDefinitionAcquiresNamedControlsAndAliasesOnceWithoutEngine() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long time : List.of(0L, 3L)) for (boolean reverse : List.of(false, true)) {
                Tensor h = TensorFactory.create(scanDescriptor(type, false, 1, 2));
                Tensor x = TensorFactory.create(h.descriptor()), w = TensorFactory.create(scanDescriptor(type, true, 2));
                Tensor initial = TensorFactory.allocate(h.descriptor());
                Tensor lengths = TensorFactory.fromFlatArray(scanDescriptor(DataType.INT64, false, 1),
                        Optional.empty(), new long[]{Long.MAX_VALUE});
                Tensor weight = type == DataType.FLOAT32
                        ? TensorFactory.fromFlatArray(w.descriptor(), Optional.empty(), new float[]{-0.0f, 2})
                        : TensorFactory.fromFlatArray(w.descriptor(), Optional.empty(), new double[]{-0.0, 2});
                var body = new io.github.pho001.synaptik.model.tensor.ScanBody(List.of(h), List.of(x), List.of(w),
                        List.of(h.add(x).add(w).tanh()), List.of());
                AtomicInteger definitions = new AtomicInteger(), forwards = new AtomicInteger();
                try (var model = Model.defineTensor(scanDescriptor(type, false, time, 1, 2), topology -> {
                    definitions.incrementAndGet();
                    topology.addModule("a", new Weight(weight)); topology.addModule("b", new Weight(weight));
                    return input -> {
                        forwards.incrementAndGet();
                        return io.github.pho001.synaptik.model.tensor.Scan.apply(List.of(initial), List.of(input),
                                List.of(weight), lengths, 1, List.of(java.util.OptionalInt.of(0)), body, reverse)
                                .finalCarry().getFirst();
                    };
                })) {
                    Tensor fresh = TensorFactory.create(scanDescriptor(type, false, time, 1, 2));
                    Tensor result = model.forward(fresh);
                    var p = result.provenance().orElseThrow().producer();
                    assertSame(fresh, p.inputs().get(1));
                    assertNotSame(initial, p.inputs().getFirst()); assertNotSame(weight, p.inputs().get(2));
                    assertNotSame(lengths, p.inputs().getLast());
                    assertEquals(Long.MAX_VALUE, p.inputs().getLast().hostStorage().orElseThrow().segment()
                            .get(ValueLayout.JAVA_LONG_UNALIGNED, 0));
                    var entries = model.stateDictionary().entries();
                    assertSame(entries.get(0).value(), entries.get(1).value());
                    assertEquals(1, definitions.get()); assertEquals(1, forwards.get());
                    assertNull(field(field(model, "execution"), "engine"));
                }
            }
    }

    /**
     * Supplies exact canonical metadata for stored parameters and external scan prototypes.
     * @param type logical dtype, including INT64 for lengths
     * @param grad gradient eligibility, false for lengths
     * @param shape fully static nonnegative extents
     * @return immutable canonical offset-zero descriptor
     */
    private static TensorDescriptor scanDescriptor(DataType type, boolean grad, long... shape) {
        Shape geometry = Shape.of(shape);
        return new TensorDescriptor(type, geometry, Optional.of(LayoutDescriptor.contiguous(geometry)), grad);
    }
}
