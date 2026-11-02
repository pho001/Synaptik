package io.github.pho001.synaptik.checkpoint.graph;

import static io.github.pho001.synaptik.checkpoint.graph.GraphCheckpointTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.scan.*;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Fixed wire IDs, preallocation bounds and malformed topology tests with explicit encoded fixtures. */
final class GraphArtifactCodecTest {
    @TempDir Path directory;

    @Test void codecLocalAdmissionRejectsNewTopologiesWithoutCompilerOrSizing() throws Exception {
        // Invoke the existing semantic admission predicate, not Compiler or the writer. Without
        // the codec-local topology guard it returns an opcode and this regression must fail.
        var admission = GraphArtifactCodec.class.getDeclaredMethod("opcode", CompiledNode.class);
        admission.setAccessible(true);
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long time : new long[] {0, 1}) for (long batch : new long[] {0, 1})
                for (boolean reverse : List.of(false, true)) for (int topology = 0; topology < 3; topology++) {
                    var raw = sourceOnlyScanFixture(type, time, batch, reverse, topology);
                    Tensor before = prototype(type, Shape.scalar(), false, true);
                    var failure = assertThrows(java.lang.reflect.InvocationTargetException.class,
                            () -> admission.invoke(null, raw.graph.nodes().getFirst()));
                    assertInstanceOf(IllegalArgumentException.class, failure.getCause());
                    assertEquals(before.id().value() + 1, prototype(type, Shape.scalar(), false, true).id().value());
                }
    }

    @Test void rawEncoderRejectsNewTopologiesWithCompleteSourceBoundaries() throws Exception {
        // End-to-end rejection is separate evidence: these graphs have reachable Scan outputs,
        // contiguous input IDs before produced IDs, and the exact matching source boundary.
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long time : new long[] {0, 1}) for (long batch : new long[] {0, 1})
                for (boolean reverse : List.of(false, true)) for (int topology = 0; topology < 3; topology++) {
                    var raw = sourceOnlyScanFixture(type, time, batch, reverse, topology);
                    Tensor before = prototype(type, Shape.scalar(), false, true);
                    assertThrows(IllegalArgumentException.class, () -> GraphArtifactCodec.encode(raw, LIMITS));
                    assertThrows(IllegalArgumentException.class, () -> GraphArtifactCodec.encode(raw, new ArtifactLimits(48, 8, 0, 1, 0, 0)));
                    assertEquals(before.id().value() + 1, prototype(type, Shape.scalar(), false, true).id().value());
                }
    }

    @Test void declaredHistoryWideningDoesNotChangeColdOrWireAdmissionEvenWhenUnused() throws Exception {
        var admission = GraphArtifactCodec.class.getDeclaredMethod("opcode", CompiledNode.class);
        admission.setAccessible(true);
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long time : new long[] {0, 1}) for (long batch : new long[] {0, 1, 2})
                for (boolean reverse : List.of(false, true))
                    for (int[] geometry : List.of(new int[] {2, 0}, new int[] {2, 1},
                            new int[] {3, 0}, new int[] {3, 1}, new int[] {3, 2}, new int[] {4, 3})) {
                        int rank = geometry[0], row = geometry[1];
                        var raw = declaredHistoryFixture(type, time, batch, reverse, rank, row);
                        boolean retained = row == 0 || row == 1 && rank == 2;
                        Tensor before = prototype(type, Shape.scalar(), false, true);
                        if (retained) {
                            new io.github.pho001.synaptik.compiler.GraphCompilationPort.SourceCapture(
                                    raw.graph, raw.source.inputs());
                            assertEquals("graph-model.v5", GraphArtifactCodec.encode(raw, LIMITS).kind());
                        } else {
                            assertThrows(IllegalArgumentException.class, () ->
                                    new io.github.pho001.synaptik.compiler.GraphCompilationPort.SourceCapture(
                                            raw.graph, raw.source.inputs()));
                            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.copy(raw.source));
                            assertThrows(IllegalArgumentException.class, () -> GraphArtifactCodec.encode(raw, LIMITS));
                            assertThrows(IllegalArgumentException.class, () -> GraphArtifactCodec.encode(raw,
                                    new ArtifactLimits(48, 8, 0, 1, 0, 0)));
                            var failure = assertThrows(java.lang.reflect.InvocationTargetException.class,
                                    () -> admission.invoke(null, raw.graph.nodes().getFirst()));
                            assertInstanceOf(IllegalArgumentException.class, failure.getCause());
                        }
                        assertEquals(before.id().value() + 1,
                                prototype(type, Shape.scalar(), false, true).id().value());
                    }
    }

    /**
     * Associates a Model-valid unused history declaration with independently authored raw metadata.
     * The sole root is an identity ROW carry; no ordinary operation can mask a header rejection.
     * @param type ordinary homogeneous floating dtype retained by old cold source
     * @param time static time, including zero
     * @param batch exact static B, including empty and coincident feature extents
     * @param reverse original-time traversal direction
     * @param rank positive step rank
     * @param row valid identified step axis, not inferred from equal extents
     * @return complete raw source association without invoking cold admission
     * @throws ReflectiveOperationException if the existing raw association cannot be constructed
     */
    private static GraphSource declaredHistoryFixture(DataType type, long time, long batch,
            boolean reverse, int rank, int row) throws ReflectiveOperationException {
        long[] extents = new long[rank]; Arrays.fill(extents, 2); extents[row] = batch;
        long[] stacked = new long[rank + 1]; stacked[0] = time;
        System.arraycopy(extents, 0, stacked, 1, rank);
        Tensor h = prototype(type, Shape.of(batch, 2), false, true);
        Tensor step = prototype(type, Shape.of(extents), false, true);
        var template = new io.github.pho001.synaptik.model.tensor.ScanBody(
                List.of(h), List.of(step), List.of(), List.of(h), List.of());
        Tensor initial = prototype(type, h.descriptor().shape(), false, true);
        Tensor sequence = prototype(type, Shape.of(stacked), false, true);
        Tensor lengths = prototype(DataType.INT64, Shape.of(batch), false, true);
        var result = io.github.pho001.synaptik.model.tensor.Scan.apply(List.of(initial), List.of(sequence),
                List.of(), lengths, 1, List.of(OptionalInt.of(row)), template, reverse);
        var producer = result.finalCarry().getFirst().provenance().orElseThrow().producer();
        ValueId localH = new ValueId(0), localStep = new ValueId(1);
        var local = new CompiledGraphModel(List.of(new GraphValue(localH, h.descriptor()),
                new GraphValue(localStep, step.descriptor())), List.of(),
                List.of(localH, localStep), List.of(localH), Map.of());
        var body = new CompiledScanBody(local, List.of(localH), List.of(), List.of(localH), List.of());
        var values = new ArrayList<GraphValue>(); var inputs = new ArrayList<ValueId>();
        for (Tensor actual : producer.inputs()) {
            ValueId id = new ValueId(values.size()); inputs.add(id);
            values.add(new GraphValue(id, actual.descriptor()));
        }
        var outputs = new ArrayList<ValueId>();
        for (TensorDescriptor descriptor : producer.outputDescriptors()) {
            ValueId id = new ValueId(values.size()); outputs.add(id);
            values.add(new GraphValue(id, descriptor));
        }
        var node = new CompiledNode(new NodeId(0), producer.operation(), inputs, outputs, Optional.of(body));
        var graph = new CompiledGraphModel(values, List.of(node), inputs, List.of(outputs.getFirst()),
                Map.of(node.id(), GraphPhase.FORWARD));
        var source = new GraphCheckpoint.Source(producer.inputs(), result.finalCarry(), List.of());
        var constructor = GraphSource.class.getDeclaredConstructors()[0]; constructor.setAccessible(true);
        return (GraphSource) constructor.newInstance(source, graph, List.of(), List.of(), List.of());
    }

    /**
     * Authors canonical raw Scan metadata and its matching Model-valid source without cold capture.
     * @param type homogeneous floating body dtype
     * @param time fixed T, including zero
     * @param batch fixed B, including zero
     * @param reverse original-time traversal direction
     * @param topology zero for carry-free all-valid, one for carry-free length, two for SHARED-only length
     * @return raw encoder association with complete boundaries and a reachable semantic root
     * @throws ReflectiveOperationException if construction of the existing raw association fails
     */
    private static GraphSource sourceOnlyScanFixture(DataType type, long time, long batch,
            boolean reverse, int topology) throws ReflectiveOperationException {
        boolean shared = topology == 2, length = topology != 0;
        Shape shape = shared ? Shape.of(2) : Shape.of(batch, 2);
        Shape sequence = shared ? Shape.of(time, 2) : Shape.of(time, batch, 2);
        java.util.function.BiFunction<DataType, Shape, TensorDescriptor> descriptor = (dtype, s) ->
                new TensorDescriptor(dtype, s, Optional.of(LayoutDescriptor.contiguous(s)), false);
        ValueId x = new ValueId(0), y = new ValueId(1), sum = new ValueId(2);
        var localValues = new ArrayList<GraphValue>(); localValues.add(new GraphValue(x, descriptor.apply(type, shape)));
        var formals = new ArrayList<ValueId>(); formals.add(x);
        var localNodes = new ArrayList<CompiledNode>();
        if (shared) {
            formals.add(y); localValues.add(new GraphValue(y, descriptor.apply(type, shape)));
            localValues.add(new GraphValue(sum, descriptor.apply(type, shape)));
            localNodes.add(new CompiledNode(new NodeId(0), new Operation(
                    io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.ADD,
                    io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE), List.of(x, y), List.of(sum)));
        }
        ValueId root = shared ? sum : x;
        var local = new CompiledGraphModel(localValues, localNodes, formals, shared ? List.of(root, x) : List.of(root),
                shared ? Map.of(new NodeId(0), GraphPhase.FORWARD) : Map.of());
        var body = new CompiledScanBody(local, shared ? List.of(root) : List.of(),
                shared ? List.of() : List.of(root), shared ? List.of(x, root) : List.of(), List.of());
        var values = new ArrayList<GraphValue>();
        var inputs = new ArrayList<ValueId>();
        if (shared) {
            var id = new ValueId(values.size()); inputs.add(id); values.add(new GraphValue(id, descriptor.apply(type, shape)));
        }
        var seq = new ValueId(values.size()); inputs.add(seq); values.add(new GraphValue(seq, descriptor.apply(type, sequence)));
        if (length) {
            var id = new ValueId(values.size()); inputs.add(id);
            values.add(new GraphValue(id, descriptor.apply(DataType.INT64, Shape.of(batch))));
        }
        var out = new ValueId(values.size()); values.add(new GraphValue(out, descriptor.apply(type, shared ? shape : sequence)));
        var outputs = new ArrayList<ValueId>(); outputs.add(out);
        if (shared) for (int r = 0; r < 2; r++) {
            var id = new ValueId(values.size()); outputs.add(id);
            values.add(new GraphValue(id, descriptor.apply(type, sequence)));
        }
        var attrs = new ScanAttrs(time, shared ? 1 : 0, 1, 0, shared ? 0 : 1, shared ? 2 : 0,
                reverse, length, 0, length ? List.of(shared ? OptionalInt.empty() : OptionalInt.of(0)) : List.of());
        var node = new CompiledNode(new NodeId(0), new Operation(ScanKind.SCAN, attrs), inputs, outputs, Optional.of(body));
        var graph = new CompiledGraphModel(values, List.of(node), inputs, List.of(out), Map.of(node.id(), GraphPhase.FORWARD));
        Tensor h = io.github.pho001.synaptik.model.tensor.TensorFactory.create(descriptor.apply(type, shape));
        Tensor step = io.github.pho001.synaptik.model.tensor.TensorFactory.create(descriptor.apply(type, shape));
        var sourceBody = new io.github.pho001.synaptik.model.tensor.ScanBody(shared ? List.of(h) : List.of(), List.of(step),
                List.of(), shared ? List.of(h.add(step)) : List.of(), shared ? List.of() : List.of(step));
        var actuals = inputs.stream().map(id -> io.github.pho001.synaptik.model.tensor.TensorFactory.create(
                values.get(Math.toIntExact(id.value())).descriptor())).toList();
        int carryEnd = shared ? 1 : 0;
        var result = length ? io.github.pho001.synaptik.model.tensor.Scan.apply(actuals.subList(0, carryEnd),
                List.of(actuals.get(carryEnd)), List.of(), actuals.getLast(), 0, attrs.sequenceRowAxes(), sourceBody, reverse)
                : io.github.pho001.synaptik.model.tensor.Scan.apply(List.of(), actuals, List.of(), sourceBody, reverse);
        var source = new GraphCheckpoint.Source(actuals, shared ? result.finalCarry() : result.outputs(), List.of());
        var constructor = GraphSource.class.getDeclaredConstructors()[0]; constructor.setAccessible(true);
        return (GraphSource) constructor.newInstance(source, graph, List.of(), List.of(), List.of());
    }

    @Test
    void directCodecDeadLengthOccurrenceRejectsBeforeStorageEvenWithoutColdPreflight() throws Exception {
        for (boolean symbolic : List.of(false, true)) for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
                for (long time : List.of(0L, 3L)) {
            var captured = GraphSource.capture(scalar());
            var batch = symbolic ? new io.github.pho001.synaptik.model.shape.DynamicDimension("Customers")
                    : new io.github.pho001.synaptik.model.shape.StaticDimension(2);
            var width = new io.github.pho001.synaptik.model.shape.StaticDimension(2);
            Shape row = Shape.ofDimensions(batch, width), seq = Shape.ofDimensions(
                    new io.github.pho001.synaptik.model.shape.StaticDimension(time), batch, width), lengths = Shape.ofDimensions(batch);
            java.util.function.BiFunction<DataType, Shape, TensorDescriptor> descriptor = (dtype, shape) -> new TensorDescriptor(
                    dtype, shape, shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), false);
            var h = descriptor.apply(type, row); var x = descriptor.apply(type, seq); var l = descriptor.apply(DataType.INT64, lengths);
            var local = new CompiledGraphModel(List.of(new GraphValue(new ValueId(0), h), new GraphValue(new ValueId(1), h)),
                    List.of(), List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(0)), Map.of());
            var body = new CompiledScanBody(local, List.of(new ValueId(0)), List.of(), List.of(new ValueId(0)), List.of());
            int base = captured.graph.values().size();
            var inputs = List.of(new ValueId(base), new ValueId(base + 1), new ValueId(base + 2));
            var outputs = List.of(new ValueId(base + 3), new ValueId(base + 4));
            var n = new CompiledNode(new NodeId(1), new Operation(ScanKind.SCAN,
                    new ScanAttrs(time, 1, 1, 0, 0, 1, false, true, 1, List.of(OptionalInt.of(0)))), inputs, outputs, Optional.of(body));
            var values = new ArrayList<>(captured.graph.values());
            values.add(new GraphValue(inputs.get(0), h)); values.add(new GraphValue(inputs.get(1), x));
            values.add(new GraphValue(inputs.get(2), l)); values.add(new GraphValue(outputs.get(0), h));
            values.add(new GraphValue(outputs.get(1), x));
            var nodes = new ArrayList<>(captured.graph.nodes()); nodes.add(n);
            var graphInputs = new ArrayList<>(captured.graph.inputs()); graphInputs.addAll(inputs);
            var phases = new java.util.HashMap<>(captured.graph.nodePhases()); phases.put(n.id(), GraphPhase.FORWARD);
            // Keep the length scan dead and late: codec admission must inspect the stored inventory.
            var graph = new CompiledGraphModel(values, nodes, graphInputs, captured.graph.outputs(), phases);
            var constructor = GraphSource.class.getDeclaredConstructors()[0]; constructor.setAccessible(true);
            var injected = (GraphSource) constructor.newInstance(captured.source, graph, captured.states, captured.roles, captured.registrations);
            captured.states.getFirst().clearHostStorage();
            Tensor probe = io.github.pho001.synaptik.model.tensor.TensorFactory.create(h);
            for (ArtifactLimits limits : List.of(LIMITS, new ArtifactLimits(48, 8, 0, 1, 0, 0))) {
                assertThrows(IllegalArgumentException.class, () -> GraphArtifactCodec.encode(injected, limits));
            }
            assertEquals(probe.id().value() + 1, io.github.pho001.synaptik.model.tensor.TensorFactory.create(h).id().value());
        }
    }

    @Test
    void scalarWireHasExactFixedFieldsAndIndependentGoldenBytes() throws Exception {
        ArtifactEnvelope image = encoded();
        assertEquals("graph-model.v1", image.kind());
        byte[] golden = java.util.HexFormat.of().parseHex(
                "53594e4752463031" + "00000001" + "00000001" + "02010000000000"
                + "00000001" + "02010000000000" + "01" + "0000000000000004"
                + "00000001" + "00000001" + "78" + "00000000"
                + "00000001" + "0101" + "00000000" + "00000001" + "02010000000000"
                + "00000001" + "00000002");
        assertArrayEquals(golden, schema(image));
        assertEquals(85, golden.length);
        assertEquals("state/00000000", image.entries().get(1).key());
        assertArrayEquals(new byte[] {0x3f, (byte) 0x80, 0, 0}, payload(image));
        var source = GraphArtifactCodec.decode(image(golden, new byte[] {0x3f, (byte) 0x80, 0, 0}), LIMITS);
        assertEquals(GraphCheckpoint.Role.PARAMETER, source.leaves().getFirst().role());
        assertArrayEquals(golden, schema(GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS)));
    }

    @Test
    void activationGoldenOpcodesAndExactVersionTuplesAreClosed() throws Exception {
        Tensor input = prototype(DataType.FLOAT32, Shape.scalar(), true, false);
        List<Tensor> activations = List.of(input.relu(), input.tanh(), input.sigmoid());
        for (int i = 0; i < activations.size(); i++) {
            byte[] golden = java.util.HexFormat.of().parseHex(
                    "53594e4752463032" + "00000002" + "00000001" + "02010000000000"
                    + "00000000" + "00000000" + "00000001" + "01" + "0" + (i + 6)
                    + "00000000" + "02010000000000" + "00000001" + "00000001");
            var source = new GraphCheckpoint.Source(List.of(input), List.of(activations.get(i)), List.of());
            var encoded = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
            assertEquals("graph-model.v2", encoded.kind());
            assertArrayEquals(golden, schema(encoded));
            var loaded = GraphArtifactCodec.decode(versioned("graph-model.v2", golden), LIMITS);
            assertEquals(activations.get(i).provenance().orElseThrow().operation(),
                    loaded.outputs().getFirst().provenance().orElseThrow().operation());
            assertArrayEquals(golden, schema(GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS)));
            // New opcodes are never legal v1, even with an otherwise matching v1 header.
            byte[] v1 = golden.clone(); v1[7] = '1'; ByteBuffer.wrap(v1).putInt(8, 1);
            assertThrows(ArtifactFormatException.class,
                    () -> GraphArtifactCodec.decode(versioned("graph-model.v1", v1), LIMITS));
            assertThrows(ArtifactFormatException.class,
                    () -> GraphArtifactCodec.decode(versioned("graph-model.v2", v1), LIMITS));
            for (String kind : List.of("graph-model.v1", "graph-model.v3"))
                assertThrows(ArtifactFormatException.class,
                        () -> GraphArtifactCodec.decode(versioned(kind, golden), LIMITS));
            for (int position : List.of(7, 8, 16, 17, 22, 35, 36, 37, 41, 42, 47, 52)) {
                byte[] bad = golden.clone(); bad[position] = (byte) 0xff;
                assertThrows(ArtifactFormatException.class,
                        () -> GraphArtifactCodec.decode(versioned("graph-model.v2", bad), LIMITS));
            }
            for (int length = 0; length < golden.length; length++) {
                byte[] truncated = Arrays.copyOf(golden, length);
                assertThrows(ArtifactFormatException.class,
                        () -> GraphArtifactCodec.decode(versioned("graph-model.v2", truncated), LIMITS));
            }
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(
                    versioned("graph-model.v2", Arrays.copyOf(golden, golden.length + 1)), LIMITS));
            for (int type : List.of(4, 5, 6)) {
                byte[] bad = golden.clone(); bad[16] = (byte) type; bad[41] = (byte) type;
                bad[17] = 0; bad[42] = 0;
                assertThrows(ArtifactFormatException.class,
                        () -> GraphArtifactCodec.decode(versioned("graph-model.v2", bad), LIMITS));
            }
            // A five-kind-only v2 is valid input, but saves using the smallest sufficient version.
            byte[] neg = golden.clone(); neg[36] = 3;
            var downgraded = GraphArtifactCodec.encode(GraphSource.capture(GraphArtifactCodec.decode(
                    versioned("graph-model.v2", neg), LIMITS)), LIMITS);
            assertEquals("graph-model.v1", downgraded.kind());
            assertEquals('1', schema(downgraded)[7]);
            assertEquals(1, ByteBuffer.wrap(schema(downgraded)).getInt(8));
        }
    }

    @Test
    void versionTwoAllSixExactLimitsAndOneBelowPreservePriorTarget() throws Exception {
        var base = scalar();
        var source = new GraphCheckpoint.Source(base.inputs(), List.of(base.outputs().getFirst().tanh()), base.leaves());
        var exact = new ArtifactLimits(280, 130, 2, 14, 98, 102);
        var image = GraphArtifactCodec.encode(GraphSource.capture(source), exact);
        Path target = directory.resolve("v2-exact");
        GraphCheckpoint.save(target, source, exact);
        byte[] before = Files.readAllBytes(target);
        assertEquals(280, before.length);
        assertEquals(1, GraphCheckpoint.load(target, exact).leaves().size());
        for (var below : List.of(new ArtifactLimits(279, 130, 2, 14, 98, 102),
                new ArtifactLimits(280, 129, 2, 14, 98, 102), new ArtifactLimits(280, 130, 1, 14, 98, 102),
                new ArtifactLimits(280, 130, 2, 13, 98, 102), new ArtifactLimits(280, 130, 2, 14, 97, 102),
                new ArtifactLimits(280, 130, 2, 14, 98, 101))) {
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.save(target, source, below));
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image, below));
            assertThrows(ArtifactFormatException.class, () -> GraphCheckpoint.load(target, below));
            assertArrayEquals(before, Files.readAllBytes(target));
        }
        source.leaves().getFirst().tensor().clearHostStorage();
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.save(target, source,
                new ArtifactLimits(279, 130, 2, 14, 98, 102)));
    }

    @Test
    void encoderIndependentlyRejectsUnknownKindBeforeStorageAdmission() throws Exception {
        var captured = GraphSource.capture(scalar());
        var old = captured.graph.nodes().getFirst();
        var node = new io.github.pho001.synaptik.model.graph.CompiledNode(old.id(),
                new io.github.pho001.synaptik.model.operation.Operation(
                        io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind.ABS,
                        io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE),
                List.of(old.inputs().getFirst()), old.outputs());
        var graph = new io.github.pho001.synaptik.model.graph.CompiledGraphModel(captured.graph.values(),
                List.of(node), captured.graph.inputs(), captured.graph.outputs(), captured.graph.nodePhases());
        var constructor = GraphSource.class.getDeclaredConstructors()[0]; constructor.setAccessible(true);
        var injected = (GraphSource) constructor.newInstance(captured.source, graph, captured.states,
                captured.roles, captured.registrations);
        captured.states.getFirst().clearHostStorage();
        assertThrows(IllegalArgumentException.class, () -> GraphArtifactCodec.encode(injected, LIMITS));
    }

    /**
     * Wraps independent zero-state schema bytes without consulting the production encoder.
     * @param kind exact or deliberately malformed graph artifact kind
     * @param bytes schema fixture in big-endian wire order
     * @return immutable envelope owning its schema snapshot
     */
    private static ArtifactEnvelope versioned(String kind, byte[] bytes) {
        return new ArtifactEnvelope(kind, List.of(new ArtifactEnvelope.Entry("schema", bytes)));
    }

    @Test
    void allSixLimitsAcceptExactAndRejectOneBelowBeforeHostAdmissionOrPublication() throws Exception {
        var exact = new ArtifactLimits(267, 130, 2, 14, 85, 89);
        ArtifactEnvelope image = GraphArtifactCodec.encode(GraphSource.capture(scalar()), exact);
        Path file = directory.resolve("exact");
        GraphCheckpoint.save(file, scalar(), exact);
        assertEquals(267, Files.size(file));
        assertEquals(1, GraphCheckpoint.load(file, exact).leaves().size());
        var below = List.of(new ArtifactLimits(266, 130, 2, 14, 85, 89),
                new ArtifactLimits(267, 129, 2, 14, 85, 89),
                new ArtifactLimits(267, 130, 1, 14, 85, 89),
                new ArtifactLimits(267, 130, 2, 13, 85, 89),
                new ArtifactLimits(267, 130, 2, 14, 84, 89),
                new ArtifactLimits(267, 130, 2, 14, 85, 88));
        for (ArtifactLimits limit : below) {
            var source = scalar(); source.leaves().getFirst().tensor().clearHostStorage();
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.save(file, source, limit));
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image, limit));
            assertThrows(ArtifactFormatException.class, () -> GraphCheckpoint.load(file, limit));
            assertEquals(267, Files.size(file));
        }
    }

    @Test
    void malformedIdsFlagsCountsRanksLengthsReferencesAndUtf8FailWithCauses() throws Exception {
        byte[] original = schema(encoded());
        for (int position : List.of(16, 27, 70, 34, 60, 61)) {
            byte[] bytes = original.clone(); bytes[position] = (byte) 0xff;
            invalid(bytes);
        }
        for (int position : List.of(17, 22, 28, 33, 71, 76)) {
            byte[] bytes = original.clone(); bytes[position] = 2;
            invalid(bytes);
        }
        for (int position : List.of(12, 18, 23, 29, 43, 47, 52, 56, 62, 66, 72, 77, 81)) {
            for (int value : List.of(-1, Integer.MAX_VALUE)) {
                byte[] bytes = original.clone(); ByteBuffer.wrap(bytes).putInt(position, value);
                invalid(bytes);
            }
        }
        for (int value : List.of(0, 2, 100)) {
            byte[] bytes = original.clone(); ByteBuffer.wrap(bytes).putInt(8, value); invalid(bytes);
        }
        byte[] badMagic = original.clone(); badMagic[0] = 0; invalid(badMagic);
        byte[] badText = original.clone(); badText[51] = (byte) 0x80; invalid(badText);
        for (char bad : new char[] {'.', ' '}) {
            byte[] badName = original.clone(); badName[51] = (byte) bad; invalid(badName);
        }
        for (long length : new long[] {-1, 0, 3, 5, Long.MAX_VALUE}) {
            byte[] bytes = original.clone(); ByteBuffer.wrap(bytes).putLong(35, length); invalid(bytes);
        }
        byte[] forward = original.clone(); ByteBuffer.wrap(forward).putInt(62, 2); invalid(forward);
        byte[] dead = original.clone(); ByteBuffer.wrap(dead).putInt(81, 0); invalid(dead);
        byte[] incompatible = original.clone(); incompatible[70] = 1; invalid(incompatible);
        byte[] untrainable = original.clone(); untrainable[28] = 0; invalid(untrainable);
    }

    @Test
    void everyTruncationAndTrailingSchemaByteRejectsWithoutPartialResults() throws Exception {
        byte[] original = schema(encoded());
        for (int length = 0; length < original.length; length++) invalid(Arrays.copyOf(original, length));
        invalid(Arrays.copyOf(original, original.length + 1));
        byte[] duplicateOutput = Arrays.copyOf(original, original.length + 4);
        ByteBuffer.wrap(duplicateOutput).putInt(77, 2).putInt(85, 2);
        invalid(duplicateOutput);
        byte[] emptyOutput = Arrays.copyOf(original, 81); ByteBuffer.wrap(emptyOutput).putInt(77, 0);
        invalid(emptyOutput);
    }

    @Test
    void completeKeysPayloadLengthsAndBoolBytesAreValidatedIndependently() throws Exception {
        ArtifactEnvelope good = encoded();
        for (ArtifactEnvelope image : List.of(
                new ArtifactEnvelope("model.v1", good.entries()),
                new ArtifactEnvelope("graph-model.v1", List.of()),
                new ArtifactEnvelope("graph-model.v1", List.of(new ArtifactEnvelope.Entry("other", schema(good)))),
                new ArtifactEnvelope("graph-model.v1", List.of(good.entries().getFirst())),
                image(schema(good), new byte[3]), image(schema(good), new byte[5]),
                new ArtifactEnvelope("graph-model.v1", List.of(good.entries().getFirst(),
                        new ArtifactEnvelope.Entry("state/00000001", payload(good)))),
                new ArtifactEnvelope("graph-model.v1", List.of(good.entries().getFirst(), good.entries().get(1),
                        new ArtifactEnvelope.Entry("unused", new byte[0])))))
            assertNotNull(assertThrows(ArtifactFormatException.class,
                    () -> GraphArtifactCodec.decode(image, LIMITS)).getCause());
        Tensor bool = tensor(DataType.BOOL, Shape.of(1), false, false, new byte[] {1});
        var boolImage = GraphArtifactCodec.encode(GraphSource.capture(new GraphCheckpoint.Source(List.of(),
                List.of(bool), List.of(leaf("flag", GraphCheckpoint.Role.CONSTANT, bool)))), LIMITS);
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(schema(boolImage), new byte[] {2}), LIMITS));
    }

    @Test
    void aliasesMustBeCanonicalReferencedAndPreserveEqualIndependentIdentities() throws Exception {
        var source = scalar(); Tensor state = source.leaves().getFirst().tensor();
        var aliased = new GraphCheckpoint.Source(source.inputs(), source.outputs(), List.of(
                leaf("x", GraphCheckpoint.Role.PARAMETER, state), leaf("y", GraphCheckpoint.Role.PARAMETER, state)));
        var good = GraphArtifactCodec.encode(GraphSource.capture(aliased), LIMITS);
        var loaded = GraphArtifactCodec.decode(good, LIMITS);
        assertSame(loaded.leaves().get(0).tensor(), loaded.leaves().get(1).tensor());
        byte[] bytes = schema(good);
        // First registration must introduce state zero; second registration cannot introduce state one.
        byte[] first = bytes.clone(); ByteBuffer.wrap(first).putInt(52, 1); invalid(first);
        byte[] second = bytes.clone(); ByteBuffer.wrap(second).putInt(61, 1); invalid(second);
        byte[] duplicateName = bytes.clone(); duplicateName[60] = 'x'; invalid(duplicateName);
        Tensor other = tensor(DataType.FLOAT32, Shape.of(), true, false, payload(encoded()));
        var independent = new GraphCheckpoint.Source(List.of(), List.of(state.add(other)), List.of(
                leaf("x", GraphCheckpoint.Role.PARAMETER, state), leaf("y", GraphCheckpoint.Role.PARAMETER, other)));
        var two = GraphArtifactCodec.encode(GraphSource.capture(independent), LIMITS);
        var twoLoaded = GraphArtifactCodec.decode(two, LIMITS);
        assertNotSame(twoLoaded.leaves().get(0).tensor(), twoLoaded.leaves().get(1).tensor());
        byte[] swapped = schema(two);
        // Second state's descriptor/role/length is 16 bytes; first registration index shifts by 16.
        ByteBuffer.wrap(swapped).putInt(68, 1);
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(
                new ArtifactEnvelope(two.kind(), List.of(new ArtifactEnvelope.Entry("schema", swapped),
                        two.entries().get(1), two.entries().get(2))), LIMITS));
    }

    @Test
    void topologyOrderDeadNodesAndPermuteAttributesFailBeforePublication() throws Exception {
        Tensor input = prototype(DataType.FLOAT32, Shape.of(), true, false);
        // Two independent occurrences; reversing roots without changing stored node order is noncanonical.
        var source = new GraphCheckpoint.Source(List.of(input), List.of(input.neg(), input.mul(input)), List.of());
        var good = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
        byte[] swapped = schema(good);
        int tail = swapped.length - 8;
        ByteBuffer buffer = ByteBuffer.wrap(swapped);
        int first = buffer.getInt(tail), second = buffer.getInt(tail + 4);
        buffer.putInt(tail, second).putInt(tail + 4, first);
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(imageNoState(swapped), LIMITS));
        Tensor matrix = prototype(DataType.FLOAT32, Shape.of(2, 3), true, false);
        var transpose = new GraphCheckpoint.Source(List.of(matrix), List.of(matrix.permute(1, 0)), List.of());
        var transposed = GraphArtifactCodec.encode(GraphSource.capture(transpose), LIMITS);
        byte[] schema = schema(transposed);
        // Header + input descriptor (23 bytes) + three collection counts = node phase at 51.
        assertEquals(5, Byte.toUnsignedInt(schema[52]));
        for (int offset : List.of(57, 61, 65)) {
            byte[] bad = schema.clone(); ByteBuffer.wrap(bad).putInt(offset, 9);
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(imageNoState(bad), LIMITS));
        }
    }

    @Test
    void ranksBeyondBackendCeilingsAndGeometryOverflowHaveCorrectOwnership() throws Exception {
        long[] extents = new long[20]; Arrays.fill(extents, 1);
        Tensor input = prototype(DataType.FLOAT32, Shape.of(extents), true, false);
        var source = new GraphCheckpoint.Source(List.of(input), List.of(input.neg()), List.of());
        var good = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
        assertEquals(20, GraphArtifactCodec.decode(good, LIMITS).inputs().getFirst().descriptor().shape().rank());
        byte[] bad = schema(good); ByteBuffer.wrap(bad).putLong(22, Long.MAX_VALUE).putLong(30, 2);
        var failure = assertThrows(ArtifactFormatException.class,
                () -> GraphArtifactCodec.decode(imageNoState(bad), LIMITS));
        assertInstanceOf(ArithmeticException.class, failure.getCause());
    }

    /**
     * Creates the scalar golden fixture with unresolved layouts and one FLOAT32 parameter.
     * @return ordinary input-plus-weight ADD source
     */
    private static GraphCheckpoint.Source scalar() {
        Tensor input = prototype(DataType.FLOAT32, Shape.of(), true, false);
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(), true, false, new byte[] {0x3f, (byte) 0x80, 0, 0});
        return new GraphCheckpoint.Source(List.of(input), List.of(input.add(weight)),
                List.of(leaf("x", GraphCheckpoint.Role.PARAMETER, weight)));
    }

    /** @return independently validated encoded scalar fixture */
    private static ArtifactEnvelope encoded() { return GraphArtifactCodec.encode(GraphSource.capture(scalar()), LIMITS); }

    /**
     * @param image immutable test image
     * @return detached schema bytes for mutation tests
     */
    private static byte[] schema(ArtifactEnvelope image) {
        byte[] result = new byte[image.entries().getFirst().size()]; image.entries().getFirst().payload().get(result); return result;
    }

    /**
     * @param image immutable one-state fixture
     * @return detached state bytes
     */
    private static byte[] payload(ArtifactEnvelope image) {
        byte[] result = new byte[image.entries().get(1).size()]; image.entries().get(1).payload().get(result); return result;
    }

    /**
     * @param schema mutated scalar schema
     * @throws Exception if fixture construction unexpectedly fails
     */
    private static void invalid(byte[] schema) throws Exception {
        var failure = assertThrows(ArtifactFormatException.class,
                () -> GraphArtifactCodec.decode(image(schema, payload(encoded())), LIMITS));
        assertNotNull(failure.getCause());
    }

    /**
     * @param schema caller-owned raw schema fixture
     * @param payload caller-owned exact state bits
     * @return immutable image with one fixed state key
     */
    private static ArtifactEnvelope image(byte[] schema, byte[] payload) {
        return new ArtifactEnvelope("graph-model.v1", List.of(new ArtifactEnvelope.Entry("schema", schema),
                new ArtifactEnvelope.Entry("state/00000000", payload)));
    }

    /**
     * @param schema caller-owned raw zero-state schema
     * @return immutable image without state payloads
     */
    private static ArtifactEnvelope imageNoState(byte[] schema) {
        return new ArtifactEnvelope("graph-model.v1", List.of(new ArtifactEnvelope.Entry("schema", schema)));
    }
}
