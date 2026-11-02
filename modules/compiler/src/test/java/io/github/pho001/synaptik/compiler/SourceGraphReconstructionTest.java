package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledScanBody;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class SourceGraphReconstructionTest {
    @Test
    void pool3dPreservesEveryTypedGeometryFieldWithoutWideningOtherPoolingFamilies() {
        var maximum = new MaxPool3dAttrs(2, 3, 2, 1, 2, 1, 1, 0, 2, 2, 1, 2, true);
        var average = new AveragePool3dAttrs(3, 2, 2, 2, 1, 3, 0, 2, 1, 1, 2, 1, false);
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (boolean gradient : List.of(false, true)) {
                Tensor volume = leaf(type, Shape.of(2, 1, 5, 6, 7), gradient);
                var outputs = List.of(volume.maxPool3d(maximum), volume.averagePool3d(average));
                assertRoundTrip(outputs);
                var captured = GraphCompilationPort.captureSource(outputs);
                var substitutions = List.of(volume.abs());
                var rebuilt = GraphCompilationPort.reconstructSource(captured.graph(), substitutions);
                assertEquals(maximum, rebuilt.getFirst().provenance().orElseThrow().operation().attrs());
                assertEquals(average, rebuilt.getLast().provenance().orElseThrow().operation().attrs());
                assertSame(substitutions.getFirst(), rebuilt.getFirst().provenance().orElseThrow().inputs().getFirst());
                assertEquals(gradient, rebuilt.getLast().descriptor().requiresGrad());
            }
            Tensor image = formal(type, 1, 1, 4, 4);
            Tensor unsupported = image.maxPool2d(new MaxPool2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false));
            assertColdFailureWithoutIds(() -> GraphCompilationPort.captureSource(List.of(unsupported)));
        }
    }

    @Test
    void pool3dRoundTripsInsideAllValidAndLengthBodiesWithExistingRowRoles() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean reverse : List.of(false, true)) for (long time : List.of(0L, 2L)) {
                Tensor carry = formal(type, 2, 1, 1, 1, 1);
                Tensor step = formal(type, 2, 1, 2, 2, 2);
                Tensor maximum = step.maxPool3d(new MaxPool3dAttrs(2, 2, 2, 2, 2, 2,
                        0, 0, 0, 1, 1, 1, false));
                Tensor average = step.averagePool3d(new AveragePool3dAttrs(2, 2, 2, 2, 2, 2,
                        0, 0, 0, 1, 1, 1, false));
                var body = new ScanBody(List.of(carry), List.of(step), List.of(),
                        List.of(carry.add(average)), List.of(maximum, average, maximum));
                assertColdBodyRoundTrip(body, time, reverse);
                Tensor initial = TensorFactory.create(carry.descriptor());
                Tensor sequence = formal(type, time, 2, 1, 2, 2, 2);
                Tensor lengths = formal(DataType.INT64, 2);
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths,
                        1, List.of(java.util.OptionalInt.of(0)), body, reverse);
                var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
                assertRoundTrip(roots);
                var captured = GraphCompilationPort.captureSource(roots);
                var local = captured.graph().nodes().getFirst().scanBody().orElseThrow();
                assertEquals(local.emissions().getFirst(), local.emissions().getLast());
                assertEquals(1, ((ScanAttrs) captured.graph().nodes().getFirst().operation().attrs()).rowCarryCount());
            }
    }

    @Test
    void metadataOnlyColdPreflightRejectsLatePoolDescriptorWithoutAllocatingAnyTensor() {
        Tensor volume = formal(DataType.FLOAT32, 1, 1, 3, 3, 3);
        Tensor maximum = volume.maxPool3d(new MaxPool3dAttrs(2, 2, 2, 1, 1, 1,
                0, 0, 0, 1, 1, 1, false));
        Tensor average = volume.averagePool3d(new AveragePool3dAttrs(2, 2, 2, 1, 1, 1,
                0, 0, 0, 1, 1, 1, false));
        var graph = GraphCompilationPort.captureSource(List.of(maximum, average)).graph();
        Tensor marker = formal(DataType.BOOL);
        GraphCompilationPort.validateSource(graph);
        assertEquals(marker.id().value() + 1, formal(DataType.BOOL).id().value());
        var values = new ArrayList<>(graph.values());
        GraphValue last = values.getLast();
        values.set(values.size() - 1, new GraphValue(last.id(),
                descriptor(DataType.FLOAT32, Shape.of(1, 1, 3, 2, 2), false)));
        var bad = new CompiledGraphModel(values, graph.nodes(), graph.inputs(), graph.outputs(), graph.nodePhases());
        assertColdFailureWithoutIds(() -> GraphCompilationPort.validateSource(bad));
        var unsupported = GraphCapture.capture(List.of(volume.abs()));
        assertColdFailureWithoutIds(() -> GraphCompilationPort.validateSource(unsupported));
        assertThrows(NullPointerException.class, () -> GraphCompilationPort.validateSource(null));
    }

    @Test
    void generalBodiesRoundTripTheWholeSelectedArithmeticReductionAndLayoutInventory() {
        for (DataType type : DataType.values()) {
            Tensor h = formal(type, 2, 3), x = formal(type, 2, 3), w = formal(type, 3);
            Tensor scalar = formal(type), empty = formal(type, 0, 3);
            var emissions = new ArrayList<>(List.of(h.reshape(Shape.of(3, 2)),
                    h.expandDims(0).squeeze(0), h.expandDims(2).squeeze(2), h.permute(1, 0),
                    h.contiguous(), h.cast(type), scalar.expand(Shape.of(2, 3)),
                    scalar.reshape(Shape.scalar()), empty.reshape(Shape.of(3, 0))));
            // Body-only mappings must remain unavailable at the flat boundary.
            for (Tensor bodyOnly : List.of(h.expandDims(0), h.contiguous(), h.cast(type),
                    empty.reshape(Shape.of(3, 0)), scalar.expand(Shape.of(2, 3)))) {
                assertThrows(IllegalArgumentException.class,
                        () -> GraphCompilationPort.captureSource(List.of(bodyOnly)));
            }
            if (type != DataType.BOOL) {
                ScalarValue exact = exactScalar(type);
                emissions.addAll(List.of(h.add(w), h.sub(w), h.mul(w), h.add(exact), h.sub(exact),
                        h.mul(exact), h.sum(), h.sum(0, false), h.sum(1, true),
                        h.sum(new int[] {1, 0}, false), h.sum(new int[] {1, 0}, true),
                        h.sum(new int[] {}, false), h.sum(new int[] {}, true),
                        h.sumToShape(Shape.of(1, 3)), empty.sum(0), scalar.sum()));
                if (type.isFloating()) emissions.addAll(List.of(h.div(w), h.div(exact), h.neg(),
                        h.relu(), h.tanh(), h.sigmoid()));
                for (Tensor bodyOnly : List.of(h.add(exact), h.sum(), h.sumToShape(Shape.of(1, 3)))) {
                    assertThrows(IllegalArgumentException.class,
                            () -> GraphCompilationPort.captureSource(List.of(bodyOnly)));
                }
                if (type.isIntegral()) assertThrows(IllegalArgumentException.class,
                        () -> GraphCompilationPort.captureSource(List.of(h.sub(w))));
            }
            var body = new ScanBody(List.of(h), List.of(x), List.of(w, scalar, empty), List.of(h), emissions);
            for (boolean reverse : List.of(false, true)) for (long time : List.of(0L, 3L)) {
                assertColdBodyRoundTrip(body, time, reverse);
            }
        }
    }

    @Test
    void generalBodiesReuseNarrowIndexCnnAndMatmulConstructorsWithoutBackendRestrictions() {
        for (DataType type : DataType.values()) {
            Tensor h = formal(type), x = formal(type), matrix = formal(type, 2, 3);
            Tensor cube = formal(type, 1, 2, 3), indices = formal(DataType.INT64, 1);
            var invariants = new ArrayList<>(List.of(matrix, cube, indices));
            var emissions = new ArrayList<>(List.of(cube.select(0, 0), matrix.gather(indices, 0),
                    matrix.sliceByLength(new long[] {1}, new long[] {2}, new int[] {1}, new long[] {1}),
                    Tensor.concat(0, cube, cube), Tensor.concat(0, cube, cube).select(0, 1),
                    Tensor.concat(0, cube, cube).reshape(Shape.of(4, 3))));
            if (type != DataType.BOOL) {
                Tensor weights = formal(type, 3, 2), vector = formal(type, 3);
                Tensor batch = formal(type, 5, 3, 2);
                invariants.addAll(List.of(weights, vector, batch));
                emissions.addAll(List.of(matrix.matmul(weights), vector.matmul(vector),
                        vector.matmul(weights), matrix.matmul(vector), matrix.matmul(batch),
                        matrix.matmul(weights).reshape(Shape.of(4)),
                        matrix.matmul(weights).sliceByLength(new long[] {0}, new long[] {1},
                                new int[] {1}, new long[] {1})));
            }
            if (type.isFloating()) {
                Tensor image = formal(type, 1, 1, 4, 4), kernel = formal(type, 1, 1, 2, 2), bias = formal(type, 1);
                Tensor volume = formal(type, 1, 1, 3, 3, 3), kernel3 = formal(type, 1, 1, 2, 2, 2);
                invariants.addAll(List.of(image, kernel, bias, volume, kernel3));
                emissions.addAll(List.of(image.conv2d(kernel, bias, Conv2dAttrs.defaults()),
                        volume.conv3d(kernel3, bias, Conv3dAttrs.defaults()),
                        image.maxPool2d(new MaxPool2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false)),
                        volume.maxPool3d(new MaxPool3dAttrs(2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, true)),
                        volume.averagePool3d(new AveragePool3dAttrs(2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, false))));
            }
            assertColdBodyRoundTrip(new ScanBody(List.of(h), List.of(x), invariants, List.of(h), emissions), 2, true);
        }
        for (List<DataType> types : List.of(List.of(DataType.INT32, DataType.INT64),
                List.of(DataType.FLOAT32, DataType.FLOAT64), List.of(DataType.BFLOAT16, DataType.FLOAT16))) {
            Tensor h = formal(types.getFirst()), x = formal(types.getFirst());
            Tensor a = formal(types.getFirst(), 2, 3), b = formal(types.getLast(), 3, 4);
            assertColdBodyRoundTrip(new ScanBody(List.of(h), List.of(x), List.of(a, b),
                    List.of(h), List.of(a.matmul(b), a.sum().add(b.sum()))), 0, false);
        }
    }

    @Test
    void generalScanRebuildsIndependentScopesAllCanonicalPortsAndRepeatedRolesActuals() {
        Tensor h = formal(DataType.FLOAT32, 2), h2 = formal(DataType.FLOAT32, 2);
        Tensor x = formal(DataType.FLOAT32, 2), x2 = formal(DataType.FLOAT32, 2);
        Tensor w = formal(DataType.FLOAT32, 2), unused = formal(DataType.FLOAT32, 2);
        Tensor next = h.add(x).add(w).tanh();
        var template = new ScanBody(List.of(h, h2), List.of(x, x2), List.of(w, unused),
                List.of(next, next), List.of(next, h, next));
        Tensor initial = formal(DataType.FLOAT32, 2), seq = formal(DataType.FLOAT32, 5, 2);
        Tensor constant = formal(DataType.FLOAT32, 2);
        constant.replaceHostStorage(new io.github.pho001.synaptik.model.storage.MemorySegmentStorage(
                DataType.FLOAT32, 2, java.lang.foreign.MemorySegment.ofArray(new float[] {1, -0.0f})));
        // Stored constants enter only through explicit invariant positions.
        var a = Scan.apply(List.of(initial, initial), List.of(seq, seq), List.of(constant, initial), template, false);
        var b = Scan.apply(List.of(initial, initial), List.of(seq, seq), List.of(constant, initial), template, true);
        var roots = new ArrayList<Tensor>();
        roots.add(a.residuals().getLast().neg());
        roots.add(a.finalCarry().get(1));
        roots.addAll(a.outputs());
        roots.add(a.residuals().getLast());
        roots.add(b.finalCarry().getFirst());
        var source = GraphCompilationPort.captureSource(roots);
        var rebuilt = GraphCompilationPort.reconstructSource(source.graph(), source.leaves());
        assertEquals(source.graph(), GraphCompilationPort.captureSource(rebuilt).graph());
        TensorProducer pa = rebuilt.get(1).provenance().orElseThrow().producer();
        TensorProducer pb = rebuilt.getLast().provenance().orElseThrow().producer();
        assertNotSame(pa, pb);
        assertSame(pa.output(1), rebuilt.get(1));
        assertSame(pa.output(2), rebuilt.get(2));
        assertSame(pa.output(3), rebuilt.get(3));
        assertSame(pa.output(4), rebuilt.get(4));
        assertSame(rebuilt.get(5), rebuilt.getFirst().provenance().orElseThrow().inputs().getFirst());
        assertSame(pa.inputs().get(0), pa.inputs().get(1));
        assertSame(pa.inputs().get(2), pa.inputs().get(3));
        assertSame(initial, pa.inputs().get(5));
        assertNotSame(template, pa.scanBody().orElseThrow());
        assertNotSame(pa.scanBody().orElseThrow(), pb.scanBody().orElseThrow());
        var newFormals = ScanBodyInference.formals(pa.scanBody().orElseThrow());
        assertEquals(6, newFormals.size());
        assertEquals(6, newFormals.stream().distinct().count());
        for (Tensor fresh : newFormals) {
            assertFalse(ScanBodyInference.formals(template).contains(fresh));
            assertFalse(pa.inputs().contains(fresh));
            assertFalse(ScanBodyInference.formals(pb.scanBody().orElseThrow()).contains(fresh));
            assertTrue(fresh.hostStorage().isEmpty()); assertTrue(fresh.provenance().isEmpty());
        }
        var newBody = pa.scanBody().orElseThrow();
        assertSame(newBody.nextCarry().get(0), newBody.nextCarry().get(1));
        assertSame(newBody.nextCarry().getFirst(), newBody.emissions().getFirst());
        for (CompiledNode scan : source.graph().nodes().stream().filter(n -> n.scanBody().isPresent()).toList()) {
            var body = scan.scanBody().orElseThrow();
            var ordered = new java.util.LinkedHashSet<>(body.nextCarry());
            ordered.addAll(body.emissions()); ordered.addAll(body.residuals());
            assertEquals(List.copyOf(ordered), body.graph().outputs());
            assertTrue(body.demandedOutputIndices().isEmpty());
        }
        // Explicit substitutions can be produced, repeated and cold-unsupported upstream.
        var substitutions = source.leaves().stream().map(t -> t.abs().contiguous()).toList();
        var opaque = GraphCompilationPort.reconstructSource(source.graph(), substitutions);
        var opaqueProducer = opaque.get(1).provenance().orElseThrow().producer();
        assertSame(substitutions.get(0), opaqueProducer.inputs().get(0));
        assertSame(substitutions.get(0), opaqueProducer.inputs().get(1));
        assertThrows(IllegalArgumentException.class, () -> GraphCompilationPort.captureSource(opaque));
        // Original reusable templates still capture successfully after reconstruction.
        assertEquals(source.graph(), GraphCompilationPort.captureSource(roots).graph());
    }

    /**
     * Binds fresh ordinary actuals, then verifies all ports and O(body) reconstruction count.
     * @param body source template with canonical declarations
     * @param time static time count, including zero
     * @param reverse decreasing traversal flag
     */
    private static void assertColdBodyRoundTrip(ScanBody body, long time, boolean reverse) {
        var carry = body.carry().stream().map(t -> TensorFactory.create(t.descriptor())).toList();
        var sequences = body.step().stream().map(t -> {
            long[] dims = new long[t.descriptor().shape().rank() + 1]; dims[0] = time;
            System.arraycopy(t.descriptor().shape().toLongArray(), 0, dims, 1, dims.length - 1);
            return formal(t.descriptor().dataType(), dims);
        }).toList();
        var invariants = body.invariants().stream().map(t -> TensorFactory.create(t.descriptor())).toList();
        var scan = Scan.apply(carry, sequences, invariants, body, reverse);
        var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs()); roots.addAll(scan.residuals());
        var source = GraphCompilationPort.captureSource(roots);
        Tensor before = formal(DataType.BOOL);
        var rebuilt = GraphCompilationPort.reconstructSource(source.graph(), source.leaves());
        Tensor after = formal(DataType.BOOL);
        var local = source.graph().nodes().getFirst().scanBody().orElseThrow().graph();
        long expected = local.inputs().size() + local.nodes().stream().mapToInt(n -> n.outputs().size()).sum() + roots.size();
        assertEquals(expected + 1, after.id().value() - before.id().value());
        assertEquals(roots.stream().map(Tensor::descriptor).toList(), rebuilt.stream().map(Tensor::descriptor).toList());
        assertEquals(source.graph(), GraphCompilationPort.captureSource(rebuilt).graph());
        var freshBody = rebuilt.getFirst().provenance().orElseThrow().producer().scanBody().orElseThrow();
        assertEquals(ScanBodyInference.formals(body).size(), ScanBodyInference.formals(freshBody).size());
        assertNotSame(body, freshBody);
    }

    /**
     * Creates a canonical, storage-free non-gradient formal of any Model dtype.
     * @param type exact Model dtype
     * @param dims static extents, possibly empty or zero
     * @return fresh ordinary canonical leaf
     */
    private static Tensor formal(DataType type, long... dims) {
        return ScanBodyInferenceTest.tensor(type, false, dims);
    }

    /**
     * Supplies typed raw parameters that would lose information through double conversion.
     * @param type numeric Model dtype
     * @return exact typed NaN payload or large integral scalar
     */
    private static ScalarValue exactScalar(DataType type) {
        return switch (type) {
            case FLOAT64 -> ScalarValue.float64(Double.longBitsToDouble(0xfff8000000000123L));
            case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(0xffc00123));
            case FLOAT16 -> ScalarValue.float16Bits((short) 0xfe23);
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0xffc3);
            case INT64 -> ScalarValue.int64(9_007_199_254_740_993L);
            case INT32 -> ScalarValue.int32(Integer.MIN_VALUE);
            default -> throw new AssertionError(type);
        };
    }

    @Test
    void preflightsEveryLateDeadAndEmptyScanScopeBeforeAllocatingAnyTensor() {
        for (boolean lengthMode : List.of(false, true)) for (long time : List.of(0L, 3L)) {
            Tensor h = formal(DataType.FLOAT32, 2), x = formal(DataType.FLOAT32, 2);
            Tensor unused = formal(DataType.FLOAT32, 2);
            var body = new ScanBody(List.of(h), List.of(x), List.of(unused),
                    List.of(h.add(x).tanh()), List.of(h));
            Tensor initial = formal(DataType.FLOAT32, 2), sequence = formal(DataType.FLOAT32, time, 2);
            Tensor invariant = formal(DataType.FLOAT32, 2), lengths = formal(DataType.INT64, 2);
            var scan = lengthMode
                    ? Scan.apply(List.of(initial), List.of(sequence), List.of(invariant), lengths, 1,
                            List.of(java.util.OptionalInt.of(0)), body, false)
                    : Scan.apply(List.of(initial), List.of(sequence), List.of(invariant), body, false);
            Tensor early = formal(DataType.FLOAT32, 2).neg();
            var source = GraphCompilationPort.captureSource(List.of(early, scan.finalCarry().getFirst()));
            var node = source.graph().nodes().getLast();
            var raw = node.scanBody().orElseThrow();
            var invalidBodies = new ArrayList<CompiledScanBody>();
            invalidBodies.add(new CompiledScanBody(raw.graph(), raw.nextCarry(), raw.emissions(), raw.residuals(), List.of(0)));
            var reordered = new ArrayList<>(raw.graph().outputs()); java.util.Collections.reverse(reordered);
            invalidBodies.add(new CompiledScanBody(withOutputs(raw.graph(), reordered),
                    raw.nextCarry(), raw.emissions(), raw.residuals(), List.of()));
            var residuals = new ArrayList<>(raw.residuals()); java.util.Collections.reverse(residuals);
            invalidBodies.add(new CompiledScanBody(raw.graph(), raw.nextCarry(), raw.emissions(), residuals, List.of()));
            var phases = new HashMap<>(raw.graph().nodePhases());
            phases.put(raw.graph().nodes().getLast().id(), GraphPhase.BACKWARD);
            invalidBodies.add(withBodyGraph(raw, new CompiledGraphModel(raw.graph().values(), raw.graph().nodes(),
                    raw.graph().inputs(), raw.graph().outputs(), phases)));
            var nodes = new ArrayList<>(raw.graph().nodes());
            var last = nodes.getLast();
            nodes.set(nodes.size() - 1, new CompiledNode(last.id(), operation(UnaryElementwiseKind.ABS), last.inputs(), last.outputs()));
            invalidBodies.add(withBodyGraph(raw, new CompiledGraphModel(raw.graph().values(), nodes,
                    raw.graph().inputs(), raw.graph().outputs(), raw.graph().nodePhases())));
            // Every stored value is checked, including unused formals and wrong raw layout/flags.
            for (ValueId valueId : List.of(raw.graph().inputs().getLast(), last.outputs().getFirst())) {
                for (TensorDescriptor invalid : List.of(descriptor(DataType.FLOAT32, Shape.of(2), false),
                        descriptor(DataType.FLOAT64, Shape.of(2), false),
                        descriptor(DataType.FLOAT32, Shape.of(2), true),
                        descriptor(DataType.FLOAT32, Shape.of(Long.MAX_VALUE, 2), false),
                        descriptor(DataType.FLOAT32, Shape.ofDimensions(new DynamicDimension("N")), false))) {
                    var values = raw.graph().values().stream().map(v -> v.id().equals(valueId)
                            ? new GraphValue(v.id(), invalid) : v).toList();
                    // An unresolved layout is already correct for the TANH result.
                    if (values.equals(raw.graph().values())) continue;
                    invalidBodies.add(withBodyGraph(raw, new CompiledGraphModel(values, raw.graph().nodes(),
                            raw.graph().inputs(), raw.graph().outputs(), raw.graph().nodePhases())));
                }
            }
            for (CompiledScanBody invalidBody : invalidBodies) {
                var badNode = new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs(), Optional.of(invalidBody));
                var badNodes = new ArrayList<>(source.graph().nodes()); badNodes.set(badNodes.size() - 1, badNode);
                var bad = new CompiledGraphModel(source.graph().values(), badNodes, source.graph().inputs(),
                        source.graph().outputs(), source.graph().nodePhases());
                for (CompiledGraphModel graph : List.of(bad, withOutputs(bad, List.of(bad.outputs().getFirst())))) {
                    assertColdFailureWithoutIds(() -> GraphCompilationPort.reconstructSource(graph, source.leaves()));
                    assertColdFailureWithoutIds(() -> GraphCompilationPort.validateSource(graph));
                    assertColdFailureWithoutIds(() -> new GraphCompilationPort.SourceCapture(graph, source.leaves()));
                }
            }
            // Structural missing/extra role outputs and missing attachments remain Model failures.
            assertColdFailureWithoutIds(() -> new CompiledScanBody(withOutputs(raw.graph(),
                    raw.graph().outputs().subList(1, raw.graph().outputs().size())),
                    raw.nextCarry(), raw.emissions(), raw.residuals(), List.of()));
            var extra = new ArrayList<>(raw.graph().outputs()); extra.add(raw.graph().inputs().get(1));
            assertColdFailureWithoutIds(() -> new CompiledScanBody(withOutputs(raw.graph(), extra),
                    raw.nextCarry(), raw.emissions(), raw.residuals(), List.of()));
            assertColdFailureWithoutIds(() -> new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs()));
        }
    }

    @Test
    void symbolicLengthModeStillRejectsLateDeadAndZeroTimeBeforeAnyIds() {
        for (long time : List.of(0L, 3L)) {
            var batch = new io.github.pho001.synaptik.model.shape.DynamicDimension("Customers");
            var width = new io.github.pho001.synaptik.model.shape.StaticDimension(2);
            var row = Shape.ofDimensions(batch, width);
            java.util.function.BiFunction<DataType, Shape, Tensor> source = (type, shape) -> TensorFactory.create(
                    new TensorDescriptor(type, shape, shape.isFullyStatic()
                            ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), false));
            Tensor h = source.apply(DataType.FLOAT32, row), x = source.apply(DataType.FLOAT32, row);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), List.of());
            Tensor initial = source.apply(DataType.FLOAT32, row), sequence = source.apply(DataType.FLOAT32,
                    Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.StaticDimension(time), batch, width));
            Tensor lengths = source.apply(DataType.INT64, Shape.ofDimensions(batch)), earlyInput = formal(DataType.FLOAT32, 2, 2);
            Tensor early = earlyInput.neg();
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths, 1,
                    List.of(java.util.OptionalInt.of(0)), body, false);
            var raw = GraphCapture.capture(List.of(early, scan.finalCarry().getFirst()));
            List<Tensor> leaves = List.of(earlyInput, initial, sequence, lengths);
            assertEquals(4, raw.inputs().size());
            for (var graph : List.of(raw, withOutputs(raw, List.of(raw.outputs().getFirst())))) {
                assertColdFailureWithoutIds(() -> SourceGraphReconstruction.preflight(graph));
                assertColdFailureWithoutIds(() -> GraphCompilationPort.reconstructSource(graph, leaves));
                assertColdFailureWithoutIds(() -> new GraphCompilationPort.SourceCapture(graph, leaves));
            }
            assertColdFailureWithoutIds(() -> GraphCompilationPort.captureSource(List.of(early, scan.finalCarry().getFirst())));
            assertColdFailureWithoutIds(() -> GraphCapture.captureSource(List.of(early, scan.finalCarry().getFirst())));
        }
    }

    @Test
    void lengthRoundTripsEveryRolePortAndScopeWithTimeIndependentConstruction() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long batch : List.of(0L, 1L, 3L)) for (long time : List.of(0L, 3L, 100_000L))
                for (boolean reverse : List.of(false, true)) {
                    Tensor h = formal(type, batch, 2), unusedCarry = formal(type, batch, 2);
                    Tensor accumulator = formal(type, 2);
                    Tensor x = formal(type, batch, 2), transposed = formal(type, 2, batch);
                    Tensor shared = formal(type, 2), weight = formal(type, 2), unused = formal(type, 2);
                    Tensor next = h.add(x).add(transposed.permute(1, 0)).add(shared).add(weight)
                            .mul(type == DataType.FLOAT32 ? ScalarValue.float32(-0.0f) : ScalarValue.float64(-0.0))
                            .cast(type).tanh();
                    var body = new ScanBody(List.of(h, unusedCarry, accumulator), List.of(x, transposed, shared),
                            List.of(weight, unused), List.of(next, next, accumulator.add(next.sum(0, false))),
                            List.of(next, h, next));
                    Tensor initial = formal(type, batch, 2), total = formal(type, 2);
                    Tensor sequence = formal(type, time, batch, 2), other = formal(type, time, 2, batch);
                    Tensor sharedSequence = formal(type, time, 2), invariant = formal(type, 2);
                    Tensor lengths = formal(DataType.INT64, batch);
                    var a = Scan.apply(List.of(initial, initial, total), List.of(sequence, other, sharedSequence),
                            List.of(invariant, invariant), lengths, 2,
                            List.of(java.util.OptionalInt.of(0), java.util.OptionalInt.of(1), java.util.OptionalInt.empty()),
                            body, reverse);
                    var b = Scan.apply(List.of(initial, initial, total), List.of(sequence, other, sharedSequence),
                            List.of(invariant, invariant), lengths, 2,
                            List.of(java.util.OptionalInt.of(0), java.util.OptionalInt.of(1), java.util.OptionalInt.empty()),
                            body, !reverse);
                    var roots = new ArrayList<>(a.finalCarry());
                    roots.addAll(a.outputs()); roots.addAll(a.residuals()); roots.add(b.finalCarry().getFirst());
                    var source = GraphCompilationPort.captureSource(roots);
                    var arbitrary = relabel(source.graph());
                    Tensor before = formal(DataType.BOOL);
                    var rebuilt = GraphCompilationPort.reconstructSource(arbitrary, source.leaves());
                    long expected = source.graph().nodes().stream().mapToLong(n -> {
                        var local = n.scanBody().orElseThrow().graph();
                        return local.inputs().size() + local.nodes().stream().mapToLong(p -> p.outputs().size()).sum()
                                + n.outputs().size();
                    }).sum();
                    assertEquals(expected + 1, formal(DataType.BOOL).id().value() - before.id().value());
                    assertEquals(source.graph(), GraphCompilationPort.captureSource(rebuilt).graph());
                    var pa = rebuilt.getFirst().provenance().orElseThrow().producer();
                    var pb = rebuilt.getLast().provenance().orElseThrow().producer();
                    assertNotSame(pa, pb);
                    for (int i = 0; i < pa.outputCount(); i++) assertSame(pa.output(i), rebuilt.get(i));
                    assertEquals(9, pa.inputs().size());
                    assertSame(lengths, pa.inputs().getLast());
                    assertSame(invariant, pa.inputs().get(6)); assertSame(invariant, pa.inputs().get(7));
                    assertSame(pa.inputs().getFirst(), pa.inputs().get(1));
                    var fresh = pa.scanBody().orElseThrow();
                    assertEquals(8, ScanBodyInference.formals(fresh).stream().distinct().count());
                    assertNotSame(body, fresh); assertNotSame(fresh, pb.scanBody().orElseThrow());
                    for (Tensor f : ScanBodyInference.formals(fresh)) {
                        assertFalse(ScanBodyInference.formals(body).contains(f));
                        assertFalse(ScanBodyInference.formals(pb.scanBody().orElseThrow()).contains(f));
                        assertTrue(f.hostStorage().isEmpty()); assertTrue(f.provenance().isEmpty());
                    }
                    assertSame(fresh.nextCarry().getFirst(), fresh.nextCarry().get(1));
                    assertSame(fresh.nextCarry().getFirst(), fresh.emissions().getFirst());
                }
    }

    @Test
    void lengthSubstitutionTiesAreCheckedPerOccurrenceBeforeEarlierProducerIds() {
        for (long batch : List.of(0L, 1L, 2L)) {
            Tensor h = formal(DataType.FLOAT32, batch, 2), x = formal(DataType.FLOAT32, batch, 2);
            Tensor invariant = formal(DataType.FLOAT32, batch, 2);
            var body = new ScanBody(List.of(h), List.of(x), List.of(invariant), List.of(h.add(x)), List.of(h));
            Tensor initial = formal(DataType.FLOAT32, batch, 2), state = formal(DataType.FLOAT32, batch, 2);
            Tensor sequence = formal(DataType.FLOAT32, 3, batch, 2), lengths = formal(DataType.INT64, batch);
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(state), lengths, 1,
                    List.of(java.util.OptionalInt.of(0)), body, false);
            var source = GraphCompilationPort.captureSource(List.of(initial.neg(), scan.finalCarry().getFirst()));
            var bad = source.leaves().stream().map(t -> t == state ? initial : t).toList();
            for (var graph : List.of(source.graph(), withOutputs(source.graph(), List.of(source.graph().outputs().getFirst()))))
                assertColdFailureWithoutIds(() -> GraphCompilationPort.reconstructSource(graph, bad));
            Tensor opaque = initial.abs().contiguous();
            var good = source.leaves().stream().map(t -> t == initial ? opaque : t).toList();
            var rebuilt = GraphCompilationPort.reconstructSource(source.graph(), good);
            assertSame(opaque, rebuilt.getLast().provenance().orElseThrow().inputs().getFirst());
            // The same replacement can be ROW in one occurrence and SHARED in another.
            var second = Scan.apply(List.of(state), List.of(sequence), List.of(initial), lengths, 1,
                    List.of(java.util.OptionalInt.of(0)), body, true);
            var independent = GraphCompilationPort.captureSource(List.of(scan.finalCarry().getFirst(), second.finalCarry().getFirst()));
            assertEquals(independent.graph(), GraphCompilationPort.captureSource(
                    GraphCompilationPort.reconstructSource(independent.graph(), independent.leaves())).graph());
        }
        // Equal sequence descriptors do not prove equal outer row axes (or SHARED).
        for (java.util.OptionalInt secondRole : List.of(java.util.OptionalInt.of(1), java.util.OptionalInt.empty())) {
            Tensor h = formal(DataType.FLOAT64, 2, 2), x = formal(DataType.FLOAT64, 2, 2), y = formal(DataType.FLOAT64, 2, 2);
            var body = new ScanBody(List.of(h), List.of(x, y), List.of(), List.of(h.add(x)), List.of());
            Tensor initial = formal(DataType.FLOAT64, 2, 2), first = formal(DataType.FLOAT64, 0, 2, 2);
            Tensor second = formal(DataType.FLOAT64, 0, 2, 2), lengths = formal(DataType.INT64, 2);
            var scan = Scan.apply(List.of(initial), List.of(first, second), List.of(), lengths, 1,
                    List.of(java.util.OptionalInt.of(0), secondRole), body, false);
            var source = GraphCompilationPort.captureSource(scan.finalCarry());
            var bad = source.leaves().stream().map(t -> t == second ? first : t).toList();
            assertColdFailureWithoutIds(() -> GraphCompilationPort.reconstructSource(source.graph(), bad));
        }
    }

    @Test
    void newlyTypedLengthDeclarationsAndHiddenCastInteriorsStayColdUnsupported() {
        for (long batch : List.of(0L, 1L, 2L)) for (long time : List.of(0L, 3L))
            for (DataType type : DataType.values()) if (type != DataType.FLOAT32) {
                for (int role = -2; role < 3; role++) {
                    if (role == -2 && type == DataType.FLOAT64) continue;
                    var scan = LengthAwareScanBodyInferenceTest.typedLengthSource(type,
                            new io.github.pho001.synaptik.model.shape.StaticDimension(batch), time, role);
                    var actuals = scan.finalCarry().getFirst().provenance().orElseThrow().inputs();
                    Tensor earlyInput = formal(DataType.FLOAT32, 2), early = earlyInput.neg();
                    var raw = GraphCapture.capture(List.of(early, scan.finalCarry().getFirst()));
                    var leaves = new ArrayList<>(List.of(earlyInput)); leaves.addAll(actuals);
                    for (var graph : List.of(raw, withOutputs(raw, List.of(raw.outputs().getFirst())))) {
                        assertColdFailureWithoutIds(() -> GraphCompilationPort.reconstructSource(graph, leaves));
                        assertColdFailureWithoutIds(() -> new GraphCompilationPort.SourceCapture(graph, leaves));
                    }
                    assertColdFailureWithoutIds(() -> GraphCompilationPort.captureSource(List.of(early, scan.finalCarry().getFirst())));
                }
            }
    }

    @Test
    void separateLengthOccurrencesMayUseDifferentHomogeneousDtypes() {
        var roots = new ArrayList<Tensor>();
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var scan = LengthAwareScanBodyInferenceTest.typedLengthSource(type,
                    new io.github.pho001.synaptik.model.shape.StaticDimension(2), 3, -2);
            roots.addAll(scan.finalCarry());
        }
        assertRoundTrip(roots);
    }

    @Test
    void rawLengthHeadersAndRowFeedbackRejectLateDeadAndEmptyBeforeAnyIds() {
        for (long batch : List.of(0L, 1L, 2L)) for (long time : List.of(0L, 3L)) {
            Tensor h = formal(DataType.FLOAT64, batch, 2), x = formal(DataType.FLOAT64, batch, 2);
            Tensor initial = formal(DataType.FLOAT64, batch, 2), sequence = formal(DataType.FLOAT64, time, batch, 2);
            Tensor early = initial.neg();
            for (Tensor root : List.of(h, h.sum(0, true).expand(h.descriptor().shape()),
                    h.add(x.sum(0, true).expand(h.descriptor().shape())))) {
                var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(root), List.of(h));
                var allValid = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
                var raw = GraphCapture.capture(List.of(early, allValid.finalCarry().getFirst()));
                var controls = new ArrayList<>(List.of(formal(DataType.INT32, batch),
                        formal(DataType.INT64, batch, 1), formal(DataType.INT64, batch + 1)));
                if (root != h) controls.add(formal(DataType.INT64, batch));
                for (Tensor control : controls) {
                    var bad = LengthAwareScanBodyInferenceTest.lengthIR(raw, 1,
                            List.of(java.util.OptionalInt.of(0)), control.descriptor());
                    List<Tensor> leaves = List.of(initial, sequence, control);
                    for (var graph : List.of(bad, withOutputs(bad, List.of(bad.outputs().getFirst())))) {
                        assertColdFailureWithoutIds(() -> GraphCompilationPort.reconstructSource(graph, leaves));
                        assertColdFailureWithoutIds(() -> new GraphCompilationPort.SourceCapture(graph, leaves));
                    }
                }
            }
            Tensor old = formal(DataType.FLOAT64, 2), total = formal(DataType.FLOAT64, 2);
            for (Tensor badUpdate : List.of(old, old.add(old), old.add(h.sum(0, false)).add(old))) {
                var body = new ScanBody(List.of(h, old), List.of(x), List.of(), List.of(h, badUpdate), List.of());
                var allValid = Scan.apply(List.of(initial, total), List.of(sequence), List.of(), body, true);
                var raw = GraphCapture.capture(List.of(early, allValid.finalCarry().getFirst()));
                Tensor lengths = formal(DataType.INT64, batch);
                var bad = LengthAwareScanBodyInferenceTest.lengthIR(raw, 1,
                        List.of(java.util.OptionalInt.of(0)), lengths.descriptor());
                for (var graph : List.of(bad, withOutputs(bad, List.of(bad.outputs().getFirst())))) {
                    assertColdFailureWithoutIds(() -> GraphCompilationPort.reconstructSource(graph, List.of(initial, total, sequence, lengths)));
                    assertColdFailureWithoutIds(() -> new GraphCompilationPort.SourceCapture(graph, List.of(initial, total, sequence, lengths)));
                }
            }
        }
    }

    @Test
    void emptyDomainsCannotHideUnusedFeatureGeometryOverflowBeforeAnyIds() {
        for (long batch : List.of(0L, 1L, 2L)) for (long time : List.of(0L, 3L)) {
            Tensor h = formal(DataType.FLOAT32, batch, 2), x = formal(DataType.FLOAT32, batch, 2);
            Tensor initial = formal(DataType.FLOAT32, batch, 2), sequence = formal(DataType.FLOAT32, time, batch, 2);
            Tensor lengths = formal(DataType.INT64, batch);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), List.of());
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths, 1,
                    List.of(java.util.OptionalInt.of(0)), body, false);
            var source = GraphCompilationPort.captureSource(List.of(initial.neg(), scan.finalCarry().getFirst()));
            var node = source.graph().nodes().getLast();
            var attachment = node.scanBody().orElseThrow();
            TensorDescriptor overflow = descriptor(DataType.FLOAT32, Shape.of(batch, Long.MAX_VALUE, 2), false);
            TensorDescriptor sequenceOverflow = descriptor(DataType.FLOAT32, Shape.of(time, batch, Long.MAX_VALUE, 2), false);
            var localValues = attachment.graph().values().stream().map(v ->
                    v.id().equals(attachment.graph().inputs().getLast()) ? new GraphValue(v.id(), overflow) : v).toList();
            var local = new CompiledGraphModel(localValues, attachment.graph().nodes(), attachment.graph().inputs(),
                    attachment.graph().outputs(), attachment.graph().nodePhases());
            var nodes = new ArrayList<>(source.graph().nodes());
            nodes.set(nodes.size() - 1, new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs(),
                    Optional.of(withBodyGraph(attachment, local))));
            var values = source.graph().values().stream().map(v ->
                    v.id().equals(node.inputs().get(1)) ? new GraphValue(v.id(), sequenceOverflow) : v).toList();
            var bad = new CompiledGraphModel(values, nodes, source.graph().inputs(),
                    source.graph().outputs(), source.graph().nodePhases());
            for (var graph : List.of(bad, withOutputs(bad, List.of(bad.outputs().getFirst())))) {
                assertColdFailureWithoutIds(() -> SourceGraphReconstruction.preflight(graph));
                assertColdFailureWithoutIds(() -> GraphCompilationPort.reconstructSource(graph, source.leaves()));
                assertColdFailureWithoutIds(() -> new GraphCompilationPort.SourceCapture(graph, source.leaves()));
            }
        }
    }

    @Test
    void admittedLengthRnnAlgebraRetainsOrdinaryFirstOrderAutograd() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long time : List.of(0L, 3L)) for (boolean reverse : List.of(false, true)) {
                Tensor h = ScanBodyInferenceTest.tensor(type, false, 2, 2);
                Tensor x = ScanBodyInferenceTest.tensor(type, true, 2, 2);
                Tensor w = ScanBodyInferenceTest.tensor(type, true, 2, 2);
                Tensor initial = formal(type, 2, 2), sequence = ScanBodyInferenceTest.tensor(type, true, time, 2, 2);
                Tensor weight = ScanBodyInferenceTest.tensor(type, true, 2, 2), lengths = formal(DataType.INT64, 2);
                var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.add(x.matmul(w)).tanh()), List.of(h));
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), lengths, 1,
                        List.of(java.util.OptionalInt.of(0)), body, reverse);
                var source = GraphCompilationPort.captureSource(List.of(scan.finalCarry().getFirst()));
                var rebuilt = GraphCompilationPort.reconstructSource(source.graph(), source.leaves());
                var originalRoots = List.of(scan.finalCarry().getFirst().sum());
                var rebuiltRoots = List.of(rebuilt.getFirst().sum());
                var targets = List.of(sequence, weight);
                if (time == 0) {
                    // No cell is entered: final=initial, with no sequence or parameter route.
                    assertThrows(IllegalArgumentException.class, () -> compile(originalRoots, targets, true));
                    assertThrows(IllegalArgumentException.class, () -> compile(rebuiltRoots, targets, true));
                }
                assertEquivalentCompile(originalRoots, rebuiltRoots, targets, targets, time == 0
                        ? FunctionalGradientRequest.DisconnectedPolicy.ZERO
                        : FunctionalGradientRequest.DisconnectedPolicy.ERROR);
            }
    }

    @Test
    void generalScanSupportsArbitraryScopeQualifiedIdsAndTimeIndependentObjectCounts() {
        Tensor h = formal(DataType.FLOAT64, 2), x = formal(DataType.FLOAT64, 2);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).tanh()), List.of(h));
        for (long time : List.of(0L, 3L, 100_000L)) for (boolean reverse : List.of(false, true)) {
            assertColdBodyRoundTrip(body, time, reverse);
        }
        var scan = Scan.apply(List.of(formal(DataType.FLOAT64, 2)), List.of(formal(DataType.FLOAT64, 3, 2)),
                List.of(), body, true);
        var source = GraphCompilationPort.captureSource(List.of(scan.residuals().getLast(), scan.finalCarry().getFirst()));
        var arbitrary = relabel(source.graph());
        var local = arbitrary.nodes().getFirst().scanBody().orElseThrow().graph();
        assertEquals(arbitrary.inputs().getFirst(), local.inputs().getFirst(), "numeric IDs deliberately coincide across scopes");
        assertEquals(source.graph(), GraphCompilationPort.captureSource(
                GraphCompilationPort.reconstructSource(arbitrary, source.leaves())).graph());
        // Logical geometry only: no physical-byte limit or provider restriction belongs here.
        Tensor hugeH = formal(DataType.FLOAT64, Long.MAX_VALUE), hugeX = formal(DataType.FLOAT64, Long.MAX_VALUE);
        assertColdBodyRoundTrip(new ScanBody(List.of(hugeH), List.of(hugeX), List.of(), List.of(hugeH), List.of()), 0, false);
    }

    @Test
    void generalColdRoundTripPreservesMixedFormalFlagsAndOrdinaryAutograd() {
        for (long time : List.of(0L, 3L)) for (boolean reverse : List.of(false, true)) {
            Tensor h = ScanBodyInferenceTest.tensor(DataType.FLOAT32, false);
            Tensor x = ScanBodyInferenceTest.tensor(DataType.FLOAT32, true);
            Tensor w = ScanBodyInferenceTest.tensor(DataType.FLOAT32, true);
            Tensor initial = formal(DataType.FLOAT32);
            Tensor sequence = ScanBodyInferenceTest.tensor(DataType.FLOAT32, true, time);
            Tensor weight = ScanBodyInferenceTest.tensor(DataType.FLOAT32, true);
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.add(x).mul(w).tanh()), List.of(h));
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse);
            var source = GraphCompilationPort.captureSource(List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst()));
            var rebuilt = GraphCompilationPort.reconstructSource(source.graph(), source.leaves());
            var rebuiltBody = rebuilt.getFirst().provenance().orElseThrow().producer().scanBody().orElseThrow();
            assertFalse(rebuiltBody.carry().getFirst().descriptor().requiresGrad());
            assertTrue(rebuiltBody.step().getFirst().descriptor().requiresGrad());
            assertTrue(rebuiltBody.invariants().getFirst().descriptor().requiresGrad());
            assertFalse(rebuiltBody.emissions().getFirst().descriptor().requiresGrad());
            assertTrue(rebuilt.get(1).descriptor().requiresGrad());
            var originalRoots = List.of(scan.finalCarry().getFirst());
            var rebuiltRoots = List.of(rebuilt.getFirst());
            var targets = List.of(sequence, weight);
            if (time == 0) {
                // Mixed formal flags do not create value routes through an unentered cell.
                assertThrows(IllegalArgumentException.class, () -> compile(originalRoots, targets, true));
                assertThrows(IllegalArgumentException.class, () -> compile(rebuiltRoots, targets, true));
            }
            assertEquivalentCompile(originalRoots, rebuiltRoots, targets, targets, time == 0
                    ? FunctionalGradientRequest.DisconnectedPolicy.ZERO
                    : FunctionalGradientRequest.DisconnectedPolicy.ERROR);
        }
    }

    @Test
    void unsupportedPureBodiesAndBodyOnlySignaturesNeverBroadenFlatAdmission() {
        Tensor h = formal(DataType.FLOAT32, 2, 2), x = formal(DataType.FLOAT32, 2, 2);
        Tensor scalar = formal(DataType.FLOAT32);
        for (Tensor unsupported : List.of(h.abs(), h.cast(DataType.FLOAT64), h.permute(0, 1), h.mean(), h.add(x).pow(x))) {
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), List.of(unsupported));
            var scan = Scan.apply(List.of(formal(DataType.FLOAT32, 2, 2)), List.of(formal(DataType.FLOAT32, 0, 2, 2)),
                    List.of(), body, false);
            assertColdFailureWithoutIds(() -> GraphCompilationPort.captureSource(scan.finalCarry()));
            var raw = GraphCapture.capture(scan.finalCarry());
            List<Tensor> leaves = List.of(scan.finalCarry().getFirst().provenance().orElseThrow().inputs().get(0),
                    scan.finalCarry().getFirst().provenance().orElseThrow().inputs().get(1));
            assertColdFailureWithoutIds(() -> GraphCompilationPort.reconstructSource(raw, leaves));
        }
        for (Tensor unsupported : List.of(h.div(x), h.div(ScalarValue.float32(2)),
                scalar.expand(Shape.of(2, 2)), h.contiguous(), h.cast(DataType.FLOAT32), h.sum(),
                h.sum(new int[] {}, true), h.sumToShape(Shape.of(2, 2)))) {
            assertColdFailureWithoutIds(() -> GraphCompilationPort.captureSource(List.of(unsupported)));
        }
    }

    /**
     * Changes only a graph's ordered publication roots, preserving dead stored nodes and values.
     * @param graph structurally closed input graph
     * @param outputs valid unique value IDs in desired publication order
     * @return immutable graph with identical stored facts except outputs
     */
    private static CompiledGraphModel withOutputs(CompiledGraphModel graph, List<ValueId> outputs) {
        return new CompiledGraphModel(graph.values(), graph.nodes(), graph.inputs(), outputs, graph.nodePhases());
    }

    /**
     * Replaces a body graph while retaining its raw role bindings and empty demand.
     * @param body source attachment
     * @param graph replacement structurally valid local graph with the same IDs
     * @return immutable attachment retaining ordered roles
     */
    private static CompiledScanBody withBodyGraph(CompiledScanBody body, CompiledGraphModel graph) {
        return new CompiledScanBody(graph, body.nextCarry(), body.emissions(), body.residuals(), List.of());
    }

    /**
     * Verifies known cold/structural rejection consumes neither formal nor producer IDs.
     * @param action fully prepared failure action, with no test-input allocation inside it
     */
    private static void assertColdFailureWithoutIds(Runnable action) {
        Tensor before = formal(DataType.BOOL);
        assertThrows(IllegalArgumentException.class, action::run);
        assertEquals(before.id().value() + 1, formal(DataType.BOOL).id().value());
    }

    /**
     * Assigns sparse, descending IDs independently in each lexical scope.
     * @param graph structurally valid source graph
     * @return equivalent graph with arbitrary local identities and unchanged descriptors/order
     */
    private static CompiledGraphModel relabel(CompiledGraphModel graph) {
        var ids = new HashMap<ValueId, ValueId>();
        for (int i = 0; i < graph.values().size(); i++) ids.put(graph.values().get(i).id(), id(900 - i * 7L));
        var phases = new HashMap<NodeId, GraphPhase>();
        var nodes = new ArrayList<CompiledNode>();
        for (int i = 0; i < graph.nodes().size(); i++) {
            var node = graph.nodes().get(i); var nodeId = new NodeId(700 - i * 3L);
            phases.put(nodeId, graph.nodePhases().get(node.id()));
            var attachment = node.scanBody().map(b -> {
                var local = relabel(b.graph());
                var remap = new HashMap<ValueId, ValueId>();
                for (int j = 0; j < b.graph().values().size(); j++) remap.put(b.graph().values().get(j).id(), local.values().get(j).id());
                return new CompiledScanBody(local, b.nextCarry().stream().map(remap::get).toList(),
                        b.emissions().stream().map(remap::get).toList(), b.residuals().stream().map(remap::get).toList(), List.of());
            });
            nodes.add(new CompiledNode(nodeId, node.operation(), node.inputs().stream().map(ids::get).toList(),
                    node.outputs().stream().map(ids::get).toList(), attachment));
        }
        return new CompiledGraphModel(graph.values().stream().map(v -> new GraphValue(ids.get(v.id()), v.descriptor())).toList(),
                nodes, graph.inputs().stream().map(ids::get).toList(), graph.outputs().stream().map(ids::get).toList(), phases);
    }

    @Test
    void activationsPreserveExactKindsSharingDescriptorsAndOrdinaryAutogradAcrossFloatingTypes() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            Tensor x = leaf(type, Shape.of(2, 3), true);
            for (var activation : List.of(UnaryElementwiseKind.RELU, UnaryElementwiseKind.TANH,
                    UnaryElementwiseKind.SIGMOID)) {
                Tensor active = activate(x, activation);
                var outputs = List.of(active.mul(active), activate(x, activation), active, x);
                var source = GraphCompilationPort.captureSource(outputs);
                var rebuilt = GraphCompilationPort.reconstructSource(source.graph(), source.leaves());
                assertEquals(source.graph(), GraphCompilationPort.captureSource(rebuilt).graph());
                assertEquals(activation, source.graph().nodes().getFirst().operation().kind());
                assertSame(NoOperationAttrs.INSTANCE, source.graph().nodes().getFirst().operation().attrs());
                assertEquals(active.descriptor(), rebuilt.get(2).descriptor());
                assertSame(rebuilt.get(2), rebuilt.getFirst().provenance().orElseThrow().inputs().getFirst());
                assertSame(rebuilt.get(2), rebuilt.getFirst().provenance().orElseThrow().inputs().get(1));
                assertNotSame(rebuilt.get(1), rebuilt.get(2));
                assertSame(x, rebuilt.get(3));
                assertEquivalentCompile(List.of(active, active.sum()),
                        List.of(rebuilt.get(2), rebuilt.get(2).sum()), List.of(x), List.of(x));
            }
        }
    }

    @Test
    void lateDeadIntegralOrBoolActivationsRejectBeforeAnyProducerAllocation() {
        for (DataType type : List.of(DataType.INT32, DataType.INT64, DataType.BOOL)) {
            for (var kind : List.of(UnaryElementwiseKind.RELU, UnaryElementwiseKind.TANH,
                    UnaryElementwiseKind.SIGMOID)) {
                var d = descriptor(type, Shape.scalar(), false);
                var invalid = single(operation(kind), List.of(d), d, GraphPhase.FORWARD);
                Tensor input = TensorFactory.create(d);
                Tensor good = leaf(DataType.FLOAT32, Shape.scalar(), false);
                var values = List.of(new GraphValue(id(0), good.descriptor()), new GraphValue(id(1), d),
                        new GraphValue(id(2), good.descriptor()), new GraphValue(id(3), d));
                var nodes = List.of(new CompiledNode(new NodeId(0), operation(UnaryElementwiseKind.NEG),
                        List.of(id(0)), List.of(id(2))), new CompiledNode(new NodeId(1),
                        invalid.nodes().getFirst().operation(), List.of(id(1)), List.of(id(3))));
                var dead = new CompiledGraphModel(values, nodes, List.of(id(0), id(1)), List.of(id(2)),
                        Map.of(new NodeId(0), GraphPhase.FORWARD, new NodeId(1), GraphPhase.FORWARD));
                Tensor marker = TensorFactory.create(d);
                assertThrows(IllegalArgumentException.class,
                        () -> GraphCompilationPort.reconstructSource(dead, List.of(good, input)));
                assertEquals(marker.id().value() + 1, TensorFactory.create(d).id().value());
                assertThrows(IllegalArgumentException.class,
                        () -> new GraphCompilationPort.SourceCapture(dead, List.of(good, input)));
                assertThrows(IllegalArgumentException.class,
                        () -> new Operation(kind, new PermutationAttrs(List.of(1, 0))));
                assertThrows(IllegalArgumentException.class, () -> new CompiledNode(new NodeId(0),
                        operation(kind), List.of(id(0), id(1)), List.of(id(2))));
            }
        }
    }

    /**
     * Constructs only the selected ordinary activation, independently of cold reconstruction.
     * @param input source input with a floating descriptor
     * @param kind one of the three newly admitted activations
     * @return ordinary unary expression
     */
    private static Tensor activate(Tensor input, UnaryElementwiseKind kind) {
        return switch (kind) {
            case RELU -> input.relu();
            case TANH -> input.tanh();
            case SIGMOID -> input.sigmoid();
            default -> throw new AssertionError(kind);
        };
    }

    @Test
    void rebuildsEachOccurrenceOnceAndPreservesSharedCanonicalWrappersAndOutputOrder() {
        Tensor x = leaf(DataType.FLOAT32, Shape.of(2), true);
        Tensor y = leaf(DataType.FLOAT32, Shape.of(2), true);
        Tensor shared = x.add(y);
        Tensor separate = x.add(y);
        var source = GraphCompilationPort.captureSource(List.of(
                shared.mul(shared), separate, shared.neg(), x, shared));
        Tensor freshX = TensorFactory.create(x.descriptor());
        Tensor freshY = TensorFactory.create(y.descriptor());
        var supplied = new ArrayList<>(List.of(freshX, freshY));

        List<Tensor> rebuilt = GraphCompilationPort.reconstructSource(source.graph(), supplied);
        supplied.clear();
        Tensor rebuiltShared = rebuilt.get(4);
        assertSame(rebuiltShared, rebuilt.getFirst().provenance().orElseThrow().inputs().get(0));
        assertSame(rebuiltShared, rebuilt.getFirst().provenance().orElseThrow().inputs().get(1));
        assertSame(rebuiltShared, rebuilt.get(2).provenance().orElseThrow().inputs().getFirst());
        assertSame(rebuiltShared, rebuiltShared.provenance().orElseThrow().producer().output(0));
        assertNotSame(rebuiltShared, rebuilt.get(1));
        assertSame(freshX, rebuilt.get(3));
        assertSame(freshX, rebuiltShared.provenance().orElseThrow().inputs().getFirst());
        assertSame(freshY, rebuiltShared.provenance().orElseThrow().inputs().get(1));
        assertEquals(source.graph(), GraphCompilationPort.captureSource(rebuilt).graph());
        assertThrows(UnsupportedOperationException.class, () -> rebuilt.clear());
        assertNotSame(rebuiltShared, GraphCompilationPort.reconstructSource(
                source.graph(), List.of(freshX, freshY)).get(4));
    }

    @Test
    void preservesBroadcastPromotionVectorBatchedAndEmptyMatmulSemantics() {
        for (List<DataType> pair : List.of(
                List.of(DataType.INT32, DataType.INT64),
                List.of(DataType.FLOAT32, DataType.FLOAT64),
                List.of(DataType.BFLOAT16, DataType.FLOAT16))) {
            Tensor a = leaf(pair.get(0), Shape.of(2, 1), false);
            Tensor b = leaf(pair.get(1), Shape.of(1, 3), false);
            assertRoundTrip(List.of(a.add(b), a.mul(b)));
            for (List<Shape> shapes : List.of(
                    List.of(Shape.of(3), Shape.of(3)),
                    List.of(Shape.of(3), Shape.of(3, 4)),
                    List.of(Shape.of(2, 3), Shape.of(3)),
                    List.of(Shape.of(2, 3), Shape.of(3, 4)),
                    List.of(Shape.of(1, 2, 3), Shape.of(5, 3, 4)),
                    List.of(Shape.of(5, 2, 3), Shape.of(1, 3, 4)),
                    List.of(Shape.of(2, 0), Shape.of(0, 4)))) {
                Tensor left = leaf(pair.get(0), shapes.get(0), false);
                Tensor right = leaf(pair.get(1), shapes.get(1), false);
                assertRoundTrip(List.of(left.matmul(right)));
            }
        }
        for (DataType type : DataType.values()) {
            Tensor scalar = leaf(type, Shape.scalar(), false);
            Tensor empty = leaf(type, Shape.of(0, Long.MAX_VALUE, 2), false);
            assertRoundTrip(List.of(empty, scalar));
            if (type.isFloating()) assertRoundTrip(List.of(scalar.neg(), empty.neg()));
        }
    }

    @Test
    void acceptsArbitraryIdsUnusedInputsDeadValidNodesAndPassThrough() {
        Tensor x = leaf(DataType.FLOAT32, Shape.scalar(), false);
        Tensor y = TensorFactory.create(x.descriptor());
        var graph = new CompiledGraphModel(
                List.of(new GraphValue(id(70), x.descriptor()), new GraphValue(id(11), x.descriptor()),
                        new GraphValue(id(95), x.descriptor())),
                List.of(new CompiledNode(new NodeId(43), operation(UnaryElementwiseKind.NEG),
                        List.of(id(70)), List.of(id(95)))),
                List.of(id(11), id(70)), List.of(id(70), id(11)),
                Map.of(new NodeId(43), GraphPhase.FORWARD));
        assertEquals(List.of(y, x), GraphCompilationPort.reconstructSource(graph, List.of(x, y)));
        var passThrough = new CompiledGraphModel(graph.values().subList(0, 2), List.of(),
                graph.inputs(), graph.outputs(), Map.of());
        assertEquals(List.of(y, x), GraphCompilationPort.reconstructSource(passThrough, List.of(x, y)));
        var duplicates = GraphCompilationPort.reconstructSource(passThrough, List.of(x, x));
        assertSame(x, duplicates.getFirst());
        assertSame(x, duplicates.get(1));
        assertThrows(IllegalArgumentException.class, () -> compile(duplicates, List.of(), false));
        assertEquals(List.of(x, y), GraphCompilationPort.reconstructSource(passThrough, List.of(y, x)));
    }

    @Test
    void acceptsOpaqueProducedSubstitutionsPreservingSharedUpstreamProvenanceAndGradients() {
        Tensor prototype = leaf(DataType.FLOAT32, Shape.scalar(), true);
        var source = GraphCompilationPort.captureSource(List.of(prototype.mul(prototype).neg()));
        Tensor upstream = TensorFactory.create(prototype.descriptor());
        Tensor substitution = upstream.abs(); // ABS is deliberately outside source admission.
        Tensor rebuilt = GraphCompilationPort.reconstructSource(source.graph(), List.of(substitution)).getFirst();
        Tensor product = rebuilt.provenance().orElseThrow().inputs().getFirst();
        assertSame(substitution, product.provenance().orElseThrow().inputs().get(0));
        assertSame(substitution, product.provenance().orElseThrow().inputs().get(1));
        assertSame(upstream, substitution.provenance().orElseThrow().inputs().getFirst());
        assertThrows(IllegalArgumentException.class, () -> GraphCompilationPort.captureSource(List.of(rebuilt)));
        Tensor manual = substitution.mul(substitution).neg();
        assertEquivalentCompile(List.of(manual), List.of(rebuilt), List.of(upstream), List.of(upstream));
        assertEquals(upstream.id(), compile(List.of(rebuilt), List.of(upstream), true)
                .publication().gradientBindings().getFirst().target());
    }

    @Test
    void originalAndFreshLeafRoundTripsKeepOrdinaryCompileAndAutogradArtifacts() {
        Tensor x = leaf(DataType.FLOAT32, Shape.scalar(), true);
        Tensor y = leaf(DataType.FLOAT32, Shape.scalar(), true);
        Tensor shared = x.mul(x).add(y);
        List<Tensor> outputs = List.of(x.mul(x), shared.mul(shared).neg());
        var scalar = GraphCompilationPort.captureSource(outputs);
        var reconstructed = GraphCompilationPort.reconstructSource(scalar.graph(), scalar.leaves());
        assertEquivalentCompile(outputs, reconstructed, List.of(y, x), List.of(y, x));
        Tensor freshX = TensorFactory.create(x.descriptor());
        Tensor freshY = TensorFactory.create(y.descriptor());
        var fresh = GraphCompilationPort.reconstructSource(scalar.graph(), List.of(freshX, freshY));
        assertEquivalentCompile(outputs, fresh, List.of(y, x), List.of(freshY, freshX));

        Tensor input = leaf(DataType.FLOAT32, Shape.of(2, 3), true);
        Shape weightShape = Shape.of(4, 3);
        Tensor weight = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, weightShape,
                Optional.of(LayoutDescriptor.contiguous(weightShape)), true));
        Tensor bias = leaf(DataType.FLOAT32, Shape.of(4), true);
        Tensor linear = input.matmul(weight.permute(1, 0)).add(bias);
        var source = GraphCompilationPort.captureSource(List.of(linear));
        Tensor rebuilt = GraphCompilationPort.reconstructSource(source.graph(), source.leaves()).getFirst();
        // SUM is a later ordinary objective, never a new source-codec family.
        assertEquivalentCompile(List.of(linear, linear.sum()), List.of(rebuilt, rebuilt.sum()),
                List.of(bias, weight, input), List.of(bias, weight, input));
        var closed = compile(List.of(linear), List.of(), false).graph();
        assertThrows(IllegalArgumentException.class,
                () -> GraphCompilationPort.reconstructSource(closed, source.leaves()));
        assertThrows(IllegalArgumentException.class,
                () -> new GraphCompilationPort.SourceCapture(closed, source.leaves()));
    }

    @Test
    void rejectsNullsAndFullDescriptorAssociationMismatchesBeforeAnyRebuild() {
        Tensor x = leaf(DataType.FLOAT32, Shape.of(2, 3), true);
        var source = GraphCompilationPort.captureSource(List.of(x.neg()));
        assertThrows(NullPointerException.class, () -> GraphCompilationPort.reconstructSource(null, List.of(x)));
        assertThrows(NullPointerException.class, () -> GraphCompilationPort.reconstructSource(source.graph(), null));
        assertThrows(NullPointerException.class,
                () -> GraphCompilationPort.reconstructSource(source.graph(), Arrays.asList((Tensor) null)));
        assertThrows(IllegalArgumentException.class,
                () -> GraphCompilationPort.reconstructSource(source.graph(), List.of()));
        assertThrows(IllegalArgumentException.class,
                () -> GraphCompilationPort.reconstructSource(source.graph(), List.of(x, x)));
        for (TensorDescriptor bad : List.of(
                descriptor(DataType.FLOAT64, x.descriptor().shape(), true),
                descriptor(DataType.FLOAT32, Shape.of(3, 2), true),
                descriptor(DataType.FLOAT32, x.descriptor().shape(), false),
                new TensorDescriptor(DataType.FLOAT32, x.descriptor().shape(),
                        Optional.of(LayoutDescriptor.contiguous(x.descriptor().shape())), true))) {
            Tensor replacement = TensorFactory.create(bad);
            var failure = assertThrows(IllegalArgumentException.class,
                    () -> GraphCompilationPort.reconstructSource(source.graph(), List.of(replacement)));
            assertTrue(failure.getMessage().contains("leaves[0]"));
            assertThrows(IllegalArgumentException.class,
                    () -> new GraphCompilationPort.SourceCapture(source.graph(), List.of(replacement)));
        }
        Shape shape = Shape.of(2, 3);
        Tensor resolved = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {3, 1}, 1, true)), true));
        var viewSource = GraphCompilationPort.captureSource(List.of(resolved.neg()));
        Tensor wrongOffset = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {3, 1}, 0, true)), true));
        assertThrows(IllegalArgumentException.class,
                () -> GraphCompilationPort.reconstructSource(viewSource.graph(), List.of(wrongOffset)));
    }

    @Test
    void rejectsEveryInvalidStoredDescriptorFamilyPhaseAndGeometryIncludingDeadWork() {
        TensorDescriptor scalar = descriptor(DataType.FLOAT32, Shape.scalar(), false);
        var neg = operation(UnaryElementwiseKind.NEG);
        for (CompiledGraphModel invalid : List.of(
                single(neg, List.of(scalar), scalar, GraphPhase.BACKWARD),
                single(operation(CustomKind.ADD), List.of(scalar), scalar, GraphPhase.FORWARD),
                single(operation(UnaryElementwiseKind.ABS), List.of(scalar), scalar, GraphPhase.FORWARD),
                single(neg, List.of(scalar), descriptor(DataType.FLOAT64, Shape.scalar(), false), GraphPhase.FORWARD),
                single(neg, List.of(scalar), descriptor(DataType.FLOAT32, Shape.of(1), false), GraphPhase.FORWARD),
                single(neg, List.of(scalar), descriptor(DataType.FLOAT32, Shape.scalar(), true), GraphPhase.FORWARD),
                single(neg, List.of(descriptor(DataType.INT32, Shape.scalar(), false)),
                        descriptor(DataType.INT32, Shape.scalar(), false), GraphPhase.FORWARD),
                single(operation(BinaryArithmeticKind.ADD), List.of(scalar,
                        descriptor(DataType.INT32, Shape.scalar(), false)), scalar, GraphPhase.FORWARD),
                single(operation(BinaryArithmeticKind.MUL), List.of(
                        descriptor(DataType.FLOAT32, Shape.of(2), false),
                        descriptor(DataType.FLOAT32, Shape.of(3), false)), scalar, GraphPhase.FORWARD),
                single(operation(MatmulKind.MATMUL), List.of(scalar, scalar), scalar, GraphPhase.FORWARD),
                single(operation(MatmulKind.MATMUL), List.of(
                        descriptor(DataType.FLOAT32, Shape.of(2, 3), false),
                        descriptor(DataType.FLOAT32, Shape.of(4, 2), false)), scalar, GraphPhase.FORWARD),
                single(new Operation(AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(0, 1))),
                        List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3), false)),
                        descriptor(DataType.FLOAT32, Shape.of(2, 3), false), GraphPhase.FORWARD),
                single(new Operation(AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1, 0))),
                        List.of(scalar), scalar, GraphPhase.FORWARD))) {
            List<Tensor> replacements = invalid.inputs().stream().map(input -> TensorFactory.create(
                    invalid.values().stream().filter(value -> value.id().equals(input))
                            .findFirst().orElseThrow().descriptor())).toList();
            // Return an input instead: invalid work is dead but must still be checked.
            var dead = new CompiledGraphModel(invalid.values(), invalid.nodes(), invalid.inputs(),
                    List.of(invalid.inputs().getFirst()), invalid.nodePhases());
            Tensor marker = TensorFactory.create(scalar);
            assertThrows(IllegalArgumentException.class,
                    () -> GraphCompilationPort.reconstructSource(dead, replacements));
            assertEquals(marker.id().value() + 1, TensorFactory.create(scalar).id().value());
            assertThrows(IllegalArgumentException.class,
                    () -> new GraphCompilationPort.SourceCapture(dead, replacements));
        }
        for (Shape shape : List.of(Shape.ofDimensions(new DynamicDimension("N")), Shape.of(Long.MAX_VALUE, 2))) {
            TensorDescriptor bad = descriptor(DataType.FLOAT32, shape, false);
            // An unused input still belongs to complete source admission.
            var graph = new CompiledGraphModel(List.of(new GraphValue(id(3), scalar), new GraphValue(id(8), bad)),
                    List.of(), List.of(id(3), id(8)), List.of(id(3)), Map.of());
            assertThrows(IllegalArgumentException.class, () -> GraphCompilationPort.reconstructSource(
                    graph, List.of(TensorFactory.create(scalar), TensorFactory.create(bad))));
        }
        Tensor a = leaf(DataType.FLOAT32, Shape.of(Long.MAX_VALUE, 0), false);
        Tensor b = leaf(DataType.FLOAT32, Shape.of(0, 2), false);
        Tensor overflowingResult = a.matmul(b); // Inputs are empty; only the result count overflows.
        var raw = GraphCapture.capture(List.of(overflowingResult));
        var failure = assertThrows(IllegalArgumentException.class,
                () -> GraphCompilationPort.reconstructSource(raw, List.of(a, b)));
        assertInstanceOf(ArithmeticException.class, failure.getCause());
        assertThrows(IllegalArgumentException.class,
                () -> GraphCompilationPort.captureSource(List.of(overflowingResult)));
    }

    @Test
    void preflightsLateFailuresBeforeConstructingEvenAnEarlierValidProducer() {
        Tensor x = leaf(DataType.FLOAT32, Shape.scalar(), false);
        var raw = GraphCapture.capture(List.of(x.neg().abs()));
        Tensor marker = TensorFactory.create(x.descriptor());
        assertThrows(IllegalArgumentException.class,
                () -> GraphCompilationPort.reconstructSource(raw, List.of(x)));
        assertEquals(marker.id().value() + 1, TensorFactory.create(x.descriptor()).id().value());

        var values = new ArrayList<>(raw.values());
        GraphValue last = values.getLast();
        values.set(values.size() - 1, new GraphValue(last.id(),
                descriptor(DataType.FLOAT32, Shape.of(1), false)));
        var nodes = List.of(raw.nodes().getFirst(), new CompiledNode(raw.nodes().get(1).id(),
                operation(UnaryElementwiseKind.NEG), raw.nodes().get(1).inputs(), raw.nodes().get(1).outputs()));
        var badDescriptor = new CompiledGraphModel(values, nodes, raw.inputs(), raw.outputs(), raw.nodePhases());
        marker = TensorFactory.create(x.descriptor());
        assertThrows(IllegalArgumentException.class,
                () -> GraphCompilationPort.reconstructSource(badDescriptor, List.of(x)));
        assertEquals(marker.id().value() + 1, TensorFactory.create(x.descriptor()).id().value());
    }

    @Test
    void malformedAttributesArityAndTopologyRemainModelConstructorFailures() {
        assertThrows(IllegalArgumentException.class, () -> new Operation(
                BinaryArithmeticKind.ADD, new PermutationAttrs(List.of(1, 0))));
        assertThrows(IllegalArgumentException.class, () -> new CompiledNode(new NodeId(1),
                operation(UnaryElementwiseKind.NEG), List.of(id(0), id(1)), List.of(id(2))));
        // Full missing/duplicate producer, unknown/forward/self/cyclic reference, duplicate
        // output and missing/extra phase coverage is owned by CompiledGraphModelTest.
        TensorDescriptor d = descriptor(DataType.FLOAT32, Shape.scalar(), false);
        assertThrows(IllegalArgumentException.class, () -> new CompiledGraphModel(
                List.of(new GraphValue(id(0), d), new GraphValue(id(1), d)),
                List.of(new CompiledNode(new NodeId(1), operation(UnaryElementwiseKind.NEG),
                        List.of(id(1)), List.of(id(1)))), List.of(id(0)), List.of(id(1)),
                Map.of(new NodeId(1), GraphPhase.FORWARD)));
    }

    private static void assertRoundTrip(List<Tensor> outputs) {
        var source = GraphCompilationPort.captureSource(outputs);
        var rebuilt = GraphCompilationPort.reconstructSource(source.graph(), source.leaves());
        assertEquals(outputs.stream().map(Tensor::descriptor).toList(), rebuilt.stream().map(Tensor::descriptor).toList());
        assertEquals(source.graph(), GraphCompilationPort.captureSource(rebuilt).graph());
    }

    private static void assertEquivalentCompile(List<Tensor> original, List<Tensor> rebuilt,
            List<Tensor> originalTargets, List<Tensor> rebuiltTargets) {
        assertEquivalentCompile(original, rebuilt, originalTargets, rebuiltTargets,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR);
    }

    private static void assertEquivalentCompile(List<Tensor> original, List<Tensor> rebuilt,
            List<Tensor> originalTargets, List<Tensor> rebuiltTargets,
            FunctionalGradientRequest.DisconnectedPolicy policy) {
        Map<TensorId, TensorId> freshIds = new HashMap<>();
        for (int index = 0; index < originalTargets.size(); index++) {
            freshIds.put(originalTargets.get(index).id(), rebuiltTargets.get(index).id());
        }
        for (boolean backward : List.of(false, true)) {
            CompileArtifacts expected = compile(original, originalTargets, backward, policy);
            CompileArtifacts actual = compile(rebuilt, rebuiltTargets, backward, policy);
            assertEquals(expected.mode(), actual.mode());
            assertEquals(expected.graph(), actual.graph());
            assertEquals(expected.partitions(), actual.partitions());
            assertEquals(expected.memory(), actual.memory());
            assertEquals(expected.constants().constantSources(), actual.constants().constantSources());
            assertEquals(expected.constants().bindableInputBindings().stream()
                            .map(binding -> new CompileConstantPlan.BindableInput(
                                    freshIds.getOrDefault(binding.tensorId(), binding.tensorId()), binding.valueId()))
                            .toList(), actual.constants().bindableInputBindings());
            assertEquals(expected.derivatives().derivativeOrderByNode(), actual.derivatives().derivativeOrderByNode());
            assertEquals(expected.publication().gradientBindings().stream()
                            .map(binding -> new GradientPublicationBinding(binding.derivativeOrder(),
                                    binding.targetIndex(), freshIds.getOrDefault(binding.target(), binding.target()),
                                    binding.valueId())).toList(), actual.publication().gradientBindings());
            assertEquals(expected.publication().forwardBindings().stream().map(binding -> binding.valueId()).toList(),
                    actual.publication().forwardBindings().stream().map(binding -> binding.valueId()).toList());
            for (int index = 0; index < rebuilt.size(); index++) {
                assertEquals(rebuilt.get(index).id(), actual.publication().forwardBindings().get(index).tensorId());
            }
            assertEquals(expected.diagnostics().deferredConstraints(), actual.diagnostics().deferredConstraints());
        }
    }

    private static CompileArtifacts compile(List<Tensor> outputs, List<Tensor> targets, boolean backward) {
        return compile(outputs, targets, backward, FunctionalGradientRequest.DisconnectedPolicy.ERROR);
    }

    private static CompileArtifacts compile(List<Tensor> outputs, List<Tensor> targets, boolean backward,
            FunctionalGradientRequest.DisconnectedPolicy policy) {
        BackendId backend = new BackendId("cpu");
        BackendCapabilityProvider provider = new BackendCapabilityProvider() {
            @Override public BackendId backendId() { return backend; }
            @Override public boolean supports(OperationCapabilityQuery query) { return true; }
        };
        return GraphCompilationPort.compile(backward ? CompileMode.FORWARD_AND_BACKWARD : CompileMode.FORWARD_ONLY,
                outputs, backward ? Optional.of(FunctionalGradientTestSupport.request(outputs.getLast(), targets, policy))
                        : Optional.empty(), GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(), List.of(provider), List.of(new BackendAvailabilitySnapshot(
                        backend, Map.of(new BackendDeviceId(backend, "0"), DeviceClass.CPU))));
    }

    private static CompiledGraphModel single(Operation op, List<TensorDescriptor> inputs,
            TensorDescriptor output, GraphPhase phase) {
        List<GraphValue> values = new ArrayList<>();
        for (int index = 0; index < inputs.size(); index++) values.add(new GraphValue(id(index), inputs.get(index)));
        ValueId result = id(inputs.size());
        values.add(new GraphValue(result, output));
        List<ValueId> inputIds = values.subList(0, inputs.size()).stream().map(GraphValue::id).toList();
        return new CompiledGraphModel(values,
                List.of(new CompiledNode(new NodeId(7), op, inputIds, List.of(result))),
                inputIds, List.of(result), Map.of(new NodeId(7), phase));
    }

    private static ValueId id(long value) { return new ValueId(value); }
    private static Operation operation(OperationKind kind) { return new Operation(kind, NoOperationAttrs.INSTANCE); }
    private static TensorDescriptor descriptor(DataType type, Shape shape, boolean grad) {
        return new TensorDescriptor(type, shape, Optional.empty(), grad);
    }
    private static Tensor leaf(DataType type, Shape shape, boolean grad) {
        return TensorFactory.create(descriptor(type, shape, grad));
    }

    private enum CustomKind implements OperationKind {
        ADD;
        @Override public List<OperationSignature> signatures() {
            return List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));
        }
    }
}
