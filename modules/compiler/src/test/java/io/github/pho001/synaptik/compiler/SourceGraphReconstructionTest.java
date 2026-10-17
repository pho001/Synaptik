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
        Map<TensorId, TensorId> freshIds = new HashMap<>();
        for (int index = 0; index < originalTargets.size(); index++) {
            freshIds.put(originalTargets.get(index).id(), rebuiltTargets.get(index).id());
        }
        for (boolean backward : List.of(false, true)) {
            CompileArtifacts expected = compile(original, originalTargets, backward);
            CompileArtifacts actual = compile(rebuilt, rebuiltTargets, backward);
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
        BackendId backend = new BackendId("cpu");
        BackendCapabilityProvider provider = new BackendCapabilityProvider() {
            @Override public BackendId backendId() { return backend; }
            @Override public boolean supports(OperationCapabilityQuery query) { return true; }
        };
        return GraphCompilationPort.compile(backward ? CompileMode.FORWARD_AND_BACKWARD : CompileMode.FORWARD_ONLY,
                outputs, backward ? Optional.of(FunctionalGradientTestSupport.request(outputs.getLast(), targets))
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
