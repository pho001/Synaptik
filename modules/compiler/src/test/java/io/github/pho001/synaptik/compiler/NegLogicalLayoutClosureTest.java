package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class NegLogicalLayoutClosureTest {
    @Test
    void exposesOnlyOnePackagePrivateStatelessClosure() throws Exception {
        var method = NegLogicalLayoutClosure.class.getDeclaredMethod(
                "close", ValidatedGraph.class);
        var constructor = NegLogicalLayoutClosure.class.getDeclaredConstructor();

        assertAll(
                () -> assertTrue(Modifier.isFinal(
                        NegLogicalLayoutClosure.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        NegLogicalLayoutClosure.class.getModifiers())),
                () -> assertEquals(0, NegLogicalLayoutClosure.class.getDeclaredFields().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertSame(ValidatedGraph.class, method.getReturnType()),
                () -> assertEquals("validatedGraph", assertThrows(
                        NullPointerException.class,
                        () -> NegLogicalLayoutClosure.close(null)).getMessage()));
    }

    @Test
    void closesStaticNegOutputsAndOneSharedDirectSplatInput() {
        Shape shape = Shape.of(2, 3);
        CompiledNode first = negNode(10, 0, 1);
        CompiledNode second = negNode(11, 0, 2);
        CompiledGraphModel graph = graph(
                List.of(descriptor(shape), descriptor(shape), descriptor(shape)),
                List.of(first, second),
                List.of(0),
                List.of(1, 2),
                Map.of(first.id(), GraphPhase.FORWARD, second.id(), GraphPhase.FORWARD));
        var splat = new CompileTimeConstantGraph.Splat(ScalarValue.float32(-0.0f));
        ValidatedGraph source = validated(
                graph,
                Map.of(id(0), splat),
                Map.of(),
                derivativeOrders(first, second),
                List.of());

        ValidatedGraph result = NegLogicalLayoutClosure.close(source);

        assertAll(
                () -> assertNotSame(source, result),
                () -> assertEquals(LayoutDescriptor.contiguous(shape),
                        value(result, 0).descriptor().layout().orElseThrow()),
                () -> assertEquals(LayoutDescriptor.contiguous(shape),
                        value(result, 1).descriptor().layout().orElseThrow()),
                () -> assertEquals(LayoutDescriptor.contiguous(shape),
                        value(result, 2).descriptor().layout().orElseThrow()),
                () -> assertSame(splat, result.constants().get(id(0))),
                () -> assertSame(first, result.graph().nodes().get(0)),
                () -> assertSame(second, result.graph().nodes().get(1)));
    }

    @Test
    void closesScalarAndZeroExtentLogicalGeometry() {
        CompiledNode scalarNeg = negNode(20, 0, 1);
        CompiledNode emptyNeg = negNode(21, 2, 3);
        CompiledGraphModel graph = graph(
                List.of(
                        descriptor(Shape.scalar()), descriptor(Shape.scalar()),
                        descriptor(Shape.of(2, 0, 3)), descriptor(Shape.of(2, 0, 3))),
                List.of(scalarNeg, emptyNeg),
                List.of(0, 2),
                List.of(1, 3),
                Map.of(scalarNeg.id(), GraphPhase.FORWARD, emptyNeg.id(), GraphPhase.FORWARD));
        ValidatedGraph source = validated(
                graph,
                Map.of(
                        id(0), new CompileTimeConstantGraph.Splat(ScalarValue.float32(1.0f)),
                        id(2), new CompileTimeConstantGraph.Splat(ScalarValue.float32(2.0f))),
                Map.of(),
                derivativeOrders(scalarNeg, emptyNeg),
                List.of());

        ValidatedGraph result = NegLogicalLayoutClosure.close(source);
        LayoutDescriptor scalar = value(result, 1).descriptor().layout().orElseThrow();
        LayoutDescriptor empty = value(result, 3).descriptor().layout().orElseThrow();

        assertAll(
                () -> assertEquals(0, scalar.rank()),
                () -> assertArrayEquals(new long[0], scalar.strides()),
                () -> assertEquals(1, scalar.referencedElementSpan()),
                () -> assertArrayEquals(new long[] {0, 3, 1}, empty.strides()),
                () -> assertEquals(0, empty.referencedElementSpan()));
    }

    @Test
    void preservesBindableIndirectDynamicResolvedNonNegAndUnrelatedValues() {
        Shape staticShape = Shape.of(2);
        Shape dynamicShape = Shape.ofDimensions(new DynamicDimension("N"));
        TensorDescriptor resolved = resolvedDescriptor(staticShape);
        CompiledNode bindableNeg = negNode(30, 0, 1);
        CompiledNode abs = node(31, UnaryElementwiseKind.ABS, 2, 3);
        CompiledNode indirectNeg = negNode(32, 3, 4);
        CompiledNode dynamicNeg = negNode(33, 5, 6);
        CompiledNode resolvedNeg = negNode(34, 7, 8);
        CompiledGraphModel graph = graph(
                List.of(
                        descriptor(staticShape), descriptor(staticShape),
                        descriptor(staticShape), descriptor(staticShape), descriptor(staticShape),
                        descriptor(dynamicShape), descriptor(dynamicShape),
                        resolved, resolved,
                        descriptor(staticShape)),
                List.of(bindableNeg, abs, indirectNeg, dynamicNeg, resolvedNeg),
                List.of(0, 2, 5, 7, 9),
                List.of(1, 4, 6, 8, 9),
                phases(bindableNeg, abs, indirectNeg, dynamicNeg, resolvedNeg));
        var indirectSplat = new CompileTimeConstantGraph.Splat(ScalarValue.float32(2.0f));
        var dynamicSplat = new CompileTimeConstantGraph.Splat(ScalarValue.float32(3.0f));
        var resolvedSplat = new CompileTimeConstantGraph.Splat(ScalarValue.float32(4.0f));
        var unrelatedSplat = new CompileTimeConstantGraph.Splat(ScalarValue.float32(5.0f));
        TensorId bindableId = new TensorId(900);
        ValidatedGraph source = validated(
                graph,
                Map.of(
                        id(2), indirectSplat,
                        id(5), dynamicSplat,
                        id(7), resolvedSplat,
                        id(9), unrelatedSplat),
                Map.of(id(0), bindableId),
                derivativeOrders(bindableNeg, abs, indirectNeg, dynamicNeg, resolvedNeg),
                List.of());

        ValidatedGraph result = NegLogicalLayoutClosure.close(source);

        assertAll(
                () -> assertSame(graph.values().get(0), result.graph().values().get(0)),
                () -> assertTrue(value(result, 0).descriptor().layout().isEmpty()),
                () -> assertTrue(value(result, 1).descriptor().layout().isPresent()),
                () -> assertSame(graph.values().get(2), result.graph().values().get(2)),
                () -> assertTrue(value(result, 2).descriptor().layout().isEmpty()),
                () -> assertSame(graph.values().get(3), result.graph().values().get(3)),
                () -> assertTrue(value(result, 4).descriptor().layout().isPresent()),
                () -> assertSame(graph.values().get(5), result.graph().values().get(5)),
                () -> assertSame(graph.values().get(6), result.graph().values().get(6)),
                () -> assertSame(resolved, value(result, 7).descriptor()),
                () -> assertSame(resolved, value(result, 8).descriptor()),
                () -> assertSame(graph.values().get(9), result.graph().values().get(9)),
                () -> assertSame(bindableId,
                        result.constantGraph().bindableTensorIds().get(id(0))));
    }

    @Test
    void returnsExactInputWhenNothingQualifies() {
        Shape dynamicShape = Shape.ofDimensions(new DynamicDimension("N"));
        CompiledNode neg = negNode(40, 0, 1);
        CompiledNode abs = node(41, UnaryElementwiseKind.ABS, 2, 3);
        CompiledGraphModel graph = graph(
                List.of(
                        descriptor(dynamicShape), descriptor(dynamicShape),
                        descriptor(Shape.of(2)), descriptor(Shape.of(2))),
                List.of(neg, abs),
                List.of(0, 2),
                List.of(1, 3),
                phases(neg, abs));
        ValidatedGraph source = validated(
                graph,
                Map.of(),
                Map.of(id(0), new TensorId(1), id(2), new TensorId(2)),
                derivativeOrders(neg, abs),
                List.of());

        assertSame(source, NegLogicalLayoutClosure.close(source));
    }

    @Test
    void rejectsNonExactNegAttributesBeforeGraphConstruction() {
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> new Operation(UnaryElementwiseKind.NEG, WrongNegAttrs.INSTANCE));

        assertTrue(failure.getMessage().contains("does not accept attributes type"));
    }

    @Test
    void rejectsMalformedNegCardinalityBeforeGraphConstruction() {
        Operation neg = new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE);

        assertAll(
                () -> assertEquals(
                        "input count 0 is outside accepted range [1, 1]",
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new CompiledNode(
                                        new NodeId(42), neg, List.of(), List.of(id(1))))
                                .getMessage()),
                () -> assertEquals(
                        "input count 2 is outside accepted range [1, 1]",
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new CompiledNode(
                                        new NodeId(43), neg,
                                        List.of(id(0), id(1)), List.of(id(2))))
                                .getMessage()),
                () -> assertEquals(
                        "output count 2 is outside accepted range [1, 1]",
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new CompiledNode(
                                        new NodeId(44), neg, List.of(id(0)),
                                        List.of(id(1), id(2))))
                                .getMessage()));
    }

    @Test
    void preservesDescriptorsAndExactGraphForNearestLegalGuardVariants() {
        Shape shape = Shape.of(2);
        CompiledNode wrongAttrs = new CompiledNode(
                new NodeId(45),
                new Operation(NearNegKind.NEG, WrongNegAttrs.INSTANCE),
                List.of(id(0)),
                List.of(id(1)));
        CompiledNode wrongCardinality = new CompiledNode(
                new NodeId(46),
                new Operation(NearNegKind.NEG, NoOperationAttrs.INSTANCE),
                List.of(id(2), id(3)),
                List.of(id(4), id(5)));
        CompiledGraphModel graph = graph(
                List.of(
                        descriptor(shape), descriptor(shape), descriptor(shape),
                        descriptor(shape), descriptor(shape), descriptor(shape)),
                List.of(wrongAttrs, wrongCardinality),
                List.of(0, 2, 3),
                List.of(1, 4, 5),
                phases(wrongAttrs, wrongCardinality));
        ValidatedGraph source = validated(
                graph,
                Map.of(
                        id(0), new CompileTimeConstantGraph.Splat(ScalarValue.float32(1.0f)),
                        id(2), new CompileTimeConstantGraph.Splat(ScalarValue.float32(2.0f)),
                        id(3), new CompileTimeConstantGraph.Splat(ScalarValue.float32(3.0f))),
                Map.of(),
                derivativeOrders(wrongAttrs, wrongCardinality),
                List.of());

        ValidatedGraph result = NegLogicalLayoutClosure.close(source);

        assertSame(source, result);
        for (int index = 0; index < graph.values().size(); index++) {
            assertSame(graph.values().get(index), result.graph().values().get(index));
            assertTrue(result.graph().values().get(index).descriptor().layout().isEmpty());
        }
    }

    @Test
    void rebuildPreservesEveryGraphSidecarAndIdentity() {
        CompiledNode backward = negNode(50, 0, 1);
        CompiledGraphModel graph = graph(
                List.of(descriptor(Shape.of(2)), descriptor(Shape.of(2))),
                List.of(backward),
                List.of(0),
                List.of(1),
                Map.of(backward.id(), GraphPhase.BACKWARD));
        DeferredGraphConstraint constraint = new DeferredGraphConstraint(
                backward.id(), "preserved", new DimensionAtLeast(new DynamicDimension("N"), 1));
        var splat = new CompileTimeConstantGraph.Splat(ScalarValue.float32(7.0f));
        ValidatedGraph source = validated(
                graph,
                Map.of(id(0), splat),
                Map.of(),
                Map.of(backward.id(), 2),
                List.of(constraint));

        ValidatedGraph result = NegLogicalLayoutClosure.close(source);

        assertAll(
                () -> assertEquals(graph.inputs(), result.graph().inputs()),
                () -> assertEquals(graph.outputs(), result.graph().outputs()),
                () -> assertEquals(graph.nodePhases(), result.graph().nodePhases()),
                () -> assertSame(backward, result.graph().nodes().getFirst()),
                () -> assertSame(backward.operation(),
                        result.graph().nodes().getFirst().operation()),
                () -> assertSame(splat, result.constants().get(id(0))),
                () -> assertSame(constraint, result.constraints().getFirst()),
                () -> assertEquals(Map.of(backward.id(), 2),
                        result.derivatives().derivativeOrderByNode()),
                () -> assertSame(result.graph(), result.derivatives().graph()));
    }

    @Test
    void failsInputClosureWithNodeValueRoleContextAndArithmeticCause() {
        Shape overflowing = Shape.of(2, Long.MAX_VALUE, 2);
        CompiledNode neg = negNode(60, 0, 1);
        CompiledGraphModel graph = graph(
                List.of(descriptor(overflowing), descriptor(overflowing)),
                List.of(neg), List.of(0), List.of(1), Map.of(neg.id(), GraphPhase.FORWARD));
        ValidatedGraph source = validated(
                graph,
                Map.of(id(0), new CompileTimeConstantGraph.Splat(ScalarValue.float32(1.0f))),
                Map.of(), Map.of(neg.id(), 0), List.of());

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> NegLogicalLayoutClosure.close(source));

        assertAll(
                () -> assertEquals(
                        "cannot close NEG logical layout for " + neg.id() + " input " + id(0),
                        failure.getMessage()),
                () -> assertTrue(failure.getCause() instanceof ArithmeticException));
    }

    @Test
    void failsOutputClosureWithNodeValueRoleContextAndArithmeticCause() {
        Shape overflowing = Shape.of(2, Long.MAX_VALUE, 2);
        CompiledNode neg = negNode(61, 0, 1);
        CompiledGraphModel graph = graph(
                List.of(descriptor(overflowing), descriptor(overflowing)),
                List.of(neg), List.of(0), List.of(1), Map.of(neg.id(), GraphPhase.FORWARD));
        ValidatedGraph source = validated(
                graph, Map.of(), Map.of(id(0), new TensorId(61)), Map.of(neg.id(), 0), List.of());

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> NegLogicalLayoutClosure.close(source));

        assertAll(
                () -> assertEquals(
                        "cannot close NEG logical layout for " + neg.id() + " output " + id(1),
                        failure.getMessage()),
                () -> assertTrue(failure.getCause() instanceof ArithmeticException));
    }

    private static ValidatedGraph validated(
            CompiledGraphModel graph,
            Map<ValueId, CompileTimeConstantGraph.Splat> constants,
            Map<ValueId, TensorId> bindableTensorIds,
            Map<NodeId, Integer> derivativeOrders,
            List<DeferredGraphConstraint> constraints) {
        return new ValidatedGraph(
                new CompileTimeConstantGraph(graph, constants, bindableTensorIds),
                constraints,
                new DerivativeGraphMetadata(graph, derivativeOrders));
    }

    private static CompiledGraphModel graph(
            List<TensorDescriptor> descriptors,
            List<CompiledNode> nodes,
            List<Integer> inputs,
            List<Integer> outputs,
            Map<NodeId, GraphPhase> phases) {
        List<GraphValue> values = new ArrayList<>();
        for (int index = 0; index < descriptors.size(); index++) {
            values.add(new GraphValue(id(index), descriptors.get(index)));
        }
        return new CompiledGraphModel(
                values,
                nodes,
                inputs.stream().map(NegLogicalLayoutClosureTest::id).toList(),
                outputs.stream().map(NegLogicalLayoutClosureTest::id).toList(),
                phases);
    }

    private static CompiledNode negNode(long nodeId, long input, long output) {
        return node(nodeId, UnaryElementwiseKind.NEG, input, output);
    }

    private static CompiledNode node(
            long nodeId, UnaryElementwiseKind kind, long input, long output) {
        return new CompiledNode(
                new NodeId(nodeId),
                new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(id(input)),
                List.of(id(output)));
    }

    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape, Optional.empty(), false);
    }

    private static TensorDescriptor resolvedDescriptor(Shape shape) {
        return new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
    }

    private static GraphValue value(ValidatedGraph graph, long id) {
        return graph.graph().values().stream()
                .filter(value -> value.id().equals(id(id)))
                .findFirst()
                .orElseThrow();
    }

    private static Map<NodeId, GraphPhase> phases(CompiledNode... nodes) {
        Map<NodeId, GraphPhase> phases = new LinkedHashMap<>();
        for (CompiledNode node : nodes) {
            phases.put(node.id(), GraphPhase.FORWARD);
        }
        return phases;
    }

    private static Map<NodeId, Integer> derivativeOrders(CompiledNode... nodes) {
        Map<NodeId, Integer> orders = new LinkedHashMap<>();
        for (CompiledNode node : nodes) {
            orders.put(node.id(), 0);
        }
        return orders;
    }

    private static ValueId id(long value) {
        return new ValueId(value);
    }

    private enum WrongNegAttrs implements OperationAttrs {
        INSTANCE
    }

    private enum NearNegKind implements OperationKind {
        NEG;

        private static final List<OperationSignature> SIGNATURES = List.of(
                OperationSignature.fixed(WrongNegAttrs.class, 1, 1),
                OperationSignature.fixed(NoOperationAttrs.class, 2, 2));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
