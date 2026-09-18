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
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class ConvolutionLogicalLayoutClosureTest {
    @Test
    void exposesOnlyOnePackagePrivateStatelessClosure() throws Exception {
        var method = ConvolutionLogicalLayoutClosure.class.getDeclaredMethod(
                "close", ValidatedGraph.class);
        var constructor = ConvolutionLogicalLayoutClosure.class.getDeclaredConstructor();

        assertAll(
                () -> assertTrue(Modifier.isFinal(
                        ConvolutionLogicalLayoutClosure.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        ConvolutionLogicalLayoutClosure.class.getModifiers())),
                () -> assertEquals(0,
                        ConvolutionLogicalLayoutClosure.class.getDeclaredFields().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertSame(ValidatedGraph.class, method.getReturnType()),
                () -> assertEquals("validatedGraph", assertThrows(
                        NullPointerException.class,
                        () -> ConvolutionLogicalLayoutClosure.close(null)).getMessage()));
    }

    @Test
    void closesStaticConv2dAndConv3dToCanonicalLayouts() {
        Shape conv2dShape = Shape.of(2, 3, 4, 5);
        Shape conv3dShape = Shape.of(2, 3, 4, 5, 6);
        CompiledNode conv2d = node(10, Conv2dKind.CONV2D, Conv2dAttrs.defaults(),
                List.of(0, 1), List.of(2));
        CompiledNode conv3d = node(11, Conv3dKind.CONV3D, Conv3dAttrs.defaults(),
                List.of(3, 4), List.of(5));
        ValidatedGraph source = validated(
                graph(
                        List.of(
                                descriptor(Shape.of(1)), descriptor(Shape.of(1)),
                                descriptor(conv2dShape), descriptor(Shape.of(1)),
                                descriptor(Shape.of(1)), descriptor(conv3dShape)),
                        List.of(conv2d, conv3d),
                        List.of(0, 1, 3, 4),
                        List.of(2, 5)),
                List.of());

        ValidatedGraph result = ConvolutionLogicalLayoutClosure.close(source);

        assertAll(
                () -> assertNotSame(source, result),
                () -> assertEquals(LayoutDescriptor.contiguous(conv2dShape),
                        result.graph().values().get(2).descriptor().layout().orElseThrow()),
                () -> assertEquals(LayoutDescriptor.contiguous(conv3dShape),
                        result.graph().values().get(5).descriptor().layout().orElseThrow()),
                () -> assertSame(conv2d, result.graph().nodes().get(0)),
                () -> assertSame(conv3d, result.graph().nodes().get(1)));
    }

    @Test
    void closesOnlyTheDirectAxisTwoSqueezeOfANewlyClosedConv2d() {
        Shape convShape = Shape.of(1, 2, 1, 3);
        Shape visibleShape = Shape.of(1, 2, 3);
        CompiledNode conv = node(20, Conv2dKind.CONV2D, Conv2dAttrs.defaults(),
                List.of(0, 1), List.of(2));
        CompiledNode squeeze = node(21, AxisTransformKind.SQUEEZE,
                new AxisTransformAttrs(2), List.of(2), List.of(3));
        ValidatedGraph source = validated(
                graph(
                        List.of(
                                descriptor(Shape.of(1)), descriptor(Shape.of(1)),
                                descriptor(convShape), descriptor(visibleShape)),
                        List.of(conv, squeeze),
                        List.of(0, 1),
                        List.of(2, 3)),
                List.of());

        ValidatedGraph result = ConvolutionLogicalLayoutClosure.close(source);

        LayoutDescriptor convLayout = result.graph().values().get(2).descriptor()
                .layout().orElseThrow();
        LayoutDescriptor visibleLayout = result.graph().values().get(3).descriptor()
                .layout().orElseThrow();
        assertAll(
                () -> assertArrayEquals(new long[] {6, 3, 3, 1}, convLayout.strides()),
                () -> assertFalse(convLayout.isView()),
                () -> assertArrayEquals(new long[] {6, 3, 1}, visibleLayout.strides()),
                () -> assertEquals(convLayout.storageOffset(), visibleLayout.storageOffset()),
                () -> assertTrue(visibleLayout.isView()),
                () -> assertSame(conv, result.graph().nodes().get(0)),
                () -> assertSame(squeeze, result.graph().nodes().get(1)));
    }

    @Test
    void preservesDynamicAlreadyResolvedAndIneligibleSqueezeDescriptors() {
        Shape dynamicConvShape = Shape.ofDimensions(
                new StaticDimension(1), new StaticDimension(2),
                new StaticDimension(1), new DynamicDimension("W"));
        Shape resolvedConvShape = Shape.of(1, 2, 1, 3);
        TensorDescriptor resolvedConv = resolvedDescriptor(resolvedConvShape);
        TensorDescriptor unresolvedVisible = descriptor(Shape.of(1, 2, 3));
        CompiledNode dynamicConv = node(30, Conv2dKind.CONV2D, Conv2dAttrs.defaults(),
                List.of(0, 1), List.of(2));
        CompiledNode resolvedConvNode = node(31, Conv2dKind.CONV2D, Conv2dAttrs.defaults(),
                List.of(3, 4), List.of(5));
        CompiledNode squeezeNotNew = node(32, AxisTransformKind.SQUEEZE,
                new AxisTransformAttrs(2), List.of(5), List.of(6));
        CompiledNode wrongAxis = node(33, AxisTransformKind.SQUEEZE,
                new AxisTransformAttrs(1), List.of(7), List.of(8));
        ValidatedGraph source = validated(
                graph(
                        List.of(
                                descriptor(Shape.of(1)), descriptor(Shape.of(1)),
                                descriptor(dynamicConvShape), descriptor(Shape.of(1)),
                                descriptor(Shape.of(1)), resolvedConv, unresolvedVisible,
                                descriptor(Shape.of(1, 1, 2, 3)), descriptor(Shape.of(1, 2, 3))),
                        List.of(dynamicConv, resolvedConvNode, squeezeNotNew, wrongAxis),
                        List.of(0, 1, 3, 4, 7),
                        List.of(2, 6, 8)),
                List.of());

        ValidatedGraph result = ConvolutionLogicalLayoutClosure.close(source);

        assertSame(source, result);
        assertSame(resolvedConv, result.graph().values().get(5).descriptor());
        assertSame(unresolvedVisible, result.graph().values().get(6).descriptor());
    }

    @Test
    void returnsExactInputForNonConvolutionGraph() {
        CompiledNode neg = node(40, UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE,
                List.of(0), List.of(1));
        ValidatedGraph source = validated(
                graph(
                        List.of(descriptor(Shape.of(2)), descriptor(Shape.of(2))),
                        List.of(neg), List.of(0), List.of(1)),
                List.of());

        assertSame(source, ConvolutionLogicalLayoutClosure.close(source));
    }

    @Test
    void leavesPartiallyDynamicConv3dUnresolved() {
        Shape outputShape = Shape.ofDimensions(
                new StaticDimension(1), new StaticDimension(2),
                new DynamicDimension("D"), new StaticDimension(3), new StaticDimension(4));
        CompiledNode conv = node(41, Conv3dKind.CONV3D, Conv3dAttrs.defaults(),
                List.of(0, 1), List.of(2));
        ValidatedGraph source = validated(
                graph(
                        List.of(descriptor(Shape.of(1)), descriptor(Shape.of(1)),
                                descriptor(outputShape)),
                        List.of(conv), List.of(0, 1), List.of(2)),
                List.of());

        assertSame(source, ConvolutionLogicalLayoutClosure.close(source));
        assertTrue(source.graph().values().get(2).descriptor().layout().isEmpty());
    }

    @Test
    void preservesGraphIdentityStateAndUnchangedValueReferences() {
        Shape outputShape = Shape.of(1, 2, 2, 2);
        CompiledNode conv = node(50, Conv2dKind.CONV2D, Conv2dAttrs.defaults(),
                List.of(0, 1), List.of(2));
        CompiledGraphModel graph = graph(
                List.of(descriptor(Shape.of(1)), descriptor(Shape.of(1)),
                        descriptor(outputShape)),
                List.of(conv), List.of(0, 1), List.of(2));
        DeferredGraphConstraint constraint = new DeferredGraphConstraint(
                conv.id(), "preserved", new DimensionAtLeast(new DynamicDimension("N"), 1));
        ValidatedGraph source = validated(graph, List.of(constraint));

        ValidatedGraph result = ConvolutionLogicalLayoutClosure.close(source);

        assertAll(
                () -> assertEquals(graph.inputs(), result.graph().inputs()),
                () -> assertEquals(graph.outputs(), result.graph().outputs()),
                () -> assertEquals(graph.nodePhases(), result.graph().nodePhases()),
                () -> assertSame(conv, result.graph().nodes().getFirst()),
                () -> assertSame(graph.values().get(0), result.graph().values().get(0)),
                () -> assertSame(graph.values().get(1), result.graph().values().get(1)),
                () -> assertSame(source.constantGraph().bindableTensorIds(),
                        result.constantGraph().bindableTensorIds()),
                () -> assertSame(source.constraints().getFirst(), result.constraints().getFirst()),
                () -> assertEquals(source.derivatives().derivativeOrderByNode(),
                        result.derivatives().derivativeOrderByNode()),
                () -> assertSame(result.graph(), result.derivatives().graph()));
    }

    @Test
    void failsCanonicalClosureWithOperationNodeValueAndArithmeticCause() {
        CompiledNode conv = node(61, Conv3dKind.CONV3D, Conv3dAttrs.defaults(),
                List.of(0, 1), List.of(2));
        ValidatedGraph source = validated(
                graph(
                        List.of(descriptor(Shape.of(1)), descriptor(Shape.of(1)),
                                descriptor(Shape.of(2, Long.MAX_VALUE, 1, 1, 2))),
                        List.of(conv), List.of(0, 1), List.of(2)),
                List.of());

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> ConvolutionLogicalLayoutClosure.close(source));

        assertAll(
                () -> assertEquals(
                        "cannot close CONV3D logical layout for " + conv.id()
                                + " output " + id(2),
                        failure.getMessage()),
                () -> assertTrue(failure.getCause() instanceof ArithmeticException));
    }

    private static ValidatedGraph validated(
            CompiledGraphModel graph, List<DeferredGraphConstraint> constraints) {
        Map<ValueId, TensorId> bindable = new LinkedHashMap<>();
        for (int index = 0; index < graph.inputs().size(); index++) {
            bindable.put(graph.inputs().get(index), new TensorId(100 + index));
        }
        Map<NodeId, Integer> derivativeOrders = new LinkedHashMap<>();
        graph.nodes().forEach(node -> derivativeOrders.put(node.id(), 0));
        return new ValidatedGraph(
                new CompileTimeConstantGraph(graph, Map.of(), bindable),
                constraints,
                new DerivativeGraphMetadata(graph, derivativeOrders));
    }

    private static CompiledGraphModel graph(
            List<TensorDescriptor> descriptors,
            List<CompiledNode> nodes,
            List<Integer> inputs,
            List<Integer> outputs) {
        List<GraphValue> values = new ArrayList<>();
        for (int index = 0; index < descriptors.size(); index++) {
            values.add(new GraphValue(id(index), descriptors.get(index)));
        }
        Map<NodeId, GraphPhase> phases = new LinkedHashMap<>();
        nodes.forEach(node -> phases.put(node.id(), GraphPhase.FORWARD));
        return new CompiledGraphModel(
                values,
                nodes,
                inputs.stream().map(ConvolutionLogicalLayoutClosureTest::id).toList(),
                outputs.stream().map(ConvolutionLogicalLayoutClosureTest::id).toList(),
                phases);
    }

    private static CompiledNode node(
            long nodeId,
            io.github.pho001.synaptik.model.operation.OperationKind kind,
            io.github.pho001.synaptik.model.operation.OperationAttrs attrs,
            List<Integer> inputs,
            List<Integer> outputs) {
        return new CompiledNode(
                new NodeId(nodeId),
                new Operation(kind, attrs),
                inputs.stream().map(ConvolutionLogicalLayoutClosureTest::id).toList(),
                outputs.stream().map(ConvolutionLogicalLayoutClosureTest::id).toList());
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

    private static ValueId id(long value) {
        return new ValueId(value);
    }
}
