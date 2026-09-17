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
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
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

final class PublishedCompileTimeConstantDescriptorClosureTest {
    @Test
    void exposesOnlyOnePackagePrivateStatelessClosure() throws Exception {
        var method = PublishedCompileTimeConstantDescriptorClosure.class.getDeclaredMethod(
                "close", ValidatedGraph.class);
        var constructor = PublishedCompileTimeConstantDescriptorClosure.class
                .getDeclaredConstructor();

        assertAll(
                () -> assertTrue(Modifier.isFinal(
                        PublishedCompileTimeConstantDescriptorClosure.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        PublishedCompileTimeConstantDescriptorClosure.class.getModifiers())),
                () -> assertEquals(0,
                        PublishedCompileTimeConstantDescriptorClosure.class
                                .getDeclaredFields().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertSame(ValidatedGraph.class, method.getReturnType()),
                () -> assertEquals("validatedGraph", assertThrows(
                        NullPointerException.class,
                        () -> PublishedCompileTimeConstantDescriptorClosure.close(null))
                        .getMessage()));
    }

    @Test
    void closesScalarZeroElementAndOrdinaryStaticGeometry() {
        List<Shape> shapes = List.of(
                Shape.scalar(), Shape.of(0), Shape.of(2, 0, 3), Shape.of(2, 3, 4));
        ValidatedGraph result = closeInputs(
                shapes.stream().map(shape -> descriptor(DataType.INT32, shape)).toList(),
                List.of(
                        ScalarValue.int32(1),
                        ScalarValue.int32(2),
                        ScalarValue.int32(3),
                        ScalarValue.int32(4)));

        LayoutDescriptor scalar = result.graph().values().get(0).descriptor()
                .layout().orElseThrow();
        LayoutDescriptor rankOneEmpty = result.graph().values().get(1).descriptor()
                .layout().orElseThrow();
        LayoutDescriptor rankThreeEmpty = result.graph().values().get(2).descriptor()
                .layout().orElseThrow();
        LayoutDescriptor ordinary = result.graph().values().get(3).descriptor()
                .layout().orElseThrow();
        assertAll(
                () -> assertEquals(0, scalar.rank()),
                () -> assertArrayEquals(new long[0], scalar.strides()),
                () -> assertEquals(1, scalar.referencedElementSpan()),
                () -> assertArrayEquals(new long[] {1}, rankOneEmpty.strides()),
                () -> assertEquals(0, rankOneEmpty.referencedElementSpan()),
                () -> assertEquals(3, rankThreeEmpty.rank()),
                () -> assertArrayEquals(new long[] {0, 3, 1}, rankThreeEmpty.strides()),
                () -> assertEquals(0, rankThreeEmpty.referencedElementSpan()),
                () -> assertArrayEquals(new long[] {12, 4, 1}, ordinary.strides()),
                () -> assertEquals(24, ordinary.referencedElementSpan()));
        for (GraphValue value : result.graph().values()) {
            LayoutDescriptor layout = value.descriptor().layout().orElseThrow();
            assertAll(
                    () -> assertEquals(0, layout.storageOffset()),
                    () -> assertEquals(LayoutKind.DENSE_CONTIGUOUS, layout.kind()),
                    () -> assertFalse(layout.isView()));
        }
    }

    @Test
    void closesAllSixDataTypesAndPreservesExactSplatAndDescriptorSemantics() {
        List<DataType> dataTypes = List.of(DataType.values());
        List<ScalarValue> splats = List.of(
                ScalarValue.float64(-0.0d),
                ScalarValue.float32(-0.0f),
                ScalarValue.bfloat16Bits((short) 0x8000),
                ScalarValue.int32(Integer.MIN_VALUE),
                ScalarValue.int64(Long.MAX_VALUE),
                ScalarValue.bool(true));
        Shape shape = Shape.of(2, 3);

        ValidatedGraph result = closeInputs(
                dataTypes.stream().map(dataType -> descriptor(dataType, shape)).toList(), splats);

        for (int index = 0; index < dataTypes.size(); index++) {
            int current = index;
            GraphValue value = result.graph().values().get(index);
            assertAll(
                    () -> assertSame(dataTypes.get(current), value.descriptor().dataType()),
                    () -> assertSame(shape, value.descriptor().shape()),
                    () -> assertFalse(value.descriptor().requiresGrad()),
                    () -> assertEquals(LayoutDescriptor.contiguous(shape),
                            value.descriptor().layout().orElseThrow()),
                    () -> assertSame(splats.get(current),
                            result.constants().get(value.id()).value()));
        }
    }

    @Test
    void preservesEveryNonEligibleCategoryAndAlreadyResolvedIdentity() {
        TensorDescriptor dynamic = descriptor(
                DataType.INT32,
                Shape.ofDimensions(new DynamicDimension("N")));
        TensorDescriptor unresolved = descriptor(DataType.INT32, Shape.of(2));
        TensorDescriptor resolved = new TensorDescriptor(
                DataType.INT32,
                Shape.of(2),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(2))),
                false);
        CompiledNode consumer = node(0, 1, 5);
        CompiledGraphModel graph = graph(
                List.of(dynamic, unresolved, unresolved, unresolved, unresolved, unresolved,
                        resolved),
                List.of(consumer),
                List.of(0, 1, 2, 3, 4, 6),
                List.of(0, 1, 3, 5, 6),
                Map.of(consumer.id(), GraphPhase.FORWARD));
        var dynamicSplat = new CompileTimeConstantGraph.Splat(ScalarValue.int32(1));
        var consumedSplat = new CompileTimeConstantGraph.Splat(ScalarValue.int32(2));
        var unpublishedSplat = new CompileTimeConstantGraph.Splat(ScalarValue.int32(3));
        var resolvedSplat = new CompileTimeConstantGraph.Splat(ScalarValue.int32(4));
        ValidatedGraph source = validated(
                graph,
                Map.of(
                        id(0), dynamicSplat,
                        id(1), consumedSplat,
                        id(2), unpublishedSplat,
                        id(6), resolvedSplat),
                Map.of(id(3), new TensorId(30), id(4), new TensorId(40)),
                Map.of(consumer.id(), 0));

        ValidatedGraph result = PublishedCompileTimeConstantDescriptorClosure.close(source);

        assertSame(source, result);
        for (int index = 0; index < graph.values().size(); index++) {
            assertSame(graph.values().get(index), result.graph().values().get(index));
        }
        assertSame(resolved, result.graph().values().get(6).descriptor());
    }

    @Test
    void rebuildPreservesTopologyBoundariesSourceRolesAndDerivativeMetadata() {
        TensorDescriptor unresolved = descriptor(DataType.FLOAT32, Shape.of(2));
        CompiledNode backward = node(8, 1, 2);
        CompiledGraphModel graph = graph(
                List.of(unresolved, unresolved, unresolved),
                List.of(backward),
                List.of(0, 1),
                List.of(2, 0),
                Map.of(backward.id(), GraphPhase.BACKWARD));
        var splat = new CompileTimeConstantGraph.Splat(ScalarValue.float32(7.0f));
        TensorId bindableTensorId = new TensorId(91);
        ValidatedGraph source = validated(
                graph,
                Map.of(id(0), splat),
                Map.of(id(1), bindableTensorId),
                Map.of(backward.id(), 2));

        ValidatedGraph result = PublishedCompileTimeConstantDescriptorClosure.close(source);

        assertAll(
                () -> assertNotSame(source, result),
                () -> assertNotSame(graph, result.graph()),
                () -> assertEquals(graph.inputs(), result.graph().inputs()),
                () -> assertEquals(graph.outputs(), result.graph().outputs()),
                () -> assertEquals(graph.nodePhases(), result.graph().nodePhases()),
                () -> assertSame(backward, result.graph().nodes().getFirst()),
                () -> assertSame(graph.values().get(1), result.graph().values().get(1)),
                () -> assertSame(graph.values().get(2), result.graph().values().get(2)),
                () -> assertSame(splat, result.constants().get(id(0))),
                () -> assertSame(bindableTensorId,
                        result.constantGraph().bindableTensorIds().get(id(1))),
                () -> assertSame(result.graph(), result.derivatives().graph()),
                () -> assertEquals(Map.of(backward.id(), 2),
                        result.derivatives().derivativeOrderByNode()),
                () -> assertTrue(result.constraints().isEmpty()));
    }

    @Test
    void failsClosedOnCanonicalLayoutOverflowWithValueContextAndCause() {
        ValueId valueId = new ValueId(37);
        TensorDescriptor descriptor = descriptor(
                DataType.INT64, Shape.of(2, Long.MAX_VALUE, 2));
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(new GraphValue(valueId, descriptor)),
                List.of(),
                List.of(valueId),
                List.of(valueId),
                Map.of());
        ValidatedGraph source = validated(
                graph,
                Map.of(valueId, new CompileTimeConstantGraph.Splat(ScalarValue.int64(1))),
                Map.of(),
                Map.of());

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> PublishedCompileTimeConstantDescriptorClosure.close(source));

        assertAll(
                () -> assertEquals(
                        "cannot close published compile-time constant descriptor for " + valueId,
                        failure.getMessage()),
                () -> assertTrue(failure.getCause() instanceof ArithmeticException));
    }

    private static ValidatedGraph closeInputs(
            List<TensorDescriptor> descriptors, List<ScalarValue> scalarValues) {
        List<GraphValue> values = new ArrayList<>();
        List<ValueId> boundaries = new ArrayList<>();
        Map<ValueId, CompileTimeConstantGraph.Splat> constants = new LinkedHashMap<>();
        for (int index = 0; index < descriptors.size(); index++) {
            ValueId id = id(index);
            values.add(new GraphValue(id, descriptors.get(index)));
            boundaries.add(id);
            constants.put(id, new CompileTimeConstantGraph.Splat(scalarValues.get(index)));
        }
        CompiledGraphModel graph = new CompiledGraphModel(
                values, List.of(), boundaries, boundaries, Map.of());
        return PublishedCompileTimeConstantDescriptorClosure.close(
                validated(graph, constants, Map.of(), Map.of()));
    }

    private static ValidatedGraph validated(
            CompiledGraphModel graph,
            Map<ValueId, CompileTimeConstantGraph.Splat> constants,
            Map<ValueId, TensorId> bindableTensorIds,
            Map<NodeId, Integer> derivativeOrders) {
        return new ValidatedGraph(
                new CompileTimeConstantGraph(graph, constants, bindableTensorIds),
                List.of(),
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
                inputs.stream().map(PublishedCompileTimeConstantDescriptorClosureTest::id).toList(),
                outputs.stream().map(PublishedCompileTimeConstantDescriptorClosureTest::id).toList(),
                phases);
    }

    private static CompiledNode node(long nodeId, long input, long output) {
        return new CompiledNode(
                new NodeId(nodeId),
                new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                List.of(id(input)),
                List.of(id(output)));
    }

    private static TensorDescriptor descriptor(DataType dataType, Shape shape) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), false);
    }

    private static ValueId id(long value) {
        return new ValueId(value);
    }
}
