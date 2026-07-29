package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertAll;
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
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class GraphCanonicalizationTest {
    @Test
    void exposesOnlyThePackagePrivateStatelessCanonicalizationContract() throws Exception {
        var method = GraphCanonicalization.class.getDeclaredMethod(
                "canonicalize", CompiledGraphModel.class);
        var constructor = GraphCanonicalization.class.getDeclaredConstructor();

        assertAll(
                () -> assertTrue(Modifier.isFinal(GraphCanonicalization.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(GraphCanonicalization.class.getModifiers())),
                () -> assertEquals(0, GraphCanonicalization.class.getDeclaredFields().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertSame(CompiledGraphModel.class, method.getReturnType()),
                () -> assertEquals(2, Arrays.stream(GraphCanonicalization.class.getDeclaredMethods())
                        .filter(declared -> !declared.isSynthetic())
                        .filter(declared -> !Modifier.isPrivate(declared.getModifiers()))
                        .count()));
    }

    @Test
    void rejectsNullGraphWithSpecifiedMessage() {
        assertEquals("graph", assertThrows(NullPointerException.class,
                () -> GraphCanonicalization.canonicalize(
                        (CompiledGraphModel) null)).getMessage());
    }

    @Test
    void canonicalizesSparseIdsInBoundaryAndTopologicalOrderWithExactReferences() {
        TensorDescriptor firstInputDescriptor = descriptor(Shape.of(2));
        TensorDescriptor secondInputDescriptor = descriptor(Shape.of(2));
        TensorDescriptor sumDescriptor = descriptor(Shape.of(2));
        TensorDescriptor backwardDescriptor = descriptor(Shape.of(2));
        TensorDescriptor fanoutDescriptor = descriptor(Shape.of(2));
        ValueId firstInput = new ValueId(90);
        ValueId secondInput = new ValueId(3);
        ValueId sum = new ValueId(77);
        ValueId backward = new ValueId(8);
        ValueId fanout = new ValueId(6);
        Operation add = operation(BinaryArithmeticKind.ADD);
        Operation neg = operation(UnaryElementwiseKind.NEG);
        Operation abs = operation(UnaryElementwiseKind.ABS);
        NodeId addId = new NodeId(44);
        NodeId backwardId = new NodeId(2);
        NodeId fanoutId = new NodeId(99);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(fanout, fanoutDescriptor),
                        new GraphValue(firstInput, firstInputDescriptor),
                        new GraphValue(backward, backwardDescriptor),
                        new GraphValue(secondInput, secondInputDescriptor),
                        new GraphValue(sum, sumDescriptor)),
                List.of(
                        new CompiledNode(addId, add,
                                List.of(firstInput, firstInput), List.of(sum)),
                        new CompiledNode(backwardId, neg, List.of(sum), List.of(backward)),
                        new CompiledNode(fanoutId, abs, List.of(sum), List.of(fanout))),
                List.of(secondInput, firstInput),
                List.of(backward, fanout),
                Map.of(
                        addId, GraphPhase.FORWARD,
                        backwardId, GraphPhase.BACKWARD,
                        fanoutId, GraphPhase.FORWARD));

        CompiledGraphModel canonical = GraphCanonicalization.canonicalize(graph);

        assertAll(
                () -> assertNotSame(graph, canonical),
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1)), canonical.inputs()),
                () -> assertEquals(List.of(new ValueId(3), new ValueId(4)), canonical.outputs()),
                () -> assertEquals(List.of(0L, 1L, 2L, 3L, 4L), canonical.values().stream()
                        .map(value -> value.id().value()).toList()),
                () -> assertEquals(List.of(0L, 1L, 2L), canonical.nodes().stream()
                        .map(node -> node.id().value()).toList()),
                () -> assertEquals(List.of(new ValueId(1), new ValueId(1)),
                        canonical.nodes().get(0).inputs()),
                () -> assertEquals(List.of(new ValueId(2)), canonical.nodes().get(1).inputs()),
                () -> assertEquals(List.of(new ValueId(2)), canonical.nodes().get(2).inputs()),
                () -> assertSame(add, canonical.nodes().get(0).operation()),
                () -> assertSame(neg, canonical.nodes().get(1).operation()),
                () -> assertSame(abs, canonical.nodes().get(2).operation()),
                () -> assertSame(secondInputDescriptor,
                        canonical.values().get(0).descriptor()),
                () -> assertSame(firstInputDescriptor,
                        canonical.values().get(1).descriptor()),
                () -> assertSame(sumDescriptor, canonical.values().get(2).descriptor()),
                () -> assertSame(backwardDescriptor,
                        canonical.values().get(3).descriptor()),
                () -> assertSame(fanoutDescriptor,
                        canonical.values().get(4).descriptor()),
                () -> assertEquals(GraphPhase.BACKWARD,
                        canonical.nodePhases().get(new NodeId(1))),
                () -> assertEquals(List.of(firstInput, firstInput),
                        graph.nodes().get(0).inputs()),
                () -> assertEquals(List.of(backward, fanout), graph.outputs()));
    }

    @Test
    void isDeterministicImmutableAndHandlesZeroNodePassThrough() {
        TensorDescriptor descriptor = descriptor(Shape.of(2, 3));
        ValueId sparseInput = new ValueId(42);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(new GraphValue(sparseInput, descriptor)),
                List.of(),
                List.of(sparseInput),
                List.of(sparseInput),
                Map.of());

        CompiledGraphModel first = GraphCanonicalization.canonicalize(graph);
        CompiledGraphModel second = GraphCanonicalization.canonicalize(graph);
        CompiledGraphModel recanonicalized = GraphCanonicalization.canonicalize(first);

        assertAll(
                () -> assertEquals(first, second),
                () -> assertEquals(first, recanonicalized),
                () -> assertNotSame(first, second),
                () -> assertEquals(List.of(new ValueId(0)), first.inputs()),
                () -> assertEquals(first.inputs(), first.outputs()),
                () -> assertTrue(first.nodes().isEmpty()),
                () -> assertSame(descriptor, first.values().getFirst().descriptor()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> first.values().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> first.inputs().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> first.outputs().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> first.nodePhases().clear()));
    }

    private static Operation operation(io.github.pho001.synaptik.model.operation.OperationKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }

    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape, Optional.empty(), true);
    }
}
