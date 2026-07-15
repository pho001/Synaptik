package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class ForwardCommonSubexpressionEliminationTest {
    @Test
    void exposesOnlyThePackagePrivateStatelessCseContract() throws Exception {
        var method = ForwardCommonSubexpressionElimination.class.getDeclaredMethod(
                "eliminate", CompiledGraphModel.class);
        var constructor = ForwardCommonSubexpressionElimination.class.getDeclaredConstructor();

        assertAll(
                () -> assertTrue(Modifier.isFinal(
                        ForwardCommonSubexpressionElimination.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        ForwardCommonSubexpressionElimination.class.getModifiers())),
                () -> assertEquals(0,
                        ForwardCommonSubexpressionElimination.class.getDeclaredFields().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertSame(CompiledGraphModel.class, method.getReturnType()),
                () -> assertEquals(1, Arrays.stream(
                                ForwardCommonSubexpressionElimination.class.getDeclaredMethods())
                        .filter(declared -> !declared.isSynthetic())
                        .filter(declared -> !Modifier.isPrivate(declared.getModifiers()))
                        .count()));
    }

    @Test
    void rejectsNullAndReturnsExactGraphWhenNoMergeOccurs() {
        CompiledGraphModel graph = graphWithOutputExcludedDuplicates();

        assertAll(
                () -> assertEquals("graph", assertThrows(NullPointerException.class,
                        () -> ForwardCommonSubexpressionElimination.eliminate(null)).getMessage()),
                () -> assertSame(graph,
                        ForwardCommonSubexpressionElimination.eliminate(graph)));
    }

    @Test
    void retainsFirstExactRepresentativeAndUsesAlreadyRemappedInputs() {
        TensorDescriptor descriptor = descriptor(DataType.FLOAT32, Shape.of(4), true);
        Operation firstAbs = operation(UnaryElementwiseKind.ABS);
        Operation equalAbs = operation(UnaryElementwiseKind.ABS);
        Operation firstNeg = operation(UnaryElementwiseKind.NEG);
        Operation secondNeg = operation(UnaryElementwiseKind.NEG);
        CompiledGraphModel graph = graph(
                List.of(descriptor, descriptor, descriptor, descriptor, descriptor),
                List.of(
                        node(0, firstAbs, List.of(0L), List.of(1L)),
                        node(1, equalAbs, List.of(0L), List.of(2L)),
                        node(2, firstNeg, List.of(1L), List.of(3L)),
                        node(3, secondNeg, List.of(2L), List.of(4L))),
                List.of(0L),
                List.of(3L, 4L),
                List.of(
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD));

        CompiledGraphModel result =
                ForwardCommonSubexpressionElimination.eliminate(graph);

        assertAll(
                () -> assertEquals(3, result.nodes().size()),
                () -> assertSame(firstAbs, result.nodes().get(0).operation()),
                () -> assertEquals(List.of(new ValueId(1)), result.nodes().get(1).inputs()),
                () -> assertEquals(List.of(new ValueId(1)), result.nodes().get(2).inputs()),
                () -> assertEquals(List.of(new ValueId(2), new ValueId(3)), result.outputs()),
                () -> assertSame(firstAbs, graph.nodes().get(0).operation()),
                () -> assertSame(equalAbs, graph.nodes().get(1).operation()));
    }

    @Test
    void distinguishesCompleteKeysIncludingOrderRepetitionDescriptorsOperationsAndPhases() {
        TensorDescriptor vector = descriptor(DataType.FLOAT32, Shape.of(2), true);
        TensorDescriptor otherDescriptor = descriptor(DataType.FLOAT32, Shape.of(3), true);
        Operation add = operation(BinaryArithmeticKind.ADD);
        Operation otherAdd = operation(BinaryArithmeticKind.ADD);
        CompiledGraphModel graph = graph(
                List.of(vector, vector, vector, vector, vector, vector, vector, otherDescriptor),
                List.of(
                        node(0, add, List.of(0L, 1L), List.of(2L)),
                        node(1, otherAdd, List.of(1L, 0L), List.of(3L)),
                        node(2, operation(BinaryArithmeticKind.ADD),
                                List.of(0L, 0L), List.of(4L)),
                        node(3, operation(BinaryArithmeticKind.SUB),
                                List.of(0L, 1L), List.of(5L)),
                        node(4, operation(UnaryElementwiseKind.ABS),
                                List.of(0L), List.of(6L)),
                        node(5, operation(UnaryElementwiseKind.ABS),
                                List.of(0L), List.of(7L))),
                List.of(0L, 1L),
                List.of(0L),
                List.of(
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD,
                        GraphPhase.BACKWARD,
                        GraphPhase.FORWARD));

        assertSame(graph, ForwardCommonSubexpressionElimination.eliminate(graph));
    }

    @Test
    void excludesGraphOutputProducersEncounteredBeforeAndAfterAnEqualInternalNode() {
        CompiledGraphModel graph = graphWithOutputExcludedDuplicates();

        CompiledGraphModel result =
                ForwardCommonSubexpressionElimination.eliminate(graph);

        assertAll(
                () -> assertSame(graph, result),
                () -> assertEquals(3, result.nodes().stream()
                        .filter(node -> node.operation().kind() == UnaryElementwiseKind.ABS)
                        .count()),
                () -> assertEquals(List.of(new ValueId(1), new ValueId(3), new ValueId(4)),
                        result.outputs()));
    }

    @Test
    void mergesMultiOutputOccurrenceAllOrNothingAndKeepsDescriptorsImmutable() {
        TensorDescriptor input = descriptor(DataType.FLOAT32, Shape.of(4), true);
        TensorDescriptor selected = descriptor(DataType.FLOAT32, Shape.of(2), true);
        TensorDescriptor indices = descriptor(DataType.INT64, Shape.of(2), false);
        Operation first = new Operation(
                TopKKind.TOP_K, new TopKAttrs(0, 2, true, true));
        Operation second = new Operation(
                TopKKind.TOP_K, new TopKAttrs(0, 2, true, true));
        CompiledGraphModel graph = graph(
                List.of(input, selected, indices, selected, indices),
                List.of(
                        node(0, first, List.of(0L), List.of(1L, 2L)),
                        node(1, second, List.of(0L), List.of(3L, 4L))),
                List.of(0L),
                List.of(0L),
                List.of(GraphPhase.FORWARD, GraphPhase.FORWARD));

        CompiledGraphModel result =
                ForwardCommonSubexpressionElimination.eliminate(graph);

        assertAll(
                () -> assertEquals(1, result.nodes().size()),
                () -> assertEquals(List.of(new ValueId(1), new ValueId(2)),
                        result.nodes().getFirst().outputs()),
                () -> assertSame(first, result.nodes().getFirst().operation()),
                () -> assertSame(selected, result.values().get(1).descriptor()),
                () -> assertSame(indices, result.values().get(2).descriptor()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> result.values().clear()),
                () -> assertSame(result,
                        CapturedGraphInference.inferAndValidate(result).graph()));
    }

    @Test
    void mergesEqualExplicitRngAndDropoutOccurrencesSlotwise() {
        TensorDescriptor input = descriptor(DataType.FLOAT32, Shape.of(4), true);
        TensorDescriptor state = descriptor(DataType.INT64, Shape.of(2), false);
        TensorDescriptor output = descriptor(DataType.FLOAT32, Shape.of(4), true);
        TensorDescriptor mask = descriptor(DataType.BOOL, Shape.of(4), false);
        Operation firstState = new Operation(
                GraphRngKind.INITIAL_STATE, new GraphRngStateAttrs(7, 11));
        Operation secondState = new Operation(
                GraphRngKind.INITIAL_STATE, new GraphRngStateAttrs(7, 11));
        Operation firstDropout = new Operation(
                DropoutKind.DROPOUT, new DropoutAttrs(0.25));
        Operation secondDropout = new Operation(
                DropoutKind.DROPOUT, new DropoutAttrs(0.25));
        CompiledGraphModel graph = graph(
                List.of(
                        input,
                        state,
                        state,
                        output,
                        mask,
                        state,
                        output,
                        mask,
                        state),
                List.of(
                        node(0, firstState, List.of(), List.of(1L)),
                        node(1, secondState, List.of(), List.of(2L)),
                        node(2, firstDropout, List.of(0L, 1L), List.of(3L, 4L, 5L)),
                        node(3, secondDropout, List.of(0L, 2L), List.of(6L, 7L, 8L))),
                List.of(0L),
                List.of(0L),
                List.of(
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD));

        CompiledGraphModel result =
                ForwardCommonSubexpressionElimination.eliminate(graph);

        assertAll(
                () -> assertEquals(2, result.nodes().size()),
                () -> assertSame(firstState, result.nodes().get(0).operation()),
                () -> assertSame(firstDropout, result.nodes().get(1).operation()),
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1)),
                        result.nodes().get(1).inputs()),
                () -> assertEquals(List.of(new ValueId(2), new ValueId(3), new ValueId(4)),
                        result.nodes().get(1).outputs()),
                () -> assertSame(result,
                        CapturedGraphInference.inferAndValidate(result).graph()));
    }

    private static CompiledGraphModel graphWithOutputExcludedDuplicates() {
        TensorDescriptor descriptor = descriptor(DataType.FLOAT32, Shape.of(4), true);
        return graph(
                List.of(descriptor, descriptor, descriptor, descriptor, descriptor),
                List.of(
                        node(0, operation(UnaryElementwiseKind.ABS),
                                List.of(0L), List.of(1L)),
                        node(1, operation(UnaryElementwiseKind.ABS),
                                List.of(0L), List.of(2L)),
                        node(2, operation(UnaryElementwiseKind.ABS),
                                List.of(0L), List.of(3L)),
                        node(3, operation(UnaryElementwiseKind.NEG),
                                List.of(2L), List.of(4L))),
                List.of(0L),
                List.of(1L, 3L, 4L),
                List.of(
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD));
    }

    private static CompiledGraphModel graph(
            List<TensorDescriptor> descriptors,
            List<CompiledNode> nodes,
            List<Long> inputIds,
            List<Long> outputIds,
            List<GraphPhase> phases) {
        List<GraphValue> values = new ArrayList<>(descriptors.size());
        for (int index = 0; index < descriptors.size(); index++) {
            values.add(new GraphValue(new ValueId(index), descriptors.get(index)));
        }
        Map<NodeId, GraphPhase> phaseMap = new LinkedHashMap<>();
        for (int index = 0; index < nodes.size(); index++) {
            phaseMap.put(nodes.get(index).id(), phases.get(index));
        }
        return new CompiledGraphModel(
                values,
                nodes,
                ids(inputIds),
                ids(outputIds),
                phaseMap);
    }

    private static CompiledNode node(
            long id, Operation operation, List<Long> inputs, List<Long> outputs) {
        return new CompiledNode(new NodeId(id), operation, ids(inputs), ids(outputs));
    }

    private static List<ValueId> ids(List<Long> values) {
        return values.stream().map(ValueId::new).toList();
    }

    private static Operation operation(OperationKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }

    private static TensorDescriptor descriptor(
            DataType dataType, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad);
    }
}
