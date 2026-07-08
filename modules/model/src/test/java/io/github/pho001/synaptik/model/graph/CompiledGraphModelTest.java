package io.github.pho001.synaptik.model.graph;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CompiledGraphModelTest {
    @Test
    void hasExactlyTheRequiredFiveComponentRecordState() {
        var components = CompiledGraphModel.class.getRecordComponents();
        var instanceFields = Arrays.stream(CompiledGraphModel.class.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();

        assertAll(
                () -> assertTrue(CompiledGraphModel.class.isRecord()),
                () -> assertEquals(5, components.length),
                () -> assertEquals("values", components[0].getName()),
                () -> assertEquals(List.class, components[0].getType()),
                () -> assertEquals("nodes", components[1].getName()),
                () -> assertEquals(List.class, components[1].getType()),
                () -> assertEquals("inputs", components[2].getName()),
                () -> assertEquals(List.class, components[2].getType()),
                () -> assertEquals("outputs", components[3].getName()),
                () -> assertEquals(List.class, components[3].getType()),
                () -> assertEquals("nodePhases", components[4].getName()),
                () -> assertEquals(Map.class, components[4].getType()),
                () -> assertEquals(
                        List.of("values", "nodes", "inputs", "outputs", "nodePhases"),
                        instanceFields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(instanceFields.stream().allMatch(
                        field -> Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(1, CompiledGraphModel.class.getDeclaredConstructors().length));
    }

    @Test
    void rejectsNullComponentReferencesInDeclarationOrder() {
        List<GraphValue> values = List.of(value(0));
        List<CompiledNode> nodes = List.of();
        List<ValueId> inputs = List.of(id(0));
        List<ValueId> outputs = List.of(id(0));
        Map<NodeId, GraphPhase> phases = Map.of();

        NullPointerException nullValues = assertThrows(
                NullPointerException.class,
                () -> new CompiledGraphModel(null, null, null, null, null));
        NullPointerException nullNodes = assertThrows(
                NullPointerException.class,
                () -> new CompiledGraphModel(values, null, null, null, null));
        NullPointerException nullInputs = assertThrows(
                NullPointerException.class,
                () -> new CompiledGraphModel(values, nodes, null, null, null));
        NullPointerException nullOutputs = assertThrows(
                NullPointerException.class,
                () -> new CompiledGraphModel(values, nodes, inputs, null, null));
        NullPointerException nullPhases = assertThrows(
                NullPointerException.class,
                () -> new CompiledGraphModel(values, nodes, inputs, outputs, null));

        assertAll(
                () -> assertEquals("values", nullValues.getMessage()),
                () -> assertEquals("nodes", nullNodes.getMessage()),
                () -> assertEquals("inputs", nullInputs.getMessage()),
                () -> assertEquals("outputs", nullOutputs.getMessage()),
                () -> assertEquals("nodePhases", nullPhases.getMessage()));
    }

    @Test
    void validatesValueElementsAndIdsBeforeLaterComponents() {
        List<GraphValue> nullValues = new ArrayList<>(Arrays.asList(value(0), null, null));
        NullPointerException nullElement = assertThrows(
                NullPointerException.class,
                () -> new CompiledGraphModel(nullValues, null, null, null, null));

        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(7), value(8), value(7), value(8)),
                        List.of(),
                        List.of(),
                        List.of(id(7)),
                        Map.of()));

        assertAll(
                () -> assertEquals("nodes", nullElement.getMessage()),
                () -> assertEquals(
                        "values[2] duplicates ValueId[value=7]", duplicate.getMessage()));
    }

    @Test
    void rejectsIndexedNullAndDuplicateValueElementsAfterContainerChecks() {
        List<GraphValue> nullValues = new ArrayList<>(Arrays.asList(value(0), null, null));
        NullPointerException nullElement = assertThrows(
                NullPointerException.class,
                () -> new CompiledGraphModel(
                        nullValues, List.of(), List.of(), List.of(id(0)), Map.of()));

        assertEquals("values[1]", nullElement.getMessage());
    }

    @Test
    void rejectsIndexedNullAndDuplicateNodesInEncounterOrder() {
        List<CompiledNode> nullNodes = new ArrayList<>(Arrays.asList(
                node(1, List.of(id(0)), List.of(id(1))), null, null));
        NullPointerException nullElement = assertThrows(
                NullPointerException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0), value(1)),
                        nullNodes,
                        List.of(id(0)),
                        List.of(id(1)),
                        Map.of()));
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0), value(1), value(2)),
                        List.of(
                                node(4, List.of(id(0)), List.of(id(1))),
                                node(5, List.of(id(1)), List.of(id(2))),
                                node(4, List.of(id(0)), List.of(id(2)))),
                        List.of(id(0)),
                        List.of(id(2)),
                        Map.of()));

        assertAll(
                () -> assertEquals("nodes[1]", nullElement.getMessage()),
                () -> assertEquals(
                        "nodes[2] duplicates NodeId[value=4]", duplicate.getMessage()));
    }

    @Test
    void rejectsIndexedNullAndDuplicateInputBoundaryIds() {
        List<ValueId> nullInputs = new ArrayList<>(Arrays.asList(id(0), null, null));
        NullPointerException nullElement = assertThrows(
                NullPointerException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0)),
                        List.of(),
                        nullInputs,
                        List.of(id(0)),
                        Map.of()));
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0), value(1)),
                        List.of(),
                        List.of(id(0), id(1), id(0), id(1)),
                        List.of(id(0)),
                        Map.of()));

        assertAll(
                () -> assertEquals("inputs[1]", nullElement.getMessage()),
                () -> assertEquals(
                        "inputs[2] duplicates ValueId[value=0]", duplicate.getMessage()));
    }

    @Test
    void requiresOutputsBeforeInspectingTheirElementsAndRejectsDuplicates() {
        IllegalArgumentException empty = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(), List.of(), List.of(), List.of(), Map.of()));
        List<ValueId> nullOutputs = new ArrayList<>(Arrays.asList(id(0), null, null));
        NullPointerException nullElement = assertThrows(
                NullPointerException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0)), List.of(), List.of(id(0)), nullOutputs, Map.of()));
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0), value(1)),
                        List.of(),
                        List.of(id(0), id(1)),
                        List.of(id(0), id(1), id(0), id(1)),
                        Map.of()));

        assertAll(
                () -> assertEquals("outputs must not be empty", empty.getMessage()),
                () -> assertEquals("outputs[1]", nullElement.getMessage()),
                () -> assertEquals(
                        "outputs[2] duplicates ValueId[value=0]", duplicate.getMessage()));
    }

    @Test
    void validatesPhaseMapNullsBeforeGraphReferencesWithDeterministicKeyOrder() {
        Map<NodeId, GraphPhase> nullKey = new HashMap<>();
        nullKey.put(null, GraphPhase.FORWARD);
        nullKey.put(new NodeId(9), null);
        NullPointerException keyFailure = assertThrows(
                NullPointerException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0)), List.of(), List.of(id(99)), List.of(id(98)), nullKey));

        Map<NodeId, GraphPhase> nullPhases = new LinkedHashMap<>();
        nullPhases.put(new NodeId(9), null);
        nullPhases.put(new NodeId(2), null);
        NullPointerException phaseFailure = assertThrows(
                NullPointerException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0)),
                        List.of(),
                        List.of(id(99)),
                        List.of(id(98)),
                        nullPhases));

        assertAll(
                () -> assertEquals("nodePhases contains null key", keyFailure.getMessage()),
                () -> assertEquals(
                        "nodePhases[NodeId[value=2]]", phaseFailure.getMessage()));
    }

    @Test
    void resolvesDeclaredInputsThenOutputsInBoundaryOrder() {
        IllegalArgumentException unknownInput = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0)),
                        List.of(),
                        List.of(id(0), id(9), id(8)),
                        List.of(id(7)),
                        Map.of()));
        IllegalArgumentException unknownOutput = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0)),
                        List.of(),
                        List.of(id(0)),
                        List.of(id(0), id(9), id(8)),
                        Map.of()));
        IllegalArgumentException emptyValues = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(), List.of(), List.of(), List.of(id(5)), Map.of()));

        assertAll(
                () -> assertEquals(
                        "inputs[1] references unknown ValueId[value=9]",
                        unknownInput.getMessage()),
                () -> assertEquals(
                        "outputs[1] references unknown ValueId[value=9]",
                        unknownOutput.getMessage()),
                () -> assertEquals(
                        "outputs[0] references unknown ValueId[value=5]",
                        emptyValues.getMessage()));
    }

    @Test
    void rejectsUnknownAndUnavailableNodeInputsWithIndexedMessages() {
        IllegalArgumentException unknown = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0), value(1)),
                        List.of(node(3, List.of(id(0), id(9)), List.of(id(1)))),
                        List.of(id(0)),
                        List.of(id(1)),
                        Map.of(new NodeId(3), GraphPhase.FORWARD)));
        IllegalArgumentException laterDependency = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0), value(1), value(2)),
                        List.of(
                                node(3, List.of(id(2)), List.of(id(1))),
                                node(4, List.of(id(0)), List.of(id(2)))),
                        List.of(id(0)),
                        List.of(id(1)),
                        Map.of(
                                new NodeId(3), GraphPhase.FORWARD,
                                new NodeId(4), GraphPhase.FORWARD)));
        IllegalArgumentException selfDependency = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(1)),
                        List.of(node(6, List.of(id(1)), List.of(id(1)))),
                        List.of(),
                        List.of(id(1)),
                        Map.of(new NodeId(6), GraphPhase.BACKWARD)));

        assertAll(
                () -> assertEquals(
                        "nodes[0].inputs[1] references unknown ValueId[value=9]",
                        unknown.getMessage()),
                () -> assertEquals(
                        "nodes[0].inputs[0] is not available before NodeId[value=3]: ValueId[value=2]",
                        laterDependency.getMessage()),
                () -> assertEquals(
                        "nodes[0].inputs[0] is not available before NodeId[value=6]: ValueId[value=1]",
                        selfDependency.getMessage()));
    }

    @Test
    void rejectsUnknownOutputsProducedInputsAndMultipleProducers() {
        IllegalArgumentException unknown = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0)),
                        List.of(node(1, List.of(id(0)), List.of(id(9)))),
                        List.of(id(0)),
                        List.of(id(0)),
                        Map.of(new NodeId(1), GraphPhase.FORWARD)));
        IllegalArgumentException producedInput = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0)),
                        List.of(node(1, List.of(), List.of(id(0)))),
                        List.of(id(0)),
                        List.of(id(0)),
                        Map.of(new NodeId(1), GraphPhase.FORWARD)));
        IllegalArgumentException secondProducer = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0), value(1), value(2)),
                        List.of(
                                node(4, List.of(id(0)), List.of(id(1))),
                                node(7, List.of(id(0)), List.of(id(2), id(1)))),
                        List.of(id(0)),
                        List.of(id(2)),
                        Map.of(
                                new NodeId(4), GraphPhase.FORWARD,
                                new NodeId(7), GraphPhase.BACKWARD)));

        assertAll(
                () -> assertEquals(
                        "nodes[0].outputs[0] references unknown ValueId[value=9]",
                        unknown.getMessage()),
                () -> assertEquals(
                        "nodes[0].outputs[0] produces graph input ValueId[value=0]",
                        producedInput.getMessage()),
                () -> assertEquals(
                        "nodes[1].outputs[1] gives ValueId[value=1] a second producer; first producer is NodeId[value=4]",
                        secondProducer.getMessage()));
    }

    @Test
    void rejectsFirstUnproducedNonInputValueInValueOrder() {
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0), value(8), value(7)),
                        List.of(),
                        List.of(id(0)),
                        List.of(id(0)),
                        Map.of()));

        assertEquals(
                "values[1] is neither a graph input nor a node output: ValueId[value=8]",
                failure.getMessage());
    }

    @Test
    void rejectsMissingPhasesInNodeOrderThenUnknownKeysInNumericOrder() {
        CompiledNode first = node(9, List.of(id(0)), List.of(id(1)));
        CompiledNode second = node(2, List.of(id(1)), List.of(id(2)));
        IllegalArgumentException missing = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0), value(1), value(2)),
                        List.of(first, second),
                        List.of(id(0)),
                        List.of(id(2)),
                        Map.of(new NodeId(2), GraphPhase.BACKWARD)));

        Map<NodeId, GraphPhase> overCoverage = new LinkedHashMap<>();
        overCoverage.put(new NodeId(9), GraphPhase.FORWARD);
        overCoverage.put(new NodeId(8), GraphPhase.BACKWARD);
        overCoverage.put(new NodeId(3), GraphPhase.FORWARD);
        IllegalArgumentException unknown = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledGraphModel(
                        List.of(value(0), value(1)),
                        List.of(first),
                        List.of(id(0)),
                        List.of(id(1)),
                        overCoverage));

        assertAll(
                () -> assertEquals("nodePhases missing NodeId[value=9]", missing.getMessage()),
                () -> assertEquals(
                        "nodePhases contains unknown NodeId[value=3]", unknown.getMessage()));
    }

    @Test
    void acceptsTopologicalForwardBackwardGraphAndPreservesAllListOrders() {
        GraphValue input = value(0);
        GraphValue forward = value(1);
        GraphValue backward = value(2);
        CompiledNode forwardNode = node(10, List.of(input.id()), List.of(forward.id()));
        CompiledNode backwardNode =
                node(11, List.of(forward.id(), input.id()), List.of(backward.id()));
        Map<NodeId, GraphPhase> phases = Map.of(
                forwardNode.id(), GraphPhase.FORWARD,
                backwardNode.id(), GraphPhase.BACKWARD);

        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(input, forward, backward),
                List.of(forwardNode, backwardNode),
                List.of(input.id()),
                List.of(backward.id(), forward.id()),
                phases);

        assertAll(
                () -> assertEquals(List.of(input, forward, backward), graph.values()),
                () -> assertEquals(List.of(forwardNode, backwardNode), graph.nodes()),
                () -> assertEquals(List.of(input.id()), graph.inputs()),
                () -> assertEquals(List.of(backward.id(), forward.id()), graph.outputs()),
                () -> assertEquals(phases, graph.nodePhases()));
    }

    @Test
    void acceptsPassThroughZeroInputRepeatedInputsAndUnusedGraphInputs() {
        CompiledGraphModel passThrough = new CompiledGraphModel(
                List.of(value(0)),
                List.of(),
                List.of(id(0)),
                List.of(id(0)),
                Map.of());

        CompiledNode source = node(1, List.of(), List.of(id(1)));
        CompiledGraphModel zeroInput = new CompiledGraphModel(
                List.of(value(1)),
                List.of(source),
                List.of(),
                List.of(id(1)),
                Map.of(source.id(), GraphPhase.FORWARD));

        CompiledNode repeated = node(2, List.of(id(2), id(2), id(2)), List.of(id(4)));
        CompiledGraphModel repeatedAndUnused = new CompiledGraphModel(
                List.of(value(2), value(3), value(4)),
                List.of(repeated),
                List.of(id(2), id(3)),
                List.of(id(4)),
                Map.of(repeated.id(), GraphPhase.BACKWARD));

        assertAll(
                () -> assertTrue(passThrough.nodes().isEmpty()),
                () -> assertTrue(passThrough.nodePhases().isEmpty()),
                () -> assertTrue(zeroInput.inputs().isEmpty()),
                () -> assertTrue(source.inputs().isEmpty()),
                () -> assertEquals(List.of(id(2), id(2), id(2)), repeated.inputs()),
                () -> assertEquals(List.of(id(2), id(3)), repeatedAndUnused.inputs()));
    }

    @Test
    void snapshotsEveryCollectionAgainstCallerAndAccessorMutation() {
        GraphValue input = value(0);
        GraphValue output = value(1);
        CompiledNode node = node(3, List.of(input.id()), List.of(output.id()));
        List<GraphValue> values = new ArrayList<>(List.of(input, output));
        List<CompiledNode> nodes = new ArrayList<>(List.of(node));
        List<ValueId> inputs = new ArrayList<>(List.of(input.id()));
        List<ValueId> outputs = new ArrayList<>(List.of(output.id()));
        Map<NodeId, GraphPhase> phases = new HashMap<>(Map.of(node.id(), GraphPhase.FORWARD));

        CompiledGraphModel graph =
                new CompiledGraphModel(values, nodes, inputs, outputs, phases);
        values.clear();
        nodes.clear();
        inputs.clear();
        outputs.clear();
        phases.clear();

        assertAll(
                () -> assertEquals(List.of(input, output), graph.values()),
                () -> assertEquals(List.of(node), graph.nodes()),
                () -> assertEquals(List.of(input.id()), graph.inputs()),
                () -> assertEquals(List.of(output.id()), graph.outputs()),
                () -> assertEquals(Map.of(node.id(), GraphPhase.FORWARD), graph.nodePhases()),
                () -> assertThrows(UnsupportedOperationException.class, () -> graph.values().clear()),
                () -> assertThrows(UnsupportedOperationException.class, () -> graph.nodes().clear()),
                () -> assertThrows(UnsupportedOperationException.class, () -> graph.inputs().clear()),
                () -> assertThrows(UnsupportedOperationException.class, () -> graph.outputs().clear()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> graph.nodePhases().clear()));
    }

    @Test
    void recordEqualityUsesListOrderAndStructuralMapSemantics() {
        GraphValue input = value(0);
        GraphValue firstOutput = value(1);
        GraphValue secondOutput = value(2);
        CompiledNode firstNode = node(1, List.of(id(0)), List.of(id(1)));
        CompiledNode secondNode = node(2, List.of(id(1)), List.of(id(2)));
        CompiledGraphModel first = new CompiledGraphModel(
                List.of(input, firstOutput, secondOutput),
                List.of(firstNode, secondNode),
                List.of(input.id()),
                List.of(firstOutput.id(), secondOutput.id()),
                Map.of(
                        firstNode.id(), GraphPhase.FORWARD,
                        secondNode.id(), GraphPhase.BACKWARD));
        Map<NodeId, GraphPhase> differentlyInserted = new LinkedHashMap<>();
        differentlyInserted.put(secondNode.id(), GraphPhase.BACKWARD);
        differentlyInserted.put(firstNode.id(), GraphPhase.FORWARD);
        CompiledGraphModel equal = new CompiledGraphModel(
                List.of(input, firstOutput, secondOutput),
                List.of(firstNode, secondNode),
                List.of(input.id()),
                List.of(firstOutput.id(), secondOutput.id()),
                differentlyInserted);
        CompiledGraphModel reorderedOutputs = new CompiledGraphModel(
                List.of(input, firstOutput, secondOutput),
                List.of(firstNode, secondNode),
                List.of(input.id()),
                List.of(secondOutput.id(), firstOutput.id()),
                differentlyInserted);
        CompiledGraphModel changedPhase = new CompiledGraphModel(
                first.values(),
                first.nodes(),
                first.inputs(),
                first.outputs(),
                Map.of(
                        firstNode.id(), GraphPhase.BACKWARD,
                        secondNode.id(), GraphPhase.BACKWARD));

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, reorderedOutputs),
                () -> assertNotEquals(first, changedPhase));
    }

    @Test
    void diagnosticTextNamesEveryComponentWithoutDefiningAnExactFormat() {
        CompiledNode node = node(3, List.of(id(0)), List.of(id(1)));
        String text = new CompiledGraphModel(
                        List.of(value(0), value(1)),
                        List.of(node),
                        List.of(id(0)),
                        List.of(id(1)),
                        Map.of(node.id(), GraphPhase.FORWARD))
                .toString();

        assertAll(
                () -> assertTrue(text.contains("CompiledGraphModel")),
                () -> assertTrue(text.contains("values=")),
                () -> assertTrue(text.contains("nodes=")),
                () -> assertTrue(text.contains("inputs=")),
                () -> assertTrue(text.contains("outputs=")),
                () -> assertTrue(text.contains("nodePhases=")),
                () -> assertTrue(text.contains("FORWARD")));
    }

    private static GraphValue value(long value) {
        return new GraphValue(id(value), descriptor());
    }

    private static ValueId id(long value) {
        return new ValueId(value);
    }

    private static CompiledNode node(long nodeId, List<ValueId> inputs, List<ValueId> outputs) {
        return new CompiledNode(
                new NodeId(nodeId),
                new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE),
                inputs,
                outputs);
    }

    private static TensorDescriptor descriptor() {
        return new TensorDescriptor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), false);
    }

    private enum SampleKind implements OperationKind {
        SAMPLE;

        private static final List<OperationSignature> SIGNATURES = List.of(new OperationSignature(
                NoOperationAttrs.class, 0, Integer.MAX_VALUE, 1, Integer.MAX_VALUE));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
