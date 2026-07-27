package io.github.pho001.synaptik.planning.partition;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendId;
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
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MaximalSameOwnerPartitioningTest {
    @Test
    void hasTheExactPublicStatelessGeneratorShape() throws ReflectiveOperationException {
        Class<MaximalSameOwnerPartitioning> type = MaximalSameOwnerPartitioning.class;
        Constructor<?>[] constructors = type.getDeclaredConstructors();
        Method[] methods = type.getDeclaredMethods();
        Method partition = type.getDeclaredMethod("partition", CompiledGraphModel.class, Map.class);
        ParameterizedType mapType = (ParameterizedType) partition.getGenericParameterTypes()[1];
        ParameterizedType returnType = (ParameterizedType) partition.getGenericReturnType();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.planning.partition", type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertFalse(type.isRecord()),
                () -> assertFalse(type.isEnum()),
                () -> assertEquals(0, type.getDeclaredFields().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(1, methods.length),
                () -> assertEquals(partition, methods[0]),
                () -> assertTrue(Modifier.isStatic(partition.getModifiers())),
                () -> assertTrue(Modifier.isPublic(partition.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(partition.getModifiers())),
                () -> assertFalse(Modifier.isProtected(partition.getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {CompiledGraphModel.class, Map.class},
                        partition.getParameterTypes()),
                () -> assertEquals(Map.class, mapType.getRawType()),
                () -> assertArrayEquals(
                        new Type[] {NodeId.class, BackendId.class},
                        mapType.getActualTypeArguments()),
                () -> assertEquals(List.class, returnType.getRawType()),
                () -> assertArrayEquals(
                        new Type[] {PlannedPartition.class}, returnType.getActualTypeArguments()));
    }

    @Test
    void validatesTopLevelInputsAndNullKeysInExactOrder() {
        CompiledGraphModel graph = linearGraph(List.of(new NodeId(1)), List.of(GraphPhase.FORWARD));
        Map<NodeId, BackendId> nullKey = new HashMap<>();
        nullKey.put(new NodeId(99), new BackendId("extra"));
        nullKey.put(null, new BackendId("null-key"));

        NullPointerException nullGraph = assertThrows(
                NullPointerException.class,
                () -> MaximalSameOwnerPartitioning.partition(null, null));
        NullPointerException nullMap = assertThrows(
                NullPointerException.class,
                () -> MaximalSameOwnerPartitioning.partition(graph, null));
        NullPointerException nullKeyFailure = assertThrows(
                NullPointerException.class,
                () -> MaximalSameOwnerPartitioning.partition(graph, nullKey));

        assertAll(
                () -> assertEquals("graph", nullGraph.getMessage()),
                () -> assertEquals("ownershipByNodeId", nullMap.getMessage()),
                () -> assertEquals(
                        "ownershipByNodeId contains null key", nullKeyFailure.getMessage()));
    }

    @Test
    void rejectsUnknownKeysInNumericOrderBeforeCoverageAndOwners() {
        CompiledGraphModel graph = linearGraph(List.of(new NodeId(7)), List.of(GraphPhase.FORWARD));
        Map<NodeId, BackendId> ownership = new LinkedHashMap<>();
        ownership.put(new NodeId(11), null);
        ownership.put(new NodeId(3), null);

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> MaximalSameOwnerPartitioning.partition(graph, ownership));

        assertEquals(
                "ownershipByNodeId contains unknown NodeId[value=3]", failure.getMessage());
    }

    @Test
    void validatesCompleteCoverageBeforeOwnersThenOwnersInGraphOrder() {
        NodeId first = new NodeId(8);
        NodeId second = new NodeId(2);
        CompiledGraphModel graph = linearGraph(
                List.of(first, second), List.of(GraphPhase.FORWARD, GraphPhase.FORWARD));
        Map<NodeId, BackendId> missingAfterNull = new HashMap<>();
        missingAfterNull.put(new NodeId(8), null);

        IllegalArgumentException missing = assertThrows(
                IllegalArgumentException.class,
                () -> MaximalSameOwnerPartitioning.partition(graph, missingAfterNull));

        Map<NodeId, BackendId> bothNull = new HashMap<>();
        bothNull.put(new NodeId(2), null);
        bothNull.put(new NodeId(8), null);
        NullPointerException owner = assertThrows(
                NullPointerException.class,
                () -> MaximalSameOwnerPartitioning.partition(graph, bothNull));

        assertAll(
                () -> assertEquals(
                        "ownershipByNodeId missing NodeId[value=2]", missing.getMessage()),
                () -> assertEquals(
                        "ownershipByNodeId[NodeId[value=8]]", owner.getMessage()));
    }

    @Test
    void formsMaximalRunsByOwnerEqualityAndRetainsExactGraphAndFirstOwnerReferences() {
        List<NodeId> graphIds = List.of(
                new NodeId(40),
                new NodeId(10),
                new NodeId(50),
                new NodeId(20),
                new NodeId(30));
        CompiledGraphModel graph = linearGraph(
                graphIds,
                List.of(
                        GraphPhase.FORWARD,
                        GraphPhase.BACKWARD,
                        GraphPhase.BACKWARD,
                        GraphPhase.FORWARD,
                        GraphPhase.FORWARD));
        BackendId firstCpu = new BackendId(new String("cpu"));
        BackendId equalCpu = new BackendId(new String("cpu"));
        BackendId firstMetal = new BackendId(new String("metal"));
        BackendId equalMetal = new BackendId(new String("metal"));
        BackendId laterCpu = new BackendId(new String("cpu"));
        Map<NodeId, BackendId> ownership = new LinkedHashMap<>();
        ownership.put(new NodeId(20), equalMetal);
        ownership.put(new NodeId(10), equalCpu);
        ownership.put(new NodeId(30), laterCpu);
        ownership.put(new NodeId(40), firstCpu);
        ownership.put(new NodeId(50), firstMetal);
        Map<NodeId, BackendId> before = Map.copyOf(ownership);

        List<PlannedPartition> result =
                MaximalSameOwnerPartitioning.partition(graph, ownership);

        assertAll(
                () -> assertEquals(3, result.size()),
                () -> assertSame(firstCpu, result.get(0).owner()),
                () -> assertSame(firstMetal, result.get(1).owner()),
                () -> assertSame(laterCpu, result.get(2).owner()),
                () -> assertEquals(graphIds.subList(0, 2), result.get(0).nodeIds()),
                () -> assertEquals(graphIds.subList(2, 4), result.get(1).nodeIds()),
                () -> assertEquals(graphIds.subList(4, 5), result.get(2).nodeIds()),
                () -> assertSame(graph.nodes().get(0).id(), result.get(0).nodeIds().get(0)),
                () -> assertSame(graph.nodes().get(1).id(), result.get(0).nodeIds().get(1)),
                () -> assertSame(graph.nodes().get(2).id(), result.get(1).nodeIds().get(0)),
                () -> assertSame(graph.nodes().get(3).id(), result.get(1).nodeIds().get(1)),
                () -> assertSame(graph.nodes().get(4).id(), result.get(2).nodeIds().get(0)),
                () -> assertEquals(before, ownership),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> result.add(new PlannedPartition(firstCpu, List.of(new NodeId(99))))),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> result.get(0).nodeIds().add(new NodeId(99))));
    }

    @Test
    void consecutiveIndependentNodesAndGraphStructureDoNotRedefineAdjacency() {
        ValueId inputA = valueId(0);
        ValueId inputB = valueId(1);
        ValueId firstOutput = valueId(2);
        ValueId secondOutput = valueId(3);
        ValueId independentOutput = valueId(4);
        ValueId mergeOutput = valueId(5);
        ValueId fanOutOutput = valueId(6);
        CompiledNode multiOutput = node(
                new NodeId(10), List.of(inputA), List.of(firstOutput, secondOutput));
        CompiledNode independent = node(
                new NodeId(11), List.of(inputB), List.of(independentOutput));
        CompiledNode repeatedMerge = node(
                new NodeId(12),
                List.of(firstOutput, firstOutput, independentOutput),
                List.of(mergeOutput));
        CompiledNode fanOutMerge = node(
                new NodeId(13), List.of(firstOutput, secondOutput), List.of(fanOutOutput));
        List<CompiledNode> nodes =
                List.of(multiOutput, independent, repeatedMerge, fanOutMerge);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(
                        value(inputA),
                        value(inputB),
                        value(firstOutput),
                        value(secondOutput),
                        value(independentOutput),
                        value(mergeOutput),
                        value(fanOutOutput)),
                nodes,
                List.of(inputA, inputB),
                List.of(secondOutput, mergeOutput, fanOutOutput),
                Map.of(
                        multiOutput.id(), GraphPhase.FORWARD,
                        independent.id(), GraphPhase.BACKWARD,
                        repeatedMerge.id(), GraphPhase.BACKWARD,
                        fanOutMerge.id(), GraphPhase.FORWARD));
        BackendId owner = new BackendId("cpu");
        Map<NodeId, BackendId> ownership = new HashMap<>();
        for (CompiledNode node : nodes) {
            ownership.put(new NodeId(node.id().value()), new BackendId("cpu"));
        }
        ownership.put(new NodeId(10), owner);

        List<PlannedPartition> result =
                MaximalSameOwnerPartitioning.partition(graph, ownership);

        assertAll(
                () -> assertEquals(1, result.size()),
                () -> assertSame(owner, result.getFirst().owner()),
                () -> assertEquals(
                        nodes.stream().map(CompiledNode::id).toList(),
                        result.getFirst().nodeIds()),
                () -> assertSame(multiOutput.id(), result.getFirst().nodeIds().get(0)),
                () -> assertSame(independent.id(), result.getFirst().nodeIds().get(1)),
                () -> assertSame(repeatedMerge.id(), result.getFirst().nodeIds().get(2)),
                () -> assertSame(fanOutMerge.id(), result.getFirst().nodeIds().get(3)));
    }

    @Test
    void zeroNodePassThroughGraphRequiresNoOwnersAndReturnsImmutableEmptyList() {
        ValueId boundary = valueId(0);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(value(boundary)),
                List.of(),
                List.of(boundary),
                List.of(boundary),
                Map.of());

        List<PlannedPartition> result =
                MaximalSameOwnerPartitioning.partition(graph, Map.of());
        IllegalArgumentException extra = assertThrows(
                IllegalArgumentException.class,
                () -> MaximalSameOwnerPartitioning.partition(
                        graph, Map.of(new NodeId(0), new BackendId("cpu"))));

        assertAll(
                () -> assertTrue(result.isEmpty()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> result.add(new PlannedPartition(
                                new BackendId("cpu"), List.of(new NodeId(0))))),
                () -> assertEquals(
                        "ownershipByNodeId contains unknown NodeId[value=0]",
                        extra.getMessage()));
    }

    private static CompiledGraphModel linearGraph(
            List<NodeId> nodeIds, List<GraphPhase> phases) {
        List<GraphValue> values = new ArrayList<>();
        List<CompiledNode> nodes = new ArrayList<>();
        Map<NodeId, GraphPhase> nodePhases = new HashMap<>();
        ValueId input = valueId(0);
        values.add(value(input));
        ValueId previous = input;
        for (int index = 0; index < nodeIds.size(); index++) {
            ValueId output = valueId(index + 1L);
            CompiledNode node = node(nodeIds.get(index), List.of(previous), List.of(output));
            nodes.add(node);
            values.add(value(output));
            nodePhases.put(node.id(), phases.get(index));
            previous = output;
        }
        return new CompiledGraphModel(
                values, nodes, List.of(input), List.of(previous), nodePhases);
    }

    private static GraphValue value(ValueId id) {
        return new GraphValue(id, descriptor());
    }

    private static ValueId valueId(long value) {
        return new ValueId(value);
    }

    private static CompiledNode node(NodeId id, List<ValueId> inputs, List<ValueId> outputs) {
        return new CompiledNode(
                id,
                new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE),
                inputs,
                outputs);
    }

    private static TensorDescriptor descriptor() {
        return new TensorDescriptor(DataType.FLOAT32, Shape.scalar(), Optional.empty(), false);
    }

    private enum SampleKind implements OperationKind {
        SAMPLE;

        private static final List<OperationSignature> SIGNATURES = List.of(
                new OperationSignature(
                        NoOperationAttrs.class, 0, Integer.MAX_VALUE, 1, Integer.MAX_VALUE));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
