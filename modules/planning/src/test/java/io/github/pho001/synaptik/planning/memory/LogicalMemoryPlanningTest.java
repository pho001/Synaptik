package io.github.pho001.synaptik.planning.memory;

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
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class LogicalMemoryPlanningTest {
    @Test
    void hasTheExactPublicStatelessGeneratorShape() throws ReflectiveOperationException {
        Class<LogicalMemoryPlanning> type = LogicalMemoryPlanning.class;
        Constructor<?>[] constructors = type.getDeclaredConstructors();
        Method[] methods = type.getDeclaredMethods();
        Method plan = type.getDeclaredMethod("plan", CompiledGraphModel.class, List.class);
        ParameterizedType listType = (ParameterizedType) plan.getGenericParameterTypes()[1];

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.planning.memory", type.getPackageName()),
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
                () -> assertEquals(plan, methods[0]),
                () -> assertTrue(Modifier.isStatic(plan.getModifiers())),
                () -> assertTrue(Modifier.isPublic(plan.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(plan.getModifiers())),
                () -> assertFalse(Modifier.isProtected(plan.getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {CompiledGraphModel.class, List.class},
                        plan.getParameterTypes()),
                () -> assertEquals(List.class, listType.getRawType()),
                () -> assertArrayEquals(
                        new Type[] {PlannedPartition.class}, listType.getActualTypeArguments()),
                () -> assertEquals(LogicalMemoryPlan.class, plan.getReturnType()));
    }

    @Test
    void validatesTopLevelInputsAndPartitionElementsInExactOrder() {
        CompiledGraphModel graph = linearGraph(1, 2);
        List<PlannedPartition> withNull =
                new ArrayList<>(Arrays.asList(partition("cpu", 1), null, null));

        NullPointerException nullGraph = assertThrows(
                NullPointerException.class, () -> LogicalMemoryPlanning.plan(null, null));
        NullPointerException nullPartitions = assertThrows(
                NullPointerException.class, () -> LogicalMemoryPlanning.plan(graph, null));
        NullPointerException nullElement = assertThrows(
                NullPointerException.class, () -> LogicalMemoryPlanning.plan(graph, withNull));

        assertAll(
                () -> assertEquals("graph", nullGraph.getMessage()),
                () -> assertEquals("partitions", nullPartitions.getMessage()),
                () -> assertEquals("partitions[1]", nullElement.getMessage()));
    }

    @Test
    void validatesUnknownThenDuplicateMembershipBeforeCoverage() {
        CompiledGraphModel graph = linearGraph(1, 2);

        IllegalArgumentException unknown = assertThrows(
                IllegalArgumentException.class,
                () -> LogicalMemoryPlanning.plan(
                        graph,
                        List.of(
                                partition("cpu", 1),
                                partition("metal", 99),
                                partition("cpu", 1))));
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class,
                () -> LogicalMemoryPlanning.plan(
                        graph,
                        List.of(partition("cpu", 1), partition("metal", 1))));

        assertAll(
                () -> assertEquals(
                        "partitions[1].nodeIds[0] references unknown NodeId[value=99]",
                        unknown.getMessage()),
                () -> assertEquals(
                        "partitions[1].nodeIds[0] duplicates NodeId[value=1]",
                        duplicate.getMessage()));
    }

    @Test
    void validatesMissingCoverageThenGraphOrderThenAdjacentOwnerMaximality() {
        CompiledGraphModel graph = linearGraph(1, 2, 3);

        IllegalArgumentException missing = assertThrows(
                IllegalArgumentException.class,
                () -> LogicalMemoryPlanning.plan(
                        graph, List.of(partition("cpu", 1), partition("metal", 3))));
        IllegalArgumentException order = assertThrows(
                IllegalArgumentException.class,
                () -> LogicalMemoryPlanning.plan(
                        graph,
                        List.of(
                                partition("cpu", 2),
                                partition("metal", 1),
                                partition("cpu", 3))));
        IllegalArgumentException maximality = assertThrows(
                IllegalArgumentException.class,
                () -> LogicalMemoryPlanning.plan(
                        graph,
                        List.of(
                                partition("cpu", 1),
                                partition("cpu", 2),
                                partition("metal", 3))));

        assertAll(
                () -> assertEquals(
                        "partitions missing NodeId[value=2]", missing.getMessage()),
                () -> assertEquals(
                        "partitions[0].nodeIds[0] is out of graph order: expected NodeId[value=1]",
                        order.getMessage()),
                () -> assertEquals(
                        "partitions[1].owner equals previous owner BackendId[value=cpu]",
                        maximality.getMessage()));
    }

    @Test
    void derivesEveryPrimitiveFactInGraphValueAndPartitionOrderWithExactReferences() {
        ValueId inputA = valueId(0);
        ValueId inputB = valueId(1);
        ValueId unusedInput = valueId(2);
        ValueId firstOutput = valueId(3);
        ValueId secondOutput = valueId(4);
        ValueId independentOutput = valueId(5);
        ValueId mergeOutput = valueId(6);
        ValueId fanOutOutput = valueId(7);
        ValueId backwardOutput = valueId(8);
        ValueId unusedProduced = valueId(9);

        TensorDescriptor dynamicDescriptor = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("batch")),
                Optional.empty(),
                false);
        TensorDescriptor expressionDescriptor = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(DimensionExpressions.addConstant(
                        new DynamicDimension("sequence"), 1)),
                Optional.empty(),
                false);

        CompiledNode multiOutput = node(
                new NodeId(10),
                List.of(inputA, inputA),
                List.of(firstOutput, secondOutput));
        CompiledNode independent = node(
                new NodeId(11), List.of(inputB), List.of(independentOutput));
        CompiledNode merge = node(
                new NodeId(12),
                List.of(firstOutput, firstOutput, independentOutput),
                List.of(mergeOutput));
        CompiledNode fanOut = node(
                new NodeId(13), List.of(firstOutput), List.of(fanOutOutput));
        CompiledNode backward = node(
                new NodeId(14),
                List.of(secondOutput, mergeOutput),
                List.of(backwardOutput, unusedProduced));

        List<GraphValue> values = List.of(
                value(mergeOutput),
                value(inputA),
                new GraphValue(unusedProduced, expressionDescriptor),
                new GraphValue(firstOutput, dynamicDescriptor),
                value(inputB),
                value(secondOutput),
                value(independentOutput),
                value(fanOutOutput),
                value(backwardOutput),
                value(unusedInput));
        CompiledGraphModel graph = new CompiledGraphModel(
                values,
                List.of(multiOutput, independent, merge, fanOut, backward),
                List.of(inputA, inputB, unusedInput),
                List.of(secondOutput, fanOutOutput, backwardOutput),
                Map.of(
                        multiOutput.id(), GraphPhase.FORWARD,
                        independent.id(), GraphPhase.FORWARD,
                        merge.id(), GraphPhase.FORWARD,
                        fanOut.id(), GraphPhase.FORWARD,
                        backward.id(), GraphPhase.BACKWARD));

        PlannedPartition cpuFirst = new PlannedPartition(
                new BackendId(new String("cpu")),
                List.of(new NodeId(10), new NodeId(11)));
        PlannedPartition metal = partition("metal", 12);
        PlannedPartition cpuLater = new PlannedPartition(
                new BackendId(new String("cpu")),
                List.of(new NodeId(13), new NodeId(14)));
        List<PlannedPartition> supplied =
                new ArrayList<>(List.of(cpuFirst, metal, cpuLater));

        LogicalMemoryPlan result = LogicalMemoryPlanning.plan(graph, supplied);
        supplied.clear();

        Map<ValueId, LogicalMemoryRequirement> byValue = new HashMap<>();
        for (LogicalMemoryRequirement requirement : result.requirements()) {
            byValue.put(requirement.valueId(), requirement);
        }
        LogicalMemoryRequirement inputARequirement = byValue.get(inputA);
        LogicalMemoryRequirement unusedInputRequirement = byValue.get(unusedInput);
        LogicalMemoryRequirement firstOutputRequirement = byValue.get(firstOutput);
        LogicalMemoryRequirement secondOutputRequirement = byValue.get(secondOutput);
        LogicalMemoryRequirement mergeRequirement = byValue.get(mergeOutput);
        LogicalMemoryRequirement fanOutRequirement = byValue.get(fanOutOutput);
        LogicalMemoryRequirement unusedProducedRequirement = byValue.get(unusedProduced);

        assertAll(
                () -> assertEquals(
                        values.stream().map(GraphValue::id).toList(),
                        result.requirements().stream()
                                .map(LogicalMemoryRequirement::valueId)
                                .toList()),
                () -> assertSame(values.get(0).id(), result.requirements().get(0).valueId()),
                () -> assertSame(values.get(0).descriptor(),
                        result.requirements().get(0).descriptor()),
                () -> assertTrue(inputARequirement.producerPartition().isEmpty()),
                () -> assertEquals(List.of(cpuFirst), inputARequirement.consumerPartitions()),
                () -> assertSame(cpuFirst, inputARequirement.consumerPartitions().getFirst()),
                () -> assertTrue(unusedInputRequirement.producerPartition().isEmpty()),
                () -> assertTrue(unusedInputRequirement.consumerPartitions().isEmpty()),
                () -> assertFalse(unusedInputRequirement.graphOutput()),
                () -> assertSame(dynamicDescriptor, firstOutputRequirement.descriptor()),
                () -> assertSame(cpuFirst,
                        firstOutputRequirement.producerPartition().orElseThrow()),
                () -> assertEquals(
                        List.of(metal, cpuLater),
                        firstOutputRequirement.consumerPartitions()),
                () -> assertSame(metal, firstOutputRequirement.consumerPartitions().get(0)),
                () -> assertSame(cpuLater, firstOutputRequirement.consumerPartitions().get(1)),
                () -> assertTrue(secondOutputRequirement.graphOutput()),
                () -> assertEquals(
                        List.of(cpuLater), secondOutputRequirement.consumerPartitions()),
                () -> assertSame(metal, mergeRequirement.producerPartition().orElseThrow()),
                () -> assertEquals(List.of(cpuLater), mergeRequirement.consumerPartitions()),
                () -> assertTrue(fanOutRequirement.graphOutput()),
                () -> assertTrue(fanOutRequirement.consumerPartitions().isEmpty()),
                () -> assertSame(expressionDescriptor, unusedProducedRequirement.descriptor()),
                () -> assertSame(cpuLater,
                        unusedProducedRequirement.producerPartition().orElseThrow()),
                () -> assertTrue(unusedProducedRequirement.consumerPartitions().isEmpty()),
                () -> assertFalse(unusedProducedRequirement.graphOutput()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> result.requirements().clear()));
    }

    @Test
    void zeroNodePassThroughGraphProducesInputOutputRequirementFromEmptyPartitions() {
        ValueId passThrough = valueId(5);
        GraphValue value = value(passThrough);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(value),
                List.of(),
                List.of(passThrough),
                List.of(new ValueId(5)),
                Map.of());

        LogicalMemoryPlan result = LogicalMemoryPlanning.plan(graph, List.of());
        LogicalMemoryRequirement requirement = result.requirements().getFirst();
        IllegalArgumentException extra = assertThrows(
                IllegalArgumentException.class,
                () -> LogicalMemoryPlanning.plan(graph, List.of(partition("cpu", 99))));

        assertAll(
                () -> assertEquals(1, result.requirements().size()),
                () -> assertSame(value.id(), requirement.valueId()),
                () -> assertSame(value.descriptor(), requirement.descriptor()),
                () -> assertTrue(requirement.producerPartition().isEmpty()),
                () -> assertTrue(requirement.consumerPartitions().isEmpty()),
                () -> assertTrue(requirement.graphOutput()),
                () -> assertEquals(
                        "partitions[0].nodeIds[0] references unknown NodeId[value=99]",
                        extra.getMessage()));
    }

    private static CompiledGraphModel linearGraph(long... ids) {
        List<GraphValue> values = new ArrayList<>();
        List<CompiledNode> nodes = new ArrayList<>();
        Map<NodeId, GraphPhase> phases = new HashMap<>();
        ValueId input = valueId(100);
        values.add(value(input));
        ValueId previous = input;
        for (int index = 0; index < ids.length; index++) {
            ValueId output = valueId(101L + index);
            CompiledNode node = node(
                    new NodeId(ids[index]), List.of(previous), List.of(output));
            nodes.add(node);
            values.add(value(output));
            phases.put(node.id(), GraphPhase.FORWARD);
            previous = output;
        }
        return new CompiledGraphModel(
                values, nodes, List.of(input), List.of(previous), phases);
    }

    private static PlannedPartition partition(String owner, long... nodeIds) {
        return new PlannedPartition(
                new BackendId(owner),
                Arrays.stream(nodeIds).mapToObj(NodeId::new).toList());
    }

    private static GraphValue value(ValueId valueId) {
        return new GraphValue(valueId, descriptor());
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
