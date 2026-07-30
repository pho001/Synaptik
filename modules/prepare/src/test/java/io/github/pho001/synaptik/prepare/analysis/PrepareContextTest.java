package io.github.pho001.synaptik.prepare.analysis;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class PrepareContextTest {
    @Test
    void validatesTopLevelReferencesAndNodeOrderBeforeProjectionDetails() {
        Fixture fixture = fixture();
        List<CompiledNode> nodesWithNull =
                new ArrayList<>(Arrays.asList(fixture.nodes.getFirst(), null, null));

        NullPointerException nullPartition = assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(null, null, null, null, null, null));
        NullPointerException nullNodes = assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(
                        fixture.partition, null, null, null, null, null));
        NullPointerException nullValues = assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(
                        fixture.partition, fixture.nodes, null, null, null, null));
        NullPointerException nullRequirements = assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(
                        fixture.partition,
                        fixture.nodes,
                        fixture.values,
                        null,
                        null,
                        null));
        NullPointerException nullConstants = assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(
                        fixture.partition,
                        fixture.nodes,
                        fixture.values,
                        fixture.requirements,
                        null,
                        null));
        NullPointerException nullBackendInputs = assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(
                        fixture.partition,
                        fixture.nodes,
                        fixture.values,
                        fixture.requirements,
                        fixture.constants,
                        null));
        NullPointerException nullNode = assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(
                        fixture.partition,
                        nodesWithNull,
                        fixture.values,
                        fixture.requirements,
                        fixture.constants,
                        fixture.inputs));
        IllegalArgumentException wrongSize = assertThrows(
                IllegalArgumentException.class,
                () -> new PrepareContext<>(
                        fixture.partition,
                        List.of(fixture.nodes.getFirst()),
                        fixture.values,
                        fixture.requirements,
                        fixture.constants,
                        fixture.inputs));
        IllegalArgumentException wrongOrder = assertThrows(
                IllegalArgumentException.class,
                () -> new PrepareContext<>(
                        fixture.partition,
                        List.of(fixture.nodes.get(1), fixture.nodes.get(0)),
                        fixture.values,
                        fixture.requirements,
                        fixture.constants,
                        fixture.inputs));

        assertAll(
                () -> assertEquals("partition", nullPartition.getMessage()),
                () -> assertEquals("nodes", nullNodes.getMessage()),
                () -> assertEquals("values", nullValues.getMessage()),
                () -> assertEquals("memoryRequirements", nullRequirements.getMessage()),
                () -> assertEquals("constants", nullConstants.getMessage()),
                () -> assertEquals("backendInputs", nullBackendInputs.getMessage()),
                () -> assertEquals("nodes[1]", nullNode.getMessage()),
                () -> assertEquals(
                        "nodes size 1 does not match partition nodeIds size 2",
                        wrongSize.getMessage()),
                () -> assertEquals(
                        "nodes[0].id must equal partition.nodeIds[0]: expected "
                                + fixture.partition.nodeIds().get(0)
                                + " but was "
                                + fixture.nodes.get(1).id(),
                        wrongOrder.getMessage()));
    }

    @Test
    void validatesValueAndRequirementProjectionInDeterministicOrder() {
        Fixture fixture = fixture();
        List<GraphValue> valuesWithNull =
                new ArrayList<>(Arrays.asList(fixture.values.getFirst(), null, null));
        GraphValue duplicate = new GraphValue(
                fixture.values.getFirst().id(), fixture.values.getFirst().descriptor());
        GraphValue dynamic = new GraphValue(
                new ValueId(9),
                new TensorDescriptor(
                        DataType.FLOAT32,
                        Shape.ofDimensions(new DynamicDimension("N")),
                        Optional.empty(),
                        false));
        List<LogicalMemoryRequirement> requirementsWithNull =
                new ArrayList<>(Arrays.asList(fixture.requirements.getFirst(), null, null));
        LogicalMemoryRequirement duplicateRequirement = new LogicalMemoryRequirement(
                fixture.requirements.getFirst().valueId(),
                fixture.requirements.getFirst().descriptor(),
                Optional.empty(),
                List.of(fixture.partition),
                false);

        NullPointerException nullValue = assertThrows(
                NullPointerException.class,
                () -> context(fixture, valuesWithNull, fixture.requirements, fixture.constants));
        IllegalArgumentException duplicateValue = assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        List.of(fixture.values.getFirst(), duplicate),
                        List.of(
                                fixture.requirements.getFirst(),
                                fixture.requirements.getFirst()),
                        Map.of()));
        IllegalArgumentException dynamicShape = assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        List.of(dynamic),
                        List.of(requirement(dynamic, Optional.empty())),
                        Map.of()));
        NullPointerException nullRequirement = assertThrows(
                NullPointerException.class,
                () -> context(fixture, fixture.values, requirementsWithNull, fixture.constants));
        IllegalArgumentException duplicateMemory = assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        fixture.values,
                        List.of(fixture.requirements.getFirst(), duplicateRequirement),
                        Map.of()));
        IllegalArgumentException missingNodeValue = assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        fixture.values.subList(1, fixture.values.size()),
                        fixture.requirements.subList(1, fixture.requirements.size()),
                        Map.of()));

        assertAll(
                () -> assertEquals("values[1]", nullValue.getMessage()),
                () -> assertEquals(
                        "values[1].id duplicates " + duplicate.id(),
                        duplicateValue.getMessage()),
                () -> assertEquals(
                        "values[0].descriptor.shape must be fully static: "
                                + dynamic.descriptor().shape(),
                        dynamicShape.getMessage()),
                () -> assertEquals("memoryRequirements[1]", nullRequirement.getMessage()),
                () -> assertEquals(
                        "memoryRequirements[1].valueId duplicates "
                                + duplicateRequirement.valueId(),
                        duplicateMemory.getMessage()),
                () -> assertEquals(
                        "nodes[0].inputs[0] is absent from values: "
                                + fixture.values.getFirst().id(),
                        missingNodeValue.getMessage()));
    }

    @Test
    void validatesOneMatchingRequirementAndExactTypedGraphInputConstants() {
        Fixture fixture = fixture();
        LogicalMemoryRequirement mismatched = new LogicalMemoryRequirement(
                fixture.values.getFirst().id(),
                descriptor(DataType.FLOAT64),
                Optional.empty(),
                List.of(fixture.partition),
                false);
        LogicalMemoryRequirement extra = requirement(
                new GraphValue(new ValueId(20), descriptor(DataType.FLOAT32)),
                Optional.empty());
        Map<ValueId, ScalarValue> absentConstant =
                Map.of(new ValueId(50), ScalarValue.float32(1.0f));
        Map<ValueId, ScalarValue> producedConstant =
                Map.of(fixture.values.get(1).id(), ScalarValue.float32(1.0f));
        Map<ValueId, ScalarValue> wrongType =
                Map.of(fixture.values.getFirst().id(), ScalarValue.float64(1.0));

        IllegalArgumentException missing = assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        fixture.values,
                        fixture.requirements.subList(0, fixture.requirements.size() - 1),
                        Map.of()));
        IllegalArgumentException descriptorMismatch = assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        fixture.values,
                        List.of(
                                mismatched,
                                fixture.requirements.get(1),
                                fixture.requirements.get(2)),
                        Map.of()));
        IllegalArgumentException extraRequirement = assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        fixture.values,
                        List.of(
                                fixture.requirements.get(0),
                                fixture.requirements.get(1),
                                fixture.requirements.get(2),
                                extra),
                        Map.of()));
        IllegalArgumentException absent = assertThrows(
                IllegalArgumentException.class,
                () -> context(fixture, fixture.values, fixture.requirements, absentConstant));
        IllegalArgumentException notInput = assertThrows(
                IllegalArgumentException.class,
                () -> context(fixture, fixture.values, fixture.requirements, producedConstant));
        IllegalArgumentException typeMismatch = assertThrows(
                IllegalArgumentException.class,
                () -> context(fixture, fixture.values, fixture.requirements, wrongType));

        assertAll(
                () -> assertEquals(
                        "memoryRequirements has no entry for values[2].id "
                                + fixture.values.get(2).id(),
                        missing.getMessage()),
                () -> assertEquals(
                        "memoryRequirements[0].descriptor does not match values[0].descriptor for "
                                + fixture.values.getFirst().id(),
                        descriptorMismatch.getMessage()),
                () -> assertEquals(
                        "memoryRequirements[3].valueId is absent from values: "
                                + extra.valueId(),
                        extraRequirement.getMessage()),
                () -> assertEquals(
                        "constants key is absent from values: " + absentConstant.keySet().iterator().next(),
                        absent.getMessage()),
                () -> assertEquals(
                        "constants key is not a projected graph input: "
                                + producedConstant.keySet().iterator().next(),
                        notInput.getMessage()),
                () -> assertEquals(
                        "constants["
                                + fixture.values.getFirst().id()
                                + "] data type FLOAT64 does not match descriptor data type FLOAT32",
                        typeMismatch.getMessage()));
    }

    @Test
    void snapshotsCollectionsRetainsExactReferencesAndPreservesEncounterOrder() {
        Fixture fixture = fixture();
        List<CompiledNode> suppliedNodes = new ArrayList<>(fixture.nodes);
        List<GraphValue> suppliedValues = new ArrayList<>(fixture.values);
        List<LogicalMemoryRequirement> suppliedRequirements =
                new ArrayList<>(fixture.requirements);
        Map<ValueId, ScalarValue> suppliedConstants = new LinkedHashMap<>();
        ScalarValue firstConstant = fixture.constants.values().iterator().next();
        suppliedConstants.put(fixture.values.getFirst().id(), firstConstant);

        PrepareContext<FakeInputs> context = new PrepareContext<>(
                fixture.partition,
                suppliedNodes,
                suppliedValues,
                suppliedRequirements,
                suppliedConstants,
                fixture.inputs);
        suppliedNodes.clear();
        suppliedValues.clear();
        suppliedRequirements.clear();
        suppliedConstants.clear();

        assertAll(
                () -> assertSame(fixture.partition, context.partition()),
                () -> assertEquals(fixture.nodes, context.nodes()),
                () -> assertSame(fixture.nodes.getFirst(), context.nodes().getFirst()),
                () -> assertNotSame(suppliedNodes, context.nodes()),
                () -> assertEquals(fixture.values, context.values()),
                () -> assertSame(fixture.values.getFirst(), context.values().getFirst()),
                () -> assertEquals(fixture.requirements, context.memoryRequirements()),
                () -> assertSame(
                        fixture.requirements.getFirst(),
                        context.memoryRequirements().getFirst()),
                () -> assertEquals(
                        List.of(fixture.values.getFirst().id()),
                        context.constants().keySet().stream().toList()),
                () -> assertSame(firstConstant, context.constants().values().iterator().next()),
                () -> assertSame(fixture.inputs, context.backendInputs()),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> context.nodes().clear()),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> context.values().clear()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> context.memoryRequirements().clear()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> context.constants().clear()));
    }

    private static PrepareContext<FakeInputs> context(
            Fixture fixture,
            List<GraphValue> values,
            List<LogicalMemoryRequirement> requirements,
            Map<ValueId, ScalarValue> constants) {
        return new PrepareContext<>(
                fixture.partition,
                fixture.nodes,
                values,
                requirements,
                constants,
                fixture.inputs);
    }

    private static Fixture fixture() {
        ValueId input = new ValueId(0);
        ValueId middle = new ValueId(1);
        ValueId output = new ValueId(2);
        CompiledNode first = node(10, List.of(input), List.of(middle));
        CompiledNode second = node(11, List.of(middle), List.of(output));
        PlannedPartition partition = new PlannedPartition(
                new BackendId("cpu"), List.of(first.id(), second.id()));
        List<GraphValue> values = List.of(
                new GraphValue(input, descriptor(DataType.FLOAT32)),
                new GraphValue(middle, descriptor(DataType.FLOAT32)),
                new GraphValue(output, descriptor(DataType.FLOAT32)));
        List<LogicalMemoryRequirement> requirements = List.of(
                new LogicalMemoryRequirement(
                        input,
                        values.get(0).descriptor(),
                        Optional.empty(),
                        List.of(partition),
                        false),
                new LogicalMemoryRequirement(
                        middle,
                        values.get(1).descriptor(),
                        Optional.of(partition),
                        List.of(partition),
                        false),
                new LogicalMemoryRequirement(
                        output,
                        values.get(2).descriptor(),
                        Optional.of(partition),
                        List.of(),
                        true));
        ScalarValue constant = ScalarValue.float32(3.0f);
        return new Fixture(
                partition,
                List.of(first, second),
                values,
                requirements,
                Map.of(input, constant),
                new FakeInputs("cpu-v1"));
    }

    private static LogicalMemoryRequirement requirement(
            GraphValue value, Optional<PlannedPartition> producer) {
        return new LogicalMemoryRequirement(
                value.id(), value.descriptor(), producer, List.of(), false);
    }

    private static CompiledNode node(long id, List<ValueId> inputs, List<ValueId> outputs) {
        return new CompiledNode(
                new NodeId(id),
                new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE),
                inputs,
                outputs);
    }

    private static TensorDescriptor descriptor(DataType dataType) {
        return new TensorDescriptor(dataType, Shape.of(2, 3), Optional.empty(), false);
    }

    private record FakeInputs(String target) implements BackendAnalysisInputs {}

    private record Fixture(
            PlannedPartition partition,
            List<CompiledNode> nodes,
            List<GraphValue> values,
            List<LogicalMemoryRequirement> requirements,
            Map<ValueId, ScalarValue> constants,
            FakeInputs inputs) {}

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
