package io.github.pho001.synaptik.prepare.analysis;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class PartitionDagTest {
    @Test
    void projectsBranchingRepeatedMultiOutputAndDisconnectedFactsInPortOrder() {
        ValueId externalA = value(0);
        ValueId externalB = value(1);
        ValueId left = value(2);
        ValueId right = value(3);
        ValueId repeatedResult = value(4);
        ValueId branchResult = value(5);
        ValueId disconnectedResult = value(6);
        ValueId diamondResult = value(7);
        CompiledNode source = node(10, List.of(externalA), List.of(left, right));
        CompiledNode repeated = node(11, List.of(left, left), List.of(repeatedResult));
        CompiledNode branch = node(12, List.of(right), List.of(branchResult));
        CompiledNode disconnected =
                node(13, List.of(externalB), List.of(disconnectedResult));
        CompiledNode diamond =
                node(14, List.of(repeatedResult, branchResult), List.of(diamondResult));
        List<CompiledNode> nodes = List.of(source, repeated, branch, disconnected, diamond);
        PlannedPartition partition = partition(nodes);

        PartitionDag dag = new PartitionDag(partition, nodes);

        assertAll(
                () -> assertSame(partition, dag.partition()),
                () -> assertEquals(nodes, dag.nodes()),
                () -> assertSame(source, dag.nodes().getFirst()),
                () -> assertSame(source, dag.node(source.id()).orElseThrow()),
                () -> assertTrue(dag.node(new NodeId(99)).isEmpty()),
                () -> assertEquals(
                        new PartitionDag.ProducerOccurrence(right, 0, source, 1),
                        dag.producer(right).orElseThrow()),
                () -> assertTrue(dag.producer(externalA).isEmpty()),
                () -> assertEquals(
                        List.of(
                                new PartitionDag.ConsumerOccurrence(left, 1, repeated, 0),
                                new PartitionDag.ConsumerOccurrence(left, 1, repeated, 1)),
                        dag.consumers(left)),
                () -> assertEquals(
                        List.of(
                                new PartitionDag.ConsumerOccurrence(externalA, 0, source, 0),
                                new PartitionDag.ConsumerOccurrence(
                                        externalB, 3, disconnected, 0)),
                        dag.externalInputs()),
                () -> assertEquals(
                        List.of(
                                new PartitionDag.Edge(
                                        new PartitionDag.ProducerOccurrence(left, 0, source, 0),
                                        new PartitionDag.ConsumerOccurrence(
                                                left, 1, repeated, 0)),
                                new PartitionDag.Edge(
                                        new PartitionDag.ProducerOccurrence(left, 0, source, 0),
                                        new PartitionDag.ConsumerOccurrence(
                                                left, 1, repeated, 1)),
                                new PartitionDag.Edge(
                                        new PartitionDag.ProducerOccurrence(right, 0, source, 1),
                                        new PartitionDag.ConsumerOccurrence(
                                                right, 2, branch, 0)),
                                new PartitionDag.Edge(
                                        new PartitionDag.ProducerOccurrence(
                                                repeatedResult, 1, repeated, 0),
                                        new PartitionDag.ConsumerOccurrence(
                                                repeatedResult, 4, diamond, 0)),
                                new PartitionDag.Edge(
                                        new PartitionDag.ProducerOccurrence(
                                                branchResult, 2, branch, 0),
                                        new PartitionDag.ConsumerOccurrence(
                                                branchResult, 4, diamond, 1))),
                        dag.edges()),
                () -> assertEquals(List.of(disconnected, diamond), dag.localSinks()),
                () -> assertTrue(dag.consumers(value(99)).isEmpty()),
                () -> assertThrows(NullPointerException.class, () -> dag.node(null)),
                () -> assertThrows(NullPointerException.class, () -> dag.producer(null)),
                () -> assertThrows(NullPointerException.class, () -> dag.consumers(null)),
                () -> assertThrows(UnsupportedOperationException.class, () -> dag.nodes().clear()),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> dag.consumers(left).clear()),
                () -> assertThrows(UnsupportedOperationException.class, () -> dag.edges().clear()),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> dag.externalInputs().clear()),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> dag.localSinks().clear()));
    }

    @Test
    void supportsOneNodeAndLinearTopologiesAndSnapshotsMembership() {
        ValueId input = value(0);
        ValueId middle = value(1);
        ValueId output = value(2);
        CompiledNode first = node(10, List.of(input), List.of(middle));
        CompiledNode second = node(11, List.of(middle), List.of(output));
        var supplied = new ArrayList<>(List.of(first, second));
        PlannedPartition partition = partition(supplied);
        PartitionDag linear = new PartitionDag(partition, supplied);
        supplied.clear();
        PartitionDag one = new PartitionDag(partition(List.of(first)), List.of(first));

        assertAll(
                () -> assertEquals(List.of(first, second), linear.nodes()),
                () -> assertEquals(1, linear.edges().size()),
                () -> assertEquals(List.of(second), linear.localSinks()),
                () -> assertEquals(List.of(first), one.nodes()),
                () -> assertTrue(one.edges().isEmpty()),
                        () -> assertEquals(List.of(first), one.localSinks()));
    }

    @Test
    void reliesOnUpstreamImmutableContractsForNonEmptyPartitionsAndWellFormedPorts() {
        ValueId value = value(0);
        List<ValueId> inputWithNull = new ArrayList<>(Arrays.asList((ValueId) null));

        assertAll(
                () -> assertEquals(
                        "nodeIds must not be empty",
                        assertThrows(
                                        IllegalArgumentException.class,
                                        () -> new PlannedPartition(
                                                new BackendId("cpu"), List.of()))
                                .getMessage()),
                () -> assertEquals(
                        "inputs[0]",
                        assertThrows(
                                        NullPointerException.class,
                                        () -> node(10, inputWithNull, List.of(value)))
                                .getMessage()),
                () -> assertEquals(
                        "outputs[1] duplicates " + value,
                        assertThrows(
                                        IllegalArgumentException.class,
                                        () -> node(10, List.of(), List.of(value, value)))
                                .getMessage()));
    }

    @Test
    void rejectsNullDuplicateAndPartitionDisagreementBeforePublishingFacts() {
        ValueId firstValue = value(0);
        ValueId secondValue = value(1);
        CompiledNode first = node(10, List.of(), List.of(firstValue));
        CompiledNode second = node(11, List.of(firstValue), List.of(secondValue));
        PlannedPartition partition = partition(List.of(first, second));
        List<CompiledNode> withNull = new ArrayList<>(Arrays.asList(first, null));
        CompiledNode duplicateId = node(10, List.of(), List.of(value(9)));

        assertAll(
                () -> assertEquals(
                        "partition",
                        assertThrows(NullPointerException.class,
                                        () -> new PartitionDag(null, List.of(first)))
                                .getMessage()),
                () -> assertEquals(
                        "nodes",
                        assertThrows(NullPointerException.class,
                                        () -> new PartitionDag(partition, null))
                                .getMessage()),
                () -> assertEquals(
                        "nodes[1]",
                        assertThrows(NullPointerException.class,
                                        () -> new PartitionDag(partition, withNull))
                                .getMessage()),
                () -> assertTrue(
                        assertThrows(IllegalArgumentException.class,
                                        () -> new PartitionDag(
                                                partition, List.of(first, duplicateId)))
                                .getMessage()
                                .contains("id duplicates")),
                () -> assertTrue(
                        assertThrows(IllegalArgumentException.class,
                                        () -> new PartitionDag(partition, List.of(first)))
                                .getMessage()
                                .contains("does not match partition nodeIds size")),
                () -> assertTrue(
                        assertThrows(IllegalArgumentException.class,
                                        () -> new PartitionDag(
                                                partition, List.of(second, first)))
                                .getMessage()
                                .contains("must equal partition.nodeIds[0]")));
    }

    @Test
    void rejectsDuplicateProducersAndNonTopologicalDependencies() {
        ValueId shared = value(0);
        CompiledNode firstProducer = node(10, List.of(), List.of(shared));
        CompiledNode duplicateProducer = node(11, List.of(), List.of(shared));
        CompiledNode selfDependent = node(12, List.of(shared), List.of(shared));
        CompiledNode earlyConsumer = node(13, List.of(shared), List.of(value(1)));
        CompiledNode laterProducer = node(14, List.of(), List.of(shared));

        assertAll(
                () -> assertTrue(
                        assertThrows(IllegalArgumentException.class,
                                        () -> new PartitionDag(
                                                partition(List.of(
                                                        firstProducer, duplicateProducer)),
                                                List.of(firstProducer, duplicateProducer)))
                                .getMessage()
                                .contains("duplicates produced value")),
                () -> assertTrue(
                        assertThrows(IllegalArgumentException.class,
                                        () -> new PartitionDag(
                                                partition(List.of(selfDependent)),
                                                List.of(selfDependent)))
                                .getMessage()
                                .contains("own or a later node")),
                () -> assertTrue(
                        assertThrows(IllegalArgumentException.class,
                                        () -> new PartitionDag(
                                                partition(List.of(earlyConsumer, laterProducer)),
                                                List.of(earlyConsumer, laterProducer)))
                                .getMessage()
                                .contains("own or a later node")));
    }

    @Test
    void pinsEqualityHashingTextAndOccurrenceValidation() {
        ValueId input = value(0);
        ValueId output = value(1);
        CompiledNode node = node(10, List.of(input), List.of(output));
        PlannedPartition partition = partition(List.of(node));
        PartitionDag first = new PartitionDag(partition, List.of(node));
        PartitionDag equal = new PartitionDag(partition, List.of(node));
        PartitionDag different = new PartitionDag(
                new PlannedPartition(new BackendId("gpu"), List.of(node.id())), List.of(node));

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, different),
                () -> assertEquals(
                        "PartitionDag[partition=" + partition + ", nodes=[" + node + "]]",
                        first.toString()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new PartitionDag.ProducerOccurrence(input, 0, node, 0)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new PartitionDag.ConsumerOccurrence(output, 0, node, 0)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new PartitionDag.Edge(
                                first.producer(output).orElseThrow(),
                                first.externalInputs().getFirst())));
    }

    private static PlannedPartition partition(List<CompiledNode> nodes) {
        return new PlannedPartition(
                new BackendId("cpu"), nodes.stream().map(CompiledNode::id).toList());
    }

    private static CompiledNode node(long id, List<ValueId> inputs, List<ValueId> outputs) {
        return new CompiledNode(
                new NodeId(id),
                new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE),
                inputs,
                outputs);
    }

    private static ValueId value(long value) {
        return new ValueId(value);
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
