package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
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
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.partition.MaximalSameOwnerPartitioning;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Conformance checks for public Metal capability truth and Planning maximal closure. */
final class MetalNegCapabilityPartitionConformanceTest {
    /** Proves NEG is public while all binary arithmetic and ABS fail closed. */
    @Test void advertisesExactNegationDomain() {
        var provider = new MetalCapabilityProvider();
        TensorDescriptor matrix = descriptor(Shape.of(2, 3));
        TensorDescriptor row = descriptor(Shape.of(3));
        assertTrue(provider.supports(new OperationCapabilityQuery(
                operation(UnaryElementwiseKind.NEG), List.of(matrix), List.of(matrix))));
        for (BinaryArithmeticKind kind : List.of(
                BinaryArithmeticKind.ADD,
                BinaryArithmeticKind.SUB,
                BinaryArithmeticKind.MUL,
                BinaryArithmeticKind.DIV)) {
            assertFalse(provider.supports(new OperationCapabilityQuery(
                    operation(kind), List.of(matrix, row), List.of(matrix))));
        }
        assertFalse(provider.supports(new OperationCapabilityQuery(
                operation(UnaryElementwiseKind.ABS), List.of(matrix), List.of(matrix))));
    }

    /** Proves an eligible NEG chain becomes one whole maximal Metal partition. */
    @Test void eligibleNegOccurrencesBecomeOneMaximalPartition() {
        TensorDescriptor descriptor = descriptor(Shape.of(4));
        ValueId input = new ValueId(0);
        ValueId negated = new ValueId(1);
        ValueId restored = new ValueId(2);
        ValueId output = new ValueId(3);
        CompiledNode first = new CompiledNode(
                new NodeId(0),
                operation(UnaryElementwiseKind.NEG),
                List.of(input),
                List.of(negated));
        CompiledNode second = new CompiledNode(
                new NodeId(1),
                operation(UnaryElementwiseKind.NEG),
                List.of(negated),
                List.of(restored));
        CompiledNode third = new CompiledNode(
                new NodeId(2),
                operation(UnaryElementwiseKind.NEG),
                List.of(restored),
                List.of(output));
        List<CompiledNode> nodes = List.of(first, second, third);
        var graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(input, descriptor),
                        new GraphValue(negated, descriptor),
                        new GraphValue(restored, descriptor),
                        new GraphValue(output, descriptor)),
                nodes,
                List.of(input),
                List.of(restored, output),
                Map.of(
                        first.id(), GraphPhase.FORWARD,
                        second.id(), GraphPhase.FORWARD,
                        third.id(), GraphPhase.FORWARD));

        var partitions = MaximalSameOwnerPartitioning.partition(
                graph,
                Map.of(
                        first.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        second.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        third.id(), MetalCapabilityProvider.METAL_BACKEND_ID));

        assertEquals(1, partitions.size());
        assertSame(MetalCapabilityProvider.METAL_BACKEND_ID, partitions.getFirst().owner());
        assertEquals(nodes.stream().map(CompiledNode::id).toList(),
                partitions.getFirst().nodeIds());
    }

    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static Operation operation(UnaryElementwiseKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }

    private static Operation operation(BinaryArithmeticKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }
}
