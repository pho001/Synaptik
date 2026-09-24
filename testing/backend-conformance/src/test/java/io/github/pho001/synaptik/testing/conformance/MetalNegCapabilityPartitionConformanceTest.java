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

/** Conformance checks for public Metal elementwise truth and Planning maximal closure. */
final class MetalNegCapabilityPartitionConformanceTest {
    /** Proves all five typed kinds and exact broadcasting are public while ABS fails closed. */
    @Test void advertisesExactElementwiseDomain() {
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
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    operation(kind), List.of(matrix, row), List.of(matrix))));
        }
        assertFalse(provider.supports(new OperationCapabilityQuery(
                operation(UnaryElementwiseKind.ABS), List.of(matrix), List.of(matrix))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                operation(BinaryArithmeticKind.ADD),
                List.of(matrix, descriptor(Shape.of(2, 2))),
                List.of(matrix))));
    }

    /** Proves a heterogeneous eligible chain becomes one whole maximal Metal partition. */
    @Test void mixedEligibleOccurrencesBecomeOneMaximalPartition() {
        TensorDescriptor descriptor = descriptor(Shape.of(4));
        ValueId input = new ValueId(0);
        ValueId right = new ValueId(1);
        ValueId negated = new ValueId(2);
        ValueId added = new ValueId(3);
        ValueId subtracted = new ValueId(4);
        ValueId multiplied = new ValueId(5);
        ValueId output = new ValueId(6);
        CompiledNode neg = new CompiledNode(
                new NodeId(0),
                operation(UnaryElementwiseKind.NEG),
                List.of(input),
                List.of(negated));
        CompiledNode add = new CompiledNode(
                new NodeId(1),
                operation(BinaryArithmeticKind.ADD),
                List.of(negated, right),
                List.of(added));
        CompiledNode sub = new CompiledNode(
                new NodeId(2),
                operation(BinaryArithmeticKind.SUB),
                List.of(added, right),
                List.of(subtracted));
        CompiledNode mul = new CompiledNode(
                new NodeId(3),
                operation(BinaryArithmeticKind.MUL),
                List.of(subtracted, right),
                List.of(multiplied));
        CompiledNode div = new CompiledNode(
                new NodeId(4),
                operation(BinaryArithmeticKind.DIV),
                List.of(multiplied, right),
                List.of(output));
        List<CompiledNode> nodes = List.of(neg, add, sub, mul, div);
        var graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(input, descriptor),
                        new GraphValue(right, descriptor),
                        new GraphValue(negated, descriptor),
                        new GraphValue(added, descriptor),
                        new GraphValue(subtracted, descriptor),
                        new GraphValue(multiplied, descriptor),
                        new GraphValue(output, descriptor)),
                nodes,
                List.of(input, right),
                List.of(added, output),
                Map.of(
                        neg.id(), GraphPhase.FORWARD,
                        add.id(), GraphPhase.FORWARD,
                        sub.id(), GraphPhase.FORWARD,
                        mul.id(), GraphPhase.FORWARD,
                        div.id(), GraphPhase.FORWARD));

        var partitions = MaximalSameOwnerPartitioning.partition(
                graph,
                Map.of(
                        neg.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        add.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        sub.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        mul.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        div.id(), MetalCapabilityProvider.METAL_BACKEND_ID));

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
