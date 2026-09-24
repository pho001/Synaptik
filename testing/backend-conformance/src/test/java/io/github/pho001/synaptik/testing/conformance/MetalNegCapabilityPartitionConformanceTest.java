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
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
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
        assertTrue(provider.supports(new OperationCapabilityQuery(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, operation(UnaryElementwiseKind.NEG), List.of(matrix), List.of(matrix))));
        for (BinaryArithmeticKind kind : List.of(
                BinaryArithmeticKind.ADD,
                BinaryArithmeticKind.SUB,
                BinaryArithmeticKind.MUL,
                BinaryArithmeticKind.DIV)) {
            assertFalse(provider.supports(new OperationCapabilityQuery(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, operation(kind), List.of(matrix, row), List.of(matrix))));
        }
        assertFalse(provider.supports(new OperationCapabilityQuery(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, operation(UnaryElementwiseKind.ABS), List.of(matrix), List.of(matrix))));
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

    /** Proves chained affine views, CONTIGUOUS, and NEG remain one maximal Metal partition. */
    @Test
    void eligibleAffineCompositionBecomesOneMaximalPartition() {
        ValueId input = new ValueId(10);
        ValueId reshaped = new ValueId(11);
        ValueId expanded = new ValueId(12);
        ValueId contiguous = new ValueId(13);
        ValueId negated = new ValueId(14);
        ValueId output = new ValueId(15);
        List<CompiledNode> nodes = List.of(
                new CompiledNode(
                        new NodeId(10),
                        new Operation(
                                ShapeTransformKind.RESHAPE,
                                new TargetShapeAttrs(Shape.of(2, 1, 3))),
                        List.of(input),
                        List.of(reshaped)),
                new CompiledNode(
                        new NodeId(11),
                        new Operation(
                                ShapeTransformKind.EXPAND,
                                new TargetShapeAttrs(Shape.of(2, 4, 3))),
                        List.of(reshaped),
                        List.of(expanded)),
                new CompiledNode(
                        new NodeId(12),
                        new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE),
                        List.of(expanded),
                        List.of(contiguous)),
                new CompiledNode(
                        new NodeId(13),
                        operation(UnaryElementwiseKind.NEG),
                        List.of(contiguous),
                        List.of(negated)),
                new CompiledNode(
                        new NodeId(14),
                        new Operation(
                                ShapeTransformKind.RESHAPE,
                                new TargetShapeAttrs(Shape.of(4, 6))),
                        List.of(negated),
                        List.of(output)));
        List<GraphValue> values = List.of(
                new GraphValue(input, descriptor(Shape.of(6))),
                new GraphValue(reshaped, view(Shape.of(2, 1, 3), 3, 3, 1)),
                new GraphValue(expanded, view(Shape.of(2, 4, 3), 3, 0, 1)),
                new GraphValue(contiguous, descriptor(Shape.of(2, 4, 3))),
                new GraphValue(negated, descriptor(Shape.of(2, 4, 3))),
                new GraphValue(output, view(Shape.of(4, 6), 6, 1)));
        Map<NodeId, GraphPhase> phases = nodes.stream().collect(
                java.util.stream.Collectors.toMap(
                        CompiledNode::id, ignored -> GraphPhase.FORWARD));
        var graph = new CompiledGraphModel(
                values, nodes, List.of(input), List.of(expanded, output), phases);
        Map<NodeId, io.github.pho001.synaptik.backend.contract.BackendId> owners =
                nodes.stream().collect(java.util.stream.Collectors.toMap(
                        CompiledNode::id,
                        ignored -> MetalCapabilityProvider.METAL_BACKEND_ID));

        var partitions = MaximalSameOwnerPartitioning.partition(graph, owners);
        assertEquals(1, partitions.size());
        assertSame(MetalCapabilityProvider.METAL_BACKEND_ID, partitions.getFirst().owner());
        assertEquals(nodes.stream().map(CompiledNode::id).toList(),
                partitions.getFirst().nodeIds());
    }

    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static TensorDescriptor view(Shape shape, long... strides) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, strides, 0L, true)), false);
    }

    private static Operation operation(UnaryElementwiseKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }

    private static Operation operation(BinaryArithmeticKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }
}
