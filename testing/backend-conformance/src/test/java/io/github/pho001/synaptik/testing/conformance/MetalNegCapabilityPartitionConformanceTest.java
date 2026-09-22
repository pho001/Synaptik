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
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.partition.MaximalSameOwnerPartitioning;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Conformance checks for public Metal NEG truth and Planning's unchanged maximal closure. */
final class MetalNegCapabilityPartitionConformanceTest {
    /** Proves exact supported truth remains public while a neighboring operation fails closed. */
    @Test void advertisesOnlyEligibleFloat32Neg() {
        var provider = new MetalCapabilityProvider();
        TensorDescriptor descriptor = descriptor(Shape.of(2, 3));
        assertTrue(provider.supports(new OperationCapabilityQuery(
                operation(UnaryElementwiseKind.NEG), List.of(descriptor), List.of(descriptor))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                operation(UnaryElementwiseKind.ABS), List.of(descriptor), List.of(descriptor))));
    }

    /** Proves adjacent eligible NEG occurrences become one whole maximal Metal partition. */
    @Test void adjacentEligibleOccurrencesRemainOneMaximalPartition() {
        ValueId input = new ValueId(0), middle = new ValueId(1), output = new ValueId(2);
        TensorDescriptor descriptor = descriptor(Shape.of(4));
        CompiledNode first = new CompiledNode(new NodeId(0), operation(UnaryElementwiseKind.NEG),
                List.of(input), List.of(middle));
        CompiledNode second = new CompiledNode(new NodeId(1), operation(UnaryElementwiseKind.NEG),
                List.of(middle), List.of(output));
        var graph = new CompiledGraphModel(
                List.of(new GraphValue(input, descriptor), new GraphValue(middle, descriptor),
                        new GraphValue(output, descriptor)),
                List.of(first, second), List.of(input), List.of(output),
                Map.of(first.id(), GraphPhase.FORWARD, second.id(), GraphPhase.FORWARD));

        var partitions = MaximalSameOwnerPartitioning.partition(graph,
                Map.of(first.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        second.id(), MetalCapabilityProvider.METAL_BACKEND_ID));

        assertEquals(1, partitions.size());
        assertSame(MetalCapabilityProvider.METAL_BACKEND_ID, partitions.getFirst().owner());
        assertEquals(List.of(first.id(), second.id()), partitions.getFirst().nodeIds());
    }

    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static Operation operation(UnaryElementwiseKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }
}
