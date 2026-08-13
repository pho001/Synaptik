package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.random.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import org.junit.jupiter.api.Test;

public class CpuRandomLoweringTest {
    @Test void lowersZeroInputInitializerAndFiveBoundaryDropoutExactly() {
        var initial = new CpuPartitionLowering().lower(initialContext(0x1234L, -7L));
        var dropout = new CpuPartitionLowering().lower(dropoutContext(DataType.FLOAT32,
                Shape.of(2, 3), .25d));
        assertAll(
                () -> assertEquals(CpuRandomIr.Family.INITIAL_STATE,
                        ((CpuRandomIr) initial.portableKernelIr()).family()),
                () -> assertEquals(1, initial.boundaryValues().size()),
                () -> assertEquals(0, initial.elementCount()),
                () -> assertEquals(List.of(DataType.INT64), initial.boundaryDataTypes()),
                () -> assertEquals(CpuRandomIr.Family.DROPOUT,
                        ((CpuRandomIr) dropout.portableKernelIr()).family()),
                () -> assertEquals(List.of(DataType.FLOAT32, DataType.INT64, DataType.FLOAT32,
                        DataType.BOOL, DataType.INT64), dropout.boundaryDataTypes()),
                () -> assertEquals(6, dropout.elementCount()),
                () -> assertTrue(dropout.randomGeometry().isPresent()));
    }

    @Test void rejectsBfloat16DynamicUnresolvedAndNonInjectiveOutputs() {
        assertThrows(IllegalArgumentException.class, () -> new CpuPartitionLowering().lower(
                dropoutContext(DataType.BFLOAT16, Shape.of(3), .2d)));
        assertThrows(IllegalArgumentException.class, () -> {
            var base = dropoutContext(DataType.FLOAT64, Shape.of(2), .2d);
            var values = new ArrayList<>(base.values());
            var memory = new ArrayList<>(base.memoryRequirements());
            TensorDescriptor old = values.get(2).descriptor();
            var unresolved = new TensorDescriptor(old.dataType(), old.shape(), Optional.empty(),
                    old.requiresGrad());
            values.set(2, new GraphValue(values.get(2).id(), unresolved));
            memory.set(2, new LogicalMemoryRequirement(values.get(2).id(), unresolved,
                    Optional.of(base.partition()), List.of(), true));
            var invalid = new PrepareContext<>(base.partition(), base.nodes(), values,
                    memory, base.constants(), base.backendInputs());
            new CpuPartitionLowering().lower(invalid);
        });
    }

    @Test void rejectsZeroStrideStateInputBeforeResourceDeclaration() {
        var base = dropoutContext(DataType.FLOAT64, Shape.of(2), .2d);
        var values = new ArrayList<>(base.values());
        var memory = new ArrayList<>(base.memoryRequirements());
        Shape stateShape = Shape.of(2);
        var aliasedState = new TensorDescriptor(DataType.INT64, stateShape, Optional.of(
                LayoutDescriptor.of(stateShape, new long[] {0}, 0, true)), false);
        values.set(1, new GraphValue(values.get(1).id(), aliasedState));
        memory.set(1, new LogicalMemoryRequirement(values.get(1).id(), aliasedState,
                Optional.empty(), List.of(base.partition()), false));
        var invalid = new PrepareContext<>(base.partition(), base.nodes(), values, memory,
                base.constants(), base.backendInputs());
        assertThrows(IllegalArgumentException.class,
                () -> new CpuPartitionLowering().lower(invalid));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> initialContext(long key, long counter) {
        Shape stateShape = Shape.of(2); TensorDescriptor state = descriptor(DataType.INT64, stateShape);
        ValueId output = new ValueId(0); NodeId nodeId = new NodeId(0);
        var node = new CompiledNode(nodeId, new Operation(GraphRngKind.INITIAL_STATE,
                new GraphRngStateAttrs(key, counter)), List.of(), List.of(output));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(nodeId));
        return new PrepareContext<>(partition, List.of(node), List.of(new GraphValue(output, state)),
                List.of(new LogicalMemoryRequirement(output, state, Optional.of(partition), List.of(), true)),
                Map.of(), CpuPartitionAnalysisInputs.DEFAULT);
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> dropoutContext(
            DataType type, Shape shape, double probability) {
        Shape stateShape = Shape.of(2);
        return dropoutContext(type, shape, probability, List.of(
                LayoutDescriptor.contiguous(shape), LayoutDescriptor.contiguous(stateShape),
                LayoutDescriptor.contiguous(shape), LayoutDescriptor.contiguous(shape),
                LayoutDescriptor.contiguous(stateShape)));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> dropoutContext(
            DataType type, Shape shape, double probability, List<LayoutDescriptor> layouts) {
        if (layouts.size() != 5) throw new IllegalArgumentException("five layouts required");
        Shape stateShape = Shape.of(2);
        TensorDescriptor value = descriptor(type, shape, layouts.get(0));
        TensorDescriptor state = descriptor(DataType.INT64, stateShape, layouts.get(1));
        TensorDescriptor output = descriptor(type, shape, layouts.get(2));
        TensorDescriptor mask = descriptor(DataType.BOOL, shape, layouts.get(3));
        TensorDescriptor next = descriptor(DataType.INT64, stateShape, layouts.get(4));
        List<TensorDescriptor> descriptors = List.of(value, state, output, mask, next);
        List<ValueId> ids = java.util.stream.IntStream.range(0, 5)
                .mapToObj(ValueId::new).toList();
        NodeId nodeId = new NodeId(0);
        var node = new CompiledNode(nodeId, new Operation(DropoutKind.DROPOUT,
                new DropoutAttrs(probability)), ids.subList(0, 2), ids.subList(2, 5));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(nodeId));
        var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < 5; i++) {
            values.add(new GraphValue(ids.get(i), descriptors.get(i)));
            boolean input = i < 2;
            memory.add(new LogicalMemoryRequirement(ids.get(i), descriptors.get(i),
                    input ? Optional.empty() : Optional.of(partition), input ? List.of(partition) : List.of(),
                    !input));
        }
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(type, shape, Optional.of(layout), false);
    }
}
