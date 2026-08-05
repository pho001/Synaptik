package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable.BufferAccess;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

class CpuPointwiseAddCandidateSourceTest {
    @Test void buildsOrderedKernelsAndDeduplicatesSharedValuesByFirstEncounter() {
        PrepareContext<CpuPortableAnalysisInputs> context = chain(DataType.FLOAT32, Shape.of(4));
        var partition = new CpuPointwiseAddCandidateSource().candidates(context).getFirst();
        assertEquals(List.of(new ValueId(0), new ValueId(1), new ValueId(2), new ValueId(3)),
                partition.requirements().stream().map(requirement ->
                        ((io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Buffer)
                                requirement).valueId()).toList());
        assertEquals(2, partition.kernels().size());
        var first = partition.kernels().get(0);
        var second = partition.kernels().get(1);
        assertAll(
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1), new ValueId(2)),
                        first.bufferUses().stream().map(use -> use.requirement().valueId()).toList()),
                () -> assertEquals(List.of(new ValueId(2), new ValueId(0), new ValueId(3)),
                        second.bufferUses().stream().map(use -> use.requirement().valueId()).toList()),
                () -> assertSame(first.bufferUses().get(2).requirement(),
                        second.bufferUses().get(0).requirement()),
                () -> assertEquals(List.of(BufferAccess.READ_ONLY, BufferAccess.READ_ONLY,
                                BufferAccess.WRITE_ONLY),
                        first.specialization().arguments().stream()
                                .map(CpuKernelSpecialization.Argument::access).toList()),
                () -> assertTrue(first.specialization().arguments().stream().allMatch(argument ->
                        argument.carrier() == CpuKernelSpecialization.Carrier.MEMORY_SEGMENT)),
                () -> assertSame(CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                        first.specialization().executionMode()),
                () -> assertNotEquals(first.specialization().specializationFingerprint(),
                        new CpuPointwiseAddCandidateSource().candidates(
                                chain(DataType.INT32, Shape.of(4))).getFirst().kernels().getFirst()
                                .specialization().specializationFingerprint()));
    }

    @Test void rejectsAnyMixedUnsupportedPartitionBeforeReturningDeclarations() {
        var context = chain(DataType.FLOAT32, Shape.of(4));
        var first = context.nodes().getFirst();
        var unsupported = new CompiledNode(context.nodes().get(1).id(),
                new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(2)), List.of(new ValueId(3)));
        var mixed = new PrepareContext<>(context.partition(), List.of(first, unsupported),
                context.values(), context.memoryRequirements(), Map.of(), context.backendInputs());
        assertEquals("nodes[1] is not supported by CPU pointwise ADD",
                assertThrows(IllegalArgumentException.class,
                        () -> new CpuPointwiseAddCandidateSource().candidates(mixed)).getMessage());
    }

    static PrepareContext<CpuPortableAnalysisInputs> chain(DataType type, Shape shape) {
        ValueId left = new ValueId(0);
        ValueId right = new ValueId(1);
        ValueId intermediate = new ValueId(2);
        ValueId output = new ValueId(3);
        Operation add = new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE);
        var first = new CompiledNode(new NodeId(0), add,
                List.of(left, right), List.of(intermediate));
        var second = new CompiledNode(new NodeId(1), add,
                List.of(intermediate, left), List.of(output));
        return graph(type, shape, List.of(first, second), 4);
    }

    static PrepareContext<CpuPortableAnalysisInputs> graph(
            DataType type, Shape shape, List<CompiledNode> nodes, int valueCount) {
        var descriptor = new TensorDescriptor(type, shape, Optional.empty(), false);
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        Set<ValueId> outputs = nodes.stream().flatMap(node -> node.outputs().stream())
                .collect(java.util.stream.Collectors.toSet());
        Set<ValueId> inputs = nodes.stream().flatMap(node -> node.inputs().stream())
                .collect(java.util.stream.Collectors.toSet());
        var values = new java.util.ArrayList<GraphValue>();
        var memory = new java.util.ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < valueCount; index++) {
            var id = new ValueId(index);
            values.add(new GraphValue(id, descriptor));
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    outputs.contains(id) ? Optional.of(partition) : Optional.empty(),
                    inputs.contains(id) ? List.of(partition) : List.of(), outputs.contains(id)));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                new CpuPortableAnalysisInputs(List.of(),
                        new CpuPreparedParallelConfiguration(1, 1, true)));
    }
}
