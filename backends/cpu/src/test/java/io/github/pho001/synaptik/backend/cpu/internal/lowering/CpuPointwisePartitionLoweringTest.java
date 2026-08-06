package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuNativeBuffer;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

class CpuPointwisePartitionLoweringTest {
    @Test void lowersOneThroughEightOccurrencesWithDerivedBoundaries() {
        for (int count = 1; count <= 8; count++) {
            int expectedCount = count;
            var lowered = new CpuPartitionLowering().lower(chain(count));
            assertAll(
                    () -> assertEquals(expectedCount, lowered.kernelIr().instructions().size()),
                    () -> assertEquals(2, lowered.boundaryValues().size()),
                    () -> assertEquals(expectedCount - 1, lowered.virtualValues().size()),
                    () -> assertEquals(expectedCount == 1 ? 0 : expectedCount - 1,
                            lowered.kernelIr().values().stream()
                                    .filter(value -> value.kind() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.VIRTUAL)
                                    .count()));
        }
    }

    @Test void rejectsOverBudgetAndDisconnectedPartitionsBeforeDeclarations() {
        assertThrows(IllegalArgumentException.class, () -> new CpuPartitionLowering().lower(chain(9)));
        var context = chain(2);
        var nodes = new ArrayList<>(context.nodes());
        CompiledNode second = nodes.get(1);
        nodes.set(1, new CompiledNode(second.id(), second.operation(),
                List.of(new ValueId(0)), second.outputs()));
        var disconnected = new PrepareContext<>(context.partition(), nodes, context.values(),
                context.memoryRequirements(), Map.of(), context.backendInputs());
        assertThrows(IllegalArgumentException.class,
                () -> new CpuPartitionLowering().lower(disconnected));
    }

    @Test void preparationDerivesDefaultSegmentPatternAndScalarFallback() {
        var plan = new CpuPartitionPreparer().analyze(chain(8)).plan();
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(chain(8)), Optional.empty());
        assertAll(
                () -> assertEquals(2, plan.bufferDeclarations().size()),
                () -> assertEquals(List.of(
                        io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                        plan.carrierPattern()),
                () -> assertEquals("scalar", plan.executionStrategy().toString()),
                () -> assertEquals(List.of(DataType.INT32, DataType.INT32),
                        plan.units().getFirst().portablePlan().specialization().boundaryDataTypes()),
                () -> assertEquals(2, executable.bufferSelectionCount()),
                () -> assertNotNull(executable.artifact().hiddenClass()));
    }

    @Test void finalizedEightInstructionExecutableRunsThroughOneBoundInvocation() {
        var analysis = new CpuPartitionPreparer().analyze(chain(8));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        var input = CpuNativeBuffer.allocate(DataType.INT32, 20, 4);
        var output = CpuNativeBuffer.allocate(DataType.INT32, 20, 4);
        var state = new RunState(executable.memoryPlan(), List.of(
                List.of(new BufferRepresentationBinding(input, RunResourceOwnership.RUN_OWNED)),
                List.of(new BufferRepresentationBinding(output, RunResourceOwnership.RUN_OWNED))), List.of());
        var layout = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
        try {
            for (int i = 0; i < 5; i++) input.segment().set(layout, i * 4L, Integer.MAX_VALUE - i);
            executable.bind(state).execute();
            for (int i = 0; i < 5; i++) assertEquals(Integer.MAX_VALUE - i + 36,
                    output.segment().get(layout, i * 4L));
        } finally { state.close(); }
    }

    @Test void coldBindingRejectsNonCanonicalWhereCondition() {
        var analysis = new CpuPartitionPreparer().analyze(where());
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        var condition = CpuNativeBuffer.allocate(DataType.BOOL, 5, 1);
        var whenTrue = CpuNativeBuffer.allocate(DataType.FLOAT32, 20, 4);
        var whenFalse = CpuNativeBuffer.allocate(DataType.FLOAT32, 20, 4);
        var output = CpuNativeBuffer.allocate(DataType.FLOAT32, 20, 4);
        condition.segment().set(ValueLayout.JAVA_BYTE, 0, (byte) 2);
        var state = new RunState(executable.memoryPlan(), List.of(condition, whenTrue, whenFalse, output)
                .stream().map(buffer -> List.of(new BufferRepresentationBinding(buffer,
                        RunResourceOwnership.RUN_OWNED))).toList(), List.of());
        try {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(state));
        } finally { state.close(); }
    }

    static PrepareContext<CpuPartitionAnalysisInputs> chain(int count) {
        Shape shape = Shape.of(5);
        var descriptor = new TensorDescriptor(DataType.INT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var nodes = new ArrayList<CompiledNode>();
        ValueId input = new ValueId(0);
        ValueId previous = input;
        for (int i = 0; i < count; i++) {
            ValueId output = new ValueId(i + 1L);
            nodes.add(new CompiledNode(new NodeId(i), new Operation(ScalarElementwiseKind.ADD,
                    new ScalarValueAttrs(ScalarValue.int32(i + 1))), List.of(previous), List.of(output)));
            previous = output;
        }
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i <= count; i++) {
            ValueId id = new ValueId(i);
            values.add(new GraphValue(id, descriptor));
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    i == 0 ? Optional.empty() : Optional.of(partition),
                    i == count ? List.of() : List.of(partition), i == count));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> where() {
        Shape shape = Shape.of(5);
        var bool = new TensorDescriptor(DataType.BOOL, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var f32 = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var ids = List.of(new ValueId(0), new ValueId(1), new ValueId(2), new ValueId(3));
        var node = new CompiledNode(new NodeId(0),
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                ids.subList(0, 3), List.of(ids.get(3)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        var descriptors = List.of(bool, f32, f32, f32);
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < 4; i++) {
            values.add(new GraphValue(ids.get(i), descriptors.get(i)));
            memory.add(new LogicalMemoryRequirement(ids.get(i), descriptors.get(i),
                    i == 3 ? Optional.of(partition) : Optional.empty(),
                    i == 3 ? List.of() : List.of(partition), i == 3));
        }
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }
}
